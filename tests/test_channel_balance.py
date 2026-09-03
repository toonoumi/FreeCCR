"""Channel Balance: the tone-weighted per-channel control (spec/channel-balance.md).

Channel Balance replaces Temperature/Tint with one low-anchored curve node per
channel. It exists because every other per-channel control in the app is
tone-UNIFORM — Channel Levels' Shift is a density offset, its Gain a slope — and
neither can make the shadows a different colour from the highlights (crossover).

The properties worth pinning down here are: the node's endpoints stay pinned,
the curve never inverts contrast, the effect is tone-weighted toward the low
end, and the un-clamped shadow margin / highlight headroom survive the stage
untouched.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # ccr_backend pulls in Qt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.ccr_processor import (  # noqa: E402
    BALANCE_MAX_STOPS,
    BALANCE_NODE_X,
    WS_B,
    WS_W,
    _apply_channel_balance,
    _apply_working_space_recovery,
    _balance_curve,
    _balance_lut,
    apply_pre_balance_stages,
    compute_neutral_balance_for_image,
    _ws_enabled,
    adjust_image,
    adjust_image_opencl,
    balance_curve_points,
    compute_neutral_balance,
    encode_window,
)

BALANCE_KEYS = ("balance_r", "balance_g", "balance_b")


def _windowed(d):
    """Encode display values into a windowed uint16 base."""
    arr = np.asarray(d, dtype=np.float32).reshape(-1, 1, 3).copy()
    return encode_window(arr)


def _full(d):
    """Display values -> a normal full-range uint16 image."""
    arr = np.asarray(d, dtype=np.float32).reshape(-1, 1, 3)
    return np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)


@pytest.fixture
def ws_on():
    if not _ws_enabled():
        pytest.skip("working space disabled (FREECCR_WORKING_SPACE=0)")


# --- The curve itself -------------------------------------------------------

def test_node_stays_inside_the_endpoints_at_any_value():
    """The node moves in gamma, so it approaches 0 and 1 asymptotically and can
    never reach either pinned endpoint — there is no range cap to violate. A
    linear offset had to stay below NODE_X or it would cross (0,0)."""
    for s in (-100, -100.0, -37, 0, 42, 100, 1e6, -1e6):
        (_, _), (nx, ny), (_, _) = balance_curve_points(s)
        assert nx == BALANCE_NODE_X
        assert 0.0 < ny < 1.0


def test_node_gamma_is_reciprocal_about_zero():
    """+s and -s are equal and opposite in GAMMA (the slider's natural unit),
    which is what keeps the control symmetric to use even though the resulting
    y-offsets are not symmetric."""
    for mag in (25, 60, 100):
        _, (_, up), _ = balance_curve_points(mag)
        _, (_, down), _ = balance_curve_points(-mag)
        g_up = np.log(BALANCE_NODE_X) / np.log(up)
        g_down = np.log(BALANCE_NODE_X) / np.log(down)
        assert g_up * g_down == pytest.approx(1.0, rel=1e-9)
    _, (_, full), _ = balance_curve_points(100)
    assert full == pytest.approx(BALANCE_NODE_X ** (1.0 / 2.0 ** BALANCE_MAX_STOPS))


def test_travel_is_stronger_than_the_old_linear_scheme():
    """The point of the gamma move: a max correction reaches much further. The
    old linear scheme peaked at +0.13 / -0.14 deviation and could not fully
    correct a heavy cast."""
    x = np.linspace(0.0, 1.0, 401)
    up = np.asarray(_balance_curve(100, x)) - x
    down = np.asarray(_balance_curve(-100, x)) - x
    assert up.max() > 0.35
    assert down.min() < -0.20


def test_identity_at_zero():
    np.testing.assert_allclose(_balance_lut(0.0),
                               np.linspace(0.0, 1.0, len(_balance_lut(0.0))),
                               atol=1e-6)


@pytest.mark.parametrize("s", range(-100, 101, 5))
def test_monotone_and_endpoints_pinned(s):
    """The Fritsch-Carlson limiter must keep every curve non-decreasing (a tone
    curve that inverts would invert local contrast), and both endpoints pinned
    so pure black and pure white never move."""
    lut = _balance_lut(s)
    assert np.all(np.diff(lut) >= -1e-9)
    assert lut[0] == pytest.approx(0.0, abs=1e-9)
    assert lut[-1] == pytest.approx(1.0, abs=1e-6)


def test_effect_is_tone_weighted_toward_the_low_end():
    """The property that distinguishes Balance from Channel Levels Shift: the
    deviation from identity is 0 at both ends, peaks in the LOWER half, and has
    faded by the highlights. (The peak sits above NODE_X — a 3-point monotone
    cubic spreads a low node's influence through the low midtones, which is
    exactly what dragging that node in the Curves editor does.)"""
    x = np.linspace(0.0, 1.0, 201)
    dev = np.asarray(_balance_curve(50, x)) - x
    peak = float(dev.max())
    assert peak > 0.0
    assert x[int(np.argmax(dev))] < 0.5          # peak in the lower half
    assert dev[0] == pytest.approx(0.0, abs=1e-9)
    assert dev[-1] == pytest.approx(0.0, abs=1e-6)
    at_hi = float(np.asarray(_balance_curve(50, np.array([0.9])))[0] - 0.9)
    assert at_hi < peak / 5.0                    # faded by the highlights


def test_sign_direction():
    """+s raises the channel at the node, -s lowers it. The magnitudes are NOT
    equal — a node low on the curve has far more room above it than below — so only the
    direction is asserted here; the gamma symmetry is checked above."""
    up = float(_balance_curve(40, BALANCE_NODE_X)) - BALANCE_NODE_X
    down = float(_balance_curve(-40, BALANCE_NODE_X)) - BALANCE_NODE_X
    assert up > 0 and down < 0


# --- The stage --------------------------------------------------------------

def test_neutral_is_a_no_op():
    d = np.array([[[0.1, 0.4, 0.9]]], dtype=np.float32)
    before = d.copy()
    _apply_channel_balance(d, 0.0, 0.0, 0.0)
    np.testing.assert_array_equal(d, before)


def test_channels_are_independent():
    d = np.full((1, 1, 3), 0.3, dtype=np.float32)
    _apply_channel_balance(d, 80.0, 0.0, 0.0)
    assert d[0, 0, 0] > 0.3
    assert d[0, 0, 1] == pytest.approx(0.3)
    assert d[0, 0, 2] == pytest.approx(0.3)


def test_out_of_range_values_pass_through_unchanged():
    """Sub-black (film base in the shadow margin) and above-white (highlight
    headroom) must survive the stage untouched. Clamping them into the LUT
    instead would flatten the whole shadow margin onto 0 and destroy exactly the
    film-base separation spec/channel-levels-pre-clamp.md 3.4 protects."""
    d = np.array([[[-0.4, -0.01, 1.6], [1.0001, 2.5, -2.0]]], dtype=np.float32)
    before = d.copy()
    _apply_channel_balance(d, 100.0, -100.0, 60.0)
    np.testing.assert_array_equal(d, before)


def test_exact_endpoints_are_pinned_in_the_stage():
    d = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)
    _apply_channel_balance(d, 100.0, -100.0, 55.0)
    np.testing.assert_allclose(d[0, 0], [0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(d[0, 1], [1.0, 1.0, 1.0], atol=1e-6)


# --- Pipeline wiring --------------------------------------------------------

def test_windowed_neutral_byte_identical(ws_on):
    """Zeroed Balance must leave a windowed render bit-for-bit unchanged."""
    base = _windowed([[0.2, 0.5, 0.8]])
    np.testing.assert_array_equal(
        _apply_working_space_recovery(base, 0.0, 0.0, 0.0, 0.0, 1.0,
                                      *([0.0] * 12), 0.0, 0.0, 0.0),
        _apply_working_space_recovery(base, 0.0))


def test_adjust_image_neutral_byte_identical():
    img = _full([[0.2, 0.5, 0.8], [0.05, 0.5, 0.95]])
    np.testing.assert_array_equal(
        adjust_image(img.copy(), balance_r=0, balance_g=0, balance_b=0),
        adjust_image(img.copy()))


def test_windowed_balance_consumed_once(ws_on):
    """adjust_image(ws_windowed) with only Balance equals the recovery helper
    directly — the pre-stage consumes it and the look chain must not re-apply
    it. This is the double-apply guard."""
    base = _windowed([[0.15, 0.30, 0.55]])
    out = adjust_image(base, ws_windowed=True,
                       balance_r=-40, balance_g=0, balance_b=35)
    exp = _apply_working_space_recovery(base, 0.0, 0.0, 0.0, 0.0, 1.0,
                                        *([0.0] * 12), -40.0, 0.0, 35.0)
    np.testing.assert_array_equal(out, exp)


def test_windowed_balance_moves_the_shadows_not_the_white(ws_on):
    """End to end on a windowed base: a +R Balance lifts a shadow-tone red but
    leaves display white where it was."""
    base = _windowed([[BALANCE_NODE_X, BALANCE_NODE_X, BALANCE_NODE_X],
                      [1.0, 1.0, 1.0]])
    flat = _apply_working_space_recovery(base, 0.0)
    lifted = _apply_working_space_recovery(base, 0.0, 0.0, 0.0, 0.0, 1.0,
                                           *([0.0] * 12), 60.0, 0.0, 0.0)
    assert int(lifted[0, 0, 0]) > int(flat[0, 0, 0])      # shadow R lifted
    assert int(lifted[0, 0, 1]) == int(flat[0, 0, 1])     # G untouched
    assert int(lifted[1, 0, 0]) == int(flat[1, 0, 0])     # white pinned


def test_non_windowed_path_applies_the_same_curve():
    """The non-windowed path must be the SAME function on img/65535, not a
    parallel implementation that could drift from the windowed one."""
    img = _full([[BALANCE_NODE_X, 0.5, 0.9]])
    out = adjust_image(img.copy(), balance_r=70)
    d = img.astype(np.float32) / 65535.0
    _apply_channel_balance(d, 70.0, 0.0, 0.0)
    np.testing.assert_allclose(out[..., 0] / 65535.0, d[..., 0], atol=2e-5)


def test_legacy_temperature_tint_render_is_untouched():
    """The removed sliders' PIPELINE stage stays, so catalogs written before
    Channel Balance keep rendering exactly as they did."""
    img = _full([[0.2, 0.5, 0.8], [0.4, 0.4, 0.4]])
    a = adjust_image(img.copy(), 60.0, -25.0)
    b = adjust_image(img.copy(), 60.0, -25.0,
                     balance_r=0, balance_g=0, balance_b=0)
    np.testing.assert_array_equal(a, b)


# --- CPU/GPU parity ---------------------------------------------------------

def _opencl_or_skip():
    from core.ccr_processor import _initialize_opencl
    if not _initialize_opencl():
        pytest.skip("OpenCL unavailable")


@pytest.mark.parametrize("levels", [False, True])
def test_cpu_gpu_parity_non_windowed(levels):
    """Balance is applied in numpy on both paths, so the two must agree. With
    Channel Levels ALSO active the OpenCL path has to consume Levels in numpy
    too — the kernel would otherwise run it after Balance and reverse the stage
    order."""
    _opencl_or_skip()
    rng = np.random.default_rng(7)
    img = rng.integers(0, 65536, size=(16, 16, 3), dtype=np.uint16)
    kw = dict(balance_r=45.0, balance_g=-30.0, balance_b=15.0)
    if levels:
        kw.update(ch_r_shift=20.0, ch_g_gain=-15.0, ch_b_blackpoint=10.0)
    cpu = adjust_image(img.copy(), **kw)
    gpu = adjust_image_opencl(img.copy(), **kw)
    assert np.max(np.abs(cpu.astype(np.int32) - gpu.astype(np.int32))) <= 2


def test_cpu_gpu_parity_windowed(ws_on):
    _opencl_or_skip()
    rng = np.random.default_rng(11)
    base = encode_window(rng.uniform(-0.2, 1.4, size=(16, 16, 3)).astype(np.float32))
    kw = dict(ws_windowed=True, balance_r=-50.0, balance_g=10.0, balance_b=40.0)
    cpu = adjust_image(base.copy(), **kw)
    gpu = adjust_image_opencl(base.copy(), **kw)
    assert np.max(np.abs(cpu.astype(np.int32) - gpu.astype(np.int32))) <= 2


# --- The neutralising inverse (WB picker / AWB) -----------------------------

@pytest.mark.parametrize("trip", [
    (0.30, 0.22, 0.18),
    (0.12, 0.20, 0.28),
    (0.45, 0.50, 0.42),
    (0.60, 0.58, 0.62),
])
def test_inverse_neutralizes_at_the_sampled_tone(trip):
    """The solve must bring the three channels together AT the sampled tone.

    The residual is bounded by INTEGER SLIDER QUANTIZATION, not by the solve, so
    the tolerance is derived from what one slider step is worth at this tone
    rather than hardcoded — otherwise it silently becomes a strength test and
    breaks whenever BALANCE_MAX_STOPS is retuned."""
    vals = compute_neutral_balance(*trip)
    assert all(-100 <= v <= 100 for v in vals)
    got = [float(_balance_curve(s, v)) for s, v in zip(vals, trip)]
    step = max(abs(float(_balance_curve(s + 1, v)) - float(_balance_curve(s, v)))
               for s, v in zip(vals, trip))
    assert max(got) - min(got) <= 2.0 * step


def test_inverse_is_neutral_for_a_neutral_sample():
    assert compute_neutral_balance(0.5, 0.5, 0.5) == (0, 0, 0)


def test_inverse_clamps_an_unreachable_cast():
    """A cast far beyond what the node can span clamps to the endpoints rather
    than raising or running away."""
    vals = compute_neutral_balance(0.02, 0.5, 0.99)
    assert vals[0] == 100 and vals[2] == -100


def test_inverse_needs_absolute_tone_not_just_ratio():
    """Unlike compute_neutral_temp_tint (flat gains, ratio-only), Balance is
    tone-dependent: the same cast RATIO at a different tone must give different
    slider values. This is why both callers now pass normalised [0,1]."""
    low = compute_neutral_balance(0.10, 0.09, 0.08)
    high = compute_neutral_balance(0.80, 0.72, 0.64)
    assert low != high


# --- Solving against what the stage actually receives ------------------------
#
# The bare curve inverse is only correct when NOTHING moved the pixel before the
# Balance stage. Channel Levels runs first and carries the hidden Auto Gain
# offset (on by default for converted images), and Balance is tone-DEPENDENT, so
# ignoring that made every WB pick and AWB result wrong. These lock the fix in.

class _StubImage:
    """Minimal duck-type for compute_neutral_balance_for_image."""

    def __init__(self, settings=None, ws=True, converted=True, base=None):
        self.adjustment_settings = dict(settings or {})
        self._ws_windowed = ws
        self.converted = converted
        self.resized_raw = base


def test_pre_balance_stages_is_identity_when_nothing_precedes():
    rgb = (0.35, 0.30, 0.19)
    assert apply_pre_balance_stages(rgb, {}) == pytest.approx(rgb, abs=1e-6)


def test_pre_balance_stages_applies_channel_levels():
    rgb = (0.35, 0.30, 0.19)
    out = apply_pre_balance_stages(rgb, {"ch_r_shift": 30})
    assert out[0] > rgb[0]
    assert out[1] == pytest.approx(rgb[1], abs=1e-6)


def test_pre_balance_stages_folds_auto_gain_into_master_gain():
    """Auto Gain cannot create a cast (it is symmetric) but it MOVES THE TONE,
    and a tone-weighted control solved at the wrong tone lands on the wrong part
    of the curve. This is what made picks wrong with default settings."""
    rgb = (0.35, 0.30, 0.19)
    plain = apply_pre_balance_stages(rgb, {})
    gained = apply_pre_balance_stages(rgb, {}, auto_gain=40.0)
    assert gained[0] > plain[0] * 1.1
    # symmetric: the ratios are untouched, only the level moves
    assert gained[0] / gained[2] == pytest.approx(plain[0] / plain[2], rel=1e-5)


@pytest.mark.parametrize("settings", [
    {},
    {"ch_r_shift": 15, "ch_g_gain": 12, "ch_b_blackpoint": -8},
    {"ch_master_gain": 35},
    {"ch_input_gain": -20},
])
def test_image_aware_solve_beats_the_bare_inverse(settings):
    """With anything set ahead of Balance, the bare inverse misses and the
    image-aware solve lands. Checked at the stage's own input, which is where
    'neutral' has to hold for the render to come out neutral."""
    rgb = (0.35, 0.308, 0.192)
    img = _StubImage(settings, converted=False)      # no base -> no auto gain
    vals = compute_neutral_balance_for_image(img, rgb)
    pre = apply_pre_balance_stages(rgb, settings)
    got = [float(_balance_curve(v, x)) for v, x in zip(vals, pre)]
    assert max(got) - min(got) < 0.01


def test_image_aware_solve_survives_a_missing_base():
    """A stub/partial image must not break the pick (getattr defaults)."""
    assert compute_neutral_balance_for_image(_StubImage(), (0.3, 0.3, 0.3)) == (0, 0, 0)


def test_non_windowed_image_uses_the_clamped_levels_position():
    """clamp mirrors the pipeline: Channel Levels runs un-clamped on a windowed
    base and clamped on a full-range one."""
    rgb = (0.9, 0.9, 0.9)
    assert apply_pre_balance_stages(rgb, {"ch_master_gain": 60}, clamp=True)[0] == 1.0
    assert apply_pre_balance_stages(rgb, {"ch_master_gain": 60}, clamp=False)[0] > 1.0


# --- Settings keys ----------------------------------------------------------

def test_balance_keys_default_to_zero_when_absent():
    """A catalog written before this feature carries no balance_* keys; they
    must resolve to the 0 default rather than changing the render."""
    from core.ccr_processor import _channel_balance_active
    s = {"brightness": 10}
    assert not _channel_balance_active(*(s.get(k, 0) for k in BALANCE_KEYS))

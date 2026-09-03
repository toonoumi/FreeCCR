"""Channel Levels as the first pipeline stage (spec/channel-levels-pre-clamp.md).

The stage moved from late in the look domain (post window clamp) to the FRONT of
the pipeline, inside the working-space recovery where values are still un-clamped.
That is what lets a per-channel shift TRANSLATE the histogram — lifting sub-black
film base into view instead of merely adding to what is already displayed — and
what stops a +blue shift from painting the film base blue.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.ccr_processor import (  # noqa: E402
    CH_INPUT_GAIN_DIV,
    CH_MIN_RANGE,
    CH_SLIDER_DIV,
    DEFAULT_DENSITY_SLOPE,
    WS_B,
    WS_W,
    _apply_channel_levels,
    _apply_working_space_recovery,
    _default_slope_invert,
    _ws_enabled,
    _WS_HEADROOM_STOPS,
    adjust_image,
    encode_window,
)

CH_KEYS = (
    "ch_input_gain", "ch_master_shift", "ch_master_gain",
    "ch_r_shift", "ch_r_gain", "ch_r_blackpoint",
    "ch_g_shift", "ch_g_gain", "ch_g_blackpoint",
    "ch_b_shift", "ch_b_gain", "ch_b_blackpoint",
)


def _levels(**kw):
    """Positional Channel Levels args in the order the helpers take them."""
    return [float(kw.get(k, 0.0)) for k in CH_KEYS]


def _windowed(d):
    """Encode display values `d` (H,W,3 or a list of RGB triples) into a
    windowed uint16 base."""
    arr = np.asarray(d, dtype=np.float32).reshape(-1, 1, 3).copy()
    return encode_window(arr)


def _decode(u16):
    """Full-range [0,65535] uint16 → display floats."""
    return np.asarray(u16, dtype=np.float32) / 65535.0


def _recover(base_u16, **kw):
    """Run the windowed recovery (Channel Levels + WB + WP + Gain + clamp)."""
    return _apply_working_space_recovery(
        base_u16, kw.pop("exposure", 0.0), kw.pop("white_point", 0.0),
        kw.pop("kelvin_shift", 0.0), kw.pop("tint_shift", 0.0),
        kw.pop("tint_balance_factor", 1.0), *_levels(**kw))


# --- neutral identity -------------------------------------------------------

def test_neutral_is_identity_windowed():
    base = _windowed([[0.0, 0.25, 0.5], [0.75, 1.0, 0.1]])
    out = _recover(base)
    expect = _apply_working_space_recovery(base.copy(), 0.0)
    assert np.array_equal(out, expect)


def test_neutral_is_identity_full_range():
    rng = np.random.default_rng(7)
    img = rng.integers(0, 65536, size=(6, 5, 3), dtype=np.uint16)
    out = adjust_image(img.copy(), 0, 0, 0, 0, 0, 0, 0, 0, 1.0)
    ref = adjust_image(img.copy(), 0, 0, 0, 0, 0, 0, 0, 0, 1.0, *(0,) * 2,
                       *_levels())
    assert np.array_equal(out, ref)


# --- stage ordering: input gain -> per-channel -> master --------------------

def test_input_gain_applies_before_the_channel_shift():
    # (in*ig + s) != (in + s)*ig — pick values where the two differ.
    d = np.full((1, 1, 3), 0.25, dtype=np.float32)
    out = _apply_channel_levels(d.copy(), *_levels(ch_input_gain=25,
                                                   ch_r_shift=30), clamp=False)
    ig = 2.0 ** (25.0 / CH_INPUT_GAIN_DIV)
    s = 30.0 / CH_SLIDER_DIV
    assert out[0, 0, 0] == pytest.approx(0.25 * ig + s, abs=1e-6)
    # wrong order would be (0.25 + s) * ig
    assert out[0, 0, 0] != pytest.approx((0.25 + s) * ig, abs=1e-6)


def test_master_shift_is_a_separate_stage_not_summed_into_the_channel():
    # Old behaviour summed master+channel shift, then divided ONCE by the
    # channel range. Now the channel range applies first, master after — so a
    # channel gain scales the channel shift but NOT the master shift.
    d = np.full((1, 1, 3), 0.2, dtype=np.float32)
    out = _apply_channel_levels(d.copy(),
                                *_levels(ch_r_shift=30, ch_r_gain=50,
                                         ch_master_shift=30), clamp=False)
    s = 30.0 / CH_SLIDER_DIV
    g = 50.0 / CH_SLIDER_DIV
    expect = (0.2 + s) / (1.0 - g) + s          # master added AFTER the divide
    assert out[0, 0, 0] == pytest.approx(expect, abs=1e-6)
    summed = (0.2 + s + s) / (1.0 - g)          # the old, summed behaviour
    assert out[0, 0, 0] != pytest.approx(summed, abs=1e-6)


def test_master_gain_applies_to_every_channel_after_the_per_channel_work():
    d = np.full((1, 1, 3), 0.4, dtype=np.float32)
    out = _apply_channel_levels(d.copy(), *_levels(ch_master_gain=50),
                                clamp=False)
    expect = 0.4 / (1.0 - 50.0 / CH_SLIDER_DIV)
    assert out[0, 0] == pytest.approx([expect] * 3, abs=1e-6)


# --- the headline behaviour: a shift translates the histogram ---------------

def test_shift_lifts_sub_black_content_into_the_window():
    """A pixel BELOW display black is invisible (clamps to 0) until a positive
    shift translates it up into the window."""
    base = _windowed([[-0.30, -0.30, -0.30]])
    neutral = _decode(_recover(base))
    assert neutral[0, 0] == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)

    shifted = _decode(_recover(base, ch_master_shift=75))
    expect = -0.30 + 75.0 / CH_SLIDER_DIV       # = 0.20
    assert expect > 0
    assert shifted[0, 0] == pytest.approx([expect] * 3, abs=2e-3)


def test_post_clamp_shift_could_not_have_lifted_it():
    """Regression guard for the OLD ordering: applied after the window clamp the
    same shift starts from a flattened 0, so it cannot reconstruct the value."""
    base = _windowed([[-0.30, -0.30, -0.30]])
    clamped = _decode(_recover(base))            # what the look domain used to see
    post = np.clip(clamped + 75.0 / CH_SLIDER_DIV, 0.0, 1.0)
    pre = _decode(_recover(base, ch_master_shift=75))
    # The old path lands at the raw shift; the new one lands lower, because the
    # pixel genuinely started below black.
    assert post[0, 0, 0] == pytest.approx(0.5, abs=1e-3)
    assert pre[0, 0, 0] == pytest.approx(0.2, abs=2e-3)
    assert pre[0, 0, 0] < post[0, 0, 0]


def test_shift_pushed_out_the_bottom_is_recoverable():
    """Content shifted below black lands in the shadow margin, not in a clip —
    shifting back restores it."""
    base = _windowed([[0.30, 0.30, 0.30]])
    # -60 pushes 0.30 to -0.10 (out of the window), +60 brings it back.
    d = np.asarray(base, dtype=np.float32).reshape(-1, 1, 3).copy()
    d -= np.float32(WS_B)
    d /= np.float32(WS_W - WS_B)
    _apply_channel_levels(d, *_levels(ch_master_shift=-60), clamp=False)
    assert d[0, 0, 0] < 0.0                      # left the window
    _apply_channel_levels(d, *_levels(ch_master_shift=60), clamp=False)
    assert d[0, 0, 0] == pytest.approx(0.30, abs=2e-3)


def test_film_base_stays_black_while_the_shadow_lifts():
    """The motivating case. Film base sits below display black; a true image
    shadow sits just above it. A +blue shift must lift the shadow WITHOUT
    tinting the base blue."""
    base = _windowed([[-0.35, -0.35, -0.35],     # film base / rebate
                      [0.05, 0.05, 0.05]])       # a real image shadow
    out = _decode(_recover(base, ch_b_shift=30))
    film_base, shadow = out[0, 0], out[1, 0]

    s = 30.0 / CH_SLIDER_DIV                     # = 0.20
    # The shadow's blue rises...
    assert shadow[2] == pytest.approx(0.05 + s, abs=2e-3)
    assert shadow[2] > shadow[0]
    # ...while the film base stays NEUTRAL black: -0.35 + 0.20 is still < 0.
    assert film_base == pytest.approx([0.0, 0.0, 0.0], abs=1e-4)
    assert film_base[2] == film_base[0] == film_base[1]


def test_film_base_would_have_gone_blue_under_the_old_post_clamp_order():
    """Same inputs, but with the shift applied after the clamp (the old
    ordering): the base picks up the full shift and turns blue."""
    base = _windowed([[-0.35, -0.35, -0.35]])
    clamped = _decode(_recover(base))            # base flattened to 0
    old = np.clip(clamped + np.array([0, 0, 30.0 / CH_SLIDER_DIV], np.float32),
                  0.0, 1.0)
    assert old[0, 0, 2] > 0.15                   # blue-tinted black
    assert old[0, 0, 2] > old[0, 0, 0]


# --- strength ---------------------------------------------------------------

def test_slider_strength_is_doubled():
    assert CH_SLIDER_DIV == 150.0                # was 300.0
    assert CH_INPUT_GAIN_DIV == 25.0             # was 50.0
    # full-scale shift is +-2/3 of the display range, input gain +-4 stops
    assert 100.0 / CH_SLIDER_DIV == pytest.approx(2.0 / 3.0)
    assert np.log2(2.0 ** (100.0 / CH_INPUT_GAIN_DIV)) == pytest.approx(4.0)


def test_shift_offset_matches_the_mapping():
    d = np.zeros((1, 1, 3), dtype=np.float32)
    out = _apply_channel_levels(d, *_levels(ch_g_shift=45), clamp=False)
    assert out[0, 0, 1] == pytest.approx(45.0 / CH_SLIDER_DIV, abs=1e-6)


# --- the new denominator guard ---------------------------------------------

def test_extreme_gain_and_blackpoint_does_not_invert_the_channel():
    """(1 - 0.667) - 0.667 = -0.333: a negative denominator would flip the
    channel. CH_MIN_RANGE floors it."""
    d = np.asarray([[[0.2, 0.5, 0.8]]], dtype=np.float32)
    out = _apply_channel_levels(d.copy(),
                                *_levels(ch_r_gain=100, ch_r_blackpoint=100,
                                         ch_g_gain=100, ch_g_blackpoint=100,
                                         ch_b_gain=100, ch_b_blackpoint=100),
                                clamp=True)
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)
    # order is preserved (not inverted): brighter input stays brighter
    assert out[0, 0, 0] <= out[0, 0, 1] <= out[0, 0, 2]


def test_min_range_is_the_divisor_at_the_extreme():
    """At gain=+100, blackpoint=+100 the raw range is -0.333; CH_MIN_RANGE
    replaces it, so the channel is divided by +0.1 rather than a negative."""
    d = np.full((1, 1, 3), 0.05, dtype=np.float32)
    out = _apply_channel_levels(d.copy(),
                                *_levels(ch_r_gain=100, ch_r_blackpoint=100),
                                clamp=False)
    bp = 100.0 / CH_SLIDER_DIV
    assert (1.0 - bp) - bp < 0.0                 # the raw range really is negative
    assert out[0, 0, 0] == pytest.approx((0.05 - bp) / CH_MIN_RANGE, abs=1e-5)


# --- the default-slope floor -----------------------------------------------

def test_default_slope_keeps_sub_base_density(ws_on=None):
    """A pixel BRIGHTER than the sampled black point is clearer than the film
    base: its density is negative and must survive into the shadow margin."""
    assert _ws_enabled()
    black_point = [1000.0, 1000.0, 1000.0]
    img = np.full((1, 1, 3), 4000.0, dtype=np.float32)   # 2 stops brighter
    out = _default_slope_invert(img, black_point)
    assert out[0, 0, 0] < WS_B                   # below display black
    expect_d = DEFAULT_DENSITY_SLOPE * np.log10(1000.0 / 4000.0)
    got_d = (float(out[0, 0, 0]) - WS_B) / (WS_W - WS_B)
    assert got_d == pytest.approx(expect_d, abs=2e-3)


def test_default_slope_floor_kept_when_working_space_off(monkeypatch):
    monkeypatch.setenv("FREECCR_WORKING_SPACE", "0")
    black_point = [1000.0, 1000.0, 1000.0]
    img = np.full((1, 1, 3), 4000.0, dtype=np.float32)
    out = _default_slope_invert(img, black_point)
    assert int(out[0, 0, 0]) == 0                # legacy: floored at black


def test_default_slope_no_nan_with_a_display_gamma(monkeypatch):
    """The gamma branch must not raise a negative density to a fractional
    power (NaN); the sub-black region stays linear."""
    import core.ccr_processor as proc
    monkeypatch.setattr(proc, "DEFAULT_DENSITY_GAMMA", 2.2)
    img = np.asarray([[[4000.0, 4000.0, 4000.0], [200.0, 200.0, 200.0]]],
                     dtype=np.float32)
    out = proc._default_slope_invert(img, [1000.0, 1000.0, 1000.0])
    assert np.all(np.isfinite(out))
    assert out[0, 0, 0] < WS_B                   # sub-black survived
    assert out[0, 1, 0] > WS_B                   # normal pixel above black


# --- window geometry --------------------------------------------------------

def test_shadow_margin_widened_without_costing_precision():
    assert WS_B == 1024.0
    assert WS_W == 2048.0
    assert WS_W - WS_B == 1024.0                 # display precision unchanged
    assert _WS_HEADROOM_STOPS > 5.9              # highlight headroom kept
    # the margin must cover the full shift range
    assert (WS_B / (WS_W - WS_B)) >= 100.0 / CH_SLIDER_DIV


# --- CPU / GPU parity -------------------------------------------------------

def test_cpu_gpu_parity_over_a_parameter_sweep():
    from core.ccr_processor import _initialize_opencl, adjust_image_opencl
    if not _initialize_opencl():
        pytest.skip("OpenCL unavailable")
    rng = np.random.default_rng(11)
    img = rng.integers(0, 65536, size=(9, 7, 3), dtype=np.uint16)
    for _ in range(8):
        vals = rng.integers(-100, 101, size=12).astype(float)
        kw = dict(zip(CH_KEYS, vals))
        cpu = adjust_image(img.copy(), 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0,
                           *_levels(**kw))
        gpu = adjust_image_opencl(img.copy(), 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0,
                                  *_levels(**kw))
        assert np.max(np.abs(cpu.astype(int) - gpu.astype(int))) <= 1, kw


def test_cpu_gpu_parity_windowed():
    from core.ccr_processor import _initialize_opencl, adjust_image_opencl
    if not _initialize_opencl():
        pytest.skip("OpenCL unavailable")
    rng = np.random.default_rng(12)
    base = encode_window(rng.uniform(-0.5, 3.0, size=(9, 7, 3)).astype(np.float32))
    kw = dict(ch_b_shift=40, ch_r_gain=-30, ch_master_shift=15, ch_input_gain=10)
    cpu = adjust_image(base.copy(), 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0,
                       *_levels(**kw), ws_windowed=True)
    gpu = adjust_image_opencl(base.copy(), 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0,
                              *_levels(**kw), ws_windowed=True)
    assert np.max(np.abs(cpu.astype(int) - gpu.astype(int))) <= 1

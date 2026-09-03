"""No-anchor conversion (spec/no-anchor-convert.md).

Converting with NO black point and NO white point is allowed: the conversion is
NamiColor's negative transform in DENSITY space, `d = -log10(16 * p) + 1.0`, with
its fixed constants and nothing measured off the frame. The user grades it with
Channel Levels afterwards — which on this base works in the same density space
NamiColor's own sliders do.

Ported from github.com/Wavechaser/NamiColor -> NamiColor_dev/NamiColor_dev.c
(3.1, GPL-3.0). `_dctl_neg` below is a literal transcription of that source's
`neg` branch and is the reference every math test checks against.

The recipe is modelled as the existing `mode: "bw"` with `bw = (None, None)`, so
every replay site keeps working unchanged — these tests pin that too.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.ccr_processor import (  # noqa: E402
    WS_B,
    WS_W,
    NAMI_BASE_OFFSET,
    NAMI_INPUT_SCALE,
    _default_slope_invert,
    _namicolor_invert,
    _twopoint_invert,
    _ws_enabled,
    apply_bwpoint_normalization,
    compute_sprocket_alpha,
)


def _decode(base_u16):
    """Windowed container codes → display values."""
    return (np.asarray(base_u16, dtype=np.float32) - WS_B) / (WS_W - WS_B)


def _dctl_neg(p, input_gain=1.0):
    """Literal transcription of NamiColor_dev.c's `neg` branch:
        inputScale = 16.0 ; invScale = -1.0
        init = invScale * log10(inputScale * p)
        init = init * inputGain + 1.0
    `p` is the LINEAR scene value in [0,1]."""
    return -math.log10(16.0 * p) * input_gain + 1.0


# One window code is 1/1024 in display units — the quantisation floor for any
# comparison against the float reference.
WINDOW_QUANTUM = 1.5 / (WS_W - WS_B)


# --- the inversion matches the DCTL --------------------------------------

def test_matches_the_dctl_across_the_range():
    """Every sample must reproduce the upstream formula to within window
    quantisation. This is the test that says "exactly like NamiColor"."""
    assert _ws_enabled()
    vals = [1, 100, 1000, 4096, 8192, 32768, 50000, 65535]
    out = _namicolor_invert(np.asarray([[[v] * 3 for v in vals]], dtype=np.float32))
    got = _decode(out)
    for i, v in enumerate(vals):
        ref = _dctl_neg(v / 65535.0)
        assert got[0, i, 0] == pytest.approx(ref, abs=WINDOW_QUANTUM), v


def test_upstream_constants():
    assert NAMI_INPUT_SCALE == 16.0     # DCTL inputScale for negatives
    assert NAMI_BASE_OFFSET == 1.0      # DCTL "+ 1.0f" for negatives


def test_zero_density_lands_at_one():
    """`16*p == 1` is the DCTL's density zero, and the +1.0 puts it at d = 1.0."""
    d = _decode(_namicolor_invert(np.full((1, 1, 3), 65535.0 / 16.0, np.float32)))
    assert d[0, 0] == pytest.approx([1.0] * 3, abs=WINDOW_QUANTUM)


def test_half_scale_base_lands_near_cineon_black():
    """The fixed constants are chosen so a film base sitting near half scale
    maps to roughly Cineon black (93/1023) — that is what makes the unanchored
    conversion usable without measuring anything."""
    d = _decode(_namicolor_invert(np.full((1, 1, 3), 0.5 * 65535.0, np.float32)))
    assert d[0, 0, 0] == pytest.approx(93.0 / 1023.0, abs=0.01)


def test_it_is_log_not_linear():
    """A DECADE of transmission must be a constant density step — the property
    that makes a Channel Levels shift a clean per-channel multiply in linear
    light. A linear flip would fail this."""
    v = np.asarray([[[6553.5] * 3, [655.35] * 3, [65.535] * 3]], dtype=np.float32)
    d = _decode(_namicolor_invert(v))[0, :, 0]
    assert (d[1] - d[0]) == pytest.approx(1.0, abs=2 * WINDOW_QUANTUM)
    assert (d[2] - d[1]) == pytest.approx(1.0, abs=2 * WINDOW_QUANTUM)


def test_clear_film_goes_below_black_into_the_shadow_margin():
    """Brighter than the assumed base → negative density → sub-black, which the
    widened shadow margin holds and Channel Levels can lift."""
    out = _namicolor_invert(np.full((1, 1, 3), 65535.0, np.float32))
    assert out[0, 0, 0] < WS_B                     # below display black
    assert out[0, 0, 0] > 0                        # but inside the container
    assert _decode(out)[0, 0, 0] == pytest.approx(_dctl_neg(1.0),
                                                  abs=WINDOW_QUANTUM)


def test_dense_areas_go_into_the_highlight_headroom():
    out = _namicolor_invert(np.full((1, 1, 3), 1.0, np.float32))
    assert out[0, 0, 0] > WS_W                     # above display white
    assert out[0, 0, 0] <= 65535                   # still inside the container


def test_treats_each_channel_independently():
    """No cross-channel normalisation: the frame's cast survives the conversion
    (Channel Levels is what removes it)."""
    img = np.asarray([[[10000.0, 30000.0, 50000.0]]], dtype=np.float32)
    d = _decode(_namicolor_invert(img))[0, 0]
    for c, v in enumerate((10000.0, 30000.0, 50000.0)):
        assert d[c] == pytest.approx(_dctl_neg(v / 65535.0), abs=WINDOW_QUANTUM)
    assert d[0] > d[1] > d[2]


def test_no_nan_at_zero():
    """log10(0) is -inf; the density floor must keep the output finite."""
    out = _namicolor_invert(np.zeros((2, 2, 3), dtype=np.float32))
    assert np.all(np.isfinite(out))


def test_legacy_full_range_when_working_space_off(monkeypatch):
    monkeypatch.setenv("FREECCR_WORKING_SPACE", "0")
    out = _namicolor_invert(np.asarray(
        [[[65535.0, 65535.0 / 16.0, 6553.5]]], dtype=np.float32))
    assert int(out[0, 0, 0]) == 0                       # negative density clipped
    assert abs(int(out[0, 0, 1]) - 65535) <= 2          # d = 1.0 -> white
    assert abs(int(out[0, 0, 2]) - int(_dctl_neg(0.1) * 65535)) <= 4


# --- dispatch ---------------------------------------------------------------

def test_replay_dispatches_to_the_flip_with_no_anchors():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 65536, size=(8, 8, 3), dtype=np.uint16)
    got = apply_bwpoint_normalization(img, None, None)
    assert np.array_equal(got, _namicolor_invert(img.astype(np.float32)))


def test_black_point_only_still_routes_to_default_slope():
    rng = np.random.default_rng(6)
    img = rng.integers(1, 65536, size=(8, 8, 3), dtype=np.uint16)
    bp = (30000.0, 30000.0, 30000.0)
    assert np.array_equal(apply_bwpoint_normalization(img, bp, None),
                          _default_slope_invert(img.astype(np.float32), bp))


def test_two_point_still_routes_to_the_two_point_math():
    rng = np.random.default_rng(7)
    img = rng.integers(1, 65536, size=(8, 8, 3), dtype=np.uint16)
    bp, wp = (30000.0,) * 3, (3000.0,) * 3
    assert np.array_equal(
        apply_bwpoint_normalization(img, bp, wp, density=False),
        _twopoint_invert(img.astype(np.float32), bp, wp, False))


def test_no_sprocket_mask_without_a_black_point():
    """The reversal-look clear-film overlay is anchor-relative, so an unanchored
    conversion simply has none."""
    rng = np.random.default_rng(8)
    img = rng.integers(0, 65536, size=(16, 16, 3), dtype=np.uint16)
    assert compute_sprocket_alpha(img, None) is None


# --- the catalog recipe round-trip -----------------------------------------

def test_catalog_round_trips_a_no_anchor_recipe():
    from core.catalog import _ci_from_json, _ci_to_json
    ci = {"mode": "bw", "bw": (None, None), "fine_rot": 0,
          "density": False, "slopes": None}
    assert _ci_from_json(_ci_to_json(ci)) == ci


def test_catalog_still_round_trips_the_anchored_recipes():
    from core.catalog import _ci_from_json, _ci_to_json
    for bw in (((1.0, 2.0, 3.0), None),
               ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))):
        ci = {"mode": "bw", "bw": bw, "fine_rot": 0,
              "density": True, "slopes": None}
        assert _ci_from_json(_ci_to_json(ci)) == ci


def test_no_anchor_recipe_is_json_serializable():
    """_ci_to_json must produce plain JSON types — the crash this guards is
    list(None) on the black point."""
    import json
    from core.catalog import _ci_to_json
    payload = _ci_to_json({"mode": "bw", "bw": (None, None), "fine_rot": 0,
                           "density": False, "slopes": None})
    assert json.loads(json.dumps(payload))["bw"] == [None, None]


# --- the backend no longer refuses -----------------------------------------

def test_convert_all_no_longer_requires_a_black_point():
    """apply_bwpoint_to_all_images used to raise ValueError with no black point.
    With no images loaded it must now simply do nothing."""
    from core.ccr_backend import ccr_backend
    saved_images, saved_bp = ccr_backend.images, ccr_backend.black_point_bgr
    try:
        ccr_backend.images = []
        ccr_backend.black_point_bgr = None
        ccr_backend.apply_bwpoint_to_all_images()      # must not raise
    finally:
        ccr_backend.images, ccr_backend.black_point_bgr = saved_images, saved_bp


def test_warn_flag_defaults_on():
    from core.ccr_backend import ccr_backend
    assert getattr(ccr_backend, "warn_no_anchor_convert", None) is True

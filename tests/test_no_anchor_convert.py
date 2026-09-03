"""No-anchor conversion / Direct Invert (spec/no-anchor-convert.md).

Converting with NO black point and NO white point is allowed: the conversion is a
plain per-channel flip (`d = 1 - v/65535`) with nothing normalised, and the user
grades it with Channel Levels afterwards. It is modelled as the existing
`mode: "bw"` recipe with `bw = (None, None)`, so every replay site keeps working
unchanged — these tests pin that.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.ccr_processor import (  # noqa: E402
    WS_B,
    WS_W,
    _default_slope_invert,
    _flip_invert,
    _twopoint_invert,
    _ws_enabled,
    apply_bwpoint_normalization,
    compute_sprocket_alpha,
)


def _decode(base_u16):
    """Windowed container codes → display values."""
    return (np.asarray(base_u16, dtype=np.float32) - WS_B) / (WS_W - WS_B)


# --- the flip itself --------------------------------------------------------

def test_flip_maps_the_endpoints():
    """Scan black (0) → display white; scan white (65535) → display black."""
    assert _ws_enabled()
    img = np.asarray([[[0, 0, 0], [65535, 65535, 65535]]], dtype=np.float32)
    d = _decode(_flip_invert(img))
    assert d[0, 0] == pytest.approx([1.0, 1.0, 1.0], abs=1e-3)
    assert d[0, 1] == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)


def test_flip_endpoints_land_exactly_on_the_window():
    out = _flip_invert(np.asarray([[[0, 0, 0], [65535, 65535, 65535]]],
                                  dtype=np.float32))
    assert int(out[0, 0, 0]) == int(WS_W)      # scan black → display white
    assert int(out[0, 1, 0]) == int(WS_B)      # scan white → display black


def test_flip_is_linear_through_the_midpoint():
    mid = np.full((1, 1, 3), 65535.0 / 2.0, dtype=np.float32)
    assert _decode(_flip_invert(mid))[0, 0] == pytest.approx([0.5] * 3, abs=2e-3)


def test_flip_treats_each_channel_independently():
    """No cross-channel normalisation: a channel-varying scan flips per channel,
    so the cast survives the conversion (Channel Levels is what removes it)."""
    img = np.asarray([[[10000.0, 30000.0, 50000.0]]], dtype=np.float32)
    d = _decode(_flip_invert(img))[0, 0]
    assert d == pytest.approx([1 - 10000 / 65535, 1 - 30000 / 65535,
                               1 - 50000 / 65535], abs=2e-3)
    assert d[0] > d[1] > d[2]                   # the cast is preserved, not equalised


def test_flip_uses_no_headroom_or_shadow_margin():
    """uint16 in → d exactly in [0,1]; nothing lands outside the window."""
    rng = np.random.default_rng(4)
    img = rng.integers(0, 65536, size=(24, 24, 3)).astype(np.float32)
    out = _flip_invert(img)
    assert out.min() >= WS_B
    assert out.max() <= WS_W


def test_flip_legacy_full_range_when_working_space_off(monkeypatch):
    monkeypatch.setenv("FREECCR_WORKING_SPACE", "0")
    out = _flip_invert(np.asarray([[[0, 65535, 32768]]], dtype=np.float32))
    assert int(out[0, 0, 0]) == 65535
    assert int(out[0, 0, 1]) == 0
    assert abs(int(out[0, 0, 2]) - 32767) <= 2


# --- dispatch ---------------------------------------------------------------

def test_replay_dispatches_to_the_flip_with_no_anchors():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 65536, size=(8, 8, 3), dtype=np.uint16)
    got = apply_bwpoint_normalization(img, None, None)
    assert np.array_equal(got, _flip_invert(img.astype(np.float32)))


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

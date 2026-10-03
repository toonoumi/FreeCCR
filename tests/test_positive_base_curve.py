#!/usr/bin/env python3
"""Positive-mode base render curve (spec/positive-base-curve.md).

A transfer function is not a rendering: BT.709/sRGB encodes the data but does
not place the tones, which is why a positive opens flat and dark here while
Lightroom looks normal with no adjustments. Positives therefore start from a
base S-curve — toe, midtone lift, shoulder — fitted to the camera's own JPEG
rendering.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import ccr_image as ccr_image_mod  # noqa: E402
from core import catalog  # noqa: E402
from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from core.ccr_processor import (POSITIVE_BASE_CURVE,  # noqa: E402
                                positive_base_curve_points)

FULL = ccr_image_mod.POSITIVE_BASE_CURVE_STRENGTH


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FREECCR_POSITIVE_CURVE", raising=False)
    saved = ccr_backend.positive_mode
    yield
    ccr_backend.positive_mode = saved


def _stub(base_curve=0, settings=None):
    img = CCRImage.__new__(CCRImage)
    img.adjustment_settings = settings or {}
    img.color_profile = "color"
    img.decoded_mono = False
    img.contrast_base = 0
    img.temperature_base = 0
    img.brightness_base = 0
    img.exposure_base = 0
    img.base_curve = base_curve
    img.converted = False
    img._ws_windowed = False
    img.tint_balance_factor = 1.0
    return img


# --- the curve itself ------------------------------------------------------- #

def test_strength_blends_between_identity_and_the_fitted_curve():
    ident = positive_base_curve_points(0)
    assert all(abs(x - y) < 1e-9 for x, y in ident), "0 must be the diagonal"
    assert positive_base_curve_points(100) == [[x, y] for x, y in POSITIVE_BASE_CURVE]
    half = positive_base_curve_points(50)
    for (x, y), (_, full_y) in zip(half, POSITIVE_BASE_CURVE):
        assert y == pytest.approx(x + 0.5 * (full_y - x))


@pytest.mark.parametrize("strength", [0, 25, 50, 75, 100])
def test_curve_is_monotone_and_pins_endpoints(strength):
    pts = positive_base_curve_points(strength)
    ys = [y for _, y in pts]
    assert ys == sorted(ys)
    assert pts[0] == [0.0, 0.0] and pts[-1] == [255.0, 255.0]


def test_strength_is_clamped():
    assert positive_base_curve_points(999) == positive_base_curve_points(100)
    assert positive_base_curve_points(-50) == positive_base_curve_points(0)


# --- the baked strength ----------------------------------------------------- #

def test_default_strength():
    assert ccr_image_mod._positive_base_curve_strength() == FULL


@pytest.mark.parametrize("env, expected", [
    ("0", 0), ("70", 70), ("100", 100), ("250", 100), ("-10", 0), ("62.4", 62),
])
def test_env_override(monkeypatch, env, expected):
    monkeypatch.setenv("FREECCR_POSITIVE_CURVE", env)
    assert ccr_image_mod._positive_base_curve_strength() == expected


def test_garbage_env_falls_back(monkeypatch):
    monkeypatch.setenv("FREECCR_POSITIVE_CURVE", "nope")
    assert ccr_image_mod._positive_base_curve_strength() == FULL


# --- render shape: this is what separates a curve from a gamma -------------- #

def test_pins_black_and_white_lifts_mids_and_deepens_deep_shadows():
    """The measured render DEEPENS the deep shadows while lifting the midtones.
    No single gamma does that — it is the reason this is an S-curve."""
    deep, mid, upper = 3277, 32768, 49151          # 0.05, 0.50, 0.75
    src = np.array([[[0, 0, 0], [deep] * 3, [mid] * 3, [upper] * 3,
                     [65535] * 3]], dtype=np.uint16)
    out = _stub(base_curve=FULL).apply_adjustments(src)
    assert out[0, 0, 0] == 0                       # black pinned
    assert out[0, -1, 0] == 65535                  # white pinned
    assert out[0, 1, 0] < deep                     # deep shadow DEEPENED
    assert out[0, 2, 0] > mid                      # midtone lifted
    assert out[0, 3, 0] > upper                    # upper mid lifted


def test_positive_with_no_sliders_is_rendered_not_skipped():
    """The no-adjustments early return must not swallow the base curve."""
    src = np.repeat(np.linspace(2000, 60000, 24, dtype=np.uint16)
                    .reshape(1, -1, 1), 3, axis=2)
    assert _stub(base_curve=FULL).apply_adjustments(src).mean() > src.mean()


def test_negative_is_untouched():
    src = np.repeat(np.linspace(2000, 60000, 24, dtype=np.uint16)
                    .reshape(1, -1, 1), 3, axis=2)
    out = _stub(base_curve=0).apply_adjustments(src)
    assert out is src or np.array_equal(out, src)


def test_override_parameter_wins_over_the_attribute():
    src = np.repeat(np.linspace(2000, 60000, 24, dtype=np.uint16)
                    .reshape(1, -1, 1), 3, axis=2)
    np.testing.assert_array_equal(
        _stub(base_curve=FULL).apply_adjustments(src, base_curve=0),
        _stub(base_curve=0).apply_adjustments(src))


def test_areas_do_not_re_apply_the_base():
    src = np.repeat(np.linspace(2000, 60000, 24, dtype=np.uint16)
                    .reshape(1, -1, 1), 3, axis=2)
    np.testing.assert_array_equal(
        _stub(base_curve=FULL)._adjust_for_area(src, {"gamma": 10}),
        _stub(base_curve=0)._adjust_for_area(src, {"gamma": 10}))


# --- persistence + cache invalidation --------------------------------------- #

def test_catalog_round_trips_base_curve():
    img = _stub(base_curve=FULL)
    for attr, val in (("display_name", None), ("is_duplicate", False),
                      ("slice_group", None), ("slice_parent", None),
                      ("source_ops", []), ("conversion_inputs", None),
                      ("crop_rect", None), ("crop_angle", 0.0),
                      ("rotation_angle", 0), ("fine_rotation_angle", 0),
                      ("horizontal_mirrored", False), ("vertical_mirrored", False),
                      ("reference_frame", None)):
        setattr(img, attr, val)
    state = catalog.serialize_image(img)
    assert state["base_curve"] == FULL
    assert "base_curve" not in {k: v for k, v in state.items()
                                if k != "base_curve"}


def test_hires_signature_tracks_base_curve():
    from widgets.image_preview import ImagePreview
    ip = ImagePreview.__new__(ImagePreview)
    ip.current_idx = 0
    saved = ccr_backend.images
    try:
        sigs = []
        for bc in (0, FULL):
            img = _stub(base_curve=bc)
            img.dust_spots = []
            img.area_layers = []
            img._dust_plan_cache = None
            ccr_backend.images = [img]
            sigs.append(ip._current_adj_sig())
        assert sigs[0] != sigs[1], "a stale zoom tile must be invalidated"
    finally:
        ccr_backend.images = saved


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

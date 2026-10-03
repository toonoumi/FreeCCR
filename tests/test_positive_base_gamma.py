#!/usr/bin/env python3
"""Positive-mode base tone curve (spec/positive-base-gamma.md).

A positive decode starts from a baked Gamma-slider offset, the way a negative
starts from brightness_base = -8 — because the positive decode places the white
point (often with pixels already clipped) while leaving the midtones low, which
is a curve problem, not a gain problem.
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

BASE = ccr_image_mod.POSITIVE_BASE_GAMMA


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FREECCR_POSITIVE_GAMMA", raising=False)
    saved = ccr_backend.positive_mode
    yield
    ccr_backend.positive_mode = saved


def _stub(gamma_base=0, settings=None):
    img = CCRImage.__new__(CCRImage)
    img.adjustment_settings = settings or {}
    img.color_profile = "color"
    img.decoded_mono = False
    img.contrast_base = 0
    img.temperature_base = 0
    img.brightness_base = 0
    img.exposure_base = 0
    img.gamma_base = gamma_base
    img.converted = False
    img._ws_windowed = False
    img.tint_balance_factor = 1.0
    return img


def _ramp():
    v = np.linspace(2000, 60000, 24, dtype=np.uint16)
    return np.repeat(v.reshape(1, -1, 1), 3, axis=2)


# --- the baked value -------------------------------------------------------- #

def test_default_is_the_measured_baseline():
    assert ccr_image_mod._positive_base_gamma() == BASE
    assert BASE > 0


@pytest.mark.parametrize("env, expected", [
    ("0", 0), ("25", 25), ("70", 70), ("-30", -30), ("54.6", 55),
])
def test_env_override(monkeypatch, env, expected):
    monkeypatch.setenv("FREECCR_POSITIVE_GAMMA", env)
    assert ccr_image_mod._positive_base_gamma() == expected


def test_garbage_env_falls_back(monkeypatch):
    monkeypatch.setenv("FREECCR_POSITIVE_GAMMA", "nope")
    assert ccr_image_mod._positive_base_gamma() == BASE


# --- render behaviour ------------------------------------------------------- #

def test_base_curve_lifts_a_positive_with_no_sliders():
    """The no-adjustments early return must not swallow the base curve."""
    src = _ramp()
    out = _stub(gamma_base=BASE).apply_adjustments(src)
    assert out.mean() > src.mean()


def test_base_curve_pins_black_and_white():
    """The whole reason this is a curve and not a gain: white and black do NOT
    move, only the middle does. A gain would drive the already-clipped
    highlights further into the ceiling to achieve the same midtone lift."""
    src = np.array([[[0, 0, 0], [16384] * 3, [32768] * 3, [65535] * 3]],
                   dtype=np.uint16)
    out = _stub(gamma_base=BASE).apply_adjustments(src)
    assert out[0, 0, 0] == 0                      # black pinned
    assert out[0, -1, 0] == 65535                 # white pinned
    assert out[0, 1, 0] > src[0, 1, 0]            # midtones lifted
    assert out[0, 2, 0] > src[0, 2, 0]


def test_negative_is_untouched():
    src = _ramp()
    out = _stub(gamma_base=0).apply_adjustments(src)
    assert out is src or np.array_equal(out, src)


def test_base_and_slider_sum():
    src = _ramp()
    both = _stub(gamma_base=20, settings={"gamma": 30}).apply_adjustments(src)
    single = _stub(gamma_base=0, settings={"gamma": 50}).apply_adjustments(src)
    np.testing.assert_array_equal(both, single)


def test_the_sum_clamps_at_the_slider_domain():
    src = _ramp()
    over = _stub(gamma_base=BASE, settings={"gamma": 100}).apply_adjustments(src)
    maxed = _stub(gamma_base=0, settings={"gamma": 100}).apply_adjustments(src)
    np.testing.assert_array_equal(over, maxed)


def test_override_parameter_wins_over_the_attribute():
    src = _ramp()
    img = _stub(gamma_base=BASE)
    np.testing.assert_array_equal(
        img.apply_adjustments(src, gamma_base=0),
        _stub(gamma_base=0).apply_adjustments(src))


def test_areas_do_not_re_apply_the_base():
    """_adjust_for_area zeroes the base offsets; the base curve is global look
    already baked into the layer an area composites onto."""
    img = _stub(gamma_base=BASE)
    src = _ramp()
    np.testing.assert_array_equal(
        img._adjust_for_area(src, {"gamma": 10}),
        _stub(gamma_base=0)._adjust_for_area(src, {"gamma": 10}))


# --- persistence + cache invalidation --------------------------------------- #

def test_catalog_round_trips_gamma_base():
    img = _stub(gamma_base=BASE)
    for attr, val in (("display_name", None), ("is_duplicate", False),
                      ("slice_group", None), ("slice_parent", None),
                      ("source_ops", []), ("conversion_inputs", None),
                      ("crop_rect", None), ("crop_angle", 0.0),
                      ("rotation_angle", 0), ("fine_rotation_angle", 0),
                      ("horizontal_mirrored", False), ("vertical_mirrored", False),
                      ("reference_frame", None)):
        setattr(img, attr, val)
    state = catalog.serialize_image(img)
    assert state["gamma_base"] == BASE
    # A record written before this feature simply has no key.
    legacy = {k: v for k, v in state.items() if k != "gamma_base"}
    assert "gamma_base" not in legacy


def test_hires_signature_tracks_gamma_base():
    """_current_adj_sig reads the image off the backend, so stub both. Without
    gamma_base in the signature a zoom tile baked before a mode change would be
    reused and no longer match the preview."""
    from widgets.image_preview import ImagePreview
    ip = ImagePreview.__new__(ImagePreview)
    ip.current_idx = 0
    saved = ccr_backend.images
    try:
        sigs = []
        for gb in (0, BASE):
            img = _stub(gamma_base=gb)
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

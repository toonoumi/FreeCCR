#!/usr/bin/env python3
"""Tests for Auto Gain (spec/auto-gain.md).

Auto Gain is a hidden, live offset on Channel Levels' MASTER GAIN that places the
top-2%
*in-bound* highlight at 95% of the working-space window without moving the Gain
slider. "In-bound" = a de-windowed display value in [0, 1] (between the sampled
clear→0 and dense→1 conversion points); over-range/specular (>1) and sub-black
(<0) pixels are discarded. Pure numpy core, runs headless.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccr_processor import (  # noqa: E402
    compute_auto_gain_offset,
    encode_window,
    adjust_image,
    AG_PERCENTILE, AG_TARGET, AG_HI, AG_GMIN, AG_GMAX,
    CH_SLIDER_DIV,
)


def _ws_base(d):
    """Windowed base from a display array `d` (overshoot beyond 1 is kept as
    headroom up to the container ceiling)."""
    return encode_window(np.asarray(d, dtype=np.float32).copy())


def _realized_gain(v):
    """Master Gain maps slider value v → g = 1/(1 - v/CH_SLIDER_DIV)."""
    return 1.0 / (1.0 - np.clip(v, -100.0, 100.0) / CH_SLIDER_DIV)


def _render(base, v):
    """Render a windowed base with the auto-gain offset on its real stage."""
    return adjust_image(base, ch_master_gain=v, ws_windowed=True)


# --- the offset mapping ---------------------------------------------------

def test_constants():
    assert AG_PERCENTILE == 98.0
    assert AG_TARGET == 0.95
    assert AG_HI == 1.0
    assert (AG_GMIN, AG_GMAX) == (0.6, 3.0)


def test_uniform_midtone_targets_top():
    """A flat in-bound 0.5 base → an offset whose realized gain lands 0.5 at the
    target (0.95)."""
    base = _ws_base(np.full((32, 32, 3), 0.5))
    v = compute_auto_gain_offset(base, ws_windowed=True)
    assert 0.5 * _realized_gain(v) == pytest.approx(AG_TARGET, abs=2e-3)


def test_already_at_target_is_noop():
    base = _ws_base(np.full((32, 32, 3), AG_TARGET))
    v = compute_auto_gain_offset(base, ws_windowed=True)
    assert v == pytest.approx(0.0, abs=1.0)        # within ~1 slider unit


def test_dark_frame_clamps_at_max_gain():
    """A very dark base wants >3× gain; the stage caps at g=3.0 → v=+100 (the
    Master Gain slider's endpoint IS g=3.0, so the cap and the slider agree)."""
    base = _ws_base(np.full((32, 32, 3), 0.1))
    v = compute_auto_gain_offset(base, ws_windowed=True)
    assert v == pytest.approx(100.0, abs=1e-6)
    assert _realized_gain(v) == pytest.approx(AG_GMAX, abs=1e-6)


# --- req 4: discard out-of-bound (sampled clear/dense range) --------------

def test_overrange_highlights_are_excluded():
    """5% of pixels sit in headroom (d=1.5, denser than the dense sample); they
    must NOT drag the gain down — the offset equals the all-in-bound case."""
    d = np.full((40, 40, 3), 0.5, dtype=np.float32)
    d[:, :2, :] = 1.5                              # 5% of columns over-range
    over = compute_auto_gain_offset(_ws_base(d), ws_windowed=True)
    flat = compute_auto_gain_offset(_ws_base(np.full((40, 40, 3), 0.5)),
                                    ws_windowed=True)
    assert over == pytest.approx(flat, abs=1e-6)


def test_subblack_pixels_excluded_but_irrelevant_to_highlight():
    """Sub-black pixels (d<0, clearer than the clear sample) are dropped; they
    can't affect the top-2% percentile, so the offset is unchanged."""
    d = np.full((40, 40, 3), 0.5, dtype=np.float32)
    d[:, :4, :] = -0.2                              # clamped to code 0 in the window
    with_floor = compute_auto_gain_offset(_ws_base(d), ws_windowed=True)
    flat = compute_auto_gain_offset(_ws_base(np.full((40, 40, 3), 0.5)),
                                    ws_windowed=True)
    assert with_floor == pytest.approx(flat, abs=1e-6)


def test_insufficient_inbound_content_returns_zero():
    """<0.5% in-bound content (almost all over-range) → no auto-gain."""
    d = np.full((100, 100, 3), 2.0, dtype=np.float32)   # all over-range
    d[0, :30, :] = 0.5                                    # 0.3% in-bound
    assert compute_auto_gain_offset(_ws_base(d), ws_windowed=True) == 0.0


# --- round-trip through adjust_image (the real consumer) ------------------

def test_roundtrip_places_highlight_at_target():
    """Feeding the computed offset as the Master Gain value through adjust_image puts
    the 98th percentile of the output at ≈ AG_TARGET·65535. Pixels are
    near-neutral (shared gray + small channel jitter): with fully independent
    random channels the saturated top-2% pixels get single channels clipped at
    the window, deflating the output luminance percentile below the target —
    a clip-by-design of super-target headroom, not a placement error."""
    rng = np.random.default_rng(0)
    gray = rng.uniform(0.05, 0.7, size=(80, 80, 1)).astype(np.float32)
    d = np.clip(gray + rng.normal(0.0, 0.02, size=(80, 80, 3)), 0.0, 1.0
                ).astype(np.float32)
    base = _ws_base(d)
    v = compute_auto_gain_offset(base, ws_windowed=True)
    out = _render(base, v).astype(np.float32)
    lum = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
    assert np.percentile(lum, AG_PERCENTILE) == pytest.approx(AG_TARGET * 65535.0,
                                                              rel=0.02)


def test_at_target_is_near_noop_render():
    """A base already at the target produces an offset at most window-quantization
    sized (the 10-bit window can't represent the target exactly), so the render
    barely moves from the un-gained de-window. (A genuine offset of 0.0 — e.g. the
    insufficient-content case — is an exact no-op via the early-return guard.)"""
    base = _ws_base(np.full((16, 16, 3), AG_TARGET))
    v = compute_auto_gain_offset(base, ws_windowed=True)
    assert abs(v) < 1.5                                  # quantization noise only
    ref = _render(base, 0.0).astype(np.int32)
    got = _render(base, v).astype(np.int32)
    assert np.abs(ref - got).max() <= 128               # < 0.2% of 65535


# --- legacy (non-windowed) path -------------------------------------------

def test_nonws_full_range_base():
    """Without the working space the base is full-range [0,65535]; the same
    mid-grey target produces the same offset as the windowed case."""
    img = np.full((32, 32, 3), int(0.5 * 65535), dtype=np.uint16)
    v = compute_auto_gain_offset(img, ws_windowed=False)
    assert 0.5 * _realized_gain(v) == pytest.approx(AG_TARGET, abs=2e-3)


# --- wiring: Auto Gain rides MASTER GAIN, not the (slider-less) Gain stage ---
#
# apply_adjustments is exercised against a STUB self rather than a converted
# CCRImage: the real conversion path pulls in the backend + OpenCL and destabilizes
# this module. The stub carries exactly the attributes the method reads, so the
# thing under test is the routing itself.

class _StubImage:
    """Minimal stand-in for a converted CCRImage with a windowed base."""
    def __init__(self, settings):
        self._ws_windowed = True
        self.converted = True
        self.adjustment_settings = settings
        self.contrast_base = 0
        self.temperature_base = 0
        self.brightness_base = 0
        self.exposure_base = 0.0
        self.color_profile = "color"
        self.area_layers = []
        self.tint_balance_factor = 1.0

    def _apply_dust_removal(self, image, ws_windowed=False):
        return image

    def _to_grayscale(self, image):
        return image


def _capture_adjust_args(monkeypatch, base, settings, auto_gain):
    """Run apply_adjustments and return the args it passed to adjust_image_opencl."""
    from core.ccr_backend import ccr_backend
    from core.ccr_image import CCRImage
    import core.ccr_image as ccr_image_mod

    seen = {}

    def _capture(image, *args, **kw):
        seen["exposure"] = args[2]          # positional: temperature, tint, exposure
        seen["ch_master_gain"] = kw.get("ch_master_gain")
        return image

    monkeypatch.setattr(ccr_image_mod, "adjust_image_opencl", _capture)
    monkeypatch.setattr(ccr_backend, "auto_gain", auto_gain, raising=False)
    CCRImage.apply_adjustments(_StubImage(settings), base)
    return seen


def test_auto_gain_is_wired_to_master_gain(monkeypatch):
    """The offset must be ADDED to ch_master_gain and must NOT ride the old
    (now slider-less) exposure stage."""
    base = _ws_base(np.full((32, 32, 3), 0.35))
    seen = _capture_adjust_args(monkeypatch, base, {"ch_master_gain": 12},
                                auto_gain=True)
    ag = compute_auto_gain_offset(base, True)
    assert ag != 0.0, "this frame should produce a non-zero auto gain"
    assert seen["ch_master_gain"] == pytest.approx(12 + ag)
    assert seen["exposure"] == pytest.approx(0.0)


def test_auto_gain_off_leaves_master_gain_untouched(monkeypatch):
    base = _ws_base(np.full((32, 32, 3), 0.35))
    seen = _capture_adjust_args(monkeypatch, base, {"ch_master_gain": 7},
                                auto_gain=False)
    assert seen["ch_master_gain"] == pytest.approx(7)
    assert seen["exposure"] == pytest.approx(0.0)


def test_auto_gain_offset_is_in_master_gain_units(monkeypatch):
    """The routed offset, read back through the Master Gain curve, must place
    the measured highlight at AG_TARGET."""
    base = _ws_base(np.full((32, 32, 3), 0.35))
    seen = _capture_adjust_args(monkeypatch, base, {}, auto_gain=True)
    assert 0.35 * _realized_gain(seen["ch_master_gain"]) == pytest.approx(
        AG_TARGET, abs=2e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

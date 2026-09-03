#!/usr/bin/env python3
"""
Tests for the windowed working space with headroom
(spec/working-space-headroom.md).

When FREECCR_WORKING_SPACE is set, the negative inversion no longer hard-clips at
display white: the converted "base" buffer reserves a narrow display WINDOW inside
the 16-bit container (10-bit by default, codes [WS_B, WS_W]) and keeps over-range
data as recoverable headroom. apply_adjustments de-windows the base, applies the
Gain/Exposure recovery un-clamped (so blown highlights can be pulled back below
white), then clamps to the window and emits a normal full-range positive.

These exercise the pure-numpy core (no Qt), so they run headless. The feature is
read from the env at call time via _ws_enabled(), so monkeypatch toggles it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccr_processor import (  # noqa: E402
    apply_bwpoint_normalization,
    adjust_image,
    encode_window,
    _apply_working_space_recovery,
    _ws_enabled,
    WS_B, WS_W,
    DEFAULT_DENSITY_SLOPE,
)

BASE_BGR = (10000.0, 10000.0, 10000.0)   # neutral clear base for clean density math


def _img(pixels_bgr, dtype=np.uint16):
    return np.array([pixels_bgr], dtype=dtype)


@pytest.fixture
def ws_on(monkeypatch):
    monkeypatch.setenv("FREECCR_WORKING_SPACE", "1")
    assert _ws_enabled()


# --- window geometry / encode-decode -------------------------------------

def test_window_constants_are_10bit_default():
    # default 10-bit window, WS_LO=1.0 → B=1024, W=2048, width=1024.
    # WS_LO widened 0.5 → 1.0 for Channel Levels' shadow margin; the window
    # WIDTH (display precision) is unchanged. See spec/channel-levels-pre-clamp.md.
    assert WS_B == 1024.0
    assert WS_W == 2048.0
    assert WS_W - WS_B == 1024.0


def test_encode_decode_roundtrip():
    d = np.linspace(-0.5, 63.0, 4096, dtype=np.float32)
    code = encode_window(d.copy())
    assert code.dtype == np.uint16
    decoded = (code.astype(np.float32) - WS_B) / (WS_W - WS_B)
    # within one window code (1/1024 in display units), ignoring container clamp
    inside = (d >= -0.5) & (d <= 63.0)
    assert np.max(np.abs(decoded[inside] - d[inside])) < (1.5 / 1024.0)


def test_encode_anchors():
    code = encode_window(np.array([0.0, 1.0], dtype=np.float32))
    assert int(code[0]) == int(WS_B)     # display-black → B
    assert int(code[1]) == int(WS_W)     # display-white → W


# --- the toggle leaves legacy behaviour intact ---------------------------

def test_off_base_maps_to_zero(monkeypatch):
    """Feature OFF (FREECCR_WORKING_SPACE=0): a film-base pixel still inverts to
    code 0 (legacy)."""
    monkeypatch.setenv("FREECCR_WORKING_SPACE", "0")
    assert not _ws_enabled()
    out = apply_bwpoint_normalization(_img([tuple(map(round, BASE_BGR))]), BASE_BGR, None)
    assert int(out[0, 0, 0]) < 3


def test_on_base_maps_to_window_black(ws_on):
    """Feature ON: a film-base pixel maps to WS_B, not 0 (windowed black)."""
    out = apply_bwpoint_normalization(_img([tuple(map(round, BASE_BGR))]), BASE_BGR, None)
    assert abs(int(out[0, 0, 0]) - int(WS_B)) <= 2


# --- headroom is preserved instead of clipping ---------------------------

def test_headroom_preserved_default_slope(ws_on):
    """Highlights denser than the 1/slope ceiling clip to 65535 in legacy; with
    the window they survive as distinct codes above WS_W (recoverable)."""
    densities = (2.0, 3.0, 4.0)          # all above the 1/slope (=1.25) ceiling
    px = [(BASE_BGR[0] / (10.0 ** d),) * 3 for d in densities]
    out = apply_bwpoint_normalization(_img(px), BASE_BGR, None)[0, :, 0].astype(int)
    assert all(c > WS_W for c in out)            # in headroom, not clipped to white
    assert out[0] < out[1] < out[2]              # denser → higher code (distinct)
    assert out[2] < 65535 or out[2] == 65535     # bounded by the container


def test_headroom_preserved_twopoint_density(ws_on):
    """Two-point density mode: a pixel denser than the sampled dense point exceeds
    d=1 and is kept as headroom above WS_W rather than clipped."""
    base = (10000.0,) * 3
    dense = (1000.0,) * 3                 # dense point: d=1 at log10(10) = 1.0
    denser = (100.0,) * 3                 # twice the density → d≈2 (headroom)
    out = apply_bwpoint_normalization(_img([dense, denser]), base, dense,
                                      density=True)[0].astype(int)
    assert abs(out[0, 0] - int(WS_W)) <= 3       # dense point → display white (W)
    assert out[1, 0] > WS_W                       # denser → headroom


# --- recovery via the exposure/gain slider -------------------------------

def test_de_window_no_recovery_clips_like_legacy(ws_on):
    """De-window + window-clamp with no recovery reproduces the legacy clamped
    positive (within 10-bit window quantization)."""
    densities = (0.2, 0.6, 1.0)          # below / at the ceiling
    px = [(BASE_BGR[0] / (10.0 ** d),) * 3 for d in densities]
    win = apply_bwpoint_normalization(_img(px), BASE_BGR, None)
    rendered = _apply_working_space_recovery(win, 0.0)[0, :, 0].astype(int)
    legacy = [int(np.clip(DEFAULT_DENSITY_SLOPE * d, 0, 1) * 65535) for d in densities]
    for r, l in zip(rendered, legacy):
        assert abs(r - l) <= 80          # ~1 window code = 64 levels


def test_recovery_pulls_back_blown_highlights(ws_on):
    """Highlights that clip to white at zero gain become distinct, monotonic,
    sub-white values once a darkening (negative) gain is applied."""
    densities = (1.30, 1.40, 1.50)       # slope*d ∈ (1.0, 1.25): just over white
    px = [(BASE_BGR[0] / (10.0 ** d),) * 3 for d in densities]
    win = apply_bwpoint_normalization(_img(px), BASE_BGR, None)

    flat = _apply_working_space_recovery(win, 0.0)[0, :, 0].astype(int)
    assert np.all(flat == 65535)         # all blown to white at zero gain

    rec = _apply_working_space_recovery(win, -200.0)[0, :, 0].astype(int)
    assert rec.max() < 65535             # recovered below the clip
    assert rec[0] < rec[1] < rec[2]      # distinct detail restored, monotonic


# --- adjust_image honours the windowed base ------------------------------

def test_adjust_image_dewindows_identity(ws_on):
    """adjust_image with no sliders + ws_windowed == the de-window pre-stage."""
    px = [(BASE_BGR[0] / (10.0 ** 0.5),) * 3]
    win = apply_bwpoint_normalization(_img(px), BASE_BGR, None)
    out = adjust_image(win, ws_windowed=True)
    exp = _apply_working_space_recovery(win, 0.0)
    np.testing.assert_array_equal(out, exp)


def test_adjust_image_exposure_consumed_once(ws_on):
    """The recovery exposure is consumed by the pre-stage: passing it via
    adjust_image(ws_windowed=True) matches the recovery helper directly."""
    densities = (1.30, 1.40, 1.50)
    px = [(BASE_BGR[0] / (10.0 ** d),) * 3 for d in densities]
    win = apply_bwpoint_normalization(_img(px), BASE_BGR, None)
    out = adjust_image(win, exposure=-200.0, ws_windowed=True)[0, :, 0].astype(int)
    exp = _apply_working_space_recovery(win, -200.0)[0, :, 0].astype(int)
    np.testing.assert_array_equal(out, exp)


# --- White Point: wide-range recovery across the FULL headroom ------------

def test_white_point_recovers_full_headroom(ws_on):
    """White Point (negative) reaches the WHOLE headroom — densities far past the
    ceiling that Gain cannot touch come back as distinct, monotonic, sub-white
    values. (High base so the density floor doesn't cap the test densities.)"""
    base = (50000.0,) * 3
    densities = (1.25, 1.5, 2.0, 3.0, 4.0)        # 1.25 = ceiling; rest are headroom
    px = [(base[0] / (10.0 ** d),) * 3 for d in densities]
    win = apply_bwpoint_normalization(_img(px), base, None)

    flat = _apply_working_space_recovery(win, 0.0, 0.0)[0, :, 0].astype(int)
    assert np.all(flat[1:] == 65535)              # all headroom blown at WP=0

    rec = _apply_working_space_recovery(win, 0.0, -50.0)[0, :, 0].astype(int)
    assert rec.max() < 65535                       # everything recovered below white
    assert np.all(np.diff(rec) > 0)                # distinct + monotonic detail


def test_white_point_neutral_at_zero(ws_on):
    """WP=0 is byte-identical to a plain de-window (neutral edit unchanged)."""
    px = [(BASE_BGR[0] / (10.0 ** d),) * 3 for d in (0.4, 1.0, 2.0)]
    win = apply_bwpoint_normalization(_img(px), BASE_BGR, None)
    np.testing.assert_array_equal(
        _apply_working_space_recovery(win, 0.0, 0.0),
        _apply_working_space_recovery(win, 0.0))


def test_white_point_consumed_once_via_adjust(ws_on):
    """White Point recovery is consumed by the pre-stage and not re-applied by the
    look-domain B/W remap: adjust_image(whitepoint=...) == the recovery helper."""
    base = (50000.0,) * 3
    px = [(base[0] / (10.0 ** d),) * 3 for d in (1.5, 2.0, 3.0)]
    win = apply_bwpoint_normalization(_img(px), base, None)
    out = adjust_image(win, whitepoint=-50.0, ws_windowed=True)[0, :, 0].astype(int)
    exp = _apply_working_space_recovery(win, 0.0, -50.0)[0, :, 0].astype(int)
    np.testing.assert_array_equal(out, exp)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

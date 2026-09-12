#!/usr/bin/env python3
"""Tests for the sprocket-hole / clear-film white mask (spec/sprocket-hole-mask.md).

On a 135 negative the punched sprocket holes and clear rebate scan BRIGHTER than
the sampled film base, so after inversion they clamp to pure black. The optional
reversal-film look paints those clear-film regions white as the last render step.
The mask is derived from the RAW scan ("clearer than the sampled black point plus
a padding"), morphologically cleaned, and feathered. Pure numpy/cv2 core, headless.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccr_processor import (  # noqa: E402
    compute_sprocket_alpha,
    apply_sprocket_mask,
    SPROCKET_REF_LONG,
)


# --- synthetic raw scans (BGR, uint16) ------------------------------------

def _base_plate(h, w, bp=(30000, 30000, 30000)):
    """A uniform film-base plate at the (BGR) black-point value."""
    img = np.empty((h, w, 3), dtype=np.uint16)
    for c in range(3):
        img[..., c] = bp[c]
    return img


def _with_square(img, y0, y1, x0, x1, value):
    """Paint a rectangular region to a (BGR) value (a hole or a scene patch)."""
    out = img.copy()
    for c in range(3):
        out[y0:y1, x0:x1, c] = value[c]
    return out


# --- threshold isolates clear film ----------------------------------------

def test_holes_found_when_base_sits_just_under_clip():
    """Regression (5207 colour negative, trichrome merge): the decode saturates
    at 63486 (white level minus black pedestal), the base's blue is 62092 —
    only 1394 below that — and the holes clip in every channel. A threshold
    measured against 65535 put blue above the holes and masked nothing; the
    measured clear level must find them."""
    bp = (62092, 35455, 43587)
    img = _base_plate(1080, 1080, bp)
    img = _with_square(img, 400, 680, 400, 680, (63486, 63486, 63486))
    alpha = compute_sprocket_alpha(img, bp)
    assert alpha is not None
    assert alpha[540, 540] == 255
    assert alpha[100, 100] == 0                   # base


def test_near_base_shadow_not_whitened():
    """Shadows approach the base (noise, uneven light, a base sampled a touch
    denser): even 40% of the way to the measured clear level they stay
    unmasked, while the holes are caught."""
    bp = (30000, 30000, 30000)
    img = _base_plate(1080, 1080, bp)
    img = _with_square(img, 100, 300, 100, 300, (62000, 62000, 62000))  # hole
    shadow = 30000 + int(0.40 * (62000 - 30000))
    img = _with_square(img, 500, 800, 500, 800, (shadow, shadow, shadow))
    alpha = compute_sprocket_alpha(img, bp)
    assert alpha[200, 200] == 255
    assert alpha[650, 650] == 0


def test_no_clear_film_masks_nothing():
    """Without holes there is no gap to split: a mildly brighter corner
    (uneven illumination, density gap ~0.05) must not become the clear
    level and get whitened."""
    bp = (30000, 30000, 30000)
    img = _base_plate(1080, 1080, bp)
    img = _with_square(img, 0, 300, 0, 300, (33600, 33600, 33600))
    assert compute_sprocket_alpha(img, bp) is None


def test_hole_masked_scene_and_base_not():
    bp = (30000, 30000, 30000)
    img = _base_plate(1080, 1080, bp)
    # A bright square (a "sprocket hole": clearer than base in every channel).
    img = _with_square(img, 400, 680, 400, 680, (62000, 62000, 62000))
    # A darker square (exposed "scene": denser than base -> lower than base).
    img = _with_square(img, 100, 300, 100, 300, (10000, 10000, 10000))

    alpha = compute_sprocket_alpha(img, bp)
    assert alpha is not None
    assert alpha.dtype == np.uint8
    assert alpha.shape == (1080, 1080)
    # Solid white deep inside the hole, zero over base and over the scene patch.
    assert alpha[540, 540] == 255
    assert alpha[200, 200] == 0          # scene patch
    assert alpha[900, 900] == 0          # bare film base


def test_padding_rejects_base_noise():
    bp = (30000, 30000, 30000)
    rng = np.random.default_rng(0)
    img = _base_plate(1080, 1080, bp).astype(np.int32)
    # Noise amplitude well under the padding (0.20*headroom ~= 7107, floor
    # 0.02*65535 ~= 1311) -> nothing should qualify.
    img = img + rng.integers(-400, 400, size=img.shape)
    img = np.clip(img, 0, 65535).astype(np.uint16)
    assert compute_sprocket_alpha(img, bp) is None


def test_per_channel_and_requires_all_channels():
    # Brighter than base in G and R only (B stays at base) -> NOT clear film.
    bp = (30000, 30000, 30000)
    img = _base_plate(1080, 1080, bp)
    img = _with_square(img, 400, 680, 400, 680, (30000, 62000, 62000))  # B unchanged
    assert compute_sprocket_alpha(img, bp) is None


def test_no_black_point_is_none():
    img = _base_plate(256, 256)
    assert compute_sprocket_alpha(img, None) is None


def test_no_qualifying_pixels_is_none():
    bp = (30000, 30000, 30000)
    # Everything at or below base (a densely exposed frame, no clear film).
    img = _base_plate(512, 512, bp)
    assert compute_sprocket_alpha(img, bp) is None


# --- feather is a soft ramp -----------------------------------------------

def test_feather_is_soft_ramp_solid_interior():
    bp = (30000, 30000, 30000)
    img = _base_plate(1080, 1080, bp)
    img = _with_square(img, 300, 780, 300, 780, (62000, 62000, 62000))
    alpha = compute_sprocket_alpha(img, bp)
    assert alpha is not None
    assert alpha.max() == 255                      # solid interior
    # A thin feather band of intermediate alpha exists around the edge.
    assert np.any((alpha > 0) & (alpha < 255))


# --- sharpness: keep the true hole shape ----------------------------------

def test_interior_hole_is_filled():
    # A dark speck inside a hole (dust / a printed frame number) must NOT punch
    # through — the interior fills solid via the flood-fill imfill.
    bp = (30000, 30000, 30000)
    img = _base_plate(1080, 1080, bp)
    img = _with_square(img, 300, 780, 300, 780, (62000, 62000, 62000))   # hole
    img = _with_square(img, 520, 560, 520, 560, (8000, 8000, 8000))      # dark speck inside
    alpha = compute_sprocket_alpha(img, bp)
    assert alpha is not None
    assert alpha[540, 540] == 255      # filled despite the dark speck at the centre


def test_tiny_speckle_removed():
    # An isolated few-pixel bright speck (noise) far from any hole is dropped by
    # the connected-component area filter, while the real hole is kept.
    bp = (30000, 30000, 30000)
    img = _base_plate(1080, 1080, bp)
    img = _with_square(img, 300, 780, 300, 780, (62000, 62000, 62000))   # real hole
    img = _with_square(img, 60, 63, 60, 63, (62000, 62000, 62000))       # 3x3 speck
    alpha = compute_sprocket_alpha(img, bp)
    assert alpha is not None
    assert alpha[540, 540] == 255       # hole kept
    assert alpha[61, 61] == 0           # speck removed


def test_edges_stay_sharp_not_dilated():
    # No morphological dilation and only a ~1px feather: a few px OUTSIDE the true
    # boundary stays 0 (the old close + 8px feather glowed here), and the edge
    # transition is a narrow anti-alias band, not a soft ramp.
    bp = (30000, 30000, 30000)
    img = _base_plate(1080, 1080, bp)
    img = _with_square(img, 400, 680, 400, 680, (62000, 62000, 62000))   # edge at x=400
    alpha = compute_sprocket_alpha(img, bp)
    assert alpha is not None
    assert alpha[540, 540] == 255       # interior solid
    assert alpha[540, 396] == 0         # 4px outside the edge -> not whitened
    # The transition band around the left edge is only a pixel or two wide.
    window = alpha[540, 395:406]
    intermediate = int(np.sum((window > 0) & (window < 255)))
    assert intermediate <= 4


# --- resolution scaling: preview vs export agree geometrically ------------

def test_resolution_scaling_matches():
    bp = (30000, 30000, 30000)

    def masked_fraction_and_feather(long_side):
        img = _base_plate(long_side, long_side, bp)
        q = long_side // 4
        img = _with_square(img, q, 3 * q, q, 3 * q, (62000, 62000, 62000))
        alpha = compute_sprocket_alpha(img, bp)
        assert alpha is not None
        frac = float((alpha == 255).sum()) / alpha.size
        feather_band = int(((alpha > 0) & (alpha < 255)).sum())
        return frac, feather_band

    f1, band1 = masked_fraction_and_feather(SPROCKET_REF_LONG)          # 1080
    f4, band4 = masked_fraction_and_feather(SPROCKET_REF_LONG * 4)      # 4320

    # Solid-white area fraction is resolution-independent (same geometry).
    assert abs(f1 - f4) < 0.02
    # The feather band is a fixed px width -> its pixel COUNT grows ~ with the
    # perimeter (×4) and the band width (×4), i.e. ~×16 for a 4× upscale.
    assert 8 < (band4 / max(band1, 1)) < 32


# --- the composite ---------------------------------------------------------

def test_apply_paints_white_and_blends():
    rgb = np.full((4, 4, 3), 5000, dtype=np.uint16)
    alpha = np.zeros((4, 4), dtype=np.uint8)
    alpha[0, 0] = 255      # full white
    alpha[1, 1] = 128      # half blend
    out = apply_sprocket_mask(rgb, alpha)
    assert out.dtype == np.uint16
    assert tuple(out[0, 0]) == (65535, 65535, 65535)          # painted white
    assert tuple(out[2, 2]) == (5000, 5000, 5000)             # untouched
    expected = round(5000 * (1 - 128 / 255) + 65535 * (128 / 255))
    assert abs(int(out[1, 1, 0]) - expected) <= 1             # feathered blend


def test_apply_none_alpha_is_noop():
    rgb = np.full((4, 4, 3), 5000, dtype=np.uint16)
    assert apply_sprocket_mask(rgb, None) is rgb


def test_apply_shape_mismatch_is_noop():
    rgb = np.full((4, 4, 3), 5000, dtype=np.uint16)
    alpha = np.zeros((8, 8), dtype=np.uint8)
    assert apply_sprocket_mask(rgb, alpha) is rgb


# --- env overrides tune the threshold -------------------------------------

def test_env_overrides(monkeypatch):
    bp = (30000, 30000, 30000)
    img = _base_plate(1080, 1080, bp)
    img = _with_square(img, 100, 300, 100, 300, (62000, 62000, 62000))  # hole
    img = _with_square(img, 500, 800, 500, 800, (40000, 40000, 40000))  # 31% of gap
    assert compute_sprocket_alpha(img, bp)[650, 650] == 0      # default: half the gap
    monkeypatch.setenv("FREECCR_SPROCKET_GAP_FRAC", "0.2")
    assert compute_sprocket_alpha(img, bp)[650, 650] == 255    # lower threshold
    monkeypatch.delenv("FREECCR_SPROCKET_GAP_FRAC")
    monkeypatch.setenv("FREECCR_SPROCKET_MIN_CLEAR_D", "0.5")  # gap here is ~0.32
    assert compute_sprocket_alpha(img, bp) is None


# --- backend flag + convert integration -----------------------------------

def test_backend_flag_defaults_off():
    from core.ccr_backend import ccr_backend
    assert ccr_backend.sprocket_mask_white is False


class _Stub:
    """Minimal ccr_image surface for a preview-path B/W-point conversion."""
    def __init__(self, raw):
        self.resized_raw = raw
        self.fine_rotation_angle = 0
        self.horizontal_mirrored = False
        self.vertical_mirrored = False
        self.crop_rect = None
        self.crop_angle = 0.0
        self.rotation_angle = 0
        self.sprocket_alpha = None
        self.contrast_base = 0
        self.temperature_base = 0
        self.exposure_base = 0.0
        self._ws_windowed = False


def test_convert_stores_alpha_black_point_only():
    from core.ccr_processor import ccr_normalize_with_bwpoint
    bp = (30000, 30000, 30000)
    img = _base_plate(600, 900, bp)
    img = _with_square(img, 200, 400, 350, 550, (62000, 62000, 62000))
    stub = _Stub(img)
    # Black-point-only (default-slope) preview conversion, output_path=None.
    ccr_normalize_with_bwpoint(stub, bp, None)
    assert stub.sprocket_alpha is not None
    assert stub.sprocket_alpha.shape == (600, 900)


def test_convert_no_hole_leaves_alpha_none():
    from core.ccr_processor import ccr_normalize_with_bwpoint
    bp = (30000, 30000, 30000)
    img = _base_plate(600, 900, bp)          # no clear-film region
    stub = _Stub(img)
    ccr_normalize_with_bwpoint(stub, bp, None)
    assert stub.sprocket_alpha is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

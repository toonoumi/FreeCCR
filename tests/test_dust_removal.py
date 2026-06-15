#!/usr/bin/env python3
"""
Tests for the dust detection + inpainting module (src/core/dust_removal.py).

Pure numpy/cv2 — no Qt required. Synthesizes converted-positive-like images
with known injected dust (bright specks and curly streaks) and asserts that
detection finds them, inpainting removes them while preserving untouched
pixels, and the vector strokes round-trip across resolutions.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2  # noqa: E402
from core.dust_removal import (  # noqa: E402
    detect_dust, inpaint_dust, rasterize_strokes, clean_strokes, BRUSH_NORM,
)


# --- synthetic image helpers -----------------------------------------------
def _clean_positive(h=400, w=600, seed=7, noise=500.0):
    """A smooth-ish converted positive: a gentle 2D gradient + mild grain,
    sitting mid-range in 16-bit (like blue sky)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 30000 + 12000 * (xx / w) + 6000 * (yy / h)
    img = np.stack([base * 0.95, base, base * 1.08], axis=-1)   # cool, sky-like
    img += rng.normal(0, noise, img.shape)
    return np.clip(img, 1000, 64000).astype(np.uint16)


def _inject_spot(img, cy, cx, radius=2, amp=18000):
    """Add a bright circular speck (dust on the positive reads bright)."""
    img = img.copy()
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
    vals = img[m].astype(np.int32) + amp
    img[m] = np.clip(vals, 0, 65535).astype(np.uint16)
    return img, m


def _inject_streak(img, amp=16000, thickness=2):
    """Add a thin curly bright streak (a sine squiggle), like a hair/scratch."""
    img = img.copy()
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    xs = np.arange(w // 4, 3 * w // 4)
    ys = (h // 2 + (h // 8) * np.sin(xs / 18.0)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], False, 255, thickness=thickness)
    m = mask.astype(bool)
    vals = img[m].astype(np.int32) + amp
    img[m] = np.clip(vals, 0, 65535).astype(np.uint16)
    return img, m


# --- detection -------------------------------------------------------------
class TestDetect:
    def test_single_spot(self):
        img = _clean_positive()
        dusty, _ = _inject_spot(img, 200, 300, radius=3)
        strokes = detect_dust(dusty, "med")
        assert len(strokes) >= 1
        # at least one detected point lands near the injected centre
        h, w = dusty.shape[:2]
        hits = [(p[0] * w, p[1] * h) for s in strokes for p in s["points"]]
        assert any(abs(px - 300) < 12 and abs(py - 200) < 12 for px, py in hits)

    def test_multiple_spots(self):
        img = _clean_positive()
        for (cy, cx) in [(60, 80), (150, 420), (320, 250), (250, 540)]:
            img, _ = _inject_spot(img, cy, cx, radius=2)
        strokes = detect_dust(img, "med")
        assert len(strokes) >= 4

    def test_curly_streak_detected_and_covered(self):
        img = _clean_positive()
        dusty, streak = _inject_streak(img)
        strokes = detect_dust(dusty, "med")
        assert any(s["kind"] == "scratch" for s in strokes)
        h, w = dusty.shape[:2]
        covered = rasterize_strokes(strokes, h, w, pad_px=2).astype(bool)
        recall = (covered & streak).sum() / max(1, streak.sum())
        assert recall >= 0.8

    def test_clean_image_few_false_positives(self):
        img = _clean_positive(seed=21)
        assert len(detect_dust(img, "med")) <= 8

    def test_sensitivity_ordering(self):
        img = _clean_positive(seed=3)
        for (cy, cx) in [(100, 100), (200, 300), (300, 500)]:
            img, _ = _inject_spot(img, cy, cx, radius=2, amp=9000)  # faint
        low = detect_dust(img, "low")
        high = detect_dust(img, "high")
        assert len(high) >= len(low)

    def test_large_blob_rejected_by_area(self):
        img = _clean_positive()
        dusty, _ = _inject_spot(img, 200, 300, radius=30)   # far above max_area
        # The big blob must not be auto-detected as dust.
        strokes = detect_dust(dusty, "med")
        h, w = dusty.shape[:2]
        for s in strokes:
            for p in s["points"]:
                assert not (abs(p[0] * w - 300) < 25 and abs(p[1] * h - 200) < 25)

    def test_round_spot_classified_as_spot(self):
        img = _clean_positive()
        dusty, _ = _inject_spot(img, 200, 300, radius=3)
        h, w = dusty.shape[:2]
        near = [s for s in detect_dust(dusty, "med") for p in [s["points"][0]]
                if abs(p[0] * w - 300) < 12 and abs(p[1] * h - 200) < 12]
        assert near and all(s["kind"] == "spot" for s in near)


# --- inpainting ------------------------------------------------------------
class TestInpaint:
    def test_removes_spot(self):
        img = _clean_positive()
        dusty, m = _inject_spot(img, 200, 300, radius=3)
        strokes = detect_dust(dusty, "med")
        healed = inpaint_dust(dusty, strokes, grain=0.0)
        # healed region must be closer to the clean original than the dusty one
        reg = (slice(190, 211), slice(290, 311))
        mae_dust = np.abs(dusty[reg].astype(np.int32) - img[reg]).mean()
        mae_heal = np.abs(healed[reg].astype(np.int32) - img[reg]).mean()
        assert mae_heal < mae_dust

    def test_preserves_untouched_pixels(self):
        img = _clean_positive()
        dusty, _ = _inject_spot(img, 200, 300, radius=3)
        strokes = detect_dust(dusty, "med")
        healed = inpaint_dust(dusty, strokes, grain=0.0)
        # a corner far from any stroke is byte-for-byte unchanged
        np.testing.assert_array_equal(healed[:20, :20], dusty[:20, :20])

    def test_empty_strokes_is_identity(self):
        img = _clean_positive()
        assert inpaint_dust(img, []) is img

    def test_dtype_and_shape(self):
        img = _clean_positive()
        dusty, _ = _inject_spot(img, 100, 100, radius=2)
        out = inpaint_dust(dusty, detect_dust(dusty, "med"))
        assert out.dtype == np.uint16
        assert out.shape == img.shape

    def test_gradient_fidelity_no_halo(self):
        # A thin horizontal scratch across a vertical gradient should heal to
        # the exact linear interpolation (harmonic fill), with no halo.
        h, w = 300, 300
        yy = np.mgrid[0:h, 0:w][0]
        base = (2000 + (60000 - 2000) * yy / h).astype(np.uint16)
        img = np.stack([base, base, base], axis=-1)
        dusty = img.copy()
        dusty[148:151, 30:270] = 65000              # bright horizontal scratch
        stroke = {"points": [[30 / w, 149 / h], [269 / w, 149 / h]],
                  "radius": 3 / max(h, w), "connect": True,
                  "kind": "scratch", "source": "manual"}
        healed = inpaint_dust(dusty, [stroke], grain=0.0)
        band = healed[146:153, 40:260].astype(np.int32)
        ref = img[146:153, 40:260].astype(np.int32)
        assert np.abs(band - ref).max() < 600        # < ~1% of full range

    def test_grain_is_deterministic(self):
        img = _clean_positive()
        dusty, _ = _inject_spot(img, 200, 300, radius=4)
        strokes = detect_dust(dusty, "med")
        a = inpaint_dust(dusty, strokes, grain=0.4)
        b = inpaint_dust(dusty, strokes, grain=0.4)
        np.testing.assert_array_equal(a, b)


# --- rasterization / resolution independence -------------------------------
class TestRasterize:
    def test_resolution_independent(self):
        strokes = [
            {"points": [[0.3, 0.4]], "radius": 0.02, "connect": False,
             "kind": "spot", "source": "manual"},
            {"points": [[0.6, 0.6], [0.7, 0.65], [0.8, 0.6]], "radius": 0.015,
             "connect": True, "kind": "scratch", "source": "manual"},
        ]
        h, w = 200, 300
        m1 = rasterize_strokes(strokes, h, w)
        m2 = rasterize_strokes(strokes, 2 * h, 2 * w)
        m2_down = cv2.resize(m2, (w, h), interpolation=cv2.INTER_AREA) > 64
        m1b = m1 > 64
        inter = (m1b & m2_down).sum()
        union = (m1b | m2_down).sum()
        assert union > 0 and inter / union >= 0.8

    def test_clean_strokes_drops_malformed(self):
        raw = [
            {"points": [[0.1, 0.2]], "radius": 0.01},          # ok
            {"points": []},                                     # empty -> drop
            "not a dict",                                       # drop
            {"points": [["x", "y"]]},                           # bad coords -> drop
            {"points": [[0.5, 0.5]], "radius": "oops"},         # bad radius -> coerced
        ]
        out = clean_strokes(raw)
        assert len(out) == 2
        assert out[0]["source"] == "manual"        # default filled in
        assert out[1]["radius"] > 0


def test_brush_constants_present():
    assert set(BRUSH_NORM) == {"S", "M", "L"}
    assert BRUSH_NORM["S"] < BRUSH_NORM["M"] < BRUSH_NORM["L"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

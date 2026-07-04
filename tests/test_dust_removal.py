#!/usr/bin/env python3
"""
Tests for the Dust Removal feature.

Dust edits are stored as NORMALIZED spots on the image
({kind, pts:[[x,y],...], r}) and healed at render time by
ccr_processor.apply_dust_removal (clone-heal patch fill, 16-bit native, with a
cv2.inpaint diffusion fallback; masked-only feathered composite). The AI
detector (dust_detect) only finds dust; the fill is the same path.
See spec/dust-removal.md.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

import cv2  # noqa: E402
from core import catalog, dust_detect  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from core.ccr_processor import (apply_dust_removal,  # noqa: E402
                                rasterize_dust_mask)


def _flat_with_speck(h=100, w=100, base=30000, speck=60000, cx=50, cy=50, r=4):
    """Flat gray image with one bright circular speck near the center."""
    img = np.full((h, w, 3), base, dtype=np.uint16)
    cv2.circle(img, (cx, cy), r, (speck, speck, speck), -1)
    return img


# --- rasterize_dust_mask ----------------------------------------------------
class TestRasterize:
    def test_centered_spot_is_a_filled_circle(self):
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.1}
        mask = rasterize_dust_mask([spot], 100, 100)
        assert mask.dtype == np.uint8
        assert mask[50, 50] == 255          # center filled
        assert mask[50, 70] == 0            # well outside r_px=10
        area = int((mask > 0).sum())
        assert abs(area - np.pi * 100) < 60  # ~ pi*r_px^2, r_px = 10

    def test_resolution_independence(self):
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.1}
        a1 = int((rasterize_dust_mask([spot], 100, 100) > 0).sum())
        a2 = int((rasterize_dust_mask([spot], 200, 200) > 0).sum())
        # Same normalized spot covers ~4x the pixels at 2x resolution.
        assert 3.5 < a2 / a1 < 4.6

    def test_empty_spots_is_blank(self):
        mask = rasterize_dust_mask([], 50, 50)
        assert mask.shape == (50, 50)
        assert not mask.any()


# --- apply_dust_removal -----------------------------------------------------
class TestApplyDustRemoval:
    def test_identity_returns_same_object(self):
        img = _flat_with_speck()
        assert apply_dust_removal(img, []) is img
        assert apply_dust_removal(img, None) is img

    def test_speck_removed_and_far_pixels_untouched(self):
        img = _flat_with_speck(base=30000, speck=60000, cx=50, cy=50, r=4)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}
        out = apply_dust_removal(img, [spot])
        assert out.dtype == np.uint16
        assert out is not img                       # new array (non-destructive)
        # The speck is filled toward the flat surround.
        assert abs(int(out[50, 50, 0]) - 30000) < 15000
        assert int(out[50, 50, 0]) < 55000
        # A corner far from the mask is bit-for-bit unchanged.
        assert np.array_equal(out[0:10, 0:10], img[0:10, 0:10])

    def test_input_not_mutated(self):
        img = _flat_with_speck()
        before = img.copy()
        apply_dust_removal(img, [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}])
        assert np.array_equal(img, before)

    def test_fill_is_16bit_native_on_flat_field(self):
        # 30000 is NOT representable through an 8-bit round trip (the old
        # cv2.inpaint path quantized the fill to multiples of 257 -> 30069).
        # The clone heal copies real 16-bit pixels, so a flat field heals to
        # exactly the surrounding value.
        img = _flat_with_speck(base=30000, speck=60000, cx=50, cy=50, r=4)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}
        out = apply_dust_removal(img, [spot])
        core = out[47:54, 47:54]  # deep inside the healed hole (alpha == 1)
        assert int(np.abs(core.astype(np.int32) - 30000).max()) <= 1

    def test_heal_preserves_grain(self):
        # On a grainy background the fill must carry real texture, not the
        # smooth averaged patch diffusion inpainting produces (the round,
        # fan-like artifact). Healed-region noise must stay comparable to the
        # surround's.
        rng = np.random.default_rng(7)
        img = np.clip(rng.normal(30000, 3000, (100, 100, 3)), 0,
                      65535).astype(np.uint16)
        cv2.circle(img, (50, 50), 4, (60000, 60000, 60000), -1)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}
        out = apply_dust_removal(img, [spot])
        healed = out[46:55, 46:55].astype(np.float32)
        yy, xx = np.mgrid[0:100, 0:100]
        ann = (np.hypot(yy - 50, xx - 50) > 15) & (np.hypot(yy - 50, xx - 50) < 30)
        surround = out[ann].astype(np.float32)
        # Tone lands on the surround; grain survives (Telea gives ~0 std here).
        assert abs(float(healed.mean()) - float(surround.mean())) < 2000
        assert float(healed.std()) > 0.4 * float(surround.std())

    def test_heal_continues_gradient(self):
        # A linear gradient must continue through the patch (the tone
        # correction interpolates the ring differences), not flatten into a
        # single-toned blob.
        ramp = (10000 + np.arange(100, dtype=np.float32) * 400)
        img = np.broadcast_to(ramp[None, :, None], (100, 100, 3)).astype(np.uint16).copy()
        expected = img.copy()
        cv2.circle(img, (50, 50), 4, (60000, 60000, 60000), -1)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}
        out = apply_dust_removal(img, [spot])
        yy, xx = np.mgrid[0:100, 0:100]
        hole = np.hypot(yy - 50, xx - 50) <= 8
        err = np.abs(out[hole].astype(np.float32) - expected[hole].astype(np.float32))
        assert float(err.max()) < 2500

    def test_fallback_still_fills_when_no_clean_source(self):
        # A spot so large no clean source window exists anywhere must fall
        # back to diffusion inpainting rather than leaving the speck.
        img = _flat_with_speck(base=30000, speck=60000, cx=50, cy=50, r=30)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.4}
        out = apply_dust_removal(img, [spot])
        assert int(out[50, 50, 0]) < 45000  # moved toward the surround

    def test_traced_hair_leaves_no_bright_ghost(self):
        # THE reported artifact: tracing a bright warm hair with a brush about
        # its own width left a yellow-green ghost of the whole stroke. The
        # hair's edges leak past the mask; without the guard gap + robust ring
        # rejection they poisoned the tone correction (R/G lifted, B dropped)
        # across the entire stroke — and the bbox-diagonal source-search
        # minimum forced long strokes into the diffusion fallback, which
        # ghosts the same way.
        sky = np.array([20000, 25000, 40000], np.float32)
        img = np.broadcast_to(sky.astype(np.uint16), (200, 200, 3)).copy()
        # A "hair" much wider than the brush: bright/warm vs the blue sky.
        cv2.line(img, (30, 100), (170, 100), (62000, 60000, 30000), 13)
        # Brush stroke tracing the hair core, narrower than the hair.
        spot = {"kind": "brush",
                "pts": [[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]], "r": 2.0 / 200}
        out = apply_dust_removal(img, [spot])
        core = out[99:102, 60:140].astype(np.float32).reshape(-1, 3)
        err = np.abs(core.mean(axis=0) - sky)
        assert float(err.max()) < 5000   # fill stays sky-toned along the stroke

    def test_feather_alpha_ramps_inward(self):
        # The fill blends over a smooth inward ramp: 0 at the hole boundary,
        # 1 in the core, exactly 0 outside the mask; a 1-px feather is an
        # essentially hard (fully filled) edge for tight traces.
        from core.ccr_processor import _feather_alpha
        mask = np.zeros((60, 60), np.uint8)
        cv2.circle(mask, (30, 30), 20, 255, -1)
        a = _feather_alpha(mask, np.full((60, 60), 8, np.uint8))
        assert a[30, 30] == 1.0                    # core fully filled
        assert a[30, 51] == 0.0                    # outside untouched
        edge, mid = a[30, 49], a[30, 45]
        assert 0.0 < edge < 0.35                   # soft start at the rim
        assert edge < mid < 1.0                    # monotone ramp
        hard = _feather_alpha(mask, np.ones((60, 60), np.uint8))
        assert hard[30, 49] > 0.9                  # 1px feather ~ hard fill

    def test_feather_param_softens_rim(self, monkeypatch):
        # The user Feather setting widens the fill's cross-fade: with a wide
        # feather the hole's clean rim stays close to the ORIGINAL pixels
        # (low alpha), with feather 0 the rim is the clone (hard edge).
        # Auto-mask is pinned off: it would (correctly) shrink this generous
        # dab to the speck, and the dab rim this test measures would no longer
        # be part of the hole (spec/dust-auto-mask.md §7).
        monkeypatch.setenv("FREECCR_DUST_AUTOMASK", "0")
        rng = np.random.default_rng(5)
        img = np.clip(rng.normal(30000, 2000, (200, 200, 3)), 0,
                      65535).astype(np.uint16)
        cv2.circle(img, (100, 100), 6, (60000, 60000, 60000), -1)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}  # mask r=16
        hard = apply_dust_removal(img, [spot], feather=0.0)
        soft = apply_dust_removal(img, [spot], feather=0.08)
        yy, xx = np.mgrid[0:200, 0:200]
        rim = (np.hypot(yy - 100, xx - 100) > 13.5) & \
              (np.hypot(yy - 100, xx - 100) < 15.5)
        d_hard = np.abs(hard[rim].astype(np.float32) - img[rim]).mean()
        d_soft = np.abs(soft[rim].astype(np.float32) - img[rim]).mean()
        assert d_soft < 0.5 * d_hard
        # The speck itself is gone in both.
        assert int(hard[100, 100, 0]) < 40000
        assert int(soft[100, 100, 0]) < 40000

    def test_wide_feather_never_blends_defect_back(self):
        # Defect-like pixels are force-filled whatever the feather: a tight
        # hair trace with a huge feather must still remove the hair.
        sky = np.array([20000, 25000, 40000], np.float32)
        img = np.broadcast_to(sky.astype(np.uint16), (200, 200, 3)).copy()
        cv2.line(img, (30, 100), (170, 100), (62000, 60000, 30000), 13)
        spot = {"kind": "brush",
                "pts": [[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]], "r": 2.0 / 200}
        out = apply_dust_removal(img, [spot], feather=0.02)
        core = out[99:102, 60:140].astype(np.float32).reshape(-1, 3)
        assert float(np.abs(core.mean(axis=0) - sky).max()) < 5000

    def test_underscoped_dab_keeps_tone(self):
        # A click smaller than the speck (the small dot ghost): the speck's
        # edge leaks past the mask but must not lift the fill's tone.
        sky = np.array([20000, 25000, 40000], np.float32)
        img = np.broadcast_to(sky.astype(np.uint16), (100, 100, 3)).copy()
        cv2.circle(img, (50, 50), 6, (62000, 60000, 30000), -1)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.04}  # mask r=4 < speck r=6
        out = apply_dust_removal(img, [spot])
        center = out[49:52, 49:52].astype(np.float32).reshape(-1, 3).mean(axis=0)
        assert float(np.abs(center - sky).max()) < 6000


# --- auto-mask (shrink a generous selection to its outliers) -----------------
# spec/dust-auto-mask.md: a dab loosely circling a defect heals only the
# defect (+ a small buffer); the clean remainder of the selection is
# preserved bit-for-bit and doubles as local source area, which keeps heals
# on the correct side of high-contrast edges.
from core.ccr_processor import _automask_shrink  # noqa: E402


class TestAutoMaskShrink:
    """_automask_shrink directly: what shrinks, what stays whole."""

    @staticmethod
    def _dab_mask(h=100, w=100, cx=50, cy=50, r=15):
        mask = np.zeros((h, w), np.uint8)
        cv2.circle(mask, (cx, cy), r, 255, -1)
        return mask

    def test_shrink_targets_outliers_only(self):
        img = _flat_with_speck(base=30000, speck=60000, cx=50, cy=50, r=4)
        mask = self._dab_mask(r=15)
        out = _automask_shrink(img, mask)
        assert out[50, 50] == 255                    # speck stays masked
        assert out[50, 60] == 0                      # clean dab pixel released
        # Shrunken mask stays within speck + buffer (r=4 + ~2px slack).
        yy, xx = np.mgrid[0:100, 0:100]
        assert not out[np.hypot(yy - 50, xx - 50) > 7].any()

    def test_dark_speck_variant(self):
        img = _flat_with_speck(base=40000, speck=2000, cx=50, cy=50, r=4)
        out = _automask_shrink(img, self._dab_mask(r=15))
        assert out[50, 50] == 255
        assert out[50, 60] == 0

    def test_tight_trace_keeps_whole_stroke(self):
        # Brush ≈ defect width: nearly everything under the stroke is defect,
        # so the clean-fraction gate must keep the whole stroke.
        sky = np.array([20000, 25000, 40000], np.uint16)
        img = np.broadcast_to(sky, (200, 200, 3)).copy()
        cv2.line(img, (30, 100), (170, 100), (62000, 60000, 30000), 13)
        mask = np.zeros((200, 200), np.uint8)
        cv2.line(mask, (40, 100), (160, 100), 255, 9)  # inside the hair
        out = _automask_shrink(img, mask)
        assert np.array_equal(out, mask)

    def test_clean_dab_stays_whole(self):
        rng = np.random.default_rng(11)
        img = np.clip(rng.normal(30000, 2000, (100, 100, 3)), 0,
                      65535).astype(np.uint16)
        mask = self._dab_mask(r=15)
        out = _automask_shrink(img, mask)
        assert np.array_equal(out, mask)  # no confident outlier -> unchanged

    def test_edge_straddling_dab_flags_only_the_speck(self):
        # Bimodal surround (the bright-face/dark-background case): both sides
        # of the edge are legit ring modes; only the speck is an outlier.
        rng = np.random.default_rng(3)
        img = np.clip(rng.normal(8000, 500, (100, 100, 3)), 0,
                      65535).astype(np.uint16)
        img[:, 50:] = np.clip(rng.normal(50000, 500, (100, 100 - 50, 3)),
                              0, 65535).astype(np.uint16)
        cv2.circle(img, (60, 50), 3, (65000, 65000, 65000), -1)  # bright side
        mask = self._dab_mask(cx=52, cy=50, r=16)   # straddles the edge
        out = _automask_shrink(img, mask)
        assert out[50, 60] == 255                    # speck masked
        assert not out[:, :49].any()                 # dark side fully released
        yy, xx = np.mgrid[0:100, 0:100]
        assert not out[np.hypot(yy - 50, xx - 60) > 6].any()

    def test_env_knob_disables(self, monkeypatch):
        monkeypatch.setenv("FREECCR_DUST_AUTOMASK", "0")
        img = _flat_with_speck(base=30000, speck=60000, cx=50, cy=50, r=4)
        mask = self._dab_mask(r=15)
        assert np.array_equal(_automask_shrink(img, mask), mask)


class TestAutoMaskPipeline:
    """apply_dust_removal end-to-end with the shrink pass."""

    def test_generous_dab_preserves_clean_selection_pixels(self):
        rng = np.random.default_rng(9)
        img = np.clip(rng.normal(30000, 2000, (100, 100, 3)), 0,
                      65535).astype(np.uint16)
        cv2.circle(img, (50, 50), 3, (62000, 62000, 62000), -1)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.15}  # dab r=15
        out = apply_dust_removal(img, [spot])
        # Speck healed to the surround.
        assert abs(int(out[50, 50, 0]) - 30000) < 10000
        # Clean selection pixels (inside the dab, away from the speck) are
        # bit-for-bit original — the whole point of the auto-mask.
        yy, xx = np.mgrid[0:100, 0:100]
        keep = (np.hypot(yy - 50, xx - 50) < 15) & \
               (np.hypot(yy - 50, xx - 50) > 8)
        assert np.array_equal(out[keep], img[keep])

    def test_edge_straddling_dab_heals_on_the_right_side(self):
        # Speck on the bright side of a hard edge, dab straddling the edge:
        # the fill must land at bright-side statistics (not pulled dark) and
        # the dark side of the selection must stay bit-for-bit.
        rng = np.random.default_rng(4)
        img = np.clip(rng.normal(8000, 500, (100, 100, 3)), 0,
                      65535).astype(np.uint16)
        img[:, 50:] = np.clip(rng.normal(50000, 500, (100, 50, 3)),
                              0, 65535).astype(np.uint16)
        cv2.circle(img, (60, 50), 3, (65000, 65000, 65000), -1)
        spot = {"kind": "brush", "pts": [[0.52, 0.5]], "r": 0.16}
        out = apply_dust_removal(img, [spot])
        healed = out[48:53, 58:63].astype(np.float32)
        assert abs(float(healed.mean()) - 50000) < 4000   # bright-side fill
        # Dark side under the dab: untouched.
        yy, xx = np.mgrid[0:100, 0:100]
        dark = (np.hypot(yy - 50, xx - 52) < 16) & (xx < 49)
        assert np.array_equal(out[dark], img[dark])

    def test_two_specks_one_dab(self):
        rng = np.random.default_rng(6)
        img = np.clip(rng.normal(30000, 1500, (100, 100, 3)), 0,
                      65535).astype(np.uint16)
        cv2.circle(img, (43, 50), 2, (62000, 62000, 62000), -1)
        cv2.circle(img, (57, 50), 2, (1000, 1000, 1000), -1)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.15}
        out = apply_dust_removal(img, [spot])
        assert abs(int(out[50, 43, 0]) - 30000) < 9000
        assert abs(int(out[50, 57, 0]) - 30000) < 9000
        # The strip between the two specks is original.
        assert np.array_equal(out[48:53, 47:53], img[48:53, 47:53])


# --- apply_adjustments integration (dust runs before the early-return guard) -
class TestApplyAdjustmentsIntegration:
    def test_dust_only_image_still_inpaints(self, tmp_path):
        # Build a bare CCRImage; neutralize bases so apply_adjustments takes the
        # early-return path AFTER dust removal (proving dust runs before it).
        path = str(tmp_path / "x.png")
        cv2.imwrite(path, np.zeros((10, 10, 3), np.uint8))
        img = CCRImage(path)
        img.adjustment_settings = {}
        img.contrast_base = 0
        img.temperature_base = 0
        img.brightness_base = 0
        img.color_profile = "color"
        img.dust_spots = [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}]

        src = _flat_with_speck()
        out = img.apply_adjustments(src)
        # Speck healed even though every slider/base is neutral.
        assert int(out[50, 50, 0]) < 55000
        assert np.array_equal(out[0:10, 0:10], src[0:10, 0:10])

    def test_no_dust_no_change_in_neutral_pipeline(self, tmp_path):
        path = str(tmp_path / "y.png")
        cv2.imwrite(path, np.zeros((10, 10, 3), np.uint8))
        img = CCRImage(path)
        img.adjustment_settings = {}
        img.contrast_base = img.temperature_base = img.brightness_base = 0
        img.dust_spots = []
        src = _flat_with_speck()
        out = img.apply_adjustments(src)
        assert np.array_equal(out, src)


# --- dust_detect.prob_to_spots (model-free) ---------------------------------
def _bright_luma(prob):
    """Luma where high-prob regions read bright (white film dust) on a dark
    surround, so detections pass the bright-speck gate."""
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


class TestProbToSpots:
    def test_threshold_direction(self):
        prob = np.zeros((100, 100), np.float32)
        prob[10:13, 10:13] = 0.5            # a small mid-confidence blob
        luma = _bright_luma(prob)
        # sensitivity 0 -> thr 0.85 -> nothing; 100 -> thr 0.25 -> detected.
        assert dust_detect.prob_to_spots(prob, luma, 0) == []
        spots = dust_detect.prob_to_spots(prob, luma, 100)
        assert len(spots) == 1
        assert spots[0]["kind"] == "auto"
        x, y = spots[0]["pts"][0]
        assert 0.08 < x < 0.16 and 0.08 < y < 0.16
        assert spots[0]["r"] > 0

    def test_size_gate_drops_large_blobs(self):
        # max_blob = MAX_BLOB(400) * max(h,w) / 2000 = 20 px for a 100x100 map.
        prob = np.zeros((100, 100), np.float32)
        prob[10:13, 10:13] = 0.9            # 9 px  -> kept
        prob[50:62, 50:62] = 0.9            # 144 px -> dropped
        spots = dust_detect.prob_to_spots(prob, _bright_luma(prob), 50)
        assert len(spots) == 1
        x, y = spots[0]["pts"][0]
        assert x < 0.3 and y < 0.3          # the small blob, not the big one

    def test_blank_prob_yields_nothing(self):
        z = np.zeros((40, 40), np.float32)
        assert dust_detect.prob_to_spots(z, z, 100) == []

    def test_drops_elongated_keeps_compact(self):
        # A thin line is real image structure (bike frame / horizon), not dust:
        # the aspect filter drops it while keeping a compact speck. Guards the
        # AI-artifact fix (see spec/dust-removal.md §5.3/§5.4).
        prob = np.zeros((100, 100), np.float32)
        prob[50, 10:22] = 0.9       # 1x12 thin line (area 12, aspect 12) -> dropped
        prob[80:83, 80:83] = 0.9    # 3x3 compact speck (area 9)          -> kept
        spots = dust_detect.prob_to_spots(prob, _bright_luma(prob), 60)
        assert len(spots) == 1
        x, y = spots[0]["pts"][0]
        assert x > 0.5 and y > 0.5  # the compact speck, not the line

    def test_radius_is_area_equivalent_not_extent(self):
        # A 6x6 compact blob -> radius ~ sqrt(36/pi) ~ 3.4 px (+pad), NOT the
        # 0.5*extent=3 of the old bounding-box sizing blown up by elongation.
        prob = np.zeros((200, 200), np.float32)
        prob[100:106, 100:106] = 0.9
        spots = dust_detect.prob_to_spots(prob, _bright_luma(prob), 60)
        assert len(spots) == 1
        r_px = spots[0]["r"] * 200
        assert 3.0 < r_px < 7.0     # tight circle, no big smudge

    def test_bright_gate_drops_dark_blob(self):
        # Film dust inverts to WHITE specks. A compact, right-sized blob that is
        # DARKER than its surround (e.g. a face on bright sky) is NOT dust and
        # must be rejected — this is what removed a person's head before.
        # 200x200 so the 6x6 blob clears the size gate (max_blob ~ 40 here) and
        # the brightness gate is what actually decides.
        prob = np.zeros((200, 200), np.float32)
        prob[40:46, 40:46] = 0.9                 # detector fires on a 6x6 region
        dark = np.full((200, 200), 0.8, np.float32)
        dark[40:46, 40:46] = 0.2                 # dark blob, bright surround
        assert dust_detect.prob_to_spots(prob, dark, 60) == []
        bright = np.full((200, 200), 0.2, np.float32)
        bright[40:46, 40:46] = 0.9               # bright blob (real dust)
        assert len(dust_detect.prob_to_spots(prob, bright, 60)) == 1


# --- Availability / graceful degradation ------------------------------------
class TestAvailability:
    def test_unavailable_when_onnxruntime_absent(self, monkeypatch):
        # Setting the module to None makes `import onnxruntime` raise ImportError.
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        assert dust_detect.is_available() is False

    def test_model_path_is_under_freeccr(self):
        assert "FreeCCR" in dust_detect.model_path()
        assert dust_detect.model_path().endswith("detector.onnx")


# --- Persistence (catalog) --------------------------------------------------
def _scan_png(tmp_path, name="neg.png", w=120, h=80, seed=3):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(30000, 4000, (h, w, 3)), 0, 65535).astype(np.uint16)
    path = str(tmp_path / name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


class TestPersistence:
    def test_dust_spots_round_trip(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.dust_spots = [{"kind": "brush", "pts": [[0.25, 0.5], [0.3, 0.55]],
                           "r": 0.01},
                          {"kind": "auto", "pts": [[0.7, 0.2]], "r": 0.004}]
        catalog.update_for_images([img], path=cat)
        restored = catalog.create_images_for_path(path, path=cat)
        assert len(restored) == 1
        assert restored[0].dust_spots == img.dust_spots

    def test_old_entry_without_dust_defaults_empty(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.adjustment_settings = {"exposure": 5}
        state = catalog.serialize_image(img)
        del state["dust_spots"]            # simulate a pre-feature catalog entry
        restored = catalog._restore_image(path, state)
        assert restored.dust_spots == []

    def test_dust_only_image_is_not_pristine(self, tmp_path):
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.dust_spots = [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.01}]
        state = catalog.serialize_image(img)
        assert catalog._is_pristine(state) is False

    def test_dust_feather_round_trips_and_defaults(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.dust_spots = [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.01}]
        img.dust_feather = 0.007
        catalog.update_for_images([img], path=cat)
        restored = catalog.create_images_for_path(path, path=cat)
        assert abs(restored[0].dust_feather - 0.007) < 1e-9
        # Pre-feature catalog entries restore to the default.
        state = catalog.serialize_image(img)
        del state["dust_feather"]
        old = catalog._restore_image(path, state)
        assert abs(old.dust_feather - 0.003) < 1e-9


# --- Undo -------------------------------------------------------------------
class TestUndo:
    def test_capture_and_pop_restore_dust_spots(self, tmp_path):
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.dust_spots = [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.01}]
        img.push_undo_state()
        # Mutate after snapshot.
        img.dust_spots.append({"kind": "auto", "pts": [[0.1, 0.1]], "r": 0.005})
        assert len(img.dust_spots) == 2
        assert img.pop_undo_state()
        assert img.dust_spots == [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.01}]

    def test_snapshot_is_independent_deep_copy(self, tmp_path):
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.dust_spots = [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.01}]
        img.push_undo_state()
        # Mutate a NESTED structure in place; the snapshot must not change.
        img.dust_spots[0]["pts"].append([0.6, 0.6])
        img.pop_undo_state()
        assert img.dust_spots[0]["pts"] == [[0.5, 0.5]]


# --- DustRemovalPanel wiring (headless, with stubs) -------------------------
class _StubPreview:
    def __init__(self):
        self.dust_mode = True
        self.current_idx = None
        self.brush = None

    def set_dust_brush_size(self, r):
        self.brush = r

    def dust_undo_last(self):
        return False

    def dust_clear_all(self):
        return False


class _StubMain:
    def __init__(self):
        self.toggled = None

    def toggle_dust_removal(self, on):
        self.toggled = on


class TestPanelWiring:
    def test_panel_builds_without_onnxruntime(self):
        from widgets.dust_panel import DustRemovalPanel
        # Building the panel must never import onnxruntime / raise.
        panel = DustRemovalPanel(_StubMain(), _StubPreview())
        assert panel is not None

    def test_brush_slider_drives_canvas(self):
        from widgets.dust_panel import (DustRemovalPanel, brush_r_to_slider,
                                        slider_to_brush_r)
        prev = _StubPreview()
        panel = DustRemovalPanel(_StubMain(), prev)
        panel._on_brush_changed(brush_r_to_slider(0.030))
        # Log-step quantization: nearest step is within ~1% of the target r.
        assert abs(prev.brush - 0.030) < 0.030 * 0.015
        panel.sync_brush_size(0.05)            # canvas -> slider, no feedback loop
        assert panel.brush_slider.value() == brush_r_to_slider(0.05)
        # Mapping round-trips through the slider's integer steps.
        v = brush_r_to_slider(0.012)
        assert brush_r_to_slider(slider_to_brush_r(v)) == v

    def test_brush_reaches_fine_sizes(self):
        # The slider bottom is 0.05% of image width (~3 px radius on a 6000 px
        # scan) — finer than the old 0.2% floor; the 20% top is unchanged.
        from widgets.dust_panel import (DustRemovalPanel, BRUSH_STEPS,
                                        DUST_BRUSH_R_MIN, DUST_BRUSH_R_MAX)
        assert DUST_BRUSH_R_MIN <= 0.0005
        prev = _StubPreview()
        panel = DustRemovalPanel(_StubMain(), prev)
        assert panel.brush_slider.minimum() == 0
        assert panel.brush_slider.maximum() == BRUSH_STEPS
        panel._on_brush_changed(0)
        assert abs(prev.brush - DUST_BRUSH_R_MIN) < 1e-12
        panel._on_brush_changed(BRUSH_STEPS)
        assert abs(prev.brush - DUST_BRUSH_R_MAX) < 1e-12

    def test_done_button_exits_mode(self):
        from widgets.dust_panel import DustRemovalPanel
        main = _StubMain()
        panel = DustRemovalPanel(main, _StubPreview())
        panel._on_done()
        assert main.toggled is False

    def test_cancel_and_shutdown_are_safe_with_no_jobs(self):
        from widgets.dust_panel import DustRemovalPanel
        panel = DustRemovalPanel(_StubMain(), _StubPreview())
        panel.cancel_jobs()
        panel.shutdown()   # must not raise with no threads running

    def test_feather_slider_writes_image_and_label(self):
        from widgets.dust_panel import DustRemovalPanel
        from core.ccr_backend import ccr_backend

        class _Img:
            dust_feather = 0.003
            dust_spots = []

        img = _Img()
        saved = ccr_backend.images
        ccr_backend.images = [img]
        try:
            prev = _StubPreview()
            prev.current_idx = 0
            panel = DustRemovalPanel(_StubMain(), prev)
            panel.feather_slider.setValue(60)  # emits valueChanged
            assert panel.feather_value.text() == "0.60%"
            panel._apply_feather()             # bypass the debounce timer
            assert abs(img.dust_feather - 0.006) < 1e-9
        finally:
            ccr_backend.images = saved

    def test_detect_all_no_targets_is_safe(self):
        from widgets.dust_panel import DustRemovalPanel, _DetectAllWorker  # noqa: F401
        from core.ccr_backend import ccr_backend
        saved = ccr_backend.images
        ccr_backend.images = []
        try:
            panel = DustRemovalPanel(_StubMain(), _StubPreview())
            assert hasattr(panel, "detect_all_btn")
            panel._on_detect_all()                 # no convertible images
            assert panel._detecting_all is False    # no batch started
            assert panel._detect_all_thread is None
        finally:
            ccr_backend.images = saved


# --- Ctrl+Z routing in dust mode ---------------------------------------------
class TestUndoRouting:
    def test_ctrl_z_in_dust_mode_undoes_spot_and_keeps_view(self):
        # In dust mode Ctrl+Z must behave like the panel's "Undo last spot"
        # (view preserved), NOT the general undo whose _reset_zoom read as
        # "Ctrl+Z unzoomed" while spotting dust zoomed-in.
        from ui.main_window import MainWindow

        class _Panel:
            called = False

            def _on_undo_last(self):
                self.called = True

        class _Prev:
            dust_mode = True

            def _reset_zoom(self):
                raise AssertionError("dust-mode undo must preserve the view")

        class _Stub:
            image_preview = _Prev()
            dust_panel = _Panel()

        stub = _Stub()
        MainWindow.undo_last_action(stub)
        assert stub.dust_panel.called is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

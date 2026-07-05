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
        # back to diffusion inpainting rather than leaving the speck. A
        # defect this size is beyond the dab's dust scale — whole-stroke
        # engine territory (the automask dab correctly no-ops on it).
        img = _flat_with_speck(base=30000, speck=60000, cx=50, cy=50, r=30)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.4}
        out = apply_dust_removal(img, [spot], method="clone")
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

    def test_feather_param_softens_rim(self):
        # The user Feather setting widens the fill's cross-fade: with a wide
        # feather the hole's clean rim stays close to the ORIGINAL pixels
        # (low alpha), with feather 0 the rim is the clone (hard edge).
        # method="clone": auto-mask would (correctly) shrink this generous
        # dab to the speck, and the dab rim this test measures would no longer
        # be part of the hole (spec/dust-auto-mask.md §7).
        rng = np.random.default_rng(5)
        img = np.clip(rng.normal(30000, 2000, (200, 200, 3)), 0,
                      65535).astype(np.uint16)
        cv2.circle(img, (100, 100), 6, (60000, 60000, 60000), -1)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}  # mask r=16
        hard = apply_dust_removal(img, [spot], feather=0.0, method="clone")
        soft = apply_dust_removal(img, [spot], feather=0.9, method="clone")
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
        out = apply_dust_removal(img, [spot], feather=1.0)
        core = out[99:102, 60:140].astype(np.float32).reshape(-1, 3)
        assert float(np.abs(core.mean(axis=0) - sky).max()) < 5000

    def test_underscoped_dab_keeps_tone(self):
        # A click smaller than the speck (the small dot ghost): under
        # automask the speck is LARGER than the dab's dust scale, so the dab
        # is a no-op (bigger brush / trace / whole-stroke are the remedies).
        # Under the whole-stroke engine the speck's edge leaks past the mask
        # but must not lift the fill's tone (the small-dot-ghost defense).
        sky = np.array([20000, 25000, 40000], np.float32)
        img = np.broadcast_to(sky.astype(np.uint16), (100, 100, 3)).copy()
        cv2.circle(img, (50, 50), 6, (62000, 60000, 30000), -1)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.04}  # mask r=4 < speck r=6
        assert np.array_equal(apply_dust_removal(img, [spot]), img)
        out = apply_dust_removal(img, [spot], method="clone")
        center = out[49:52, 49:52].astype(np.float32).reshape(-1, 3).mean(axis=0)
        assert float(np.abs(center - sky).max()) < 6000


# --- auto-mask contracts (spec/dust-auto-mask.md §1) --------------------------
# Rebuilt suite: each test pins one maintainer-set contract. Film dust is
# WHITE; a dab heals its bright outliers or does NOTHING; samples come from
# WITHIN the stroke; the feather is the stroke's soft edge; spots heal
# independently in order; deliberate whole-area replacement is the trace
# gesture or the Whole-stroke engine.


def _grain(base=30000, sigma=1500, h=100, w=100, seed=9):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(base, sigma, (h, w, 3)), 0,
                   65535).astype(np.uint16)


def _dab(x, y, r, w=100, h=100):
    return {"kind": "brush", "pts": [[x / w, y / h]], "r": r / w}


class TestAutoMaskContracts:
    def test_dab_heals_bright_outlier_only(self):
        # Contracts 1+2: the speck goes; every clean selection pixel is
        # bit-for-bit original.
        img = _grain()
        cv2.circle(img, (50, 50), 3, (62000, 62000, 62000), -1)
        out = apply_dust_removal(img, [_dab(50, 50, 15)])
        assert abs(int(out[50, 50, 0]) - 30000) < 10000
        yy, xx = np.mgrid[0:100, 0:100]
        keep = (np.hypot(yy - 50, xx - 50) < 15) & \
               (np.hypot(yy - 50, xx - 50) > 8)
        assert np.array_equal(out[keep], img[keep])

    def test_clean_dab_is_noop(self):
        # Contract 2: nothing dusty under the dab -> the WHOLE image is
        # bit-for-bit unchanged (no more replaced discs on clean sky).
        img = _grain(seed=11)
        out = apply_dust_removal(img, [_dab(50, 50, 15)])
        assert np.array_equal(out, img)

    def test_dark_content_under_dab_is_preserved(self):
        # Contract 1: dark detail is not dust; only the white speck heals.
        img = _grain(seed=15)
        cv2.circle(img, (43, 50), 2, (62000, 62000, 62000), -1)   # dust
        cv2.circle(img, (57, 50), 2, (1500, 1500, 1500), -1)      # detail
        out = apply_dust_removal(img, [_dab(50, 50, 15)])
        assert abs(int(out[50, 43, 0]) - 30000) < 9000            # healed
        assert np.array_equal(out[47:54, 54:61], img[47:54, 54:61])  # kept

    def test_dark_only_dab_is_noop(self):
        # Contract 1: a dab over only-dark content does nothing under
        # automask; the Whole-stroke engine is the explicit way to remove it.
        img = _grain(seed=16)
        cv2.circle(img, (50, 50), 3, (1500, 1500, 1500), -1)
        spot = _dab(50, 50, 12)
        assert np.array_equal(apply_dust_removal(img, [spot]), img)
        out = apply_dust_removal(img, [spot], method="clone")
        assert abs(int(out[50, 50, 0]) - 30000) < 9000

    def test_dab_samples_within_stroke(self):
        # Contract 3 (the black-disc report): sky pocket surrounded by black
        # rebate; a dab around a speck must fill with in-stroke sky — never
        # rebate — even though rebate windows are the only clean candidates
        # outside the stroke.
        rng = np.random.default_rng(30)
        img = np.clip(rng.normal(2500, 300, (300, 300, 3)), 0,
                      65535).astype(np.uint16)
        img[88:212, 88:212] = np.clip(
            rng.normal(42000, 800, (124, 124, 3)), 0, 65535).astype(np.uint16)
        cv2.circle(img, (150, 150), 3, (65000, 65000, 65000), -1)
        out = apply_dust_removal(
            img, [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 25 / 300}])
        healed = out[144:157, 144:157].astype(np.float32)
        assert float(np.median(healed)) > 34000        # sky, not rebate
        assert abs(int(out[150, 150, 0]) - 42000) < 9000

    def test_edge_straddling_dab_heals_correct_side(self):
        # Contract 3: speck on the bright side of a hard edge; fill lands at
        # bright-side statistics; the dark side of the dab stays bit-exact.
        rng = np.random.default_rng(4)
        img = np.clip(rng.normal(8000, 500, (100, 100, 3)), 0,
                      65535).astype(np.uint16)
        img[:, 50:] = np.clip(rng.normal(50000, 500, (100, 50, 3)),
                              0, 65535).astype(np.uint16)
        cv2.circle(img, (60, 50), 3, (65000, 65000, 65000), -1)
        out = apply_dust_removal(
            img, [{"kind": "brush", "pts": [[0.52, 0.5]], "r": 0.16}])
        healed = out[48:53, 58:63].astype(np.float32)
        assert abs(float(healed.mean()) - 50000) < 4000
        yy, xx = np.mgrid[0:100, 0:100]
        dark = (np.hypot(yy - 50, xx - 52) < 16) & (xx < 49)
        assert np.array_equal(out[dark], img[dark])

    def test_halo_joins_the_heal(self):
        # A speck's soft halo (below the seed threshold) is hysteresis-grown
        # into the heal so no bright ring survives.
        img = np.full((100, 100, 3), 30000, np.uint16)
        yy, xx = np.mgrid[0:100, 0:100]
        halo = np.exp(-((yy - 50.0) ** 2 + (xx - 50.0) ** 2) / (2 * 3.0 ** 2))
        img = np.clip(img + (halo[..., None] * 30000), 0,
                      65535).astype(np.uint16)
        out = apply_dust_removal(img, [_dab(50, 50, 15)])
        win = out[38:63, 38:63].astype(np.int32)
        assert int(win.max()) - 30000 < 1500   # halo gone, no ring left

    def test_feather_softens_stroke_border_effect(self):
        # Contract 4: the feather is the STROKE's soft edge — a defect at the
        # dab border heals less than one at the center; feather 0 heals both.
        base = np.full((200, 200, 3), 30000, np.uint16)

        def scene():
            img = base.copy()
            cv2.circle(img, (100, 100), 2, (62000, 62000, 62000), -1)
            cv2.circle(img, (113, 100), 2, (62000, 62000, 62000), -1)
            return img

        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}  # dab r=16
        img = scene()
        soft = apply_dust_removal(img, [spot], feather=0.6)
        hard = apply_dust_removal(img, [spot], feather=0.0)
        res_center = int(soft[100, 100, 0]) - 30000
        res_border = int(soft[100, 113, 0]) - 30000
        assert res_center < 3000
        assert res_border > res_center + 5000
        assert int(hard[100, 113, 0]) - 30000 < 3000


class TestSequentialReplay:
    def test_adding_a_spot_never_changes_prior_heals(self):
        # Contract 5 (the "clicking adjacent dust affects the neighbors"
        # report): spot A's healed pixels are bit-identical whether or not a
        # later spot B exists next to it.
        img = _grain(seed=19)
        cv2.circle(img, (40, 50), 2, (62000, 62000, 62000), -1)
        cv2.circle(img, (60, 50), 2, (62000, 62000, 62000), -1)
        a = _dab(40, 50, 9)
        b = _dab(60, 50, 9)
        out_a = apply_dust_removal(img, [a])
        out_ab = apply_dust_removal(img, [a, b])
        assert np.array_equal(out_a[:, :50], out_ab[:, :50])  # A side identical
        assert abs(int(out_ab[50, 60, 0]) - 30000) < 9000     # B healed too

    def test_later_spot_may_source_from_healed_area(self):
        # Sequential replay heals on the running result, so two adjacent
        # dabs both converge to the background level.
        img = _grain(seed=23)
        cv2.circle(img, (46, 50), 2, (62000, 62000, 62000), -1)
        cv2.circle(img, (54, 50), 2, (62000, 62000, 62000), -1)
        out = apply_dust_removal(img, [_dab(46, 50, 8), _dab(54, 50, 8)])
        assert abs(int(out[50, 46, 0]) - 30000) < 9000
        assert abs(int(out[50, 54, 0]) - 30000) < 9000


class TestTraceGesture:
    def test_trace_heals_whole_stroke(self):
        # Contract 6: a path much longer than the brush radius means
        # "replace exactly this outline" — the whole stroke heals even when
        # its content matches the (leak-contaminated) local surround.
        sky = np.array([20000, 25000, 40000], np.float32)
        img = np.broadcast_to(sky.astype(np.uint16), (200, 200, 3)).copy()
        cv2.line(img, (30, 100), (170, 100), (62000, 60000, 30000), 13)
        spot = {"kind": "brush",
                "pts": [[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]], "r": 2.0 / 200}
        out = apply_dust_removal(img, [spot])
        core = out[99:102, 60:140].astype(np.float32).reshape(-1, 3)
        assert float(np.abs(core.mean(axis=0) - sky).max()) < 5000

    def test_click_is_a_dab_not_a_trace(self):
        # A single click (no path) over clean grain follows dab semantics:
        # no outliers -> no-op, never a whole-circle replacement.
        img = _grain(seed=27)
        out = apply_dust_removal(img, [_dab(50, 50, 12)])
        assert np.array_equal(out, img)


class TestHealEngines:
    """The Settings heal-method choice (spec/dust-auto-mask.md §3)."""

    def test_whole_stroke_replaces_clean_dab_pixels(self):
        img = _grain(seed=12)
        cv2.circle(img, (50, 50), 3, (62000, 62000, 62000), -1)
        out = apply_dust_removal(img, [_dab(50, 50, 15)], method="clone")
        assert abs(int(out[50, 50, 0]) - 30000) < 10000
        yy, xx = np.mgrid[0:100, 0:100]
        keep = (np.hypot(yy - 50, xx - 50) < 13) & \
               (np.hypot(yy - 50, xx - 50) > 8)
        assert not np.array_equal(out[keep], img[keep])

    def test_inpaint_uses_diffusion(self):
        # The legacy engine fills through the 8-bit Telea path: on a flat
        # 30000 field the fill quantizes to multiples of 257 (30069).
        img = _flat_with_speck(base=30000, speck=60000, cx=50, cy=50, r=4)
        spot = {"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}
        out = apply_dust_removal(img, [spot], method="inpaint")
        core = out[49:52, 49:52].astype(np.int32)
        assert int(np.abs(core - 30069).max()) <= 257 * 2
        assert (core % 257 == 0).all()

    def test_unknown_method_behaves_as_automask(self):
        img = _grain(seed=13)
        out = apply_dust_removal(img, [_dab(50, 50, 15)],
                                 method="not-a-method")
        assert np.array_equal(out, img)  # clean dab -> automask no-op

    def test_backend_setting_drives_apply_adjustments(self, tmp_path):
        from core.ccr_backend import ccr_backend
        path = str(tmp_path / "m.png")
        cv2.imwrite(path, np.zeros((10, 10, 3), np.uint8))
        img = CCRImage(path)
        img.adjustment_settings = {}
        img.contrast_base = img.temperature_base = img.brightness_base = 0
        img.color_profile = "color"
        img.dust_spots = [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 0.08}]
        src = _flat_with_speck()
        prev = getattr(ccr_backend, "dust_method", "automask")
        try:
            ccr_backend.dust_method = "inpaint"
            out = img.apply_adjustments(src)
        finally:
            ccr_backend.dust_method = prev
        core = out[49:52, 49:52].astype(np.int32)
        assert (core % 257 == 0).all()  # the 8-bit Telea fill signature

    def test_tone_gate_applies_to_whole_stroke_too(self):
        # The Whole-stroke engine must not clone the black rebate either:
        # with every clean candidate window on the rebate, the tone gate
        # rejects them all and the heal falls back to rim diffusion (sky).
        rng = np.random.default_rng(31)
        img = np.clip(rng.normal(2500, 300, (300, 300, 3)), 0,
                      65535).astype(np.uint16)
        img[88:212, 88:212] = np.clip(
            rng.normal(42000, 800, (124, 124, 3)), 0, 65535).astype(np.uint16)
        out = apply_dust_removal(
            img, [{"kind": "brush", "pts": [[0.5, 0.5]], "r": 25 / 300}],
            method="clone")
        healed = out[130:170, 130:170].astype(np.float32)
        assert float(np.median(healed)) > 25000  # sky-toned, never black


class TestRealScan:
    """A REAL Portra-400 scan crop (tests/data/real_scan_sky.png — the
    maintainer's DSC07237.ARW converted through the app's own pipeline):
    genuine dust specks on the film-edge band plus clean area. Synthetic
    targets alone repeatedly passed while real scans failed — every gate/
    threshold change must hold on this asset."""

    @staticmethod
    def _asset():
        p = os.path.join(os.path.dirname(__file__), "data",
                         "real_scan_sky.png")
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        assert img is not None and img.dtype == np.uint16
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def test_dab_heals_real_dust(self):
        from core.ccr_processor import dust_spot_effect_px
        img = self._asset()
        h, w = img.shape[:2]
        for x, y in ((111, 74), (131, 74)):     # real specks
            spot = {"kind": "brush", "pts": [[x / w, y / h]], "r": 14 / w}
            assert dust_spot_effect_px(img, spot) > 0   # not a no-op
            out = apply_dust_removal(img, [spot])
            ring = img[y-25:y+25, x-25:x+25].astype(np.float32).mean(axis=2)
            med = float(np.median(ring))
            before = img[y-6:y+6, x-6:x+6].astype(np.float32).mean(axis=2)
            after = out[y-6:y+6, x-6:x+6].astype(np.float32).mean(axis=2)
            assert (after.max() - med) < 0.6 * (before.max() - med)

    def test_clean_dab_on_real_scan_is_noop(self):
        from core.ccr_processor import dust_spot_effect_px
        img = self._asset()
        h, w = img.shape[:2]
        spot = {"kind": "brush", "pts": [[40 / w, 120 / h]], "r": 14 / w}
        assert dust_spot_effect_px(img, spot) == 0
        assert np.array_equal(apply_dust_removal(img, [spot]), img)

    @staticmethod
    def _load8(name):
        p = os.path.join(os.path.dirname(__file__), "data", name)
        img8 = cv2.imread(p)
        assert img8 is not None
        return cv2.cvtColor(img8, cv2.COLOR_BGR2RGB).astype(np.uint16) * 257

    def test_user_reported_crops_heal(self):
        # The maintainer's own failure crops ("save as test set image"):
        # a speck on dark background, a faint speck on sky, and a dust
        # string on sky. A dab over each must act, never no-op.
        from core.ccr_processor import dust_spot_effect_px
        for name, cx, cy, r in (("real_speck_dark.png", 34, 29, 18),
                                ("real_speck_sky.png", 36, 34, 20),
                                ("real_string_sky.png", 40, 38, 26)):
            img = self._load8(name)
            h, w = img.shape[:2]
            spot = {"kind": "brush", "pts": [[cx / w, cy / h]], "r": r / w}
            assert dust_spot_effect_px(img, spot) > 0, name
            assert not np.array_equal(apply_dust_removal(img, [spot]),
                                      img), name

    def test_user_sky_photo_dabs(self):
        # The maintainer's faint-sky-dust photo ("test against this"):
        # dabs over visible specks act; a clean-sky dab is a no-op.
        from core.ccr_processor import dust_spot_effect_px
        img = self._load8("real_sky_dust.png")
        h, w = img.shape[:2]
        hits = sum(dust_spot_effect_px(
            img, {"kind": "brush", "pts": [[x / w, y / h]], "r": 16 / w}) > 0
            for x, y in ((631, 186), (434, 182), (413, 209), (745, 143),
                         (211, 359), (668, 226)))
        assert hits >= 5
        clean = {"kind": "brush", "pts": [[0.35, 0.62]], "r": 16 / w}
        assert dust_spot_effect_px(img, clean) == 0


class TestAutoSpots:
    def test_auto_spot_halo_heals_without_ring(self):
        # AI circles hug the speck; the halo extends past them. An AUTO
        # spot's heal grows over the connected halo (machine circle, not a
        # user boundary), so no bright ring survives; the whole-circle
        # engine demonstrably leaves the ring.
        rng = np.random.default_rng(21)
        img = np.clip(rng.normal(30000, 2000, (200, 200, 3)), 0,
                      65535).astype(np.uint16)
        yy, xx = np.mgrid[0:200, 0:200]
        halo = np.exp(-((yy - 100.0) ** 2 + (xx - 100.0) ** 2)
                      / (2 * 5.0 ** 2)) * 32000
        img = np.clip(img.astype(np.float32) + halo[..., None],
                      0, 65535).astype(np.uint16)
        spot = {"kind": "auto", "pts": [[0.5, 0.5]], "r": 6.0 / 200}
        out = apply_dust_removal(img, [spot])
        win = out[85:116, 85:116].astype(np.float32)
        assert float(np.percentile(win, 99.5)) - 30000 < 8000
        ring_out = apply_dust_removal(img, [spot], method="clone")
        ring_win = ring_out[85:116, 85:116].astype(np.float32)
        assert float(np.percentile(ring_win, 99.5)) - 30000 > 10000

    def test_auto_string_polyline_heals_along_its_path(self):
        # Detection emits polyline spots for dust strings; the heal follows
        # the string, not one big circle.
        img = _grain(base=30000, sigma=1200, h=200, w=200, seed=33)
        cv2.line(img, (60, 100), (140, 112), (62000, 62000, 62000), 2)
        spot = {"kind": "auto",
                "pts": [[0.3, 0.5], [0.5, 0.53], [0.7, 0.56]], "r": 3.0 / 200}
        out = apply_dust_removal(img, [spot])
        line_px = out[98:116, 60:141].astype(np.float32)
        assert float(np.percentile(line_px, 99.5)) - 30000 < 8000

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

    def test_strings_kept_thick_elongated_dropped(self):
        # Film dust is white specks AND strings: a thin bright line is a dust
        # string and becomes a POLYLINE spot tracing it (also exempt from the
        # area cap — max_blob is 20 px here and the string is 30). Only THICK
        # elongated detections are structure (bike frame / horizon) and drop.
        prob = np.zeros((100, 100), np.float32)
        prob[50, 10:40] = 0.9       # 1x30 thin line -> dust string, kept
        prob[10:17, 40:90] = 0.9    # 7x50 thick elongated -> structure, dropped
        prob[80:83, 80:83] = 0.9    # 3x3 compact speck -> kept
        spots = dust_detect.prob_to_spots(prob, _bright_luma(prob), 60)
        assert len(spots) == 2
        strings = [s for s in spots if len(s["pts"]) > 1]
        assert len(strings) == 1
        xs = [p[0] for p in strings[0]["pts"]]
        ys = [p[1] for p in strings[0]["pts"]]
        assert min(xs) < 0.16 and max(xs) > 0.33   # waypoints span the string
        assert all(abs(y - 0.5) < 0.03 for y in ys)
        assert strings[0]["r"] * 100 < 4           # thin heal, not a smudge

    def test_string_touching_frame_edge_is_dropped(self):
        # Film borders are bright thin lines too — frame-touching strings are
        # not dust.
        prob = np.zeros((100, 100), np.float32)
        prob[50, 0:30] = 0.9
        assert dust_detect.prob_to_spots(prob, _bright_luma(prob), 60) == []

    def test_faint_blob_passes_adaptive_margin(self):
        # The old fixed 6% bright margin rejected faint wisps on smooth sky
        # ("AI says no dust found"). The margin now adapts to the surround's
        # scatter with a 1.5% floor: a 3% lift on quiet sky is detected.
        rng = np.random.default_rng(8)
        prob = np.zeros((100, 100), np.float32)
        prob[40:44, 40:44] = 0.9
        luma = (0.55 + rng.normal(0.0, 0.004, (100, 100))).astype(np.float32)
        luma[40:44, 40:44] += 0.03
        spots = dust_detect.prob_to_spots(prob, luma, 60)
        assert len(spots) == 1
        # ... while the same lift buried in heavy grain still gets rejected.
        noisy = (0.55 + rng.normal(0.0, 0.05, (100, 100))).astype(np.float32)
        noisy[40:44, 40:44] += 0.03
        assert dust_detect.prob_to_spots(prob, noisy, 60) == []

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
        img.dust_feather = 0.7
        catalog.update_for_images([img], path=cat)
        restored = catalog.create_images_for_path(path, path=cat)
        assert abs(restored[0].dust_feather - 0.7) < 1e-9
        # Legacy catalogs stored a fraction of IMAGE WIDTH under
        # "dust_feather" (slider 0..0.01) — migrated proportionally onto the
        # stroke-relative 0..1 range (0.006 -> 0.6).
        state = catalog.serialize_image(img)
        del state["dust_feather_rel"]
        state["dust_feather"] = 0.006
        old = catalog._restore_image(path, state)
        assert abs(old.dust_feather - 0.6) < 1e-9
        # Pre-feather catalog entries (neither key) restore near the default.
        del state["dust_feather"]
        ancient = catalog._restore_image(path, state)
        assert abs(ancient.dust_feather - 0.35) < 1e-6


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
            dust_feather = 0.35
            dust_spots = []

        img = _Img()
        saved = ccr_backend.images
        ccr_backend.images = [img]
        try:
            prev = _StubPreview()
            prev.current_idx = 0
            panel = DustRemovalPanel(_StubMain(), prev)
            panel.feather_slider.setValue(60)  # emits valueChanged
            assert panel.feather_value.text() == "60%"
            panel._apply_feather()             # bypass the debounce timer
            assert abs(img.dust_feather - 0.6) < 1e-9
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

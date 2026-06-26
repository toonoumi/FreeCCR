#!/usr/bin/env python3
"""
Tests for the Auto White Balance (AWB) feature.

AWB estimates a per-channel linear gain with a learning-based ONNX model
(net_awb.onnx) and applies it to the converted positive RIGHT BEFORE the manual
slider adjustments — without moving any slider. Only the per-image enable flag
(awb_enabled) is persisted/synced; the gains are a recomputed cache.
See spec/auto-white-balance.md.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Import onnxruntime BEFORE PySide6/Qt — its native DLLs fail to initialise
# (Windows access violation) when first loaded after Qt. Mirrors src/main.py's
# preload. Guarded so the suite still runs when onnxruntime is absent.
try:
    import onnxruntime as _ort  # noqa: F401
except Exception:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

import cv2  # noqa: E402
from core import awb, catalog  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402

_AWB = awb.is_available()
_needs_model = pytest.mark.skipif(not _AWB, reason="onnxruntime/net_awb.onnx unavailable")


def _lum(g):
    return 0.299 * g[0] + 0.587 * g[1] + 0.114 * g[2]


def _linear_positive(seed=1, h=200, w=300):
    """A synthetic near-linear 16-bit RGB positive with varied content."""
    rng = np.random.RandomState(seed)
    scene = np.zeros((h, w, 3), np.float32)
    scene[:] = [0.35, 0.34, 0.33]
    scene[20:80, 20:100] = [0.55, 0.30, 0.22]
    scene[20:80, w - 100:w - 20] = [0.18, 0.42, 0.20]
    scene[h - 80:h - 20, 40:140] = [0.20, 0.30, 0.60]
    scene += rng.normal(0, 0.01, scene.shape).astype(np.float32)
    scene = np.clip(scene, 0.01, 0.98)
    return scene


def _to16(lin):
    return np.clip(lin * 65535.0, 0, 65535).astype(np.uint16)


def _scan_png(tmp_path, name="pos.png", w=120, h=80, seed=3):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(30000, 4000, (h, w, 3)), 0, 65535).astype(np.uint16)
    path = str(tmp_path / name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


# --- compute_gains (needs the real model) -----------------------------------
@_needs_model
class TestComputeGains:
    def test_neutral_image_gains_near_identity(self):
        g = awb.compute_gains(_to16(_linear_positive()))
        assert g is not None
        assert all(abs(x - 1.0) < 0.1 for x in g)

    def test_warm_cast_is_corrected(self):
        # Warm/orange illuminant (R up, B down) -> gains pull R down, B up.
        warm = _to16(np.clip(_linear_positive() * np.array([1.30, 1.00, 0.70]), 0, 1))
        g = awb.compute_gains(warm)
        assert g is not None
        assert g[0] < 1.0 < g[2]

    def test_cool_cast_is_corrected(self):
        cool = _to16(np.clip(_linear_positive() * np.array([0.72, 1.00, 1.35]), 0, 1))
        g = awb.compute_gains(cool)
        assert g is not None
        assert g[0] > 1.0 > g[2]

    def test_gains_luminance_normalised(self):
        warm = _to16(np.clip(_linear_positive() * np.array([1.30, 1.00, 0.70]), 0, 1))
        g = awb.compute_gains(warm)
        assert abs(_lum(g) - 1.0) < 1e-4   # colour-only: luminance preserved

    def test_gains_within_clamp(self):
        g = awb.compute_gains(_to16(_linear_positive()))
        assert all(awb.AWB_GAIN_MIN <= x <= awb.AWB_GAIN_MAX for x in g)


# --- apply_gains (pure, no model) -------------------------------------------
class TestApplyGains:
    def test_identity_is_noop(self):
        img = _to16(_linear_positive())
        out = awb.apply_gains(img, (1.0, 1.0, 1.0))
        assert out is img                       # untouched object

    def test_none_gains_is_noop(self):
        img = _to16(_linear_positive())
        assert awb.apply_gains(img, None) is img

    def test_multiplies_per_channel_and_clips(self):
        img = np.full((4, 4, 3), 40000, np.uint16)
        out = awb.apply_gains(img, (0.5, 1.0, 2.0))
        assert out.dtype == np.uint16
        assert int(out[0, 0, 0]) == 20000      # R halved
        assert int(out[0, 0, 1]) == 40000      # G unchanged
        assert int(out[0, 0, 2]) == 65535      # B*2 clipped to max

    def test_resolution_independent(self):
        img = _to16(_linear_positive())
        half = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2),
                          interpolation=cv2.INTER_AREA)
        gains = (1.2, 1.0, 0.8)
        r_full = awb.apply_gains(img, gains).astype(np.float64).reshape(-1, 3).mean(0)
        r_half = awb.apply_gains(half, gains).astype(np.float64).reshape(-1, 3).mean(0)
        # Same gains -> same per-channel ratio at any resolution (pre-clip means).
        assert np.allclose(r_full / img.astype(np.float64).reshape(-1, 3).mean(0),
                           r_half / half.astype(np.float64).reshape(-1, 3).mean(0),
                           atol=0.02)


# --- _apply_awb gating inside CCRImage (model-independent via injected gains)-
class TestApplyAwbGating:
    def _img(self, tmp_path):
        return CCRImage(_scan_png(tmp_path))

    def test_applies_when_enabled_color(self, tmp_path):
        img = self._img(tmp_path)
        pos = _to16(_linear_positive())
        out = img._apply_awb(pos, "color", True, (1.3, 1.0, 0.7))
        assert not np.array_equal(out, pos)
        assert np.array_equal(out, awb.apply_gains(pos, (1.3, 1.0, 0.7)))

    def test_skipped_when_disabled(self, tmp_path):
        img = self._img(tmp_path)
        pos = _to16(_linear_positive())
        assert img._apply_awb(pos, "color", False, (1.3, 1.0, 0.7)) is pos

    def test_skipped_for_bw_profile(self, tmp_path):
        img = self._img(tmp_path)
        pos = _to16(_linear_positive())
        assert img._apply_awb(pos, "bw", True, (1.3, 1.0, 0.7)) is pos

    def test_override_none_falls_back_to_self(self, tmp_path):
        img = self._img(tmp_path)
        img.awb_enabled = True
        img.awb_gains = (1.1, 1.0, 0.9)
        pos = _to16(_linear_positive())
        out = img._apply_awb(pos, "color", None, None)
        assert np.array_equal(out, awb.apply_gains(pos, (1.1, 1.0, 0.9)))

    def test_does_not_touch_slider_settings(self, tmp_path):
        img = self._img(tmp_path)
        img.adjustment_settings = {"temperature": 25, "tint": -10}
        before = dict(img.adjustment_settings)
        img._apply_awb(_to16(_linear_positive()), "color", True, (1.3, 1.0, 0.7))
        assert img.adjustment_settings == before     # AWB never moves sliders


# --- apply_adjustments end-to-end (AWB before sliders) ----------------------
class TestApplyAdjustments:
    def test_awb_changes_result_without_sliders(self, tmp_path):
        img = CCRImage(_scan_png(tmp_path))
        pos = _to16(_linear_positive())
        kw = dict(settings={}, contrast_base=0, brightness_base=0, exposure_base=0)
        off = img.apply_adjustments(pos, awb_enabled=False, **kw)
        on = img.apply_adjustments(pos, awb_enabled=True, awb_gains=(1.3, 1.0, 0.7), **kw)
        assert np.array_equal(off, pos)                       # no edits -> identity
        assert np.array_equal(on, awb.apply_gains(pos, (1.3, 1.0, 0.7)))


# --- Crop region: AWB estimates the cast from the kept area only ------------
class TestAwbCropRegion:
    def _img(self, tmp_path, pos):
        img = CCRImage(_scan_png(tmp_path))
        img.resized_raw = pos
        img.awb_enabled = True
        img.awb_gains = None
        img._awb_src_id = None
        img.crop_rect = None
        img.crop_angle = 0.0
        return img

    @staticmethod
    def _capture(captured):
        def fake_compute(src):
            captured["shape"] = src.shape
            return (1.0, 1.0, 1.0)
        return fake_compute

    def test_compute_sees_only_cropped_region(self, tmp_path, monkeypatch):
        pos = np.zeros((100, 200, 3), np.uint16)
        img = self._img(tmp_path, pos)
        img.crop_rect = (0.0, 0.0, 0.5, 1.0)   # left half
        captured = {}
        monkeypatch.setattr(awb, "compute_gains", self._capture(captured))
        img._apply_awb(pos, "color", None, None, allow_compute=True)
        # Cropped to the left half: width ~100 (< full 200), full height.
        assert captured["shape"][1] < pos.shape[1]
        assert abs(captured["shape"][1] - 100) <= 5
        assert abs(captured["shape"][0] - 100) <= 5

    def test_compute_sees_full_image_when_uncropped(self, tmp_path, monkeypatch):
        pos = np.zeros((100, 200, 3), np.uint16)
        img = self._img(tmp_path, pos)              # crop_rect None
        captured = {}
        monkeypatch.setattr(awb, "compute_gains", self._capture(captured))
        img._apply_awb(pos, "color", None, None, allow_compute=True)
        assert captured["shape"] == pos.shape       # whole image

    def test_cache_key_changes_with_crop(self, tmp_path):
        img = CCRImage(_scan_png(tmp_path))
        k1 = img._awb_cache_key()
        img.crop_rect = (0.1, 0.1, 0.9, 0.9)
        assert img._awb_cache_key() != k1           # crop change invalidates
        k2 = img._awb_cache_key()
        img.crop_angle = 3.0
        assert img._awb_cache_key() != k2           # straighten angle too


# --- Availability / graceful degradation ------------------------------------
class TestAvailability:
    def test_unavailable_when_onnxruntime_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        assert awb.is_available() is False
        assert awb.availability_reason() != ""

    def test_compute_gains_none_when_onnxruntime_absent(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "onnxruntime", None)
        monkeypatch.setattr(awb, "_session", None)
        monkeypatch.setattr(awb, "_session_path", None)
        assert awb.compute_gains(_to16(_linear_positive())) is None

    def test_model_path_points_at_net_awb(self):
        assert awb.model_path().endswith("net_awb.onnx")


# --- Persistence (catalog) --------------------------------------------------
class TestPersistence:
    def test_awb_enabled_round_trip(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.awb_enabled = True
        img.awb_gains = (1.2, 1.0, 0.8)          # cache — must NOT be serialized
        # Gains are a derived cache, never written to the catalog.
        assert "awb_gains" not in catalog.serialize_image(img)
        catalog.update_for_images([img], path=cat)
        restored = catalog.create_images_for_path(path, path=cat)
        assert len(restored) == 1
        assert restored[0].awb_enabled is True
        # The persisted gains are gone; any value present is a fresh recompute
        # from the image's own pixels (not the saved 1.2/1.0/0.8 above).
        assert restored[0].awb_gains != (1.2, 1.0, 0.8)

    def test_old_entry_without_awb_defaults_false(self, tmp_path):
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.adjustment_settings = {"exposure": 5}
        state = catalog.serialize_image(img)
        del state["awb_enabled"]                 # pre-feature catalog entry
        restored = catalog._restore_image(path, state)
        assert restored.awb_enabled is False

    def test_awb_only_image_is_not_pristine(self, tmp_path):
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.awb_enabled = True
        state = catalog.serialize_image(img)
        assert catalog._is_pristine(state) is False


# --- Undo -------------------------------------------------------------------
class TestUndo:
    def test_capture_and_pop_restore_awb_enabled(self, tmp_path):
        img = CCRImage(_scan_png(tmp_path))
        img.awb_enabled = False
        img.push_undo_state()
        img.awb_enabled = True
        img.awb_gains = (1.2, 1.0, 0.8)
        assert img.pop_undo_state()
        assert img.awb_enabled is False
        assert img.awb_gains is None             # cache dropped on undo

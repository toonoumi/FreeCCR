#!/usr/bin/env python3
"""
Tests for crop-driven hi-res detail: a confirmed crop magnifies the kept
region at the fitted view, so (like zooming in) the display should fetch
crop-matched hi-res detail rather than upscaling the 1080px preview crop.

These exercise the decision gates (_crop_wants_hires / _zoomed_in_enough),
not the threaded decode itself.
"""

import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout,  # noqa: E402
                               QGraphicsPixmapItem)
from PySide6.QtGui import QPixmap, QColor, QTransform  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from widgets.image_preview import ImagePreview  # noqa: E402


class _StubPanel:
    def set_sliders_enabled(self, *a):
        pass


class _Host(QWidget):
    def __init__(self):
        super().__init__()
        self.sliders_panel = _StubPanel()
        self.mid = QWidget(self)


class _ImgStub:
    """Carries just the attributes the hi-res crop gates read."""
    def __init__(self, crop_rect, converted=True,
                 full=(6000, 4000), preview_shape=(720, 1080, 3)):
        self.crop_rect = crop_rect
        self.crop_angle = 0.0
        self.converted = converted
        self.original_full_size = full
        self.resized_raw = np.zeros(preview_shape, np.uint16)


_HOSTS = []


def _make_preview(w=540, h=360):
    host = _Host()
    _HOSTS.append(host)
    layout = QVBoxLayout(host)
    layout.addWidget(host.mid)
    mid_layout = QVBoxLayout(host.mid)
    ip = ImagePreview(host.mid)
    mid_layout.addWidget(ip)
    pm = QPixmap(w, h)
    pm.fill(QColor(128, 128, 128))
    ip.current_pixmap = pm
    ip.pixmap_item = QGraphicsPixmapItem(pm)
    ip.scene.addItem(ip.pixmap_item)
    ip.current_rotation = 0
    ip.current_horizontal_flip = False
    ip.current_vertical_flip = False
    ip.crop_mode = False
    ip.slice_mode = False
    ip.area_mode = False
    return ip


def _setup(ip, img, view_scale=1.5, zoom=1.0):
    ccr_backend.images = [img]
    ip.current_idx = 0
    ip._zoom = zoom
    ip.view.setTransform(QTransform().scale(view_scale, view_scale))


class TestCropWantsHires:
    def test_heavy_crop_wants_hires(self):
        ip = _make_preview()
        _setup(ip, _ImgStub((0.0, 0.0, 0.5, 0.5)))   # keeps 50% — >15% off
        assert ip._crop_wants_hires() is True

    def test_light_crop_does_not(self):
        ip = _make_preview()
        _setup(ip, _ImgStub((0.0, 0.0, 0.9, 0.95)))  # keeps >=90% on both sides
        assert ip._crop_wants_hires() is False

    def test_threshold_boundary(self):
        ip = _make_preview()
        # keeps >85% on both sides (<15% off) -> preview is sharp enough
        _setup(ip, _ImgStub((0.0, 0.0, 0.88, 1.0)))
        assert ip._crop_wants_hires() is False
        # keeps <85% on a side (>15% off) -> crosses the threshold
        _setup(ip, _ImgStub((0.0, 0.0, 0.80, 1.0)))
        assert ip._crop_wants_hires() is True

    def test_no_crop(self):
        ip = _make_preview()
        _setup(ip, _ImgStub(None))
        assert ip._crop_wants_hires() is False

    def test_unconverted_image(self):
        ip = _make_preview()
        _setup(ip, _ImgStub((0.0, 0.0, 0.5, 0.5), converted=False))
        assert ip._crop_wants_hires() is False

    def test_no_higher_res_source(self):
        ip = _make_preview()
        # source no bigger than the preview -> a hi-res decode adds nothing
        _setup(ip, _ImgStub((0.0, 0.0, 0.5, 0.5),
                            full=(1080, 720), preview_shape=(720, 1080, 3)))
        assert ip._crop_wants_hires() is False

    def test_crop_mode_shows_full_image(self):
        ip = _make_preview()
        _setup(ip, _ImgStub((0.0, 0.0, 0.5, 0.5)))
        ip.crop_mode = True                 # full image shown, no crop magnify
        assert ip._crop_wants_hires() is False


class TestHiresDisplayModeGuards:
    """_hires_display_pixmap must NOT apply the display crop in dust/area
    modes: both show the full un-cropped frame and normalize coordinates
    against it, so cropped hi-res pixels under them would make strokes heal
    the wrong place (the 'dust coordinates off after cropping' bug)."""

    @staticmethod
    def _inject_hires(ip, img, w=1080, h=720):
        full_pm = QPixmap(w, h)
        full_pm.fill(QColor(64, 64, 64))
        ip._hires = {"img": img, "sig": ip._hires_signature(img),
                     "adj_sig": None, "base": None, "full_pm": full_pm,
                     "display_pm": None, "crop_sig": None}
        return full_pm

    def test_dust_mode_gets_full_frame(self):
        ip = _make_preview()
        _setup(ip, _ImgStub((0.25, 0.25, 0.75, 0.75)))
        self._inject_hires(ip, ccr_backend.images[0])
        ip.dust_mode = True                 # full image shown; hires must match
        pm = ip._hires_display_pixmap()
        assert pm is not None
        assert (pm.width(), pm.height()) == (1080, 720)

    def test_area_mode_gets_full_frame(self):
        ip = _make_preview()
        _setup(ip, _ImgStub((0.25, 0.25, 0.75, 0.75)), zoom=2.0)
        self._inject_hires(ip, ccr_backend.images[0])
        ip.area_mode = True                 # full image shown; hires must match
        pm = ip._hires_display_pixmap()
        assert pm is not None
        assert (pm.width(), pm.height()) == (1080, 720)

    def test_normal_display_still_gets_cropped(self):
        ip = _make_preview()
        _setup(ip, _ImgStub((0.25, 0.25, 0.75, 0.75)))
        self._inject_hires(ip, ccr_backend.images[0])
        pm = ip._hires_display_pixmap()     # no mode active -> crop applies
        assert pm is not None
        assert (pm.width(), pm.height()) == (540, 360)


class TestZoomedInEnoughWithCrop:
    def test_heavy_crop_at_fit_is_enough(self):
        ip = _make_preview()
        _setup(ip, _ImgStub((0.0, 0.0, 0.5, 0.5)), view_scale=1.5, zoom=1.0)
        assert ip._zoomed_in_enough() is True

    def test_crop_not_magnified_on_screen_is_not_enough(self):
        ip = _make_preview()
        # crop is heavy, but the view isn't actually upscaling preview pixels
        _setup(ip, _ImgStub((0.0, 0.0, 0.5, 0.5)), view_scale=0.9, zoom=1.0)
        assert ip._zoomed_in_enough() is False

    def test_no_crop_at_fit_is_not_enough(self):
        ip = _make_preview()
        _setup(ip, _ImgStub(None), view_scale=1.5, zoom=1.0)
        assert ip._zoomed_in_enough() is False

    def test_zoomed_in_path_unchanged(self):
        ip = _make_preview()
        # No crop, but genuinely zoomed in -> the existing behaviour holds.
        _setup(ip, _ImgStub(None), view_scale=2.0, zoom=2.0)
        assert ip._zoomed_in_enough() is True


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

#!/usr/bin/env python3
"""
Slice mode displays the CROPPED frame, and the Slice button's long-press
menu arms the slicing scope.

Cut lines are fractions of the canvas the user is looking at, and the backend
bakes the parent's crop before cutting — so slice mode must show the cropped
frame, or the lines the user places and the pieces produced would disagree.
See spec/slice-crop-and-slice-all.md.
"""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from widgets.image_preview import ImagePreview  # noqa: E402


def _scan_png(tmp_path, name="negative.png", w=600, h=400, seed=21):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 20000 + 25000 * (xx / w) + 10000 * (yy / h)
    img = np.stack([base * 1.2, base, base * 0.7], axis=-1)
    img += rng.normal(0, 1500, img.shape)
    img = np.clip(img, 1000, 64000).astype(np.uint16)
    path = str(tmp_path / name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


class _StubPanel:
    def __init__(self):
        self.hints = []

    def set_sliders_enabled(self, *a):
        pass

    def set_current_idx(self, *a):
        pass

    def set_histogram(self, *a):
        pass

    def set_hint(self, *a, **kw):
        pass

    def set_temporary_hint(self, text, **kw):
        self.hints.append(text)


class _Host(QWidget):
    def __init__(self):
        super().__init__()
        self.sliders_panel = _StubPanel()
        self.mid = QWidget(self)


_HOSTS = []


def _converted_preview(tmp_path, crop_rect=(0.25, 0.25, 0.75, 0.75),
                       crop_angle=0.0, count=1):
    """`count` real 600x400 CCRImages converted through the backend, the first
    carrying the given crop, displayed in a real ImagePreview."""
    paths = [_scan_png(tmp_path, name=f"negative{i}.png", seed=21 + i)
             for i in range(count)]
    images = []
    for p in paths:
        img = CCRImage(p)
        img.reference_frame = (20, 20, 580, 380)
        images.append(img)
    ccr_backend.images = images
    ccr_backend.file_paths = list(paths)
    for i in range(count):
        ccr_backend.convert_negative_by_index(i)
    images[0].crop_rect = crop_rect
    images[0].crop_angle = crop_angle

    host = _Host()
    _HOSTS.append(host)  # keep alive
    layout = QVBoxLayout(host)
    layout.addWidget(host.mid)
    mid_layout = QVBoxLayout(host.mid)
    ip = ImagePreview(host.mid)
    mid_layout.addWidget(ip)
    ip.update_preview(0)
    return ip, images


class TestSliceModeShowsCrop:
    def test_slice_mode_keeps_cropped_display(self, tmp_path):
        ip, _ = _converted_preview(tmp_path)
        assert ip.current_pixmap.width() == 300   # crop displayed before entry
        assert ip.enter_slice_mode() is True
        assert (ip.current_pixmap.width(), ip.current_pixmap.height()) == (300, 200)
        assert ip._crop_display_transform is not None

    def test_no_crop_shows_full_frame(self, tmp_path):
        ip, _ = _converted_preview(tmp_path, crop_rect=None)
        assert ip.enter_slice_mode() is True
        assert (ip.current_pixmap.width(), ip.current_pixmap.height()) == (600, 400)
        assert ip._crop_display_transform is None

    def test_cut_fractions_are_measured_on_the_cropped_canvas(self, tmp_path):
        """A click at the middle of the displayed (cropped) canvas is a 0.5
        cut — the fraction the backend resolves against the crop."""
        ip, _ = _converted_preview(tmp_path)
        ip.enter_slice_mode()
        # Scene coords == cropped-pixmap coords here (no coarse rotation/flip)
        spec = ip._slice_ghost_spec(QPointF(150, 1))
        assert spec is not None
        assert spec[0] == 'v'
        assert spec[1] == pytest.approx(0.5, abs=0.01)

    def test_cancel_restores_the_cropped_display(self, tmp_path):
        ip, _ = _converted_preview(tmp_path)
        ip.enter_slice_mode()
        ip.cancel_slice_mode()
        assert ip.slice_mode is False
        assert ip.current_pixmap.width() == 300


class TestSliceScope:
    def test_scope_defaults_to_this_image(self, tmp_path):
        ip, _ = _converted_preview(tmp_path, count=2)
        assert ip.slice_scope == "image"
        ip.enter_slice_mode()
        assert ip.slice_scope == "image"

    def test_arming_all_enters_slice_mode(self, tmp_path):
        ip, _ = _converted_preview(tmp_path, count=2)
        assert ip.set_slice_scope("all") is True
        assert ip.slice_mode is True
        assert ip.slice_scope == "all"

    def test_rearming_keeps_the_placed_lines(self, tmp_path):
        """Switching scope after positioning the cuts must not discard them."""
        ip, _ = _converted_preview(tmp_path, count=2)
        ip.enter_slice_mode()
        ip.slice_press(QPointF(150, 1))
        ip.slice_release()
        assert len(ip._slice_lines) == 1
        assert ip.set_slice_scope("all") is True
        assert len(ip._slice_lines) == 1
        assert ip.slice_scope == "all"

    def test_cancel_disarms_all(self, tmp_path):
        """'All' must never survive into a later slice."""
        ip, _ = _converted_preview(tmp_path, count=2)
        ip.set_slice_scope("all")
        ip.cancel_slice_mode()
        assert ip.slice_scope == "image"

    def test_unknown_scope_rejected(self, tmp_path):
        ip, _ = _converted_preview(tmp_path, count=2)
        assert ip.set_slice_scope("every-other") is False
        assert ip.slice_mode is False


class TestSliceButtonLongPress:
    """The Slice button's hold gesture opens the scope menu and arms the
    preview from it."""

    def _panel(self, tmp_path, count=2):
        from widgets.sliders_panel import SlidersPanel
        ip, _ = _converted_preview(tmp_path, count=count)
        panel = SlidersPanel()
        _HOSTS.append(panel)   # keep alive
        panel.image_preview = ip
        return panel, ip

    @staticmethod
    def _pick(monkeypatch, label):
        """Make QMenu.exec_ return the action whose text matches, without
        opening a real (blocking) popup."""
        from PySide6.QtWidgets import QMenu
        chosen = {}

        def fake_exec(self, *args, **kwargs):
            chosen["actions"] = list(self.actions())
            if label is None:
                return None
            return next(a for a in self.actions() if a.text() == label)

        monkeypatch.setattr(QMenu, "exec_", fake_exec)
        return chosen

    def test_menu_offers_both_scopes_and_marks_the_armed_one(self, tmp_path,
                                                             monkeypatch):
        panel, ip = self._panel(tmp_path)
        seen = self._pick(monkeypatch, None)
        panel._on_slice_long_press()
        texts = [a.text() for a in seen["actions"]]
        assert texts == ["Slice this image", "Slice all images at the same cuts"]
        assert seen["actions"][0].isChecked() is True
        assert seen["actions"][1].isChecked() is False
        # Dismissing the menu changes nothing
        assert ip.slice_mode is False

    def test_choosing_all_arms_and_enters_slice_mode(self, tmp_path, monkeypatch):
        panel, ip = self._panel(tmp_path)
        self._pick(monkeypatch, "Slice all images at the same cuts")
        panel._on_slice_long_press()
        assert ip.slice_mode is True
        assert ip.slice_scope == "all"
        # The hint has to say so — Enter is about to cut every image
        assert "all images" in panel.hint_label.text()

    def test_all_is_disabled_with_a_single_image(self, tmp_path, monkeypatch):
        panel, ip = self._panel(tmp_path, count=1)
        seen = self._pick(monkeypatch, None)
        panel._on_slice_long_press()
        assert seen["actions"][1].isEnabled() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

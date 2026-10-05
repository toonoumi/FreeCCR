#!/usr/bin/env python3
"""
Dust mode survives an image switch.

Dusting is a per-ROLL task: selecting another thumbnail must land the brush on
the next frame instead of dropping the user out of the mode (crop/area are
per-frame geometry and still exit). What the switch MUST still do:

- discard the in-progress stroke, so a half-drawn stroke can never commit to
  the newly selected image;
- rebind the panel's per-image state (DustRemovalPanel.bind_image) so the
  feather value / detector cache / AI section follow the current image — but
  only on a real switch, never on a same-image refresh;
- exit the mode when the new image is not dust-eligible (un-converted and not
  in Positive mode — the toolbar action's own gate).

See spec/dust-removal.md §3.1 and refinement note 10.
"""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from widgets.image_preview import ImagePreview  # noqa: E402


def _scan_png(tmp_path, name, w=600, h=400, seed=21):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 20000 + 25000 * (xx / w) + 10000 * (yy / h)
    img = np.stack([base * 1.2, base, base * 0.7], axis=-1)
    img += rng.normal(0, 1500, img.shape)
    img = np.clip(img, 1000, 64000).astype(np.uint16)
    path = str(tmp_path / name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


class _StubSliders:
    def set_sliders_enabled(self, *a):
        pass

    def set_current_idx(self, *a):
        pass

    def set_histogram(self, *a):
        pass

    def set_hint(self, *a, **kw):
        pass

    def clear_hint(self, *a):
        pass


class _StubDustPanel:
    """Records the rebinds MainWindow's real panel would perform."""

    def __init__(self):
        self.binds = 0

    def bind_image(self):
        self.binds += 1

    def sync_brush_size(self, r):
        pass


class _Host(QWidget):
    """Stands in for MainWindow: owns the sliders panel and records the
    panel-restore call the real window makes when dust mode ends."""

    def __init__(self):
        super().__init__()
        self.sliders_panel = _StubSliders()
        self.mid = QWidget(self)
        self.sliders_restored = 0

    def _show_sliders_panel(self):
        self.sliders_restored += 1


_HOSTS = []


def _two_image_preview(tmp_path, convert_second=True):
    """Two real 600x400 CCRImages in the backend, displayed through the real
    ImagePreview/update_preview, with a stub dust panel attached."""
    paths = [_scan_png(tmp_path, "neg_a.png", seed=21),
             _scan_png(tmp_path, "neg_b.png", seed=99)]
    imgs = []
    for p in paths:
        img = CCRImage(p)
        img.reference_frame = (20, 20, 580, 380)
        imgs.append(img)
    ccr_backend.images = imgs
    ccr_backend.file_paths = list(paths)
    ccr_backend.positive_mode = False
    ccr_backend.convert_negative_by_index(0)
    if convert_second:
        ccr_backend.convert_negative_by_index(1)

    host = _Host()
    _HOSTS.append(host)            # keep alive (parentless top-level)
    layout = QVBoxLayout(host)
    layout.addWidget(host.mid)
    mid_layout = QVBoxLayout(host.mid)
    ip = ImagePreview(host.mid)
    mid_layout.addWidget(ip)
    ip.dust_panel = _StubDustPanel()
    ip._maybe_request_hires = lambda: None   # keep every test in-thread
    ip.update_preview(0)
    return ip, host, imgs


class TestDustModeSurvivesSwitch:
    def test_switch_keeps_dust_mode(self, tmp_path):
        ip, host, imgs = _two_image_preview(tmp_path)
        assert ip.enter_dust_mode() is True
        ip.update_preview(1)
        # The whole point: still dusting, now on the second image.
        assert ip.dust_mode is True
        assert ip.current_idx == 1
        assert ip._current_image_ref is imgs[1]
        # MainWindow is never asked to put the sliders back.
        assert host.sliders_restored == 0

    def test_switch_keeps_the_brush_cursor(self, tmp_path):
        ip, host, imgs = _two_image_preview(tmp_path)
        ip.enter_dust_mode()
        ip.update_preview(1)
        assert ip.view.cursor().shape() == Qt.CrossCursor

    def test_switch_rebinds_the_panel_to_the_new_image(self, tmp_path):
        ip, host, imgs = _two_image_preview(tmp_path)
        ip.enter_dust_mode()
        ip.dust_panel.binds = 0
        ip.update_preview(1)
        # Feather value / detector prob cache / AI section must follow the
        # image now on the canvas.
        assert ip.dust_panel.binds == 1

    def test_same_image_refresh_does_not_rebind(self, tmp_path):
        ip, host, imgs = _two_image_preview(tmp_path)
        ip.enter_dust_mode()
        ip.dust_panel.binds = 0
        ip.update_preview(0)       # slider tick / undo / dust commit
        assert ip.dust_panel.binds == 0
        assert ip.dust_mode is True

    def test_in_progress_stroke_is_discarded_on_switch(self, tmp_path):
        ip, host, imgs = _two_image_preview(tmp_path)
        ip.enter_dust_mode()
        ip.dust_press(QPointF(300, 200))     # start painting, never released
        assert ip._dust_painting is True and ip._dust_pts
        ip.update_preview(1)
        assert ip._dust_painting is False
        assert ip._dust_pts == []
        # The abandoned stroke reached NEITHER image.
        assert not (getattr(imgs[0], "dust_spots", None) or [])
        assert not (getattr(imgs[1], "dust_spots", None) or [])
        # A release arriving after the switch cannot resurrect it.
        ip.dust_release()
        assert not (getattr(imgs[1], "dust_spots", None) or [])

    def test_spots_land_on_their_own_image_across_a_switch(self, tmp_path):
        ip, host, imgs = _two_image_preview(tmp_path)
        ip.enter_dust_mode()
        ip.dust_press(QPointF(150, 100))
        ip.dust_release()
        ip.update_preview(1)
        ip.dust_press(QPointF(450, 300))
        ip.dust_release()
        # One spot each, at its own location — no leakage either way.
        assert len(imgs[0].dust_spots) == 1
        assert len(imgs[1].dust_spots) == 1
        assert imgs[0].dust_spots[0]["pts"][0] == pytest.approx([0.25, 0.25],
                                                                abs=1e-6)
        assert imgs[1].dust_spots[0]["pts"][0] == pytest.approx([0.75, 0.75],
                                                                abs=1e-6)


class TestDustModeStillExits:
    def test_switch_to_unconverted_image_exits(self, tmp_path):
        ip, host, imgs = _two_image_preview(tmp_path, convert_second=False)
        assert ip.enter_dust_mode() is True
        ip.update_preview(1)
        # Nothing to spot on an un-converted negative — same gate as the
        # toolbar action, so the mode ends and the sliders come back.
        assert ip.dust_mode is False
        assert host.sliders_restored == 1
        assert ip.view.cursor().shape() == Qt.ArrowCursor
        assert ip._dust_pts == []

    def test_unconverted_switch_does_not_rebind_the_panel(self, tmp_path):
        ip, host, imgs = _two_image_preview(tmp_path, convert_second=False)
        ip.enter_dust_mode()
        ip.dust_panel.binds = 0
        ip.update_preview(1)
        assert ip.dust_panel.binds == 0

    def test_positive_mode_keeps_dust_mode_on_unconverted(self, tmp_path):
        # In Positive mode every loaded image is editable, so an "un-converted"
        # image is still dust-eligible and the mode must persist.
        ip, host, imgs = _two_image_preview(tmp_path, convert_second=False)
        ip.enter_dust_mode()
        ccr_backend.positive_mode = True
        try:
            ip.update_preview(1)
            assert ip.dust_mode is True
            assert host.sliders_restored == 0
        finally:
            ccr_backend.positive_mode = False

    def test_clear_preview_still_exits(self, tmp_path):
        ip, host, imgs = _two_image_preview(tmp_path)
        ip.enter_dust_mode()
        ccr_backend.images = []
        ccr_backend.file_paths = []
        ip.clear_preview()          # last image removed
        assert ip.dust_mode is False
        assert host.sliders_restored == 1

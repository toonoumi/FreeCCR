#!/usr/bin/env python3
"""Tests for the anamorphic de-squeeze (Edit ▸ Anamorphic De-squeeze): the
export stretch, persistence, undo, the Copy/Sync group and the menu.
See spec/anamorphic-desqueeze.md."""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import cv2  # noqa: E402
import tifffile  # noqa: E402
from PySide6.QtWidgets import QApplication, QDialog  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core import catalog  # noqa: E402
from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from core.ccr_processor import apply_desqueeze, write_export_image  # noqa: E402
from widgets.sliders_panel import SyncSettingsDialog, SYNC_GROUPS  # noqa: E402

from test_orientation_sync_group import _ccr_image, _panel, _two_images  # noqa: E402


# --- apply_desqueeze ------------------------------------------------------------

class TestApplyDesqueeze:
    def test_identity_at_one(self):
        img = np.zeros((10, 20, 3), np.uint16)
        assert apply_desqueeze(img, 1.0) is img

    @pytest.mark.parametrize("k", [1.33, 1.5, 2.0])
    def test_widens_only(self, k):
        img = np.zeros((40, 60, 3), np.uint16)
        out = apply_desqueeze(img, k)
        assert out.shape == (40, round(60 * k), 3)
        assert out.dtype == np.uint16

    def test_flat_stays_flat(self):
        img = np.full((20, 30, 3), 12345, np.uint16)
        assert np.all(apply_desqueeze(img, 1.5) == 12345)

    def test_float_has_no_ringing_outside_input_range(self):
        img = np.zeros((16, 32, 3), np.float32)
        img[:, 16:] = 1.0                      # hard edge: Lanczos would ring
        out = apply_desqueeze(img, 1.8)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_single_channel_axis_kept(self):
        img = np.zeros((10, 20, 1), np.uint16)
        assert apply_desqueeze(img, 1.5).shape == (10, 30, 1)


class _Stub:
    color_profile = "color"

    def __init__(self, k):
        self.desqueeze = k

    @staticmethod
    def resize_image_to_max_pixel(image, max_long_side):
        return CCRImage.resize_image_to_max_pixel(None, image, max_long_side)


class TestExportWrite:
    def test_export_is_stretched(self, tmp_path):
        out = str(tmp_path / "out.tiff")
        img = np.full((40, 60, 3), 30000, np.uint16)
        path = write_export_image(_Stub(1.5), img, out, False, 95, None)
        assert tifffile.imread(path or out).shape[:2] == (40, 90)

    def test_long_edge_limit_bounds_the_stretched_file(self, tmp_path):
        out = str(tmp_path / "out.tiff")
        img = np.full((40, 60, 3), 30000, np.uint16)
        path = write_export_image(_Stub(2.0), img, out, False, 95, 60)
        h, w = tifffile.imread(path or out).shape[:2]
        assert w == 60 and h == 20


# --- model: undo + catalog ----------------------------------------------------

class TestModel:
    def test_default_is_off(self, tmp_path):
        assert _ccr_image(tmp_path).desqueeze == 1.0

    def test_undo_restores_factor(self, tmp_path):
        img = _ccr_image(tmp_path)
        img.push_undo_state()
        img.desqueeze = 1.5
        assert img.pop_undo_state()
        assert img.desqueeze == 1.0

    def test_legacy_snapshot_restores_off(self, tmp_path):
        img = _ccr_image(tmp_path)
        img.desqueeze = 2.0
        snap = img.capture_undo_state()
        del snap["desqueeze"]
        img.undo_stack.append(snap)
        img.pop_undo_state()
        assert img.desqueeze == 1.0

    def test_catalog_round_trip(self, tmp_path):
        cat = str(tmp_path / "catalog.json")
        img = _ccr_image(tmp_path)
        img.desqueeze = 1.44
        state = catalog.serialize_image(img)
        assert state["desqueeze"] == 1.44
        assert not catalog._is_pristine(state)     # a de-squeeze alone is an edit
        catalog.update_for_images([img], path=cat)
        restored = catalog.create_images_for_path(img.file_path, path=cat)
        assert restored[0].desqueeze == 1.44


# --- Copy / Sync group --------------------------------------------------------

def _stub_dialog(monkeypatch, selection):
    full = {gid: bool(selection.get(gid)) for gid, _l, _k in SYNC_GROUPS}
    monkeypatch.setattr(SyncSettingsDialog, "exec_",
                        lambda self: QDialog.Accepted, raising=False)
    monkeypatch.setattr(SyncSettingsDialog, "selection", lambda self: dict(full))


class TestSyncGroup:
    def test_group_follows_orientation(self):
        ids = [gid for gid, _l, _k in SYNC_GROUPS]
        assert ids[ids.index("orientation") + 1] == "desqueeze"
        assert dict((g, k) for g, _l, k in SYNC_GROUPS)["desqueeze"] == ()

    def test_copy_paste(self, tmp_path, monkeypatch):
        src, tgt = _two_images(tmp_path)
        src.desqueeze = 1.8
        panel = _panel()
        panel.current_idx = 0
        _stub_dialog(monkeypatch, {"desqueeze": True})
        panel.copy_adjustment_settings()
        assert panel.copied_desqueeze == 1.8
        panel.current_idx = 1
        panel.paste_adjustment_settings()
        assert tgt.desqueeze == 1.8
        assert tgt.rotation_angle == 0

    def test_uncopied_group_leaves_target_alone(self, tmp_path, monkeypatch):
        src, tgt = _two_images(tmp_path)
        src.desqueeze = 1.8
        tgt.desqueeze = 1.33
        panel = _panel()
        panel.current_idx = 0
        _stub_dialog(monkeypatch, {"orientation": True})
        panel.copy_adjustment_settings()
        assert panel.copied_desqueeze is None
        panel.current_idx = 1
        panel.paste_adjustment_settings()
        assert tgt.desqueeze == 1.33

    def test_sync_to_all_without_reprocess(self, tmp_path, monkeypatch):
        src, tgt = _two_images(tmp_path)
        src.desqueeze = 2.0
        calls = []
        monkeypatch.setattr(CCRImage, "update_thumbnail_and_preview",
                            lambda self: calls.append(self))
        panel = _panel()
        panel.current_idx = 0
        panel._sync_group_selection = {"desqueeze": True}
        panel._perform_sync_to_all()
        assert tgt.desqueeze == 2.0
        assert calls == []                         # display-level only
        assert len(tgt.undo_stack) == 1


# --- Edit menu ------------------------------------------------------------------

@pytest.fixture(scope="module")
def main_window():
    # Never closed: MainWindow.closeEvent raises a modal confirm-exit dialog
    # that would hang (same as tests/test_panel_width.py).
    from ui.main_window import MainWindow
    ccr_backend.images = []
    ccr_backend.file_paths = []
    return MainWindow()


class TestEditMenu:
    def test_menu_lists_presets_and_custom(self, main_window):
        w = main_window
        top = [a.text() for a in w.menuBar().actions()]
        assert top.index("File") < top.index("Edit") < top.index("Help")
        sub = w.desqueeze_menu
        assert sub.menuAction() in w.edit_menu.actions()
        assert sub.title() == "Anamorphic De-squeeze"
        labels = [a.text() for a in sub.actions() if not a.isSeparator()]
        assert labels[0] == "None (1.0×)"
        for f in ("1.2×", "1.4×", "1.44×", "1.5×", "1.6×", "2×"):
            assert f in labels
        assert labels[-1] == "Custom…"

    def test_preset_sets_selected_images(self, main_window, tmp_path, monkeypatch):
        src, tgt = _two_images(tmp_path)
        w = main_window
        monkeypatch.setattr(w.thumbnail_list, "selected_indices", lambda: [0, 1])
        monkeypatch.setattr(w.thumbnail_list, "update_thumbnail", lambda idx: None)
        monkeypatch.setattr(w.image_preview, "update_preview", lambda idx: None)
        w.image_preview.current_idx = 0
        w._desqueeze_actions[1.5].trigger()
        assert src.desqueeze == 1.5 and tgt.desqueeze == 1.5
        w._refresh_desqueeze_menu()
        assert w._desqueeze_actions[1.5].isChecked()

    def test_custom_factor_shows_in_label(self, main_window, tmp_path, monkeypatch):
        src, _tgt = _two_images(tmp_path)
        w = main_window
        monkeypatch.setattr(w.thumbnail_list, "selected_indices", lambda: [])
        monkeypatch.setattr(w.thumbnail_list, "update_thumbnail", lambda idx: None)
        monkeypatch.setattr(w.image_preview, "update_preview", lambda idx: None)
        w.image_preview.current_idx = 0
        from ui import main_window as mw_mod
        monkeypatch.setattr(mw_mod.QInputDialog, "getDouble",
                            staticmethod(lambda *a, **k: (1.37, True)))
        w.desqueeze_custom_action.trigger()
        assert src.desqueeze == 1.37
        w._refresh_desqueeze_menu()
        assert w.desqueeze_custom_action.text() == "Custom (1.37×)…"

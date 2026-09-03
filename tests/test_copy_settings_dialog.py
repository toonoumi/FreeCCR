#!/usr/bin/env python3
"""Tests for the selective Ctrl/Cmd+C copy — the dialog that asks which
setting groups to copy, and the group-aware merge Ctrl/Cmd+V performs.
See spec/copy-settings-dialog.md."""

import os
import sys

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QWidget,  # noqa: E402
                               QVBoxLayout)

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from widgets.sliders_panel import (SlidersPanel, SyncSettingsDialog,  # noqa: E402
                                   SYNC_GROUPS)


# --- harness -----------------------------------------------------------------

def _ccr_image(tmp_path, name="scan.png"):
    """A CCRImage from a tiny uniform on-disk scan so __init__ succeeds."""
    path = str(tmp_path / name)
    base = np.full((40, 60, 3), 20000, np.uint16)
    cv2.imwrite(path, cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
    return CCRImage(path)


def _half_image(h=40, w=60):
    """Left half red, right half blue — the halves' histograms differ, so a
    crop must change the computed histogram."""
    a = np.zeros((h, w, 3), np.uint16)
    a[:, : w // 2] = (60000, 0, 0)
    a[:, w // 2:] = (0, 0, 60000)
    return a


def _hist_bytes(img):
    return img.histogram_data.tobytes()


class _ImagePreviewStub:
    def __init__(self):
        self.updated = []

    def update_preview(self, idx):
        self.updated.append(idx)


class _Host(QWidget):
    """parent().parent() target carrying image_preview (as MainWindow does)."""
    def __init__(self):
        super().__init__()
        self.image_preview = _ImagePreviewStub()
        self.mid = QWidget(self)


_HOSTS = []


def _panel():
    host = _Host()
    _HOSTS.append(host)               # keep alive for the test's lifetime
    mid_layout = QVBoxLayout(host.mid)
    panel = SlidersPanel(host.mid)
    mid_layout.addWidget(panel)
    return panel


def _stub_dialog(monkeypatch, selection, accepted=True):
    """Replace the modal group picker with a canned answer. `selection` is a
    partial {gid: bool}; unlisted groups count as unchecked."""
    full = {gid: bool(selection.get(gid)) for gid, _l, _k in SYNC_GROUPS}

    def _exec(self):
        return QDialog.Accepted if accepted else QDialog.Rejected

    monkeypatch.setattr(SyncSettingsDialog, "exec_", _exec, raising=False)
    monkeypatch.setattr(SyncSettingsDialog, "selection", lambda self: dict(full))
    return full


def _set(panel, key, value):
    """Set one slider by adjustment key, as a user drag would."""
    panel.sliders[panel.adjustment_keys.index(key)].setValue(value)


def _val(img, key):
    return img.adjustment_settings.get(key)


def _copy(panel, monkeypatch, selection, accepted=True):
    _stub_dialog(monkeypatch, selection, accepted)
    panel.copy_adjustment_settings()


# --- the dialog itself --------------------------------------------------------

class TestDialog:
    def test_copy_dialog_offers_the_sync_groups(self):
        dlg = SyncSettingsDialog(None, None, title="Copy Settings",
                                 prompt="Copy these settings:",
                                 action_label="Copy")
        assert dlg.windowTitle() == "Copy Settings"
        assert list(dlg._checkboxes) == [gid for gid, _l, _k in SYNC_GROUPS]
        assert all(dlg.selection().values())          # all checked by default

    def test_select_all_and_deselect_all(self):
        dlg = SyncSettingsDialog(None, None)
        dlg._set_all(False)
        assert not any(dlg.selection().values())
        dlg._set_all(True)
        assert all(dlg.selection().values())

    def test_seeds_from_remembered_selection(self):
        dlg = SyncSettingsDialog(None, {"wb": True, "tone": False})
        sel = dlg.selection()
        assert sel["wb"] is True and sel["tone"] is False
        assert sel["crop"] is True                    # unlisted → checked

    def test_sync_defaults_unchanged(self):
        """Sync to All's call site passes no wording — its strings are the
        constructor defaults."""
        dlg = SyncSettingsDialog(None, None)
        assert dlg.windowTitle() == "Sync to All"


# --- copy ---------------------------------------------------------------------

class TestCopy:
    def test_copies_only_selected_groups(self, tmp_path, monkeypatch):
        img = _ccr_image(tmp_path)
        ccr_backend.images = [img]
        ccr_backend.file_paths = [img.file_path]
        panel = _panel()
        panel.current_idx = 0
        _set(panel, "balance_r", 40)
        _set(panel, "contrast", 25)

        _copy(panel, monkeypatch, {"wb": True})

        assert panel.copied_adjustment == {"balance_r": 40, "balance_g": 0,
                                           "balance_b": 0}
        assert "contrast" not in panel.copied_adjustment
        assert panel.copied_profile is None and panel.copied_crop is None

    def test_cancel_keeps_previous_clipboard(self, tmp_path, monkeypatch):
        img = _ccr_image(tmp_path)
        ccr_backend.images = [img]
        ccr_backend.file_paths = [img.file_path]
        panel = _panel()
        panel.current_idx = 0
        _set(panel, "balance_r", 40)
        _copy(panel, monkeypatch, {"wb": True})
        first = dict(panel.copied_adjustment)

        _set(panel, "balance_r", -40)
        _copy(panel, monkeypatch, {"tone": True}, accepted=False)

        assert panel.copied_adjustment == first
        assert panel.copied_selection["wb"] is True

    def test_empty_selection_copies_nothing(self, tmp_path, monkeypatch):
        img = _ccr_image(tmp_path)
        ccr_backend.images = [img]
        ccr_backend.file_paths = [img.file_path]
        panel = _panel()
        panel.current_idx = 0

        _copy(panel, monkeypatch, {})

        assert panel.copied_selection is None
        assert panel.copied_adjustment is None
        assert "Nothing selected" in panel.hint_label.text()

    def test_no_image_shows_hint_and_no_dialog(self, monkeypatch):
        panel = _panel()
        panel.current_idx = None

        def _boom(self):
            raise AssertionError("dialog must not open without an image")

        monkeypatch.setattr(SyncSettingsDialog, "exec_", _boom, raising=False)
        panel.copy_adjustment_settings()
        assert panel.copied_selection is None
        assert "No image selected" in panel.hint_label.text()

    def test_whole_image_groups_read_from_the_image(self, tmp_path, monkeypatch):
        img = _ccr_image(tmp_path)
        img.color_profile = "bw"
        img.crop_rect = (0.1, 0.2, 0.8, 0.9)
        img.crop_angle = 3.0
        ccr_backend.images = [img]
        ccr_backend.file_paths = [img.file_path]
        panel = _panel()
        panel.current_idx = 0

        _copy(panel, monkeypatch, {"profile": True, "crop": True})

        assert panel.copied_profile == "bw"
        assert panel.copied_crop == ((0.1, 0.2, 0.8, 0.9), 3.0)


# --- paste --------------------------------------------------------------------

class TestPasteMerge:
    def _two_images(self, tmp_path):
        src = _ccr_image(tmp_path, "src.png")
        tgt = _ccr_image(tmp_path, "tgt.png")
        for i in (src, tgt):
            i.converted = True
            i.adjustment_settings = {}
        ccr_backend.images = [src, tgt]
        ccr_backend.file_paths = [src.file_path, tgt.file_path]
        return src, tgt

    def test_uncopied_groups_keep_target_values(self, tmp_path, monkeypatch):
        src, tgt = self._two_images(tmp_path)
        panel = _panel()

        panel.current_idx = 0
        _set(panel, "balance_r", 40)
        _set(panel, "contrast", 25)
        _copy(panel, monkeypatch, {"wb": True})

        panel.current_idx = 1
        _set(panel, "balance_r", -10)
        _set(panel, "contrast", -30)
        panel.paste_adjustment_settings()

        assert _val(tgt, "balance_r") == 40      # copied group followed
        assert _val(tgt, "contrast") == -30        # un-copied group untouched

    def test_pasted_dict_carries_every_key(self, tmp_path, monkeypatch):
        src, tgt = self._two_images(tmp_path)
        panel = _panel()
        panel.current_idx = 0
        _set(panel, "balance_r", 40)
        _copy(panel, monkeypatch, {"wb": True})

        panel.current_idx = 1
        panel.paste_adjustment_settings()

        for key in SlidersPanel.ADJUSTMENT_KEYS:
            assert key in tgt.adjustment_settings, key
        # Defaults are honoured, not zero-filled blindly.
        assert _val(tgt, "band_feather") == panel._default_for("band_feather")

    def test_sliders_reflect_the_merge_not_the_clipboard(self, tmp_path, monkeypatch):
        src, tgt = self._two_images(tmp_path)
        panel = _panel()
        panel.current_idx = 0
        _set(panel, "balance_r", 40)
        _set(panel, "contrast", 25)
        _copy(panel, monkeypatch, {"wb": True})

        panel.current_idx = 1
        _set(panel, "contrast", -30)
        panel.paste_adjustment_settings()

        assert panel.sliders[panel.adjustment_keys.index("balance_r")].value() == 40
        assert panel.sliders[panel.adjustment_keys.index("contrast")].value() == -30

    def test_paste_pushes_one_undo_state(self, tmp_path, monkeypatch):
        src, tgt = self._two_images(tmp_path)
        panel = _panel()
        panel.current_idx = 0
        _set(panel, "balance_r", 40)
        _copy(panel, monkeypatch, {"wb": True})

        panel.current_idx = 1
        before = len(tgt.undo_stack)
        panel.paste_adjustment_settings()
        assert len(tgt.undo_stack) == before + 1

    def test_paste_without_a_copy_is_a_no_op(self, tmp_path):
        src, tgt = self._two_images(tmp_path)
        panel = _panel()
        panel.current_idx = 1
        panel.paste_adjustment_settings()
        assert tgt.adjustment_settings == {}
        assert "No settings to paste" in panel.hint_label.text()


class TestPasteCurves:
    CURVE = {"rgb": [[0.0, 0.0], [128.0, 200.0], [255.0, 255.0]]}
    OTHER = {"rgb": [[0.0, 0.0], [128.0, 60.0], [255.0, 255.0]]}

    def _panel_with(self, tmp_path):
        src = _ccr_image(tmp_path, "src.png")
        tgt = _ccr_image(tmp_path, "tgt.png")
        for i in (src, tgt):
            i.converted = True
            i.adjustment_settings = {}
        ccr_backend.images = [src, tgt]
        ccr_backend.file_paths = [src.file_path, tgt.file_path]
        return _panel(), src, tgt

    def test_curves_copied_replaces_target_curves(self, tmp_path, monkeypatch):
        panel, src, tgt = self._panel_with(tmp_path)
        panel.current_idx = 0
        panel.curve_editor.set_curves(self.CURVE)
        _copy(panel, monkeypatch, {"curves": True})

        panel.current_idx = 1
        tgt.adjustment_settings = {"curves": self.OTHER}
        panel.paste_adjustment_settings()

        assert tgt.adjustment_settings["curves"] == self.CURVE

    def test_identity_source_curves_clear_the_target(self, tmp_path, monkeypatch):
        panel, src, tgt = self._panel_with(tmp_path)
        panel.current_idx = 0
        panel.curve_editor.set_curves(None)           # identity
        _copy(panel, monkeypatch, {"curves": True})

        panel.current_idx = 1
        tgt.adjustment_settings = {"curves": self.OTHER}
        panel.paste_adjustment_settings()

        assert "curves" not in tgt.adjustment_settings

    def test_uncopied_curves_survive_the_rebuild(self, tmp_path, monkeypatch):
        panel, src, tgt = self._panel_with(tmp_path)
        panel.current_idx = 0
        panel.curve_editor.set_curves(self.CURVE)
        _set(panel, "balance_r", 40)
        _copy(panel, monkeypatch, {"wb": True})       # curves NOT copied

        panel.current_idx = 1
        tgt.adjustment_settings = {"curves": self.OTHER}
        panel.paste_adjustment_settings()

        assert tgt.adjustment_settings["curves"] == self.OTHER
        assert _val(tgt, "balance_r") == 40


class TestPasteWholeImageGroups:
    def test_profile_group_switches_the_target_and_combo(self, tmp_path, monkeypatch):
        src = _ccr_image(tmp_path, "src.png")
        tgt = _ccr_image(tmp_path, "tgt.png")
        src.color_profile = "bw"
        tgt.color_profile = "color"
        for i in (src, tgt):
            i.converted = True
            i.adjustment_settings = {}
        ccr_backend.images = [src, tgt]
        ccr_backend.file_paths = [src.file_path, tgt.file_path]

        panel = _panel()
        panel.current_idx = 0
        _copy(panel, monkeypatch, {"profile": True})
        panel.current_idx = 1
        panel.paste_adjustment_settings()

        assert tgt.color_profile == "bw"
        assert panel.color_profile_combo.currentIndex() == \
            SlidersPanel.COLOR_PROFILES.index("bw")

    def test_profile_left_alone_when_not_copied(self, tmp_path, monkeypatch):
        src = _ccr_image(tmp_path, "src.png")
        tgt = _ccr_image(tmp_path, "tgt.png")
        src.color_profile = "bw"
        tgt.color_profile = "color"
        for i in (src, tgt):
            i.converted = True
            i.adjustment_settings = {}
        ccr_backend.images = [src, tgt]
        ccr_backend.file_paths = [src.file_path, tgt.file_path]

        panel = _panel()
        panel.current_idx = 0
        _copy(panel, monkeypatch, {"wb": True})
        panel.current_idx = 1
        panel.paste_adjustment_settings()

        assert tgt.color_profile == "color"

    def test_crop_group_reprocesses_the_target_histogram(self, tmp_path, monkeypatch):
        """Pasting a crop must regenerate the crop-aware histogram, not just
        copy the value (mirrors TestSyncToAllCropHistogram)."""
        full = _half_image()

        src = _ccr_image(tmp_path, "src.png")
        src.converted = True
        src.adjustment_settings = {}
        src.resized_raw = full.copy()
        src.crop_rect = (0.5, 0.0, 1.0, 1.0)
        src.update_thumbnail_and_preview()
        src_cropped = _hist_bytes(src)

        tgt = _ccr_image(tmp_path, "tgt.png")
        tgt.converted = True
        tgt.adjustment_settings = {}
        tgt.resized_raw = full.copy()
        tgt.crop_rect = None
        tgt.update_thumbnail_and_preview()
        assert _hist_bytes(tgt) != src_cropped        # precondition

        ccr_backend.images = [src, tgt]
        ccr_backend.file_paths = [src.file_path, tgt.file_path]

        panel = _panel()
        panel.current_idx = 0
        _copy(panel, monkeypatch, {"crop": True})
        panel.current_idx = 1
        panel.paste_adjustment_settings()

        assert tgt.crop_rect == (0.5, 0.0, 1.0, 1.0)
        assert _hist_bytes(tgt) == src_cropped

    def test_copied_no_crop_clears_the_target_crop(self, tmp_path, monkeypatch):
        src = _ccr_image(tmp_path, "src.png")
        tgt = _ccr_image(tmp_path, "tgt.png")
        for i in (src, tgt):
            i.converted = True
            i.adjustment_settings = {}
        src.crop_rect = None
        src.crop_angle = 0.0
        tgt.crop_rect = (0.1, 0.1, 0.9, 0.9)
        tgt.crop_angle = 5.0
        ccr_backend.images = [src, tgt]
        ccr_backend.file_paths = [src.file_path, tgt.file_path]

        panel = _panel()
        panel.current_idx = 0
        _copy(panel, monkeypatch, {"crop": True})
        panel.current_idx = 1
        panel.paste_adjustment_settings()

        assert tgt.crop_rect is None and tgt.crop_angle == 0.0

    def test_crop_left_alone_when_not_copied(self, tmp_path, monkeypatch):
        src = _ccr_image(tmp_path, "src.png")
        tgt = _ccr_image(tmp_path, "tgt.png")
        for i in (src, tgt):
            i.converted = True
            i.adjustment_settings = {}
        src.crop_rect = (0.5, 0.0, 1.0, 1.0)
        tgt.crop_rect = (0.1, 0.1, 0.9, 0.9)
        ccr_backend.images = [src, tgt]
        ccr_backend.file_paths = [src.file_path, tgt.file_path]

        panel = _panel()
        panel.current_idx = 0
        _copy(panel, monkeypatch, {"wb": True})
        panel.current_idx = 1
        panel.paste_adjustment_settings()

        assert tgt.crop_rect == (0.1, 0.1, 0.9, 0.9)


class TestPasteCineonFlag:
    """cineon_log rides the channels group and is whole-image only."""

    def _pair(self, tmp_path):
        src = _ccr_image(tmp_path, "src.png")
        tgt = _ccr_image(tmp_path, "tgt.png")
        for i in (src, tgt):
            i.converted = True
            i.adjustment_settings = {}
        ccr_backend.images = [src, tgt]
        ccr_backend.file_paths = [src.file_path, tgt.file_path]
        return src, tgt

    def test_channels_group_carries_the_flag(self, tmp_path, monkeypatch):
        src, tgt = self._pair(tmp_path)
        src.adjustment_settings = {"cineon_log": True}
        panel = _panel()
        panel.current_idx = 0
        _copy(panel, monkeypatch, {"channels": True})
        panel.current_idx = 1
        panel.paste_adjustment_settings()

        assert tgt.adjustment_settings.get("cineon_log") is True

    def test_target_flag_survives_an_unrelated_paste(self, tmp_path, monkeypatch):
        src, tgt = self._pair(tmp_path)
        panel = _panel()
        panel.current_idx = 0
        _set(panel, "balance_r", 40)
        _copy(panel, monkeypatch, {"wb": True})

        panel.current_idx = 1
        tgt.adjustment_settings = {"cineon_log": True}
        panel.paste_adjustment_settings()

        assert tgt.adjustment_settings.get("cineon_log") is True

    def test_channels_paste_clears_an_unset_flag(self, tmp_path, monkeypatch):
        src, tgt = self._pair(tmp_path)
        panel = _panel()
        panel.current_idx = 0                 # source has no flag
        _copy(panel, monkeypatch, {"channels": True})

        panel.current_idx = 1
        tgt.adjustment_settings = {"cineon_log": True}
        panel.paste_adjustment_settings()

        assert "cineon_log" not in tgt.adjustment_settings

    def test_never_written_into_an_area_layer(self, tmp_path, monkeypatch):
        src, tgt = self._pair(tmp_path)
        src.adjustment_settings = {"cineon_log": True}
        panel = _panel()
        panel.current_idx = 0
        _copy(panel, monkeypatch, {"channels": True})

        panel.current_idx = 1
        tgt.area_layers = [{"id": "a1", "kind": "circle", "enabled": True,
                            "feather": 0.2, "angle": 0,
                            "geometry": {"cx": 0.5, "cy": 0.5, "rx": 0.3, "ry": 0.3},
                            "settings": {}}]
        tgt.active_area_id = "a1"
        panel.paste_adjustment_settings()

        assert "cineon_log" not in tgt.get_area("a1")["settings"]
        assert tgt.adjustment_settings.get("cineon_log") is not True

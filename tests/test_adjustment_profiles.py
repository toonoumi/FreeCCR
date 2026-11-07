#!/usr/bin/env python3
"""Tests for saved adjustment profiles in the Color Profile dropdown — the
pure store/merge/match helpers and the panel's save / apply / derived-display
/ delete flow. See spec/adjustment-profiles.md. Modal dialogs are
monkeypatched; the panel's QSettings is redirected to a temp ini so tests
never touch the user's real store."""

import json
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import (QApplication, QDialog, QMessageBox,  # noqa: E402
                               QVBoxLayout, QWidget)

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.adjustment_profiles import (  # noqa: E402
    build_profile, decode_profiles, encode_profiles, find_profile,
    merge_profile_settings, profile_matches, remove_profile, upsert_profile)
from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from widgets.sliders_panel import (PROFILE_GROUP_IDS,  # noqa: E402
                                   PROFILE_GROUP_KEYS, SYNC_GROUPS,
                                   SlidersPanel, SyncSettingsDialog)

GROUP_KEYS = {"tone": ("brightness", "contrast"),
              "wb": ("temperature", "tint"),
              "channels": ("ch_r_shift", "cineon_log"),
              "curves": (), "profile": ()}
ADJ_KEYS = ["brightness", "contrast", "temperature", "tint", "ch_r_shift"]
CURVE = {"rgb": [[0.0, 0.0], [0.5, 0.6], [1.0, 1.0]]}


def _default(_k):
    return 0


def _profile(name="Look", groups=("tone",), settings=None, **kw):
    record = {"name": name, "groups": list(groups),
              "settings": settings or {"brightness": 10, "contrast": 5}}
    record.update(kw)
    return decode_profiles(json.dumps([record]))[0]


# --- store --------------------------------------------------------------------

class TestStore:
    def test_garbage_decodes_empty(self):
        assert decode_profiles("") == []
        assert decode_profiles(None) == []
        assert decode_profiles("not json {") == []
        assert decode_profiles('{"name": "x"}') == []

    def test_bad_records_dropped(self):
        text = json.dumps([
            {"name": "", "groups": ["tone"]},
            {"name": "no groups", "groups": []},
            {"name": "bad groups", "groups": "tone"},
            {"name": "ok", "groups": ["tone"],
             "settings": {"brightness": 3, "x": "str", "y": float("nan")}},
        ])
        out = decode_profiles(text)
        assert [p["name"] for p in out] == ["ok"]
        assert out[0]["settings"] == {"brightness": 3}
        assert out[0]["color_profile"] == "color"
        assert out[0]["curves"] is None

    def test_round_trip_and_duplicates(self):
        a = _profile("Alpha", curves=CURVE, color_profile="bw")
        out = decode_profiles(encode_profiles([a]))
        assert out == [a]
        dup = json.dumps([{"name": "Same", "groups": ["tone"]},
                          {"name": "SAME", "groups": ["wb"]}])
        out = decode_profiles(dup)
        assert len(out) == 1 and out[0]["groups"] == ["tone"]

    def test_upsert_find_remove(self):
        ps = upsert_profile([], _profile("Zed"))
        ps = upsert_profile(ps, _profile("alpha"))
        assert [p["name"] for p in ps] == ["alpha", "Zed"]
        ps = upsert_profile(ps, _profile("ZED", groups=("wb",)))
        assert len(ps) == 2 and find_profile(ps, " zed ")["groups"] == ["wb"]
        assert find_profile(ps, "") is None
        assert [p["name"] for p in remove_profile(ps, "Alpha")] == ["ZED"]
        with pytest.raises(ValueError):
            upsert_profile(ps, {"name": "x", "groups": []})


# --- merge / match ------------------------------------------------------------

class TestMergeAndMatch:
    def test_build_fills_defaults_for_every_group_key(self):
        p = build_profile("L", ["tone", "channels"], {"brightness": 7},
                          "color", GROUP_KEYS, _default)
        assert p["settings"] == {"brightness": 7, "contrast": 0,
                                 "ch_r_shift": 0, "cineon_log": False}

    def test_merge_overwrites_only_group_keys(self):
        current = {"brightness": 1, "temperature": 30}
        merged = merge_profile_settings(current, _profile(), GROUP_KEYS,
                                        ADJ_KEYS, _default)
        assert merged == {"brightness": 10, "contrast": 5, "temperature": 30,
                          "tint": 0, "ch_r_shift": 0}

    def test_merge_curves(self):
        current = {"curves": {"r": [[0, 0], [1, 0.9]]}}
        with_curves = _profile(groups=("tone", "curves"), curves=CURVE)
        assert merge_profile_settings(current, with_curves, GROUP_KEYS,
                                      ADJ_KEYS, _default)["curves"] == CURVE
        identity = _profile(groups=("curves",))
        assert "curves" not in merge_profile_settings(
            current, identity, GROUP_KEYS, ADJ_KEYS, _default)
        carried = merge_profile_settings(current, _profile(), GROUP_KEYS,
                                         ADJ_KEYS, _default)
        assert carried["curves"] == current["curves"]

    def test_merge_cineon(self):
        on = {"cineon_log": True}
        off_profile = _profile(groups=("channels",),
                               settings={"ch_r_shift": 4, "cineon_log": False})
        assert "cineon_log" not in merge_profile_settings(
            on, off_profile, GROUP_KEYS, ADJ_KEYS, _default)
        assert merge_profile_settings(on, _profile(), GROUP_KEYS, ADJ_KEYS,
                                      _default)["cineon_log"] is True

    def test_matches(self):
        p = _profile(groups=("tone", "profile", "curves"),
                     color_profile="bw")
        current = {"brightness": 10, "contrast": 5, "temperature": 99}
        # No curves on either side: identity ≡ None, so it matches.
        assert profile_matches(current, "bw", p, GROUP_KEYS, _default)
        assert not profile_matches(current, "color", p, GROUP_KEYS, _default)
        assert not profile_matches(dict(current, contrast=6), "bw", p,
                                   GROUP_KEYS, _default)
        assert not profile_matches(dict(current, curves=CURVE), "bw", p,
                                   GROUP_KEYS, _default)
        # Without the "profile" group the Color/B&W flag is not compared.
        assert profile_matches(current, "color", _profile(), GROUP_KEYS,
                               _default)


# --- panel --------------------------------------------------------------------

class _ImagePreviewStub:
    def update_preview(self, idx):
        pass


class _Host(QWidget):
    def __init__(self):
        super().__init__()
        self.image_preview = _ImagePreviewStub()
        self.mid = QWidget(self)


_HOSTS = []


@pytest.fixture
def panel(tmp_path):
    host = _Host()
    _HOSTS.append(host)
    layout = QVBoxLayout(host.mid)
    p = SlidersPanel(host.mid)
    layout.addWidget(p)
    p._settings = QSettings(str(tmp_path / "test.ini"), QSettings.IniFormat)
    p._adjust_profiles = []
    p._reload_color_profile_combo()
    return p


@pytest.fixture
def image(tmp_path):
    path = str(tmp_path / "scan.png")
    cv2.imwrite(path, np.full((40, 60, 3), 20000, np.uint16))
    img = CCRImage(path)
    img.converted = True
    img.adjustment_settings = {}
    ccr_backend.images = [img]
    ccr_backend.file_paths = [img.file_path]
    yield img
    ccr_backend.images = []
    ccr_backend.file_paths = []


def _stub_save_dialog(monkeypatch, name, groups):
    monkeypatch.setattr(SyncSettingsDialog, "exec_",
                        lambda self: QDialog.Accepted, raising=False)
    monkeypatch.setattr(SyncSettingsDialog, "selection",
                        lambda self: {g: g in groups for g in PROFILE_GROUP_IDS})
    monkeypatch.setattr(SyncSettingsDialog, "name", lambda self: name)


def _set(panel, key, value):
    panel.sliders[panel.adjustment_keys.index(key)].setValue(value)


def _items(panel):
    c = panel.color_profile_combo
    return [c.itemText(i) for i in range(c.count())]


def test_profile_groups_exclude_geometry():
    assert set(PROFILE_GROUP_IDS) == (
        {g for g, _l, _k in SYNC_GROUPS} - {"crop", "orientation", "desqueeze"})


def test_save_dialog_shows_only_profile_groups_and_a_name():
    dlg = SyncSettingsDialog(None, None, groups=PROFILE_GROUP_IDS,
                             name_prompt="Profile name:")
    assert list(dlg._checkboxes) == list(PROFILE_GROUP_IDS)
    dlg._name_edit.setText("  Roll look ")
    assert dlg.name() == "Roll look"


def test_builtins_only_without_profiles(panel):
    assert _items(panel) == ["Color", "Black & White"]
    assert not panel.delete_adjust_profile_btn.isEnabled()


def test_save_apply_display_delete(panel, image, monkeypatch):
    panel.set_current_idx(0)
    image.color_profile = "bw"
    image.adjustment_settings = {k: panel._default_for(k)
                                 for k in panel.adjustment_keys}
    image.adjustment_settings["brightness"] = 20
    image.adjustment_settings["temperature"] = 15

    _stub_save_dialog(monkeypatch, "Roll look", {"tone", "profile"})
    panel._on_save_adjust_profile()

    stored = decode_profiles(panel._settings.value("adjust/profiles", "", type=str))
    assert [p["name"] for p in stored] == ["Roll look"]
    assert stored[0]["settings"]["brightness"] == 20
    assert "temperature" not in stored[0]["settings"]
    assert stored[0]["color_profile"] == "bw"
    # Built-ins, separator, profile — and the source matches it by construction.
    assert _items(panel) == ["Color", "Black & White", "", "Roll look"]
    assert panel.color_profile_combo.currentText() == "Roll look"

    # A second image state: a different tone + Color. Applying the profile
    # overwrites tone + the flag, leaves WB alone, and is one undo step.
    image.color_profile = "color"
    image.adjustment_settings["brightness"] = -5
    image.adjustment_settings["temperature"] = 40
    panel._sync_color_profile_combo(0)
    assert panel.color_profile_combo.currentText() == "Color"
    undo_before = len(image.undo_stack)
    panel.color_profile_combo.setCurrentIndex(
        panel.color_profile_combo.findData("user:Roll look"))
    assert image.adjustment_settings["brightness"] == 20
    assert image.adjustment_settings["temperature"] == 40
    assert image.color_profile == "bw"
    assert len(image.undo_stack) == undo_before + 1
    assert panel.sliders[panel.adjustment_keys.index("brightness")].value() == 20
    assert panel.delete_adjust_profile_btn.isEnabled()

    # Moving a covered slider falls the display back to the built-in row.
    image.adjustment_settings["brightness"] = 21
    panel._sync_color_profile_combo(0)
    assert panel.color_profile_combo.currentText() == "Black & White"
    assert not panel.delete_adjust_profile_btn.isEnabled()

    # Delete: store emptied, image untouched, combo back to built-ins.
    image.adjustment_settings["brightness"] = 20
    panel._sync_color_profile_combo(0)
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    panel._on_delete_adjust_profile()
    assert panel._adjust_profiles == []
    assert decode_profiles(panel._settings.value("adjust/profiles", "", type=str)) == []
    assert image.adjustment_settings["brightness"] == 20
    assert _items(panel) == ["Color", "Black & White"]
    assert panel.color_profile_combo.currentText() == "Black & White"


def test_builtin_rows_still_switch_the_flag(panel, image):
    panel.set_current_idx(0)
    panel.color_profile_combo.setCurrentIndex(
        panel.color_profile_combo.findData("builtin:bw"))
    assert image.color_profile == "bw"
    assert panel.color_profile_combo.currentIndex() == \
        SlidersPanel.COLOR_PROFILES.index("bw")


def test_save_requires_name_and_groups(panel, image, monkeypatch):
    panel.set_current_idx(0)
    _stub_save_dialog(monkeypatch, "", {"tone"})
    panel._on_save_adjust_profile()
    _stub_save_dialog(monkeypatch, "X", set())
    panel._on_save_adjust_profile()
    assert panel._adjust_profiles == []


def test_reapplying_a_matched_profile_pushes_no_undo(panel, image, monkeypatch):
    panel.set_current_idx(0)
    _stub_save_dialog(monkeypatch, "Same", {"tone"})
    panel._on_save_adjust_profile()
    before = len(image.undo_stack)
    panel.on_color_profile_changed(panel.color_profile_combo.findData("user:Same"))
    assert len(image.undo_stack) == before

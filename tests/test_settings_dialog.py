#!/usr/bin/env python3
"""
Tests for the Settings dialog (DaVinci-style Color Management tab), the
disable-camera-profile toggle, the per-image profile signature + thumbnail
mismatch flagging, and "Replace with current camera profile". See
spec/settings-page.md (§8b).
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication, QListWidgetItem, QWidget  # noqa: E402
from PySide6.QtCore import Qt  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

import cv2  # noqa: E402
from core import color_management as cm  # noqa: E402
from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402


class _StubICC:
    description = "MyCam"

    def apply(self, arr):           # no-op identity (decode still succeeds)
        return arr


class _StubDCP:
    name = "Nikon"


@pytest.fixture(autouse=True)
def _reset_profile_state():
    """Each test starts with no profile, not disabled, negative mode, no Camera
    Matrix, no active selection."""
    def _reset():
        cm.set_active_input_profile(None)
        cm.set_active_dcp_profile(None)
        cm.set_input_profile_disabled(False)
        cm.set_camera_matrix_mode(False)
        ccr_backend.positive_mode = False
        ccr_backend.density_bwpoint = False        # product default: opt-in
        ccr_backend.gamma_luminance = False        # product default: per-channel
        ccr_backend.dust_method = "automask"       # product default
        ccr_backend.active_profile_path = None
        ccr_backend.images = []
    _reset()
    yield
    _reset()


def _scan_png(tmp_path, name="negative.png", w=320, h=240, seed=7):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 20000 + 25000 * (xx / w) + 10000 * (yy / h)
    img = np.stack([base * 1.2, base, base * 0.7], axis=-1)
    img += rng.normal(0, 1200, img.shape)
    img = np.clip(img, 1000, 64000).astype(np.uint16)
    path = str(tmp_path / name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


# --------------------------------------------------------------------------- #
# color_management: signature + disable + camera_profile_active
# --------------------------------------------------------------------------- #
class TestSignatureAndDisable:
    def test_none_when_no_profile(self):
        assert cm.active_profile_signature() == "none"
        assert cm.camera_profile_active() is False

    def test_icc_active(self):
        cm.set_active_input_profile(_StubICC())
        assert cm.active_profile_signature() == "icc:MyCam"
        assert cm.camera_profile_active() is True

    def test_dcp_takes_precedence(self):
        cm.set_active_input_profile(_StubICC())
        cm.set_active_dcp_profile(_StubDCP())
        assert cm.active_profile_signature() == "dcp:Nikon"

    def test_disabled_reads_as_none(self):
        cm.set_active_input_profile(_StubICC())
        cm.set_input_profile_disabled(True)
        assert cm.active_profile_signature() == "none"
        assert cm.camera_profile_active() is False
        assert cm.input_profile_disabled() is True

    def test_backend_signature_none_in_positive_mode(self):
        cm.set_active_input_profile(_StubICC())
        ccr_backend.positive_mode = True
        assert ccr_backend.active_profile_signature() == "none"
        ccr_backend.positive_mode = False
        assert ccr_backend.active_profile_signature() == "icc:MyCam"


# --------------------------------------------------------------------------- #
# CCRImage: profile-signature stamping + decode gating
# --------------------------------------------------------------------------- #
class TestProfileSignatureStamping:
    def test_stamped_none_on_plain_load(self, tmp_path):
        img = CCRImage(_scan_png(tmp_path))
        assert img.profile_signature == "none"

    def test_stamp_reflects_active_profile(self, tmp_path):
        img = CCRImage(_scan_png(tmp_path))
        cm.set_active_input_profile(_StubICC())
        img._stamp_profile_signature()
        assert img.profile_signature == "icc:MyCam"
        cm.set_input_profile_disabled(True)
        img._stamp_profile_signature()
        assert img.profile_signature == "none"

    def test_reload_restamps_under_current_profile(self, tmp_path):
        img = CCRImage(_scan_png(tmp_path))
        img.profile_signature = "icc:Stale"        # pretend graded under another
        cm.set_active_input_profile(_StubICC())
        assert img.reload_image_decode_only() is True
        assert img.profile_signature == "icc:MyCam"

    def test_will_apply_follows_disable(self, tmp_path):
        img = CCRImage(_scan_png(tmp_path))
        cm.set_active_input_profile(_StubICC())
        assert img._input_icc_will_apply() is True
        cm.set_input_profile_disabled(True)
        assert img._input_icc_will_apply() is False


# --------------------------------------------------------------------------- #
# Backend: reset re-grades an image under the current profile
# --------------------------------------------------------------------------- #
class TestRegradePrimitive:
    def test_reset_restamps_signature(self, tmp_path):
        img = CCRImage(_scan_png(tmp_path))
        ccr_backend.images = [img]
        ccr_backend.file_paths = [img.file_path]
        img.profile_signature = "icc:Stale"
        assert ccr_backend.active_profile_signature() == "none"  # nothing active
        assert ccr_backend.reset_images_by_indices([0]) is True
        assert img.profile_signature == "none"                   # re-graded to active


# --------------------------------------------------------------------------- #
# ThumbnailList: mismatch flagging + replace-action selection
# --------------------------------------------------------------------------- #
class _StubImg:
    def __init__(self, path, sig):
        self.file_path = path
        self.profile_signature = sig
        self.display_name = None


def _thumb_list_with(sigs):
    from widgets.thumbnail_list import ThumbnailList
    ccr_backend.images = [_StubImg(f"{i}.dng", s) for i, s in enumerate(sigs)]
    tl = ThumbnailList(lambda i: None)
    for i in range(len(sigs)):
        it = QListWidgetItem("x")
        it.setData(Qt.UserRole, i)
        tl.thumbnail_list.addItem(it)
    return tl


class TestThumbnailMismatch:
    WARN = "⚠"

    def test_flags_only_mismatched(self):
        cm.set_active_input_profile(_StubICC())            # active = icc:MyCam
        tl = _thumb_list_with(["icc:MyCam", "none", "dcp:Old"])
        tl.refresh_profile_warnings()
        warned = [tl.thumbnail_list.item(i).text().startswith(self.WARN)
                  for i in range(3)]
        assert warned == [False, True, True]
        assert [tl._profile_mismatch(i) for i in range(3)] == [False, True, True]

    def test_reflags_when_profile_disabled(self):
        cm.set_active_input_profile(_StubICC())
        tl = _thumb_list_with(["icc:MyCam", "none"])
        tl.refresh_profile_warnings()
        assert tl._profile_mismatch(0) is False
        cm.set_input_profile_disabled(True)                # active -> none
        tl.refresh_profile_warnings()
        assert tl._profile_mismatch(0) is True             # icc:MyCam now mismatches
        assert tl._profile_mismatch(1) is False

    def test_matched_items_have_no_warning_prefix(self):
        # No profile active: an image graded under 'none' matches -> plain name.
        tl = _thumb_list_with(["none"])
        tl.refresh_profile_warnings()
        assert tl.thumbnail_list.item(0).text() == "0.dng"

    def test_positive_mode_refresh_clears_warnings(self):
        # Regression: toggling Positive mode re-stamps every signature to 'none'
        # and active becomes 'none'; refresh_profile_warnings (now called by
        # on_positive_mode_toggled) must clear the stale ⚠.
        cm.set_active_input_profile(_StubICC())
        tl = _thumb_list_with(["icc:Other", "icc:Other"])
        tl.refresh_profile_warnings()
        assert all(tl._profile_mismatch(i) for i in range(2))
        ccr_backend.positive_mode = True            # active signature -> 'none'
        for img in ccr_backend.images:
            img.profile_signature = "none"          # re-stamped by the re-decode
        tl.refresh_profile_warnings()
        assert not any(tl._profile_mismatch(i) for i in range(2))
        assert not tl.thumbnail_list.item(0).text().startswith(self.WARN)


class TestContentIdSignature:
    def test_content_id_distinguishes_same_named_profiles(self):
        a = _StubICC(); a.content_id = "aaaaaa"
        b = _StubICC(); b.content_id = "bbbbbb"     # same description, different file
        cm.set_active_input_profile(a)
        assert cm.active_profile_signature() == "icc:aaaaaa"
        cm.set_active_input_profile(b)
        assert cm.active_profile_signature() == "icc:bbbbbb"

    def test_falls_back_to_description_without_content_id(self):
        cm.set_active_input_profile(_StubICC())     # no content_id attribute
        assert cm.active_profile_signature() == "icc:MyCam"


class TestIT8FormatSelector:
    """The IT8 wizard exposes an explicit ICC-vs-DCP Output-format selector."""

    def _dlg(self):
        from widgets.it8_profile_dialog import IT8ProfileDialog
        return IT8ProfileDialog(None)

    def test_default_is_icc_with_type_choice(self):
        d = self._dlg()
        assert d._is_dcp() is False
        assert d.type_combo.isEnabled() is True          # matrix/cLUT available
        assert d._default_save_path().endswith(".icc")

    def test_dcp_forces_matrix_and_dcp_extension(self):
        d = self._dlg()
        d.format_combo.setCurrentIndex(1)                # DCP
        assert d._is_dcp() is True
        assert d.type_combo.isEnabled() is False         # cLUT is ICC-only
        assert d.type_combo.currentIndex() == 0          # forced 3×3 matrix
        assert d._default_save_path().endswith(".dcp")

    def test_format_switch_syncs_path_extension(self):
        d = self._dlg()
        d.format_combo.setCurrentIndex(1)
        d.save_path_edit.setText("X/p.dcp")
        d.format_combo.setCurrentIndex(0)                # back to ICC
        assert d.save_path_edit.text().endswith(".icc")


# --------------------------------------------------------------------------- #
# SettingsDialog: structure, status reflection, disable + positive wiring
# --------------------------------------------------------------------------- #
class _StubMW(QWidget):
    def __init__(self):
        super().__init__()
        self.calls = []

    def import_camera_profile_dialog(self): self.calls.append("import")
    def delete_camera_profile(self, path): self.calls.append(("delete", path))
    def create_camera_profile_from_it8(self): self.calls.append("it8")

    def on_positive_mode_toggled(self, c):
        self.calls.append(("positive", bool(c)))
        ccr_backend.positive_mode = bool(c)

    def on_rgb_merge_mode_toggled(self, c):
        self.calls.append(("rgb_merge", bool(c)))
        ccr_backend.rgb_merge_mode = bool(c)

    def on_density_bwpoint_toggled(self, c):
        self.calls.append(("density", bool(c)))
        ccr_backend.density_bwpoint = bool(c)

    def on_auto_gain_toggled(self, c):
        self.calls.append(("auto_gain", bool(c)))
        ccr_backend.auto_gain = bool(c)

    def on_gamma_mode_toggled(self, c):
        self.calls.append(("gamma_lum", bool(c)))
        ccr_backend.gamma_luminance = bool(c)

    def on_dust_method_changed(self, m):
        self.calls.append(("dust_method", str(m)))
        ccr_backend.dust_method = str(m)


# Keep every created dialog (and its stub parent) alive for the whole run:
# letting them become garbage mid-run destroys their C++ widgets at an
# arbitrary later moment, which intermittently access-violates inside
# unrelated native calls (same mitigation as test_crop_hires._HOSTS).
_DIALOGS = []


def _dialog():
    from widgets.settings_dialog import SettingsDialog
    d = SettingsDialog(_StubMW())
    _DIALOGS.append(d)
    return d


class TestSettingsDialog:
    def test_has_categories(self):
        d = _dialog()
        names = [d._sidebar.item(i).text() for i in range(d._sidebar.count())]
        assert names == ["General", "Color Management"]
        # The General tab carries the Auto gain toggle (spec/auto-gain.md).
        assert hasattr(d, "_cb_auto_gain")

    def test_no_disable_checkbox(self):
        # The disable checkbox is gone — "None" in the picker replaces it.
        d = _dialog()
        assert not hasattr(d, "_cb_disable")

    def test_status_reflects_active_profile(self):
        ccr_backend.input_icc_name = None
        ccr_backend.input_dcp_name = None
        d = _dialog()
        assert "None" in d._status.text()
        ccr_backend.input_icc_name = "MyCam.icc"
        d.refresh_color_management()
        assert "ICC" in d._status.text() and "MyCam.icc" in d._status.text()
        ccr_backend.input_icc_name = None
        ccr_backend.input_dcp_name = "Nikon.dcp"
        d.refresh_color_management()
        assert "DCP" in d._status.text() and "Nikon.dcp" in d._status.text()
        ccr_backend.input_dcp_name = None
        ccr_backend.active_profile_path = ccr_backend.CAMERA_MATRIX
        d.refresh_color_management()
        assert "Camera Matrix" in d._status.text()

    def test_buttons_delegate_to_main_window(self):
        d = _dialog()
        d._import(); d._create_it8()
        assert d._mw.calls == ["import", "it8"]

    # --- Staged toggles: nothing applies until Done (accept) -------------- #
    def test_toggles_are_staged_until_done(self):
        ccr_backend.positive_mode = False
        d = _dialog()
        d._cb_positive.setChecked(True)
        # Staged: no handler fired, backend unchanged.
        assert d._mw.calls == []
        assert ccr_backend.positive_mode is False
        d.accept()                                     # Done
        assert ("positive", True) in d._mw.calls
        assert ccr_backend.positive_mode is True

    def test_done_applies_only_changed_toggles(self):
        ccr_backend.positive_mode = False
        ccr_backend.rgb_merge_mode = False
        ccr_backend.density_bwpoint = True
        d = _dialog()
        d._cb_rgb_merge.setChecked(True)               # change one only
        d.accept()
        assert ("rgb_merge", True) in d._mw.calls
        assert not any(c[0] == "positive" for c in d._mw.calls)   # untouched
        assert not any(c[0] == "density" for c in d._mw.calls)
        assert ccr_backend.rgb_merge_mode is True

    def test_done_with_no_change_applies_nothing(self):
        d = _dialog()                                  # all at backend defaults
        d.accept()
        assert d._mw.calls == []

    def test_dust_method_staged_and_applied(self):
        # The heal-method combo follows the same staged contract as the
        # toggles, and carries the three engines (spec/dust-auto-mask.md §3).
        d = _dialog()
        ids = [d._combo_dust_method.itemData(i)
               for i in range(d._combo_dust_method.count())]
        assert ids == ["automask", "clone", "inpaint"]
        assert d._combo_dust_method.currentData() == "automask"  # seeded
        d._combo_dust_method.setCurrentIndex(
            d._combo_dust_method.findData("inpaint"))
        assert d._mw.calls == []                       # staged, not applied
        assert ccr_backend.dust_method == "automask"
        d.accept()                                     # Done
        assert ("dust_method", "inpaint") in d._mw.calls
        assert ccr_backend.dust_method == "inpaint"

    def test_close_without_done_discards_toggles(self):
        ccr_backend.positive_mode = False
        ccr_backend.density_bwpoint = True
        d = _dialog()
        d._cb_positive.setChecked(True)
        d._cb_density.setChecked(False)
        d.reject()                                     # Escape / close, not Done
        assert d._mw.calls == []
        assert ccr_backend.positive_mode is False      # nothing applied
        assert ccr_backend.density_bwpoint is True

    def test_density_checkbox_defaults_off(self):
        # Product default is OFF (density is opt-in); fixture leaves it False.
        d = _dialog()
        assert d._cb_density.isChecked() is False

    # --- Gamma mode toggle (spec/gamma-luminance-mode.md) ----------------- #
    def test_general_page_has_gamma_toggle(self):
        d = _dialog()
        assert hasattr(d, "_cb_gamma_lum")
        assert d._cb_gamma_lum.isChecked() is False    # per-channel by default

    def test_gamma_toggle_seeds_from_backend(self):
        ccr_backend.gamma_luminance = True
        d = _dialog()
        assert d._cb_gamma_lum.isChecked() is True

    def test_gamma_toggle_staged_until_done(self):
        ccr_backend.gamma_luminance = False
        d = _dialog()
        d._cb_gamma_lum.setChecked(True)
        assert d._mw.calls == []                        # staged: nothing yet
        assert ccr_backend.gamma_luminance is False
        d.accept()                                      # Done
        assert ("gamma_lum", True) in d._mw.calls
        assert ccr_backend.gamma_luminance is True

    def test_gamma_toggle_discarded_on_close(self):
        ccr_backend.gamma_luminance = False
        d = _dialog()
        d._cb_gamma_lum.setChecked(True)
        d.reject()                                      # Escape / close, not Done
        assert d._mw.calls == []
        assert ccr_backend.gamma_luminance is False

    def test_toggles_seed_from_backend_at_open(self):
        ccr_backend.rgb_merge_mode = True
        ccr_backend.density_bwpoint = False
        d = _dialog()                                   # seeded in __init__
        assert d._cb_rgb_merge.isChecked() is True
        assert d._cb_density.isChecked() is False
        # A profile refresh must NOT clobber the (possibly staged) toggles.
        d._cb_rgb_merge.setChecked(False)               # stage a change
        d.refresh_color_management()
        assert d._cb_rgb_merge.isChecked() is False     # staged value preserved


class TestCameraProfileLibrary:
    """The persistent profile library: import / list / activate / delete + the
    thumbnail picker and the Settings management list."""

    def _make(self, name):
        return cm.build_matrix_shaper_icc(
            name, (0.6, 0.3, 0.0), (0.2, 0.7, 0.06), (0.0, 0.06, 0.8),
            (1.0, 1.0, 0.0, 1.0, 0.0))

    @pytest.fixture
    def lib(self, tmp_path, monkeypatch):
        from core import catalog
        monkeypatch.setattr(catalog, "default_catalog_path",
                            lambda: str(tmp_path / "catalog.json"))
        d = ccr_backend.camera_profiles_dir()
        for nm in ("Alpha", "Beta"):
            with open(os.path.join(d, nm + ".icc"), "wb") as f:
                f.write(self._make(nm))
        yield d
        ccr_backend.set_active_profile(None)

    def test_list_sorted_with_kind(self, lib):
        ps = ccr_backend.list_camera_profiles()
        assert [p["name"] for p in ps] == ["Alpha", "Beta"]
        assert all(p["kind"] == "icc" for p in ps)

    def test_activate_and_clear_without_copy_or_delete(self, lib):
        path = os.path.join(lib, "Beta.icc")
        assert ccr_backend.set_active_profile(path) is not None
        assert ccr_backend.active_profile_path == path
        assert os.path.exists(path)                          # not moved/deleted
        assert ccr_backend.set_active_profile(None) is None
        assert ccr_backend.active_profile_path is None
        assert os.path.exists(path)

    def test_import_validates_and_dedupes(self, lib, tmp_path):
        src = tmp_path / "ext.icc"
        src.write_bytes(self._make("Ext"))
        p1 = ccr_backend.import_camera_profile(str(src))
        p2 = ccr_backend.import_camera_profile(str(src))     # same name -> de-duped
        assert os.path.exists(p1) and os.path.exists(p2)
        assert os.path.basename(p1) != os.path.basename(p2)
        bad = tmp_path / "bad.icc"
        bad.write_bytes(b"not a profile")
        with pytest.raises(Exception):
            ccr_backend.import_camera_profile(str(bad))

    def test_delete_removes_and_deactivates_active(self, lib):
        path = os.path.join(lib, "Alpha.icc")
        ccr_backend.set_active_profile(path)
        assert ccr_backend.delete_camera_profile(path) is True
        assert not os.path.exists(path)
        assert ccr_backend.active_profile_path is None        # deactivated
        assert [p["name"] for p in ccr_backend.list_camera_profiles()] == ["Beta"]

    def test_thumbnail_picker_lists_and_reflects_active(self, lib):
        from widgets.thumbnail_list import ThumbnailList
        ccr_backend.set_active_profile(os.path.join(lib, "Beta.icc"))
        tl = ThumbnailList(lambda i: None)
        tl.refresh_profile_combo()
        items = [tl.profile_combo.itemText(i) for i in range(tl.profile_combo.count())]
        assert items[0] == "None"
        assert any("Alpha" in t for t in items) and any("Beta" in t for t in items)
        assert "Beta" in tl.profile_combo.currentText()

    def test_dialog_list_marks_active_and_delete_delegates(self, lib):
        ccr_backend.set_active_profile(os.path.join(lib, "Beta.icc"))
        d = _dialog()
        labels = [d._profile_list.item(i).text() for i in range(d._profile_list.count())]
        assert any("Beta" in t and "active" in t for t in labels)
        d._profile_list.setCurrentRow(0)
        d._delete()
        assert d._mw.calls and d._mw.calls[-1][0] == "delete"


class TestRegradeVsReset:
    """Replace (regrade) keeps EVERYTHING — conversion (bwpoint/reference),
    adjustments, crop, flips, rotation, slice — and replays the conversion on the
    new decode. Reset drops it all. Both re-stamp the signature (clearing the ⚠)."""

    def _img(self, tmp_path):
        from core.ccr_image import CCRImage
        path = _scan_png(tmp_path)
        img = CCRImage(path)
        img.rotation_angle = 90
        img.fine_rotation_angle = 150
        img.crop_rect = (0.1, 0.1, 0.9, 0.9)
        img.crop_angle = 2.0
        img.horizontal_mirrored = True
        img.source_ops = [(0, (0.0, 0.0, 0.5, 0.5))]      # a slice region
        img.adjustment_settings = {"temperature": 12}
        img.converted = True
        img.profile_signature = "icc:Stale"               # graded under another profile
        ccr_backend.images = [img]
        ccr_backend.file_paths = [path]
        return img

    def test_regrade_keeps_everything_and_replays_reference(self, tmp_path, monkeypatch):
        import core.ccr_backend as B
        img = self._img(tmp_path)
        img.reference_frame = (10, 10, 200, 150)
        img.conversion_inputs = {"mode": "ref", "ref": (10, 10, 200, 150), "fine_rot": 0}
        called = {}
        def fake_ref(im, **kw):
            called["ref"] = True
            return im.resized_raw
        monkeypatch.setattr(B, "ccr_normalize_with_reference", fake_ref)
        assert ccr_backend.regrade_images_by_indices([0]) is True
        # conversion REPLAYED on the new decode
        assert called.get("ref") is True
        assert img.converted is True
        assert img.conversion_inputs == {"mode": "ref", "ref": (10, 10, 200, 150), "fine_rot": 0}
        # adjustments + geometry KEPT
        assert img.adjustment_settings == {"temperature": 12}
        assert img.rotation_angle == 90
        assert img.fine_rotation_angle == 150
        assert img.crop_rect == (0.1, 0.1, 0.9, 0.9)
        assert img.crop_angle == 2.0
        assert img.horizontal_mirrored is True
        assert img.source_ops == [(0, (0.0, 0.0, 0.5, 0.5))]
        # signature re-stamped under the current profile
        assert img.profile_signature == ccr_backend.active_profile_signature()

    def test_regrade_replays_bwpoint(self, tmp_path, monkeypatch):
        import core.ccr_backend as B
        img = self._img(tmp_path)
        img.reference_frame = None
        bw = ((100, 90, 80), (2000, 1900, 1800))
        img.conversion_inputs = {"mode": "bw", "bw": bw, "fine_rot": 0}
        called = {}
        def fake_bw(im, b, w, **kw):
            called["bw"] = (b, w)
            return im.resized_raw
        monkeypatch.setattr(B, "ccr_normalize_with_bwpoint", fake_bw)
        assert ccr_backend.regrade_images_by_indices([0]) is True
        assert called.get("bw") == bw                          # bwpoint replayed with baked anchors
        assert img.converted is True
        assert img.adjustment_settings == {"temperature": 12}
        assert img.profile_signature == ccr_backend.active_profile_signature()

    def test_reset_drops_geometry_too(self, tmp_path):
        img = self._img(tmp_path)
        assert ccr_backend.reset_images_by_indices([0]) is True
        assert img.rotation_angle == 0
        assert img.fine_rotation_angle == 0
        assert img.crop_rect is None
        assert img.horizontal_mirrored is False
        assert img.source_ops == [(0, (0.0, 0.0, 0.5, 0.5))]   # slice still preserved
        assert img.profile_signature == ccr_backend.active_profile_signature()


class TestCameraMatrixMode:
    """The non-removable 'Camera Matrix' picker entry (Adobe RGB + auto-scale),
    distinct from 'None' (bare RAW)."""

    def test_backend_activates_signature_and_clears(self):
        assert ccr_backend.set_active_profile(ccr_backend.CAMERA_MATRIX) == "Camera Matrix"
        assert cm.camera_matrix_mode() is True
        assert ccr_backend.active_profile_path == ccr_backend.CAMERA_MATRIX
        assert ccr_backend.active_profile_signature() == "camera_matrix"
        assert ccr_backend.set_active_profile(None) is None     # None -> bare RAW
        assert cm.camera_matrix_mode() is False
        assert ccr_backend.active_profile_signature() == "none"

    def test_profile_clears_camera_matrix(self, tmp_path):
        ccr_backend.set_active_profile(ccr_backend.CAMERA_MATRIX)
        blob = cm.build_matrix_shaper_icc(
            "P", (0.6, 0.3, 0.0), (0.2, 0.7, 0.06), (0.0, 0.06, 0.8),
            (1.0, 1.0, 0.0, 1.0, 0.0))
        p = tmp_path / "p.icc"; p.write_bytes(blob)
        ccr_backend.set_active_profile(str(p))
        assert cm.camera_matrix_mode() is False
        assert ccr_backend.active_profile_signature().startswith("icc:")
        ccr_backend.set_active_profile(None)

    def test_picker_lists_none_then_camera_matrix(self):
        from widgets.thumbnail_list import ThumbnailList
        ccr_backend.set_active_profile(ccr_backend.CAMERA_MATRIX)
        tl = ThumbnailList(lambda i: None)
        tl.refresh_profile_combo()
        assert [tl.profile_combo.itemText(i) for i in range(2)] == ["None", "Camera Matrix"]
        assert "Camera Matrix" in tl.profile_combo.currentText()
        ccr_backend.set_active_profile(None)

#!/usr/bin/env python3
"""Input transfer function for non-RAW decodes (spec/input-transfer-function.md).

FreeCCR's density math assumes LINEAR data, but the non-RAW reader historically
interpreted no transfer function at all — a gamma-encoded scanner TIFF was
inverted as though it were linear transmission. These tests cover the resolver
(what a file's metadata says), the math, the reader integration, the per-image
persistence, and the import dialog's gating.
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

import tifffile  # noqa: E402

from core import color_management as cm  # noqa: E402
from core import input_transfer as it  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402

# sRGB-encoded mid-grey: srgb_encode(0.5) * 65535.
SRGB_MID = int(round(float(cm.srgb_encode(np.array([0.5]))[0]) * 65535))
LINEAR_MID = 32768


def _flat(value, h=24, w=32):
    return np.full((h, w, 3), value, np.uint16)


def _write(path, value=SRGB_MID, **kwargs):
    tifffile.imwrite(path, _flat(value), **kwargs)
    return path


# --- resolver ---------------------------------------------------------------- #
class TestResolver:
    def test_untagged_tiff_is_linear(self, tmp_path):
        r = it.resolve_file_transfer(_write(str(tmp_path / "plain.tif")))
        assert r.token == it.LINEAR
        assert r.tagged is False
        assert r.identity

    def test_embedded_srgb_icc_resolves_to_lut(self, tmp_path):
        path = _write(str(tmp_path / "srgb.tif"),
                      iccprofile=cm.export_icc_bytes("srgb"))
        r = it.resolve_file_transfer(path)
        assert r.token == it.LUT and r.tagged
        assert r.luts is not None and len(r.luts) == 3
        # The curve must actually be sRGB: encoded mid-grey -> linear 0.5.
        assert abs(float(r.luts[0][SRGB_MID]) - 0.5) < 0.01

    def test_garbage_icc_degrades_to_linear(self, tmp_path):
        path = _write(str(tmp_path / "bad.tif"), iccprofile=b"not an icc profile")
        r = it.resolve_file_transfer(path)          # must not raise
        assert r.token == it.LINEAR

    def test_transferfunction_tag_resolves_to_lut(self, tmp_path):
        # A gamma-2.0 encoded->linear table over 256 entries, 3 channels.
        n = 256
        curve = (np.linspace(0, 1, n) ** 2.0 * 65535).astype(np.uint16)
        path = str(tmp_path / "tf.tif")
        tifffile.imwrite(path, _flat(SRGB_MID),
                         extratags=[(301, 'H', n * 3,
                                     tuple(np.tile(curve, 3).tolist()), False)])
        r = it.resolve_file_transfer(path)
        assert r.token == it.LUT and r.tagged
        assert abs(float(r.luts[0][32768]) - 0.25) < 0.02   # 0.5**2

    def test_missing_file_is_linear_not_raise(self, tmp_path):
        assert it.resolve_file_transfer(str(tmp_path / "nope.tif")).token == it.LINEAR


# --- tokens + math ----------------------------------------------------------- #
class TestMath:
    def test_identity_tokens(self):
        for tok in (None, "", it.LINEAR, "gamma:1.0"):
            assert it.is_identity(tok)
        for tok in (it.SRGB, it.REC709, "gamma:2.2", it.LUT):
            assert not it.is_identity(tok)

    def test_identity_returns_same_object(self):
        src = _flat(1234)
        assert it.apply_transfer(src, None) is src
        assert it.apply_transfer(src, it.LINEAR) is src

    def test_srgb_decodes_to_linear(self):
        out = it.apply_transfer(_flat(SRGB_MID), it.SRGB)
        assert abs(int(out[0, 0, 0]) - LINEAR_MID) < 250

    def test_gamma_matches_power_law(self):
        out = it.apply_transfer(_flat(32768), "gamma:2.2")
        assert abs(int(out[0, 0, 0]) - int((0.5 ** 2.2) * 65535)) < 2

    def test_rec709_round_trip(self):
        lin = np.array([0.02, 0.2, 0.5, 0.9])
        enc = np.where(lin < 0.018, lin * 4.5,
                       1.099 * np.power(lin, 0.45) - 0.099)
        back = cm.rec709_decode(enc)
        assert np.allclose(back, lin, atol=1e-4)

    def test_unknown_token_passes_through(self):
        src = _flat(1000)
        out = it.apply_transfer(src, "nonsense")
        assert np.array_equal(out, src)

    def test_gamma_of(self):
        assert it.gamma_of("gamma:2.2") == pytest.approx(2.2)
        assert it.gamma_of(it.SRGB) is None


# --- reader integration ------------------------------------------------------ #
class TestReader:
    def test_untagged_load_is_unchanged(self, tmp_path):
        """The regression guard: with no transfer set the decode is what it
        always was."""
        path = _write(str(tmp_path / "asis.tif"))
        img = CCRImage(path)
        assert abs(int(img.resized_raw[0, 0, 0]) - SRGB_MID) < 2

    def test_srgb_transfer_linearises_through_the_load_path(self, tmp_path):
        path = _write(str(tmp_path / "srgb.tif"))
        img = CCRImage(path, input_transfer=it.SRGB)
        assert abs(int(img.resized_raw[0, 0, 0]) - LINEAR_MID) < 250

    def test_embedded_transfer_uses_the_files_own_profile(self, tmp_path):
        path = _write(str(tmp_path / "emb.tif"),
                      iccprofile=cm.export_icc_bytes("srgb"))
        img = CCRImage(path, input_transfer=it.EMBEDDED)
        assert abs(int(img.resized_raw[0, 0, 0]) - LINEAR_MID) < 400

    def test_embedded_on_untagged_file_is_identity(self, tmp_path):
        path = _write(str(tmp_path / "untagged.tif"))
        img = CCRImage(path, input_transfer=it.EMBEDDED)
        assert abs(int(img.resized_raw[0, 0, 0]) - SRGB_MID) < 2

    def test_resolution_independent(self, tmp_path):
        """The export/zoom path (max_long_side=None) applies the same transform —
        the point of doing this inside the reader."""
        path = _write(str(tmp_path / "res.tif"))
        img = CCRImage(path, input_transfer=it.SRGB)
        full = img.read_image(path, max_long_side=None)
        assert abs(int(full[0, 0, 0]) - LINEAR_MID) < 250


class TestPositiveMode:
    def test_srgb_tagged_is_exact_identity_in_positive_mode(self):
        src = _flat(SRGB_MID)
        assert it.apply_transfer(src, it.SRGB, positive=True) is src

    def test_gamma18_is_re_encoded_to_srgb_in_positive_mode(self):
        out = it.apply_transfer(_flat(32768), "gamma:1.8", positive=True)
        expect = float(cm.srgb_encode(np.array([0.5 ** 1.8]))[0]) * 65535
        assert abs(int(out[0, 0, 0]) - expect) < 150


# --- batch helpers ----------------------------------------------------------- #
class TestBatch:
    def test_freeccr_merge_tiff_is_excluded(self, tmp_path):
        from core import ccr_merge
        marked = str(tmp_path / "merged.tif")
        tifffile.imwrite(marked, _flat(100),
                         software=ccr_merge.FREECCR_MERGE_TIFF_MARKER)
        plain = _write(str(tmp_path / "scan.tif"))
        got = it.tiffs_in([marked, plain, str(tmp_path / "x.arw")])
        assert got == [plain]

    def test_summarize_counts_tagged(self, tmp_path):
        a = _write(str(tmp_path / "a.tif"), iccprofile=cm.export_icc_bytes("srgb"))
        b = _write(str(tmp_path / "b.tif"))
        tagged, parts = it.summarize_batch([a, b])
        assert tagged == 1
        assert any("untagged" in p for p in parts)


# --- persistence ------------------------------------------------------------- #
class TestCatalog:
    def test_round_trip(self, tmp_path, monkeypatch):
        from core import catalog
        path = _write(str(tmp_path / "cat.tif"))
        img = CCRImage(path, input_transfer=it.SRGB)
        state = catalog.serialize_image(img)
        assert state["input_transfer"] == it.SRGB
        monkeypatch.setattr(catalog, "_import_transfer_override", lambda: None)
        back = catalog._restore_image(path, state)
        assert back.input_transfer == it.SRGB

    def test_absent_key_restores_as_none(self, tmp_path, monkeypatch):
        from core import catalog
        path = _write(str(tmp_path / "old.tif"))
        img = CCRImage(path, input_transfer=it.SRGB)
        state = catalog.serialize_image(img)
        del state["input_transfer"]              # a pre-feature catalog record
        monkeypatch.setattr(catalog, "_import_transfer_override", lambda: None)
        assert catalog._restore_image(path, state).input_transfer is None

    def test_import_decision_overrides_stored(self, tmp_path, monkeypatch):
        from core import catalog
        path = _write(str(tmp_path / "ov.tif"))
        img = CCRImage(path, input_transfer=it.SRGB)
        state = catalog.serialize_image(img)
        monkeypatch.setattr(catalog, "_import_transfer_override",
                            lambda: "gamma:2.2")
        assert catalog._restore_image(path, state).input_transfer == "gamma:2.2"


# --- dialog + gating --------------------------------------------------------- #
class TestDialog:
    def test_decision_for(self):
        from widgets.input_transfer_dialog import decision_for
        assert decision_for("embedded", "srgb") == it.EMBEDDED
        assert decision_for("linear", "srgb") == it.LINEAR
        assert decision_for("manual", "gamma:1.8") == "gamma:1.8"

    def test_defaults_to_metadata_only_when_something_is_tagged(self, tmp_path):
        from widgets.input_transfer_dialog import InputTransferDialog
        tagged = _write(str(tmp_path / "t.tif"),
                        iccprofile=cm.export_icc_bytes("srgb"))
        plain = _write(str(tmp_path / "p.tif"))
        assert InputTransferDialog([tagged]).decision() == it.EMBEDDED
        assert InputTransferDialog([plain]).decision() == it.LINEAR

    def test_manual_combo_enabled_only_for_manual(self, tmp_path):
        from widgets.input_transfer_dialog import InputTransferDialog
        dlg = InputTransferDialog([_write(str(tmp_path / "m.tif"))])
        assert not dlg._combo.isEnabled()
        dlg._rb_manual.setChecked(True)
        assert dlg._combo.isEnabled()
        assert dlg.decision() == dlg._combo.currentData()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

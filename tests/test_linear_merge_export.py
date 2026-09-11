#!/usr/bin/env python3
"""Trichrome linear TIFF export (spec/trichrome-linear-tiff-export.md).

The linear-merge format writes merge_raw_channels' output byte-for-byte as an
untagged 16-bit TIFF — no conversion, no adjustments, no geometry, no colour
management — and is offered only when merged images are loaded. No real
trichrome triplet exists in the repo, so merge_raw_channels is monkeypatched
with a known array.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

import tifffile  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402

from core import ccr_merge  # noqa: E402
from core.ccr_backend import ccr_backend  # noqa: E402
from widgets.export_dialog import ExportSettingsDialog  # noqa: E402

MERGED = (np.linspace(0, 65535, 48 * 64 * 3, dtype=np.float64)
          .reshape(48, 64, 3).astype(np.uint16))


@pytest.fixture(autouse=True)
def _clean_backend():
    """CCRBackend is an enforced singleton; leave it empty for other modules."""
    yield
    ccr_backend.images = []
    ccr_backend.file_paths = []


@pytest.fixture
def fake_merge(monkeypatch):
    calls = []

    def _fake(sources, preview=False, demosaic=False, mono=False):
        calls.append((tuple(sources), preview))
        return MERGED.copy(), MERGED.shape[:2]

    monkeypatch.setattr(ccr_merge, "merge_raw_channels", _fake)
    return calls


def _merged_stub(tmp_path, name="tri_RGB.arw", converted=False):
    return SimpleNamespace(
        is_merged=True,
        merge_sources=[str(tmp_path / f"{c}.arw") for c in "rgb"],
        file_path=str(tmp_path / name),
        display_name=name,
        converted=converted,
        conversion_inputs={"mode": "bw", "bw": ((1, 1, 1), None),
                           "fine_rot": 0, "density": False} if converted else None,
        reference_frame=None,
        resized_raw=MERGED,
        original_full_size=MERGED.shape[:2],
    )


def _normal_stub(tmp_path, name="plain.png", converted=True):
    return SimpleNamespace(
        is_merged=False, merge_sources=None,
        file_path=str(tmp_path / name), display_name=name,
        converted=converted, conversion_inputs=None, reference_frame=None,
        resized_raw=MERGED, original_full_size=MERGED.shape[:2],
    )


# --- backend write path ------------------------------------------------------

def test_linear_export_is_byte_exact_untagged_tiff(tmp_path, fake_merge):
    ccr_backend.images = [_merged_stub(tmp_path)]
    out = str(tmp_path / "out.arw")  # wrong ext on purpose
    assert ccr_backend.export_image_by_index(0, out, linear_merge=True)
    written = str(tmp_path / "out.tiff")
    assert os.path.exists(written)
    with tifffile.TiffFile(written) as tf:
        arr = tf.asarray()
        page = tf.pages[0]
        assert 34675 not in page.tags  # no InterColorProfile (untagged)
    assert arr.dtype == np.uint16
    assert np.array_equal(arr, MERGED)
    # Full-resolution re-merge, not the preview
    assert fake_merge == [(tuple(ccr_backend.images[0].merge_sources), False)]


def test_linear_export_bypasses_conversion_and_adjustments(tmp_path, fake_merge,
                                                           monkeypatch):
    """A CONVERTED merged image must export through the linear path without
    touching any conversion pipeline or apply_adjustments."""
    import core.ccr_backend as backend_mod

    def _boom(*a, **k):
        raise AssertionError("conversion pipeline must not run for linear export")

    for fn in ("ccr_normalize_with_bwpoint", "ccr_normalize_with_reference",
               "ccr_normalize_with_refparams", "ccr_export_positive"):
        monkeypatch.setattr(backend_mod, fn, _boom)

    stub = _merged_stub(tmp_path, converted=True)
    stub.apply_adjustments = _boom
    ccr_backend.images = [stub]
    assert ccr_backend.export_image_by_index(
        0, str(tmp_path / "conv.tiff"), linear_merge=True)
    assert np.array_equal(tifffile.imread(str(tmp_path / "conv.tiff")), MERGED)


def test_linear_export_rejects_non_merged(tmp_path, fake_merge):
    ccr_backend.images = [_normal_stub(tmp_path)]
    assert not ccr_backend.export_image_by_index(
        0, str(tmp_path / "no.tiff"), linear_merge=True)
    assert not os.path.exists(str(tmp_path / "no.tiff"))
    result = ccr_backend.export_items([(0, str(tmp_path / "no2.tiff"))],
                                      linear_merge=True)
    assert result["failed"] == 1 and result["exported"] == 0


def test_normal_export_unaffected_by_default_flag(tmp_path, fake_merge,
                                                  monkeypatch):
    """linear_merge defaults False: a normal export still routes to the
    conversion pipelines."""
    import core.ccr_backend as backend_mod
    routed = []
    monkeypatch.setattr(backend_mod, "ccr_normalize_with_bwpoint",
                        lambda *a, **k: routed.append("bw"))
    stub = _merged_stub(tmp_path, converted=True)
    ccr_backend.images = [stub]
    assert ccr_backend.export_image_by_index(0, str(tmp_path / "n.tiff"))
    assert routed == ["bw"]


# --- framing: crop + slice chain (round 3) -----------------------------------

def test_linear_export_applies_axis_aligned_crop_bit_exact(tmp_path, fake_merge):
    from core.ccr_processor import apply_crop_to_image
    stub = _merged_stub(tmp_path)
    stub.crop_rect = (0.25, 0.25, 0.75, 0.75)
    stub.crop_angle = 0.0
    ccr_backend.images = [stub]
    out = str(tmp_path / "crop.tiff")
    assert ccr_backend.export_image_by_index(0, out, linear_merge=True)
    written = tifffile.imread(out)
    expected = apply_crop_to_image(MERGED, stub.crop_rect, 0.0)
    assert written.dtype == np.uint16
    assert written.shape[0] < MERGED.shape[0] and written.shape[1] < MERGED.shape[1]
    assert np.array_equal(written, expected)  # pure sub-array, values untouched


def test_linear_export_applies_angled_crop(tmp_path, fake_merge):
    stub = _merged_stub(tmp_path)
    stub.crop_rect = (0.25, 0.25, 0.75, 0.75)
    stub.crop_angle = 10.0
    ccr_backend.images = [stub]
    out = str(tmp_path / "angled.tiff")
    assert ccr_backend.export_image_by_index(0, out, linear_merge=True)
    written = tifffile.imread(out)
    # The rotated crop outputs the box's own dimensions (interpolated values).
    assert written.shape == (MERGED.shape[0] // 2, MERGED.shape[1] // 2, 3)
    assert written.dtype == np.uint16


def test_linear_export_applies_slice_chain_and_crop(tmp_path, fake_merge):
    """A sliced merged image exports its framed region (slice chain, then the
    crop defined in the sliced frame's space) — values stay bit-exact for
    unrotated framing."""
    from types import MethodType
    from core.ccr_image import CCRImage

    stub = _merged_stub(tmp_path)
    stub.source_ops = [(0, (0.5, 0.0, 1.0, 0.5))]  # right-top quadrant
    stub._apply_source_ops = MethodType(CCRImage._apply_source_ops, stub)
    stub.crop_rect = (0.0, 0.0, 0.5, 1.0)          # left half OF THE SLICE
    stub.crop_angle = 0.0
    ccr_backend.images = [stub]
    out = str(tmp_path / "slice.tiff")
    assert ccr_backend.export_image_by_index(0, out, linear_merge=True)
    written = tifffile.imread(out)
    h, w = MERGED.shape[:2]
    assert np.array_equal(written, MERGED[0:h // 2, w // 2:w // 2 + w // 4])


# --- dialog ------------------------------------------------------------------

@pytest.fixture
def _export_settings_guard():
    """Snapshot/restore the persisted export/* keys the dialog writes."""
    s = QSettings("FreeCCR", "FreeCCR")
    keys = ["export/destination", "export/scope", "export/filename_template",
            "export/format", "export/colorspace", "export/jpeg_quality",
            "export/resize", "export/conflict", "export/open_folder"]
    saved = {k: s.value(k) for k in keys}
    yield
    for k, v in saved.items():
        if v is None:
            s.remove(k)
        else:
            s.setValue(k, v)


def _select_format(dialog, data):
    idx = dialog.format_combo.findData(data)
    assert idx >= 0
    dialog.format_combo.setCurrentIndex(idx)


def test_linear_entry_absent_without_merged_images(tmp_path,
                                                   _export_settings_guard):
    ccr_backend.images = [_normal_stub(tmp_path)]
    dlg = ExportSettingsDialog()
    assert dlg.format_combo.findData("linear") == -1


def test_linear_entry_and_scope_with_merged_images(tmp_path,
                                                   _export_settings_guard):
    merged = _merged_stub(tmp_path)               # NOT converted
    normal = _normal_stub(tmp_path, converted=True)
    ccr_backend.images = [merged, normal]
    dlg = ExportSettingsDialog(current_idx=1, selected_indices=[0, 1])
    assert dlg.format_combo.findData("linear") >= 0

    # Normal format: converted set (the normal image only)
    _select_format(dlg, "tiff")
    assert dlg._scope_indices() == [1]

    # Linear: merged set only, regardless of conversion; current (normal
    # image) is not exportable here
    _select_format(dlg, "linear")
    assert dlg._scope_indices() == [0]
    assert "All merged images (1)" in dlg.scope_all_radio.text()
    assert not dlg.scope_current_radio.isEnabled()
    assert dlg.scope_selected_radio.isEnabled()
    assert "(1)" in dlg.scope_selected_radio.text()
    assert not dlg.resize_combo.isEnabled()
    assert not dlg.colorspace_combo.isVisible()

    # Switching back restores the normal semantics
    _select_format(dlg, "tiff")
    assert dlg._scope_indices() == [1]
    assert dlg.resize_combo.isEnabled()


def test_plan_resolution_for_linear(tmp_path, _export_settings_guard):
    ccr_backend.images = [_merged_stub(tmp_path)]
    dlg = ExportSettingsDialog(current_idx=0)
    _select_format(dlg, "linear")
    dlg.dest_edit.setText(str(tmp_path / "outdir"))
    dlg._on_export_clicked()
    assert dlg.plan is not None
    assert dlg.plan.linear_merge is True
    assert dlg.plan.jpg_output is False
    assert dlg.plan.max_long_side is None
    assert len(dlg.plan.items) == 1
    assert dlg.plan.items[0][1].endswith(".tiff")


def test_linear_save_does_not_clobber_resize_pref(tmp_path,
                                                  _export_settings_guard):
    s = QSettings("FreeCCR", "FreeCCR")
    s.setValue("export/resize", 2000)
    s.setValue("export/colorspace", "prophoto")
    ccr_backend.images = [_merged_stub(tmp_path)]
    dlg = ExportSettingsDialog(current_idx=0)
    _select_format(dlg, "linear")
    dlg._save_settings()
    assert s.value("export/resize", type=int) == 2000
    assert s.value("export/colorspace", type=str) == "prophoto"
    assert s.value("export/format", type=str) == "linear"


def test_restored_linear_format_parks_resize_pref(tmp_path,
                                                  _export_settings_guard):
    s = QSettings("FreeCCR", "FreeCCR")
    s.setValue("export/format", "linear")
    s.setValue("export/resize", 2000)
    ccr_backend.images = [_merged_stub(tmp_path)]
    dlg = ExportSettingsDialog(current_idx=0)
    # Restored into linear: resize locked at Original size…
    assert dlg._is_linear()
    assert not dlg.resize_combo.isEnabled()
    assert dlg.resize_combo.currentData() == 0
    # …and switching to a normal format brings the saved preference back.
    _select_format(dlg, "tiff")
    assert dlg.resize_combo.isEnabled()
    assert dlg.resize_combo.currentData() == 2000

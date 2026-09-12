#!/usr/bin/env python3
"""Trichrome merge detail: Monochrome read (spec/trichrome-mono-read.md).

Monochrome reads every source RAW's whole visible mosaic as a monochrome
sensor — one luminance sample per photosite, full sensor resolution, no
demosaic, the declared CFA ignored — for mono-converted bodies whose RAW still
reports RGGB. Captured per image at import (CCRImage.merge_mono) and reproduced
by every re-read.
"""
import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core import ccr_merge  # noqa: E402
from core import catalog  # noqa: E402
from core import it8_profile as it8  # noqa: E402
from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402

ARW = os.path.join(os.path.dirname(__file__), "..", "example_raw", "DSC07096.ARW")


@pytest.fixture(autouse=True)
def _clean_backend():
    saved = (ccr_backend.rgb_merge_demosaic, ccr_backend.rgb_merge_mono,
             ccr_backend.rgb_merge_mode)
    yield
    ccr_backend.images = []
    ccr_backend.file_paths = []
    (ccr_backend.rgb_merge_demosaic, ccr_backend.rgb_merge_mono,
     ccr_backend.rgb_merge_mode) = saved


@pytest.fixture
def capture_merge(monkeypatch):
    calls = []

    def _fake(sources, preview=False, demosaic=False, mono=False):
        calls.append({"sources": tuple(sources), "preview": preview,
                      "demosaic": demosaic, "mono": mono})
        return np.full((40, 60, 3), 1000, dtype=np.uint16), (40, 60)

    monkeypatch.setattr(ccr_merge, "merge_raw_channels", _fake)
    return calls


def _png(tmp_path, name="r.png"):
    """A tiny real image file (the merge decode is monkeypatched; only the
    file's existence and name matter)."""
    path = str(tmp_path / name)
    tmp = str(tmp_path / "_w.png")
    cv2.imwrite(tmp, np.full((32, 48, 3), 90, dtype=np.uint8))
    os.replace(tmp, path)
    return path


# --- pure read ----------------------------------------------------------------

def test_mono_plane_subtracts_per_site_black_at_full_resolution():
    colors = np.tile(np.array([[0, 1], [3, 2]]), (3, 4))        # 6x8 RGGB tile
    black = [100, 200, 300, 400]
    value = np.arange(48, dtype=np.float32).reshape(6, 8) * 10
    mosaic = (value + np.asarray(black, dtype=np.float32)[colors]).astype(np.uint16)
    plane = ccr_merge.mono_plane_from_mosaic(mosaic, colors, black)
    assert plane.shape == (6, 8) and plane.dtype == np.float32
    np.testing.assert_array_equal(plane, value)                 # no 2x2 print


def test_mono_plane_equal_levels_and_clip():
    mosaic = np.array([[500, 10], [512, 2000]], dtype=np.uint16)
    plane = ccr_merge.mono_plane_from_mosaic(
        mosaic, np.zeros((2, 2), int), [512, 512, 512, 512])
    np.testing.assert_array_equal(plane, [[0, 0], [0, 1488]])  # clipped at 0
    # No black levels: values pass through untouched.
    np.testing.assert_array_equal(
        ccr_merge.mono_plane_from_mosaic(mosaic), mosaic.astype(np.float32))


def test_mono_plane_rejects_non_mosaic():
    with pytest.raises(ValueError):
        ccr_merge.mono_plane_from_mosaic(np.zeros((4, 4, 3), np.uint16))


def test_bin2x2_averages_and_drops_odd_edge():
    p = np.arange(15, dtype=np.float32).reshape(3, 5)
    out = ccr_merge.bin2x2(p)
    assert out.shape == (1, 2)
    np.testing.assert_allclose(out, [[(0 + 1 + 5 + 6) / 4, (2 + 3 + 7 + 8) / 4]])


# --- real RAW (decodes the example ARW; skipped if absent) ---------------------

@pytest.mark.skipif(not os.path.exists(ARW), reason="example ARW not present")
def test_mono_read_is_the_whole_mosaic():
    import rawpy
    with rawpy.imread(ARW) as raw:
        mosaic = np.asarray(raw.raw_image_visible).copy()
        colors = np.asarray(raw.raw_colors_visible).copy()
        black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
        white = float(raw.white_level)

    merged, full = ccr_merge.merge_raw_channels([ARW] * 3, mono=True)
    assert merged.dtype == np.uint16 and merged.shape == mosaic.shape + (3,)
    assert full == mosaic.shape                       # full sensor, no halving
    # Same file for all three frames -> identical planes.
    np.testing.assert_array_equal(merged[..., 0], merged[..., 1])
    np.testing.assert_array_equal(merged[..., 0], merged[..., 2])
    # Every photosite is its own sample: (raw - its black) * 65535/white.
    y, x = slice(1000, 1064), slice(2000, 2064)
    expect = np.clip((mosaic[y, x] - black[colors[y, x]]) * (65535.0 / white),
                     0, 65535)
    assert np.abs(merged[y, x, 0].astype(np.float32) - expect).max() <= 1.0

    prev, full_prev = ccr_merge.merge_raw_channels([ARW] * 3, preview=True,
                                                   mono=True)
    assert full_prev == full                          # canonical size stable
    assert prev.shape[:2] == (mosaic.shape[0] // 2, mosaic.shape[1] // 2)


@pytest.mark.skipif(not os.path.exists(ARW), reason="example ARW not present")
def test_mono_overrides_demosaic_and_photosite():
    a, _ = ccr_merge.merge_raw_channels([ARW] * 3, preview=True, mono=True,
                                        demosaic=True)
    b, _ = ccr_merge.merge_raw_channels([ARW] * 3, preview=True, mono=True,
                                        demosaic=False)
    np.testing.assert_array_equal(a, b)


# --- per-image capture and threading -----------------------------------------

def test_ccrimage_captures_and_forwards_mono(tmp_path, capture_merge):
    p = _png(tmp_path)
    img = CCRImage(p, is_merged=True, merge_sources=[p, p, p], merge_mono=True)
    assert img.merge_mono is True
    assert capture_merge[-1]["mono"] is True

    img2 = CCRImage(p, is_merged=True, merge_sources=[p, p, p])
    assert img2.merge_mono is False                    # default: off
    assert capture_merge[-1]["mono"] is False


def test_linear_export_forwards_mono(tmp_path, capture_merge):
    stub = SimpleNamespace(is_merged=True, merge_sources=["r", "g", "b"],
                           merge_demosaic=True, merge_mono=True)
    ccr_backend.images = [stub]
    assert ccr_backend.export_image_by_index(
        0, str(tmp_path / "lin.tiff"), linear_merge=True)
    assert capture_merge[-1]["mono"] is True


def test_duplicate_inherits_mono(tmp_path, capture_merge):
    p = _png(tmp_path)
    img = CCRImage(p, is_merged=True, merge_sources=[p, p, p], merge_mono=True)
    ccr_backend.images = [img]
    ccr_backend.file_paths = [p]
    assert ccr_backend.duplicate_images_by_indices([0]) == 1
    assert ccr_backend.images[1].merge_mono is True


def test_loader_stamps_the_backend_mono_flag(tmp_path, capture_merge):
    paths = [_png(tmp_path, f"tri_{c}.arw") for c in "abc"]  # RAW-named
    ccr_backend.rgb_merge_mode = True
    ccr_backend.rgb_merge_mono = True
    try:
        count = ccr_backend.load_images_from_files(paths)
    finally:
        ccr_backend.rgb_merge_mode = False
    assert count == 1
    assert ccr_backend.images[0].merge_mono is True
    assert capture_merge[-1]["mono"] is True


def test_catalog_round_trips_mono(tmp_path, capture_merge):
    p = _png(tmp_path)
    img = CCRImage(p, is_merged=True, merge_sources=[p, p, p], merge_mono=True)
    state = catalog.serialize_image(img)
    assert state["merge_mono"] is True
    assert catalog._restore_image(p, state).merge_mono is True
    # A record written before this feature has no key -> restores as off.
    legacy = {k: v for k, v in state.items() if k != "merge_mono"}
    assert catalog._restore_image(p, legacy).merge_mono is False
    # The live global setting wins over the stored copy, like merge_demosaic.
    assert catalog._restore_image(p, state, live_merge_mono=False).merge_mono is False


def test_it8_target_forwards_mono(tmp_path, capture_merge):
    p = _png(tmp_path)
    it8.decode_target_merged([p, p, p], demosaic=True, mono=True)
    assert capture_merge[-1]["mono"] is True


# --- settings dialog + MainWindow handler --------------------------------------

class _StubMW(QWidget):
    def __init__(self):
        super().__init__()
        self.detail_calls = []

    def on_rgb_merge_detail_changed(self, mode):
        self.detail_calls.append(mode)


# Keep dialogs alive until exit — see the note in test_settings_dialog.py
# (collecting a parentless dialog mid-run corrupts the heap on Windows).
_LIVE_DIALOGS = []


def _dialog():
    from widgets.settings_dialog import SettingsDialog
    d = SettingsDialog(_StubMW())
    _LIVE_DIALOGS.append(d)
    return d


def test_dialog_offers_three_modes_and_seeds_mono():
    ccr_backend.rgb_merge_mono = True
    d = _dialog()
    combo = d._combo_merge_detail
    assert [combo.itemData(i) for i in range(combo.count())] == [
        "demosaic", "photosite", "mono"]
    assert combo.currentData() == "mono"


def test_dialog_applies_only_a_changed_mode():
    ccr_backend.rgb_merge_mono = False
    ccr_backend.rgb_merge_demosaic = True
    d = _dialog()
    assert d._combo_merge_detail.currentData() == "demosaic"
    d._apply_pending()
    assert d._mw.detail_calls == []                    # unchanged -> no call
    d._combo_merge_detail.setCurrentIndex(d._combo_merge_detail.findData("mono"))
    d._apply_pending()
    assert d._mw.detail_calls == ["mono"]


def test_main_window_handler_sets_and_persists_flags():
    from ui.main_window import MainWindow
    stored = {}
    fake = SimpleNamespace(
        _MERGE_DETAIL_LABELS=MainWindow._MERGE_DETAIL_LABELS,
        _settings=SimpleNamespace(setValue=stored.__setitem__),
        sliders_panel=SimpleNamespace(set_temporary_hint=lambda *a, **k: None))
    ccr_backend.rgb_merge_demosaic = False              # photosite before

    MainWindow.on_rgb_merge_detail_changed(fake, "mono")
    assert ccr_backend.rgb_merge_mono is True
    assert ccr_backend.rgb_merge_demosaic is False      # left as it was
    assert stored == {"import/rgb_merge_mono": True,
                      "import/rgb_merge_demosaic": False}

    MainWindow.on_rgb_merge_detail_changed(fake, "demosaic")
    assert ccr_backend.rgb_merge_mono is False
    assert ccr_backend.rgb_merge_demosaic is True
    assert stored["import/rgb_merge_mono"] is False

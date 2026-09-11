#!/usr/bin/env python3
"""Trichrome merge detail: demosaic vs single photosite
(spec/trichrome-demosaic-mode.md).

Demosaic (the new default) decodes each frame through a LINEAR (bilinear,
per-channel — no inter-channel mixing) demosaic at full sensor resolution;
single photosite is the original raw-mosaic phase slice at half resolution.
The choice is captured per image at import and reproduced by every re-read.
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

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core import ccr_merge  # noqa: E402
from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402

ARW = os.path.join(os.path.dirname(__file__), "..", "example_raw", "DSC07096.ARW")


@pytest.fixture(autouse=True)
def _clean_backend():
    saved_demosaic = ccr_backend.rgb_merge_demosaic
    saved_mode = ccr_backend.rgb_merge_mode
    yield
    ccr_backend.images = []
    ccr_backend.file_paths = []
    ccr_backend.rgb_merge_demosaic = saved_demosaic
    ccr_backend.rgb_merge_mode = saved_mode


@pytest.fixture
def capture_merge(monkeypatch):
    calls = []

    def _fake(sources, preview=False, demosaic=False, mono=False):
        calls.append({"sources": tuple(sources), "preview": preview,
                      "demosaic": demosaic})
        return np.full((40, 60, 3), 1000, dtype=np.uint16), (40, 60)

    monkeypatch.setattr(ccr_merge, "merge_raw_channels", _fake)
    return calls


def _png(tmp_path, name="r.png"):
    """A tiny real image file; a non-image extension (e.g. .arw for loader
    tests) is written as PNG bytes then renamed — the merge decode is
    monkeypatched, only the file's existence and name matter."""
    path = str(tmp_path / name)
    tmp = str(tmp_path / "_w.png")
    cv2.imwrite(tmp, np.full((32, 48, 3), 90, dtype=np.uint8))
    os.replace(tmp, path)
    return path


# --- real-RAW integration (decodes the example ARW; skipped if absent) -------

@pytest.mark.skipif(not os.path.exists(ARW), reason="example ARW not present")
def test_demosaic_is_full_resolution_photosite_is_half():
    m_ph, full_ph = ccr_merge.merge_raw_channels([ARW] * 3, demosaic=False)
    m_dm, full_dm = ccr_merge.merge_raw_channels([ARW] * 3, demosaic=True)
    assert m_dm.dtype == np.uint16 and m_dm.ndim == 3 and m_dm.shape[2] == 3
    assert m_dm.mean() > 0
    assert full_ph == m_ph.shape[:2]
    assert full_dm == m_dm.shape[:2]
    # Demosaic is the full sensor; photosite is the 2x2-binned half.
    assert abs(m_dm.shape[0] - 2 * m_ph.shape[0]) <= 4
    assert abs(m_dm.shape[1] - 2 * m_ph.shape[1]) <= 4


@pytest.mark.skipif(not os.path.exists(ARW), reason="example ARW not present")
def test_demosaic_preview_decodes_half_but_reports_full():
    m_full, full_full = ccr_merge.merge_raw_channels([ARW] * 3, demosaic=True)
    m_prev, full_prev = ccr_merge.merge_raw_channels([ARW] * 3, preview=True,
                                                     demosaic=True)
    assert full_prev == full_full                      # canonical size stable
    assert abs(2 * m_prev.shape[0] - m_full.shape[0]) <= 4
    assert abs(2 * m_prev.shape[1] - m_full.shape[1]) <= 4


# --- per-image capture and threading -----------------------------------------

def test_ccrimage_captures_and_forwards_the_flag(tmp_path, capture_merge):
    p = _png(tmp_path)
    img = CCRImage(p, is_merged=True, merge_sources=[p, p, p],
                   merge_demosaic=False)
    assert img.merge_demosaic is False
    assert capture_merge[-1]["demosaic"] is False

    img2 = CCRImage(p, is_merged=True, merge_sources=[p, p, p])
    assert img2.merge_demosaic is True                 # default: demosaic
    assert capture_merge[-1]["demosaic"] is True


def test_linear_export_reproduces_the_captured_decode(tmp_path, capture_merge):
    stub = SimpleNamespace(is_merged=True, merge_sources=["r", "g", "b"],
                           merge_demosaic=False)
    ccr_backend.images = [stub]
    assert ccr_backend.export_image_by_index(
        0, str(tmp_path / "lin.tiff"), linear_merge=True)
    assert capture_merge[-1] == {"sources": ("r", "g", "b"),
                                 "preview": False, "demosaic": False}


def test_duplicate_inherits_the_flag(tmp_path, capture_merge):
    p = _png(tmp_path)
    img = CCRImage(p, is_merged=True, merge_sources=[p, p, p],
                   merge_demosaic=False)
    ccr_backend.images = [img]
    ccr_backend.file_paths = [p]
    assert ccr_backend.duplicate_images_by_indices([0]) == 1
    dup = ccr_backend.images[1]           # copy is inserted after its source
    assert dup.is_duplicate
    assert dup.is_merged and dup.merge_demosaic is False


def test_loader_stamps_the_backend_flag(tmp_path, capture_merge):
    paths = [_png(tmp_path, f"tri_{c}.arw") for c in "abc"]  # RAW-named
    ccr_backend.rgb_merge_mode = True
    ccr_backend.rgb_merge_demosaic = False
    try:
        count = ccr_backend.load_images_from_files(paths)
    finally:
        ccr_backend.rgb_merge_mode = False
    assert count == 1
    assert ccr_backend.images[0].merge_demosaic is False
    assert capture_merge[-1]["demosaic"] is False

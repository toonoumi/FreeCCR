#!/usr/bin/env python3
"""Replace trichrome originals with a linear TIFF (spec/merge-linear-tiff-replace.md).

bake_merged_to_linear_tiff writes each merged image's full-resolution linear
TIFF, verifies it reads back valid, then PERMANENTLY deletes its source RAWs —
never deleting a source whose replacement failed or when cancelled. No real
trichrome triplet exists in the repo, so merge_raw_channels is monkeypatched;
the "source RAWs" are real empty temp files so deletion is observable.
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

from core import ccr_merge, ccr_processor  # noqa: E402
from core.ccr_backend import ccr_backend  # noqa: E402

MERGED = (np.linspace(0, 65535, 48 * 64 * 3, dtype=np.float64)
          .reshape(48, 64, 3).astype(np.uint16))


@pytest.fixture(autouse=True)
def _clean_backend():
    """CCRBackend is an enforced singleton; leave it empty + merge off for others."""
    yield
    ccr_backend.images = []
    ccr_backend.file_paths = []
    ccr_backend.rgb_merge_mode = False
    ccr_backend.last_merge_error = None


@pytest.fixture
def fake_merge(monkeypatch):
    def _fake(sources, preview=False, demosaic=False):
        return MERGED.copy(), MERGED.shape[:2]
    monkeypatch.setattr(ccr_merge, "merge_raw_channels", _fake)


def _make_merged(tmp_path, prefix):
    """A stub merged image backed by three real (empty) source files, so the
    delete step has something to remove and assertions can see it happen."""
    sources = []
    for c in "RGB":
        p = tmp_path / f"{prefix}_{c}.arw"
        p.write_bytes(b"\x00" * 16)
        sources.append(str(p))
    return SimpleNamespace(
        is_merged=True,
        merge_sources=sources,
        file_path=sources[0],
        display_name=f"{prefix}_RGB.arw",
        source_ops=[],
        crop_rect=None,
        crop_angle=0.0,
        merge_demosaic=True,
    )


def test_bake_writes_tiffs_and_deletes_sources(tmp_path, fake_merge):
    im1, im2 = _make_merged(tmp_path, "a"), _make_merged(tmp_path, "b")
    res = ccr_backend.bake_merged_to_linear_tiff([im1, im2])

    assert res["cancelled"] is False
    assert res["failures"] == []
    assert len(res["tiff_paths"]) == 2
    for out in res["tiff_paths"]:
        assert os.path.exists(out) and out.endswith(".tiff")
        with tifffile.TiffFile(out) as tf:
            arr = tf.asarray()
        assert arr.dtype == np.uint16 and arr.shape == (48, 64, 3)
        np.testing.assert_array_equal(arr, MERGED)
    # every original RAW is gone
    assert len(res["deleted"]) == 6
    for im in (im1, im2):
        for s in im.merge_sources:
            assert not os.path.exists(s)


def test_write_failure_preserves_that_images_sources(tmp_path, monkeypatch):
    good, bad = _make_merged(tmp_path, "good"), _make_merged(tmp_path, "bad")

    def _fake(sources, preview=False, demosaic=False):
        if any("bad" in s for s in sources):
            raise ValueError("boom")
        return MERGED.copy(), MERGED.shape[:2]
    monkeypatch.setattr(ccr_merge, "merge_raw_channels", _fake)

    res = ccr_backend.bake_merged_to_linear_tiff([good, bad])

    assert len(res["tiff_paths"]) == 1
    assert any("bad" in name for name, _reason in res["failures"])
    # the good frame's sources are deleted; the failed frame's are untouched
    for s in good.merge_sources:
        assert not os.path.exists(s)
    for s in bad.merge_sources:
        assert os.path.exists(s)


def test_verify_rejects_invalid_tiff_and_keeps_sources(tmp_path, fake_merge, monkeypatch):
    im = _make_merged(tmp_path, "c")

    def _bad_write(out, image, **kwargs):
        # A valid TIFF but the WRONG shape (2-D) — the readback verify must
        # reject it, and the sources (only copy) must survive.
        tifffile.imwrite(ccr_processor.safe_unicode_path(out),
                         np.zeros((10, 10), np.uint16))
        return True
    monkeypatch.setattr(ccr_processor, "safe_tifffile_imwrite", _bad_write)

    res = ccr_backend.bake_merged_to_linear_tiff([im])

    assert res["tiff_paths"] == []
    assert res["deleted"] == []
    assert res["failures"] and "verification" in res["failures"][0][1]
    for s in im.merge_sources:
        assert os.path.exists(s)


def test_cancel_deletes_nothing_and_removes_written_tiffs(tmp_path, fake_merge):
    im1, im2 = _make_merged(tmp_path, "a"), _make_merged(tmp_path, "b")
    calls = {"n": 0}

    def cancel_flag():
        calls["n"] += 1
        return calls["n"] > 1     # let the first image write, cancel before the 2nd

    res = ccr_backend.bake_merged_to_linear_tiff([im1, im2], cancel_flag=cancel_flag)

    assert res["cancelled"] is True
    assert res["tiff_paths"] == [] and res["deleted"] == []
    for im in (im1, im2):
        for s in im.merge_sources:
            assert os.path.exists(s)          # nothing deleted
    assert not list(tmp_path.glob("*.tiff"))  # the written TIFF was cleaned up


def test_conflict_safe_naming_never_overwrites(tmp_path, fake_merge):
    im = _make_merged(tmp_path, "a")
    stem = os.path.splitext(im.display_name)[0]
    existing = tmp_path / f"{stem}.tiff"
    existing.write_bytes(b"OLD")

    res = ccr_backend.bake_merged_to_linear_tiff([im])

    out = res["tiff_paths"][0]
    assert out.endswith(f"{stem}_2.tiff")
    assert existing.read_bytes() == b"OLD"    # pre-existing file untouched


def test_baked_tiff_is_marked_and_detected(tmp_path, fake_merge):
    im = _make_merged(tmp_path, "a")
    res = ccr_backend.bake_merged_to_linear_tiff([im])
    out = res["tiff_paths"][0]
    # the baked TIFF carries the FreeCCR merge marker; a plain TIFF does not
    assert ccr_merge.is_freeccr_merge_tiff(out) is True
    plain = str(tmp_path / "plain.tiff")
    tifffile.imwrite(plain, MERGED)
    assert ccr_merge.is_freeccr_merge_tiff(plain) is False
    # a RAW extension / missing file is never a merge TIFF
    assert ccr_merge.is_freeccr_merge_tiff(str(tmp_path / "x.arw")) is False


def test_marked_tiff_opens_in_merge_mode_without_toggling(tmp_path, fake_merge):
    im = _make_merged(tmp_path, "a")
    out = ccr_backend.bake_merged_to_linear_tiff([im])["tiff_paths"][0]
    # merge mode ON: a marked TIFF loads as one NORMAL image, no error, no toggle
    ccr_backend.rgb_merge_mode = True
    ccr_backend.last_merge_error = None
    n = ccr_backend.load_images_from_files([out])
    assert n == 1
    assert len(ccr_backend.images) == 1
    assert not getattr(ccr_backend.images[0], "is_merged", False)
    assert not ccr_backend.last_merge_error


def test_force_no_merge_loads_tiff_as_normal(tmp_path):
    p = str(tmp_path / "lin.tiff")
    tifffile.imwrite(p, MERGED)
    ccr_backend.rgb_merge_mode = True
    n = ccr_backend.load_images_from_files([p], force_no_merge=True)
    assert n == 1
    assert len(ccr_backend.images) == 1
    assert not getattr(ccr_backend.images[0], "is_merged", False)


def test_without_force_no_merge_a_tiff_is_rejected_in_merge_mode(tmp_path):
    p = str(tmp_path / "lin.tiff")
    tifffile.imwrite(p, MERGED)
    ccr_backend.rgb_merge_mode = True
    ccr_backend.last_merge_error = None
    n = ccr_backend.load_images_from_files([p])   # merge on, TIFF is not RAW
    assert n == 0
    assert ccr_backend.last_merge_error          # a rejection was recorded


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

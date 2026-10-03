#!/usr/bin/env python3
"""Monochrome rendering for a mono-CONVERTED camera
(spec/mono-positive-render.md).

Two gaps this covers, both found by comparing our decode against the camera's
own embedded JPEG:

* A converted sensor keeps a fixed per-phase sensitivity difference, which the
  monochrome read prints as a 2x2 checkerboard at full resolution.
* The monochrome read returns SCENE-LINEAR data and ignored Positive mode, so
  the display-referred base render curve landed on linear values and DARKENED
  them instead of lifting them.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import ccr_merge  # noqa: E402
from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402


@pytest.fixture(autouse=True)
def _clean():
    saved = (ccr_backend.positive_mode, getattr(ccr_backend, "mono_raw", False))
    yield
    ccr_backend.positive_mode, ccr_backend.mono_raw = saved


# --- per-phase normalisation (pure) ----------------------------------------- #

def _patterned(h=8, w=8, gains=(0.92, 1.06, 1.07, 0.96)):
    """A flat field carrying a fixed 2x2 gain pattern, like a converted sensor."""
    base = np.full((h, w), 1000.0, dtype=np.float32)
    for (dy, dx), g in zip([(0, 0), (0, 1), (1, 0), (1, 1)], gains):
        base[dy::2, dx::2] *= g
    return base


def test_normalisation_flattens_the_fixed_pattern():
    out = ccr_merge.normalize_cfa_phases(_patterned())
    quads = [out[dy::2, dx::2].mean() for dy in (0, 1) for dx in (0, 1)]
    assert max(quads) - min(quads) < 1e-3          # 15.75% -> 0.00%, as measured
    assert out.mean() == pytest.approx(_patterned().mean(), rel=1e-5)


def test_normalisation_preserves_detail_not_just_the_mean():
    """It is a per-phase GAIN, so real structure must survive untouched."""
    src = _patterned()
    src[4, 4] = 5000.0                              # a feature
    out = ccr_merge.normalize_cfa_phases(src)
    assert out[4, 4] > out.mean() * 3               # still the brightest thing


def test_normalisation_is_a_no_op_on_an_already_flat_plane():
    flat = np.full((6, 6), 500.0, dtype=np.float32)
    np.testing.assert_allclose(ccr_merge.normalize_cfa_phases(flat), flat,
                               rtol=1e-6)


def test_normalisation_does_not_mutate_its_input():
    src = _patterned()
    before = src.copy()
    ccr_merge.normalize_cfa_phases(src)
    np.testing.assert_array_equal(src, before)


def test_normalisation_survives_odd_dimensions_and_zero_planes():
    ccr_merge.normalize_cfa_phases(_patterned(7, 9))          # must not raise
    zeros = np.zeros((4, 4), dtype=np.float32)
    np.testing.assert_array_equal(ccr_merge.normalize_cfa_phases(zeros), zeros)
    with pytest.raises(ValueError):
        ccr_merge.normalize_cfa_phases(np.zeros((4, 4, 3), np.float32))


# --- the read applies it BEFORE binning ------------------------------------- #

class _FakeRaw:
    def __init__(self, plane):
        self.raw_image_visible = plane.astype(np.uint16)
        self.raw_colors_visible = np.tile(np.array([[0, 1], [3, 2]]),
                                          (plane.shape[0] // 2,
                                           plane.shape[1] // 2))
        self.black_level_per_channel = [0, 0, 0, 0]


def test_mono_read_normalises_phases_before_binning():
    """bin2x2 averages the four phases, so a preview would hide the pattern
    while full-res zoom/export still showed it. Order matters."""
    raw = _FakeRaw(_patterned(16, 16) * 10)
    plane, full = CCRImage._read_mono_mosaic(raw, preview=False)
    quads = [plane[dy::2, dx::2].mean() for dy in (0, 1) for dx in (0, 1)]
    assert max(quads) - min(quads) < 1e-2
    assert full == (16, 16)
    # The binned preview reports the same canonical full size.
    binned, full_b = CCRImage._read_mono_mosaic(raw, preview=True)
    assert binned.shape == (8, 8) and full_b == (16, 16)


# --- monochrome + Positive mode is display-referred -------------------------- #

def _srgb_encode(x):
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)


def test_mono_positive_is_encoded_not_linear(monkeypatch):
    """The whole point: in Positive mode a monochrome frame must come out
    display-referred, or the base render curve darkens it instead of lifting."""
    import core.ccr_image as mod

    lin_level = 0.08                                # a dark-ish linear midtone
    plane = np.full((8, 8), lin_level * 16383.0, dtype=np.float32)

    class _Raw(_FakeRaw):
        white_level = 16383
        camera_whitebalance = [1.0, 1.0, 1.0, 1.0]
        sizes = type("S", (), {"height": 8, "width": 8})()
        num_colors = 3
        color_desc = b"RGBG"
        raw_pattern = np.array([[0, 1], [3, 2]])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod.rawpy, "imread", lambda *a, **k: _Raw(plane))
    ccr_backend.mono_raw = True

    def decode(positive):
        ccr_backend.positive_mode = positive
        img = CCRImage.__new__(CCRImage)
        img.source_ops = []
        img.decoded_mono = False
        img.input_transfer = None
        img.last_read_error = None
        out = img.read_image("x.nef", preview=False)
        return float(np.median(out)) / 65535.0, img

    neg, img_neg = decode(False)
    pos, img_pos = decode(True)

    assert img_neg.decoded_mono and img_pos.decoded_mono
    # Negative mode keeps the scene-linear value the density math needs.
    assert neg == pytest.approx(lin_level, abs=0.01)
    # Positive mode encodes it -> markedly brighter, and matches sRGB exactly.
    assert pos == pytest.approx(float(_srgb_encode(np.array([lin_level]))[0]),
                                abs=0.01)
    assert pos > neg * 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

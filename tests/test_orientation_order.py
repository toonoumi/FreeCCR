#!/usr/bin/env python3
"""Orientation order: the export bakes rotation/mirrors the way the canvas
composes them (spec/orientation-sync-group.md § Canonical order).

The canvas applies the 90-degree rotation, then the fine rotation, then the
mirrors LAST in screen space. The export used to flip FIRST, which lands a
mirrored quarter turn 180 degrees away (and leans a mirrored fine rotation the
wrong way) — a mirrored frame exported rotated wrongly while the preview showed
it right. These tests pin the export to the canvas for every combination.
"""
import os
import sys

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtGui import QTransform  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_processor import apply_orientation  # noqa: E402

H, W = 40, 60
MARK = (5, 10)          # row, col — off-centre in both axes, so every
                        # rotation/mirror combination lands somewhere distinct
ROTATIONS = (0, 90, 180, 270)
FLIPS = ((False, False), (True, False), (False, True), (True, True))


def _marker_image():
    img = np.zeros((H, W), np.uint8)
    img[MARK] = 255
    return img


def _canvas_position(rot, h_flip, v_flip, fine_deg=0.0):
    """Where the canvas puts the marker, as a (row, col) fraction of the
    displayed frame. Mirrors the transform image_preview builds: translate to
    centre, scale (mirrors), rotate, translate back, then the fine rotation —
    Qt applies the LAST call first, so the point sees fine, rotation, mirrors."""
    cx, cy = W / 2.0, H / 2.0
    t = QTransform()
    t.translate(cx, cy)
    if v_flip:
        t.scale(1, -1)
    if h_flip:
        t.scale(-1, 1)
    if rot:
        t.rotate(rot)
    t.translate(-cx, -cy)
    if fine_deg:
        t.translate(cx, cy)
        t.rotate(fine_deg)
        t.translate(-cx, -cy)
    corners = [t.map(QPointF(x, y)) for x, y in ((0, 0), (W, 0), (0, H), (W, H))]
    xs = [p.x() for p in corners]
    ys = [p.y() for p in corners]
    p = t.map(QPointF(MARK[1] + 0.5, MARK[0] + 0.5))
    return ((p.y() - min(ys)) / (max(ys) - min(ys)),
            (p.x() - min(xs)) / (max(xs) - min(xs)))


def _export_position(rot, h_flip, v_flip, fine_deg=0.0):
    """Where apply_orientation puts the marker, same fractional coordinates."""
    out = apply_orientation(_marker_image(), rot, fine_deg, h_flip, v_flip)
    r, c = np.unravel_index(np.argmax(out), out.shape)
    return (r / out.shape[0], c / out.shape[1])


@pytest.mark.parametrize("rot", ROTATIONS)
@pytest.mark.parametrize("h_flip,v_flip", FLIPS)
def test_export_matches_canvas_for_every_orientation(rot, h_flip, v_flip):
    er, ec = _export_position(rot, h_flip, v_flip)
    cr, cc = _canvas_position(rot, h_flip, v_flip)
    assert abs(er - cr) < 0.04 and abs(ec - cc) < 0.04, (
        f"rot={rot} h={h_flip} v={v_flip}: export {(er, ec)} vs canvas {(cr, cc)}")


@pytest.mark.parametrize("rot", ROTATIONS)
@pytest.mark.parametrize("h_flip,v_flip", FLIPS)
def test_export_matches_canvas_with_fine_rotation(rot, h_flip, v_flip):
    """A mirror reverses which way a fine rotation leans, so the mirrors must
    come after it here too — this failed at EVERY rotation, 0 included."""
    er, ec = _export_position(rot, h_flip, v_flip, 10.0)
    cr, cc = _canvas_position(rot, h_flip, v_flip, 10.0)
    assert abs(er - cr) < 0.04 and abs(ec - cc) < 0.04, (
        f"rot={rot} h={h_flip} v={v_flip} fine: export {(er, ec)} vs canvas {(cr, cc)}")


def test_mirrored_quarter_turn_is_not_the_flip_first_result():
    """The specific regression: mirroring BEFORE a quarter turn lands 180
    degrees from mirroring after it. Guards against a silent revert."""
    good = apply_orientation(_marker_image(), 90, 0.0, True, False)
    flip_first = np.rot90(cv2.flip(_marker_image(), 1), k=3)
    assert good.shape == flip_first.shape
    assert not np.array_equal(good, flip_first)
    # ...and "flip first" is exactly the 180-degree-wrong version.
    assert np.array_equal(good, np.rot90(flip_first, k=2))


def test_no_mirror_is_unchanged_by_the_reorder():
    """Without a mirror the order is irrelevant — these exports must be
    bit-identical to the old behaviour, so no existing file changes."""
    for rot in ROTATIONS:
        old = _marker_image()
        if rot == 90:
            old = np.rot90(old, k=3)
        elif rot == 180:
            old = np.rot90(old, k=2)
        elif rot == 270:
            old = np.rot90(old, k=1)
        assert np.array_equal(apply_orientation(_marker_image(), rot), old)


def test_both_mirrors_are_unchanged_by_the_reorder():
    """Both mirrors together are a 180-degree turn, which commutes with the
    quarter turns — also bit-identical to the old order."""
    for rot in ROTATIONS:
        old = cv2.flip(_marker_image(), -1)
        if rot == 90:
            old = np.rot90(old, k=3)
        elif rot == 180:
            old = np.rot90(old, k=2)
        elif rot == 270:
            old = np.rot90(old, k=1)
        assert np.array_equal(
            apply_orientation(_marker_image(), rot, 0.0, True, True), old)


def test_fine_rotation_expands_the_canvas():
    out = apply_orientation(_marker_image(), 0, 10.0)
    assert out.shape[0] > H and out.shape[1] > W


# --- end to end through a real export ----------------------------------------

def test_bwpoint_export_uses_the_canvas_order(tmp_path):
    """The pipeline, not just the helper: a mirrored + rotated export must land
    where the canvas says."""
    import tifffile
    from core.ccr_image import CCRImage
    from core.ccr_processor import ccr_normalize_with_bwpoint

    scan = np.full((H * 10, W * 10, 3), 30000, np.uint16)
    # A dark block (bright after inversion) in one corner region.
    scan[50:150, 100:200] = 12000
    src = str(tmp_path / "scan.tiff")
    tifffile.imwrite(src, scan)

    img = CCRImage(src)
    img.rotation_angle = 90
    img.horizontal_mirrored = True
    out_path = str(tmp_path / "out.tiff")
    ccr_normalize_with_bwpoint(img, [30000, 30000, 30000], None,
                               output_path=out_path)
    written = cv2.imread(out_path, cv2.IMREAD_UNCHANGED)
    assert written is not None
    assert written.shape[:2] == (W * 10, H * 10)        # quarter turn

    # Centroid of the bright region — the block is uniform, so argmax would
    # return its corner while the expectation below is its centre.
    lum = written.max(axis=2).astype(np.float32)
    ys, xs = np.nonzero(lum >= lum.max() * 0.9)
    r, c = ys.mean(), xs.mean()
    # The same block, traced through the canvas order.
    marker_rel = ((50 + 150) / 2 / (H * 10), (100 + 200) / 2 / (W * 10))
    cx, cy = 0.5, 0.5
    t = QTransform()
    t.translate(cx, cy)
    t.scale(-1, 1)
    t.rotate(90)
    t.translate(-cx, -cy)
    p = t.map(QPointF(marker_rel[1], marker_rel[0]))
    assert abs(r / lum.shape[0] - p.y()) < 0.08
    assert abs(c / lum.shape[1] - p.x()) < 0.08

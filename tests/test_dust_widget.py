#!/usr/bin/env python3
"""
Widget-level smoke test for the dust-healing canvas workflow.

Drives the real ImagePreview/MainWindow (offscreen) through the full loop:
enter heal mode -> auto-detect -> manual heal -> click-to-remove -> clear,
asserting the backend strokes and the scene overlay update as expected. This
guards the signal wiring and coordinate/canvas code the pure-numpy tests can't.
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def converted_image(tmp_path):
    """Append a synthetic converted positive (one bright spot) to the backend;
    yield (main_window, idx); restore the singleton afterward."""
    from core.ccr_backend import ccr_backend
    from core.ccr_image import CCRImage
    from ui.main_window import MainWindow

    path = str(tmp_path / "scan.png")
    cv2.imwrite(path, cv2.cvtColor(np.full((200, 300, 3), 40000, np.uint16),
                                   cv2.COLOR_RGB2BGR))
    img = CCRImage(path)
    pos = (np.full((200, 300, 3), 40000.0)
           + np.random.default_rng(0).normal(0, 300, (200, 300, 3)))
    pos = np.clip(pos, 0, 65535).astype(np.uint16)
    yy, xx = np.ogrid[:200, :300]
    pos[(yy - 100) ** 2 + (xx - 150) ** 2 <= 9] = 62000   # spot at (150, 100)
    img.resized_raw = pos
    img.converted = True
    img.conversion_inputs = {"mode": "bw",
                             "bw": ((1000, 1000, 1000), (60000, 60000, 60000)),
                             "fine_rot": 0}
    img.update_thumbnail_and_preview()

    n0 = len(ccr_backend.images)
    mw = MainWindow()
    ccr_backend.images.append(img)
    idx = n0
    mw.image_preview.update_preview(idx)
    try:
        yield mw, idx
    finally:
        del ccr_backend.images[n0:]


def test_full_heal_workflow(converted_image):
    from core.ccr_backend import ccr_backend
    mw, idx = converted_image
    ip = mw.image_preview
    assert ip.current_converted

    # enter heal mode
    ip.heal_action.setChecked(True)
    assert ip.heal_mode

    # auto-detect finds the spot and draws an overlay marker
    ip.auto_detect_dust()
    assert len(ccr_backend.get_dust_strokes_by_index(idx)) >= 1
    assert len(ip._dust_overlay_items) >= 1

    # manual heal at an empty location (no crop/rotation -> scene==full coords)
    n = len(ccr_backend.get_dust_strokes_by_index(idx))
    ip.heal_press(QPointF(30, 30))
    ip.heal_release(QPointF(30, 30))
    assert len(ccr_backend.get_dust_strokes_by_index(idx)) == n + 1

    # click the detected spot to remove that detection
    n = len(ccr_backend.get_dust_strokes_by_index(idx))
    ip.heal_press(QPointF(150, 100))
    assert len(ccr_backend.get_dust_strokes_by_index(idx)) == n - 1

    # visualize toggles without error; clear removes everything
    ip.toggle_visualize(True)
    ip.update_preview(idx)
    ip.clear_dust()
    assert ccr_backend.get_dust_strokes_by_index(idx) == []


def test_autodetect_preserves_manual_strokes(converted_image):
    """Re-running Auto-Detect (even at a new sensitivity) keeps manual strokes
    and replaces — not accumulates — the auto ones (spec §3.6 / §9.5)."""
    from core.ccr_backend import ccr_backend
    mw, idx = converted_image
    ip = mw.image_preview
    ip.heal_action.setChecked(True)

    ip.heal_press(QPointF(30, 30))      # a manual heal far from the spot
    ip.heal_release(QPointF(30, 30))
    manual_before = [s for s in ccr_backend.get_dust_strokes_by_index(idx)
                     if s["source"] == "manual"]
    assert len(manual_before) == 1

    ip.sensitivity_combo.setCurrentIndex(2)   # High
    ip.auto_detect_dust()
    ip.sensitivity_combo.setCurrentIndex(0)   # Low
    ip.auto_detect_dust()

    strokes = ccr_backend.get_dust_strokes_by_index(idx)
    manual_after = [s for s in strokes if s["source"] == "manual"]
    autos = [s for s in strokes if s["source"] == "auto"]
    assert len(manual_after) == 1                       # preserved
    assert manual_after[0]["points"] == manual_before[0]["points"]
    assert len(autos) <= 3                              # replaced, not accumulated


def test_clicks_outside_image_are_noops(converted_image):
    """Clicking far outside the image bounds must not create/delete strokes."""
    from core.ccr_backend import ccr_backend
    mw, idx = converted_image
    ip = mw.image_preview
    ip.heal_action.setChecked(True)
    assert ip._heal_scene_to_full(QPointF(-500, -500)) is None
    n = len(ccr_backend.get_dust_strokes_by_index(idx))
    ip.heal_press(QPointF(-500, -500))
    ip.heal_release(QPointF(-500, -500))
    assert len(ccr_backend.get_dust_strokes_by_index(idx)) == n


def test_heal_mode_blocked_until_converted(converted_image):
    from core.ccr_backend import ccr_backend
    mw, idx = converted_image
    ip = mw.image_preview
    ccr_backend.images[idx].converted = False
    ip.update_preview(idx)            # refresh -> controls disable
    assert not ip.heal_action.isEnabled()
    ip.heal_action.setChecked(True)   # guarded: cannot enter heal mode
    assert not ip.heal_mode


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

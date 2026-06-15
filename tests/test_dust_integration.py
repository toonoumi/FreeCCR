#!/usr/bin/env python3
"""
Integration tests for dust healing wiring into CCRImage / catalog (no canvas).

Verifies the single apply_adjustments hook heals, and that strokes round-trip
through serialize + the undo snapshot. CCRImage pulls QtGui, so a headless
QApplication is created, but no widgets are shown.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import cv2  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from core import catalog  # noqa: E402
from core.dust_removal import clean_strokes  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])


def _load_blank(tmp_path, h=200, w=300):
    """Construct a CCRImage from a flat synthetic file, then return it."""
    flat = np.full((h, w, 3), 40000, np.uint16)
    path = str(tmp_path / "scan.png")
    cv2.imwrite(path, cv2.cvtColor(flat, cv2.COLOR_RGB2BGR))
    return CCRImage(path)


def _positive_with_spot(h=200, w=300, cy=100, cx=150, r=4, seed=1):
    rng = np.random.default_rng(seed)
    base = np.full((h, w, 3), 40000.0) + rng.normal(0, 300, (h, w, 3))
    img = np.clip(base, 0, 65535).astype(np.uint16)
    yy, xx = np.ogrid[:h, :w]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
    img[m] = 62000
    return img, m


def _spot_stroke(cy, cx, r, h, w, source="manual"):
    return {"points": [[cx / w, cy / h]], "radius": (r + 2) / max(h, w),
            "connect": False, "kind": "spot", "source": source}


def test_apply_adjustments_heals_with_no_other_edits(tmp_path):
    ci = _load_blank(tmp_path)
    ci.converted = True
    ci.brightness_base = 0          # isolate healing from the default look offset
    img16, m = _positive_with_spot()
    ci.resized_raw = img16
    ci.dust_heal_strokes = [_spot_stroke(100, 150, 4, 200, 300)]
    out = ci.apply_adjustments(ci.resized_raw)
    assert out[m].mean() < 50000                       # spot pulled to background
    np.testing.assert_array_equal(out[:20, :20], img16[:20, :20])   # far corner exact


def test_no_strokes_is_passthrough_identity(tmp_path):
    ci = _load_blank(tmp_path)
    ci.converted = True
    ci.brightness_base = 0
    img16, _ = _positive_with_spot()
    ci.resized_raw = img16
    out = ci.apply_adjustments(ci.resized_raw)
    assert out is ci.resized_raw                        # early-return fast path


def test_serialize_includes_strokes_and_loader_roundtrips(tmp_path):
    ci = _load_blank(tmp_path)
    ci.dust_heal_strokes = [_spot_stroke(100, 150, 4, 200, 300, source="auto"),
                            _spot_stroke(40, 60, 3, 200, 300, source="manual")]
    state = catalog.serialize_image(ci)
    assert len(state["dust_heal_strokes"]) == 2
    assert not catalog._is_pristine(state)
    loaded = clean_strokes(state["dust_heal_strokes"])
    assert [s["source"] for s in loaded] == ["auto", "manual"]


def test_undo_roundtrip_strokes(tmp_path):
    ci = _load_blank(tmp_path)
    ci.push_undo_state()                       # snapshot: no strokes
    ci.dust_heal_strokes = [_spot_stroke(100, 150, 4, 200, 300)]
    ci.push_undo_state()                       # snapshot: one stroke
    ci.dust_heal_strokes = []
    ci.pop_undo_state()                        # restore one-stroke snapshot
    assert len(ci.dust_heal_strokes) == 1
    ci.pop_undo_state()                        # restore empty snapshot
    assert ci.dust_heal_strokes == []


def test_pristine_until_strokes_added(tmp_path):
    ci = _load_blank(tmp_path)
    assert catalog._is_pristine(catalog.serialize_image(ci))
    ci.dust_heal_strokes = [_spot_stroke(100, 150, 4, 200, 300)]
    assert not catalog._is_pristine(catalog.serialize_image(ci))


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))

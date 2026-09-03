"""No-anchor conversion: the panel gating and the suppressible warning
(spec/no-anchor-convert.md §6).

Converting with no black point is ALLOWED but confirmed first, and that
confirmation is switchable off in Settings.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from widgets.sliders_panel import SlidersPanel  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return SlidersPanel()


@pytest.fixture(autouse=True)
def _clean_backend():
    saved = (ccr_backend.black_point_bgr, ccr_backend.white_point_bgr,
             getattr(ccr_backend, "warn_no_anchor_convert", True))
    yield
    (ccr_backend.black_point_bgr, ccr_backend.white_point_bgr,
     ccr_backend.warn_no_anchor_convert) = saved


class _FakeBox:
    """Stands in for the QMessageBox the confirm helper builds, recording
    whether it was shown and answering with a chosen button."""
    shown = 0
    answer_proceed = False

    def __init__(self, *a, **kw):
        self._proceed = object()
        self._cancel = object()

    def setIcon(self, *a): pass
    def setWindowTitle(self, *a): pass
    def setText(self, *a): pass
    def setInformativeText(self, *a): pass
    def setDefaultButton(self, *a): pass

    def addButton(self, *a):
        # (label, role) → the proceed button; (StandardButton,) → cancel
        return self._proceed if len(a) == 2 else self._cancel

    def exec_(self):
        type(self).shown += 1

    def clickedButton(self):
        return self._proceed if type(self).answer_proceed else self._cancel


def _install_fake_box(monkeypatch):
    import widgets.sliders_panel as sp
    _FakeBox.shown = 0
    monkeypatch.setattr(sp, "QMessageBox", type(
        "QMB", (), {**{k: getattr(QMessageBox, k)
                       for k in ("Warning", "Cancel", "AcceptRole")},
                    "__new__": lambda cls, *a, **kw: _FakeBox()}))
    return _FakeBox


# --- the warning ------------------------------------------------------------

def test_warns_and_cancels_without_a_black_point(panel, monkeypatch):
    box = _install_fake_box(monkeypatch)
    box.answer_proceed = False
    ccr_backend.black_point_bgr = None
    ccr_backend.warn_no_anchor_convert = True

    assert panel._confirm_no_anchor_convert() is False
    assert box.shown == 1


def test_warns_and_proceeds_when_confirmed(panel, monkeypatch):
    box = _install_fake_box(monkeypatch)
    box.answer_proceed = True
    ccr_backend.black_point_bgr = None
    ccr_backend.warn_no_anchor_convert = True

    assert panel._confirm_no_anchor_convert() is True
    assert box.shown == 1


def test_no_warning_when_the_setting_is_off(panel, monkeypatch):
    box = _install_fake_box(monkeypatch)
    ccr_backend.black_point_bgr = None
    ccr_backend.warn_no_anchor_convert = False

    assert panel._confirm_no_anchor_convert() is True
    assert box.shown == 0, "the setting must suppress the dialog entirely"


def test_no_warning_when_a_black_point_is_set(panel, monkeypatch):
    box = _install_fake_box(monkeypatch)
    ccr_backend.black_point_bgr = (30000.0, 30000.0, 30000.0)
    ccr_backend.warn_no_anchor_convert = True

    assert panel._confirm_no_anchor_convert() is True
    assert box.shown == 0


# --- the mode label ---------------------------------------------------------

def test_label_names_the_no_anchor_state(panel):
    ccr_backend.black_point_bgr = None
    ccr_backend.white_point_bgr = None
    panel._update_bwp_mode_label()
    text = panel.bwp_mode_label.text()
    assert "none" in text and "namicolor density invert" in text.lower()
    assert panel.bwp_mode_label.isVisibleTo(panel)


def test_label_still_names_the_anchored_states(panel):
    ccr_backend.white_point_bgr = None
    ccr_backend.black_point_bgr = (30000.0,) * 3
    panel._update_bwp_mode_label()
    assert "black point only" in panel.bwp_mode_label.text()

    ccr_backend.white_point_bgr = (3000.0,) * 3
    panel._update_bwp_mode_label()
    assert "two-point" in panel.bwp_mode_label.text()


# --- the convert buttons are no longer gated --------------------------------

def test_convert_buttons_are_not_disabled_without_a_black_point(panel):
    ccr_backend.black_point_bgr = None
    panel.set_negative_controls_enabled(True)
    assert panel.convert_current_bwp_btn.isEnabled()
    assert panel.convert_all_bwp_btn.isEnabled()

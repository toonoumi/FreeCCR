#!/usr/bin/env python3
"""Tests for LongPressButton: the hold gesture fires once, swallows the click
it consumed, and never interferes with a normal short press."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtCore import QEvent, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from widgets.long_press_button import LongPressButton  # noqa: E402


def _mouse(kind, button, pos=QPointF(5, 5)):
    return QMouseEvent(kind, pos, pos, button, button, Qt.NoModifier)


class _Recorder:
    def __init__(self, button):
        self.clicks = 0
        self.holds = 0
        button.clicked.connect(self._click)
        button.longPressed.connect(self._hold)

    def _click(self):
        self.clicks += 1

    def _hold(self):
        self.holds += 1


@pytest.fixture
def button():
    btn = LongPressButton("Slice")
    btn.resize(80, 24)
    btn.show()
    yield btn
    btn.hide()
    btn.deleteLater()


def _press(btn, button=Qt.LeftButton):
    btn.mousePressEvent(_mouse(QEvent.MouseButtonPress, button))


def _release(btn, button=Qt.LeftButton):
    btn.mouseReleaseEvent(_mouse(QEvent.MouseButtonRelease, button))


def test_short_press_clicks_without_holding(button):
    rec = _Recorder(button)
    _press(button)
    _release(button)
    assert (rec.clicks, rec.holds) == (1, 0)


def test_hold_emits_long_press_and_swallows_the_click(button):
    rec = _Recorder(button)
    _press(button)
    # Fire the hold timer directly instead of sleeping for HOLD_MS
    button._hold_timer.timeout.emit()
    assert (rec.clicks, rec.holds) == (0, 1)
    _release(button)
    assert (rec.clicks, rec.holds) == (0, 1)
    assert not button.isDown()


def test_hold_does_not_arm_the_next_press(button):
    rec = _Recorder(button)
    _press(button)
    button._hold_timer.timeout.emit()
    _release(button)
    # A normal click afterwards must behave normally again
    _press(button)
    _release(button)
    assert (rec.clicks, rec.holds) == (1, 1)


def test_right_click_opens_the_menu_without_a_hold(button):
    rec = _Recorder(button)
    _press(button, Qt.RightButton)
    _release(button, Qt.RightButton)
    assert (rec.clicks, rec.holds) == (0, 1)


def test_leaving_the_button_cancels_the_hold(button):
    rec = _Recorder(button)
    _press(button)
    button.leaveEvent(QEvent(QEvent.Leave))
    assert not button._hold_timer.isActive()
    _release(button)
    assert (rec.clicks, rec.holds) == (1, 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

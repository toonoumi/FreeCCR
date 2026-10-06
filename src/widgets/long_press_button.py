"""A push button with a secondary "hold" gesture.

Used where one button owns a primary action plus a rarely-needed variant that
does not deserve a second button in a crowded row (Slice: this image vs. every
image). Press and hold for HOLD_MS, or right-click, and `longPressed` fires;
the press that triggered it is swallowed, so the primary `clicked` never also
runs. Lifting before HOLD_MS behaves exactly like a normal QPushButton.
"""

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QPushButton


class LongPressButton(QPushButton):
    longPressed = Signal()

    HOLD_MS = 450

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hold_timer = QTimer(self)
        self._hold_timer.setSingleShot(True)
        self._hold_timer.setInterval(self.HOLD_MS)
        self._hold_timer.timeout.connect(self._fire_long_press)
        # Set while the hold has already fired, so the release that follows
        # does not also emit clicked().
        self._long_press_fired = False

    def _fire_long_press(self):
        self._long_press_fired = True
        # Drop the pressed-in look: the menu is about to take the focus, and a
        # button left visually held down reads as a stuck control.
        self.setDown(False)
        self.longPressed.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            # Right-click is the keyboard-free shortcut to the same menu —
            # no hold, and never a click.
            self._long_press_fired = True
            self.longPressed.emit()
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            self._long_press_fired = False
            self._hold_timer.start()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._hold_timer.stop()
        if self._long_press_fired:
            # Swallow the release so QPushButton never emits clicked() for the
            # press the hold already consumed.
            self._long_press_fired = False
            self.setDown(False)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        # Dragging off the button cancels the hold, matching how a click is
        # cancelled by releasing outside.
        self._hold_timer.stop()
        super().leaveEvent(event)

"""Photoshop-style tone curve editor widget.

Lets the user reshape image tonality by dragging control points on a curve, per
composite channel ("All"/rgb) and per individual channel (R/G/B). The curve
state is exposed as a plain dict of channel -> list of [x, y] control points in
the 0..255 domain, matching what ccr_processor.apply_curves consumes.

See spec/curves-tone-control.md for the interaction contract.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPainterPath

from ui import theme

# Mirror ccr_processor's channel set / identity so the two agree by construction.
CURVE_CHANNELS = ("rgb", "r", "g", "b")


def _identity_points():
    return [[0.0, 0.0], [255.0, 255.0]]


def identity_curves():
    return {ch: _identity_points() for ch in CURVE_CHANNELS}


def _monotone_cubic(xs, ys, xq):
    """Local copy of the Fritsch-Carlson interpolation used for drawing so the
    rendered line matches ccr_processor.apply_curves. Pure-Python, small N."""
    n = len(xs)
    # Endpoints can be dragged inward, so clamp the query into the endpoint range:
    # outside it the curve holds its endpoint value. Without this the n==2 branch
    # below would extrapolate a straight line (np.interp in the processor's copy
    # holds instead), and the n>=3 branch would extrapolate the polynomial — the
    # drawn line would disagree with the rendered pixels. See build_channel_lut.
    lo, hi = xs[0], xs[-1]
    xq = [min(hi, max(lo, q)) for q in xq]
    if n == 2:
        # Linear
        x0, x1 = xs[0], xs[1]
        y0, y1 = ys[0], ys[1]
        span = (x1 - x0) or 1.0
        return [y0 + (y1 - y0) * (q - x0) / span for q in xq]
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    delta = [(ys[i + 1] - ys[i]) / (h[i] or 1.0) for i in range(n - 1)]
    m = [0.0] * n
    m[0] = delta[0]
    m[-1] = delta[-1]
    for i in range(1, n - 1):
        m[i] = (delta[i - 1] + delta[i]) / 2.0
    for i in range(n - 1):
        if delta[i] == 0.0:
            m[i] = 0.0
            m[i + 1] = 0.0
        else:
            a = m[i] / delta[i]
            b = m[i + 1] / delta[i]
            s = a * a + b * b
            if s > 9.0:
                t = 3.0 / (s ** 0.5)
                m[i] = t * a * delta[i]
                m[i + 1] = t * b * delta[i]
    out = []
    seg = 0
    for q in xq:
        while seg < n - 2 and q > xs[seg + 1]:
            seg += 1
        x0, x1 = xs[seg], xs[seg + 1]
        y0, y1 = ys[seg], ys[seg + 1]
        hh = (x1 - x0) or 1.0
        t = (q - x0) / hh
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        out.append(h00 * y0 + h10 * hh * m[seg] + h01 * y1 + h11 * hh * m[seg + 1])
    return out


class CurveCanvas(QWidget):
    """The interactive drawing surface for one set of per-channel curves."""

    POINT_HIT = 11          # px square hit area ("button size") around a point
    POINT_RADIUS = 4
    CURVE_HIT = 8           # px vertical tolerance for "clicked on the line"

    curveChanged = Signal()      # live, during a drag / on add / on remove
    editFinished = Signal()      # drag released / edit settled

    _LINE_COLORS = {
        "rgb": QColor("#e8e8e8"),
        "r": QColor(theme.CH_R),
        "g": QColor(theme.CH_G),
        "b": QColor(theme.CH_B),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setFixedHeight(220)
        self.setMouseTracking(True)
        self._channel = "rgb"
        self._points = identity_curves()
        self._drag_index = None
        self._grabbed = False
        self._enabled = True

    # --- public API -------------------------------------------------------
    def set_active_channel(self, channel: str):
        if channel in CURVE_CHANNELS and channel != self._channel:
            self._channel = channel
            self.update()

    def active_channel(self) -> str:
        return self._channel

    def is_dragging(self) -> bool:
        return self._drag_index is not None

    def get_curves(self):
        """Return a fresh dict of non-identity channels, or None if every
        channel is identity."""
        out = {}
        for ch in CURVE_CHANNELS:
            pts = self._points[ch]
            if pts != _identity_points():
                out[ch] = [[float(x), float(y)] for x, y in pts]
        return out or None

    def set_curves(self, curves):
        """Load curve state (None / missing channels => identity)."""
        new = identity_curves()
        if curves:
            for ch in CURVE_CHANNELS:
                pts = curves.get(ch)
                if pts and len(pts) >= 2:
                    cleaned = []
                    for p in pts:
                        try:
                            x = float(p[0]); y = float(p[1])
                        except (TypeError, ValueError, IndexError):
                            continue
                        cleaned.append([min(255.0, max(0.0, x)),
                                        min(255.0, max(0.0, y))])
                    cleaned.sort(key=lambda q: q[0])
                    if len(cleaned) >= 2:
                        new[ch] = cleaned
        self._points = new
        self._release_grab()
        self.update()

    def _release_grab(self):
        self._drag_index = None
        if self._grabbed:
            self.releaseMouse()
            self._grabbed = False

    def reset(self):
        self._points = identity_curves()
        self._release_grab()
        # repaint() (synchronous) rather than update() (deferred): the
        # curveChanged handler below reprocesses the image preview synchronously
        # and would otherwise block the event loop before the scheduled canvas
        # repaint runs, making the curve appear to reset only after a lag.
        self.repaint()
        self.curveChanged.emit()
        self.editFinished.emit()

    def setEnabled(self, enabled: bool):  # noqa: N802 (Qt override style)
        self._enabled = enabled
        super().setEnabled(enabled)
        self.update()

    # --- geometry helpers -------------------------------------------------
    def _plot_rect(self) -> QRectF:
        # Larger right inset keeps the x=255 edge clear of the panel's vertical
        # scrollbar, which overlays the canvas's rightmost pixels.
        left = top = bottom = 10.0
        right = 22.0
        return QRectF(left, top,
                      self.width() - left - right,
                      self.height() - top - bottom)

    def _curve_y_at(self, x_curve: float) -> float:
        """Output (0..255) of the active channel's curve at input x_curve."""
        pts = self._points[self._channel]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return _monotone_cubic(xs, ys, [x_curve])[0]

    def _to_widget(self, x, y) -> QPointF:
        r = self._plot_rect()
        wx = r.left() + (x / 255.0) * r.width()
        wy = r.bottom() - (y / 255.0) * r.height()
        return QPointF(wx, wy)

    def _to_curve(self, pos) -> tuple:
        r = self._plot_rect()
        x = (pos.x() - r.left()) / r.width() * 255.0
        y = (r.bottom() - pos.y()) / r.height() * 255.0
        return (min(255.0, max(0.0, x)), min(255.0, max(0.0, y)))

    def _point_at(self, pos):
        """Index of the control point whose hit area contains pos, else None."""
        pts = self._points[self._channel]
        half = self.POINT_HIT / 2.0
        for i, (x, y) in enumerate(pts):
            w = self._to_widget(x, y)
            if abs(w.x() - pos.x()) <= half and abs(w.y() - pos.y()) <= half:
                return i
        return None

    # --- mouse ------------------------------------------------------------
    def mousePressEvent(self, event):
        if not self._enabled:
            return
        pts = self._points[self._channel]
        pos = event.position()
        idx = self._point_at(pos)

        if event.button() == Qt.RightButton:
            # Remove an interior point; ignore endpoints / empty canvas.
            if idx is not None and idx != 0 and idx != len(pts) - 1:
                del pts[idx]
                self._drag_index = None
                self.update()
                self.curveChanged.emit()
                self.editFinished.emit()
            event.accept()
            return

        if event.button() != Qt.LeftButton:
            return

        if idx is not None:
            # Grab the existing point (no new point created).
            self._drag_index = idx
        else:
            # Create a point ONLY when the click lands on the curve line.
            xc, yc = self._to_curve(pos)
            curve_y = self._curve_y_at(xc)
            on_line = abs(pos.y() - self._to_widget(xc, curve_y).y()) <= self.CURVE_HIT
            if not on_line:
                event.accept()
                return  # empty space away from the line — do nothing
            # New point sits exactly on the existing curve, strictly between
            # the endpoints, then the drag reshapes it. Index 0 and the last
            # index ARE the endpoints, so a point inserted outside their x range
            # would silently demote an endpoint into a removable interior point;
            # clicks on the flat clipped segments therefore add nothing. With
            # endpoints at 0/255 these bounds are the former 1..254.
            lo = pts[0][0] + 1.0
            hi = pts[-1][0] - 1.0
            if not (lo <= xc <= hi):
                event.accept()
                return
            x = xc
            insert_at = 0
            while insert_at < len(pts) and pts[insert_at][0] < x:
                insert_at += 1
            pts.insert(insert_at, [x, curve_y])
            self._drag_index = insert_at
        # The canvas sits inside the panel's QScrollArea; grab the mouse so move
        # events keep coming to us for the whole drag instead of being stolen.
        self.grabMouse()
        self._grabbed = True
        self.update()
        self.curveChanged.emit()
        event.accept()

    def mouseMoveEvent(self, event):
        if not self._enabled or self._drag_index is None:
            return
        pts = self._points[self._channel]
        i = self._drag_index
        x, y = self._to_curve(event.position())
        last = len(pts) - 1
        # Every point — endpoints included — moves in both axes. Dragging an
        # endpoint horizontally sets the black/white input point (tones beyond it
        # clip to its output level). The clamp is therefore uniform: stay strictly
        # between the neighbours, with the canvas edge standing in for the outer
        # neighbour an endpoint doesn't have.
        lo = pts[i - 1][0] + 1.0 if i > 0 else 0.0
        hi = pts[i + 1][0] - 1.0 if i < last else 255.0
        x = min(hi, max(lo, x))
        pts[i] = [x, min(255.0, max(0.0, y))]
        self.update()
        self.curveChanged.emit()
        event.accept()

    def mouseReleaseEvent(self, event):
        # Always drop the grab if we hold it, even if _drag_index was cleared
        # mid-drag (e.g. by a set_curves refresh), so the mouse can't get stuck.
        if self._grabbed:
            self.releaseMouse()
            self._grabbed = False
        if self._drag_index is not None:
            self._drag_index = None
            self.editFinished.emit()
            event.accept()

    # --- painting ---------------------------------------------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = self._plot_rect()

        # Background
        p.fillRect(self.rect(), QColor(theme.Paint.CURVE_BG))
        p.setPen(QPen(QColor(theme.BORDER_STRONG), 1))
        p.drawRect(r)

        # Grid (4x4)
        p.setPen(QPen(QColor(theme.Paint.CURVE_GRID), 1))
        for k in range(1, 4):
            gx = r.left() + r.width() * k / 4.0
            gy = r.top() + r.height() * k / 4.0
            p.drawLine(QPointF(gx, r.top()), QPointF(gx, r.bottom()))
            p.drawLine(QPointF(r.left(), gy), QPointF(r.right(), gy))

        # Identity diagonal reference
        p.setPen(QPen(QColor(theme.Paint.CURVE_IDENTITY), 1, Qt.DashLine))
        p.drawLine(self._to_widget(0, 0), self._to_widget(255, 255))

        pts = self._points[self._channel]
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]

        # Smooth curve line (sample across the plot width)
        n_samples = max(2, int(r.width()))
        xq = [255.0 * s / (n_samples - 1) for s in range(n_samples)]
        yq = _monotone_cubic(xs, ys, xq)
        path = QPainterPath()
        first = self._to_widget(xq[0], min(255.0, max(0.0, yq[0])))
        path.moveTo(first)
        for qx, qy in zip(xq[1:], yq[1:]):
            path.lineTo(self._to_widget(qx, min(255.0, max(0.0, qy))))
        line_color = self._LINE_COLORS.get(self._channel, QColor("#e8e8e8"))
        if not self._enabled:
            line_color = QColor(theme.Paint.CURVE_NODE_DISABLED)
        p.setPen(QPen(line_color, 2))
        p.drawPath(path)

        # Control points
        p.setPen(QPen(QColor(theme.Paint.CURVE_NODE_OUTLINE), 1))
        p.setBrush(QBrush(QColor(theme.Paint.CURVE_NODE) if self._enabled
                          else QColor(theme.Paint.CURVE_NODE_DISABLED)))
        for x, y in pts:
            w = self._to_widget(x, y)
            p.drawEllipse(w, self.POINT_RADIUS, self.POINT_RADIUS)
        p.end()


class CurveEditor(QWidget):
    """Channel selector + curve canvas + Reset Curve button."""

    curveChanged = Signal()
    editFinished = Signal()

    _CHANNEL_LABELS = (("rgb", "All"), ("r", "R"), ("g", "G"), ("b", "B"))
    _BTN_TINT = {"rgb": theme.TEXT_MUTED, "r": theme.CH_R, "g": theme.CH_G, "b": theme.CH_B}

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.GAP_ROW)

        # Channel selector buttons (mutually exclusive, checkable)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(theme.GAP_BTN)
        self._channel_buttons = {}
        for ch, label in self._CHANNEL_LABELS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(theme.CONTROL_H)
            btn.setStyleSheet(
                f"QPushButton {{ background: {theme.SURFACE}; border: 1px solid {theme.BORDER}; "
                f"border-radius: {theme.RADIUS_SM}px; color: {self._BTN_TINT[ch]}; font-size: 11px; }}"
                f"QPushButton:checked {{ background: {theme.SURFACE_ACTIVE}; border: 2px solid {theme.ACCENT}; }}")
            btn.clicked.connect(lambda _=False, c=ch: self._on_channel(c))
            btn_row.addWidget(btn, 1)
            self._channel_buttons[ch] = btn
        layout.addLayout(btn_row)

        self.canvas = CurveCanvas()
        self.canvas.curveChanged.connect(self.curveChanged)
        self.canvas.editFinished.connect(self.editFinished)
        layout.addWidget(self.canvas)

        self.reset_button = QPushButton("Reset Curve")
        self.reset_button.clicked.connect(self.canvas.reset)
        theme.style_button(self.reset_button, "danger")  # discards the curve
        self.reset_button.setFixedHeight(theme.CONTROL_H)
        layout.addWidget(self.reset_button)

        self._channel_buttons["rgb"].setChecked(True)

    def _on_channel(self, channel: str):
        for ch, btn in self._channel_buttons.items():
            btn.setChecked(ch == channel)
        self.canvas.set_active_channel(channel)

    # Pass-through API used by SlidersPanel.
    def get_curves(self):
        return self.canvas.get_curves()

    def set_curves(self, curves):
        self.canvas.set_curves(curves)

    def is_dragging(self) -> bool:
        return self.canvas.is_dragging()

    def reset(self):
        self.canvas.reset()

    def setEnabled(self, enabled: bool):  # noqa: N802
        self.canvas.setEnabled(enabled)
        for btn in self._channel_buttons.values():
            btn.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        super().setEnabled(enabled)

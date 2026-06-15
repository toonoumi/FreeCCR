from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QToolBar, QMessageBox, QSlider, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsPathItem, QGraphicsEllipseItem,
    QGraphicsLineItem, QStyleOptionSlider, QDialog,
    QLabel, QPushButton, QStyle, QSizePolicy, QApplication, QComboBox
)
from PySide6.QtGui import (QPixmap, QIcon, QTransform, QPen, QColor, QAction, QPainter,
                           QCursor, QDesktopServices, QPainterPath, QBrush, QPolygonF,
                           QImage)
from PySide6.QtCore import Qt, QSize, Signal, QRect, QRectF, QPointF, QThread, QTimer, QUrl
from core.ccr_backend import ccr_backend
from widgets.export_dialog import ExportSettingsDialog
import math
import copy
import sys
import os

_eyedropper_cursor_cache = None

# Dust-healing overlay tints (RGB; QColor built lazily to avoid pre-QApplication
# construction). auto = cyan-blue, manual = green, live drag ghost = amber.
_HEAL_AUTO_RGB = (0, 180, 255)
_HEAL_MANUAL_RGB = (0, 210, 0)
_HEAL_GHOST_RGB = (255, 210, 0)

def _eyedropper_cursor():
    """Small painter-drawn eyedropper cursor, hotspot at the tip (bottom-left)."""
    global _eyedropper_cursor_cache
    if _eyedropper_cursor_cache is None:
        pm = QPixmap(24, 24)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        # White underlay then dark dropper on top, so it reads on any image.
        for color, body_w, bulb_w in ((QColor(255, 255, 255), 5, 7),
                                      (QColor(25, 25, 25), 3, 5)):
            pen = QPen(color, body_w, Qt.SolidLine, Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(4, 20, 13, 11)        # body, tip toward bottom-left
            pen.setWidth(bulb_w)
            p.setPen(pen)
            p.drawLine(11, 7, 17, 13)        # bulb across the top end
        p.end()
        _eyedropper_cursor_cache = QCursor(pm, 3, 21)
    return _eyedropper_cursor_cache

_rotate_cursor_cache = None

def _rotate_cursor():
    """Small painter-drawn curved-arrow cursor for the crop rotate handle."""
    global _rotate_cursor_cache
    if _rotate_cursor_cache is None:
        import math as _math
        pm = QPixmap(28, 28)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        # White underlay then dark arrow on top, so it reads on any image.
        for color, w in ((QColor(255, 255, 255), 4.0), (QColor(25, 25, 25), 2.0)):
            pen = QPen(color, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            # ~270deg arc, leaving a gap at the top-right
            p.drawArc(6, 6, 16, 16, 20 * 16, 280 * 16)
            # Arrowhead at the arc's end (near 20deg, right side)
            cx, cy, r = 14.0, 14.0, 8.0
            ang = _math.radians(20)
            ex, ey = cx + r * _math.cos(ang), cy - r * _math.sin(ang)
            for da in (130, 200):  # two short barbs forming the head
                bx = ex + 5.0 * _math.cos(_math.radians(20 + da))
                by = ey - 5.0 * _math.sin(_math.radians(20 + da))
                p.drawLine(int(ex), int(ey), int(bx), int(by))
        p.end()
        _rotate_cursor_cache = QCursor(pm, 14, 14)
    return _rotate_cursor_cache

def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for Nuitka bundle.
    - In bundle: icons are in Contents/MacOS/icons/
    - In dev: icons are in src/icons/
    """
    # If running as a bundled app (Nuitka, PyInstaller, etc.)
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
        abs_path = os.path.join(base_path, relative_path)
        if os.path.exists(abs_path):
            return abs_path
    # In development: look for src/icons/...
    dev_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "icons"))
    dev_path = os.path.join(dev_base, os.path.basename(relative_path))
    if os.path.exists(dev_path):
        return dev_path
    # Fallback: try relative to current file (for edge cases)
    fallback_path = os.path.join(os.path.dirname(__file__), relative_path)
    if os.path.exists(fallback_path):
        return fallback_path
    # Not found, return as-is (QIcon will fail gracefully)
    return relative_path


class GraphicsImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setMouseTracking(True)
        self.drawing_reference = False
        self._drag_start = None
        self._drag_end = None
        self.reference_rect_item = None
        self.parent_widget = parent

        self.bwpoint_mode = None   # None | "black" | "white"
        self._bw_drag_start = None
        self._bw_drag_end = None
        self._bw_rect_item = None

        self.wb_pick_mode = False  # eyedropper: click a neutral point for auto temp/tint

        self._space_pan = False    # space held -> hand-drag panning
        self._mid_pan = False      # middle button held -> drag panning
        self._mid_pan_last = None  # last viewport pos during a middle-button pan

        # Zoom anchoring is done manually in ImagePreview._apply_zoom (exact
        # cursor/center anchoring); clicking focuses the view so the
        # space-pan key is received.
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setFocusPolicy(Qt.ClickFocus)

        # Disable scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def mousePressEvent(self, event):
        if self._mid_pan:
            # Already panning — swallow any other button so it can't start a
            # reference/crop/B-W interaction underneath the pan.
            event.accept()
            return
        if (event.button() == Qt.MiddleButton
                and self.parent_widget.pixmap_item is not None):
            # Middle-button drag pans the (zoomed) image — works in any mode,
            # but not while another interaction or space-pan is already active.
            interaction_active = (self.drawing_reference or self._space_pan
                                  or self._bw_drag_start is not None
                                  or self.parent_widget.crop_mode
                                  or self.parent_widget._area_drag is not None
                                  or self.parent_widget._slice_drag is not None
                                  or self.parent_widget.heal_mode)
            if not interaction_active:
                self._mid_pan = True
                self._mid_pan_last = event.pos()
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        if self._space_pan:
            # Space+drag pans the view (ScrollHandDrag handles the motion)
            return super().mousePressEvent(event)
        if self.parent_widget.crop_mode and self.parent_widget.pixmap_item is not None:
            if event.button() == Qt.LeftButton:
                self.parent_widget.begin_crop_drag(self.mapToScene(event.pos()))
            elif event.button() == Qt.RightButton:
                self.parent_widget.clear_crop()
            return
        if self.parent_widget.area_mode and self.parent_widget.pixmap_item is not None:
            if event.button() == Qt.LeftButton:
                self.parent_widget.begin_area_drag(self.mapToScene(event.pos()))
            # Swallow both buttons so area editing never starts a reference draw.
            return
        if self.parent_widget.slice_mode and self.parent_widget.pixmap_item is not None:
            if event.button() == Qt.LeftButton:
                self.parent_widget.slice_press(self.mapToScene(event.pos()))
            elif event.button() == Qt.RightButton:
                self.parent_widget.slice_delete_at(self.mapToScene(event.pos()))
            return
        if self.parent_widget.heal_mode and self.parent_widget.pixmap_item is not None:
            if event.button() == Qt.LeftButton:
                self.parent_widget.heal_press(self.mapToScene(event.pos()))
            elif event.button() == Qt.RightButton:
                self.parent_widget.heal_remove_at(self.mapToScene(event.pos()))
            return
        if event.button() == Qt.LeftButton and self.wb_pick_mode and self.parent_widget.pixmap_item is not None:
            scene_pos = self.mapToScene(event.pos())
            self.wb_pick_mode = False
            self.setCursor(Qt.ArrowCursor)
            self._sample_wb_point(scene_pos)
            return
        if event.button() == Qt.LeftButton and self.bwpoint_mode and self.parent_widget.pixmap_item is not None:
            self._bw_drag_start = self.mapToScene(event.pos())
            self._bw_drag_end = self._bw_drag_start
            if self._bw_rect_item:
                self.scene().removeItem(self._bw_rect_item)
                self._bw_rect_item = None
            self.viewport().update()
        elif event.button() == Qt.LeftButton and self.parent_widget.pixmap_item is not None:
            self.drawing_reference = True
            self._drag_start = self.mapToScene(event.pos())
            self._drag_end = self._drag_start
            if self.reference_rect_item:
                self.scene().removeItem(self.reference_rect_item)
                self.reference_rect_item = None
            self.viewport().update()
        elif event.button() == Qt.RightButton:
            # Remove reference frame on right click
            if self.reference_rect_item:
                self.scene().removeItem(self.reference_rect_item)
                self.reference_rect_item = None
            # Also clear in parent widget
            self.parent_widget.reference_rect_item = None
            # Clear in backend
            ccr_backend.set_reference_frame_by_index(self.parent_widget.current_idx, None)
            self.drawing_reference = False
            self.viewport().update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._mid_pan and self._mid_pan_last is not None:
            # Pan by adjusting the (hidden) scrollbars — works even with
            # ScrollBarAlwaysOff, the same mechanism ScrollHandDrag uses.
            delta = event.pos() - self._mid_pan_last
            self._mid_pan_last = event.pos()
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            event.accept()
            return
        if self._space_pan:
            return super().mouseMoveEvent(event)
        if self.parent_widget.crop_mode:
            scene_pos = self.mapToScene(event.pos())
            if self.parent_widget._crop_drag is not None:
                self.parent_widget.update_crop_drag(scene_pos)
            else:
                # Hover: show the transform cursor for the handle underneath
                self.parent_widget.update_crop_hover_cursor(scene_pos)
            return
        if self.parent_widget.area_mode:
            scene_pos = self.mapToScene(event.pos())
            if self.parent_widget._area_drag is not None:
                self.parent_widget.update_area_drag(scene_pos)
            else:
                self.parent_widget.update_area_hover_cursor(scene_pos)
            return
        if self.parent_widget.slice_mode:
            self.parent_widget.slice_move(self.mapToScene(event.pos()))
            return
        if self.parent_widget.heal_mode:
            self.parent_widget.heal_move(self.mapToScene(event.pos()))
            return
        if self.bwpoint_mode and self._bw_drag_start is not None:
            self._bw_drag_end = self.mapToScene(event.pos())
            self._update_bw_rect(self._bw_drag_start, self._bw_drag_end)
        elif self.drawing_reference:
            self._drag_end = self.mapToScene(event.pos())
            self.viewport().update()
            self.parent_widget.update_reference_rect(self._drag_start, self._drag_end)

    def _update_bw_rect(self, start, end):
        """Draw the B/W point selection rect on the canvas."""
        pw = self.parent_widget
        cx = pw.current_pixmap.width() / 2
        cy = pw.current_pixmap.height() / 2
        base_transform = QTransform()
        base_transform.translate(cx, cy)
        if pw.current_vertical_flip:
            base_transform.scale(1, -1)
        if pw.current_horizontal_flip:
            base_transform.scale(-1, 1)
        if pw.current_rotation:
            base_transform.rotate(pw.current_rotation)
        base_transform.translate(-cx, -cy)

        if not base_transform.isInvertible():
            return
        inv = base_transform.inverted()[0]
        s = inv.map(start)
        e = inv.map(end)
        rect = QRectF(min(s.x(), e.x()), min(s.y(), e.y()),
                      abs(e.x() - s.x()), abs(e.y() - s.y()))
        color = QColor(0, 100, 255, 140) if self.bwpoint_mode == "white" else QColor(255, 140, 0, 140)
        if self._bw_rect_item is None:
            self._bw_rect_item = QGraphicsRectItem(rect)
            pen = QPen(color, 2, Qt.DashLine)
            self._bw_rect_item.setPen(pen)
            self.scene().addItem(self._bw_rect_item)
        else:
            self._bw_rect_item.setPen(QPen(color, 2, Qt.DashLine))
            self._bw_rect_item.setRect(rect)
        self._bw_rect_item.setTransform(base_transform)

    def _sample_wb_point(self, scene_pos):
        """Sample a small neighborhood at the clicked point and auto-set the
        temperature/tint sliders so that point becomes neutral."""
        pw = self.parent_widget
        cx = pw.current_pixmap.width() / 2
        cy = pw.current_pixmap.height() / 2
        base_transform = QTransform()
        base_transform.translate(cx, cy)
        if pw.current_vertical_flip:
            base_transform.scale(1, -1)
        if pw.current_horizontal_flip:
            base_transform.scale(-1, 1)
        if pw.current_rotation:
            base_transform.rotate(pw.current_rotation)
        base_transform.translate(-cx, -cy)
        if not base_transform.isInvertible():
            return
        local = base_transform.inverted()[0].map(scene_pos)
        # Displayed pixmap may be a crop of the full image — map back to
        # full-image coordinates before indexing resized_raw.
        fx, fy = pw.map_displayed_to_full(local.x(), local.y())
        x, y = int(fx), int(fy)

        img_obj = ccr_backend.get_image_by_index(pw.current_idx)
        if img_obj is None or img_obj.resized_raw is None:
            return
        data = img_obj.resized_raw
        h, w = data.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        import numpy as np
        rad = 3
        crop = data[max(0, y - rad):min(h, y + rad + 1),
                    max(0, x - rad):min(w, x + rad + 1), :]
        means = np.mean(crop.reshape(-1, 3), axis=0)

        from core.ccr_processor import compute_neutral_temp_tint
        temp, tint = compute_neutral_temp_tint(
            means[0], means[1], means[2],
            getattr(img_obj, 'tint_balance_factor', 1.0))
        try:
            pw.parent().parent().sliders_panel.on_wb_sampled(temp, tint)
        except AttributeError:
            pass

    def mouseReleaseEvent(self, event):
        if self._mid_pan:
            if event.button() == Qt.MiddleButton:
                self._mid_pan = False
                self._mid_pan_last = None
                self._restore_mode_cursor()
            # Swallow any release while panning (other buttons never started
            # an interaction, so nothing to finalize).
            event.accept()
            return
        if self._space_pan:
            return super().mouseReleaseEvent(event)
        if self.parent_widget.crop_mode:
            if event.button() == Qt.LeftButton:
                self.parent_widget.end_crop_drag(self.mapToScene(event.pos()))
            return
        if self.parent_widget.area_mode:
            if event.button() == Qt.LeftButton:
                self.parent_widget.end_area_drag(self.mapToScene(event.pos()))
            return
        if self.parent_widget.slice_mode:
            if event.button() == Qt.LeftButton:
                self.parent_widget.slice_release()
            return
        if self.parent_widget.heal_mode:
            if event.button() == Qt.LeftButton:
                self.parent_widget.heal_release(self.mapToScene(event.pos()))
            return
        if self.bwpoint_mode and event.button() == Qt.LeftButton and self._bw_drag_start is not None:
            drag_start_scene = self._bw_drag_start
            drag_end_scene = self.mapToScene(event.pos())

            pw = self.parent_widget
            cx = pw.current_pixmap.width() / 2
            cy = pw.current_pixmap.height() / 2
            base_transform = QTransform()
            base_transform.translate(cx, cy)
            if pw.current_vertical_flip:
                base_transform.scale(1, -1)
            if pw.current_horizontal_flip:
                base_transform.scale(-1, 1)
            if pw.current_rotation:
                base_transform.rotate(pw.current_rotation)
            base_transform.translate(-cx, -cy)

            mode = self.bwpoint_mode
            self.bwpoint_mode = None
            self._bw_drag_start = None
            self._bw_drag_end = None
            self.setCursor(Qt.ArrowCursor)

            if base_transform.isInvertible():
                inv = base_transform.inverted()[0]
                s = inv.map(drag_start_scene)
                e = inv.map(drag_end_scene)
                # Map from displayed (possibly cropped/rotated) coords to
                # full-image coords. All FOUR rect corners must be mapped —
                # under a rotated crop, the AABB of just the two diagonal
                # corners degenerates (zero width at 45 degrees).
                pts = [pw.map_displayed_to_full(px, py)
                       for px, py in ((s.x(), s.y()), (e.x(), s.y()),
                                      (s.x(), e.y()), (e.x(), e.y()))]
                x1 = int(min(p[0] for p in pts))
                y1 = int(min(p[1] for p in pts))
                x2 = int(max(p[0] for p in pts))
                y2 = int(max(p[1] for p in pts))

                img_obj = ccr_backend.get_image_by_index(pw.current_idx)
                if img_obj is not None:
                    import numpy as np
                    # Always sample from original scan data, not processed data
                    if img_obj.converted:
                        raw_data = img_obj.read_image(img_obj.file_path, max_long_side=1080)
                    else:
                        raw_data = img_obj.resized_raw
                    if raw_data is not None:
                        h, w = raw_data.shape[:2]
                        x1 = max(0, min(x1, w - 1))
                        y1 = max(0, min(y1, h - 1))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        if (x2 - x1) >= 5 and (y2 - y1) >= 5:
                            crop = raw_data[y1:y2, x1:x2, :]
                            means = np.mean(crop.reshape(-1, 3), axis=0)
                            bgr_tuple = (float(means[0]), float(means[1]), float(means[2]))
                            if mode == "black":
                                ccr_backend.set_black_point(bgr_tuple)
                            else:
                                ccr_backend.set_white_point(bgr_tuple)
                            try:
                                pw.parent().parent().sliders_panel.on_bwpoint_sampled(mode)
                            except AttributeError:
                                pass
            return

        if self.drawing_reference and event.button() == Qt.LeftButton:
            self._drag_end = self.mapToScene(event.pos())
            self.drawing_reference = False

            # Build the base (coarse) transform (same as in update_reference_rect)
            cx = self.parent_widget.current_pixmap.width() / 2
            cy = self.parent_widget.current_pixmap.height() / 2
            base_transform = QTransform()
            base_transform.translate(cx, cy)

            if self.parent_widget.current_vertical_flip:
                base_transform.scale(1, -1)
            if self.parent_widget.current_horizontal_flip:
                base_transform.scale(-1, 1)
            if self.parent_widget.current_rotation:
                base_transform.rotate(self.parent_widget.current_rotation)
            base_transform.translate(-cx, -cy)

            if not base_transform.isInvertible():
                return
            inv_transform = base_transform.inverted()[0]
            start_local = inv_transform.map(self._drag_start)
            end_local = inv_transform.map(self._drag_end)

            x1, y1 = start_local.x(), start_local.y()
            x2, y2 = end_local.x(), end_local.y()
            x, y = min(x1, x2), min(y1, y2)
            x2, y2 = max(x1, x2), max(y1, y2)
            w, h = x2 - x, y2 - y
            if w > 20 and h > 20:
                # Map from displayed (possibly cropped/rotated) coords to
                # full-image coords. All FOUR corners must be mapped — under
                # a rotated crop the two-diagonal AABB degenerates — and the
                # minimum size re-checked in full-image space.
                pw = self.parent_widget
                pts = [pw.map_displayed_to_full(px, py)
                       for px, py in ((x, y), (x2, y), (x, y2), (x2, y2))]
                fx1 = int(min(p[0] for p in pts))
                fy1 = int(min(p[1] for p in pts))
                fx2 = int(max(p[0] for p in pts))
                fy2 = int(max(p[1] for p in pts))
                if (fx2 - fx1) > 20 and (fy2 - fy1) > 20:
                    ccr_backend.set_reference_frame_by_index(
                        pw.current_idx, (fx1, fy1, fx2, fy2))
                    pw.parent().parent().sliders_panel.set_temporary_hint(
                        "<b>Hint:</b><br>Reference frame set! Click the Convert button to view the updates.",
                        duration=5000
                    )
                    print("Set reference frame:", (fx1, fy1, fx2, fy2))
        self.parent_widget.update_preview(self.parent_widget.current_idx)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Let the parent widget handle the fitting to avoid conflicts
        if self.parent_widget and self.parent_widget.pixmap_item:
            self.parent_widget._fit_view_to_content()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if hasattr(self.parent_widget, "show_grid") and self.parent_widget.show_grid:
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing, False)  # <-- Use QPainter.Antialiasing
            pen = QPen(QColor(0, 80, 0, 80), 2)
            painter.setPen(pen)

            # Draw grid in scene (canvas) coordinates, so it stays stable
            scene_rect = self.scene().sceneRect()
            left, top, width, height = scene_rect.left(), scene_rect.top(), scene_rect.width(), scene_rect.height()

            # Draw vertical lines
            for i in range(0, 13):
                x = left + i * width / 12.0
                painter.drawLine(QPointF(x, top), QPointF(x, top + height))
            # Draw horizontal lines
            for i in range(0, 13):
                y = top + i * height / 12.0
                painter.drawLine(QPointF(left, y), QPointF(left + width, y))

            painter.restore()

    def wheelEvent(self, event):
        # Mouse wheel zooms in/out, anchored under the cursor
        delta = event.angleDelta().y()
        if delta == 0 or self.parent_widget.pixmap_item is None:
            event.ignore()
            return
        # Take keyboard focus so space-pan is immediately available after a
        # zoom (ClickFocus alone would leave space going to other widgets)
        self.setFocus()
        # The zoom center is the mouse location
        self.parent_widget.zoom_by(1.25 if delta > 0 else 0.8,
                                   anchor_vp=event.position().toPoint())
        event.accept()

    def _restore_mode_cursor(self):
        pw = self.parent_widget
        if pw is not None and pw.crop_mode:
            self.setCursor(Qt.CrossCursor)
        elif pw is not None and pw.heal_mode:
            self.setCursor(Qt.BlankCursor)   # the brush ring is the cursor
        elif self.wb_pick_mode:
            self.setCursor(_eyedropper_cursor())
        elif self.bwpoint_mode:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def _end_space_pan(self):
        self._space_pan = False
        self.setDragMode(QGraphicsView.NoDrag)
        self._restore_mode_cursor()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            # Hold space to pan with the mouse (hand drag). Refused while a
            # mouse button is down: engaging pan mid-drag would swallow the
            # release and leave the crop/reference/B-W interaction dangling.
            if QApplication.mouseButtons() == Qt.NoButton:
                self._space_pan = True
                self.setDragMode(QGraphicsView.ScrollHandDrag)
            event.accept()
            return
        # Disable up/down/left/right keys from scrolling the view
        if event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right):
            event.ignore()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._end_space_pan()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def focusOutEvent(self, event):
        # The space release may be delivered to whichever widget takes focus
        # next — never leave the hand-drag mode stuck on.
        if self._space_pan:
            self._end_space_pan()
        super().focusOutEvent(event)

    def leaveEvent(self, event):
        # Don't leave a ghost slice line painted when the cursor exits the
        # canvas (slice_move only fires while over the viewport).
        pw = self.parent_widget
        if pw is not None and pw.slice_mode:
            pw._set_slice_ghost(None)
        super().leaveEvent(event)

class CenteringSlider(QSlider):
    def mouseDoubleClickEvent(self, event):
        super().mouseDoubleClickEvent(event)
        self.setValue(0)

    def mousePressEvent(self, event):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        handle_rect = self.style().subControlRect(
            QStyle.CC_Slider,  # <-- Use QStyle.CC_Slider
            opt,
            QStyle.SC_SliderHandle,  # <-- Use QStyle.SC_SliderHandle
            self
        )
        if handle_rect.contains(event.pos()):
            super().mousePressEvent(event)
        else:
            event.ignore()

class ImagePreview(QWidget):
    ccr_converted = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(40, 0, 40, 0)

        # Toolbar
        self.toolbar = QToolBar()
        self.toolbar.setIconSize(QSize(24, 24))
        self.toolbar.setStyleSheet("""
    QToolButton { color: red; }
    QToolButton:enabled { color: black; }
    QToolButton:!hover { color: black; }
    QToolButton:disabled { color: gray; }
""")

        def add_spacer(width=5):
            spacer = QWidget()
            spacer.setFixedWidth(width)
            spacer.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.toolbar.addWidget(spacer)

        auto_icon = QIcon(resource_path("icons/auto.png"))
        if auto_icon.isNull():
            auto_icon = QIcon.fromTheme("view-refresh")
        auto_frame_action = QAction(auto_icon, "Auto Frame", self)
        auto_frame_action.triggered.connect(self.auto_frame)
        self.toolbar.addAction(auto_frame_action)
        add_spacer()

        rotate_left_icon = QIcon(resource_path("icons/rotate-left-icon-size_512.png"))
        if rotate_left_icon.isNull():
            rotate_left_icon = QIcon.fromTheme("view-refresh")
        rotate_left_action = QAction(rotate_left_icon, "Rotate Left", self)
        rotate_left_action.triggered.connect(self.rotate_left)
        self.toolbar.addAction(rotate_left_action)
        add_spacer()

        rotate_right_icon = QIcon(resource_path("icons/rotate-right-icon-size_512.png"))
        if rotate_right_icon.isNull():
            rotate_right_icon = QIcon.fromTheme("view-refresh")
        rotate_right_action = QAction(rotate_right_icon, "Rotate Right", self)
        rotate_right_action.triggered.connect(self.rotate_right)
        self.toolbar.addAction(rotate_right_action)
        add_spacer()

        mirror_v_icon = QIcon(resource_path("icons/vertical-mirror-icon.png"))
        if mirror_v_icon.isNull():
            mirror_v_icon = QIcon.fromTheme("view-refresh")
        mirror_v_action = QAction(mirror_v_icon, "Mirror Vertical", self)
        mirror_v_action.triggered.connect(self.mirror_vertical)
        self.toolbar.addAction(mirror_v_action)
        add_spacer()

        mirror_h_icon = QIcon(resource_path("icons/horizontal-mirror-icon.png"))
        if mirror_h_icon.isNull():
            mirror_h_icon = QIcon.fromTheme("view-refresh")
        mirror_h_action = QAction(mirror_h_icon, "Mirror Horizontal", self)
        mirror_h_action.triggered.connect(self.mirror_horizontal)
        self.toolbar.addAction(mirror_h_action)
        add_spacer()

        # Area-editing mask pickers — create a local masked adjustment area
        # (the global panel is the implicit whole-image layer). Guarded to
        # converted images inside add_area().
        self.add_circle_action = QAction("◯ Area", self)
        self.add_circle_action.setToolTip(
            "Add a circular local-adjustment area (squishable ellipse). "
            "Edit its adjustments in the right panel; drag on the canvas to "
            "move/resize/rotate it.")
        self.add_circle_action.triggered.connect(lambda: self.add_area("circle"))
        self.toolbar.addAction(self.add_circle_action)
        add_spacer()

        self.add_gradient_action = QAction("▤ Gradient", self)
        self.add_gradient_action.setToolTip(
            "Add a linear-gradient local-adjustment area. Drag its endpoints "
            "on the canvas to aim the ramp.")
        self.add_gradient_action.triggered.connect(lambda: self.add_area("gradient"))
        self.toolbar.addAction(self.add_gradient_action)
        add_spacer()

        convert_action = QAction("Convert", self)
        convert_action.triggered.connect(self.convert_ccr)
        self.toolbar.addAction(convert_action)
        add_spacer()

        self.unconvert_action = QAction("Un-convert", self)
        self.unconvert_action.triggered.connect(self.unconvert_ccr)
        self.toolbar.addAction(self.unconvert_action)
        add_spacer()

        # --- Dust healing tools (gated to converted images) ---------------
        self.heal_action = QAction("🩹 Heal", self)
        self.heal_action.setCheckable(True)
        self.heal_action.setToolTip(
            "Dust healing: auto-detect spots, click a marked spot to remove the "
            "detection, or click/drag to heal a missed spot or scratch.")
        self.heal_action.toggled.connect(self.toggle_heal_mode)
        self.toolbar.addAction(self.heal_action)
        add_spacer()

        self.auto_dust_action = QAction("Auto-Detect", self)
        self.auto_dust_action.setToolTip("Automatically detect dust spots and "
                                         "scratches on the current image.")
        self.auto_dust_action.triggered.connect(self.auto_detect_dust)
        self.toolbar.addAction(self.auto_dust_action)
        add_spacer()

        self.visualize_action = QAction("Visualize", self)
        self.visualize_action.setCheckable(True)
        self.visualize_action.setToolTip("Show a high-contrast view that makes "
                                         "dust easy to spot (display only).")
        self.visualize_action.toggled.connect(self.toggle_visualize)
        self.toolbar.addAction(self.visualize_action)
        add_spacer()

        self.clear_dust_action = QAction("Clear Spots", self)
        self.clear_dust_action.setToolTip("Remove all dust heal strokes.")
        self.clear_dust_action.triggered.connect(self.clear_dust)
        self.toolbar.addAction(self.clear_dust_action)
        add_spacer()

        self.brush_combo = QComboBox()
        self.brush_combo.addItems(["Brush S", "Brush M", "Brush L"])
        self.brush_combo.setCurrentIndex(1)   # M
        self.brush_combo.setFixedWidth(78)
        self.brush_combo.setToolTip("Manual heal brush size.")
        self.brush_combo.currentIndexChanged.connect(self._on_brush_changed)
        self.toolbar.addWidget(self.brush_combo)
        add_spacer()

        self.sensitivity_combo = QComboBox()
        self.sensitivity_combo.addItems(["Sens Low", "Sens Med", "Sens High"])
        self.sensitivity_combo.setCurrentIndex(1)   # Med
        self.sensitivity_combo.setFixedWidth(86)
        self.sensitivity_combo.setToolTip("Auto-detect sensitivity.")
        self.toolbar.addWidget(self.sensitivity_combo)
        add_spacer()

        self.export_action = QAction("Export…", self)
        self.export_action.triggered.connect(self.open_export_dialog)
        self.toolbar.addAction(self.export_action)

        # Zoom percentage dropdown (before Export). Percentages are relative
        # to the source image's ACTUAL full resolution, not the preview size.
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["Fit", "25%", "50%", "100%", "200%"])
        self.zoom_combo.setCurrentIndex(0)
        self.zoom_combo.setFixedWidth(78)
        # Editable + read-only line edit: lets the box DISPLAY arbitrary live
        # percentages from wheel zoom while still being a pick-only dropdown.
        self.zoom_combo.setEditable(True)
        self.zoom_combo.lineEdit().setReadOnly(True)
        self.zoom_combo.lineEdit().setAlignment(Qt.AlignCenter)
        self.zoom_combo.setInsertPolicy(QComboBox.NoInsert)
        self.zoom_combo.setToolTip(
            "Zoom level — percentages are relative to the image's actual size "
            "(100% = one source pixel per screen pixel)")
        self.zoom_combo.activated.connect(self._on_zoom_combo)
        zoom_spacer_l = QWidget()
        zoom_spacer_l.setFixedWidth(8)
        zoom_spacer_l.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        zoom_spacer_r = QWidget()
        zoom_spacer_r.setFixedWidth(8)
        zoom_spacer_r.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.toolbar.insertWidget(self.export_action, zoom_spacer_l)
        self.toolbar.insertWidget(self.export_action, self.zoom_combo)
        self.toolbar.insertWidget(self.export_action, zoom_spacer_r)

        self.layout.addWidget(self.toolbar)

        # Graphics Scene/View
        self.scene = QGraphicsScene(self)
        self.view = GraphicsImageView(self)
        self.view.setScene(self.scene)
        self.layout.addWidget(self.view)

        # Fine rotation slider
        self.rotation_slider = CenteringSlider(Qt.Horizontal)
        self.rotation_slider.setMinimum(-4500)
        self.rotation_slider.setMaximum(4500)
        self.rotation_slider.setValue(0)
        self.rotation_slider.setTickInterval(450)

        self.rotation_slider.setTickPosition(QSlider.TicksBelow)
        self.rotation_slider.valueChanged.connect(self._on_slider_rotate)
        self.rotation_slider.setEnabled(False)
        self.rotation_slider.sliderPressed.connect(self._on_slider_pressed)
        self.rotation_slider.sliderReleased.connect(self._on_slider_released)
        self.layout.addWidget(self.rotation_slider)

        self.setLayout(self.layout)

        # State
        self.current_pixmap = None
        self.current_idx = None
        self.current_converted = False
        self.current_fine_rotation = 0
        self.current_rotation = 0
        self.current_vertical_flip = False
        self.current_horizontal_flip = False

        self.pixmap_item = None
        self.reference_rect_item = None
        self.group_item = None

        # Crop state. crop_mode shows the full image with a selection overlay;
        # outside crop mode the confirmed crop is applied to the displayed
        # pixmap only (display-level — no reprocessing/re-conversion).
        self.crop_mode = False
        self._crop_rerender = False          # guards update_preview while entering crop mode
        self._pending_crop_local = None      # QRectF box (un-rotated) in full-pixmap coords
        self._pending_crop_angle = 0.0       # box rotation, degrees (Qt clockwise-positive)
        self._crop_drag = None               # in-progress handle/new-rect drag state
        self._crop_overlay_item = None
        self._crop_handle_items = []
        # Maps displayed-pixmap coords -> full-image coords while a confirmed
        # crop is displayed (None = identity / no crop shown)
        self._crop_display_transform = None
        self._crop_display_angle = 0.0

        # Area editing (local masked adjustment layers). Like crop, area mode
        # shows the FULL un-cropped image so the active area's normalized mask
        # geometry maps directly; the overlay reflects/edits that area's mask.
        self.area_mode = False
        self._area_rerender = False          # guards update_preview while entering
        self._area_drag = None               # in-progress handle drag state
        self._area_overlay_items = []
        # Coalesce the live masked-effect recompute during an area drag so the
        # effect follows the overlay (not just on release) without recomputing
        # on every single mouse-move.
        self._area_live_timer = QTimer(self)
        self._area_live_timer.setSingleShot(True)
        self._area_live_timer.timeout.connect(self._area_live_refresh)

        # Slice mode: split a scan containing several photos into separate
        # images along user-placed cut lines. Lines live in un-rotated image
        # coords: orient 'v' = vertical cut at an x-fraction, 'h' =
        # horizontal cut at a y-fraction.
        self.slice_mode = False
        self._slice_rerender = False     # guards update_preview during entry
        self._slice_lines = []           # {"orient": 'v'|'h', "frac": float, "item": item}
        self._slice_ghost_item = None    # preview line following the cursor
        self._slice_dim_item = None      # darker cast marking slice mode
        self._slice_drag = None          # line dict being dragged, or None
        self._slice_worker = None

        # Dust healing mode: click a marked spot to remove its detection, or
        # click/drag to heal. Strokes live on the image (normalized, un-cropped
        # space) and replay through apply_adjustments. See spec/dust-healing.md.
        self.heal_mode = False
        self.dust_visualize = False      # high-contrast "find the dust" view
        self._heal_drag = False          # a manual stroke is being drawn
        self._heal_stroke_pts = []       # in-progress stroke, full-image coords
        self._heal_ring_item = None      # brush-size ring following the cursor
        self._heal_ghost_items = []      # live preview of the stroke being drawn
        self._dust_overlay_items = []    # markers for committed strokes
        self._last_cursor_scene = None   # last cursor scene pos (for brush ring)

        # Coalesce a fine-rotation drag into a single undo step
        self._fine_rot_burst_active = False
        self._fine_rot_burst_timer = QTimer(self)
        self._fine_rot_burst_timer.setSingleShot(True)
        self._fine_rot_burst_timer.timeout.connect(self._end_fine_rot_burst)

        # --- Zoom & hi-res detail state ---
        # _zoom is relative to the fitted view (1.0 = fit). While zoomed past
        # the preview's native resolution, a hi-res re-render of the current
        # conversion is loaded asynchronously and displayed under identical
        # scene geometry. The rendered detail is retained until the image is
        # switched (zooming out keeps it cached for instant re-zoom).
        self._zoom = 1.0
        self._item_prescale = 1.0   # preview_width / displayed_pixmap_width
        self._hires = None          # cache for the CURRENT image only
        self._hires_workers = set()
        self._hires_timer = QTimer(self)
        self._hires_timer.setSingleShot(True)
        self._hires_timer.timeout.connect(self._maybe_request_hires)
        # Identity of the displayed image — indices shift when images are
        # removed, so "same image?" must never be answered by index alone.
        self._current_image_ref = None

        self._update_unconvert_action_state()

        # --- Add hotkey support ---
        self._install_shortcuts()
        # --- End hotkey support ---

    def _install_shortcuts(self):
        from PySide6.QtGui import QShortcut, QKeySequence
        # Left bracket: rotate left
        QShortcut(QKeySequence("["), self, self.rotate_left)
        # Right bracket: rotate right
        QShortcut(QKeySequence("]"), self, self.rotate_right)
        # Enter: confirm crop while in crop mode, otherwise convert image
        QShortcut(QKeySequence(Qt.Key_Return), self, self._on_enter_key)
        QShortcut(QKeySequence(Qt.Key_Enter), self, self._on_enter_key)
        # Esc: leave crop mode without changing the crop
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._on_escape_key)

    def clear_preview(self):
        """Empty the canvas after the last image is removed: drop all
        references to the deleted image (incl. the hi-res cache memory) and
        reset the view state."""
        if self.crop_mode:
            self._exit_crop_mode()
        if self.slice_mode:
            self._exit_slice_mode()
        if self.area_mode:
            self._exit_area_mode()
        if self.heal_mode:
            self._exit_heal_mode()
            self.heal_action.blockSignals(True)
            self.heal_action.setChecked(False)
            self.heal_action.blockSignals(False)
        self._release_hires(refresh=False)
        self.current_idx = None
        self._current_image_ref = None
        self.current_pixmap = None
        self.pixmap_item = None
        self.reference_rect_item = None
        self.view._bw_rect_item = None
        self.scene.clear()
        self._crop_overlay_item = None
        self._crop_handle_items = []
        self._slice_ghost_item = None
        self._slice_dim_item = None
        self._zoom = 1.0
        self._item_prescale = 1.0
        self.rotation_slider.setEnabled(False)
        self._sync_zoom_combo()
        self._update_unconvert_action_state()

    def _on_enter_key(self):
        if self.crop_mode:
            # Confirming a crop is display-level only — never re-converts.
            self.confirm_crop()
        elif self.slice_mode:
            self.confirm_slices()
        else:
            self.convert_ccr()

    def _on_escape_key(self):
        if self.crop_mode:
            self.cancel_crop_mode()
        elif self.slice_mode:
            self.cancel_slice_mode()
        elif self.area_mode:
            # Esc leaves area editing: deselect back to the Whole Image layer.
            self._select_whole_image_layer()
        elif self.heal_mode:
            self.heal_action.setChecked(False)   # toggled -> toggle_heal_mode(False)

    def update_preview(self, idx):
        ''' Update the UI image based on the backend, using the index from the thumbnail list. '''
        if idx is None or not (0 <= idx < len(ccr_backend.images)):
            print("Invalid index for update_preview:", idx)
            return
        # Any external refresh while picking a crop abandons crop mode
        # (image switch, slider change, conversion, ...).
        if self.crop_mode and not self._crop_rerender:
            self._exit_crop_mode()
        # Same for slice mode
        if self.slice_mode and not self._slice_rerender:
            self._exit_slice_mode()
        # Identity-based same-image check: indices shift when images are
        # removed from the list, so idx alone is not enough.
        img_obj_now = ccr_backend.images[idx]
        same_image = (idx == self.current_idx
                      and img_obj_now is self._current_image_ref)
        # Area-edit mode PERSISTS across same-image refreshes (so editing the
        # area's sliders/feather/geometry keeps the overlay visible); an image
        # switch leaves it, and so does the active area disappearing (e.g. an
        # undo that removed it). _area_rerender guards the deliberate in-mode
        # re-renders (enter / drag-commit).
        if self.area_mode and not self._area_rerender:
            active_gone = img_obj_now.get_area(
                getattr(img_obj_now, "active_area_id", None)) is None
            if not same_image or active_gone or not img_obj_now.converted:
                self._exit_area_mode()
        if not same_image:
            # The fine-rotation burst belongs to the previous image; a switch
            # must end it so the next image's first drag gets its own snapshot.
            self._end_fine_rot_burst()
            self._fine_rot_burst_timer.stop()
            # New image: back to the fitted view, free hi-res detail memory
            self._zoom = 1.0
            self._release_hires(refresh=False)
        else:
            # Same image refreshed (sliders, crop, convert, undo, ...): the
            # hi-res display layer is stale — fall back to preview pixels now
            # and re-render the detail once the controls go idle.
            self._invalidate_hires_display()
        self._current_image_ref = img_obj_now

        preview_img = ccr_backend.get_preview_by_index(idx)
        self.current_idx = idx

        # Display-level crop: show only the cropped region, except in crop
        # mode where the full image is shown so the user can adjust/regret.
        # Skipped for un-converted images: the crop tool is disabled there,
        # and the full negative (incl. film base) is needed to draw frames.
        prev_display_transform = self._crop_display_transform
        self._crop_display_transform = None
        self._crop_display_angle = 0.0
        crop = getattr(ccr_backend.images[idx], "crop_rect", None)
        crop_angle = getattr(ccr_backend.images[idx], "crop_angle", 0.0) or 0.0
        if (preview_img is not None and not preview_img.isNull()
                and crop is not None and not self.crop_mode and not self.slice_mode
                and not self.area_mode
                and ccr_backend.images[idx].converted):
            if crop_angle:
                extracted = self._extract_rotated_crop(preview_img, crop, crop_angle)
                if extracted is not None:
                    preview_img, self._crop_display_transform = extracted
                    self._crop_display_angle = crop_angle
            else:
                px_rect = self._crop_rect_to_pixels(crop, preview_img)
                if px_rect is not None:
                    t = QTransform()
                    t.translate(px_rect.x(), px_rect.y())
                    self._crop_display_transform = t
                    preview_img = preview_img.copy(px_rect)

        # If the displayed pixmap's geometry changed while zoomed in (crop
        # applied/cleared, crop mode entered/left, undo), a preserved zoom
        # would strand the viewport on a stale region — fall back to fit.
        if (same_image and self._zoom > 1.0 and preview_img is not None
                and self.current_pixmap is not None
                and (preview_img.size() != self.current_pixmap.size()
                     or prev_display_transform != self._crop_display_transform)):
            self._zoom = 1.0
            self._release_hires(refresh=False)

        # Visualize mode: swap in a high-contrast "find the dust" rendering.
        # Same geometry as the normal pixmap, so all heal coordinates are
        # unaffected; display-only (never stored/exported).
        if (self.dust_visualize and preview_img is not None
                and not preview_img.isNull()
                and ccr_backend.images[idx].converted):
            preview_img = self._visualize_pixmap(preview_img)

        self.current_pixmap = preview_img
        self.current_fine_rotation = ccr_backend.get_image_fine_rotation_by_index(idx)
        self.rotation_slider.setValue(self.current_fine_rotation)
        self.current_rotation = ccr_backend.get_image_rotation_by_index(idx)
        self.current_vertical_flip = ccr_backend.get_image_vertical_flip_by_index(idx)
        self.current_horizontal_flip = ccr_backend.get_image_horizontal_flip_by_index(idx)

        self.scene.clear()
        self.pixmap_item = None
        self.reference_rect_item = None
        self.view._bw_rect_item = None
        self._crop_overlay_item = None
        self._crop_handle_items = []
        self._area_overlay_items = []
        self._slice_ghost_item = None
        self._slice_dim_item = None
        # scene.clear() above dropped the dust overlay / brush ring / ghost items
        self._dust_overlay_items = []
        self._heal_ring_item = None
        self._heal_ghost_items = []
        for _slice_line in self._slice_lines:
            _slice_line["item"] = None

        self.parent().parent().sliders_panel.set_current_idx(idx)

        if preview_img and not preview_img.isNull():
            self.rotation_slider.setEnabled(True)
            self.pixmap_item = QGraphicsPixmapItem(preview_img)
            self.scene.addItem(self.pixmap_item)

            ref = getattr(ccr_backend.images[idx], "reference_frame", None)
            print("Loaded reference_frame from backend:", ref)
            if ref:
                # Backend rect is in full-image coords; map into the displayed
                # (possibly cropped) pixmap's coordinate space and clip to it.
                # With a rotated crop displayed the frame would not be
                # axis-aligned in display space, so it is not drawn at all.
                if self._crop_display_angle == 0.0:
                    x1, y1, x2, y2 = ref
                    rect = QRectF(x1, y1, x2 - x1, y2 - y1)
                    if self._crop_display_transform is not None:
                        inv = self._crop_display_transform.inverted()[0]
                        rect = inv.mapRect(rect).intersected(
                            QRectF(0, 0, preview_img.width(), preview_img.height()))
                    if not rect.isEmpty():
                        self.reference_rect_item = QGraphicsRectItem(rect)
                        pen = QPen(QColor(255, 0, 0, 180), 2, Qt.DashLine)
                        self.reference_rect_item.setPen(pen)
                        self.scene.addItem(self.reference_rect_item)

            else:
                # Show hint when no reference frame exists

                self.parent().parent().sliders_panel.set_hint(
                    "<b>Hint:</b><br>Draw a frame around the image + some film base (orange/brown). "
                    "Avoid white backlight or black film holder areas. Left-drag to draw, right-click to remove."
                )


            # Apply transformations which will handle fitting consistently
            # (it also redraws the crop/area overlay after the scene.clear()).
            self.apply_transformations()
            histogram = ccr_backend.get_histogram_image_by_index(idx)
            self.parent().parent().sliders_panel.set_histogram(histogram)

            # A confirmed crop magnifies the kept region even at the fitted
            # view, so — like zooming in — request crop-matched hi-res detail
            # instead of leaving the upscaled preview crop looking soft. The
            # zoom paths handle their own requests; this covers crop-at-fit
            # (apply/clear crop, undo, switching to an already-cropped image).
            # Debounced; the request self-validates against current state.
            if self._zoom <= 1.0 + 1e-9 and self._zoomed_in_enough():
                self._hires_timer.start(200)

        else:
            self.rotation_slider.setEnabled(False)

        self._update_unconvert_action_state()
        self._sync_zoom_combo()

    def update_reference_rect(self, start, end):
        # Build the base (coarse) transform
        cx = self.current_pixmap.width() / 2
        cy = self.current_pixmap.height() / 2
        base_transform = QTransform()
        base_transform.translate(cx, cy)

        if self.current_vertical_flip:
            base_transform.scale(1, -1)
        if self.current_horizontal_flip:
            base_transform.scale(-1, 1)
        if self.current_rotation:
            base_transform.rotate(self.current_rotation)
        base_transform.translate(-cx, -cy)

        if not base_transform.isInvertible():
            return
        inv_transform = base_transform.inverted()[0]
        start_local = inv_transform.map(start)
        end_local = inv_transform.map(end)

        x1, y1 = start_local.x(), start_local.y()
        x2, y2 = end_local.x(), end_local.y()
        rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        if self.reference_rect_item is None:
            self.reference_rect_item = QGraphicsRectItem(rect)
            pen = QPen(QColor(255, 0, 0, 120), 2, Qt.DashLine)
            self.reference_rect_item.setPen(pen)
            self.scene.addItem(self.reference_rect_item)
        else:
            self.reference_rect_item.setRect(rect)
        self.reference_rect_item.setTransform(base_transform)

    def _fit_view_to_content(self):
        """Consistently fit the view to the content, prioritizing the pixmap
        item — unless the user is zoomed in, in which case the current
        zoom/pan must never be yanked back to fit."""
        if self._zoom > 1.0:
            pass
        elif self.pixmap_item:
            # Fit to the transformed pixmap item for consistent zoom behavior
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        elif self.scene.sceneRect() and not self.scene.sceneRect().isNull():
            # Fallback to scene rect if no pixmap item
            self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        # Crop handle glyph sizes are baked at draw time from the view scale,
        # which fitInView just changed (window resize, rotate, ...) — rebuild
        # so the drawn handles keep matching their hit-test zones.
        if self.crop_mode and self._crop_overlay_item is not None:
            self._draw_crop_overlay()
        # Same for the area-edit overlay (handles sized from the view scale).
        if self.area_mode:
            self._draw_area_overlay()
        # Dust heal markers are redrawn here too (scene.clear / zoom rescale).
        if self.heal_mode:
            self._draw_dust_overlay()
        # Slice overlay follows the (possibly changed) display transform; the
        # ghost is simply dropped (it reappears on the next mouse move). The
        # dim cast is rebuilt first so the lines render on top of it.
        if self.slice_mode:
            self._set_slice_ghost(None)
            self._draw_slice_dim()
            if self._slice_lines:
                self._redraw_slice_lines()

    def apply_transformations(self):
        if not self.pixmap_item:
            return

        # Swap preview/hi-res pixmap first so the prescale below is current
        self._refresh_item_pixmap()

        # Center of the image
        cx = self.current_pixmap.width() / 2
        cy = self.current_pixmap.height() / 2

        # Coarse rotation and flips
        base_transform = QTransform()
        base_transform.translate(cx, cy)

        if self.current_vertical_flip:
            base_transform.scale(1, -1)
        if self.current_horizontal_flip:
            base_transform.scale(-1, 1)
        if self.current_rotation:
            base_transform.rotate(self.current_rotation)
        base_transform.translate(-cx, -cy)

        # Fine rotation (only for image). Suppressed while in crop mode so
        # the displayed image, the dim overlay, and the drag-to-rect mapping
        # (all base-transform-only) share the same coordinate space — the
        # selection then bounds exactly the pixels the crop keeps.
        # (Slice mode keeps the fine rotation displayed: cuts are placed on
        # the rotated frame and the rotation is baked into the slices.)
        img_transform = QTransform(base_transform)
        if self.current_fine_rotation and not self.crop_mode:
            img_transform.translate(cx, cy)
            img_transform.rotate(self.current_fine_rotation / 100.0)
            img_transform.translate(-cx, -cy)

        # A displayed hi-res pixmap is prescaled down so it occupies exactly
        # the preview pixmap's scene footprint — all geometry (crop, frames,
        # sampling) keeps working in preview coordinates.
        if self._item_prescale != 1.0:
            pre = QTransform()
            pre.scale(self._item_prescale, self._item_prescale)
            self.pixmap_item.setTransform(pre * img_transform)
        else:
            self.pixmap_item.setTransform(img_transform)

        # Pin the scene rect to the current image extent. QGraphicsScene's
        # default sceneRect only ever grows, so after a crop (smaller pixmap)
        # the stale larger rect would bound panning to empty space on one side
        # — the image gets stuck against a wall and can't be re-centred.
        self.scene.setSceneRect(
            self.pixmap_item.mapToScene(self.pixmap_item.boundingRect()).boundingRect())

        # Rectangle only gets coarse rotation/flips
        if self.reference_rect_item:
            self.reference_rect_item.setTransform(base_transform)

        # Always fit the view to the content consistently. (This also
        # rebuilds the crop overlay when crop mode is active, keeping the
        # dim region and handles aligned after rotates/flips.)
        self._fit_view_to_content()

    def _end_fine_rot_burst(self):
        self._fine_rot_burst_active = False

    def end_undo_bursts(self):
        """End any in-progress fine-rotation undo burst, e.g. after an undo,
        so the next change gets its own snapshot."""
        self._end_fine_rot_burst()
        self._fine_rot_burst_timer.stop()

    def _on_slider_rotate(self, value):
        # Snapshot once per burst, and only for real user changes (the slider
        # is also set programmatically to the backend value on image switch).
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is not None and img.fine_rotation_angle != value:
            if not self._fine_rot_burst_active:
                img.push_undo_state()
                self._fine_rot_burst_active = True
            self._fine_rot_burst_timer.start(800)
        self.current_fine_rotation = value
        ccr_backend.set_image_fine_rotation_by_index(self.current_idx, value)
        self.apply_transformations()

    def _on_slider_pressed(self):
        self.show_grid = True
        self.view.viewport().update()

    def _on_slider_released(self):
        self.show_grid = False
        self.view.viewport().update()

    def _push_undo_for_current(self):
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is not None:
            img.push_undo_state()

    def rotate_left(self):
        self._push_undo_for_current()
        self._crop_drag = None  # base transform change invalidates an active crop drag
        self._reset_zoom()      # orientation change returns to the fitted view
        self.current_rotation = (self.current_rotation - 90) % 360
        ccr_backend.set_image_rotation_by_index(self.current_idx, self.current_rotation)
        self.apply_transformations()

    def rotate_right(self):
        self._push_undo_for_current()
        self._crop_drag = None  # base transform change invalidates an active crop drag
        self._reset_zoom()      # orientation change returns to the fitted view
        self.current_rotation = (self.current_rotation + 90) % 360
        ccr_backend.set_image_rotation_by_index(self.current_idx, self.current_rotation)
        self.apply_transformations()

    def mirror_vertical(self):
        self._push_undo_for_current()
        self._crop_drag = None  # base transform change invalidates an active crop drag
        self._reset_zoom()      # orientation change returns to the fitted view
        flip = ccr_backend.get_image_vertical_flip_by_index(self.current_idx)
        ccr_backend.set_image_vertical_flip_by_index(self.current_idx, not flip)
        self.current_vertical_flip = not flip
        self.apply_transformations()

    def mirror_horizontal(self):
        self._push_undo_for_current()
        self._crop_drag = None  # base transform change invalidates an active crop drag
        self._reset_zoom()      # orientation change returns to the fitted view
        flip = ccr_backend.get_image_horizontal_flip_by_index(self.current_idx)
        ccr_backend.set_image_horizontal_flip_by_index(self.current_idx, not flip)
        self.current_horizontal_flip = not flip
        self.apply_transformations()

    def convert_ccr(self):
        if self.current_idx is None:
            QMessageBox.warning(self, "No Image Selected", "Please select an image to convert.")
            return
        if ccr_backend.get_reference_frame_by_index(self.current_idx) is None:
            QMessageBox.warning(self, "No Reference Frame", "Please draw a reference frame before converting.")
            return
        ccr_backend.convert_negative_by_index(self.current_idx)
        self.update_preview(self.current_idx)
        self.parent().parent().thumbnail_list.update_thumbnail(self.current_idx)
        self._update_unconvert_action_state()
        ccr_backend.save_catalog()

    def unconvert_ccr(self):
        if self.current_idx is None:
            QMessageBox.warning(self, "No Image Selected", "Please select an image to convert.")
            return
        ccr_backend.unconvert_negative_by_index(self.current_idx)
        self.update_preview(self.current_idx)
        self.parent().parent().thumbnail_list.update_thumbnail(self.current_idx)
        self._update_unconvert_action_state()
        ccr_backend.save_catalog()

    def _update_unconvert_action_state(self):
        self.current_converted = ccr_backend.get_converted_state_by_index(self.current_idx) if self.current_idx is not None else False
        self.unconvert_action.setEnabled(self.current_converted)
        self.export_action.setEnabled(any(img.converted for img in ccr_backend.images))
        # Dust healing is only meaningful on the converted positive.
        conv = self.current_converted
        for w in (self.heal_action, self.auto_dust_action, self.visualize_action,
                  self.clear_dust_action, self.brush_combo, self.sensitivity_combo):
            w.setEnabled(conv)
        if not conv and (self.heal_mode or self.dust_visualize):
            # Leaving the converted state drops out of heal/visualize cleanly,
            # without re-entering update_preview (we're often inside it here).
            self._exit_heal_mode()
            for act in (self.heal_action, self.visualize_action):
                act.blockSignals(True)
                act.setChecked(False)
                act.blockSignals(False)
        parent = self.parent()
        if hasattr(parent.parent(), "sliders_panel"):
            print("Setting sliders enabled based on current_converted:", self.current_converted)
            parent.parent().sliders_panel.set_sliders_enabled(self.current_converted)


    # ===== Dust healing ===================================================
    # Click a marked spot to remove its detection; click/drag to heal a missed
    # spot or scratch. Strokes live on the image in normalized, un-cropped space
    # and replay through apply_adjustments. Coordinate authoring/overlay use
    # _image_transform() (coarse + fine rotation) so they are correct even when
    # the straighten slider is non-zero. See spec/dust-healing.md.

    def _image_transform(self):
        """Scene<->preview-local map including coarse AND fine rotation (no
        prescale): scene == img_transform(preview_local). Use this, not the
        coarse-only _base_transform, for heal authoring and overlays."""
        base = self._base_transform()
        if base is None:
            return None
        t = QTransform(base)
        if self.current_fine_rotation:
            cx = self.current_pixmap.width() / 2.0
            cy = self.current_pixmap.height() / 2.0
            t.translate(cx, cy)
            t.rotate(self.current_fine_rotation / 100.0)
            t.translate(-cx, -cy)
        return t if t.isInvertible() else None

    def map_full_to_displayed(self, x, y):
        """Inverse of map_displayed_to_full: full-image -> displayed-pixmap
        coords (identity when no crop is being displayed)."""
        if self._crop_display_transform is None:
            return x, y
        p = self._crop_display_transform.inverted()[0].map(QPointF(x, y))
        return p.x(), p.y()

    def _full_dims(self):
        """(W, H) of the full un-cropped working image (resized_raw), or None."""
        img = (ccr_backend.get_image_by_index(self.current_idx)
               if self.current_idx is not None else None)
        if img is None or img.resized_raw is None:
            return None
        h, w = img.resized_raw.shape[:2]
        return w, h

    def _brush_radius_norm(self):
        from core.dust_removal import BRUSH_NORM
        key = {0: "S", 1: "M", 2: "L"}.get(self.brush_combo.currentIndex(), "M")
        return BRUSH_NORM[key]

    def _on_brush_changed(self, _idx):
        if self.heal_mode and self._last_cursor_scene is not None:
            self._update_brush_ring(self._last_cursor_scene)

    # ---- mode enter/leave ----
    def toggle_heal_mode(self, checked):
        if checked:
            if self.current_idx is None or not getattr(self, "current_converted", False):
                self.heal_action.blockSignals(True)
                self.heal_action.setChecked(False)
                self.heal_action.blockSignals(False)
                return
            if self.crop_mode:
                self._exit_crop_mode()
            if self.slice_mode:
                self._exit_slice_mode()
            if self.area_mode:
                self._exit_area_mode()
            self.view.bwpoint_mode = None
            self.view.wb_pick_mode = False
            self.heal_mode = True
            self.update_preview(self.current_idx)   # redraws the stroke overlay
            self.view.setCursor(Qt.BlankCursor)
            self._heal_hint()
        else:
            self._exit_heal_mode()
            if self.current_idx is not None:
                self.update_preview(self.current_idx)

    def _exit_heal_mode(self):
        """Pure cleanup — never re-enters update_preview (callers refresh)."""
        self.heal_mode = False
        self._heal_drag = False
        self._heal_stroke_pts = []
        self._remove_brush_ring()
        self._remove_heal_ghost()
        self._remove_dust_overlay_items()
        self.view.setCursor(Qt.ArrowCursor)

    def toggle_visualize(self, checked):
        self.dust_visualize = bool(checked) and getattr(self, "current_converted", False)
        if self.current_idx is not None:
            self.update_preview(self.current_idx)

    # ---- toolbar actions ----
    def auto_detect_dust(self):
        if self.current_idx is None or not getattr(self, "current_converted", False):
            return
        if not self.heal_mode:
            self.heal_action.setChecked(True)   # enters heal mode (via toggled)
        key = {0: "low", 1: "med", 2: "high"}.get(
            self.sensitivity_combo.currentIndex(), "med")
        self._push_undo_for_current()
        n_detected, n_replaced = ccr_backend.detect_dust_by_index(self.current_idx, key)
        self._after_heal_change()
        panel = self._sliders_panel()
        if panel is not None:
            msg = f"Detected {n_detected} spot(s)"
            if n_replaced:
                msg += f" (replaced {n_replaced} previous auto)"
            try:
                panel.set_temporary_hint(
                    msg + ". Click a marked spot to remove it; click/drag to heal.",
                    duration=5000)
            except AttributeError:
                pass

    def clear_dust(self):
        if self.current_idx is None:
            return
        if not ccr_backend.get_dust_strokes_by_index(self.current_idx):
            return
        self._push_undo_for_current()
        ccr_backend.clear_dust_strokes_by_index(self.current_idx)
        self._after_heal_change()

    def _after_heal_change(self):
        """Strokes changed: refresh canvas + thumbnail, update hint, persist."""
        self.update_preview(self.current_idx)
        try:
            self.parent().parent().thumbnail_list.update_thumbnail(self.current_idx)
        except Exception:
            pass
        self._heal_hint()
        ccr_backend.save_catalog()

    # ---- canvas interaction (called by GraphicsImageView) ----
    def _heal_scene_to_full(self, scene_pos):
        t = self._image_transform()
        dims = self._full_dims()
        if t is None or dims is None:
            return None
        local = t.inverted()[0].map(scene_pos)
        return self.map_displayed_to_full(local.x(), local.y())

    def _heal_hit_test(self, fx, fy):
        """Index of the topmost stroke whose footprint contains (fx, fy), or None."""
        dims = self._full_dims()
        if dims is None:
            return None
        W, H = dims
        L = max(W, H)
        tol_img = 8.0 / max(self._view_scale(), 1e-6)
        strokes = ccr_backend.get_dust_strokes_by_index(self.current_idx)
        best = None
        for i, st in enumerate(strokes):
            hit = max(2.0, st.get("radius", 0.006) * L) + tol_img
            for nx, ny in st.get("points", []):
                if (nx * W - fx) ** 2 + (ny * H - fy) ** 2 <= hit * hit:
                    best = i
                    break
        return best

    def heal_press(self, scene_pos):
        full = self._heal_scene_to_full(scene_pos)
        if full is None:
            return
        hit = self._heal_hit_test(*full)
        if hit is not None:
            self._remove_stroke(hit)
            self._heal_drag = False
            self._heal_stroke_pts = []
            return
        self._heal_drag = True
        self._heal_stroke_pts = [tuple(full)]
        self._draw_heal_ghost()

    def heal_move(self, scene_pos):
        self._update_brush_ring(scene_pos)
        if not self._heal_drag:
            return
        full = self._heal_scene_to_full(scene_pos)
        if full is None:
            return
        fx, fy = full
        dims = self._full_dims()
        L = max(dims) if dims else 1080
        min_gap = max(1.0, 0.5 * self._brush_radius_norm() * L)
        if self._heal_stroke_pts:
            lx, ly = self._heal_stroke_pts[-1]
            if (fx - lx) ** 2 + (fy - ly) ** 2 < min_gap * min_gap:
                return
        self._heal_stroke_pts.append((fx, fy))
        self._draw_heal_ghost()

    def heal_release(self, scene_pos):
        if not self._heal_drag:
            return
        full = self._heal_scene_to_full(scene_pos)
        if full is not None:
            self._heal_stroke_pts.append(tuple(full))
        self._heal_drag = False
        self._remove_heal_ghost()
        self._commit_manual_stroke()

    def heal_remove_at(self, scene_pos):
        full = self._heal_scene_to_full(scene_pos)
        if full is None:
            return
        hit = self._heal_hit_test(*full)
        if hit is not None:
            self._remove_stroke(hit)

    def _remove_stroke(self, index):
        self._push_undo_for_current()
        ccr_backend.remove_dust_stroke_by_index(self.current_idx, index)
        self._after_heal_change()

    def _commit_manual_stroke(self):
        pts = self._heal_stroke_pts
        self._heal_stroke_pts = []
        dims = self._full_dims()
        if not pts or dims is None:
            return
        W, H = dims
        norm = [[min(1.0, max(0.0, x / W)), min(1.0, max(0.0, y / H))]
                for x, y in pts]
        stroke = {
            "points": norm,
            "radius": self._brush_radius_norm(),
            "connect": len(norm) > 1,
            "kind": "scratch" if len(norm) > 1 else "spot",
            "source": "manual",
        }
        self._push_undo_for_current()
        ccr_backend.add_dust_stroke_by_index(self.current_idx, stroke)
        self._after_heal_change()

    # ---- overlay drawing ----
    def _update_brush_ring(self, scene_pos):
        self._last_cursor_scene = scene_pos
        dims = self._full_dims()
        if dims is None or not self.heal_mode:
            self._remove_brush_ring()
            return
        r = self._brush_radius_norm() * max(dims)
        rect = QRectF(scene_pos.x() - r, scene_pos.y() - r, 2 * r, 2 * r)
        if self._heal_ring_item is None:
            self._heal_ring_item = QGraphicsEllipseItem(rect)
            pen = QPen(QColor(255, 255, 255, 235), 0, Qt.DashLine)
            pen.setCosmetic(True)
            self._heal_ring_item.setPen(pen)
            self._heal_ring_item.setBrush(QBrush(QColor(255, 255, 255, 25)))
            self._heal_ring_item.setZValue(50)
            self.scene.addItem(self._heal_ring_item)
        else:
            self._heal_ring_item.setRect(rect)

    def _remove_brush_ring(self):
        if self._heal_ring_item is not None:
            try:
                self.scene.removeItem(self._heal_ring_item)
            except Exception:
                pass
            self._heal_ring_item = None

    def _draw_heal_ghost(self):
        self._remove_heal_ghost()
        t = self._image_transform()
        dims = self._full_dims()
        if t is None or dims is None or not self._heal_stroke_pts:
            return
        r = max(1.0, self._brush_radius_norm() * max(dims))
        disp = [self.map_full_to_displayed(x, y) for x, y in self._heal_stroke_pts]
        path = QPainterPath()
        if len(disp) >= 2:
            path.moveTo(disp[0][0], disp[0][1])
            for dx, dy in disp[1:]:
                path.lineTo(dx, dy)
            item = QGraphicsPathItem(path)
            pen = QPen(QColor(*_HEAL_GHOST_RGB, 130), 2 * r)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            item.setPen(pen)
        else:
            path.addEllipse(QPointF(disp[0][0], disp[0][1]), r, r)
            item = QGraphicsPathItem(path)
            item.setPen(QPen(QColor(*_HEAL_GHOST_RGB, 200), 0))
            item.setBrush(QBrush(QColor(*_HEAL_GHOST_RGB, 90)))
        item.setTransform(t)
        item.setZValue(40)
        self.scene.addItem(item)
        self._heal_ghost_items.append(item)

    def _remove_heal_ghost(self):
        for it in self._heal_ghost_items:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self._heal_ghost_items = []

    def _draw_dust_overlay(self):
        self._remove_dust_overlay_items()
        if (not self.heal_mode or self.current_pixmap is None
                or self.current_idx is None):
            return
        t = self._image_transform()
        dims = self._full_dims()
        if t is None or dims is None:
            return
        strokes = ccr_backend.get_dust_strokes_by_index(self.current_idx)
        if not strokes:
            return
        W, H = dims
        L = max(W, H)
        scale = self._view_scale()
        for st in strokes:
            r = max(1.0, st.get("radius", 0.006) * L)
            rgb = _HEAL_AUTO_RGB if st.get("source") == "auto" else _HEAL_MANUAL_RGB
            disp = [self.map_full_to_displayed(nx * W, ny * H)
                    for nx, ny in st.get("points", [])]
            if not disp:
                continue
            path = QPainterPath()
            if st.get("connect") and len(disp) >= 2:
                path.moveTo(disp[0][0], disp[0][1])
                for dx, dy in disp[1:]:
                    path.lineTo(dx, dy)
                item = QGraphicsPathItem(path)
                pen = QPen(QColor(*rgb, 110), 2 * r)
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                item.setPen(pen)
            else:
                for dx, dy in disp:
                    path.addEllipse(QPointF(dx, dy), r, r)
                item = QGraphicsPathItem(path)
                item.setPen(QPen(QColor(*rgb, 235), max(1.2 / scale, 0.4)))
                item.setBrush(QBrush(QColor(*rgb, 70)))
            item.setTransform(t)
            item.setZValue(30)
            self.scene.addItem(item)
            self._dust_overlay_items.append(item)

    def _remove_dust_overlay_items(self):
        for it in self._dust_overlay_items:
            try:
                self.scene.removeItem(it)
            except Exception:
                pass
        self._dust_overlay_items = []

    def _heal_hint(self):
        panel = self._sliders_panel()
        if panel is None or self.current_idx is None:
            return
        strokes = ccr_backend.get_dust_strokes_by_index(self.current_idx)
        n_auto = sum(1 for s in strokes if s.get("source") == "auto")
        n_manual = len(strokes) - n_auto
        try:
            panel.set_hint(
                f"<b>Dust healing.</b> {len(strokes)} spot(s) "
                f"({n_auto} auto, {n_manual} manual).<br>"
                "Click a marked spot to remove it; click or drag to heal a "
                "missed spot/scratch. <b>Auto-Detect</b> finds spots; "
                "<b>Visualize</b> reveals them. Ctrl+Z undo · Esc to exit.")
        except AttributeError:
            pass

    def _visualize_pixmap(self, pixmap):
        """A high-contrast 'find the dust' rendering of the displayed pixmap
        (same geometry — coordinates are unaffected)."""
        try:
            import numpy as np
            import cv2
            qimg = pixmap.toImage().convertToFormat(QImage.Format_RGB888)
            w, h = qimg.width(), qimg.height()
            bpl = qimg.bytesPerLine()
            buf = np.frombuffer(qimg.constBits(), np.uint8, count=bpl * h)
            arr = buf.reshape(h, bpl)[:, :w * 3].reshape(h, w, 3)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
            hp = np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 2.0))
            hp = np.clip(hp * 8.0, 0, 255).astype(np.uint8)
            vis = np.ascontiguousarray(cv2.cvtColor(hp, cv2.COLOR_GRAY2RGB))
            out = QImage(vis.data, w, h, w * 3, QImage.Format_RGB888).copy()
            return QPixmap.fromImage(out)
        except Exception as e:
            print(f"Dust visualize failed: {e}")
            return pixmap

    def set_bwpoint_mode(self, mode):
        """mode: 'black' | 'white' | None"""
        if mode and self.crop_mode:
            # Pick modes and crop mode are mutually exclusive in both directions
            self.cancel_crop_mode()
        if mode and self.slice_mode:
            self.cancel_slice_mode()
        self.view.bwpoint_mode = mode
        self.view.wb_pick_mode = False
        self.view.setCursor(Qt.CrossCursor if mode else Qt.ArrowCursor)

    def set_wb_pick_mode(self, enabled):
        """Eyedropper mode: next click on the image picks a neutral point
        for automatic temperature/tint adjustment."""
        if enabled and self.crop_mode:
            self.cancel_crop_mode()
        if enabled and self.slice_mode:
            self.cancel_slice_mode()
        self.view.wb_pick_mode = enabled
        self.view.bwpoint_mode = None
        self.view.setCursor(_eyedropper_cursor() if enabled else Qt.ArrowCursor)

    # --- Zoom & hi-res detail --------------------------------------------
    # Wheel zooms relative to the fitted view (Lightroom-style). Once the
    # 1080px preview is displayed past its native resolution, a hi-res
    # re-render of the same conversion is produced on a worker thread and
    # swapped in under identical scene geometry; zooming back out (or
    # switching images) frees that memory again.

    MAX_ZOOM = 8.0        # wheel ceiling, relative to the fitted view
    MAX_PERCENT = 4.0     # absolute ceiling: 400% of the source's actual size

    def _compute_fit_scale(self):
        """View scale (view px per scene px) that fitInView would produce for
        the current pixmap item — recomputed from geometry so it stays correct
        after window resizes while zoomed."""
        if self.pixmap_item is None:
            return 1.0
        br = self.pixmap_item.mapToScene(
            self.pixmap_item.boundingRect()).boundingRect()
        vp = self.view.viewport().rect()
        if br.width() <= 0 or br.height() <= 0:
            return 1.0
        return min(vp.width() / br.width(), vp.height() / br.height())

    def _preview_to_source_ratio(self):
        """preview-pixmap pixels per source (actual-size) pixel. The displayed
        scene is in preview-pixmap pixel units, so percentage-of-actual needs
        this factor. Returns None when the source size is unknown.

        The factor is taken from the ACTUAL working-image resolution
        (resized_raw), not re-derived from the 1080 resize policy: RAW is
        decoded at HALF size before the 1080 downscale, so a low-res RAW's
        preview can be well below min(1080, full_long)."""
        img = ccr_backend.get_image_by_index(self.current_idx) if self.current_idx is not None else None
        full = getattr(img, "original_full_size", None) if img is not None else None
        rr = getattr(img, "resized_raw", None) if img is not None else None
        if not full or rr is None:
            return None
        source_long = max(full)
        preview_long = max(rr.shape[:2])  # the working/preview pixel resolution
        if source_long <= 0 or preview_long <= 0:
            return None
        return float(preview_long) / float(source_long)

    def _max_zoom(self):
        """Wheel/zoom ceiling as a multiple of fit: the larger of 8x fit or
        whatever reaches MAX_PERCENT of the actual size."""
        cap = self.MAX_ZOOM
        ratio = self._preview_to_source_ratio()
        fit = self._compute_fit_scale()
        if ratio and fit > 0:
            cap = max(cap, (self.MAX_PERCENT / ratio) / fit)
        return cap

    def zoom_by(self, factor, anchor_vp=None):
        """Zoom by a factor, anchored under anchor_vp (the cursor) or the
        viewport center; clamped to [fit, max]."""
        if self.pixmap_item is None or self.current_pixmap is None:
            return
        # Re-derive _zoom from the live transform first: a window resize while
        # zoomed preserves the absolute transform but changes the fit scale, so
        # the stored _zoom (a multiple of fit) would otherwise be stale and the
        # next wheel step would jump discontinuously.
        fit = self._compute_fit_scale()
        if fit > 0:
            self._zoom = max(1.0, self._view_scale() / fit)
        new_zoom = max(1.0, min(self._max_zoom(), self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-9:
            return
        if new_zoom <= 1.0:
            self._zoom = 1.0
            self._fit_view_to_content()  # snap back to the exact fit (centered)
        else:
            self._apply_zoom_scale(self._compute_fit_scale() * new_zoom,
                                   new_zoom, anchor_vp=anchor_vp)
        self._update_hires_state()
        self._sync_zoom_combo()

    def _apply_zoom_scale(self, target_view_scale, new_zoom, anchor_vp=None):
        """Set the absolute view scale, keeping the scene point under anchor_vp
        (the cursor, or the viewport center) fixed — manual anchoring, since
        the view uses NoAnchor."""
        cur = self._view_scale()
        if cur <= 0:
            return
        self._zoom = new_zoom
        factor = target_view_scale / cur
        if anchor_vp is None:
            anchor_vp = self.view.viewport().rect().center()
        old_scene = self.view.mapToScene(anchor_vp)
        self.view.scale(factor, factor)
        new_scene = self.view.mapToScene(anchor_vp)
        delta = new_scene - old_scene
        self.view.translate(delta.x(), delta.y())
        if self.crop_mode and self._crop_overlay_item is not None:
            self._draw_crop_overlay()  # handle sizes track the view scale
        if self.area_mode:
            self._draw_area_overlay()  # area handle sizes track the view scale too

    def zoom_to_percent(self, pct):
        """Zoom to an absolute percentage of the source's ACTUAL size (1.0 =
        100% = one source pixel per screen pixel). Anchors on the current
        frame center; if the whole image fits at that scale, centers it."""
        if self.pixmap_item is None or self.current_pixmap is None:
            return
        ratio = self._preview_to_source_ratio()
        fit = self._compute_fit_scale()
        if not ratio or fit <= 0:
            return
        pct = min(pct, self.MAX_PERCENT)
        target_view_scale = pct / ratio
        new_zoom = target_view_scale / fit
        # Whole image is visible whenever the requested scale is <= fit
        if new_zoom <= 1.0 + 1e-9:
            self._zoom = 1.0
            self._fit_view_to_content()  # fit shows the entire image, centered
        else:
            self._apply_zoom_scale(target_view_scale, new_zoom)
        self._update_hires_state()
        self._sync_zoom_combo()

    def zoom_to_fit(self):
        self._zoom = 1.0
        self._fit_view_to_content()
        self._update_hires_state()
        self._sync_zoom_combo()

    def _current_percent(self):
        """Current zoom as a fraction of the source's actual size, or None."""
        ratio = self._preview_to_source_ratio()
        if not ratio:
            return None
        return self._view_scale() * ratio

    # A confirmed crop that keeps less than this fraction of a side (i.e. more
    # than ~15% cropped off width or height) magnifies the kept region enough
    # at the fitted view that the upscaled 1080px preview reads as soft —
    # worth fetching crop-matched hi-res detail, just like zooming in.
    CROP_HIRES_KEEP_MAX = 0.85

    def _zoomed_in_enough(self):
        """True when preview pixels are magnified on screen — i.e. higher-res
        detail would help. Magnification comes from zooming in, or from cropping
        into the image (the kept region is scaled up to fill the view even at
        the fitted zoom)."""
        if self._view_scale() <= 1.05:
            return False
        return self._zoom > 1.0 or self._crop_wants_hires()

    def _crop_wants_hires(self):
        """At the fitted view a confirmed crop magnifies the kept region. Worth
        a crop-matched hi-res fetch when the crop removes a meaningful slice
        (>15% of a side) and the source carries more detail than the preview.
        Crop/slice/area modes show the FULL image, so no crop magnification."""
        if self.crop_mode or self.slice_mode or self.area_mode:
            return False
        img = (ccr_backend.get_image_by_index(self.current_idx)
               if self.current_idx is not None else None)
        if img is None or not getattr(img, "converted", False):
            return False
        crop = getattr(img, "crop_rect", None)
        if crop is None:
            return False
        if min(crop[2] - crop[0], crop[3] - crop[1]) > self.CROP_HIRES_KEEP_MAX:
            return False
        # Only worthwhile if a higher-resolution decode actually exists.
        ratio = self._preview_to_source_ratio()
        return ratio is not None and ratio < 0.98

    def _update_hires_state(self):
        # Keep any rendered detail in memory until the image is switched
        # (per user request) — only request when zoomed in; never release on
        # zoom-out. Falling back to the preview pixmap for display is handled
        # by _refresh_item_pixmap (which gates on _zoomed_in_enough()).
        if self._zoomed_in_enough():
            self._maybe_request_hires()
        elif self.pixmap_item is not None:
            self._refresh_item_pixmap()
            self.apply_transformations()

    def _reset_zoom(self):
        self._zoom = 1.0
        # Image-level reset (rotate/flip/undo/crop entry): the cached detail
        # no longer matches the geometry, so free it here.
        self._release_hires(refresh=False)
        self._sync_zoom_combo()

    def _on_zoom_combo(self, index):
        text = self.zoom_combo.itemText(index)
        if text == "Fit":
            self.zoom_to_fit()
        else:
            try:
                pct = float(text.rstrip('%')) / 100.0
            except ValueError:
                return
            self.zoom_to_percent(pct)

    def _sync_zoom_combo(self):
        """Reflect the current zoom in the toolbar dropdown without firing it."""
        if not hasattr(self, "zoom_combo"):
            return
        if self._zoom <= 1.0 + 1e-9:
            label = "Fit"
        else:
            pct = self._current_percent()
            label = f"{round(pct * 100)}%" if pct else "Fit"
        self.zoom_combo.blockSignals(True)
        self.zoom_combo.setCurrentText(label)
        # setCurrentText on an editable combo doesn't move currentIndex, which
        # leaves the open dropdown highlighting the wrong row — align it when
        # the label matches a listed item.
        match = self.zoom_combo.findText(label)
        if match >= 0:
            self.zoom_combo.setCurrentIndex(match)
        self.zoom_combo.blockSignals(False)

    def _hires_signature(self, img):
        """Identity of the conversion baked into the preview, taken from the
        snapshot captured at convert time — never from live editable state
        (reference frame redraws, global B/W point resampling, or fine
        rotation nudges change the NEXT conversion, not the displayed one).
        Returns None when no color-matched replay is possible."""
        if not img.converted:
            return ("unconverted",)
        ci = getattr(img, "conversion_inputs", None)
        if ci is None:
            return None
        return ("converted", tuple(sorted(ci.items())))

    def _current_adj_sig(self):
        """Identity of the adjustment state baked into the hi-res DISPLAY."""
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None:
            return None
        areas_sig = tuple(
            (a.get("id"), a.get("kind"), bool(a.get("enabled", True)),
             float(a.get("feather", 0.25)), float(a.get("angle", 0.0)),
             tuple(sorted((a.get("geometry") or {}).items())),
             tuple(sorted((a.get("settings") or {}).items())))
            for a in getattr(img, "area_layers", []))
        return (tuple(sorted(img.adjustment_settings.items())),
                img.contrast_base, img.temperature_base, img.brightness_base,
                getattr(img, "color_profile", "color"),
                areas_sig)

    HIRES_MAX_LONG_SIDE = 4500   # bounds non-RAW decodes (RAW half-size passes through)

    def _maybe_request_hires(self):
        if not self._zoomed_in_enough() or self.current_idx is None:
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None:
            return
        sig = self._hires_signature(img)
        if sig is None:
            return  # no color-matched replay possible for this image
        adj_sig = self._current_adj_sig()
        cache = self._hires
        base = None
        if cache is not None and cache["img"] is img and cache["sig"] == sig:
            if cache.get("full_pm") is not None and cache.get("adj_sig") == adj_sig:
                # Cache is current — just make sure it is displayed
                self._refresh_item_pixmap()
                self.apply_transformations()
                return
            base = cache.get("base")  # re-adjust only; skip decode+convert
        # A matching render may already be in flight; its result is accepted
        # by current-state validation, so waiting for it is always safe.
        for w in self._hires_workers:
            if (w.isRunning() and w.req_img is img
                    and w.req_sig == sig and w.req_adj_sig == adj_sig):
                return
        worker = HiResDetailWorker(img, sig, adj_sig, base, self.HIRES_MAX_LONG_SIDE)
        worker.finished_hires.connect(self._on_hires_ready)
        worker.finished.connect(lambda w=worker: self._hires_workers.discard(w))
        self._hires_workers.add(worker)
        worker.start()

    def _on_hires_ready(self, img_obj, sig, adj_sig, base, display8):
        # Accept by validating against CURRENT state (not a generation
        # counter): the result is useful iff the same image object is still
        # displayed, its conversion snapshot still matches, and we are still
        # zoomed in. This also re-adopts renders that were "released" by a
        # zoom-out/zoom-in round trip while in flight.
        current = (ccr_backend.get_image_by_index(self.current_idx)
                   if self.current_idx is not None else None)
        if (current is None or current is not img_obj
                or not self._zoomed_in_enough()
                or sig != self._hires_signature(current)):
            return  # no longer needed — arrays just get GC'd
        h, w = display8.shape[:2]
        qimg = QImage(display8.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        if adj_sig != self._current_adj_sig():
            # Sliders moved while rendering: the decoded base is still valid,
            # the adjusted display is not — keep the base, re-render shortly.
            self._hires = {"img": img_obj, "sig": sig, "adj_sig": None,
                           "base": base, "full_pm": None,
                           "display_pm": None, "crop_sig": None}
            self._hires_timer.start(150)
            return
        self._hires = {"img": img_obj, "sig": sig, "adj_sig": adj_sig, "base": base,
                       "full_pm": QPixmap.fromImage(qimg),
                       "display_pm": None, "crop_sig": None}
        self._refresh_item_pixmap()
        self.apply_transformations()

    def _invalidate_hires_display(self):
        """Same-image refresh (adjustments/crop/conversion changed): drop the
        stale hi-res display layer, keep the decoded base where possible, and
        re-render once the controls go idle."""
        if self._hires is None and not self._zoomed_in_enough():
            return
        if self._hires is not None:
            self._hires["display_pm"] = None
            self._hires["crop_sig"] = None
            if self._hires.get("adj_sig") != self._current_adj_sig():
                self._hires["full_pm"] = None
                self._hires["adj_sig"] = None
        if self._zoomed_in_enough():
            self._hires_timer.start(350)

    def _hires_display_pixmap(self):
        """Crop-matched hi-res pixmap for the current display, or None."""
        cache = self._hires
        if (cache is None or cache.get("full_pm") is None
                or not self._zoomed_in_enough()):
            return None
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None or cache["img"] is not img:
            return None
        # The base must still describe the conversion baked into the preview
        if cache["sig"] != self._hires_signature(img):
            return None
        crop = getattr(img, "crop_rect", None)
        angle = getattr(img, "crop_angle", 0.0) or 0.0
        crop_active = (crop is not None and not self.crop_mode
                       and not self.slice_mode and img.converted)
        crop_sig = (crop, angle) if crop_active else None
        if cache.get("display_pm") is not None and cache.get("crop_sig") == crop_sig:
            return cache["display_pm"]
        pm = cache["full_pm"]
        if crop_active:
            if angle:
                ext = self._extract_rotated_crop(pm, crop, angle)
                pm = ext[0] if ext is not None else None
            else:
                r = self._crop_rect_to_pixels(crop, pm)
                pm = pm.copy(r) if r is not None else None
        if pm is None or pm.isNull():
            return None
        cache["display_pm"] = pm
        cache["crop_sig"] = crop_sig
        return pm

    def _refresh_item_pixmap(self):
        """Display either the preview pixmap or the hi-res detail pixmap;
        _item_prescale keeps the scene geometry identical either way."""
        if self.pixmap_item is None or self.current_pixmap is None:
            self._item_prescale = 1.0
            return
        pm = self._hires_display_pixmap()
        if pm is not None and pm.width() > 0:
            self._item_prescale = self.current_pixmap.width() / pm.width()
            self.pixmap_item.setPixmap(pm)
        else:
            self._item_prescale = 1.0
            self.pixmap_item.setPixmap(self.current_pixmap)

    def _release_hires(self, refresh=True):
        """Free hi-res detail memory (zoomed out / image switched). In-flight
        renders are not cancelled; their results are validated against the
        CURRENT state on arrival and re-adopted only if still applicable."""
        self._hires_timer.stop()
        had = self._hires is not None
        self._hires = None
        if had and refresh and self.pixmap_item is not None:
            self._refresh_item_pixmap()
            self.apply_transformations()

    # --- Slice mode ------------------------------------------------------
    # Splits one scan containing several photos into separate images. Moving
    # the mouse near the top/bottom rim arms a VERTICAL cut line at the
    # cursor's x; moving along the left/right rim arms a HORIZONTAL cut at
    # the cursor's y. Click places the line (then drag to fine-tune,
    # right-click to delete), Enter performs the slicing.

    SLICE_EDGE_BAND_PX = 60.0   # rim proximity that arms a ghost line (view px)
    SLICE_GRAB_PX = 10.0        # grab tolerance around a placed line (view px)

    def enter_slice_mode(self) -> bool:
        """Show the whole image at the fitted view and start placing cuts."""
        if self.current_idx is None:
            return False
        img_obj = ccr_backend.get_image_by_index(self.current_idx)
        if img_obj is None or self.current_pixmap is None or self.current_pixmap.isNull():
            return False
        ci = getattr(img_obj, "conversion_inputs", None)
        if ci is not None and ci.get("mode") == "bw" and ci.get("fine_rot"):
            # The B/W-point conversion baked this fine rotation into the
            # preview pixels, so cut lines placed here would land offset in
            # the source file. Steer to the supported workflow instead.
            try:
                self.parent().parent().sliders_panel.set_temporary_hint(
                    "This image's B/W conversion has a fine rotation baked "
                    "in — <b>Un-convert</b> first (or slice before "
                    "converting) to slice it accurately.", duration=8000)
            except AttributeError:
                pass
            return False
        if self.crop_mode:
            self._exit_crop_mode()
        if self.area_mode:
            self._exit_area_mode()
        self.view.bwpoint_mode = None
        self.view.wb_pick_mode = False
        self.slice_mode = True
        self._slice_lines = []
        self._slice_drag = None
        # Slice mode always starts at the fitted view of the entire image
        self._zoom = 1.0
        self._release_hires(refresh=False)
        self._slice_rerender = True
        try:
            self.update_preview(self.current_idx)  # re-render full image
        finally:
            self._slice_rerender = False
        self._draw_slice_dim()
        self.view.setCursor(Qt.CrossCursor)
        self._sync_zoom_combo()
        return True

    def cancel_slice_mode(self):
        """Esc / toggling Slice: leave without slicing."""
        if not self.slice_mode:
            return
        idx = self.current_idx
        self._exit_slice_mode()
        if idx is not None:
            self.update_preview(idx)

    def _exit_slice_mode(self):
        self.slice_mode = False
        self._slice_drag = None
        self._teardown_slice_items()
        self.view.setCursor(Qt.ArrowCursor)

    def _slice_display_transform(self):
        """Transform the PIXMAP and dim cast are displayed with in slice
        mode: coarse flips/rotation PLUS the fine rotation (which stays
        visible — the rotation gets baked into the slices)."""
        t = self._base_transform()
        if t is None or self.current_pixmap is None:
            return None
        if self.current_fine_rotation:
            cx = self.current_pixmap.width() / 2
            cy = self.current_pixmap.height() / 2
            t = QTransform(t)
            t.translate(cx, cy)
            t.rotate(self.current_fine_rotation / 100.0)
            t.translate(-cx, -cy)
        return t if t.isInvertible() else None

    def _slice_local(self, scene_pos):
        """Map a scene point into the WARPED-CANVAS frame — the frame the
        backend actually cuts. The backend rotates the decode about its
        center into the same WxH canvas (identical to the on-screen warp),
        so canvas coordinates relate to the scene through the BASE transform
        only; cut fractions must be measured here, NOT in un-rotated content
        coords, or the cuts would land offset from the placed lines."""
        t = self._base_transform()
        if t is None or self.current_pixmap is None:
            return None
        return t.inverted()[0].map(scene_pos)

    def _slice_line_at(self, p):
        """Placed line near local point p, or None."""
        if self.current_pixmap is None:
            return None
        tol = self.SLICE_GRAB_PX / self._view_scale()
        w, h = self.current_pixmap.width(), self.current_pixmap.height()
        best, best_d = None, None
        for line in self._slice_lines:
            if line["orient"] == 'v':
                d = abs(p.x() - line["frac"] * w)
            else:
                d = abs(p.y() - line["frac"] * h)
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = line, d
        return best

    def _slice_ghost_spec(self, p):
        """('v'|'h', frac) for a ghost line when the cursor is near a rim,
        else None. Near top/bottom -> vertical cut; near left/right ->
        horizontal cut. (In image-local coords, so a 90-degree-rotated view
        still behaves as the user sees it.)"""
        w, h = self.current_pixmap.width(), self.current_pixmap.height()
        band = self.SLICE_EDGE_BAND_PX / self._view_scale()
        if not (-band <= p.x() <= w + band and -band <= p.y() <= h + band):
            return None
        x = min(max(p.x(), 0.0), float(w))
        y = min(max(p.y(), 0.0), float(h))
        dist_tb = min(abs(p.y()), abs(h - p.y()))   # to top/bottom rim
        dist_lr = min(abs(p.x()), abs(w - p.x()))   # to left/right rim
        if dist_tb > band and dist_lr > band:
            return None
        if dist_tb <= dist_lr:
            return ('v', min(max(x / w, 0.0), 1.0))
        return ('h', min(max(y / h, 0.0), 1.0))

    def slice_move(self, scene_pos):
        p = self._slice_local(scene_pos)
        if p is None:
            return
        if self._slice_drag is not None:
            w, h = self.current_pixmap.width(), self.current_pixmap.height()
            line = self._slice_drag
            # Clamp matches the backend's keep-range (_clean_slice_cuts), so
            # a visibly placed line can never be silently discarded at Enter.
            if line["orient"] == 'v':
                line["frac"] = min(max(p.x() / w, 0.01), 0.99)
            else:
                line["frac"] = min(max(p.y() / h, 0.01), 0.99)
            self._redraw_slice_lines()
            return
        # Hover: grab cursor near an existing line, else rim ghost line
        near = self._slice_line_at(p)
        if near is not None:
            self._set_slice_ghost(None)
            # The grab cursor is screen-space: a 90/270-degree view rotation
            # swaps which way the line runs on screen.
            screen_vertical = ((near["orient"] == 'v')
                               != (self.current_rotation % 180 == 90))
            self.view.setCursor(Qt.SizeHorCursor if screen_vertical
                                else Qt.SizeVerCursor)
            return
        self.view.setCursor(Qt.CrossCursor)
        self._set_slice_ghost(self._slice_ghost_spec(p))

    def slice_press(self, scene_pos):
        p = self._slice_local(scene_pos)
        if p is None:
            return
        near = self._slice_line_at(p)
        if near is not None:
            self._slice_drag = near
            return
        spec = self._slice_ghost_spec(p)
        if spec is not None:
            orient, frac = spec
            line = {"orient": orient, "frac": min(max(frac, 0.01), 0.99),
                    "item": None}
            self._slice_lines.append(line)
            self._slice_drag = line   # keep dragging while the button is held
            self._set_slice_ghost(None)
            self._redraw_slice_lines()

    def slice_release(self):
        self._slice_drag = None

    def slice_delete_at(self, scene_pos):
        p = self._slice_local(scene_pos)
        if p is None:
            return
        line = self._slice_line_at(p)
        if line is not None:
            self._slice_lines.remove(line)
            if line["item"] is not None:
                try:
                    self.scene.removeItem(line["item"])
                except RuntimeError:
                    pass
            self._redraw_slice_lines()

    def _make_slice_line_item(self, orient, frac, ghost=False):
        w, h = self.current_pixmap.width(), self.current_pixmap.height()
        if orient == 'v':
            item = QGraphicsLineItem(frac * w, 0, frac * w, h)
        else:
            item = QGraphicsLineItem(0, frac * h, w, frac * h)
        pen = QPen(QColor(0, 200, 255, 150 if ghost else 235), 2,
                   Qt.DashLine if ghost else Qt.SolidLine)
        pen.setCosmetic(True)  # constant on-screen width at any zoom
        item.setPen(pen)
        # Lines live in the warped-canvas frame (base transform only): they
        # render screen-straight, exactly where the backend will cut.
        t = self._base_transform()
        if t is not None:
            item.setTransform(t)
        self.scene.addItem(item)
        return item

    def _draw_slice_dim(self):
        """Darker cast over the whole image while in slice mode, so the mode
        is unmistakable; cut lines render on top of it."""
        if self._slice_dim_item is not None:
            try:
                self.scene.removeItem(self._slice_dim_item)
            except RuntimeError:
                pass
            self._slice_dim_item = None
        if not self.slice_mode or self.current_pixmap is None:
            return
        rect = QGraphicsRectItem(QRectF(0, 0, self.current_pixmap.width(),
                                        self.current_pixmap.height()))
        rect.setBrush(QBrush(QColor(0, 0, 0, 80)))
        rect.setPen(QPen(Qt.NoPen))
        t = self._slice_display_transform()
        if t is not None:
            rect.setTransform(t)
        self.scene.addItem(rect)
        self._slice_dim_item = rect

    def _set_slice_ghost(self, spec):
        if self._slice_ghost_item is not None:
            try:
                self.scene.removeItem(self._slice_ghost_item)
            except RuntimeError:
                pass
            self._slice_ghost_item = None
        if spec is None or self.current_pixmap is None:
            return
        self._slice_ghost_item = self._make_slice_line_item(spec[0], spec[1], ghost=True)

    def _redraw_slice_lines(self):
        for line in self._slice_lines:
            if line["item"] is not None:
                try:
                    self.scene.removeItem(line["item"])
                except RuntimeError:
                    pass
            line["item"] = self._make_slice_line_item(line["orient"], line["frac"])

    def _teardown_slice_items(self):
        self._set_slice_ghost(None)
        if self._slice_dim_item is not None:
            try:
                self.scene.removeItem(self._slice_dim_item)
            except RuntimeError:
                pass
            self._slice_dim_item = None
        for line in self._slice_lines:
            if line["item"] is not None:
                try:
                    self.scene.removeItem(line["item"])
                except RuntimeError:
                    pass
            line["item"] = None
        self._slice_lines = []

    def confirm_slices(self):
        """Enter pressed in slice mode: split the image along the placed
        lines. The slices replace the original in the list, each reading its
        own region of the source file at full quality."""
        if not self.slice_mode:
            return
        x_cuts = [l["frac"] for l in self._slice_lines if l["orient"] == 'v']
        y_cuts = [l["frac"] for l in self._slice_lines if l["orient"] == 'h']
        idx = self.current_idx
        self._exit_slice_mode()
        if (not x_cuts and not y_cuts) or idx is None:
            if idx is not None:
                self.update_preview(idx)
            try:
                self.parent().parent().sliders_panel.set_temporary_hint(
                    "No slice lines placed — nothing to slice.", duration=3000)
            except AttributeError:
                pass
            return
        dialog = SliceProgressDialog(self)
        self._slice_worker = SliceWorker(idx, x_cuts, y_cuts)
        self._slice_worker.progress.connect(dialog.set_progress)
        self._slice_worker.finished_slice.connect(
            lambda count: self._on_slice_done(dialog, idx, count))
        self._slice_worker.start()
        dialog.exec_()

    def _on_slice_done(self, dialog, idx, count):
        dialog.accept()
        mw = self.parent().parent()
        if count > 0:
            # Rebuild the whole thumbnail list (the image count changed) and
            # land on the first slice.
            mw.thumbnail_list.load_thumbnails()
            if idx < ccr_backend.get_image_count():
                mw.thumbnail_list.thumbnail_list.setCurrentRow(idx)
            mw.sliders_panel.set_temporary_hint(
                f"Sliced into {count} images. Each can now be framed, "
                f"converted, and exported separately.", duration=6000)
            ccr_backend.save_catalog()
        else:
            self.update_preview(idx)
            mw.sliders_panel.set_temporary_hint(
                "Slicing did not produce multiple pieces (lines too close "
                "to the edge?).", duration=5000)

    # --- Crop mode -----------------------------------------------------

    def map_displayed_to_full(self, x, y):
        """Map a point from displayed-pixmap coords to full-image coords.
        Identity unless a confirmed crop is currently being displayed."""
        if self._crop_display_transform is None:
            return x, y
        p = self._crop_display_transform.map(QPointF(x, y))
        return p.x(), p.y()

    @staticmethod
    def _extract_rotated_crop(pixmap, crop, angle):
        """Render the content under a rotated crop box into an upright pixmap.
        Returns (pixmap, displayed->full QTransform), or None when degenerate.
        Display twin of apply_crop_to_image's rotated branch: displayed pixel
        (u, v) shows the source at C + R(angle) . (u - bw/2, v - bh/2)."""
        w, h = pixmap.width(), pixmap.height()
        bw = int(round((crop[2] - crop[0]) * w))
        bh = int(round((crop[3] - crop[1]) * h))
        if bw < 2 or bh < 2 or w <= 0 or h <= 0:
            return None
        cx = (crop[0] + crop[2]) / 2.0 * w
        cy = (crop[1] + crop[3]) / 2.0 * h
        to_full = QTransform()
        to_full.translate(cx, cy)
        to_full.rotate(angle)
        to_full.translate(-bw / 2.0, -bh / 2.0)
        inverted, ok = to_full.inverted()
        if not ok:
            return None
        out = QPixmap(bw, bh)
        out.fill(QColor(0, 0, 0))
        painter = QPainter(out)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setTransform(inverted)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()
        return out, to_full

    @staticmethod
    def _crop_rect_to_pixels(crop, pixmap) -> "QRect | None":
        """Convert a normalized (x1, y1, x2, y2) crop to a QRect in pixmap
        pixels, or None when the rect is degenerate."""
        w, h = pixmap.width(), pixmap.height()
        if w <= 0 or h <= 0:
            return None
        x1 = max(0, min(w - 1, int(round(crop[0] * w))))
        y1 = max(0, min(h - 1, int(round(crop[1] * h))))
        x2 = max(x1 + 1, min(w, int(round(crop[2] * w))))
        y2 = max(y1 + 1, min(h, int(round(crop[3] * h))))
        if (x2 - x1) < 2 or (y2 - y1) < 2:
            return None
        return QRect(x1, y1, x2 - x1, y2 - y1)

    def _base_transform(self):
        """Coarse flip/rotation transform around the displayed pixmap center
        (same construction the reference-frame and B/W tools use)."""
        if self.current_pixmap is None:
            return None
        cx = self.current_pixmap.width() / 2
        cy = self.current_pixmap.height() / 2
        t = QTransform()
        t.translate(cx, cy)
        if self.current_vertical_flip:
            t.scale(1, -1)
        if self.current_horizontal_flip:
            t.scale(-1, 1)
        if self.current_rotation:
            t.rotate(self.current_rotation)
        t.translate(-cx, -cy)
        return t if t.isInvertible() else None

    def enter_crop_mode(self) -> bool:
        """Show the full image with the current crop (if any) as the starting
        selection, so the user can adjust or undo a previous crop."""
        if self.current_idx is None:
            return False
        img_obj = ccr_backend.get_image_by_index(self.current_idx)
        if img_obj is None or self.current_pixmap is None or self.current_pixmap.isNull():
            return False
        # Crop mode replaces any other interactive pick mode
        if self.slice_mode:
            self._exit_slice_mode()
        if self.area_mode:
            self._exit_area_mode()
        self.view.bwpoint_mode = None
        self.view.wb_pick_mode = False
        self.crop_mode = True
        self._pending_crop_local = None
        self._pending_crop_angle = 0.0
        self._crop_drag = None
        # Entering crop always shows the entire image at the fitted view, so
        # the whole frame (and the crop box over it) is visible.
        self._zoom = 1.0
        self._release_hires(refresh=False)
        self._crop_rerender = True
        try:
            self.update_preview(self.current_idx)  # re-render un-cropped
        finally:
            self._crop_rerender = False
        self._sync_zoom_combo()
        crop = getattr(img_obj, "crop_rect", None)
        if crop is not None and self.current_pixmap is not None:
            w, h = self.current_pixmap.width(), self.current_pixmap.height()
            self._pending_crop_local = QRectF(
                crop[0] * w, crop[1] * h,
                (crop[2] - crop[0]) * w, (crop[3] - crop[1]) * h)
            self._pending_crop_angle = getattr(img_obj, "crop_angle", 0.0) or 0.0
        self._draw_crop_overlay()
        self.view.setCursor(Qt.CrossCursor)
        return True

    # ---- Crop drag interaction: new rect + 10 control handles -----------
    # 4 corners (resize), 4 edge midpoints (move that edge), a rotate knob
    # above the top edge, and a center handle (move the whole box).

    HANDLE_VIEW_PX = 14.0          # hit tolerance, in view pixels
    HANDLE_DRAW_PX = 9.0           # handle glyph size, in view pixels
    ROTATE_HANDLE_VIEW_PX = 28.0   # rotate-knob stem length, in view pixels
    _CORNER_SIGNS = {"tl": (-1, -1), "tr": (1, -1), "bl": (-1, 1), "br": (1, 1)}

    def _view_scale(self):
        m = self.view.transform().m11()
        return abs(m) if m else 1.0

    def _box_transform(self, sel):
        """Rotation of the crop box about its own center (image coords)."""
        c = sel.center()
        t = QTransform()
        t.translate(c.x(), c.y())
        t.rotate(self._pending_crop_angle)
        t.translate(-c.x(), -c.y())
        return t

    def _crop_handle_at(self, p):
        """Return the handle name under local image point p, or None."""
        sel = self._pending_crop_local
        if sel is None or sel.width() < 2 or sel.height() < 2:
            return None
        c = sel.center()
        unrot = QTransform()
        unrot.rotate(-self._pending_crop_angle)
        q = unrot.map(QPointF(p.x() - c.x(), p.y() - c.y()))
        hw, hh = sel.width() / 2.0, sel.height() / 2.0
        tol = self.HANDLE_VIEW_PX / self._view_scale()
        rot_off = self.ROTATE_HANDLE_VIEW_PX / self._view_scale()
        # Corners first (most specific), then the rotate knob, edges, center.
        handles = [
            ("corner-tl", -hw, -hh), ("corner-tr", hw, -hh),
            ("corner-bl", -hw, hh), ("corner-br", hw, hh),
            ("rotate", 0.0, -hh - rot_off),
            ("edge-t", 0.0, -hh), ("edge-b", 0.0, hh),
            ("edge-l", -hw, 0.0), ("edge-r", hw, 0.0),
            ("move", 0.0, 0.0),
        ]
        # Pick the NEAREST handle among all within tolerance — on small boxes
        # the zones overlap, and first-match priority would shadow the center
        # move handle behind corners/edges.
        best = None
        best_d2 = None
        for name, hx, hy in handles:
            dx, dy = q.x() - hx, q.y() - hy
            if abs(dx) <= tol and abs(dy) <= tol:
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best, best_d2 = name, d2
        return best

    # Base orientation (deg) of each handle's resize axis on an un-rotated box
    _HANDLE_BASE_ANGLE = {
        "edge-l": 0.0, "edge-r": 0.0, "edge-t": 90.0, "edge-b": 90.0,
        "corner-tl": 45.0, "corner-br": 45.0, "corner-tr": 135.0, "corner-bl": 135.0,
    }

    def _cursor_for_handle(self, handle):
        """Cursor that indicates the transform a handle performs, accounting
        for the box's current rotation."""
        if handle is None:
            return Qt.CrossCursor
        if handle == "move":
            return Qt.SizeAllCursor
        if handle == "rotate":
            return _rotate_cursor()
        base = self._HANDLE_BASE_ANGLE.get(handle)
        if base is None:
            return Qt.CrossCursor
        a = (base + self._pending_crop_angle) % 180.0
        if a < 22.5:
            return Qt.SizeHorCursor       # ↔
        elif a < 67.5:
            return Qt.SizeFDiagCursor     # ⤡  (\)
        elif a < 112.5:
            return Qt.SizeVerCursor       # ↕
        elif a < 157.5:
            return Qt.SizeBDiagCursor     # ⤢  (/)
        return Qt.SizeHorCursor

    def update_crop_hover_cursor(self, scene_pos):
        """Set the cursor based on the handle under the (un-dragged) cursor."""
        base = self._base_transform()
        if base is None or self._pending_crop_local is None:
            self.view.setCursor(Qt.CrossCursor)
            return
        p = base.inverted()[0].map(scene_pos)
        self.view.setCursor(self._cursor_for_handle(self._crop_handle_at(p)))

    def begin_crop_drag(self, scene_pos):
        base = self._base_transform()
        if base is None or self.current_pixmap is None:
            return
        p = base.inverted()[0].map(scene_pos)
        mode = self._crop_handle_at(p) or "new"
        sel = self._pending_crop_local
        self._crop_drag = {
            "mode": mode,
            "start": QPointF(p),
            "box0": QRectF(sel) if sel is not None else None,
            "angle0": self._pending_crop_angle,
        }
        if mode != "new":
            self.view.setCursor(self._cursor_for_handle(mode))
        self.update_crop_drag(scene_pos)

    def update_crop_drag(self, scene_pos):
        drag = self._crop_drag
        if drag is None or self.current_pixmap is None:
            return
        base = self._base_transform()
        if base is None:
            return
        p = base.inverted()[0].map(scene_pos)
        mode = drag["mode"]
        if mode == "new":
            self._update_new_selection(drag["start"], p)
        elif mode == "move":
            self._drag_move_box(drag, p)
        elif mode == "rotate":
            self._drag_rotate_box(drag, p)
        elif mode.startswith("corner-"):
            self._drag_corner(drag, p, mode[len("corner-"):])
        elif mode.startswith("edge-"):
            self._drag_edge(drag, p, mode[len("edge-"):])
        self._draw_crop_overlay()

    def end_crop_drag(self, scene_pos):
        if self._crop_drag is None:
            return
        self.update_crop_drag(scene_pos)
        self._crop_drag = None

    def _update_new_selection(self, p0, p1):
        w, h = self.current_pixmap.width(), self.current_pixmap.height()
        x1 = max(0.0, min(float(w), min(p0.x(), p1.x())))
        y1 = max(0.0, min(float(h), min(p0.y(), p1.y())))
        x2 = max(0.0, min(float(w), max(p0.x(), p1.x())))
        y2 = max(0.0, min(float(h), max(p0.y(), p1.y())))
        if (x2 - x1) < 3 and (y2 - y1) < 3:
            # Stray click, not a drag — keep the current selection (and its
            # angle) so the display keeps matching what Enter will confirm.
            return
        # A freshly drawn box starts axis-aligned
        self._pending_crop_local = QRectF(x1, y1, x2 - x1, y2 - y1)
        self._pending_crop_angle = 0.0

    def _drag_move_box(self, drag, p):
        box0 = drag["box0"]
        if box0 is None:
            return
        delta = p - drag["start"]
        if abs(delta.x()) < 1e-9 and abs(delta.y()) < 1e-9:
            # A plain click must be a no-op: if the box center sits outside
            # the image (possible after a big corner drag), the clamp below
            # would otherwise yank the box inward without any mouse motion.
            self._pending_crop_local = QRectF(box0)
            return
        w, h = self.current_pixmap.width(), self.current_pixmap.height()
        c = box0.center() + delta
        box = QRectF(box0)
        box.moveCenter(QPointF(min(max(c.x(), 0.0), float(w)),
                               min(max(c.y(), 0.0), float(h))))
        self._pending_crop_local = box

    def _drag_rotate_box(self, drag, p):
        box0 = drag["box0"]
        if box0 is None:
            return
        c = box0.center()
        v0 = drag["start"] - c
        v1 = p - c
        if (abs(v1.x()) + abs(v1.y())) < 1e-6:
            return
        a0 = math.degrees(math.atan2(v0.y(), v0.x()))
        a1 = math.degrees(math.atan2(v1.y(), v1.x()))
        angle = drag["angle0"] + (a1 - a0)
        angle = (angle + 180.0) % 360.0 - 180.0
        if abs(angle) < 0.75:
            angle = 0.0  # snap to straight
        self._pending_crop_angle = angle
        self._pending_crop_local = QRectF(box0)

    def _drag_corner(self, drag, p, which):
        box0 = drag["box0"]
        if box0 is None:
            return
        sx, sy = self._CORNER_SIGNS[which]
        angle = drag["angle0"]
        fwd = QTransform()
        fwd.rotate(angle)
        unrot = QTransform()
        unrot.rotate(-angle)
        hw0, hh0 = box0.width() / 2.0, box0.height() / 2.0
        # The opposite corner stays fixed in image space
        anchor = box0.center() + fwd.map(QPointF(-sx * hw0, -sy * hh0))
        d = unrot.map(p - anchor)  # dragged corner in box frame, rel. anchor
        min_sz = 10.0
        dx = sx * max(sx * d.x(), min_sz)
        dy = sy * max(sy * d.y(), min_sz)
        c = anchor + fwd.map(QPointF(dx / 2.0, dy / 2.0))
        bw, bh = abs(dx), abs(dy)
        self._pending_crop_local = QRectF(c.x() - bw / 2.0, c.y() - bh / 2.0, bw, bh)

    def _drag_edge(self, drag, p, which):
        box0 = drag["box0"]
        if box0 is None:
            return
        angle = drag["angle0"]
        fwd = QTransform()
        fwd.rotate(angle)
        unrot = QTransform()
        unrot.rotate(-angle)
        c0 = box0.center()
        hw0, hh0 = box0.width() / 2.0, box0.height() / 2.0
        d = unrot.map(p - c0)  # mouse in box frame, rel. the old center
        min_sz = 10.0
        left, right, top, bottom = -hw0, hw0, -hh0, hh0
        if which == "r":
            right = max(d.x(), left + min_sz)
        elif which == "l":
            left = min(d.x(), right - min_sz)
        elif which == "b":
            bottom = max(d.y(), top + min_sz)
        elif which == "t":
            top = min(d.y(), bottom - min_sz)
        c = c0 + fwd.map(QPointF((left + right) / 2.0, (top + bottom) / 2.0))
        bw, bh = right - left, bottom - top
        self._pending_crop_local = QRectF(c.x() - bw / 2.0, c.y() - bh / 2.0, bw, bh)

    def _remove_crop_overlay_items(self):
        for item in self._crop_handle_items:
            try:
                self.scene.removeItem(item)
            except RuntimeError:
                pass
        self._crop_handle_items = []
        if self._crop_overlay_item is not None:
            try:
                self.scene.removeItem(self._crop_overlay_item)
            except RuntimeError:
                pass
            self._crop_overlay_item = None

    def _draw_crop_overlay(self):
        """Dim everything outside the pending crop box (so both the entire
        image and the crop target stay visible) and draw its 10 control
        handles: 4 corners, 4 edge midpoints, a rotate knob sticking out on
        top, and a center move handle."""
        self._remove_crop_overlay_items()
        if not self.crop_mode or self.current_pixmap is None:
            return
        base = self._base_transform()
        if base is None:
            return
        sel = self._pending_crop_local
        if sel is None or sel.width() < 2 or sel.height() < 2:
            # No box drawn yet: dim the whole canvas so crop mode reads as
            # active immediately (matching slice mode). The clear hole and
            # handles appear as soon as the user starts a selection.
            dim = QGraphicsRectItem(QRectF(0, 0, self.current_pixmap.width(),
                                           self.current_pixmap.height()))
            dim.setBrush(QBrush(QColor(0, 0, 0, 110)))
            dim.setPen(QPen(Qt.NoPen))
            dim.setTransform(base)
            self.scene.addItem(dim)
            self._crop_overlay_item = dim
            return
        box_t = self._box_transform(sel)
        combined = box_t * base  # box rotation first, then coarse flips/rotation

        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        path.addRect(QRectF(0, 0, self.current_pixmap.width(), self.current_pixmap.height()))
        path.addPolygon(box_t.map(QPolygonF(sel)))
        overlay = QGraphicsPathItem(path)
        overlay.setBrush(QBrush(QColor(0, 0, 0, 110)))
        overlay.setPen(QPen(QColor(255, 255, 255, 220), 2, Qt.DashLine))
        overlay.setTransform(base)
        self.scene.addItem(overlay)
        self._crop_overlay_item = overlay

        # Handles are laid out in un-rotated box coords and carry the
        # combined transform, so they ride along with the box rotation.
        scale = self._view_scale()
        hs = self.HANDLE_DRAW_PX / scale
        rot_off = self.ROTATE_HANDLE_VIEW_PX / scale
        hw, hh = sel.width() / 2.0, sel.height() / 2.0
        c = sel.center()
        handle_pen = QPen(QColor(30, 30, 30, 230), max(1.0 / scale, 0.5))
        handle_brush = QBrush(QColor(255, 255, 255, 235))

        def add_handle(item):
            item.setPen(handle_pen)
            item.setBrush(handle_brush)
            item.setTransform(combined)
            self.scene.addItem(item)
            self._crop_handle_items.append(item)

        # 4 corners + 4 edge midpoints (squares)
        for hx, hy in ((-hw, -hh), (hw, -hh), (-hw, hh), (hw, hh),
                       (0.0, -hh), (0.0, hh), (-hw, 0.0), (hw, 0.0)):
            x, y = c.x() + hx, c.y() + hy
            add_handle(QGraphicsRectItem(QRectF(x - hs / 2, y - hs / 2, hs, hs)))

        # Rotate knob: stem + circle sticking out above the top edge
        stem = QGraphicsLineItem(c.x(), c.y() - hh, c.x(), c.y() - hh - rot_off)
        stem.setPen(QPen(QColor(255, 255, 255, 220), max(1.5 / scale, 0.5)))
        stem.setTransform(combined)
        self.scene.addItem(stem)
        self._crop_handle_items.append(stem)
        add_handle(QGraphicsEllipseItem(
            QRectF(c.x() - hs / 2, c.y() - hh - rot_off - hs / 2, hs, hs)))

        # Center (move) handle (circle)
        add_handle(QGraphicsEllipseItem(QRectF(c.x() - hs / 2, c.y() - hs / 2, hs, hs)))

    @staticmethod
    def _crops_equal(a, b, w, h):
        """Sub-pixel-tolerant comparison of normalized crop rects, so an
        unchanged selection round-tripped through pixel coords does not
        register as a change (and push a junk undo snapshot)."""
        if a is None or b is None:
            return a is b
        tol_x = 0.5 / max(w, 1)
        tol_y = 0.5 / max(h, 1)
        return (abs(a[0] - b[0]) < tol_x and abs(a[2] - b[2]) < tol_x
                and abs(a[1] - b[1]) < tol_y and abs(a[3] - b[3]) < tol_y)

    def confirm_crop(self):
        """Enter pressed in crop mode: store the selection as the new crop.
        Display-level only — the conversion result is untouched."""
        if not self.crop_mode:
            return
        img_obj = ccr_backend.get_image_by_index(self.current_idx)
        sel = self._pending_crop_local
        applied = False
        cleared = False
        too_small = False
        if img_obj is not None and sel is not None and self.current_pixmap is not None:
            w, h = self.current_pixmap.width(), self.current_pixmap.height()
            angle = self._pending_crop_angle
            if abs(angle) < 0.05:
                angle = 0.0
                # Without rotation, anything outside the image is empty —
                # commit only the visible part of the box.
                sel = sel.intersected(QRectF(0, 0, w, h))
            if sel.width() >= 10 and sel.height() >= 10:
                new_crop = (sel.left() / w, sel.top() / h,
                            sel.right() / w, sel.bottom() / h)
                # An un-rotated selection of (almost) the whole image clears the crop
                if (angle == 0.0
                        and new_crop[0] <= 0.001 and new_crop[1] <= 0.001
                        and new_crop[2] >= 0.999 and new_crop[3] >= 0.999):
                    new_crop = None
                old_angle = getattr(img_obj, "crop_angle", 0.0) or 0.0
                if (not self._crops_equal(new_crop, img_obj.crop_rect, w, h)
                        or abs(angle - old_angle) >= 0.05):
                    img_obj.push_undo_state()
                    img_obj.crop_rect = new_crop
                    img_obj.crop_angle = 0.0 if new_crop is None else angle
                    applied = True
                    cleared = new_crop is None
            else:
                too_small = True
        self._exit_crop_mode()
        if applied and img_obj is not None:
            # The kept pixel region changed, so the histogram (computed over
            # the cropped area) is stale — regenerate it before the redraw.
            img_obj.update_thumbnail_and_preview()
        self.update_preview(self.current_idx)
        if cleared:
            hint = "Crop cleared. Ctrl+Z to undo."
        elif applied:
            hint = "Crop applied. Ctrl+Z to undo, or press Crop again to adjust."
        elif too_small:
            hint = "Selection too small — crop unchanged."
        else:
            hint = "Crop unchanged."
        try:
            self.parent().parent().sliders_panel.set_temporary_hint(hint, duration=4000)
        except AttributeError:
            pass

    def cancel_crop_mode(self):
        """Esc pressed in crop mode: leave without changing the crop."""
        if not self.crop_mode:
            return
        self._exit_crop_mode()
        self.update_preview(self.current_idx)

    def clear_crop(self):
        """Right-click in crop mode: remove the crop entirely."""
        img_obj = ccr_backend.get_image_by_index(self.current_idx)
        cleared = False
        if img_obj is not None and (img_obj.crop_rect is not None
                                    or getattr(img_obj, "crop_angle", 0.0)):
            img_obj.push_undo_state()
            img_obj.crop_rect = None
            img_obj.crop_angle = 0.0
            cleared = True
        self._exit_crop_mode()
        if cleared:
            # Histogram is computed over the cropped region — regenerate it now
            # that the whole image is kept again.
            img_obj.update_thumbnail_and_preview()
        self.update_preview(self.current_idx)

    def _exit_crop_mode(self):
        self.crop_mode = False
        self._pending_crop_local = None
        self._pending_crop_angle = 0.0
        self._crop_drag = None
        self._remove_crop_overlay_items()
        self.view.setCursor(Qt.ArrowCursor)

    # ===== Area editing (local masked adjustment layers) ==================
    # Mirrors the crop tool: a scene-item overlay with constant-size handles,
    # hit-tested on the un-rotated box frame, dragged via a state dict (no
    # grabMouse). Geometry is stored NORMALIZED on the area; area mode shows
    # the full un-cropped image so geometry maps directly through _base_transform.

    def _sliders_panel(self):
        try:
            return self.parent().parent().sliders_panel
        except AttributeError:
            return None

    def _area_hint(self, msg, duration=5000):
        panel = self._sliders_panel()
        if panel is not None:
            try:
                panel.set_temporary_hint(msg, duration=duration)
            except AttributeError:
                pass

    def _area_active(self):
        """The area dict currently being edited (active layer), or None."""
        if self.current_idx is None:
            return None
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None:
            return None
        return img.get_area(img.active_area_id)

    def add_area(self, kind):
        """Toolbar handler: create a local area of `kind` ('circle'|'gradient'),
        select it, and enter area-edit mode. Requires a converted image."""
        if self.current_idx is None:
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None:
            return
        if not getattr(img, "converted", False):
            self._area_hint("Convert the image before adding a local area.")
            return
        aid = ccr_backend.add_area_by_index(self.current_idx, kind)
        if aid is None:
            return
        panel = self._sliders_panel()
        if panel is not None and hasattr(panel, "refresh_layers"):
            panel.refresh_layers(self.current_idx)
        self.enter_area_mode()
        self._area_hint(
            "Local area added. Drag it on the canvas to move/resize/rotate; "
            "edit its adjustments in the panel. Pick 'Whole Image' (or Esc) "
            "to finish.")

    def enter_area_mode(self) -> bool:
        """Show the full un-cropped image with the active area's overlay."""
        if self.current_idx is None:
            return False
        img = ccr_backend.get_image_by_index(self.current_idx)
        if (img is None or self.current_pixmap is None
                or self.current_pixmap.isNull() or img.active_area_id is None):
            return False
        # Mutually exclusive with the other interaction modes.
        if self.crop_mode:
            self._exit_crop_mode()
        if self.slice_mode:
            self._exit_slice_mode()
        self.view.bwpoint_mode = None
        self.view.wb_pick_mode = False
        self.area_mode = True
        self._area_drag = None
        self._zoom = 1.0
        self._release_hires(refresh=False)
        self._area_rerender = True
        try:
            self.update_preview(self.current_idx)  # re-render un-cropped
        finally:
            self._area_rerender = False
        self._sync_zoom_combo()
        self._draw_area_overlay()
        self.view.setCursor(Qt.CrossCursor)
        return True

    def _exit_area_mode(self):
        self.area_mode = False
        self._area_drag = None
        self._remove_area_overlay_items()
        self.view.setCursor(Qt.ArrowCursor)

    def exit_area_mode(self):
        """Leave area mode and restore the normal (possibly cropped) display."""
        if not self.area_mode:
            return
        self._exit_area_mode()
        self.update_preview(self.current_idx)

    def _select_whole_image_layer(self):
        panel = self._sliders_panel()
        if panel is not None and hasattr(panel, "_select_layer"):
            panel._select_layer(None)
        elif self.current_idx is not None:
            ccr_backend.set_active_area_by_index(self.current_idx, None)
            self.exit_area_mode()

    def on_active_layer_changed(self, idx):
        """Called by the panel when the active layer changes: show the area's
        overlay, or leave area mode when the Whole Image layer is selected."""
        if idx != self.current_idx:
            return
        img = ccr_backend.get_image_by_index(idx)
        active_id = getattr(img, "active_area_id", None) if img else None
        if active_id is None:
            if self.area_mode:
                self.exit_area_mode()
        elif not self.area_mode:
            self.enter_area_mode()
        else:
            self._draw_area_overlay()

    # ---- geometry <-> pixmap conversions --------------------------------
    def _area_circle_sel(self, area):
        """The ellipse's un-rotated bounding box (QRectF) in pixmap pixels."""
        w, h = self.current_pixmap.width(), self.current_pixmap.height()
        g = area.get("geometry") or {}
        cx = float(g.get("cx", 0.5)) * w
        cy = float(g.get("cy", 0.5)) * h
        rx = float(g.get("rx", 0.25)) * w
        ry = float(g.get("ry", 0.25)) * h
        return QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry)

    def _area_gradient_points(self, area):
        w, h = self.current_pixmap.width(), self.current_pixmap.height()
        g = area.get("geometry") or {}
        p0 = QPointF(float(g.get("x0", 0.3)) * w, float(g.get("y0", 0.5)) * h)
        p1 = QPointF(float(g.get("x1", 0.7)) * w, float(g.get("y1", 0.5)) * h)
        return p0, p1

    @staticmethod
    def _point_seg_dist(p, a, b):
        vx, vy = b.x() - a.x(), b.y() - a.y()
        wx, wy = p.x() - a.x(), p.y() - a.y()
        seg2 = vx * vx + vy * vy
        t = 0.0 if seg2 < 1e-9 else max(0.0, min(1.0, (wx * vx + wy * vy) / seg2))
        dx, dy = a.x() + t * vx - p.x(), a.y() + t * vy - p.y()
        return math.hypot(dx, dy)

    def _area_handle_at(self, p, area):
        """Handle name under local point p for the area's overlay, or None."""
        scale = self._view_scale()
        tol = self.HANDLE_VIEW_PX / scale
        if area.get("kind") == "gradient":
            p0, p1 = self._area_gradient_points(area)
            mid = QPointF((p0.x() + p1.x()) / 2.0, (p0.y() + p1.y()) / 2.0)
            best, best_d2 = None, None
            for name, pt in (("p0", p0), ("p1", p1), ("move", mid)):
                dx, dy = p.x() - pt.x(), p.y() - pt.y()
                if abs(dx) <= tol and abs(dy) <= tol:
                    d2 = dx * dx + dy * dy
                    if best_d2 is None or d2 < best_d2:
                        best, best_d2 = name, d2
            if best is not None:
                return best
            # Clicking on the gradient line itself moves the whole gradient.
            if self._point_seg_dist(p, p0, p1) <= tol:
                return "move"
            return None
        # circle / ellipse
        sel = self._area_circle_sel(area)
        angle = float(area.get("angle", 0.0))
        c = sel.center()
        unrot = QTransform()
        unrot.rotate(-angle)
        q = unrot.map(QPointF(p.x() - c.x(), p.y() - c.y()))
        hw, hh = sel.width() / 2.0, sel.height() / 2.0
        rot_off = self.ROTATE_HANDLE_VIEW_PX / scale
        handles = [
            ("corner-tl", -hw, -hh), ("corner-tr", hw, -hh),
            ("corner-bl", -hw, hh), ("corner-br", hw, hh),
            ("rotate", 0.0, -hh - rot_off),
            ("edge-t", 0.0, -hh), ("edge-b", 0.0, hh),
            ("edge-l", -hw, 0.0), ("edge-r", hw, 0.0),
            ("move", 0.0, 0.0),
        ]
        best, best_d2 = None, None
        for name, hx, hy in handles:
            dx, dy = q.x() - hx, q.y() - hy
            if abs(dx) <= tol and abs(dy) <= tol:
                d2 = dx * dx + dy * dy
                if best_d2 is None or d2 < best_d2:
                    best, best_d2 = name, d2
        if best is None:
            # Inside the ellipse body -> move.
            if (q.x() / max(hw, 1e-6)) ** 2 + (q.y() / max(hh, 1e-6)) ** 2 <= 1.0:
                return "move"
        return best

    def _area_cursor_for_handle(self, handle, angle):
        if handle is None:
            return Qt.CrossCursor
        if handle in ("move", "p0", "p1"):
            return Qt.SizeAllCursor
        if handle == "rotate":
            return _rotate_cursor()
        base = self._HANDLE_BASE_ANGLE.get(handle)
        if base is None:
            return Qt.CrossCursor
        a = (base + angle) % 180.0
        if a < 22.5:
            return Qt.SizeHorCursor
        elif a < 67.5:
            return Qt.SizeFDiagCursor
        elif a < 112.5:
            return Qt.SizeVerCursor
        elif a < 157.5:
            return Qt.SizeBDiagCursor
        return Qt.SizeHorCursor

    def update_area_hover_cursor(self, scene_pos):
        base = self._base_transform()
        area = self._area_active()
        if base is None or area is None:
            self.view.setCursor(Qt.CrossCursor)
            return
        p = base.inverted()[0].map(scene_pos)
        handle = self._area_handle_at(p, area)
        self.view.setCursor(
            self._area_cursor_for_handle(handle, float(area.get("angle", 0.0))))

    # ---- drag interaction ------------------------------------------------
    def begin_area_drag(self, scene_pos):
        base = self._base_transform()
        area = self._area_active()
        if base is None or area is None or self.current_pixmap is None:
            return
        p = base.inverted()[0].map(scene_pos)
        mode = self._area_handle_at(p, area)
        if mode is None:
            self._area_drag = None
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is not None:
            img.push_undo_state()  # one undo step per geometry drag
        self._area_drag = {
            "mode": mode,
            "id": area["id"],
            "kind": area.get("kind", "circle"),
            "start": QPointF(p),
            "geom0": dict(area.get("geometry") or {}),
            "angle0": float(area.get("angle", 0.0)),
        }
        self.view.setCursor(
            self._area_cursor_for_handle(mode, float(area.get("angle", 0.0))))

    def update_area_drag(self, scene_pos):
        drag = self._area_drag
        if drag is None or self.current_pixmap is None:
            return
        base = self._base_transform()
        if base is None:
            return
        p = base.inverted()[0].map(scene_pos)
        if drag["kind"] == "gradient":
            self._area_drag_gradient(drag, p)
        else:
            self._area_drag_circle(drag, p)
        self._draw_area_overlay()
        # Recompute the masked EFFECT live (coalesced) so it follows the drag
        # rather than only snapping into place on release.
        self._area_live_timer.start(40)

    def _area_live_refresh(self):
        """Recompute the masked preview during a drag and swap it under the
        existing overlay — surgical (no scene rebuild / refit), so it stays
        smooth and the handles don't flicker."""
        if not self.area_mode or self.current_idx is None:
            return
        if not (0 <= self.current_idx < len(ccr_backend.images)):
            return
        # When a hi-res (prescaled) pixmap is displayed (zoomed in), a
        # preview-size swap would render at the wrong scale — skip the live
        # swap; end_area_drag does the full, scale-correct refresh on release.
        if self._item_prescale != 1.0:
            return
        img = ccr_backend.images[self.current_idx]
        img.update_thumbnail_and_preview()
        preview = ccr_backend.get_preview_by_index(self.current_idx)
        if (preview is not None and not preview.isNull()
                and self.pixmap_item is not None
                and (self.current_pixmap is None
                     or preview.size() == self.current_pixmap.size())):
            self.current_pixmap = preview
            self.pixmap_item.setPixmap(preview)

    def end_area_drag(self, scene_pos):
        if self._area_drag is None:
            return
        self.update_area_drag(scene_pos)
        self._area_drag = None
        self._area_live_timer.stop()
        # Commit: render the masked effect (heavy) and refresh the thumbnail.
        if self.current_idx is not None and 0 <= self.current_idx < len(ccr_backend.images):
            ccr_backend.images[self.current_idx].update_thumbnail_and_preview()
        self._area_rerender = True
        try:
            self.update_preview(self.current_idx)
        finally:
            self._area_rerender = False
        try:
            self.parent().parent().thumbnail_list.update_thumbnail(self.current_idx)
        except AttributeError:
            pass

    def _resize_box_corner(self, box0, angle, p, which):
        sx, sy = self._CORNER_SIGNS[which]
        fwd = QTransform()
        fwd.rotate(angle)
        unrot = QTransform()
        unrot.rotate(-angle)
        hw0, hh0 = box0.width() / 2.0, box0.height() / 2.0
        anchor = box0.center() + fwd.map(QPointF(-sx * hw0, -sy * hh0))
        d = unrot.map(p - anchor)
        min_sz = 6.0
        dx = sx * max(sx * d.x(), min_sz)
        dy = sy * max(sy * d.y(), min_sz)
        c = anchor + fwd.map(QPointF(dx / 2.0, dy / 2.0))
        bw, bh = abs(dx), abs(dy)
        return QRectF(c.x() - bw / 2.0, c.y() - bh / 2.0, bw, bh)

    def _resize_box_edge(self, box0, angle, p, which):
        fwd = QTransform()
        fwd.rotate(angle)
        unrot = QTransform()
        unrot.rotate(-angle)
        c0 = box0.center()
        hw0, hh0 = box0.width() / 2.0, box0.height() / 2.0
        d = unrot.map(p - c0)
        min_sz = 6.0
        left, right, top, bottom = -hw0, hw0, -hh0, hh0
        if which == "r":
            right = max(d.x(), left + min_sz)
        elif which == "l":
            left = min(d.x(), right - min_sz)
        elif which == "b":
            bottom = max(d.y(), top + min_sz)
        elif which == "t":
            top = min(d.y(), bottom - min_sz)
        c = c0 + fwd.map(QPointF((left + right) / 2.0, (top + bottom) / 2.0))
        bw, bh = right - left, bottom - top
        return QRectF(c.x() - bw / 2.0, c.y() - bh / 2.0, bw, bh)

    def _area_drag_circle(self, drag, p):
        w, h = self.current_pixmap.width(), self.current_pixmap.height()
        g0 = drag["geom0"]
        cx = float(g0.get("cx", 0.5)) * w
        cy = float(g0.get("cy", 0.5)) * h
        rx = float(g0.get("rx", 0.25)) * w
        ry = float(g0.get("ry", 0.25)) * h
        box0 = QRectF(cx - rx, cy - ry, 2 * rx, 2 * ry)
        angle = drag["angle0"]
        mode = drag["mode"]
        sel = QRectF(box0)
        new_angle = angle
        if mode == "move":
            delta = p - drag["start"]
            c = box0.center() + delta
            sel = QRectF(box0)
            sel.moveCenter(QPointF(min(max(c.x(), 0.0), float(w)),
                                   min(max(c.y(), 0.0), float(h))))
        elif mode == "rotate":
            c = box0.center()
            v0 = drag["start"] - c
            v1 = p - c
            if (abs(v1.x()) + abs(v1.y())) >= 1e-6:
                a0 = math.degrees(math.atan2(v0.y(), v0.x()))
                a1 = math.degrees(math.atan2(v1.y(), v1.x()))
                na = angle + (a1 - a0)
                na = (na + 180.0) % 360.0 - 180.0
                if abs(na) < 0.75:
                    na = 0.0
                new_angle = na
        elif mode.startswith("corner-"):
            sel = self._resize_box_corner(box0, angle, p, mode[len("corner-"):])
        elif mode.startswith("edge-"):
            sel = self._resize_box_edge(box0, angle, p, mode[len("edge-"):])
        c = sel.center()
        g = {"cx": c.x() / w, "cy": c.y() / h,
             "rx": (sel.width() / 2.0) / w, "ry": (sel.height() / 2.0) / h}
        ccr_backend.update_area_geometry_by_index(
            self.current_idx, drag["id"], geometry=g, angle=new_angle,
            reprocess=False)

    def _area_drag_gradient(self, drag, p):
        w, h = self.current_pixmap.width(), self.current_pixmap.height()
        g0 = drag["geom0"]
        x0 = float(g0.get("x0", 0.3)) * w
        y0 = float(g0.get("y0", 0.5)) * h
        x1 = float(g0.get("x1", 0.7)) * w
        y1 = float(g0.get("y1", 0.5)) * h
        mode = drag["mode"]
        if mode == "p0":
            x0, y0 = p.x(), p.y()
        elif mode == "p1":
            x1, y1 = p.x(), p.y()
        elif mode == "move":
            delta = p - drag["start"]
            x0 += delta.x(); y0 += delta.y()
            x1 += delta.x(); y1 += delta.y()
        g = {"x0": x0 / w, "y0": y0 / h, "x1": x1 / w, "y1": y1 / h}
        ccr_backend.update_area_geometry_by_index(
            self.current_idx, drag["id"], geometry=g, reprocess=False)

    # ---- overlay drawing -------------------------------------------------
    def _remove_area_overlay_items(self):
        for item in self._area_overlay_items:
            try:
                self.scene.removeItem(item)
            except RuntimeError:
                pass
        self._area_overlay_items = []

    def _draw_area_overlay(self):
        self._remove_area_overlay_items()
        if not self.area_mode or self.current_pixmap is None:
            return
        base = self._base_transform()
        area = self._area_active()
        if base is None or area is None:
            return
        scale = self._view_scale()
        hs = self.HANDLE_DRAW_PX / scale
        handle_pen = QPen(QColor(30, 30, 30, 230), max(1.0 / scale, 0.5))
        handle_brush = QBrush(QColor(255, 255, 255, 235))
        line_pen = QPen(QColor(255, 255, 255, 220), max(2.0 / scale, 0.5), Qt.DashLine)

        if area.get("kind") == "gradient":
            p0, p1 = self._area_gradient_points(area)
            line = QGraphicsLineItem(p0.x(), p0.y(), p1.x(), p1.y())
            line.setPen(line_pen)
            line.setTransform(base)
            self.scene.addItem(line)
            self._area_overlay_items.append(line)
            mid = QPointF((p0.x() + p1.x()) / 2.0, (p0.y() + p1.y()) / 2.0)
            for pt in (p0, p1, mid):
                hdl = QGraphicsEllipseItem(QRectF(pt.x() - hs / 2, pt.y() - hs / 2, hs, hs))
                hdl.setPen(handle_pen)
                hdl.setBrush(handle_brush)
                hdl.setTransform(base)
                self.scene.addItem(hdl)
                self._area_overlay_items.append(hdl)
            return

        # circle / ellipse
        sel = self._area_circle_sel(area)
        angle = float(area.get("angle", 0.0))
        c = sel.center()
        box_t = QTransform()
        box_t.translate(c.x(), c.y())
        box_t.rotate(angle)
        box_t.translate(-c.x(), -c.y())
        combined = box_t * base
        ell = QGraphicsEllipseItem(sel)
        ell.setPen(line_pen)
        ell.setTransform(combined)
        self.scene.addItem(ell)
        self._area_overlay_items.append(ell)
        # Feather ring (inner edge of the falloff).
        feather = float(area.get("feather", 0.25))
        if feather > 0.01:
            dx = sel.width() / 2.0 * feather
            dy = sel.height() / 2.0 * feather
            inner = QRectF(sel).adjusted(dx, dy, -dx, -dy)
            if inner.width() > 1 and inner.height() > 1:
                ring = QGraphicsEllipseItem(inner)
                ring.setPen(QPen(QColor(255, 255, 255, 120),
                                 max(1.0 / scale, 0.5), Qt.DotLine))
                ring.setTransform(combined)
                self.scene.addItem(ring)
                self._area_overlay_items.append(ring)
        hw, hh = sel.width() / 2.0, sel.height() / 2.0
        rot_off = self.ROTATE_HANDLE_VIEW_PX / scale

        def add_handle(item):
            item.setPen(handle_pen)
            item.setBrush(handle_brush)
            item.setTransform(combined)
            self.scene.addItem(item)
            self._area_overlay_items.append(item)

        for hx, hy in ((-hw, -hh), (hw, -hh), (-hw, hh), (hw, hh),
                       (0.0, -hh), (0.0, hh), (-hw, 0.0), (hw, 0.0)):
            x, y = c.x() + hx, c.y() + hy
            add_handle(QGraphicsRectItem(QRectF(x - hs / 2, y - hs / 2, hs, hs)))
        stem = QGraphicsLineItem(c.x(), c.y() - hh, c.x(), c.y() - hh - rot_off)
        stem.setPen(QPen(QColor(255, 255, 255, 220), max(1.5 / scale, 0.5)))
        stem.setTransform(combined)
        self.scene.addItem(stem)
        self._area_overlay_items.append(stem)
        add_handle(QGraphicsEllipseItem(
            QRectF(c.x() - hs / 2, c.y() - hh - rot_off - hs / 2, hs, hs)))
        add_handle(QGraphicsEllipseItem(QRectF(c.x() - hs / 2, c.y() - hs / 2, hs, hs)))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_view_to_content()

    def auto_frame(self):
        # Show progress dialog
        dialog = AutoFrameDialog(self)
        worker = AutoFrameWorker()
        dialog.set_worker(worker)
        worker.progress.connect(dialog.set_progress)
        worker.finished.connect(lambda: self._on_auto_frame_finished(dialog))
        worker.start()
        dialog.exec_()
    
    def _on_auto_frame_finished(self, dialog):
        """Handle auto frame completion."""
        dialog.accept()
        self.parent().parent().thumbnail_list.update_all_thumbnails()
        if self.current_idx is not None:
            self.update_preview(self.current_idx)
        ccr_backend.save_catalog()

        # Show completion hint
        self.parent().parent().sliders_panel.set_temporary_hint("Auto frame conversion completed!", duration=2000)

    def open_export_dialog(self, checked=False, default_scope=None):
        # `checked` absorbs the bool QAction.triggered emits; `default_scope`
        # is passed by callers (e.g. the thumbnail right-click "Export").
        if not any(img.converted for img in ccr_backend.images):
            QMessageBox.information(self, "Nothing to Export",
                                    "Convert at least one image before exporting.")
            return

        selected_indices = self.parent().parent().thumbnail_list.selected_indices()
        settings_dialog = ExportSettingsDialog(self, current_idx=self.current_idx,
                                               selected_indices=selected_indices,
                                               default_scope=default_scope)
        if settings_dialog.exec_() != QDialog.Accepted or settings_dialog.plan is None:
            return
        plan = settings_dialog.plan

        progress_dialog = ExportProgressDialog(self)
        self._export_worker = ExportItemsWorker(plan)
        progress_dialog.set_worker(self._export_worker)
        self._export_worker.progress.connect(progress_dialog.set_progress)
        self._export_worker.finished.connect(
            lambda result: self._on_export_finished(progress_dialog, result, plan))
        self._export_worker.start()
        progress_dialog.exec_()

    def _on_export_finished(self, dialog, result, plan):
        dialog.accept()
        lines = [f"Exported: {result.get('exported', 0)}"]
        if plan.skipped:
            lines.append(f"Skipped (already exist): {plan.skipped}")
        if result.get("failed"):
            lines.append(f"Failed: {result['failed']}")
        if result.get("cancelled"):
            lines.append("Export was stopped before completion.")
        lines.append(f"\nFolder: {plan.destination}")
        QMessageBox.information(self, "Export Complete", "\n".join(lines))
        ccr_backend.save_catalog()
        if plan.open_folder:
            QDesktopServices.openUrl(QUrl.fromLocalFile(plan.destination))

class SliceWorker(QThread):
    """Runs the backend slice (one shared decode + per-slice previews) off
    the GUI thread."""
    finished_slice = Signal(int)
    progress = Signal(int, int)

    def __init__(self, idx, x_cuts, y_cuts, parent=None):
        super().__init__(parent)
        self._idx = idx
        self._x_cuts = list(x_cuts)
        self._y_cuts = list(y_cuts)

    def run(self):
        try:
            count = ccr_backend.slice_image_by_index(
                self._idx, self._x_cuts, self._y_cuts,
                progress_callback=lambda c, t: self.progress.emit(c, t))
        except Exception as e:
            print(f"Slice failed: {e}")
            count = 0
        self.finished_slice.emit(count)


class SliceProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Slicing...")
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumWidth(240)

        self.label = QLabel("Slicing image", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.progress_label = QLabel("", self)
        self.progress_label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_label)
        self.setLayout(layout)

        self._dot_count = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(400)

    def _animate(self):
        self._dot_count = (self._dot_count + 1) % 4
        self.label.setText("Slicing image" + "." * self._dot_count)

    def set_progress(self, current, total):
        self.progress_label.setText(f"{current} / {total}")

    def closeEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)


class HiResDetailWorker(QThread):
    """Renders the zoom detail image off the GUI thread: decode the RAW at
    half size (non-RAW capped at max_long_side), replay the conversion
    color-matched to the preview, apply the adjustments, and hand back both
    the pre-adjustment base (cached for fast slider re-renders) and the
    displayable 8-bit result. ALL mutable display state is snapshotted at
    construction time (on the GUI thread), so concurrent edits cannot bleed
    into the render."""

    finished_hires = Signal(object, object, object, object, object)
    # img_obj, sig, adj_sig, base (uint16 ndarray), display8 (uint8 ndarray)

    def __init__(self, img_obj, sig, adj_sig, base=None, max_long_side=None, parent=None):
        super().__init__(parent)
        self._img = img_obj
        self.req_img = img_obj
        self.req_sig = sig
        self.req_adj_sig = adj_sig
        self._base = base
        self._cap = max_long_side
        # Snapshots taken on the GUI thread at request time:
        self._settings = dict(img_obj.adjustment_settings)
        # Deep copy: each area nests settings + geometry dicts the GUI thread
        # may mutate while this worker runs.
        self._areas = copy.deepcopy(getattr(img_obj, "area_layers", []))
        self._dust_strokes = copy.deepcopy(getattr(img_obj, "dust_heal_strokes", []))
        self._contrast_base = img_obj.contrast_base
        self._temperature_base = img_obj.temperature_base
        self._brightness_base = img_obj.brightness_base
        self._converted = img_obj.converted
        ci = getattr(img_obj, "conversion_inputs", None)
        self._conversion_inputs = dict(ci) if ci else None

    def run(self):
        try:
            import time as _time
            import cv2 as _cv2
            import numpy as _np
            t0 = _time.perf_counter()
            base = self._base
            if base is None:
                base = self._img.render_hires_base(
                    max_long_side=self._cap,
                    conversion_inputs=self._conversion_inputs)
            if base is None:
                return
            t1 = _time.perf_counter()
            display = self._img.apply_adjustments(
                base, settings=self._settings,
                contrast_base=self._contrast_base,
                temperature_base=self._temperature_base,
                brightness_base=self._brightness_base,
                areas_override=self._areas,
                dust_strokes_override=self._dust_strokes)
            if not self._converted:
                # Mirror the preview pipeline: adjustments first, then the
                # display-only auto-brightness stretch for raw negatives
                display = self._img._auto_brightness_for_preview(display)
            display8 = _np.ascontiguousarray(
                _cv2.convertScaleAbs(display, alpha=255.0 / 65535.0))
            t2 = _time.perf_counter()
            print(f"Hi-res detail: base {(t1 - t0) * 1000:.0f} ms, "
                  f"adjust+8bit {(t2 - t1) * 1000:.0f} ms "
                  f"({display8.shape[1]}x{display8.shape[0]})")
            self.finished_hires.emit(self.req_img, self.req_sig,
                                     self.req_adj_sig, base, display8)
        except Exception as e:
            print(f"Hi-res detail render failed: {e}")


class AutoFrameWorker(QThread):
    finished = Signal()
    progress = Signal(int, int)  # current, total

    def __init__(self):
        super().__init__()
        self._stop_requested = False

    def run(self):
        # Define progress callback
        def progress_callback(current, total_count):
            if not self._stop_requested:
                self.progress.emit(current, total_count)
        
        # Use the auto_frame_all_images functionality from backend with progress
        try:
            ccr_backend.auto_frame_all_images(progress_callback=progress_callback)
        except Exception as e:
            print(f"Failed to auto frame images: {e}")
        
        self.finished.emit()

    def stop(self):
        self._stop_requested = True

class AutoFrameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Framing...")
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint)
        self.setWindowModality(Qt.ApplicationModal)

        self.label = QLabel("Auto framing and converting images", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.progress_label = QLabel("", self)
        self.progress_label.setAlignment(Qt.AlignCenter)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.on_stop_clicked)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.setMinimumWidth(250)

        self._dot_count = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(400)
        self.stopping = False

        self.worker = None  # Will be set from outside

        if parent is not None:
            geo = parent.frameGeometry()
            self.move(geo.center() - self.rect().center())

    def animate(self):
        self._dot_count = (self._dot_count + 1) % 4
        if self.stopping:
            self.label.setText("Stopping"+ "." * self._dot_count)
            return
        
        self.label.setText("Auto framing and converting images" + "." * self._dot_count)

    def set_progress(self, current, total):
        self.progress_label.setText(f"{current} / {total}")

    def set_worker(self, worker):
        self.worker = worker

    def on_stop_clicked(self):
        if self.worker is not None:
            self.worker.stop()
        self.stop_button.setEnabled(False)
        self.label.setText("Stopping...")
        self.stopping = True

    def closeEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)

class ExportItemsWorker(QThread):
    finished = Signal(dict)
    progress = Signal(int, int)  # current, total

    def __init__(self, plan):
        super().__init__()
        self.plan = plan
        self._stop_requested = False

    def run(self):
        def progress_callback(current, total_count):
            if not self._stop_requested:
                self.progress.emit(current, total_count)

        try:
            result = ccr_backend.export_items(
                self.plan.items,
                jpg_output=self.plan.jpg_output,
                jpg_quality=self.plan.jpg_quality,
                max_long_side=self.plan.max_long_side,
                output_colorspace=self.plan.output_colorspace,
                progress_callback=progress_callback,
                cancel_check=lambda: self._stop_requested,
            )
        except Exception as e:
            print(f"Failed to export images: {e}")
            result = {"exported": 0, "failed": len(self.plan.items),
                      "cancelled": False, "failures": []}
        self.finished.emit(result)

    def stop(self):
        self._stop_requested = True

class ExportProgressDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Exporting...")
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint)
        self.setWindowModality(Qt.ApplicationModal)

        self.label = QLabel("Exporting", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.progress_label = QLabel("", self)
        self.progress_label.setAlignment(Qt.AlignCenter)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self.on_stop_clicked)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self.setMinimumWidth(200)

        self._dot_count = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(400)
        self.stopping = False

        self.worker = None  # Will be set from outside

        if parent is not None:
            geo = parent.frameGeometry()
            self.move(geo.center() - self.rect().center())

    def animate(self):
        self._dot_count = (self._dot_count + 1) % 4
        if self.stopping:
            self.label.setText("Stopping"+ "." * self._dot_count)
            return
        
        self.label.setText("Exporting" + "." * self._dot_count)

    def set_progress(self, current, total):
        self.progress_label.setText(f"{current} / {total}")

    def set_worker(self, worker):
        self.worker = worker

    def on_stop_clicked(self):
        if self.worker is not None:
            self.worker.stop()
        self.stop_button.setEnabled(False)
        self.label.setText("Stopping...")
        self.stopping = True

    def closeEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)






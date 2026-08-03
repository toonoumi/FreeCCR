from PySide6.QtWidgets import QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QDialog, QApplication, QPushButton, QMenu, QComboBox
from PySide6.QtGui import QPixmap, QIcon, QTransform, QPainter
from PySide6.QtCore import Qt, QSize, QTimer
import os
import logging
import numpy as np
from io import BytesIO
from core.ccr_backend import ccr_backend  # <-- Add this import
from ui import theme

class LoadingDialog(QDialog):
    def __init__(self, parent=None, cancel_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Loading")
        self.setModal(True)
        self.setWindowModality(Qt.ApplicationModal)  # Block all input to the app
        # Remove close, minimize, maximize buttons
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setFixedSize(400, 100)  # Wider and taller for button

        layout = QVBoxLayout()
        self.label = QLabel("Loading images, please wait")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        # Add Cancel button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedWidth(100)
        self.cancel_button.setFixedHeight(theme.CONTROL_H)
        self.cancel_button.clicked.connect(self.on_cancel)
        self.cancel_callback = cancel_callback
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(theme.GAP_ROW)
        btn_layout.addWidget(self.cancel_button, alignment=Qt.AlignCenter)
        layout.addLayout(btn_layout)

        self.setLayout(layout)

        self.dot_count = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_dots)
        self.timer.start(400)  # Update every 400ms

    def update_dots(self):
        self.dot_count = (self.dot_count % 3) + 1  # Cycles 1, 2, 3
        count = ccr_backend.get_image_count()  
        total = ccr_backend.get_total_file_paths()  # Assuming this method exists
        self.label.setText("Loading images, please wait" + "." * self.dot_count + f" ({count}/{total})")

    def closeEvent(self, event):
        # Prevent closing by user
        event.ignore()

    def reject(self):
        # Prevent closing by Esc or other means
        pass

    def on_cancel(self):
        # Stop timer and close dialog
        self.timer.stop()
        # Clear backend
        
        # Call any additional cancel logic if provided
        if self.cancel_callback:
            self.cancel_callback()
        ccr_backend.clear()
        self.accept()

class ThumbnailList(QWidget):
    def __init__(self, image_selected_callback):
        super().__init__()
        self.image_selected_callback = image_selected_callback
        self.init_ui()

    def _main_window(self):
        """Return the MainWindow that owns this widget."""
        return self.parent().parent()

    def refresh_profile_combo(self):
        """Repopulate the camera-profile picker from the library and reflect the
        active selection (called on startup and after import/delete/generate)."""
        if not hasattr(self, "profile_combo"):
            return
        CM = ccr_backend.CAMERA_MATRIX
        active = getattr(ccr_backend, "active_profile_path", None)
        active_n = (os.path.normcase(os.path.abspath(active))
                    if (active and active != CM) else None)
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        # Two non-removable entries first: None (bare RAW) and Camera Matrix
        # (Adobe RGB + auto-scale, the camera's built-in matrix).
        self.profile_combo.addItem("None", None)
        self.profile_combo.addItem("Camera Matrix", CM)
        sel = 1 if active == CM else 0
        for p in ccr_backend.list_camera_profiles():
            self.profile_combo.addItem(f"{p['name']}  ({p['kind'].upper()})", p["path"])
            if active_n and os.path.normcase(os.path.abspath(p["path"])) == active_n:
                sel = self.profile_combo.count() - 1
        self.profile_combo.setCurrentIndex(sel)
        self.profile_combo.blockSignals(False)

    def _on_profile_combo_changed(self, _idx):
        """User picked a profile (or None) — forward to the MainWindow, which
        owns activation + persistence + the thumbnail mismatch refresh."""
        mw = self._main_window()
        if hasattr(mw, "set_active_profile_path"):
            mw.set_active_profile_path(self.profile_combo.currentData())

    def init_ui(self):
        self.layout = QVBoxLayout()
        theme.apply_panel_spacing(self.layout)

        # Camera-profile picker — choose which imported / IT8-generated profile
        # (kept in FreeCCR's library) is applied at decode, or None. Manage the
        # library in Settings ▸ Color Management. See spec/camera-profile-library.md.
        self.profile_combo = QComboBox()
        self.profile_combo.setToolTip(
            "Camera input profile applied at decode, from your imported / "
            "IT8-generated library. 'None' applies no profile. Add and manage "
            "profiles in Settings ▸ Color Management.")
        self.profile_combo.currentIndexChanged.connect(self._on_profile_combo_changed)
        _plabel = QLabel("Camera profile:")
        _plabel.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        self.layout.addWidget(_plabel)
        self.layout.addWidget(self.profile_combo)
        # Positive mode now lives only in Settings ▸ Color Management.

        self.thumbnail_list = QListWidget()
        self.thumbnail_list.setViewMode(QListWidget.IconMode)
        # Set the icon size up front so incrementally-appended items (e.g.
        # tether captures added via append_image_item, before any full
        # load_thumbnails rebuild has run) show at full size, not Qt's tiny
        # default. load_thumbnails also sets this, harmlessly.
        self.thumbnail_list.setIconSize(QSize(156, 156))
        self.thumbnail_list.setSpacing(10)
        self.thumbnail_list.setDragDropMode(QListWidget.NoDragDrop)
        self.thumbnail_list.setResizeMode(QListWidget.Adjust)
        self.thumbnail_list.setMovement(QListWidget.Static)
        self.thumbnail_list.setWrapping(False)
        self.thumbnail_list.setFlow(QListWidget.TopToBottom)
        self.thumbnail_list.setFixedWidth(196)
        self.thumbnail_list.setFocusPolicy(Qt.StrongFocus)  # <-- Ensure strong focus
        self.thumbnail_list.setFocus()  # <-- Optionally set focus immediately
        # Ctrl+click toggles, Shift+click selects a range
        self.thumbnail_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.thumbnail_list.itemClicked.connect(self.on_thumbnail_clicked)
        self.thumbnail_list.currentItemChanged.connect(self.on_current_item_changed)
        # Right-click menu: duplicate / remove the selected image(s)
        self.thumbnail_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.thumbnail_list.customContextMenuRequested.connect(self.show_context_menu)
        self.layout.addWidget(self.thumbnail_list)

        self.count_label = QLabel("Total: 0 image(s)")
        self.count_label.setAlignment(Qt.AlignLeft)
        self.layout.addWidget(self.count_label)

        self.setLayout(self.layout)
        
        # Set initial hint since no images are loaded when program starts
        # Use QTimer to ensure parent widgets are fully initialized
        QTimer.singleShot(100, self.set_initial_hint)

    def set_initial_hint(self):
        """
        Set the initial hint when the program starts with no images loaded.
        """
        try:
            self._main_window().sliders_panel.set_hint("<b>Hint:</b><br>Use file menu to import folder or files.")
        except AttributeError:
            # In case the parent structure isn't ready yet, try again later
            QTimer.singleShot(200, self.set_initial_hint)

    def show_loading_dialog(self):
        # Pass a callback to cancel the worker thread
        def cancel_loading():
            # Access the worker from the main window and call cancel.
            # _loader_worker can already be None (cleanup ran while the
            # dialog was still up) — Cancel must not die on None.cancel().
            main_window = self._main_window()
            worker = getattr(main_window, '_loader_worker', None)
            if worker is not None:
                worker.cancel()
        self.loading_dialog = LoadingDialog(self, cancel_callback=cancel_loading)
        self.loading_dialog.show()
        QApplication.processEvents()

    def load_thumbnails(self):
        # Guard re-entrancy: the per-5-items processEvents below can deliver
        # a right-click whose menu mutates the backend and re-enters here.
        self._rebuilding = True
        try:
            self._load_thumbnails_inner()
        finally:
            self._rebuilding = False

    def _load_thumbnails_inner(self):
        self.thumbnail_list.clear()
        image_count = ccr_backend.get_image_count()
        logging.info(f"Loading {image_count} images from backend")
        for idx in range(image_count):
            thumbnail = ccr_backend.get_thumbnail_by_index(idx)
            # Get filename from the CCRImage object itself (always in sync);
            # sliced images carry a distinguishing display name (e.g. _s2)
            img_obj = ccr_backend.get_image_by_index(idx)
            if img_obj is None:
                filename = ""
            else:
                filename = (getattr(img_obj, "display_name", None)
                            or os.path.basename(img_obj.file_path))
            item = QListWidgetItem(filename)  # Set filename as item text
            # Apply rotations and flips using PySide6 transformations
            transformed_thumbnail = self.apply_frontend_transformations(thumbnail, idx)
            icon = QIcon(transformed_thumbnail)
            item.setIcon(icon)
            item.setData(Qt.UserRole, idx)
            self.thumbnail_list.addItem(item)
            if idx % 5 == 0:
                QApplication.processEvents()
        self.thumbnail_list.setIconSize(QSize(156, 156))
        self.count_label.setText(f"Total: {image_count} image(s)")
        self.refresh_profile_warnings()

        # Set current index to 0 if there are items
        if image_count > 0:
            self.thumbnail_list.setCurrentRow(0)
            self._main_window().image_preview.update_preview(0)
            # Clear any existing hint when images are loaded
            self._main_window().sliders_panel.clear_hint()
        else:
            # Set hint when no images are loaded, and clear the preview —
            # otherwise the last removed image stays on the canvas with
            # stale state (and its memory pinned).
            self._main_window().sliders_panel.set_hint("<b>Hint:</b><br>Use file menu to import folder or files.")
            try:
                self._main_window().image_preview.clear_preview()
            except AttributeError:
                pass

        if hasattr(self, 'loading_dialog') and self.loading_dialog is not None:
            self.loading_dialog.accept()  # Close the dialog

    def update_thumbnail(self, idx):
        thumbnail = ccr_backend.get_thumbnail_by_index(idx)
        if thumbnail is not None:
            item = self.thumbnail_list.item(idx)
            if item:
                # Apply rotations and flips using PySide6 transformations
                transformed_thumbnail = self.apply_frontend_transformations(thumbnail, idx)
                icon = QIcon(transformed_thumbnail)
                item.setIcon(icon)
                                                                                 
    def update_all_thumbnails(self):
        for idx in range(ccr_backend.get_image_count()):
            self.update_thumbnail(idx)

    def append_image_item(self, idx, select=True):
        """Append a single thumbnail item for a newly added backend image
        (e.g. a tether capture) without rebuilding the whole list. When select
        is True, selecting the item drives update_preview/set_current_idx via
        on_current_item_changed (so the capture shows big on the canvas).
        Returns True if the item was appended; False if skipped (a rebuild is
        in flight, or the index is no longer valid — the next full rebuild will
        include the image either way)."""
        if getattr(self, "_rebuilding", False):
            return False
        img_obj = ccr_backend.get_image_by_index(idx)
        if img_obj is None:
            logging.warning(f"append_image_item: no backend image at index {idx}")
            return False
        filename = (getattr(img_obj, "display_name", None)
                    or os.path.basename(img_obj.file_path))
        item = QListWidgetItem(filename)
        thumbnail = ccr_backend.get_thumbnail_by_index(idx)
        if thumbnail is not None:
            item.setIcon(QIcon(self.apply_frontend_transformations(thumbnail, idx)))
        item.setData(Qt.UserRole, idx)
        self.thumbnail_list.addItem(item)
        self.count_label.setText(
            f"Total: {ccr_backend.get_image_count()} image(s)")
        self.refresh_profile_warnings()
        if select:
            self.thumbnail_list.setCurrentRow(self.thumbnail_list.count() - 1)
        return True
        
    def on_thumbnail_clicked(self, item):
        idx = item.data(Qt.UserRole)
        self._main_window().image_preview.update_preview(idx)
        self._main_window().sliders_panel.set_current_idx(idx)

    def on_current_item_changed(self, current, previous):
        if current:
            idx = current.data(Qt.UserRole)
            self._main_window().image_preview.update_preview(idx)
            self._main_window().sliders_panel.set_current_idx(idx)
            if previous:
                prev_idx = previous.data(Qt.UserRole)
                self.update_thumbnail(prev_idx)  # Update previous thumbnail
            

    def keyPressEvent(self, event):
        # Allow up/down keys to move the selection
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            self.thumbnail_list.keyPressEvent(event)
        else:
            super().keyPressEvent(event)

    def selected_indices(self):
        """Backend indices of the selected thumbnails, ascending."""
        return sorted({item.data(Qt.UserRole)
                       for item in self.thumbnail_list.selectedItems()})

    # --- Camera-profile mismatch flagging ---------------------------------
    def _profile_mismatch(self, idx) -> bool:
        """True if the image was graded under a different camera profile than the
        one active now (so it should be re-graded / flagged)."""
        img = ccr_backend.get_image_by_index(idx)
        if img is None:
            return False
        sig = getattr(img, "profile_signature", None)
        return sig is not None and sig != ccr_backend.active_profile_signature()

    def refresh_profile_warnings(self):
        """Flag every thumbnail whose image was graded under a different camera
        profile than the active one — ⚠ prefix + amber text + tooltip. Called on
        load/append and whenever the active profile (or disable / Positive mode)
        changes."""
        active = ccr_backend.active_profile_signature()
        warn = theme.qcolor(theme.WARN_TEXT)
        normal = theme.qcolor(theme.TEXT)
        for row in range(self.thumbnail_list.count()):
            item = self.thumbnail_list.item(row)
            idx = item.data(Qt.UserRole)
            img = ccr_backend.get_image_by_index(idx)
            if img is None:
                continue
            filename = (getattr(img, "display_name", None)
                        or os.path.basename(img.file_path))
            sig = getattr(img, "profile_signature", None)
            mism = sig is not None and sig != active
            item.setText(("⚠ " + filename) if mism else filename)
            item.setForeground(warn if mism else normal)
            item.setToolTip(
                "Graded under a different camera profile or field "
                "correction.\nRight-click ▸ Replace with current camera "
                "profile." if mism else "")

    def replace_with_current_profile(self, indices):
        """Re-grade the given image(s) under the current camera profile by
        re-decoding them and replaying the conversion — KEEPING everything (the
        negative conversion / bwpoint, adjustments, crop, flips, rotation, slice).
        Only the camera-profile decode changes, unlike Reset. Bulk-safe."""
        if not indices:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            changed = ccr_backend.regrade_images_by_indices(indices)
        finally:
            QApplication.restoreOverrideCursor()
        if not changed:
            return
        for idx in indices:
            self.update_thumbnail(idx)
        self.refresh_profile_warnings()
        current = self.thumbnail_list.currentItem()
        if current is not None:
            cur_idx = current.data(Qt.UserRole)
            mw = self._main_window()
            mw.image_preview.update_preview(cur_idx)
            mw.sliders_panel.set_current_idx(cur_idx)
            mw.image_preview._update_unconvert_action_state()
        ccr_backend.save_catalog()

    def show_context_menu(self, pos):
        if getattr(self, "_rebuilding", False):
            return  # list is mid-rebuild; stale items must not act
        item = self.thumbnail_list.itemAt(pos)
        if item is None:
            return
        # Right-clicking outside the current selection targets just that
        # item. clearSelection() is required: with Ctrl/Shift still held,
        # setCurrentItem alone KEEPS committed selection ranges and would
        # make the menu act on a wrong multi-item set.
        if not item.isSelected():
            self.thumbnail_list.clearSelection()
            self.thumbnail_list.setCurrentItem(item)
        indices = self.selected_indices()
        if not indices:
            return
        suffix = f" ({len(indices)})" if len(indices) > 1 else ""
        menu = QMenu(self)
        duplicate_action = menu.addAction(f"Duplicate{suffix}")
        remove_action = menu.addAction(f"Remove from list{suffix}")
        menu.addSeparator()
        export_action = menu.addAction("Export…")
        reset_action = menu.addAction(f"Reset{suffix}")
        # Offered only when the selection contains slices: restores the
        # un-sliced parent image(s) in place of all their slices.
        reset_slice_action = None
        slice_count = sum(
            1 for i in indices
            if i < len(ccr_backend.images) and ccr_backend.images[i].source_ops
            and ccr_backend.images[i].slice_group is not None)
        if slice_count:
            slice_suffix = f" ({slice_count})" if slice_count > 1 else ""
            reset_slice_action = menu.addAction(f"Reset Slice{slice_suffix}")
        # "Replace with current camera profile" — only for images in the selection
        # graded under a different profile (re-decodes them, like Reset).
        mismatched = [i for i in indices if self._profile_mismatch(i)]
        replace_action = None
        if mismatched:
            menu.addSeparator()
            rsuffix = f" ({len(mismatched)})" if len(mismatched) > 1 else ""
            replace_action = menu.addAction(
                f"Replace with current camera profile{rsuffix}")
        action = menu.exec_(self.thumbnail_list.mapToGlobal(pos))
        if action == duplicate_action:
            self.duplicate_images(indices)
        elif action == remove_action:
            self.remove_images(indices)
        elif action == export_action:
            # Open the export dialog pre-scoped to the right-clicked selection.
            self._main_window().image_preview.open_export_dialog(default_scope="selected")
        elif action == reset_action:
            self.reset_images(indices)
        elif reset_slice_action is not None and action == reset_slice_action:
            self.reset_slices(indices)
        elif replace_action is not None and action == replace_action:
            self.replace_with_current_profile(mismatched)

    def duplicate_images(self, indices):
        created = ccr_backend.duplicate_images_by_indices(indices)
        if not created:
            return
        self.load_thumbnails()
        first_copy = min(indices) + 1
        if first_copy < ccr_backend.get_image_count():
            self.thumbnail_list.setCurrentRow(first_copy)
        ccr_backend.save_catalog()

    @staticmethod
    def _next_selection_target(removed_indices, new_count):
        """Row to select after a removal: the image that followed the
        removed one(s) — it now occupies the lowest removed index — clamped
        to the new last image when the tail was removed."""
        if new_count <= 0:
            return None
        return min(min(removed_indices), new_count - 1)

    def remove_images(self, indices):
        if not ccr_backend.remove_images_by_indices(indices):
            return
        self.load_thumbnails()
        target = self._next_selection_target(indices, ccr_backend.get_image_count())
        if target is not None and target != 0:  # load_thumbnails already selected row 0
            self.thumbnail_list.setCurrentRow(target)
        ccr_backend.save_catalog()

    def remove_image(self, idx):
        """Single-image removal (kept for compatibility)."""
        self.remove_images([idx])

    def reset_images(self, indices):
        # Reset re-decodes the source file(s) synchronously; show a busy
        # cursor so the brief GUI-thread freeze reads as work, not a hang.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            changed = ccr_backend.reset_images_by_indices(indices)
        finally:
            QApplication.restoreOverrideCursor()
        if not changed:
            return
        for idx in indices:
            self.update_thumbnail(idx)
        # Reset re-decodes under the current profile, so the per-image grading
        # signature now matches — clear any stale ⚠ mismatch flags.
        self.refresh_profile_warnings()
        # Refresh the preview, sliders, and convert/unconvert state for the
        # currently selected image (the list itself is unchanged).
        current = self.thumbnail_list.currentItem()
        if current is not None:
            cur_idx = current.data(Qt.UserRole)
            main_window = self._main_window()
            main_window.image_preview.update_preview(cur_idx)
            main_window.sliders_panel.set_current_idx(cur_idx)
            main_window.image_preview._update_unconvert_action_state()
        ccr_backend.save_catalog()

    def reset_slices(self, indices):
        # Reset re-decodes the source file(s) synchronously; show a busy
        # cursor so the brief GUI-thread freeze reads as work, not a hang.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            restored = ccr_backend.reset_slice_by_indices(indices)
        finally:
            QApplication.restoreOverrideCursor()
        if restored is None:
            return
        self.load_thumbnails()
        if restored != 0:  # load_thumbnails already selected row 0
            self.thumbnail_list.setCurrentRow(restored)

    def apply_frontend_transformations(self, thumbnail, idx):
        """
        Apply rotations and flips to the thumbnail using PySide6 transformations.
        This only affects the visual display without modifying backend data.
        """
        # Get transformation settings from backend
        rotation_angle = ccr_backend.get_image_rotation_by_index(idx)
        horizontal_flip = ccr_backend.get_image_horizontal_flip_by_index(idx)
        vertical_flip = ccr_backend.get_image_vertical_flip_by_index(idx)
        
        # Start with the original thumbnail
        transformed_thumbnail = thumbnail
        
        # Apply flips first using QTransform
        if horizontal_flip or vertical_flip:
            transform = QTransform()
            if horizontal_flip:
                transform.scale(-1, 1)  # Horizontal flip
            if vertical_flip:
                transform.scale(1, -1)  # Vertical flip
            transformed_thumbnail = transformed_thumbnail.transformed(transform)
        
        # Apply 90-degree rotations (but not fine angle rotation)
        angle = rotation_angle % 360
        if angle != 0:
            transform = QTransform()
            transform.rotate(angle)
            transformed_thumbnail = transformed_thumbnail.transformed(transform)
        
        # Ensure the transformed thumbnail fits within the icon size (156x156)
        # while maintaining aspect ratio to prevent overlapping
        icon_size = 156
        if transformed_thumbnail.width() > icon_size or transformed_thumbnail.height() > icon_size:
            transformed_thumbnail = transformed_thumbnail.scaled(
                icon_size, icon_size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        
        # Create a centered thumbnail on a fixed-size canvas to ensure consistent spacing
        canvas = QPixmap(icon_size, icon_size)
        canvas.fill(Qt.transparent)  # Transparent background
        
        # Calculate position to center the thumbnail on the canvas
        x_offset = (icon_size - transformed_thumbnail.width()) // 2
        y_offset = (icon_size - transformed_thumbnail.height()) // 2
        
        # Draw the thumbnail centered on the canvas
        painter = QPainter(canvas)
        painter.drawPixmap(x_offset, y_offset, transformed_thumbnail)
        painter.end()
        
        return canvas

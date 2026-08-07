"""
Crop panel — covers the right-hand sliders panel while in crop mode (the same
cover/restore pattern as DustRemovalPanel). Exposes "all the crop options":

  - Aspect ratio: presets (Free / Original / 1:1 / 5:4 / 4:3 / 7:5 / 3:2 / 16:9 /
    cinematic Academy / 1.85 / 2:1 / 2.35 / 2.39 / Custom…) with a
    Landscape/Portrait toggle. A locked ratio constrains every
    on-canvas drag and reshapes the current box; the choice persists across
    images and sessions (QSettings) for catalogue consistency (issue #39).
  - Straighten: a ±45° slider that drives the crop box's rotation, two-way synced
    with the on-canvas rotate knob. Any pre-existing image micro-rotation is
    folded into this on entry (see ImagePreview.enter_crop_mode / spec §5.4).
  - Reset (clear the pending box + straighten) and Done (commit).

The committed result is still just CCRImage.crop_rect / crop_angle — this panel
only changes how they are produced interactively. See spec/crop-panel.md.
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QPushButton, QComboBox, QRadioButton,
                               QButtonGroup, QSpinBox)
from PySide6.QtCore import Qt, QSettings

from core import crop_aspect
from widgets.image_preview import CenteringSlider
from ui import theme


class CropPanel(QWidget):
    def __init__(self, main_window, image_preview, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_preview = image_preview
        self._settings = QSettings("FreeCCR", "FreeCCR")
        # Guards re-entrancy while widgets are updated programmatically (so a
        # setValue/setCurrentIndex doesn't fire a handler that mutates state).
        self._suppress = False
        self._build_ui()
        self._restore_prefs()

    # --- UI ---------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        theme.apply_panel_spacing(layout)

        header = QLabel("Crop")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 14px; font-weight: bold; margin: 4px;")
        layout.addWidget(header)

        # --- Aspect ratio ---
        layout.addWidget(self._section_label("Aspect Ratio"))
        self.aspect_combo = QComboBox()
        self.aspect_combo.setFixedHeight(theme.CONTROL_H)
        theme.shrinkable_combo(self.aspect_combo)
        for label, key, _ratio in crop_aspect.ASPECT_PRESETS:
            self.aspect_combo.addItem(label, key)
        self.aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        layout.addWidget(self.aspect_combo)

        # Orientation (Landscape / Portrait)
        orient_row = QHBoxLayout()
        orient_row.setSpacing(theme.GAP_TIGHT)
        self.landscape_radio = QRadioButton("Landscape")
        self.portrait_radio = QRadioButton("Portrait")
        self._orient_group = QButtonGroup(self)
        self._orient_group.addButton(self.landscape_radio)
        self._orient_group.addButton(self.portrait_radio)
        self.landscape_radio.setChecked(True)
        self.landscape_radio.toggled.connect(self._on_orientation_changed)
        orient_row.addWidget(self.landscape_radio)
        orient_row.addWidget(self.portrait_radio)
        orient_row.addStretch(1)
        layout.addLayout(orient_row)

        # Custom W:H (only shown for the Custom preset)
        self.custom_row = QHBoxLayout()
        self.custom_row.setSpacing(theme.GAP_TIGHT)
        custom_lbl = QLabel("Custom")
        custom_lbl.setFixedWidth(theme.LABEL_COL_W)
        self.custom_w_spin = QSpinBox()
        self.custom_w_spin.setRange(1, 9999)
        self.custom_w_spin.setValue(3)
        self.custom_w_spin.setFixedHeight(theme.CONTROL_H)
        self.custom_h_spin = QSpinBox()
        self.custom_h_spin.setRange(1, 9999)
        self.custom_h_spin.setValue(2)
        self.custom_h_spin.setFixedHeight(theme.CONTROL_H)
        self.custom_w_spin.valueChanged.connect(self._on_custom_changed)
        self.custom_h_spin.valueChanged.connect(self._on_custom_changed)
        custom_colon = QLabel(":")
        self.custom_row.addWidget(custom_lbl)
        self.custom_row.addWidget(self.custom_w_spin, 1)
        self.custom_row.addWidget(custom_colon)
        self.custom_row.addWidget(self.custom_h_spin, 1)
        self._custom_widgets = [custom_lbl, self.custom_w_spin, custom_colon,
                                self.custom_h_spin]
        layout.addLayout(self.custom_row)

        layout.addWidget(self._separator())

        # --- Straighten ---
        layout.addWidget(self._section_label("Straighten"))
        str_row = QHBoxLayout()
        str_row.setSpacing(theme.GAP_TIGHT)
        str_lbl = QLabel("Angle")
        str_lbl.setFixedWidth(theme.LABEL_COL_W)
        # Value is tenths of a degree: -450..450 -> -45.0°..+45.0°.
        self.straighten_slider = CenteringSlider(Qt.Horizontal)
        self.straighten_slider.setMinimum(-450)
        self.straighten_slider.setMaximum(450)
        self.straighten_slider.setValue(0)
        self.straighten_slider.setTickInterval(150)
        self.straighten_slider.setTickPosition(CenteringSlider.TicksBelow)
        self.straighten_slider.setFixedHeight(theme.CONTROL_H)
        self.straighten_slider.valueChanged.connect(self._on_straighten_changed)
        self.straighten_value = QLabel("+0.0°")
        self.straighten_value.setFixedWidth(theme.VALUE_COL_W)
        str_row.addWidget(str_lbl)
        str_row.addWidget(self.straighten_slider)
        str_row.addWidget(self.straighten_value)
        layout.addLayout(str_row)

        hint = QLabel(
            "Drag on the image to draw a box; drag handles to resize, the top "
            "knob (or the slider above) to straighten, the center to move. "
            "Enter = Done, Esc = cancel, right-click = clear.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(hint)

        layout.addWidget(self._separator())
        layout.addStretch(1)

        btns = QHBoxLayout()
        theme.apply_button_row(btns)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFixedHeight(theme.CONTROL_H)
        self.reset_btn.setToolTip("Clear the pending crop box and straighten.")
        self.reset_btn.clicked.connect(self._on_reset)
        self.done_btn = QPushButton("✓  Done")
        self.done_btn.setMinimumHeight(theme.CONTROL_H_LG)
        self.done_btn.setToolTip("Apply the crop and return to the adjustment sliders.")
        theme.style_button(self.done_btn, "primary")
        self.done_btn.clicked.connect(self._on_done)
        btns.addWidget(self.reset_btn)
        btns.addWidget(self.done_btn)
        layout.addLayout(btns)

    @staticmethod
    def _section_label(text):
        lbl = QLabel(text)
        lbl.setStyleSheet(theme.section_header_qss())
        return lbl

    @staticmethod
    def _separator():
        return theme.section_separator()

    # --- Public API used by MainWindow / ImagePreview ---------------------
    def bind_image(self):
        """Refresh panel state when crop mode opens: reflect the (possibly
        folded) straighten angle, and seed/keep a box matching the remembered
        ratio. Called by MainWindow.toggle_crop_panel after enter_crop_mode."""
        self._update_custom_visibility()
        self.image_preview.seed_crop_ratio_on_entry()
        self.on_crop_geometry_changed()
        self.image_preview.redraw_crop_overlay()

    def on_crop_geometry_changed(self):
        """Re-sync the straighten slider from the canvas (after a knob drag, a
        fresh-box reset, or a programmatic reshape)."""
        ang = getattr(self.image_preview, "_pending_crop_angle", 0.0) or 0.0
        v = int(round(max(-45.0, min(45.0, ang)) * 10))
        self._suppress = True
        self.straighten_slider.setValue(v)
        self.straighten_value.setText(f"{v / 10.0:+.1f}°")
        self._suppress = False

    def current_display_ratio(self, pixmap=None):
        """The target on-screen width/height ratio for the active preset, or
        None for Free / unresolved."""
        key = self._active_key()
        if key == "free":
            return None
        if key == "original":
            if pixmap is None:
                pixmap = self.image_preview.current_pixmap
            if pixmap is None or pixmap.height() <= 0:
                return None
            return pixmap.width() / pixmap.height()
        if key == "custom":
            w, h = self.custom_w_spin.value(), self.custom_h_spin.value()
            return (w / h) if (w >= 1 and h >= 1) else None
        entry = crop_aspect.preset_for_key(key)
        if entry is None or entry[2] in (None, "original"):
            return None
        return crop_aspect.oriented_ratio(entry[2], self._landscape())

    def current_effective_ratio(self, rotation, pixmap=None):
        """The ratio to enforce on the pending crop rect (un-rotated image
        space), accounting for coarse 90/270 rotation."""
        return crop_aspect.effective_box_ratio(
            self.current_display_ratio(pixmap), rotation)

    # --- Handlers ---------------------------------------------------------
    def _on_aspect_changed(self, *_):
        self._update_custom_visibility()
        self._save_prefs()
        self._apply_ratio_now()

    def _on_orientation_changed(self, *_):
        # QButtonGroup fires twice (off + on); act once, on the checked state.
        self._save_prefs()
        self._apply_ratio_now()

    def _on_custom_changed(self, *_):
        self._save_prefs()
        self._apply_ratio_now()

    def _on_straighten_changed(self, value):
        self.straighten_value.setText(f"{value / 10.0:+.1f}°")
        if self._suppress:
            return
        self.image_preview.set_pending_straighten(value / 10.0)

    def _on_reset(self):
        self.image_preview.reset_pending_crop()
        # Re-seed a centered ratio box when a ratio is locked (Free -> no box).
        self.image_preview.seed_crop_ratio_on_entry()
        self.on_crop_geometry_changed()
        self.image_preview.redraw_crop_overlay()

    def _on_done(self):
        self.main_window.toggle_crop_panel(False)

    # --- helpers ----------------------------------------------------------
    def _active_key(self):
        return self.aspect_combo.currentData()

    def _landscape(self):
        return self.landscape_radio.isChecked()

    def _apply_ratio_now(self):
        if self._suppress:
            return
        if self.image_preview.crop_mode:
            self.image_preview.reapply_crop_ratio(seed_if_empty=True)

    def _update_custom_visibility(self):
        key = self._active_key()
        is_custom = key == "custom"
        for wdg in self._custom_widgets:
            wdg.setVisible(is_custom)
        orient_ok = key not in crop_aspect.ORIENTATION_FIXED_KEYS
        self.landscape_radio.setEnabled(orient_ok)
        self.portrait_radio.setEnabled(orient_ok)

    def _restore_prefs(self):
        self._suppress = True
        key = self._settings.value("crop/aspect_key", "free", type=str)
        idx = self.aspect_combo.findData(key)
        self.aspect_combo.setCurrentIndex(idx if idx >= 0 else 0)
        landscape = self._settings.value(
            "crop/orientation", "landscape", type=str) != "portrait"
        (self.landscape_radio if landscape else self.portrait_radio).setChecked(True)
        self.custom_w_spin.setValue(self._settings.value("crop/custom_w", 3, type=int))
        self.custom_h_spin.setValue(self._settings.value("crop/custom_h", 2, type=int))
        self._suppress = False
        self._update_custom_visibility()

    def _save_prefs(self):
        if self._suppress:
            return
        self._settings.setValue("crop/aspect_key", self._active_key())
        self._settings.setValue(
            "crop/orientation", "landscape" if self._landscape() else "portrait")
        self._settings.setValue("crop/custom_w", self.custom_w_spin.value())
        self._settings.setValue("crop/custom_h", self.custom_h_spin.value())

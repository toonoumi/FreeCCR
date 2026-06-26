from PySide6.QtWidgets import (QWidget, QVBoxLayout, QSlider, QLabel, QHBoxLayout,
                                QSizePolicy, QStyleOptionSlider, QFrame, QStyle,
                                QPushButton, QDialog, QMessageBox, QScrollArea,
                                QCheckBox, QComboBox)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QRectF
from PySide6.QtGui import (QPixmap, QKeySequence, QShortcut, QPainter, QColor,
                           QLinearGradient, QPen)
from core.ccr_backend import ccr_backend
from core.ccr_processor import COLOR_BANDS, BAND_PARAMS, BAND_ADJUSTMENT_KEYS
from widgets.curve_editor import CurveEditor
from ui import theme
import copy

# Setting groups offered by the "Sync to All" dialog. The adjustment-key
# groups partition SlidersPanel.ADJUSTMENT_KEYS exactly; "crop" syncs the
# image's crop box (rect + angle) instead of adjustment keys.
SYNC_GROUPS = [
    ("profile", "Color Profile (Color / B&W)", ()),
    # AWB is a per-image enable flag (not a slider); only the flag is synced —
    # the gains are recomputed per image from its own pixels. Handled specially
    # in _perform_sync_to_all, like "profile" and "crop".
    ("awb", "Auto White Balance (on/off)", ()),
    ("wb", "White Balance / Tint", ("temperature", "tint")),
    ("tone", "Tone (gain, brightness, contrast, ...)",
     ("exposure", "brightness", "highlights", "white_point",
      "shadows", "black_point", "contrast")),
    ("sat", "Saturation", ("saturation", "sub_saturation")),
    ("crop", "Crop", ()),
    ("channels", "Channel Levels", (
        "ch_input_gain", "ch_master_shift", "ch_master_gain",
        "ch_r_shift", "ch_r_gain", "ch_r_blackpoint",
        "ch_g_shift", "ch_g_gain", "ch_g_blackpoint",
        "ch_b_shift", "ch_b_gain", "ch_b_blackpoint")),
    ("bands", "Subtractive Saturations (per color)",
     tuple(BAND_ADJUSTMENT_KEYS) + ("band_feather",)),
    # "curves" lives outside ADJUSTMENT_KEYS (it's a nested structure, not a
    # slider), so it's synced specially in _perform_sync_to_all, like crop.
    ("curves", "Curves", ()),
]


class SyncSettingsDialog(QDialog):
    """Pick which setting groups 'Sync to All' copies to every image."""

    def __init__(self, parent=None, selection=None):
        super().__init__(parent)
        self.setWindowTitle("Sync to All")
        self.setMinimumWidth(theme.DIALOG_W_SM)
        layout = QVBoxLayout(self)
        theme.apply_panel_spacing(layout)
        layout.addWidget(QLabel("Sync these settings to all images:"))

        self._checkboxes = {}
        for gid, label, _keys in SYNC_GROUPS:
            checkbox = QCheckBox(label)
            checkbox.setChecked(True if selection is None else selection.get(gid, True))
            layout.addWidget(checkbox)
            self._checkboxes[gid] = checkbox

        # Selection helpers — pushed to opposite ends with checkbox glyphs so they
        # no longer read as one identical pair.
        select_row = theme.apply_button_row(QHBoxLayout())
        select_all_btn = QPushButton("☑  Select All")
        deselect_all_btn = QPushButton("☐  Deselect All")
        theme.style_button(select_all_btn, "secondary")
        theme.style_button(deselect_all_btn, "secondary")
        select_all_btn.clicked.connect(lambda: self._set_all(True))
        deselect_all_btn.clicked.connect(lambda: self._set_all(False))
        select_row.addWidget(select_all_btn)
        select_row.addStretch(1)
        select_row.addWidget(deselect_all_btn)
        layout.addLayout(select_row)

        # Separator: "choose what to sync" (above) vs "commit" (below).
        layout.addWidget(theme.section_separator())

        button_row = theme.apply_button_row(QHBoxLayout())
        sync_btn = QPushButton("Sync")
        cancel_btn = QPushButton("Cancel")
        theme.style_button(sync_btn, "primary", default=True)
        theme.style_button(cancel_btn, "secondary")
        sync_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_row.addStretch(1)
        button_row.addWidget(cancel_btn)
        button_row.addWidget(sync_btn)
        layout.addLayout(button_row)

    def _set_all(self, checked: bool):
        for checkbox in self._checkboxes.values():
            checkbox.setChecked(checked)

    def selection(self) -> dict:
        return {gid: checkbox.isChecked() for gid, checkbox in self._checkboxes.items()}

class CollapsibleSection(QWidget):
    """A toggle-button header that shows/hides its content widget. Default: collapsed."""
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._toggle_btn = QPushButton(f"+ {title}")
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setStyleSheet(
            "QPushButton { text-align: left; padding: 4px 6px; "
            f"background: {theme.SURFACE}; border: none; border-radius: {theme.RADIUS_SM}px; "
            f"color: {theme.TEXT}; font-size: 11px; font-weight: bold; }}"
            f"QPushButton:checked {{ background: {theme.SURFACE_ACTIVE}; }}"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout()
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(theme.GAP_ROW)
        self._content.setLayout(self._content_layout)
        self._content.setVisible(False)

        outer = QVBoxLayout()
        outer.setContentsMargins(0, theme.GAP_SECTION, 0, 0)
        outer.setSpacing(theme.GAP_ROW)
        outer.addWidget(self._toggle_btn)
        outer.addWidget(self._content)
        self.setLayout(outer)

    def add_layout(self, layout):
        self._content_layout.addLayout(layout)

    def add_widget(self, widget):
        self._content_layout.addWidget(widget)

    def _on_toggle(self, checked: bool):
        self._content.setVisible(checked)
        label = self._toggle_btn.text()[2:]
        self._toggle_btn.setText(f"{'- ' if checked else '+ '}{label}")


class ResettableSlider(QSlider):
    def mousePressEvent(self, event):
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        handle_rect = self.style().subControlRect(
            QStyle.CC_Slider,
            option,
            QStyle.SC_SliderHandle,
            self
        )
        if handle_rect.contains(event.pos()):
            super().mousePressEvent(event)
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        # Reset to 0 and trigger adjustment update
        old_value = self.value()
        self.setValue(0)
        
        # Find the parent SlidersPanel and trigger adjustment update
        parent_widget = self.parent()
        while parent_widget and not isinstance(parent_widget, SlidersPanel):
            parent_widget = parent_widget.parent()
        
        if parent_widget and old_value != 0:
            for i, slider in enumerate(parent_widget.sliders):
                if slider is self:
                    parent_widget.slider_value_labels[i].setText("0")
                    parent_widget.on_slider_changed()
                    break
        
        super().mouseDoubleClickEvent(event)

    def initStyleOption(self, option):
        option.initFrom(self)
        option.orientation = self.orientation()
        option.minimum = self.minimum()
        option.maximum = self.maximum()
        option.sliderPosition = self.sliderPosition()
        option.sliderValue = self.value()
        option.singleStep = self.singleStep()
        option.pageStep = self.pageStep()
        option.upsideDown = False
        return option

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Up, Qt.Key_Down):
            event.ignore()
        else:
            super().keyPressEvent(event)

    def wheelEvent(self, event):
        # Override wheel event to make each wheel step equal to 1 instead of default 3
        delta = event.angleDelta().y()
        if delta > 0:
            self.setValue(self.value() + 1)
        elif delta < 0:
            self.setValue(self.value() - 1)
        event.accept()


class GradientSlider(ResettableSlider):
    """A horizontal slider whose groove is a left->right colour gradient
    (Temperature blue->amber, Tint green->magenta) so the axis reads at a glance.

    The groove + knob are painted directly rather than via QSS: a global
    ``QSlider`` stylesheet rule, matching this widget through subclassing, would
    otherwise outrank any per-widget / #id groove rule (a Qt cascade quirk), so
    QSS gradients never take here. Custom painting sidesteps that entirely.
    """

    GROOVE_H = 6
    KNOB_D = 14

    def __init__(self, stops, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self._lo = QColor(stops[0])
        self._hi = QColor(stops[1])

    def paintEvent(self, event):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        groove = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
        handle = self.style().subControlRect(
            QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Gradient groove bar, vertically centred.
        gy = self.height() / 2 - self.GROOVE_H / 2
        bar = QRectF(groove.x(), gy, groove.width(), self.GROOVE_H)
        grad = QLinearGradient(bar.left(), 0.0, bar.right(), 0.0)
        grad.setColorAt(0.0, self._lo)
        grad.setColorAt(1.0, self._hi)
        painter.setPen(Qt.NoPen)
        painter.setBrush(grad)
        painter.drawRoundedRect(bar, 3, 3)

        # Light, outlined knob at the handle position.
        knob = QRectF(0, 0, self.KNOB_D, self.KNOB_D)
        knob.moveCenter(QRectF(handle).center())
        painter.setBrush(QColor(theme.Paint.CURVE_NODE))
        painter.setPen(QPen(QColor(theme.Paint.CURVE_NODE_OUTLINE), 1))
        painter.drawEllipse(knob)
        painter.end()


class SlidersPanel(QWidget):
    # Order must match the create_slider() call order exactly — the two
    # lists are zipped positionally.
    ADJUSTMENT_KEYS = [
        "temperature", "tint", "exposure", "brightness", "highlights",
        "white_point", "shadows", "black_point", "contrast", "saturation",
        "sub_saturation",
        # Per-channel levels controls (collapsible section)
        "ch_input_gain", "ch_master_shift", "ch_master_gain",
        "ch_r_shift", "ch_r_gain", "ch_r_blackpoint",
        "ch_g_shift", "ch_g_gain", "ch_g_blackpoint",
        "ch_b_shift", "ch_b_gain", "ch_b_blackpoint",
        # Per-color-band sliders (Subtractive Saturations section, created
        # last): band_<color>_<param> for the color bands × 4 params
    ] + list(BAND_ADJUSTMENT_KEYS) + [
        # Global spatial-feather amount for the band effect (created after the
        # per-band sliders, so it stays last in the positional zip).
        "band_feather",
    ]

    # Non-zero default values for specific adjustment keys. Keys not listed
    # default to 0. band_feather defaults to 10 so band edits get a gentle
    # edge-softening out of the box (matches _BAND_FEATHER_DEFAULT in
    # ccr_processor). Used everywhere a slider is populated from a (possibly
    # partial) adjustment dict, so UI and render agree on the default.
    SLIDER_DEFAULTS = {"band_feather": 10}

    def _default_for(self, key):
        return self.SLIDER_DEFAULTS.get(key, 0)

    # Color Profile combo: row index -> CCRImage.color_profile value.
    COLOR_PROFILES = ("color", "bw")

    def __init__(self, parent=None):
        super().__init__()
        self.sliders = []
        self.slider_value_labels = []
        self.slider_labels = []
        self.image_slider_map = {}
        self.current_image_id = None
        self._layers_sig = None  # cheap signature to avoid needless rebuilds
        self.adjustment_keys = list(self.ADJUSTMENT_KEYS)
        self._sync_group_selection = None  # remembered while the app is open
        self.copied_adjustment = None  # Store copied adjustment settings
        self._hint_timer = QTimer(self)  # Timer for temporary hints
        self._hint_timer.setSingleShot(True)
        
        # Simple processing flag and debouncing
        self._processing = False
        self._pending_adjustment = None
        self._pending_idx = None

        # Coalesce rapid slider changes (one drag) into a single undo step.
        self._undo_burst_active = False
        self._undo_burst_timer = QTimer(self)
        self._undo_burst_timer.setSingleShot(True)
        self._undo_burst_timer.timeout.connect(self._end_undo_burst)

        self.initUI()
        self.setup_shortcuts()
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._process_pending_adjustment)

    def initUI(self):
        # Outer layout: histogram (fixed) + scroll area (stretchy) + hint (fixed)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(theme.GAP_ROW)

        # --- Histogram — fixed at top, outside scroll area ---
        self.histogram_label = QLabel()
        self.histogram_label.setFixedHeight(150)
        self.histogram_label.setAlignment(Qt.AlignCenter)
        self.histogram_label.setFrameShape(QFrame.NoFrame)
        self.histogram_label.setText("")
        self.histogram_label.setStyleSheet(
            f"background-color: rgb({theme.Paint.HIST_BG[0]},{theme.Paint.HIST_BG[1]},{theme.Paint.HIST_BG[2]}); border: none; border-radius: 12px;"
        )
        # Inset to match the content below so it isn't flush against the panel's
        # right border (the scroll area below keeps its scrollbar at the edge).
        hist_row = QHBoxLayout()
        hist_row.setContentsMargins(theme.GAP_PANEL, theme.GAP_PANEL, theme.GAP_PANEL, 0)
        hist_row.addWidget(self.histogram_label)
        layout.addLayout(hist_row)

        # --- Scrollable middle section ---
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_layout.setAlignment(Qt.AlignTop)
        theme.apply_panel_spacing(scroll_layout)
        scroll_content.setLayout(scroll_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area, 1)  # stretch=1: fills remaining height

        self.slider_labels = [
            "Temperature", "Tint", "Gain", "Brightness",
            "Highlights", "White Point", "Shadows", "Black Point", "Contrast", "Saturation",
            "Subtracted Sat"
        ]

        self.current_idx = None

        # --- Layers (Area Editing) — top of the scroll area ---
        # The implicit "Whole Image" layer plus one row per local area. Picking
        # a row re-points every slider/curve below at that layer's settings.
        # New areas are created from the top toolbar (circle/gradient) in
        # ImagePreview; rows here select / enable-disable / delete them.
        self.layers_container = QWidget()
        layers_vbox = QVBoxLayout(self.layers_container)
        layers_vbox.setContentsMargins(0, 0, 0, 0)
        layers_vbox.setSpacing(2)
        layers_title = QLabel("Layers")
        layers_title.setStyleSheet(theme.section_header_qss())
        layers_title.setAlignment(Qt.AlignLeft)
        layers_vbox.addWidget(layers_title)
        self._layers_list_vbox = QVBoxLayout()   # dynamic rows, rebuilt per image
        self._layers_list_vbox.setSpacing(2)
        layers_vbox.addLayout(self._layers_list_vbox)
        # Per-area feather (shown only when a circle area is the active layer).
        self.feather_row = QWidget()
        feather_layout = QHBoxLayout(self.feather_row)
        feather_layout.setContentsMargins(0, 0, 0, 0)
        feather_layout.setSpacing(theme.GAP_TIGHT)
        feather_label = QLabel("Feather")
        feather_label.setFixedWidth(theme.LABEL_COL_W)
        feather_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.feather_slider = ResettableSlider(Qt.Horizontal)
        self.feather_slider.setMinimum(0)
        self.feather_slider.setMaximum(100)
        self.feather_slider.setValue(25)
        self.feather_slider.setFixedHeight(theme.CONTROL_H)
        self.feather_value_label = QLabel("25")
        self.feather_value_label.setFixedWidth(theme.VALUE_COL_W)
        self.feather_slider.valueChanged.connect(self._on_feather_changed)
        feather_layout.addWidget(feather_label)
        feather_layout.addWidget(self.feather_slider)
        feather_layout.addWidget(self.feather_value_label)
        layers_vbox.addWidget(self.feather_row)
        self.feather_row.setVisible(False)
        self._layer_buttons = {}
        scroll_layout.addWidget(self.layers_container)
        scroll_layout.addWidget(theme.section_separator())

        # --- 10 existing sliders (sliders[0]–[9]) ---
        # --- Film B/W Point section — at the top, right below the histogram ---
        bwp_label = QLabel("Film B/W Point")
        bwp_label.setStyleSheet(theme.section_header_qss())
        bwp_label.setAlignment(Qt.AlignLeft)
        scroll_layout.addWidget(bwp_label)

        bwp_row = theme.apply_button_row(QHBoxLayout())
        self.white_point_btn = QPushButton("Set White Point")
        # Small "clear" button: drops the white point so conversion uses the
        # calibrated default slope (black point only).
        self.clear_white_point_btn = QPushButton("✕")
        self.clear_white_point_btn.setFixedWidth(theme.GLYPH_W)
        self.clear_white_point_btn.setFixedHeight(theme.CONTROL_H)
        self.clear_white_point_btn.setToolTip(
            "Clear the white point — use the calibrated default slope instead")
        theme.style_button(self.clear_white_point_btn, "danger", glyph_only=True)
        self.black_point_btn = QPushButton("Set Black Point")
        self.black_point_btn.setFixedHeight(theme.CONTROL_H)
        self.white_point_btn.setFixedHeight(theme.CONTROL_H)
        bwp_row.addWidget(self.black_point_btn)
        bwp_row.addWidget(self.white_point_btn)
        bwp_row.addWidget(self.clear_white_point_btn)
        scroll_layout.addLayout(bwp_row)
        # Shows which slope source the next conversion will use.
        self.bwp_mode_label = QLabel("")
        self.bwp_mode_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px;")
        scroll_layout.addWidget(self.bwp_mode_label)
        self.convert_current_bwp_btn = QPushButton("Convert Current")
        self.convert_all_bwp_btn = QPushButton("Convert All")
        # Convert Current is the section's primary (core single-image action);
        # Convert All stays neutral and gets a count confirmation instead.
        theme.style_button(self.convert_current_bwp_btn, "primary")
        self.convert_current_bwp_btn.setFixedHeight(theme.CONTROL_H)
        self.convert_all_bwp_btn.setFixedHeight(theme.CONTROL_H)
        convert_row = theme.apply_button_row(QHBoxLayout())
        convert_row.addWidget(self.convert_current_bwp_btn)
        convert_row.addWidget(self.convert_all_bwp_btn)
        scroll_layout.addLayout(convert_row)

        # Separator between B/W Point tools and the adjustment sliders
        scroll_layout.addWidget(theme.section_separator())

        # Auto white balance: pick a neutral point with an eyedropper.
        # Enabled only for converted images (same gating as the sliders).
        self.wb_picker_btn = QPushButton("Auto WB Picker")
        self.wb_picker_btn.setToolTip(
            "Click, then pick a neutral gray/white point on the image "
            "to auto-set Temperature and Tint.")
        # Crop: non-destructive crop of the preview/export (same gating).
        self.crop_btn = QPushButton("Crop")
        self.crop_btn.setToolTip(
            "Crop the image. Opens the Crop panel: pick an aspect ratio "
            "(or Free), straighten, and drag the box on the image. Enter "
            "confirms, Esc cancels, right-click clears the crop.")
        # Slice: split one scan containing several photos into separate
        # images. Not conversion-gated — slicing is most useful BEFORE
        # converting, so each frame gets its own reference/conversion.
        self.slice_btn = QPushButton("Slice")
        self.slice_btn.setToolTip(
            "Split a scan containing multiple photos into separate images. "
            "Move along the top/bottom edge for a vertical cut or the "
            "left/right edge for a horizontal cut; click to place a line, "
            "drag to adjust, right-click to delete, Enter to slice, "
            "Esc to cancel.")
        self.wb_picker_btn.setFixedHeight(theme.CONTROL_H)
        self.crop_btn.setFixedHeight(theme.CONTROL_H)
        self.slice_btn.setFixedHeight(theme.CONTROL_H)
        wb_crop_row = theme.apply_button_row(QHBoxLayout())
        wb_crop_row.addWidget(self.wb_picker_btn, 1)
        wb_crop_row.addWidget(self.crop_btn, 1)
        wb_crop_row.addWidget(self.slice_btn, 1)
        scroll_layout.addLayout(wb_crop_row)

        # Color Profile dropdown — sits right above Temperature. "Color" keeps
        # the full-RGB pipeline; "Black & White" maps the result to a single
        # luminance channel (preview and export). Per-image, like the sliders.
        self.color_profile_row = self._create_color_profile_row()

        # Auto White Balance — a checkbox row directly above Temperature. When
        # ticked, a learning-based per-channel gain neutralises the colour cast
        # before the manual sliders; it never moves the slider positions.
        self.awb_row = self._create_awb_row()

        self.temperature_slider_layout = self.create_slider(
            "Temperature", gradient=theme.TEMP_GRADIENT)
        self.tint_slider_layout = self.create_slider(
            "Tint", gradient=theme.TINT_GRADIENT)
        self.exposure_slider_layout = self.create_slider("Gain", min_value=-200, max_value=200)
        self.brightness_slider_layout = self.create_slider("Brightness")
        self.highlights_slider_layout = self.create_slider("Highlights")
        self.white_point_slider_layout = self.create_slider("White Point")
        self.shadows_slider_layout = self.create_slider("Shadows")
        self.black_point_slider_layout = self.create_slider("Black Point")
        self.contrast_slider_layout = self.create_slider("Contrast")
        self.saturation_slider_layout = self.create_slider("Saturation")
        self.sub_saturation_slider_layout = self.create_slider("Subtracted Sat")

        scroll_layout.addLayout(self.color_profile_row)
        scroll_layout.addLayout(self.awb_row)
        scroll_layout.addLayout(self.temperature_slider_layout)
        scroll_layout.addLayout(self.tint_slider_layout)
        scroll_layout.addLayout(self.exposure_slider_layout)
        scroll_layout.addLayout(self.brightness_slider_layout)
        scroll_layout.addLayout(self.highlights_slider_layout)
        scroll_layout.addLayout(self.white_point_slider_layout)
        scroll_layout.addLayout(self.shadows_slider_layout)
        scroll_layout.addLayout(self.black_point_slider_layout)
        scroll_layout.addLayout(self.contrast_slider_layout)
        scroll_layout.addLayout(self.saturation_slider_layout)
        scroll_layout.addLayout(self.sub_saturation_slider_layout)

        # --- Reset / Compare / Sync buttons ---
        buttons_layout = theme.apply_button_row(QHBoxLayout())
        self.reset_button = QPushButton("Reset")
        self.compare_button = QPushButton("Compare")
        # Reset discards all adjustments → danger. Gap so it isn't one block with Compare.
        theme.style_button(self.reset_button, "danger")
        self.reset_button.setFixedHeight(theme.CONTROL_H)
        self.compare_button.setFixedHeight(theme.CONTROL_H)
        buttons_layout.addWidget(self.reset_button)
        buttons_layout.addWidget(self.compare_button)
        scroll_layout.addLayout(buttons_layout)

        sync_layout = QHBoxLayout()
        self.sync_to_all_button = QPushButton("Sync to All")
        self.sync_to_all_button.setFixedHeight(theme.CONTROL_H)
        sync_layout.addWidget(self.sync_to_all_button)
        scroll_layout.addLayout(sync_layout)

        # --- Collapsible sections ---
        # Display order (top→bottom): Curves, Subtractive Saturations, Channel
        # Levels (last). The section WIDGETS are placed here in display order,
        # but their SLIDERS are created further below in the strict order that
        # ADJUSTMENT_KEYS requires (Channel Levels before bands). Placement and
        # population are decoupled because each CollapsibleSection holds its own
        # content layout, so create_slider() append order is independent of where
        # the section sits in scroll_layout.
        def _section_separator():
            return theme.section_separator()

        scroll_layout.addWidget(_section_separator())
        self.curves_section = CollapsibleSection("Curves")
        scroll_layout.addWidget(self.curves_section)

        scroll_layout.addWidget(_section_separator())
        self.band_section = CollapsibleSection("Subtractive Saturations")
        scroll_layout.addWidget(self.band_section)

        scroll_layout.addWidget(_section_separator())
        self.od_section = CollapsibleSection("Channel Levels")
        scroll_layout.addWidget(self.od_section)

        # --- Populate Channel Levels (sliders[10]–[21]) ---
        # MUST be created before the band sliders to keep the ADJUSTMENT_KEYS
        # positional mapping (channel keys precede band keys), regardless of the
        # section's display position above.
        # Master group
        master_label = QLabel("Master")
        master_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px; margin-top: 4px;")
        master_label.setAlignment(Qt.AlignCenter)
        self.od_section.add_widget(master_label)
        self.od_section.add_layout(self.create_slider("Input Gain"))
        self.od_section.add_layout(self.create_slider("Master Shift"))
        self.od_section.add_layout(self.create_slider("Master Gain"))

        # R channel group
        r_label = QLabel("R Channel")
        r_label.setStyleSheet(f"color: {theme.CH_R}; font-size: 11px; margin-top: 4px;")
        r_label.setAlignment(Qt.AlignCenter)
        self.od_section.add_widget(r_label)
        self.od_section.add_layout(self.create_slider("R Shift"))
        self.od_section.add_layout(self.create_slider("R Gain"))
        self.od_section.add_layout(self.create_slider("R Blackpoint"))

        # G channel group
        g_label = QLabel("G Channel")
        g_label.setStyleSheet(f"color: {theme.CH_G}; font-size: 11px; margin-top: 4px;")
        g_label.setAlignment(Qt.AlignCenter)
        self.od_section.add_widget(g_label)
        self.od_section.add_layout(self.create_slider("G Shift"))
        self.od_section.add_layout(self.create_slider("G Gain"))
        self.od_section.add_layout(self.create_slider("G Blackpoint"))

        # B channel group
        b_label = QLabel("B Channel")
        b_label.setStyleSheet(f"color: {theme.CH_B}; font-size: 11px; margin-top: 4px;")
        b_label.setAlignment(Qt.AlignCenter)
        self.od_section.add_widget(b_label)
        self.od_section.add_layout(self.create_slider("B Shift"))
        self.od_section.add_layout(self.create_slider("B Gain"))
        self.od_section.add_layout(self.create_slider("B Blackpoint"))

        # --- Populate Subtractive Saturations (per-color bands) ---
        # A swatch button per color selects which band's sliders are shown;
        # all 24 sliders exist (and feed adjustment_settings) regardless.
        band_swatch_colors = theme.BAND_COLORS
        self._band_buttons = {}
        self._band_pages = {}
        band_btn_row = QHBoxLayout()
        band_btn_row.setSpacing(theme.GAP_BTN)
        for color in COLOR_BANDS:
            btn = QPushButton()
            btn.setCheckable(True)
            btn.setFixedSize(28, 20)
            btn.setToolTip(color.capitalize())
            btn.setStyleSheet(
                f"QPushButton {{ background: {band_swatch_colors[color]}; "
                f"border: 1px solid {theme.BORDER}; border-radius: 3px; }}"
                f"QPushButton:checked {{ border: 2px solid {theme.ACCENT}; }}")
            btn.clicked.connect(lambda _=False, c=color: self._show_band_page(c))
            band_btn_row.addWidget(btn)
            self._band_buttons[color] = btn
        band_btn_row.addStretch()
        band_btn_widget = QWidget()
        band_btn_widget.setLayout(band_btn_row)
        self.band_section.add_widget(band_btn_widget)

        # Slider creation order must mirror BAND_ADJUSTMENT_KEYS: for each
        # color in COLOR_BANDS, the four BAND_PARAMS in order.
        band_param_labels = ("Sub Sat", "Sat", "Brightness", "Hue")
        assert len(band_param_labels) == len(BAND_PARAMS)
        for color in COLOR_BANDS:
            page = QWidget()
            page_layout = QVBoxLayout()
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.setSpacing(2)
            for param_label in band_param_labels:
                page_layout.addLayout(self.create_slider(param_label))
            page.setLayout(page_layout)
            page.setVisible(False)
            self.band_section.add_widget(page)
            self._band_pages[color] = page

        # Global feather (softness): spatially low-passes the band correction
        # to soften hard colour-selection edges. 0 = off. Applies to all
        # bands, so it lives below the per-band pages. Created LAST so it maps
        # to the trailing "band_feather" key in ADJUSTMENT_KEYS.
        self.band_section.add_layout(
            self.create_slider("Feather", min_value=0, max_value=100,
                               default_value=self._default_for("band_feather")))
        self._show_band_page("red")

        # --- Populate Curves ---
        self.curve_editor = CurveEditor()
        self.curves_section.add_widget(self.curve_editor)
        self.curve_editor.curveChanged.connect(self._on_curve_changed)
        self.curve_editor.editFinished.connect(self._on_curve_edit_finished)

        # --- Signal connections ---
        self.reset_button.clicked.connect(self.on_reset_clicked)
        self.compare_button.pressed.connect(self.on_compare_pressed)
        self.compare_button.released.connect(self.on_compare_released)
        self.compare_button.setCheckable(False)
        self.sync_to_all_button.clicked.connect(self.on_sync_to_all_clicked)
        self.wb_picker_btn.clicked.connect(self._on_pick_neutral_point)
        self.crop_btn.clicked.connect(self._on_crop_clicked)
        self.slice_btn.clicked.connect(self._on_slice_clicked)
        self.white_point_btn.clicked.connect(self._on_set_white_point)
        self.clear_white_point_btn.clicked.connect(self._on_clear_white_point)
        self.black_point_btn.clicked.connect(self._on_set_black_point)
        self.convert_current_bwp_btn.clicked.connect(self._on_convert_current_bwpoint)
        self.convert_all_bwp_btn.clicked.connect(self._on_convert_all_bwpoint)

        # --- Hint label — fixed at bottom, outside scroll area ---
        self.hint_label = QLabel()
        self.hint_label.setWordWrap(True)
        self.hint_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 12px; margin-top: 8px;")
        self.hint_label.setText("")
        layout.addWidget(self.hint_label)

        self.setLayout(layout)

    def setup_shortcuts(self):
        """
        Set up keyboard shortcuts for copy/paste functionality.
        """
        # Copy shortcut (Cmd+C on Mac, Ctrl+C on Windows/Linux)
        self.copy_shortcut = QShortcut(QKeySequence.Copy, self)
        self.copy_shortcut.activated.connect(self.copy_adjustment_settings)
        
        # Paste shortcut (Cmd+V on Mac, Ctrl+V on Windows/Linux)
        self.paste_shortcut = QShortcut(QKeySequence.Paste, self)
        self.paste_shortcut.activated.connect(self.paste_adjustment_settings)

    def set_histogram(self, pixmap: QPixmap):
        """
        Set the histogram image in the container.
        """
        if pixmap is not None:
            self.histogram_label.setPixmap(pixmap.scaled(
                self.histogram_label.width(),
                self.histogram_label.height(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            ))
            self.histogram_label.setText("")  # Remove placeholder text
        else:
            self.histogram_label.clear()
            self.histogram_label.setText("")

    def create_slider(self, label_text, min_value=-100, max_value=100,
                      default_value=0, gradient=None):
        # `gradient` (a (left, right) colour pair) gives a custom-painted
        # gradient groove — e.g. Temperature blue->amber, Tint green->magenta.
        if gradient is not None:
            slider = GradientSlider(gradient, Qt.Horizontal)
        else:
            slider = ResettableSlider(Qt.Horizontal)
        slider.setMinimum(min_value)
        slider.setMaximum(max_value)
        slider.setValue(default_value)
        slider.setOrientation(Qt.Horizontal)
        slider.setTickInterval(10)
        slider.setFixedHeight(theme.CONTROL_H)

        label = QLabel(label_text)
        label.setFixedWidth(theme.LABEL_COL_W)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label.setFixedHeight(theme.CONTROL_H)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        value_label = QLabel(str(slider.value()))
        value_label.setFixedWidth(theme.VALUE_COL_W)
        value_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        # Directly call on_slider_changed without debounce
        def handle_slider_change(val, lbl=value_label):
            lbl.setText(str(val))
            self.on_slider_changed()

        slider.valueChanged.connect(handle_slider_change)

        slider_layout = QHBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(theme.GAP_TIGHT)
        slider_layout.addWidget(label, alignment=Qt.AlignVCenter)
        slider_layout.addWidget(slider, alignment=Qt.AlignVCenter)
        slider_layout.addWidget(value_label, alignment=Qt.AlignVCenter)

        self.sliders.append(slider)
        self.slider_value_labels.append(value_label)

        return slider_layout

    def _create_color_profile_row(self):
        """Build the 'Color Profile' label + dropdown row (Color / Black &
        White). Laid out like a slider row so it lines up with Temperature."""
        label = QLabel("Color Profile")
        label.setFixedWidth(theme.LABEL_COL_W)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label.setFixedHeight(theme.CONTROL_H)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.color_profile_combo = QComboBox()
        self.color_profile_combo.addItems(["Color", "Black & White"])
        self.color_profile_combo.setFixedHeight(theme.CONTROL_H)
        self.color_profile_combo.currentIndexChanged.connect(self.on_color_profile_changed)

        row = QHBoxLayout()
        row.setSpacing(theme.GAP_TIGHT)
        row.addWidget(label, alignment=Qt.AlignVCenter)
        row.addWidget(self.color_profile_combo, alignment=Qt.AlignVCenter)
        return row

    def _create_awb_row(self):
        """Build the 'Auto WB' label + checkbox row (sits right above
        Temperature). Laid out like a slider row so the label lines up."""
        from core import awb as _awb
        label = QLabel("Auto WB")
        label.setFixedWidth(theme.LABEL_COL_W)
        label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        label.setFixedHeight(theme.CONTROL_H)
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.awb_checkbox = QCheckBox()
        self.awb_checkbox.setFixedHeight(theme.CONTROL_H)
        reason = _awb.availability_reason()
        # Remembered so set_sliders_enabled never re-enables an unavailable box.
        self._awb_available = not reason
        if reason:
            # Unavailable (onnxruntime / model missing): keep it disabled and
            # explain why, rather than silently doing nothing when ticked.
            self.awb_checkbox.setEnabled(False)
            self.awb_checkbox.setToolTip(reason)
        else:
            self.awb_checkbox.setToolTip(
                "Automatically remove the colour cast. Runs before the manual "
                "sliders and does not change their positions.")
        self.awb_checkbox.toggled.connect(self.on_awb_toggled)

        row = QHBoxLayout()
        row.setSpacing(theme.GAP_TIGHT)
        row.addWidget(label, alignment=Qt.AlignVCenter)
        row.addWidget(self.awb_checkbox, alignment=Qt.AlignVCenter)
        row.addStretch(1)
        return row

    def _sync_awb_checkbox(self, idx):
        """Reflect the image's stored AWB enable flag in the checkbox without
        firing the toggle handler."""
        img = ccr_backend.get_image_by_index(idx) if idx is not None else None
        enabled = bool(getattr(img, "awb_enabled", False)) if img is not None else False
        self.awb_checkbox.blockSignals(True)
        self.awb_checkbox.setChecked(enabled)
        self.awb_checkbox.blockSignals(False)

    def on_awb_toggled(self, checked: bool):
        """Enable/disable Auto White Balance for the current image. A single
        undoable action; reprocesses the preview/thumbnail. Does NOT touch any
        slider value — AWB is a separate gain applied before the sliders."""
        if self.current_idx is None:
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None or bool(getattr(img, "awb_enabled", False)) == bool(checked):
            return
        # Discrete action — don't merge into an in-progress slider undo burst.
        self.end_undo_burst()
        img.push_undo_state()
        img.awb_enabled = bool(checked)
        # Force a fresh estimate on next render (cheap; cached afterwards).
        img.awb_gains = None
        img._awb_src_id = None
        img.update_thumbnail_and_preview()
        mw = self.parent().parent()
        try:
            mw.thumbnail_list.update_thumbnail(self.current_idx)
        except AttributeError:
            pass
        mw.image_preview.update_preview(self.current_idx)

    def _sync_color_profile_combo(self, idx):
        """Reflect the image's stored color profile in the dropdown without
        firing the change handler."""
        img = ccr_backend.get_image_by_index(idx) if idx is not None else None
        profile = getattr(img, "color_profile", "color") if img is not None else "color"
        try:
            row = self.COLOR_PROFILES.index(profile)
        except ValueError:
            row = 0
        self.color_profile_combo.blockSignals(True)
        self.color_profile_combo.setCurrentIndex(row)
        self.color_profile_combo.blockSignals(False)

    def on_color_profile_changed(self, index):
        """Switch the current image between Color and Black & White. A single
        undoable action; reprocesses the preview/thumbnail (and invalidates
        the hi-res zoom cache via the adjustment signature)."""
        if self.current_idx is None:
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None:
            return
        profile = self.COLOR_PROFILES[index] if 0 <= index < len(self.COLOR_PROFILES) else "color"
        if img.color_profile == profile:
            return
        # Discrete action — don't merge into an in-progress slider undo burst.
        self.end_undo_burst()
        img.push_undo_state()
        img.color_profile = profile
        img.update_thumbnail_and_preview()
        mw = self.parent().parent()
        try:
            mw.thumbnail_list.update_thumbnail(self.current_idx)
        except AttributeError:
            pass
        mw.image_preview.update_preview(self.current_idx)

    def _show_band_page(self, color: str):
        """Show one color band's slider page in the Subtractive Saturations
        section; the others stay hidden (but keep their values)."""
        for c, page in self._band_pages.items():
            page.setVisible(c == color)
            self._band_buttons[c].setChecked(c == color)

    def set_sliders_enabled(self, enabled: bool):
        print(f"Setting sliders enabled: {enabled}")
        self.wb_picker_btn.setEnabled(enabled)
        self.crop_btn.setEnabled(enabled)
        self.color_profile_combo.setEnabled(enabled)
        # AWB follows the same conversion gating, but stays disabled when the
        # model/runtime isn't available regardless.
        self.awb_checkbox.setEnabled(enabled and getattr(self, "_awb_available", False))
        self.curve_editor.setEnabled(enabled)
        # Area editing presupposes a converted positive — gate the Layers list
        # with the same flag as the sliders.
        if hasattr(self, "layers_container"):
            self.layers_container.setEnabled(enabled)
        if not enabled:
            self.curve_editor.set_curves(None)
        for slider in self.sliders:
            slider.setEnabled(enabled)
            if not enabled:
                slider.blockSignals(True)
                slider.setValue(0)
                slider.blockSignals(False)
                

    def set_negative_controls_enabled(self, enabled: bool):
        """Enable/disable the film-negative-only controls (the Film B/W Point
        tools). Disabled in global Positive mode, where there is nothing to
        convert. The toolbar's Convert/Un-convert/Auto Frame are gated
        separately in ImagePreview."""
        for btn in (self.white_point_btn, self.black_point_btn,
                    self.convert_current_bwp_btn, self.convert_all_bwp_btn):
            btn.setEnabled(enabled)

    def save_slider_values(self, image_id):
        pass

    # --- Active-layer routing --------------------------------------------
    # The panel edits whichever LAYER is active: the global (whole-image)
    # adjustment_settings, or the selected area's own settings. These route
    # reads/writes through the backend's layer-aware accessors so the rest of
    # the panel needs no special-casing.
    def _read_active_settings(self, idx) -> dict:
        s = ccr_backend.get_active_settings_by_index(idx)
        return s if s is not None else {}

    def _store_active_settings(self, idx, adjustment):
        """Light, no-reprocess store of the active layer's settings (heavy
        reprocess is debounced separately)."""
        ccr_backend.set_active_settings_by_index(idx, adjustment, reprocess=False)

    def _load_active_layer(self, idx):
        """Populate the sliders + curve editor from the active layer's
        settings (global or area). Mirrors the populate logic set_current_idx
        used to do inline, but sourced from the active layer."""
        adjustment = self._read_active_settings(idx)
        # Skip reloading curves while a curve drag is active: update_preview()
        # re-enters here for the SAME image and reloading would clear the drag.
        if not self.curve_editor.is_dragging():
            self.curve_editor.set_curves(adjustment.get("curves") if adjustment else None)
        for i, key in enumerate(self.adjustment_keys):
            if i < len(self.sliders):
                val = adjustment.get(key, self._default_for(key)) if adjustment \
                    else self._default_for(key)
                self.sliders[i].blockSignals(True)
                self.sliders[i].setValue(val)
                self.sliders[i].blockSignals(False)
                self.slider_value_labels[i].setText(str(val))

    def set_current_idx(self, idx):
        # Clear any pending adjustments for the previous image
        self._pending_adjustment = None
        self._pending_idx = None
        self._debounce_timer.stop()
        # End the undo burst only on a real image switch — this method is
        # also re-entered on same-image refreshes during a slider drag.
        if idx != self.current_idx:
            self._end_undo_burst()
            self._undo_burst_timer.stop()

        self.current_idx = idx
        # Reflect this image's color profile (independent of the slider dict,
        # so it must be synced on both the empty and populated paths below).
        self._sync_color_profile_combo(idx)
        # AWB enable is likewise a per-image flag outside the slider dict.
        self._sync_awb_checkbox(idx)
        # Rebuild the Layers list only when its structure actually changed
        # (image switch, area added/removed/toggled/selected) — set_current_idx
        # is re-entered on every preview refresh (incl. each slider tick), and
        # recreating the row widgets every time would flicker.
        sig = self._layers_signature(idx)
        if sig != self._layers_sig:
            self._rebuild_layers_list(idx)
        adjustment = self._read_active_settings(idx)
        print(f"Setting current index: {idx}, active adjustment: {adjustment}")
        self._load_active_layer(idx)
        # Reprocess only when the active layer carries edits (mirrors the old
        # populated-path behavior; a blank image is already rendered elsewhere).
        if idx is not None and adjustment:
            ccr_backend.apply_adjustment_by_index(idx)

    # --- Layers list (area editing) ---------------------------------------
    def _ip(self):
        """The ImagePreview, set by MainWindow; None outside the full app."""
        return getattr(self, "image_preview", None)

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                # deleteLater (not immediate free): a rebuild can be triggered
                # from inside a row widget's own signal (e.g. the enable
                # checkbox's toggled), and freeing that widget synchronously
                # would crash with a use-after-free. Detach now, delete later.
                w.setParent(None)
                w.deleteLater()
            else:
                sub = item.layout()
                if sub is not None:
                    SlidersPanel._clear_layout(sub)

    def _layers_signature(self, idx):
        """Cheap structural signature of the Layers list — changes only when a
        rebuild is actually needed (image switch, area add/remove/toggle, or
        active-layer change), NOT on every slider tick."""
        img = ccr_backend.get_image_by_index(idx) if idx is not None else None
        if img is None:
            return None
        # NOTE: 'enabled' is intentionally excluded — toggling a checkbox does
        # not change the list STRUCTURE, so it must not trigger a rebuild (the
        # rebuild would tear down the checkbox mid-signal). Only add/remove/
        # reorder/active-change/image-switch should rebuild.
        return (id(img), img.active_area_id,
                tuple((a.get("id"), a.get("kind")) for a in img.area_layers))

    def _rebuild_layers_list(self, idx):
        """Rebuild the Layers rows (Whole Image + one per area) and sync the
        feather control for the active layer."""
        self._layers_sig = self._layers_signature(idx)
        self._clear_layout(self._layers_list_vbox)
        self._layer_buttons = {}
        img = ccr_backend.get_image_by_index(idx) if idx is not None else None
        if img is None:
            self.feather_row.setVisible(False)
            return
        active = img.active_area_id
        whole = QPushButton("Whole Image")
        whole.setCheckable(True)
        whole.setChecked(active is None)
        whole.clicked.connect(lambda _=False: self._select_layer(None))
        self._layers_list_vbox.addWidget(whole)
        self._layer_buttons[None] = whole
        counts = {}
        for a in img.area_layers:
            kind = a.get("kind", "circle")
            counts[kind] = counts.get(kind, 0) + 1
            glyph = "○" if kind == "circle" else "▤"
            aid = a.get("id")
            row = QHBoxLayout()
            row.setSpacing(theme.GAP_BTN)
            chk = QCheckBox()
            chk.setChecked(bool(a.get("enabled", True)))
            chk.setToolTip("Enable / disable this area")
            chk.toggled.connect(lambda on, _id=aid: self._on_area_enabled(_id, on))
            sel = QPushButton(f"{glyph} {kind.capitalize()} {counts[kind]}")
            sel.setCheckable(True)
            sel.setChecked(active == aid)
            sel.clicked.connect(lambda _=False, _id=aid: self._select_layer(_id))
            rem = QPushButton("✕")
            rem.setFixedWidth(theme.GLYPH_W)
            rem.setToolTip("Remove this area")
            rem.clicked.connect(lambda _=False, _id=aid: self._on_remove_area(_id))
            row.addWidget(chk)
            row.addWidget(sel, 1)
            row.addWidget(rem)
            row_w = QWidget()
            row_w.setLayout(row)
            self._layers_list_vbox.addWidget(row_w)
            self._layer_buttons[aid] = sel
        # Feather control: only meaningful for a circle area (the gradient's
        # softness is defined by its two endpoints).
        act = img.get_area(active)
        show_feather = act is not None and act.get("kind") == "circle"
        self.feather_row.setVisible(show_feather)
        if show_feather:
            fv = int(round(float(act.get("feather", 0.25)) * 100))
            self.feather_slider.blockSignals(True)
            self.feather_slider.setValue(max(0, min(100, fv)))
            self.feather_slider.blockSignals(False)
            self.feather_value_label.setText(str(fv))

    def refresh_layers(self, idx):
        """Called by ImagePreview after an area is added/changed externally:
        rebuild the rows and reload the (now-active) layer into the sliders."""
        if idx != self.current_idx:
            return
        self._rebuild_layers_list(idx)
        self._load_active_layer(idx)

    def _select_layer(self, area_id):
        """Make a layer (None = Whole Image, else an area id) the edit target."""
        if self.current_idx is None:
            return
        self.end_undo_burst()
        ccr_backend.set_active_area_by_index(self.current_idx, area_id)
        self._rebuild_layers_list(self.current_idx)
        self._load_active_layer(self.current_idx)
        ip = self._ip()
        if ip is not None and hasattr(ip, "on_active_layer_changed"):
            ip.on_active_layer_changed(self.current_idx)

    def _on_area_enabled(self, area_id, enabled):
        if self.current_idx is None:
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None:
            return
        self.end_undo_burst()
        img.push_undo_state()
        ccr_backend.set_area_enabled_by_index(self.current_idx, area_id, enabled)
        ip = self._ip()
        if ip is not None:
            ip.update_preview(self.current_idx)
        self._update_thumb()

    def _on_remove_area(self, area_id):
        if self.current_idx is None:
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None:
            return
        self.end_undo_burst()
        img.push_undo_state()
        ccr_backend.remove_area_by_index(self.current_idx, area_id)
        self._rebuild_layers_list(self.current_idx)
        self._load_active_layer(self.current_idx)
        ip = self._ip()
        if ip is not None:
            if hasattr(ip, "on_active_layer_changed"):
                ip.on_active_layer_changed(self.current_idx)
            ip.update_preview(self.current_idx)
        self._update_thumb()

    def _on_feather_changed(self, val):
        self.feather_value_label.setText(str(val))
        if self.current_idx is None:
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None or img.active_area_id is None:
            return
        self._begin_undo_burst(img)
        ccr_backend.update_area_geometry_by_index(
            self.current_idx, img.active_area_id, feather=val / 100.0,
            reprocess=False)
        ip = self._ip()
        if ip is not None:
            ip.update_preview(self.current_idx)
        # Debounce the heavy reprocess like a slider drag.
        self._pending_adjustment = self._read_active_settings(self.current_idx)
        self._pending_idx = self.current_idx
        self._debounce_timer.stop()
        self._debounce_timer.start(150)

    def _update_thumb(self):
        try:
            self.parent().parent().thumbnail_list.update_thumbnail(self.current_idx)
        except AttributeError:
            pass


    def on_slider_changed(self):
        """
        Save the current slider values to the backend when any slider changes.
        Provides immediate visual feedback while debouncing heavy processing.
        """
        if self.current_idx is not None:
            adjustment = {key: slider.value() for key, slider in zip(self.adjustment_keys, self.sliders)}
            self._attach_curves(adjustment)

            # Immediate lightweight feedback - just store the adjustment settings
            if 0 <= self.current_idx < len(ccr_backend.images):
                # Snapshot the pre-change state once per burst so a whole
                # slider drag undoes as a single Ctrl+Z step.
                self._begin_undo_burst(ccr_backend.images[self.current_idx])
                self._store_active_settings(self.current_idx, adjustment)

            # Immediate preview update for visual feedback
            self.parent().parent().image_preview.update_preview(self.current_idx)
            
            # Store the pending adjustment for debounced heavy processing
            self._pending_adjustment = adjustment
            self._pending_idx = self.current_idx
            self._debounce_timer.stop()
            self._debounce_timer.start(150)  # Slightly longer debounce for heavy processing
    
    def _attach_curves(self, adjustment: dict) -> dict:
        """Re-attach the live tone-curve state from the editor onto a freshly
        rebuilt adjustment dict (the slider->dict rebuild drops the nested
        'curves' key otherwise). No-op for identity curves."""
        curves = self.curve_editor.get_curves()
        if curves:
            adjustment["curves"] = curves
        return adjustment

    def _on_curve_changed(self):
        """Live tone-curve edit — mirror on_slider_changed's feedback +
        debounced heavy-processing path so a curve drag behaves like a slider
        drag (single undo burst, coalesced reprocessing)."""
        if self.current_idx is None:
            return
        adjustment = {key: slider.value() for key, slider in zip(self.adjustment_keys, self.sliders)}
        self._attach_curves(adjustment)
        if 0 <= self.current_idx < len(ccr_backend.images):
            self._begin_undo_burst(ccr_backend.images[self.current_idx])
            self._store_active_settings(self.current_idx, adjustment)
        self.parent().parent().image_preview.update_preview(self.current_idx)
        self._pending_adjustment = adjustment
        self._pending_idx = self.current_idx
        self._debounce_timer.stop()
        self._debounce_timer.start(150)

    def _on_curve_edit_finished(self):
        """Settle a finished curve edit (drag release, add/remove, Reset Curve).

        update_preview() displays the cached resized_preview and only
        regenerates it *afterwards* (via set_current_idx), so the live
        _on_curve_changed path leaves the final state one render behind — for a
        discrete edit nothing redraws it, so the image looked unchanged until the
        user clicked the canvas. Mirror on_reset_clicked: regenerate the preview
        first, then display it, so the result is shown immediately."""
        self.end_undo_burst()
        if self.current_idx is None or not (0 <= self.current_idx < len(ccr_backend.images)):
            return
        # Cancel the pending debounced reprocess — we render the final state now.
        self._debounce_timer.stop()
        self._pending_adjustment = None
        self._pending_idx = None
        ccr_backend.images[self.current_idx].update_thumbnail_and_preview()
        self.parent().parent().image_preview.update_preview(self.current_idx)
        try:
            self.parent().parent().thumbnail_list.update_thumbnail(self.current_idx)
        except AttributeError:
            pass

    def _process_pending_adjustment(self):
        """Process the pending adjustment if not already processing."""
        if not self._processing and self._pending_adjustment is not None:
            self._processing = True
            
            # Use QTimer to process in the next event loop iteration
            QTimer.singleShot(0, self._do_backend_processing)
    
    def _do_backend_processing(self):
        """Perform the heavier backend processing operations (thumbnail updates, etc.)."""
        # Capture which index to process and clear pending so new changes can queue up
        idx = self._pending_idx
        self._pending_adjustment = None
        self._pending_idx = None
        try:
            if idx is not None and 0 <= idx < len(ccr_backend.images):
                ccr_backend.images[idx].update_thumbnail_and_preview()
        finally:
            self._processing = False
            # Process any adjustment that arrived while we were working
            QTimer.singleShot(0, self._check_for_pending)
    
    def _check_for_pending(self):
        """Check if there's another pending adjustment to process."""
        if not self._processing and self._pending_adjustment is not None:
            self._process_pending_adjustment()

    def get_slider_values(self):
        return {key: slider.value() for key, slider in zip(self.adjustment_keys, self.sliders)}

    def _begin_undo_burst(self, img):
        """Push one undo snapshot at the start of a burst of rapid slider
        changes; the burst ends after a short idle period."""
        if not self._undo_burst_active:
            img.push_undo_state()
            self._undo_burst_active = True
        self._undo_burst_timer.start(800)

    def _end_undo_burst(self):
        self._undo_burst_active = False

    def end_undo_burst(self):
        """End any in-progress slider undo burst (after an undo or an action
        that pushes its own snapshot) so the next change gets a fresh one."""
        self._end_undo_burst()
        self._undo_burst_timer.stop()

    def on_reset_clicked(self):
        # Reset is a single undoable action
        self.end_undo_burst()
        img = ccr_backend.get_image_by_index(self.current_idx) if self.current_idx is not None else None
        if img is not None:
            img.push_undo_state()
            # Reset clears the crop too, but only when resetting the Whole Image
            # layer — the crop is an image-global property, so resetting an
            # individual area leaves it (and the crop) untouched. Folded into the
            # single undo snapshot pushed above.
            if img.active_area_id is None and (img.crop_rect is not None
                                               or getattr(img, "crop_angle", 0.0)):
                img.crop_rect = None
                img.crop_angle = 0.0
            # Reset also turns AWB off (whole-image property), folded into the
            # same undo snapshot. The checkbox is re-synced below.
            if img.active_area_id is None:
                img.awb_enabled = False
                img.awb_gains = None
                img._awb_src_id = None
                self.awb_checkbox.blockSignals(True)
                self.awb_checkbox.setChecked(False)
                self.awb_checkbox.blockSignals(False)
        # Reset every slider to its default (0 for most, 10 for band_feather).
        for i, slider in enumerate(self.sliders):
            key = self.adjustment_keys[i] if i < len(self.adjustment_keys) else None
            default = self._default_for(key)
            slider.blockSignals(True)
            slider.setValue(default)
            slider.blockSignals(False)
            self.slider_value_labels[i].setText(str(default))
        # Full reset clears tone curves too (the dedicated "Reset Curve" button
        # is the curve-only path). set_curves does not emit, so it won't start a
        # competing undo burst here.
        self.curve_editor.set_curves(None)
        # Save adjustment to the ACTIVE layer (global or selected area) and
        # update the preview. Resetting an area zeroes that area's settings;
        # resetting the Whole Image layer zeroes the global sliders (areas keep
        # their own state and have their own delete control).
        if self.current_idx is not None:
            adjustment = {key: 0 for key in self.adjustment_keys}
            ccr_backend.set_active_settings_by_index(self.current_idx, adjustment,
                                                     reprocess=True)
            self.parent().parent().image_preview.update_preview(self.current_idx)

    def on_compare_pressed(self):
        # Temporarily show the fully UNADJUSTED positive while the button is
        # held: zero the global sliders AND suppress every area layer (without
        # saving). Areas are DISABLED in place (not removed) so the area stays
        # present — keeping area-edit mode and its overlay alive during compare.
        # `_original_adjustment` doubles as the guard main_window checks to
        # suppress Undo during a compare hold.
        if self.current_idx is None:
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None:
            return
        # Remember what to restore: the active layer's settings (to refill the
        # sliders), the live global dict, and each area's enabled state.
        self._original_adjustment = self._read_active_settings(self.current_idx)
        self._compare_global = img.adjustment_settings
        self._compare_area_enabled = [(a, a.get("enabled", True))
                                      for a in img.area_layers]
        img.adjustment_settings = {}
        for a in img.area_layers:
            a["enabled"] = False
        for i, slider in enumerate(self.sliders):
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            self.slider_value_labels[i].setText("0")
        img.update_thumbnail_and_preview()
        self.parent().parent().image_preview.update_preview(self.current_idx)

    def on_compare_released(self):
        # Restore the global dict + area enabled states and refill sliders.
        if self.current_idx is None or not hasattr(self, "_original_adjustment"):
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is not None and hasattr(self, "_compare_global"):
            img.adjustment_settings = self._compare_global
            for a, enabled in self._compare_area_enabled:
                a["enabled"] = enabled
            img.update_thumbnail_and_preview()
        adjustment = self._original_adjustment or {}
        for i, key in enumerate(self.adjustment_keys):
            val = adjustment.get(key, self._default_for(key))
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(val)
            self.sliders[i].blockSignals(False)
            self.slider_value_labels[i].setText(str(val))
        self.parent().parent().image_preview.update_preview(self.current_idx)
        del self._original_adjustment
        if hasattr(self, "_compare_global"):
            del self._compare_global
            del self._compare_area_enabled

    def on_sync_to_all_clicked(self):
        """
        Apply the current image's settings to all images. A dialog picks
        which setting groups to sync; the choice is remembered while the
        app is open.
        """
        if self.current_idx is None:
            return
        dialog = SyncSettingsDialog(self, self._sync_group_selection)
        if dialog.exec_() != QDialog.Accepted:
            return
        self._sync_group_selection = dialog.selection()
        if not any(self._sync_group_selection.values()):
            self.set_temporary_hint("Nothing selected to sync.", duration=3000)
            return
        # Show syncing hint
        self.set_hint("Syncing settings to all images...")

        # Use QTimer to allow UI to update before starting the operation
        QTimer.singleShot(100, self._perform_sync_to_all)

    def _perform_sync_to_all(self):
        """Sync the selected setting groups from the current image to all
        images, leaving each image's un-synced groups untouched."""
        self.end_undo_burst()
        selection = self._sync_group_selection or {gid: True for gid, _l, _k in SYNC_GROUPS}
        keys = [k for gid, _label, group_keys in SYNC_GROUPS
                if selection.get(gid) for k in group_keys]
        sync_crop = bool(selection.get("crop"))
        sync_profile = bool(selection.get("profile"))
        sync_awb = bool(selection.get("awb"))
        sync_curves = bool(selection.get("curves"))
        # Sync always copies the SOURCE image's GLOBAL (whole-image) layer, not
        # the live sliders — those may currently reflect an active area, and
        # areas are per-image (never synced). Read from the global dict.
        src = ccr_backend.get_image_by_index(self.current_idx)
        src_global = dict(src.adjustment_settings) if src is not None else {}
        src_curves = src_global.get("curves")  # None when identity/absent
        current_adjustment = src_global
        crop_rect = src.crop_rect if src is not None else None
        crop_angle = getattr(src, "crop_angle", 0.0) if src is not None else 0.0
        src_profile = getattr(src, "color_profile", "color") if src is not None else "color"
        src_awb = bool(getattr(src, "awb_enabled", False)) if src is not None else False
        print(f"Syncing groups {sorted(g for g, on in selection.items() if on)} to all images")

        for img in ccr_backend.images:
            adj_changes = any(img.adjustment_settings.get(k, self._default_for(k))
                              != current_adjustment.get(k, self._default_for(k))
                              for k in keys)
            crop_changes = sync_crop and (img.crop_rect != crop_rect
                                          or getattr(img, "crop_angle", 0.0) != crop_angle)
            profile_changes = sync_profile and getattr(img, "color_profile", "color") != src_profile
            awb_changes = sync_awb and bool(getattr(img, "awb_enabled", False)) != src_awb
            curves_changes = sync_curves and img.adjustment_settings.get("curves") != src_curves
            if (not adj_changes and not crop_changes and not profile_changes
                    and not awb_changes and not curves_changes):
                continue  # nothing to change — and no dead undo snapshot
            img.push_undo_state()
            if adj_changes:
                # Build a COMPLETE dict (missing keys filled with 0) so the
                # rest of the app keeps its invariant that a non-empty
                # adjustment dict carries every key — set_current_idx and
                # friends rely on it.
                merged = {k: img.adjustment_settings.get(k, self._default_for(k))
                          for k in self.adjustment_keys}
                for k in keys:
                    merged[k] = current_adjustment.get(k, self._default_for(k))
                # Preserve the target's own tone curves across the slider-only
                # rebuild (they're synced separately, just below).
                existing_curves = img.adjustment_settings.get("curves")
                if existing_curves is not None:
                    merged["curves"] = existing_curves
                img.adjustment_settings = merged
            if curves_changes:
                if src_curves:
                    img.adjustment_settings["curves"] = copy.deepcopy(src_curves)
                else:
                    img.adjustment_settings.pop("curves", None)
            if sync_crop:
                img.crop_rect = crop_rect
                img.crop_angle = crop_angle
            if profile_changes:
                img.color_profile = src_profile
            if awb_changes:
                img.awb_enabled = src_awb
                img.awb_gains = None       # recompute per-image on next render
                img._awb_src_id = None
            if adj_changes or profile_changes or awb_changes or curves_changes or crop_changes:
                # Adjustments, curves, the color profile, and AWB all change
                # pixels; a crop change moves the region the histogram covers
                # (it samples only the cropped area). Any of these needs a
                # reprocess so the cached preview/histogram stays in step.
                try:
                    img.update_thumbnail_and_preview()
                except Exception as e:
                    print(f"Failed to sync settings to {img.file_path}: {e}")

        mw = self.parent().parent()
        try:
            mw.thumbnail_list.update_all_thumbnails()
        except AttributeError:
            pass
        mw.image_preview.update_preview(self.current_idx)

        # Show completion hint
        self.set_temporary_hint("Synced selected settings to all images!", duration=4000)

    def _on_pick_neutral_point(self):
        if hasattr(self, 'image_preview') and self.image_preview:
            self.image_preview.set_wb_pick_mode(True)
            self.set_temporary_hint(
                "<b>Auto WB:</b> Click a neutral gray or white point on the image.", duration=8000)

    def _on_crop_clicked(self):
        if not (hasattr(self, 'image_preview') and self.image_preview):
            return
        # Opening crop swaps the sliders panel for the dedicated Crop panel
        # (aspect ratios + straighten); MainWindow owns that swap.
        mw = self.window()
        if mw is None or not hasattr(mw, "toggle_crop_panel"):
            # Fallback (e.g. tests with a stub host): drive crop mode directly.
            if self.image_preview.crop_mode:
                self.image_preview.cancel_crop_mode()
            else:
                self.image_preview.enter_crop_mode()
            return
        mw.toggle_crop_panel(not self.image_preview.crop_mode)

    def _on_slice_clicked(self):
        if not (hasattr(self, 'image_preview') and self.image_preview):
            return
        # Toggle: clicking Slice again leaves slice mode without slicing
        if self.image_preview.slice_mode:
            self.image_preview.cancel_slice_mode()
            return
        if self.image_preview.enter_slice_mode():
            self.set_temporary_hint(
                "<b>Slice:</b> move near the top/bottom rim for a vertical "
                "cut, the left/right rim for a horizontal cut. Click = place "
                "line, drag = adjust, right-click = delete, <b>Enter</b> = "
                "slice, <b>Esc</b> = cancel.", duration=12000)

    def on_wb_sampled(self, temp_value, tint_value):
        """Apply the auto-computed temperature/tint from the WB eyedropper."""
        # The WB pick is its own undo step — don't merge it into a slider burst
        self.end_undo_burst()
        temp_idx = self.adjustment_keys.index("temperature")
        tint_idx = self.adjustment_keys.index("tint")
        for idx, val in ((temp_idx, temp_value), (tint_idx, tint_value)):
            self.sliders[idx].blockSignals(True)
            self.sliders[idx].setValue(val)
            self.sliders[idx].blockSignals(False)
            self.slider_value_labels[idx].setText(str(val))
        self.on_slider_changed()
        self.set_temporary_hint(
            f"Auto WB applied — Temperature: {temp_value}, Tint: {tint_value}.", duration=5000)

    def _on_set_white_point(self):
        if hasattr(self, 'image_preview') and self.image_preview:
            self.image_preview.set_bwpoint_mode("white")
            self.set_temporary_hint(
                "<b>White Point:</b> Draw a rect over the dense/exposed film area.", duration=6000)

    def _on_set_black_point(self):
        if hasattr(self, 'image_preview') and self.image_preview:
            self.image_preview.set_bwpoint_mode("black")
            self.set_temporary_hint(
                "<b>Black Point:</b> Draw a rect over the transparent/clear film base.", duration=6000)

    def _on_clear_white_point(self):
        """Clear BOTH sampled B/W points (black film base + white dense point),
        resetting the Film B/W Point section."""
        ccr_backend.clear_black_point()
        ccr_backend.clear_white_point()
        mw = self.parent().parent()
        if hasattr(mw, "persist_bwpoint"):
            mw.persist_bwpoint()
        self._update_bwp_mode_label()
        self.set_temporary_hint(
            "B/W points cleared. Set a <b>Black Point</b> (film base) "
            "before converting.", duration=6000)

    def _update_bwp_mode_label(self):
        """Reflect which slope source the next B/W-point conversion will use."""
        if not hasattr(self, "bwp_mode_label"):
            return
        bp_set = ccr_backend.black_point_bgr is not None
        wp_set = ccr_backend.white_point_bgr is not None
        if not bp_set:
            self.bwp_mode_label.setText("")
        elif wp_set:
            self.bwp_mode_label.setText("Slope source: white point (two-point)")
        else:
            self.bwp_mode_label.setText("Slope source: default slope (black point only)")

    def on_bwpoint_sampled(self, mode):
        label = "White Point" if mode == "white" else "Black Point"
        bp_set = ccr_backend.black_point_bgr is not None
        wp_set = ccr_backend.white_point_bgr is not None
        # Persist the global B/W point so tethering (and the next session) can
        # reuse it. See spec/camera-tethering.md §4.1.
        mw = self.parent().parent()
        if hasattr(mw, "persist_bwpoint"):
            mw.persist_bwpoint()
        self._update_bwp_mode_label()
        if bp_set and wp_set:
            self.set_temporary_hint(
                f"{label} sampled! Both points set — click <b>Convert All</b>.", duration=5000)
        elif bp_set:
            # Black point alone is enough — default slope fills in for the white.
            self.set_temporary_hint(
                "Black Point sampled! Click <b>Convert</b> to use the default slope, "
                "or set a <b>White Point</b> for a custom slope.", duration=6000)
        else:
            self.set_temporary_hint(
                "White Point sampled! Now set the <b>Black Point</b> (film base).", duration=5000)

    def _on_convert_current_bwpoint(self):
        if ccr_backend.black_point_bgr is None:
            QMessageBox.warning(self, "Black Point Missing",
                "Please set a Black Point (film base) before converting. A White "
                "Point is optional — without it the calibrated default slope is used.")
            return
        if self.current_idx is None:
            return
        img = ccr_backend.get_image_by_index(self.current_idx)
        if img is None:
            return
        try:
            if img.converted:
                img.reload_image()
            from core.ccr_processor import ccr_normalize_with_bwpoint
            white = ccr_backend.white_point_bgr  # may be None → default slope
            processed = ccr_normalize_with_bwpoint(
                img, ccr_backend.black_point_bgr, white
            )
            if processed is not None:
                img.resized_raw = processed
            img.converted = True
            img.conversion_inputs = {
                "mode": "bw",
                "bw": (tuple(ccr_backend.black_point_bgr),
                       tuple(white) if white is not None else None),
                "fine_rot": img.fine_rotation_angle,
            }
            img.update_thumbnail_and_preview()
            mw = self.parent().parent()
            mw.thumbnail_list.update_all_thumbnails()
            mw.image_preview.update_preview(self.current_idx)
            mw.image_preview._update_unconvert_action_state()
            ccr_backend.save_catalog()
            self.set_temporary_hint("Current image converted!", duration=3000)
        except Exception as e:
            QMessageBox.critical(self, "Conversion Error", str(e))

    def _on_convert_all_bwpoint(self):
        if ccr_backend.black_point_bgr is None:
            QMessageBox.warning(self, "Black Point Missing",
                "Please set a Black Point (film base) before converting. A White "
                "Point is optional — without it the calibrated default slope is used.")
            return
        n = len(ccr_backend.images)
        if n == 0:
            return
        if QMessageBox.question(
                self, "Convert All Images",
                f"Convert all {n} image{'s' if n != 1 else ''} with the current "
                "black/white point? Any existing conversions will be replaced.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        dialog = BWPointConvertDialog(self)
        worker = BWPointConvertWorker()
        dialog.set_worker(worker)
        worker.progress.connect(dialog.set_progress)
        worker.finished.connect(lambda: self._on_bwp_convert_finished(dialog))
        worker.start()
        dialog.exec_()

    def _on_bwp_convert_finished(self, dialog):
        dialog.accept()
        try:
            mw = self.parent().parent()
            mw.thumbnail_list.update_all_thumbnails()
            if self.current_idx is not None:
                mw.image_preview.update_preview(self.current_idx)
                mw.image_preview._update_unconvert_action_state()
        except AttributeError:
            pass
        ccr_backend.save_catalog()
        self.set_temporary_hint("B/W Point conversion complete!", duration=3000)

    def copy_adjustment_settings(self):
        """
        Copy the current adjustment settings to clipboard.
        """
        if self.current_idx is not None:
            # Get the current adjustment settings from sliders
            self.copied_adjustment = {key: slider.value() for key, slider in zip(self.adjustment_keys, self.sliders)}
            self._attach_curves(self.copied_adjustment)
            print(f"Copied adjustment settings: {self.copied_adjustment}")
            self.set_temporary_hint("Adjustments Copied!", duration=4000)
        else:
            print("No image selected to copy adjustment settings from.")
            self.set_temporary_hint("No image selected to copy from", duration=4000)

    def paste_adjustment_settings(self):
        """
        Paste the copied adjustment settings to the current image.
        """
        if self.current_idx is not None and self.copied_adjustment is not None:
            print(f"Pasting adjustment settings: {self.copied_adjustment}")
            self.end_undo_burst()
            img = ccr_backend.get_image_by_index(self.current_idx)
            if img is not None:
                img.push_undo_state()

            # Apply the copied settings to the current sliders
            for i, key in enumerate(self.adjustment_keys):
                if key in self.copied_adjustment and i < len(self.sliders):
                    self.sliders[i].blockSignals(True)
                    self.sliders[i].setValue(self.copied_adjustment[key])
                    self.sliders[i].blockSignals(False)
                    self.slider_value_labels[i].setText(str(self.copied_adjustment[key]))
            # Mirror the pasted tone curves into the editor (set_curves does not
            # emit, so it won't re-trigger a save).
            self.curve_editor.set_curves(self.copied_adjustment.get("curves"))

            # Save the adjustment to the ACTIVE layer and update preview. Copy
            # the dict so the clipboard isn't aliased by the image's settings.
            ccr_backend.set_active_settings_by_index(
                self.current_idx, copy.deepcopy(self.copied_adjustment),
                reprocess=True)
            self.parent().parent().image_preview.update_preview(self.current_idx)
            self.set_temporary_hint("Adjustments Pasted!", duration=2000)
            
        elif self.copied_adjustment is None:
            print("No adjustment settings to paste. Copy settings first with Cmd+C (or Ctrl+C).")
            self.set_temporary_hint("No adjustments to paste. Copy first with Cmd+C", duration=3000)
        else:
            print("No image selected to paste adjustment settings to.")
            self.set_temporary_hint("No image selected to paste to", duration=2000)

    # --- Dynamic Hint Management Methods ---
    def set_hint(self, message, temporary=False, duration=3000):
        """
        Set a hint message in the hint label.
        
        Args:
            message (str): The hint message to display
            temporary (bool): If True, the hint will be cleared after duration
            duration (int): Duration in milliseconds for temporary hints (default: 3000ms)
        """
        self.hint_label.setText(message)
        
        if temporary:
            self._hint_timer.stop()  # Stop any existing timer
            self._hint_timer.timeout.connect(lambda: self.clear_hint())
            self._hint_timer.start(duration)
    
    def clear_hint(self):
        """
        Clear the hint message.
        """
        self.hint_label.setText("")
    
    def set_temporary_hint(self, message, duration=3000):
        """
        Set a temporary hint that will automatically clear after duration.
        
        Args:
            message (str): The hint message to display
            duration (int): Duration in milliseconds (default: 3000ms)
        """
        self.set_hint(message, temporary=True, duration=duration)


class BWPointConvertWorker(QThread):
    finished = Signal()
    progress = Signal(int, int)

    def __init__(self):
        super().__init__()
        self._stop_requested = False

    def run(self):
        def progress_callback(current, total):
            if not self._stop_requested:
                self.progress.emit(current, total)
        try:
            ccr_backend.apply_bwpoint_to_all_images(progress_callback=progress_callback)
        except Exception as e:
            print(f"B/W point batch conversion failed: {e}")
        self.finished.emit()

    def stop(self):
        self._stop_requested = True


class BWPointConvertDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Converting...")
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setWindowModality(Qt.ApplicationModal)
        self.setMinimumWidth(theme.DIALOG_W_SM)

        self.label = QLabel("Applying B/W point conversion", self)
        self.label.setAlignment(Qt.AlignCenter)

        self.progress_label = QLabel("", self)
        self.progress_label.setAlignment(Qt.AlignCenter)

        self.stop_button = QPushButton("Stop", self)
        self.stop_button.clicked.connect(self._on_stop)

        layout = QVBoxLayout()
        theme.apply_panel_spacing(layout)
        layout.addWidget(self.label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.stop_button)
        self.setLayout(layout)

        self._dot_count = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._animate)
        self._timer.start(400)
        self.worker = None

    def set_worker(self, worker):
        self.worker = worker

    def set_progress(self, current, total):
        self.progress_label.setText(f"{current} / {total}")

    def _animate(self):
        self._dot_count = (self._dot_count + 1) % 4
        self.label.setText("Applying B/W point conversion" + "." * self._dot_count)

    def _on_stop(self):
        if self.worker:
            self.worker.stop()
        self.stop_button.setEnabled(False)

    def closeEvent(self, event):
        event.ignore()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            event.ignore()
        else:
            super().keyPressEvent(event)
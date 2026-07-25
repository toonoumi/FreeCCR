"""
DaVinci-style Settings dialog — a left category sidebar + a right content pane +
a footer. The first category, **Color Management**, consolidates the input camera
profile (ICC / DCP), the IT8 camera-profile wizard, and the global Positive-mode
toggle that used to live in the File menu / thumbnail panel.

UI only: it reuses MainWindow's existing colour handlers (file pickers, backend
apply, re-decode). The three global TOGGLES (Positive mode, 3-way RGB merge,
Density inversion) are STAGED — changing a checkbox does nothing until **Done**
is pressed, and closing/Escape discards the staged change (reverts to the state
the dialog opened with). Camera-profile import/delete/IT8 are immediate file
operations (they have their own dialogs). See spec/settings-page.md.
"""
import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, QScrollArea,
    QStackedWidget, QVBoxLayout, QWidget,
)

from core import color_management
from core.awb import AWB_ALGORITHMS, AWB_DEFAULT
from core.ccr_backend import ccr_backend
from ui import theme


class SettingsDialog(QDialog):
    """Modal settings dialog. Categories on the left, the selected page on the
    right, a single Done button in the footer. The global toggles are staged and
    applied on Done (accept); closing without Done discards them."""

    def __init__(self, main_window, parent=None):
        super().__init__(parent or main_window)
        self._mw = main_window
        self.setWindowTitle("Settings")
        self.setModal(True)

        scr = self.screen() or QApplication.primaryScreen()
        avail = scr.availableGeometry() if scr is not None else None
        w, h = 760, 540
        if avail is not None:
            w = min(w, avail.width() - 60)
            h = min(h, avail.height() - 80)
        self.resize(w, h)
        self.setMinimumSize(min(560, w), min(420, h))

        root = QVBoxLayout(self)
        theme.apply_panel_spacing(root)

        body = QHBoxLayout()
        body.setSpacing(theme.GAP_PANEL)
        self._sidebar = QListWidget()
        self._sidebar.setFixedWidth(170)
        self._sidebar.currentRowChanged.connect(self._stack_set)
        body.addWidget(self._sidebar)
        self._stack = QStackedWidget()
        body.addWidget(self._stack, 1)
        root.addLayout(body, 1)

        self._add_category("General", self._build_general_page())
        self._add_category("Color Management", self._build_color_management_page())

        root.addWidget(theme.section_separator())
        footer = QHBoxLayout()
        theme.apply_button_row(footer)
        footer.addStretch(1)
        done = QPushButton("Done")
        theme.style_button(done, "primary", default=True)
        done.clicked.connect(self.accept)
        footer.addWidget(done)
        root.addLayout(footer)

        self._sidebar.setCurrentRow(0)
        theme.apply_windows_dark_titlebar(self)
        self.refresh_color_management()
        self._init_toggles()        # seed staged toggles from the live backend

    def _stack_set(self, i):
        if i >= 0:
            self._stack.setCurrentIndex(i)

    def _add_category(self, name: str, page: QWidget):
        QListWidgetItem(name, self._sidebar)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self._stack.addWidget(scroll)

    # ------------------------------------------------------------------ #
    # General page
    # ------------------------------------------------------------------ #
    def _build_general_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        theme.apply_panel_spacing(lay, spacing=theme.GAP_SECTION)

        grp = QGroupBox("Exposure")
        g = QVBoxLayout(grp)
        g.setSpacing(theme.GAP_ROW)
        self._cb_auto_gain = QCheckBox("Auto gain")
        g.addWidget(self._cb_auto_gain)
        g.addWidget(self._muted(
            "Automatically place the brightest highlights at the top of the "
            "working range without moving the Gain slider — the top 0.1% of the "
            "in-range highlights are set to 99.8% of full. Pixels outside the "
            "sampled clear/dense film range are ignored. Turn off to control "
            "exposure with the Gain slider alone."))
        lay.addWidget(grp)

        grp_gamma = QGroupBox("Gamma")
        gg = QVBoxLayout(grp_gamma)
        gg.setSpacing(theme.GAP_ROW)
        self._cb_gamma_lum = QCheckBox("Hue-preserving Gamma (apply to luminance only)")
        gg.addWidget(self._cb_gamma_lum)
        gg.addWidget(self._muted(
            "Apply the Gamma slider to luminance and scale the colour channels "
            "together, so midtone adjustments don't shift hue or saturation. Off "
            "(default) applies Gamma per channel — the standard look, which adds a "
            "little saturation as it brightens/darkens. Highlights that would "
            "exceed white still clip."))
        lay.addWidget(grp_gamma)

        grp_wb = QGroupBox("White balance")
        gw = QVBoxLayout(grp_wb)
        gw.setSpacing(theme.GAP_ROW)
        self._cb_auto_awb = QCheckBox("Auto white balance after conversion")
        gw.addWidget(self._cb_auto_awb)
        gw.addWidget(self._muted(
            "When a negative is converted, automatically estimate the color "
            "cast and set the Temperature/Tint sliders — only for images with "
            "no white balance already set. The AWB button applies the same "
            "estimate on demand."))
        algo_row = QHBoxLayout()
        algo_row.setSpacing(theme.GAP_ROW)
        algo_row.addWidget(QLabel("Algorithm"))
        self._combo_awb_algo = QComboBox()
        for algo_id, label in AWB_ALGORITHMS:
            self._combo_awb_algo.addItem(label, algo_id)
        algo_row.addWidget(self._combo_awb_algo, 1)
        gw.addLayout(algo_row)
        gw.addWidget(self._muted(
            "Gray World (default) assumes the scene averages to gray. White "
            "Patch balances on the brightest content. Shades of Gray weights "
            "bright areas more than Gray World. Gray Edge balances on edge "
            "colors — robust when large areas share one color."))
        lay.addWidget(grp_wb)

        grp_border = QGroupBox("Film border")
        gb = QVBoxLayout(grp_border)
        gb.setSpacing(theme.GAP_ROW)
        self._cb_sprocket = QCheckBox(
            "White sprocket holes / clear film (reversal look)")
        gb.addWidget(self._cb_sprocket)
        gb.addWidget(self._muted(
            "After you set a Black Point (film base), paint the sprocket holes "
            "and clear film border white instead of black — the reversal-film "
            "look. The mask is built from areas brighter than the film base, "
            "cleaned up and feathered, and applied last on both preview and "
            "export, so no adjustment tints it. B/W-point conversions only."))
        lay.addWidget(grp_border)

        lay.addStretch(1)
        return page

    # ------------------------------------------------------------------ #
    # Color Management page
    # ------------------------------------------------------------------ #
    def _muted(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        return lbl

    def _build_color_management_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        theme.apply_panel_spacing(lay, spacing=theme.GAP_SECTION)

        # --- Camera-profile library ------------------------------------ #
        grp = QGroupBox("Camera profiles")
        g = QVBoxLayout(grp)
        g.setSpacing(theme.GAP_ROW)
        g.addWidget(self._muted(
            "Your imported and IT8-generated profiles, kept in FreeCCR's workspace "
            "(they survive updates). Pick the active one from the “Camera profile” "
            "dropdown above the thumbnails; manage which to keep here."))

        self._status = QLabel("Active: None")
        self._status.setStyleSheet(theme.section_header_qss())
        g.addWidget(self._status)

        self._profile_list = QListWidget()
        # Cap the height so the list stays compact (it otherwise expands to fill
        # the pane and pushes the Negative-conversion toggles below the fold).
        self._profile_list.setMinimumHeight(72)
        self._profile_list.setMaximumHeight(110)
        g.addWidget(self._profile_list)

        row = QHBoxLayout()
        theme.apply_button_row(row)
        btn_import = QPushButton("Import profile…")
        btn_import.clicked.connect(self._import)
        self._btn_delete = QPushButton("Delete")
        theme.style_button(self._btn_delete, "danger")
        self._btn_delete.clicked.connect(self._delete)
        row.addWidget(btn_import)
        row.addWidget(self._btn_delete)
        row.addStretch(1)
        g.addLayout(row)

        g.addWidget(theme.section_separator())
        it8_row = QHBoxLayout()
        theme.apply_button_row(it8_row)
        btn_it8 = QPushButton("Create Camera Profile from IT8…")
        btn_it8.clicked.connect(self._create_it8)
        it8_row.addWidget(btn_it8)
        it8_row.addStretch(1)
        g.addLayout(it8_row)
        g.addWidget(self._muted(
            "Build a camera ICC or DCP from a photographed IT8 calibration chart; "
            "it is added to the library above."))
        lay.addWidget(grp)

        # --- Negative conversion --------------------------------------- #
        grp2 = QGroupBox("Negative conversion")
        g2 = QVBoxLayout(grp2)
        g2.setSpacing(theme.GAP_ROW)
        self._cb_positive = QCheckBox(
            "Positive mode (decode RAWs as positives, skip film inversion)")
        g2.addWidget(self._cb_positive)
        g2.addWidget(self._muted(
            "Treat RAWs as ready positive photos instead of film negatives. The "
            "same toggle is on the thumbnail panel."))

        g2.addWidget(theme.section_separator())
        self._cb_density = QCheckBox(
            "Density inversion (recover optical density for clear+dense sampling)")
        g2.addWidget(self._cb_density)
        g2.addWidget(self._muted(
            "When you sample both a clear and a dense point, invert using film "
            "density (log) instead of a linear stretch. Two-point conversions "
            "only — and they look darker."))
        lay.addWidget(grp2)

        # --- Trichrome (3-way RGB-light) capture ----------------------- #
        grp3 = QGroupBox("Trichrome capture")
        g3 = QVBoxLayout(grp3)
        g3.setSpacing(theme.GAP_ROW)
        self._cb_rgb_merge = QCheckBox(
            "3-way RGB merge (combine red/green/blue-light exposures)")
        g3.addWidget(self._cb_rgb_merge)
        g3.addWidget(self._muted(
            "Shoot a static scene three times under pure red, then green, then "
            "blue light. On your NEXT import, every 3 RAWs (sorted by filename) "
            "are merged into one colour image — each frame contributes only its "
            "own channel — then converted as a negative. "
            "RAW only (Bayer or monochrome); the selected count must be a "
            "multiple of 3. "
            "Applies to the next import only. Your edits to a merged image are "
            "saved and restored the next time you merge the same three frames."))
        detail_row = QHBoxLayout()
        detail_row.setSpacing(theme.GAP_BTN)
        detail_row.addWidget(QLabel("Merge detail:"))
        self._combo_merge_detail = QComboBox()
        self._combo_merge_detail.addItem("Demosaic (full resolution)", True)
        self._combo_merge_detail.addItem(
            "Single photosite (half resolution)", False)
        detail_row.addWidget(self._combo_merge_detail, 1)
        g3.addLayout(detail_row)
        g3.addWidget(self._muted(
            "Demosaic interpolates each frame's own channel from its own "
            "photosites (bilinear — still no channel mixing) at full sensor "
            "resolution. Single photosite reads only the raw colour sites — "
            "no interpolation at all — at half resolution. Applies to the "
            "next import."))
        lay.addWidget(grp3)

        lay.addStretch(1)
        return page

    # ------------------------------------------------------------------ #
    # Profile actions — delegate to MainWindow (file pickers, backend,
    # re-decode); these apply IMMEDIATELY, then the page refreshes its status.
    # (The global toggles below are staged — applied on Done, not here.)
    # ------------------------------------------------------------------ #
    def _import(self):
        self._mw.import_camera_profile_dialog()
        self.refresh_color_management()

    def _delete(self):
        item = self._profile_list.currentItem()
        if item is None:
            return
        self._mw.delete_camera_profile(item.data(Qt.UserRole))
        self.refresh_color_management()

    def _create_it8(self):
        self._mw.create_camera_profile_from_it8()
        self.refresh_color_management()

    # --- Staged toggles: apply on Done (accept), discard on close ---------- #
    def _init_toggles(self):
        """Seed the toggle checkboxes from the live backend once, at open. They
        are NOT re-synced afterwards (refresh_color_management only touches the
        profile UI) so a profile import/delete can't clobber a staged toggle."""
        for cb, val in ((self._cb_positive, ccr_backend.positive_mode),
                        (self._cb_rgb_merge, ccr_backend.rgb_merge_mode),
                        (self._cb_density, ccr_backend.density_bwpoint),
                        (self._cb_auto_gain, ccr_backend.auto_gain),
                        (self._cb_auto_awb, ccr_backend.auto_awb),
                        (self._cb_gamma_lum, ccr_backend.gamma_luminance),
                        (self._cb_sprocket, ccr_backend.sprocket_mask_white)):
            cb.blockSignals(True)
            cb.setChecked(bool(val))
            cb.blockSignals(False)
        self._combo_merge_detail.blockSignals(True)
        self._combo_merge_detail.setCurrentIndex(
            self._combo_merge_detail.findData(
                bool(getattr(ccr_backend, "rgb_merge_demosaic", True))))
        self._combo_merge_detail.blockSignals(False)
        self._combo_awb_algo.blockSignals(True)
        idx = self._combo_awb_algo.findData(
            getattr(ccr_backend, "awb_algorithm", AWB_DEFAULT))
        self._combo_awb_algo.setCurrentIndex(
            idx if idx >= 0 else self._combo_awb_algo.findData(AWB_DEFAULT))
        self._combo_awb_algo.blockSignals(False)

    def _apply_pending(self):
        """Apply only the toggles whose checkbox differs from the live backend
        value. Each MainWindow handler persists + reprocesses as needed; calling
        them only on change avoids spurious reprocess passes. Positive first (it
        re-decodes), then merge, then density (replays on the new decode)."""
        if bool(self._cb_positive.isChecked()) != bool(ccr_backend.positive_mode):
            self._mw.on_positive_mode_toggled(bool(self._cb_positive.isChecked()))
        if bool(self._cb_rgb_merge.isChecked()) != bool(ccr_backend.rgb_merge_mode):
            self._mw.on_rgb_merge_mode_toggled(bool(self._cb_rgb_merge.isChecked()))
        staged_detail = bool(self._combo_merge_detail.currentData())
        if staged_detail != bool(getattr(ccr_backend, "rgb_merge_demosaic", True)):
            self._mw.on_rgb_merge_demosaic_changed(staged_detail)
        if bool(self._cb_density.isChecked()) != bool(ccr_backend.density_bwpoint):
            self._mw.on_density_bwpoint_toggled(bool(self._cb_density.isChecked()))
        if bool(self._cb_auto_gain.isChecked()) != bool(ccr_backend.auto_gain):
            self._mw.on_auto_gain_toggled(bool(self._cb_auto_gain.isChecked()))
        if bool(self._cb_auto_awb.isChecked()) != bool(ccr_backend.auto_awb):
            self._mw.on_auto_awb_toggled(bool(self._cb_auto_awb.isChecked()))
        staged_algo = self._combo_awb_algo.currentData()
        if staged_algo != getattr(ccr_backend, "awb_algorithm", AWB_DEFAULT):
            self._mw.on_awb_algorithm_changed(staged_algo)
        if bool(self._cb_gamma_lum.isChecked()) != bool(ccr_backend.gamma_luminance):
            self._mw.on_gamma_mode_toggled(bool(self._cb_gamma_lum.isChecked()))
        if bool(self._cb_sprocket.isChecked()) != bool(ccr_backend.sprocket_mask_white):
            self._mw.on_sprocket_mask_toggled(bool(self._cb_sprocket.isChecked()))

    def accept(self):
        """Done: commit the staged toggles, then close. (Escape/close → reject,
        which never reaches here, so staged toggles are discarded.)"""
        self._apply_pending()
        super().accept()

    def refresh_color_management(self):
        """Reflect the live active profile, the library list, and Positive mode."""
        icc = getattr(ccr_backend, "input_icc_name", None)
        dcp = getattr(ccr_backend, "input_dcp_name", None)
        if getattr(ccr_backend, "active_profile_path", None) == ccr_backend.CAMERA_MATRIX:
            self._status.setText("Active: Camera Matrix  (Adobe RGB)")
        elif dcp:
            self._status.setText(f"Active: {dcp}  (DCP)")
        elif icc:
            self._status.setText(f"Active: {icc}  (ICC)")
        else:
            self._status.setText("Active: None  (raw)")

        active = getattr(ccr_backend, "active_profile_path", None)
        active_n = os.path.normcase(os.path.abspath(active)) if active else None
        self._profile_list.clear()
        for p in ccr_backend.list_camera_profiles():
            label = f"{p['name']}  ({p['kind'].upper()})"
            if active_n and os.path.normcase(os.path.abspath(p["path"])) == active_n:
                label += "   ● active"
            it = QListWidgetItem(label)
            it.setData(Qt.UserRole, p["path"])
            self._profile_list.addItem(it)
        self._btn_delete.setEnabled(self._profile_list.count() > 0)
        # NOTE: the global toggles are intentionally NOT synced here — they are
        # staged (seeded once by _init_toggles, applied on Done). Re-syncing on
        # every profile refresh would wipe a staged toggle change.

"""
Create-Camera-Profile-from-IT8 wizard (PySide6).

A 5-step QDialog that walks the user from an IT8 target shot to a saved camera
input ICC profile: pick the shot, pick the batch reference file, locate the patch
grid (draggable 4-corner overlay), review fit quality (deltaE), and save / apply.
Core math lives in core.it8_profile; this module is UI only.
See spec/it8-camera-profile.md.
"""
import os
import re
from datetime import datetime
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QPointF, QRectF, QSettings, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QBrush, QColor, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton,
    QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from core import it8_profile as it8
from core.ccr_backend import ccr_backend
from utils.unicode_path_utils import normalize_unicode_path, validate_unicode_path

_REF_URL = "http://www.targets.coloraid.de/"


def _illuminant_enum(illum: str) -> int:
    """Map the wizard's capture-light label to a DNG CalibrationIlluminant
    (EXIF LightSource) code. The fit is D50-referenced, so this is metadata for a
    single-illuminant DCP; tungsten/fluorescent are tagged, everything else D50."""
    s = (illum or "").lower()
    if "tungsten" in s:
        return 17                                    # Standard light A
    if "fluor" in s:
        return 2                                     # Fluorescent
    return 23                                        # D50


def _gamma_stretch_to_qimage(arr_u16: np.ndarray) -> QImage:
    """8-bit gamma-stretched view of a raw-linear 16-bit RGB array (display
    only — the fit always uses the linear data)."""
    a = arr_u16.astype(np.float32)
    # Normalise exposure by a high percentile so the dark linear scan is visible.
    p = np.percentile(a, 99.5)
    if p < 1:
        p = a.max() if a.max() > 0 else 1.0
    norm = np.clip(a / p, 0.0, 1.0)
    disp = np.power(norm, 1.0 / 2.2)
    disp8 = np.ascontiguousarray((disp * 255).astype(np.uint8))
    h, w = disp8.shape[:2]
    img = QImage(disp8.data, w, h, 3 * w, QImage.Format_RGB888)
    out = img.copy()                      # detach from the temporary buffer
    return out


class IT8PatchLocator(QWidget):
    """Shows the target shot with a draggable 4-corner quad over the colour grid
    and a live overlay of all 288 sample dots."""

    changed = Signal()

    HANDLE_R = 7

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(820, 480)
        self.setMouseTracking(True)
        self._qimg: Optional[QImage] = None
        self._base = None                 # decoded array as given
        self._work = None                 # working array (mirrored if requested)
        self._mirror = False
        self._iw = self._ih = 0
        self._quad = []                   # 4 (x,y) array-space corners TL,TR,BR,BL
        self._gray_offset = 0.0
        self._invalid_ids = set()         # ids the dialog flagged clipped/off-frame
        self._mode = "classic"            # "classic" (12x22 + GS strip) | "block"
        self._rows = []                   # block mode: row letters
        self._cols = []                   # block mode: column numbers
        self._drag = -1                   # index of handle being dragged, or -1
        self._scale = 1.0
        self._ox = self._oy = 0.0

    def set_image(self, arr_u16: np.ndarray):
        self._base = arr_u16
        self._rebuild_work()

    def _rebuild_work(self):
        if self._base is None:
            return
        self._work = (np.ascontiguousarray(np.fliplr(self._base))
                      if self._mirror else self._base)
        self._qimg = _gamma_stretch_to_qimage(self._work)
        self._ih, self._iw = self._work.shape[:2]
        # Default block quad inset from the image edges (re-derived on reload).
        ix, iy = self._iw, self._ih
        self._quad = [(0.12 * ix, 0.16 * iy), (0.88 * ix, 0.16 * iy),
                      (0.88 * ix, 0.78 * iy), (0.12 * ix, 0.78 * iy)]
        self.update()
        self.changed.emit()

    # --- configuration ---
    def set_mirror(self, on: bool):
        if bool(on) != self._mirror:
            self._mirror = bool(on)
            self._rebuild_work()

    def set_layout_classic(self):
        self._mode = "classic"
        self.update(); self.changed.emit()

    def set_layout_block(self, rows, cols):
        self._mode = "block"
        self._rows, self._cols = list(rows), list(cols)
        self.update(); self.changed.emit()

    def set_invalid_ids(self, ids):
        """Ids whose sampled patch is invalid (clipped or off-frame); their dots
        are drawn red. Computed by the dialog after each (re)sample."""
        new = set(ids)
        if new != self._invalid_ids:
            self._invalid_ids = new
            self.update()

    def image_array(self):
        return self._work

    def grid_dims(self):
        """(ncols, nrows) for sampling-window sizing."""
        if self._mode == "block":
            return max(1, len(self._cols)), max(1, len(self._rows))
        return 22, 12

    # --- geometry accessors ---
    def quad(self):
        return list(self._quad)

    def gray_offset(self):
        return self._gray_offset

    def set_gray_offset(self, v: float):
        self._gray_offset = float(v)
        self.update()
        self.changed.emit()

    def flip(self):
        if self._quad:
            self._quad = it8.flip_quad(self._quad)
            self.update()
            self.changed.emit()

    def points(self):
        if not self._quad:
            return {}
        if self._mode == "block":
            if not self._rows or not self._cols:
                return {}
            return it8.block_sample_points(self._quad, self._rows, self._cols)
        return it8.grid_sample_points(self._quad, self._gray_offset)

    # --- coordinate mapping (array <-> widget) ---
    def _recompute_fit(self):
        if not self._iw:
            return
        W, H = self.width(), self.height()
        self._scale = min(W / self._iw, H / self._ih)
        self._ox = (W - self._iw * self._scale) / 2
        self._oy = (H - self._ih * self._scale) / 2

    def _a2w(self, x, y):
        return QPointF(self._ox + x * self._scale, self._oy + y * self._scale)

    def _w2a(self, x, y):
        return ((x - self._ox) / self._scale, (y - self._oy) / self._scale)

    # --- painting ---
    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if self._qimg is None:
            p.setPen(QColor(180, 180, 180))
            p.drawText(self.rect(), Qt.AlignCenter, "No image")
            return
        self._recompute_fit()
        target = QRectF(self._ox, self._oy, self._iw * self._scale,
                        self._ih * self._scale)
        p.drawImage(target, self._qimg)
        p.setRenderHint(QPainter.Antialiasing, True)

        # Quad outline.
        poly = QPolygonF([self._a2w(x, y) for (x, y) in self._quad])
        p.setPen(QPen(QColor(80, 200, 120), 2))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(poly)

        # Sample dots (colour vs gray differ in colour; invalid patches red).
        pts = self.points()
        for sid, (x, y) in pts.items():
            w = self._a2w(x, y)
            bad = (not target.contains(w)) or sid in self._invalid_ids
            if bad:
                p.setPen(QPen(QColor(230, 70, 70), 2))   # invalid/off-frame -> red
            elif sid.startswith("GS"):
                p.setPen(QPen(QColor(120, 170, 255), 1))
            else:
                p.setPen(QPen(QColor(255, 230, 90), 1))
            r = 2.6 if bad else 1.6                       # enlarge bad dots
            p.drawEllipse(w, r, r)

        # Corner handles.
        for i, (x, y) in enumerate(self._quad):
            w = self._a2w(x, y)
            p.setBrush(QBrush(QColor(80, 200, 120)))
            p.setPen(QPen(QColor(20, 20, 20), 1))
            p.drawEllipse(w, self.HANDLE_R, self.HANDLE_R)
        p.end()

    # --- mouse ---
    def mousePressEvent(self, e):
        if self._qimg is None:
            return
        self._recompute_fit()
        pos = e.position() if hasattr(e, "position") else QPointF(e.pos())
        # Pick the nearest handle within grab radius.
        best, bestd = -1, (self.HANDLE_R + 6) ** 2
        for i, (x, y) in enumerate(self._quad):
            w = self._a2w(x, y)
            d = (w.x() - pos.x()) ** 2 + (w.y() - pos.y()) ** 2
            if d < bestd:
                best, bestd = i, d
        self._drag = best

    def mouseMoveEvent(self, e):
        if self._drag < 0:
            return
        pos = e.position() if hasattr(e, "position") else QPointF(e.pos())
        ax, ay = self._w2a(pos.x(), pos.y())
        ax = max(0.0, min(self._iw, ax))
        ay = max(0.0, min(self._ih, ay))
        self._quad[self._drag] = (ax, ay)
        self.update()                      # dots follow live (cheap repaint)

    def mouseReleaseEvent(self, _):
        was_dragging = self._drag >= 0
        self._drag = -1
        if was_dragging:
            # Re-sample (expensive) only once the drag settles, not per move.
            self.changed.emit()


class IT8ProfileDialog(QDialog):
    """5-step wizard. On success the profile is written; `self.saved_path` holds
    the .icc path and `self.apply_now` whether to activate it as input profile."""

    PAGE_TARGET, PAGE_REF, PAGE_LOCATE, PAGE_BUILD, PAGE_SAVE = range(5)

    def __init__(self, parent=None, current_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Create Camera Profile from IT8")
        self.setModal(True)
        # Big enough for accurate corner placement (≥1024 wide), but clamp the
        # floor to the screen so the nav row can't be pushed off a small display.
        scr = self.screen() or QApplication.primaryScreen()
        avail = scr.availableGeometry() if scr is not None else None
        min_w, min_h, init_w, init_h = 1024, 700, 1120, 780
        if avail is not None:
            min_w = min(min_w, avail.width() - 40)
            min_h = min(min_h, avail.height() - 80)
            init_w = min(init_w, avail.width() - 40)
            init_h = min(init_h, avail.height() - 80)
        self.setMinimumSize(min_w, min_h)
        self.resize(max(min_w, init_w), max(min_h, init_h))
        self._settings = QSettings("FreeCCR", "FreeCCR")

        self._current_path = current_path
        self._target_path: Optional[str] = None
        self._target_img: Optional[np.ndarray] = None
        self._ref: Optional[it8.IT8Reference] = None
        self._fit: Optional[it8.CameraFit] = None
        self._locator_arr = None           # array currently loaded in the locator
        self._mode = "classic"             # "classic" (GS strip) | "block" (plain grid)
        self.saved_path: Optional[str] = None
        self.apply_now = False
        # Multi-card targets (block mode): samples accumulate across up to 3
        # photographed cards before the fit. _mapped_cards holds the finalised
        # sample dicts of the cards already mapped; the card in the locator is
        # the current one. _all_samples is the merged set the fit consumes.
        self._mapped_cards: list = []
        self._all_samples: dict = {}
        self._n_images = 1

        self.stack = QStackedWidget(self)
        self.stack.addWidget(self._build_target_page())
        self.stack.addWidget(self._build_ref_page())
        self.stack.addWidget(self._build_locate_page())
        self.stack.addWidget(self._build_build_page())
        self.stack.addWidget(self._build_save_page())

        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Next")
        self.cancel_btn = QPushButton("Cancel")
        self.back_btn.clicked.connect(self._go_back)
        self.next_btn.clicked.connect(self._go_next)
        self.cancel_btn.clicked.connect(self.reject)

        nav = QHBoxLayout()
        nav.addWidget(self.cancel_btn)
        nav.addStretch(1)
        nav.addWidget(self.back_btn)
        nav.addWidget(self.next_btn)

        root = QVBoxLayout(self)
        root.addWidget(self.stack, 1)
        root.addLayout(nav)
        self._update_nav()

    # ------------------------------------------------------------------ #
    # Pages
    # ------------------------------------------------------------------ #
    def _build_target_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>Step 1 — IT8 target shot</b>"))
        guide = QLabel(
            "Photograph a printed IT8 chart under your capture light, then "
            "select that photo here.<br><br>"
            "<b>Capture tips</b> — broad-spectrum daylight/strobe light (not "
            "tungsten); even, glare-free lighting; chart flat and square to the "
            "camera; fill the centre of the frame; expose so the lightest "
            "grayscale patch is bright but <i>not</i> clipped. <b>Shoot RAW</b> — "
            "the profile is built from the raw-linear sensor data, the same "
            "space FreeCCR converts negatives in.<br><br>"
            "The profile is valid only under the light it was shot in.")
        guide.setWordWrap(True)
        lay.addWidget(guide)
        row = QHBoxLayout()
        self.use_current_btn = QPushButton("Use current image")
        self.use_current_btn.setEnabled(bool(self._current_path))
        self.use_current_btn.clicked.connect(self._use_current)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_target)
        row.addWidget(self.use_current_btn)
        row.addWidget(browse)
        row.addStretch(1)
        lay.addLayout(row)
        self.target_label = QLabel("<i>No target selected.</i>")
        self.target_label.setWordWrap(True)
        lay.addWidget(self.target_label)
        lay.addStretch(1)
        return w

    def _build_ref_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>Step 2 — Batch reference file</b>"))
        guide = QLabel(
            "Load the reference data file that matches your chart's printed "
            "<b>batch/serial number</b>. Each chart batch is measured "
            "separately — the wrong file gives wrong colour. Both <b>CGATS</b> "
            "(.it8/.txt) and <b>CxF</b> (.cxf) reference files are accepted.<br>"
            f"Wolf Faust provides free per-batch files: <a href='{_REF_URL}'>"
            f"{_REF_URL}</a>")
        guide.setWordWrap(True)
        guide.setOpenExternalLinks(True)
        lay.addWidget(guide)
        row = QHBoxLayout()
        browse = QPushButton("Browse reference…")
        browse.clicked.connect(self._browse_ref)
        row.addWidget(browse)
        row.addStretch(1)
        lay.addLayout(row)
        self.ref_label = QLabel("<i>No reference loaded.</i>")
        self.ref_label.setWordWrap(True)
        lay.addWidget(self.ref_label)
        lay.addStretch(1)
        return w

    def _build_locate_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        self.locate_hint = QLabel(
            "<b>Step 3 — Locate the patches</b> — drag the four green corners "
            "onto the outer corners of the patch grid. Dots mark where each "
            "patch is sampled; nudge the corners until every dot sits inside "
            "its patch.")
        self.locate_hint.setWordWrap(True)
        lay.addWidget(self.locate_hint)
        self.locator = IT8PatchLocator()
        self.locator.changed.connect(self._update_locate_status)
        lay.addWidget(self.locator, 1)

        # Row 1: orientation controls.
        ctl = QHBoxLayout()
        flip = QPushButton("Flip 180°")
        flip.clicked.connect(self.locator.flip)
        ctl.addWidget(flip)
        self.mirror_check = QCheckBox("Mirrored capture")
        self.mirror_check.setToolTip(
            "Tick if the shot is left-right mirrored (e.g. a back-lit film "
            "target shot through its base). The image is flipped horizontally.")
        self.mirror_check.toggled.connect(self._on_mirror_toggled)
        ctl.addWidget(self.mirror_check)
        # Gray-strip nudge (classic IT8 only).
        self.gray_label = QLabel("Gray strip:")
        ctl.addWidget(self.gray_label)
        self.gray_slider = QSlider(Qt.Horizontal)
        self.gray_slider.setMinimum(-100)
        self.gray_slider.setMaximum(100)
        self.gray_slider.setValue(0)
        self.gray_slider.setFixedWidth(140)
        self.gray_slider.valueChanged.connect(
            lambda v: self.locator.set_gray_offset(v / 1000.0))
        ctl.addWidget(self.gray_slider)
        # Multi-card progress indicator (block-mode targets only).
        self.card_label = QLabel("")
        self.card_label.setStyleSheet("color:#9cc4ff; font-weight:bold;")
        ctl.addWidget(self.card_label)
        ctl.addStretch(1)
        self.locate_status = QLabel("")
        ctl.addWidget(self.locate_status)
        lay.addLayout(ctl)

        # Row 2: block patch-id range (non-classic grids only).
        self.block_row = QWidget()
        brow = QHBoxLayout(self.block_row)
        brow.setContentsMargins(0, 0, 0, 0)
        brow.addWidget(QLabel("Top-left patch:"))
        self.tl_edit = QLineEdit()
        self.tl_edit.setFixedWidth(70)
        self.tl_edit.editingFinished.connect(self._apply_block_ids)
        brow.addWidget(self.tl_edit)
        brow.addWidget(QLabel("Bottom-right patch:"))
        self.br_edit = QLineEdit()
        self.br_edit.setFixedWidth(70)
        self.br_edit.editingFinished.connect(self._apply_block_ids)
        brow.addWidget(self.br_edit)
        brow.addWidget(QLabel("(read the corner patch labels off the chart)"))
        brow.addStretch(1)
        lay.addWidget(self.block_row)
        self.block_row.setVisible(False)
        return w

    def _on_mirror_toggled(self, on):
        self.locator.set_mirror(bool(on))

    def _apply_block_ids(self):
        if self._mode != "block":
            return
        try:
            rows, cols = it8.parse_block(self.tl_edit.text(), self.br_edit.text())
        except ValueError:
            return
        self.locator.set_layout_block(rows, cols)

    def _build_build_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>Step 4 — Review fit quality</b>"))
        self.quality_label = QLabel("")
        self.quality_label.setWordWrap(True)
        lay.addWidget(self.quality_label)
        self.worst_list = QListWidget()
        self.worst_list.setMaximumHeight(150)
        lay.addWidget(self.worst_list)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.illum_combo = QComboBox()
        self.illum_combo.setEditable(True)
        self.illum_combo.addItems(["Daylight", "Strobe/Flash", "Tungsten",
                                   "Fluorescent", "Other"])
        self.format_combo = QComboBox()
        self.format_combo.addItems(["ICC profile", "DCP — DNG camera profile"])
        self.format_combo.setToolTip(
            "ICC: FreeCCR's native input profile (3×3 matrix or cLUT). "
            "DCP: a DNG camera profile (3×3 matrix only) for interchange with "
            "Lightroom / Camera Raw and other DNG tools.")
        self.format_combo.currentIndexChanged.connect(self._on_format_changed)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["3×3 matrix", "cLUT (higher accuracy)"])
        self.type_combo.setToolTip(
            "3×3 matrix: a compact, well-behaved linear profile. cLUT: a 3-D "
            "lookup table that also corrects the saturated-corner residuals a "
            "matrix can't reach (uses the full sampled patch set). cLUT is "
            "ICC-only — DCP is always a 3×3 matrix.")
        form.addRow("Profile name:", self.name_edit)
        form.addRow("Illuminant:", self.illum_combo)
        form.addRow("Output format:", self.format_combo)
        form.addRow("Profile type:", self.type_combo)
        lay.addLayout(form)
        lay.addStretch(1)
        return w

    def _build_save_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>Step 5 — Save</b>"))
        self.save_label = QLabel("")
        self.save_label.setWordWrap(True)
        lay.addWidget(self.save_label)
        row = QHBoxLayout()
        self.save_path_edit = QLineEdit()
        choose = QPushButton("Choose…")
        choose.clicked.connect(self._choose_save_path)
        row.addWidget(self.save_path_edit, 1)
        row.addWidget(choose)
        lay.addLayout(row)
        self.apply_check = QCheckBox("Set as input profile now")
        self.apply_check.setChecked(True)
        lay.addWidget(self.apply_check)
        lay.addStretch(1)
        return w

    # ------------------------------------------------------------------ #
    # Page 1 — target
    # ------------------------------------------------------------------ #
    def _use_current(self):
        if self._current_path:
            self._set_target(self._current_path)

    def _browse_target(self):
        start = self._settings.value("files/last_open_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select IT8 target shot", start,
            "Images (*.dng *.tif *.tiff *.arw *.nef *.cr2 *.cr3 *.raf *.png "
            "*.jpg *.jpeg *.rw2 *.3fr *.fff);;All Files (*)")
        if path:
            self._set_target(path)

    def _reset_card_accum(self):
        """Discard any in-progress multi-card accumulation. The target shot and
        reference define what the mapped cards belong to, so changing either
        must throw the accumulated cards away — otherwise samples from an
        abandoned session silently contaminate the next fit."""
        self._mapped_cards = []
        self._all_samples = {}
        self._n_images = 1
        self._fit = None
        # Let the block range re-seed from the (possibly new) reference extent.
        if hasattr(self, "tl_edit"):
            self.tl_edit.clear()
            self.br_edit.clear()

    def _set_target(self, path):
        self._reset_card_accum()           # a new target starts a fresh session
        self._target_path = path
        self._target_img = None            # force re-decode on Next
        raw = ", ".join((".dng", ".arw", ".nef", ".cr2", ".cr3", ".raf",
                         ".rw2", ".3fr", ".fff"))
        is_raw = os.path.splitext(path)[1].lower() in raw
        warn = ("" if is_raw else
                "<br><span style='color:#cc7a00'>Note: not a RAW file — for "
                "FreeCCR's pipeline a RAW shot gives the matching device "
                "space.</span>")
        self.target_label.setText(f"Selected: <b>{os.path.basename(path)}</b>{warn}")
        self._update_nav()

    def _decode_target(self) -> bool:
        if self._target_img is not None:
            return True
        norm = normalize_unicode_path(self._target_path)
        if not validate_unicode_path(norm):
            QMessageBox.warning(self, "Unicode Path",
                                "This file path can't be processed. Please move "
                                "it to a simpler path.")
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            arr = it8.decode_target(norm)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Decode Error",
                                 f"Could not read the target shot:\n\n{e}")
            return False
        QApplication.restoreOverrideCursor()
        if arr is None or arr.ndim != 3:
            QMessageBox.critical(self, "Decode Error",
                                 "Could not decode the target shot.")
            return False
        self._target_img = arr
        return True

    # ------------------------------------------------------------------ #
    # Page 2 — reference
    # ------------------------------------------------------------------ #
    def _browse_ref(self):
        start = self._settings.value("files/last_it8_ref_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select IT8 reference file", start,
            "Reference files (*.it8 *.txt *.cie *.cxf);;CxF (*.cxf);;"
            "CGATS (*.it8 *.txt *.cie);;All Files (*)")
        if not path:
            return
        self._settings.setValue("files/last_it8_ref_dir", os.path.dirname(path))
        try:
            ref = it8.parse_reference(path)
        except it8.IT8ReferenceError as e:
            QMessageBox.warning(self, "Reference File", str(e))
            return
        except Exception as e:
            QMessageBox.warning(self, "Reference File",
                                f"Could not read the reference file:\n\n{e}")
            return
        self._reset_card_accum()           # a new reference starts a fresh session
        self._ref = ref
        # Classic IT8 has a GS0..GS23 strip; anything else is a plain grid that
        # the user delimits with corner patch ids (block mode).
        self._mode = "classic" if "GS0" in ref.patches else "block"
        n = len(ref.matched_ids if self._mode == "classic" else ref.patches)
        extent = self._ref_extent()
        extent_txt = (f"<br>Patch ids: <b>{extent[0]}</b> … <b>{extent[1]}</b>"
                      if self._mode == "block" and extent else "")
        self.ref_label.setText(
            f"Loaded: <b>{os.path.basename(path)}</b><br>"
            f"Chart type: {ref.chart_type or 'unknown'}<br>"
            f"Batch / serial: <b>{ref.batch or 'unknown'}</b> — confirm this "
            f"matches the number printed on your chart.<br>"
            f"Patches recognised: {n}{extent_txt}")
        self._update_nav()

    def _ref_extent(self):
        """(top-left, bottom-right) patch ids spanning the reference's letter+
        number grid, e.g. ('A1','L72'). None if no such ids."""
        rows, cols = set(), set()
        for sid in self._ref.patches:
            m = re.match(r'^([A-Za-z])(\d+)$', sid)
            if m:
                rows.add(m.group(1).upper())
                cols.add(int(m.group(2)))
        if not rows or not cols:
            return None
        return (f"{min(rows)}{min(cols)}", f"{max(rows)}{max(cols)}")

    # ------------------------------------------------------------------ #
    # Page 3 — locate
    # ------------------------------------------------------------------ #
    def _update_locate_status(self):
        img = self.locator.image_array()
        if img is None or self._ref is None:
            self.locator.set_invalid_ids(set())
            return
        pts = self.locator.points()
        nc, nr = self.locator.grid_dims()
        samples = it8.sample_patches(img, pts, self.locator.quad(),
                                     ncols=nc, nrows=nr)
        in_ref = [sid for sid in samples if sid in self._ref.patches]
        invalid = [sid for sid in in_ref if not samples[sid].valid]
        self.locator.set_invalid_ids(set(invalid))
        valid = len(in_ref) - len(invalid)
        warn = ""
        if ("GS0" in samples and "GS23" in samples
                and samples["GS0"].valid and samples["GS23"].valid):
            if samples["GS0"].rgb.mean() < samples["GS23"].rgb.mean():
                warn = "  ⚠ GS0 darker than GS23 — try Flip 180°"
        if invalid:
            shown = ", ".join(invalid[:10])
            more = (f", … (+{len(invalid) - 10} more)"
                    if len(invalid) > 10 else "")
            self.locate_status.setText(
                f"Valid patches: {valid}/{len(in_ref)} — "
                f"<span style='color:#e64646;'>invalid (red): {shown}{more}"
                f"</span>{warn}")
            self.locate_status.setToolTip(
                "Invalid patches (drawn red — highlight-clipped, essentially "
                "black, or off-frame; nudge the corners or re-expose):\n"
                + ", ".join(invalid))
        else:
            self.locate_status.setText(
                f"Valid patches: {valid}/{len(in_ref)}{warn}")
            self.locate_status.setToolTip("")

    # ------------------------------------------------------------------ #
    # Page 3b — multi-card mapping (block-mode targets split across cards)
    # ------------------------------------------------------------------ #
    def _sample_current(self):
        """Sample the card currently in the locator -> {id: PatchSample}, or
        None if no image is loaded."""
        img = self.locator.image_array()
        if img is None:
            return None
        pts = self.locator.points()
        nc, nr = self.locator.grid_dims()
        return it8.sample_patches(img, pts, self.locator.quad(),
                                  ncols=nc, nrows=nr)

    def _should_ask_more(self) -> bool:
        """Offer another card only for block-mode targets, at most twice (so up
        to 3 cards total, e.g. columns 1-24 / 25-48 / 49-72), and only while no
        fit has been built yet — so Back-from-build → Next just re-fits instead
        of re-popping the prompt. A target/reference change clears _fit."""
        return (self._mode == "block" and len(self._mapped_cards) < 2
                and self._fit is None)

    def _prompt_map_another(self) -> str:
        """Ask whether to map another card. Returns 'another' or 'build'."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Map another image?")
        n = len(self._mapped_cards) + 1
        box.setText(f"Mapped {n} image(s) so far.")
        box.setInformativeText(
            "Some IT8 targets are split across several physical cards (e.g. "
            "columns 1–24, 25–48, 49–72). If you photographed more than one "
            "card, map the next image now — its patches are added to the same "
            "profile. Otherwise build the profile from what you've mapped.")
        another = box.addButton("Map another image…", QMessageBox.AcceptRole)
        box.addButton("Build profile", QMessageBox.RejectRole)
        box.setDefaultButton(another)
        box.exec()
        return "another" if box.clickedButton() is another else "build"

    def _browse_next_target(self) -> Optional[str]:
        start = self._settings.value("files/last_open_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the next IT8 card image", start,
            "Images (*.dng *.tif *.tiff *.arw *.nef *.cr2 *.cr3 *.raf *.png "
            "*.jpg *.jpeg *.rw2 *.3fr *.fff);;All Files (*)")
        return path or None

    def _load_card_image(self, path: str) -> bool:
        """Decode `path` and load it into the locator as the next card. Returns
        False (with a message shown) on a bad path / decode failure."""
        norm = normalize_unicode_path(path)
        if not validate_unicode_path(norm):
            QMessageBox.warning(self, "Unicode Path",
                                "This file path can't be processed. Please move "
                                "it to a simpler path.")
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            arr = it8.decode_target(norm)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Decode Error",
                                 f"Could not read the card image:\n\n{e}")
            return False
        QApplication.restoreOverrideCursor()
        if arr is None or arr.ndim != 3:
            QMessageBox.critical(self, "Decode Error",
                                 "Could not decode the card image.")
            return False
        self._target_path = path
        self._target_img = arr
        self.locator.set_image(arr)        # resets the quad for the new shot
        self._locator_arr = arr
        return True

    def _advance_block_ids(self):
        """Shift the block's patch-id range to the next card's columns. If the
        previous card already reached the chart edge there is no distinct next
        block to advance into (the shift would collapse to a single column or
        overlap), so leave the ids for the user to set by hand."""
        if self._mode != "block":
            return
        max_col = None
        extent = self._ref_extent()
        if extent:
            m = re.match(r'^[A-Za-z](\d+)$', extent[1])
            if m:
                max_col = int(m.group(1))
        try:
            prev_rows, prev_cols = it8.parse_block(self.tl_edit.text(),
                                                   self.br_edit.text())
            tl, br = it8.next_block_ids(self.tl_edit.text(),
                                       self.br_edit.text(), max_col)
            new_rows, new_cols = it8.parse_block(tl, br)
        except ValueError:
            return
        if len(new_cols) < 2 or min(new_cols) <= max(prev_cols):
            QMessageBox.information(
                self, "Set this card's patches",
                "Couldn't auto-advance the patch-id range — the previous card "
                "already reached the edge of the chart. Set the top-left / "
                "bottom-right patch ids for this card by hand.")
            return
        self.tl_edit.setText(tl)
        self.br_edit.setText(br)
        self._apply_block_ids()

    def _update_card_banner(self):
        if self._mode == "block":
            n = len(self._mapped_cards) + 1
            self.card_label.setText(f"Card {n} of up to 3")
            self.card_label.setVisible(True)
        else:
            self.card_label.setVisible(False)

    # ------------------------------------------------------------------ #
    # Page 4 — build
    # ------------------------------------------------------------------ #
    def _run_fit(self) -> bool:
        err = None
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # _all_samples is the merge of every mapped card; fall back to the
            # current card alone if the fit is somehow reached un-accumulated.
            samples = self._all_samples or self._sample_current() or {}
            fit = it8.fit_camera_matrix(samples, self._ref)
        except it8.IT8ReferenceError as e:
            fit, err = None, str(e)
        finally:
            QApplication.restoreOverrideCursor()
        if err is not None:
            QMessageBox.warning(self, "Fit", err)
            return False
        self._fit = fit
        if fit.avg_de < 2.0:
            chip, colour = "Good", "#3a8a3a"
        elif fit.avg_de < 4.0:
            chip, colour = "OK", "#b08a00"
        else:
            chip, colour = "Check placement / capture", "#b03030"
        combined = (f" · combined from {self._n_images} images"
                    if self._n_images > 1 else "")
        self.quality_label.setText(
            f"<span style='background:{colour};color:white;padding:2px 8px;'>"
            f"{chip}</span>  &nbsp; "
            f"avg ΔE2000 <b>{fit.avg_de:.2f}</b> · median {fit.med_de:.2f} · "
            f"95th {fit.p95_de:.2f} · max {fit.max_de:.2f}<br>"
            f"Patches used: {len(fit.used_ids)} · dropped "
            f"(clipped/missing): {len(fit.dropped_ids)} · "
            f"white-balanced on {it8.wb_patch_label(fit)}{combined}")
        self.worst_list.clear()
        for sid, de in fit.per_patch[:20]:
            self.worst_list.addItem(f"{sid}\tΔE {de:.2f}")
        if not self.name_edit.text().strip():
            illum = self.illum_combo.currentText().strip() or "Daylight"
            self.name_edit.setText(
                f"Camera {illum} {datetime.now():%Y-%m-%d}")
        return True

    # ------------------------------------------------------------------ #
    # Page 5 — save
    # ------------------------------------------------------------------ #
    def _is_dcp(self) -> bool:
        """The Output-format selector is the single source of truth for ICC vs DCP."""
        return self.format_combo.currentIndex() == 1

    def _on_format_changed(self):
        """ICC ↔ DCP: cLUT is ICC-only (DCP is matrix-only), and the suggested
        filename's extension follows the format."""
        is_dcp = self._is_dcp()
        if is_dcp:
            self.type_combo.setCurrentIndex(0)        # force 3×3 matrix
        self.type_combo.setEnabled(not is_dcp)
        cur = self.save_path_edit.text().strip()
        if cur.lower().endswith((".icc", ".dcp")):
            self.save_path_edit.setText(cur[:-4] + (".dcp" if is_dcp else ".icc"))

    def _default_save_path(self) -> str:
        from core.catalog import default_catalog_path
        folder = os.path.join(os.path.dirname(default_catalog_path()),
                              "camera_profiles")
        os.makedirs(folder, exist_ok=True)
        name = self.name_edit.text().strip() or "Camera Profile"
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
        return os.path.join(folder, f"{safe}{'.dcp' if self._is_dcp() else '.icc'}")

    def _choose_save_path(self):
        sel0 = ("DNG camera profile (*.dcp)" if self._is_dcp()
                else "ICC profile (*.icc)")
        path, sel = QFileDialog.getSaveFileName(
            self, "Save camera profile",
            self.save_path_edit.text() or self._default_save_path(),
            "ICC profile (*.icc);;DNG camera profile (*.dcp)", sel0)
        if not path:
            return
        # Picking a file type in the dialog keeps the Output-format selector in
        # step (so the combo and the path never disagree).
        if path.lower().endswith(".dcp"):
            chosen_dcp = True
        elif path.lower().endswith(".icc"):
            chosen_dcp = False
        else:
            chosen_dcp = "dcp" in sel.lower()
            path += ".dcp" if chosen_dcp else ".icc"
        self.format_combo.setCurrentIndex(1 if chosen_dcp else 0)
        self.save_path_edit.setText(path)

    def _do_save(self) -> bool:
        path = self.save_path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Save", "Choose a destination file.")
            return False
        is_dcp = self._is_dcp()
        # The Output-format selector wins: normalise the path extension to match.
        base = path[:-4] if path.lower().endswith((".icc", ".dcp")) else path
        path = base + (".dcp" if is_dcp else ".icc")
        self.save_path_edit.setText(path)
        illum = self.illum_combo.currentText().strip()
        name = self.name_edit.text().strip() or "Camera Profile"
        desc = f"{name} ({illum})" if illum else name
        clut = (not is_dcp) and self.type_combo.currentIndex() == 1
        try:
            if is_dcp:
                from core import dcp_profile
                blob = dcp_profile.build_camera_dcp(
                    self._fit, desc, illuminant=_illuminant_enum(illum))
            elif clut:
                blob = it8.build_camera_icc(
                    self._fit, desc, mode="clut", grid=17,
                    samples=self._all_samples, ref=self._ref)
            else:
                blob = it8.build_camera_icc(self._fit, desc)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "wb") as f:
                f.write(blob)
        except Exception as e:
            QMessageBox.critical(self, "Save Error",
                                 f"Could not write the profile:\n\n{e}")
            return False
        self.saved_path = path
        self.apply_now = self.apply_check.isChecked()
        return True

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #
    def _go_back(self):
        i = self.stack.currentIndex()
        if i > 0:
            self.stack.setCurrentIndex(i - 1)
            self._update_nav()

    def _go_next(self):
        i = self.stack.currentIndex()
        if i == self.PAGE_TARGET:
            if not self._target_path:
                QMessageBox.information(self, "Target", "Select a target shot.")
                return
            if not self._decode_target():
                return
        elif i == self.PAGE_REF:
            if self._ref is None:
                QMessageBox.information(self, "Reference",
                                        "Load a reference file.")
                return
        elif i == self.PAGE_LOCATE:
            cur = self._sample_current()
            if cur is None:
                QMessageBox.information(self, "Locate",
                                        "No target image to sample.")
                return
            # Offer to map another card (up to 3 total) before fitting.
            if self._should_ask_more() and self._prompt_map_another() == "another":
                path = self._browse_next_target()
                if not path:
                    return                        # browse cancelled — stay put
                self._mapped_cards.append(cur)
                if not self._load_card_image(path):
                    self._mapped_cards.pop()      # decode failed — roll back
                    return
                self._advance_block_ids()
                self._update_card_banner()
                self._update_locate_status()
                return                            # stay on locate for next card
            self._n_images = len(self._mapped_cards) + 1
            self._all_samples = it8.merge_samples(self._mapped_cards + [cur])
            if not self._run_fit():
                return
        elif i == self.PAGE_BUILD:
            if not self.save_path_edit.text().strip():
                self.save_path_edit.setText(self._default_save_path())
            self.save_label.setText(
                "The profile will be written here. With “Set as input profile "
                "now” ticked it is applied to every loaded image immediately "
                "(File ▸ Input ICC).")
        elif i == self.PAGE_SAVE:
            if self._do_save():
                self.accept()
            return

        self.stack.setCurrentIndex(i + 1)
        # On entering the locate page, configure the layout for the reference
        # type and (re)load the decoded target into the locator whenever the
        # underlying array changed (a re-decode makes a new array object).
        if self.stack.currentIndex() == self.PAGE_LOCATE:
            self._configure_locate_page()
            if self._target_img is not self._locator_arr:
                self.locator.set_image(self._target_img)
                self._locator_arr = self._target_img
            self._update_locate_status()
        self._update_nav()

    def _configure_locate_page(self):
        """Show classic (GS-strip) vs block (plain grid) controls per the
        reference, and seed the block patch-id range from its extent."""
        block = self._mode == "block"
        self.block_row.setVisible(block)
        self.gray_label.setVisible(not block)
        self.gray_slider.setVisible(not block)
        if block:
            self.locate_hint.setText(
                "<b>Step 3 — Locate the patches</b> — drag the four green "
                "corners onto the outer corners of the patch grid you "
                "photographed. Set the top-left / bottom-right patch ids below "
                "to match the block <i>in this frame</i>, then nudge the corners "
                "until every dot sits inside its patch. If the target is split "
                "across several cards (e.g. columns 1–24, 25–48, 49–72), map just "
                "this card — you'll be asked for the next one.")
            extent = self._ref_extent()
            if extent and not self.tl_edit.text():
                self.tl_edit.setText(extent[0])
                self.br_edit.setText(extent[1])
            try:
                rows, cols = it8.parse_block(self.tl_edit.text(), self.br_edit.text())
                self.locator.set_layout_block(rows, cols)
            except ValueError:
                pass
        else:
            self.locate_hint.setText(
                "<b>Step 3 — Locate the patches</b> — drag the four green "
                "corners onto the outer corners of the 12×22 colour grid. "
                "Yellow dots are colour patches, blue dots the grayscale strip.")
            self.locator.set_layout_classic()
        self._update_card_banner()

    def _update_nav(self):
        i = self.stack.currentIndex()
        self.back_btn.setEnabled(i > 0)
        self.next_btn.setText("Finish" if i == self.PAGE_SAVE else "Next")

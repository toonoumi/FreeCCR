"""
Create-Field-Correction-Profile wizard (PySide6).

A 3-step QDialog: pick a calibration shot of an evenly lit blank surface, review
the measured falloff (shot / gain map / corrected preview + validation warnings),
then name and save the profile into the library. Core math lives in
core.flat_field; this module is UI only. See spec/flat-field-correction.md §7.
"""
import os
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from core import flat_field as ff
from core.ccr_backend import ccr_backend
from utils.unicode_path_utils import normalize_unicode_path, validate_unicode_path

_RAW_EXTS = (".dng", ".arw", ".nef", ".cr2", ".cr3", ".raf", ".rw2", ".orf",
             ".srw", ".pef", ".3fr", ".fff")


def _gamma_stretch_qimage(arr_u16: np.ndarray, norm: Optional[float] = None):
    """8-bit gamma-stretched view of a linear 16-bit RGB array (display only —
    the measurement always uses the linear data). `norm` pins the exposure so two
    panes can be compared like for like."""
    a = arr_u16.astype(np.float32)
    if norm is None:
        norm = float(np.percentile(a, 99.5)) or float(a.max()) or 1.0
    disp = np.power(np.clip(a / max(norm, 1e-6), 0.0, 1.0), 1.0 / 2.2)
    disp8 = np.ascontiguousarray((disp * 255).astype(np.uint8))
    h, w = disp8.shape[:2]
    return QImage(disp8.data, w, h, 3 * w, QImage.Format_RGB888).copy(), norm


def _gain_heatmap_qimage(gain: np.ndarray):
    """False-colour view of the gain map: dark blue at 1.0 (no correction)
    through to bright yellow at the map's maximum."""
    g = gain.mean(axis=2)
    lo, hi = 1.0, max(float(g.max()), 1.0001)
    t = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    # A simple perceptual-ish ramp (navy -> magenta -> orange -> pale yellow).
    r = np.clip(1.6 * t, 0, 1)
    gr = np.clip(1.7 * t - 0.7, 0, 1)
    b = np.clip(0.45 + 0.9 * t - 1.6 * t * t, 0, 1)
    rgb = np.ascontiguousarray(
        (np.stack([r, gr, b], axis=2) * 255).astype(np.uint8))
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy(), hi


class _Pane(QWidget):
    """A titled image pane that scales its pixmap to fit."""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self._title = QLabel(title)
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setStyleSheet("font-weight: bold;")
        self._img = QLabel()
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setMinimumSize(220, 160)
        self._img.setStyleSheet("background: #1e1e1e; border: 1px solid #3a3a3a;")
        self._caption = QLabel("")
        self._caption.setAlignment(Qt.AlignCenter)
        self._caption.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(self._title)
        lay.addWidget(self._img, 1)
        lay.addWidget(self._caption)
        self._pix: Optional[QPixmap] = None

    def set_image(self, qimg: QImage, caption: str = ""):
        self._pix = QPixmap.fromImage(qimg)
        self._caption.setText(caption)
        self._rescale()

    def _rescale(self):
        if self._pix is not None and not self._pix.isNull():
            self._img.setPixmap(self._pix.scaled(
                self._img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()


class FieldCorrectionDialog(QDialog):
    """3-step wizard. On success `self.saved_path` holds the .ffc path and
    `self.apply_now` whether to activate it."""

    PAGE_SOURCE, PAGE_REVIEW, PAGE_SAVE = range(3)

    def __init__(self, parent=None, current_path: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Create Field Correction Profile")
        self.setModal(True)
        scr = self.screen() or QApplication.primaryScreen()
        avail = scr.availableGeometry() if scr is not None else None
        min_w, min_h, init_w, init_h = 900, 600, 1020, 680
        if avail is not None:
            min_w = min(min_w, avail.width() - 40)
            min_h = min(min_h, avail.height() - 80)
            init_w = min(init_w, avail.width() - 40)
            init_h = min(init_h, avail.height() - 80)
        self.setMinimumSize(min_w, min_h)
        self.resize(max(min_w, init_w), max(min_h, init_h))
        self._settings = QSettings("FreeCCR", "FreeCCR")

        self._current_path = current_path
        self._source_path: Optional[str] = None
        self._decoded: Optional[np.ndarray] = None
        self._profile: Optional[ff.FieldProfile] = None
        self.saved_path: Optional[str] = None
        self.apply_now = False

        self.stack = QStackedWidget(self)
        self.stack.addWidget(self._build_source_page())
        self.stack.addWidget(self._build_review_page())
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
    # Page 1 — calibration shot
    # ------------------------------------------------------------------ #
    def _build_source_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>Step 1 — Calibration shot</b>"))
        guide = QLabel(
            "Field correction removes the dark corners and colour shading your "
            "lens, sensor and light source put into every scan. It is measured "
            "from a single photo of an <b>evenly lit blank surface</b>.<br><br>"
            "<b>How to shoot it</b><ul>"
            "<li>Photograph <b>the light source you scan on</b> — the bare "
            "panel, no film — using the <b>same lens, aperture, focal length "
            "and distance</b> as your scans. Aperture matters most: vignetting "
            "changes by a stop or more between f/2.8 and f/8.</li>"
            "<li><b>Defocus slightly</b>, or lay a diffuser over the panel, so "
            "its texture and dust don't print into the profile.</li>"
            "<li><b>Fill the whole frame</b> — nothing dark at the edges.</li>"
            "<li>Expose so the brightest area is around <b>half to "
            "three-quarters</b> of full brightness, with <b>nothing clipped</b>."
            "</li>"
            "<li><b>Shoot RAW.</b> Other formats work but are less exact.</li>"
            "</ul>"
            "Re-shoot the profile whenever the rig changes (lens, aperture, "
            "panel or height).")
        guide.setWordWrap(True)
        lay.addWidget(guide)
        row = QHBoxLayout()
        self.use_current_btn = QPushButton("Use current image")
        self.use_current_btn.setEnabled(bool(self._current_path))
        self.use_current_btn.clicked.connect(self._use_current)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_source)
        row.addWidget(self.use_current_btn)
        row.addWidget(browse)
        row.addStretch(1)
        lay.addLayout(row)
        self.source_label = QLabel("<i>No calibration shot selected.</i>")
        self.source_label.setWordWrap(True)
        lay.addWidget(self.source_label)
        lay.addStretch(1)
        return w

    def _use_current(self):
        if self._current_path:
            self._set_source(self._current_path)

    def _browse_source(self):
        start = self._settings.value("files/last_open_dir", "", type=str)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select the calibration shot", start,
            "Images (*.dng *.tif *.tiff *.arw *.nef *.cr2 *.cr3 *.raf *.png "
            "*.jpg *.jpeg *.rw2 *.orf *.srw *.pef *.3fr *.fff);;All Files (*)")
        if path:
            self._set_source(path)

    def _set_source(self, path):
        self._source_path = path
        self._decoded = None                 # force a re-decode on Next
        self._profile = None
        is_raw = os.path.splitext(path)[1].lower() in _RAW_EXTS
        warn = ("" if is_raw else
                "<br><span style='color:#cc7a00'>Note: not a RAW file. The "
                "profile will be measured in this file's own encoding — use it "
                "with scans in the same format for an exact match.</span>")
        self.source_label.setText(
            f"Selected: <b>{os.path.basename(path)}</b>{warn}")
        self._update_nav()

    def _decode_source(self) -> bool:
        if self._decoded is not None:
            return True
        norm = normalize_unicode_path(self._source_path)
        if not validate_unicode_path(norm):
            QMessageBox.warning(self, "Unicode Path",
                                "This file path can't be processed. Please move "
                                "it to a simpler path.")
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            arr = ff.decode_calibration(norm)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Decode Error",
                                 f"Could not read the calibration shot:\n\n{e}")
            return False
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        if arr is None or arr.ndim != 3:
            QMessageBox.critical(self, "Decode Error",
                                 "Could not decode the calibration shot.")
            return False
        self._decoded = arr
        return True

    # ------------------------------------------------------------------ #
    # Page 2 — review
    # ------------------------------------------------------------------ #
    def _build_review_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>Step 2 — Review the measured falloff</b>"))
        panes = QHBoxLayout()
        self.pane_shot = _Pane("Calibration shot")
        self.pane_gain = _Pane("Correction applied")
        self.pane_fixed = _Pane("Corrected result")
        for p in (self.pane_shot, self.pane_gain, self.pane_fixed):
            panes.addWidget(p, 1)
        lay.addLayout(panes, 1)

        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        lay.addWidget(self.stats_label)
        self.warn_label = QLabel("")
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet("color: #cc7a00;")
        lay.addWidget(self.warn_label)

        row = QHBoxLayout()
        row.addWidget(QLabel("Profile name:"))
        self.name_edit = QLineEdit()
        row.addWidget(self.name_edit, 1)
        lay.addLayout(row)
        return w

    def _build_and_show(self) -> bool:
        meta = ff.source_metadata(self._source_path)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            profile = ff.build_profile(
                self._decoded, self.name_edit.text().strip()
                or ff.suggested_name(meta), meta=meta)
        except ff.FieldProfileError as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Calibration shot", str(e))
            return False
        finally:
            if QApplication.overrideCursor() is not None:
                QApplication.restoreOverrideCursor()
        self._profile = profile

        # Both photo panes share one exposure normalisation — the before/after
        # is only readable if the two are displayed at the same scale — and it is
        # taken over BOTH images so the corrected one (whose corners have been
        # lifted to the centre's level) doesn't sit clipped against white.
        corrected = ff.apply_field(self._decoded, profile)
        # ...with a stop of display headroom on top: a corrected frame sits at
        # the centre's own level, so without it the "should look even" pane is a
        # flat white rectangle and any residual unevenness is invisible.
        norm = 2.0 * max(float(np.percentile(self._decoded, 99.5)),
                         float(np.percentile(corrected, 99.5)), 1.0)
        self.pane_shot.set_image(_gamma_stretch_qimage(self._decoded, norm)[0],
                                 "as shot — dark corners")
        heat, hi = _gain_heatmap_qimage(profile.gain)
        self.pane_gain.set_image(heat, f"1.0x (dark) … {hi:.2f}x (bright)")
        self.pane_fixed.set_image(_gamma_stretch_qimage(corrected, norm)[0],
                                  "should look even")

        st = profile.stats
        cs = st["corner_stops"]
        self.stats_label.setText(
            f"Corner falloff: <b>R {cs[0]:+.2f} · G {cs[1]:+.2f} · "
            f"B {cs[2]:+.2f} EV</b> &nbsp;·&nbsp; colour spread "
            f"<b>{st['cast_stops']:.2f} EV</b> &nbsp;·&nbsp; max gain "
            f"<b>{st['max_gain']:.2f}x</b> &nbsp;·&nbsp; brightness "
            f"{st['level'] * 100:.0f}% of full"
            + ("<br><span style='color:#cc7a00'>The correction hit the "
               f"{ff.DEFAULT_MAX_GAIN:.0f}x limit "
               f"(needed {st['max_gain_uncapped']:.1f}x) — the darkest areas "
               "stay a little dark on purpose, because lifting them further "
               "would mostly amplify noise.</span>" if st["clamped"] else ""))
        warnings = profile.meta.get("warnings") or []
        self.warn_label.setText("⚠ " + "<br>⚠ ".join(warnings) if warnings else "")
        self.warn_label.setVisible(bool(warnings))
        if not self.name_edit.text().strip():
            self.name_edit.setText(profile.name)
        return True

    # ------------------------------------------------------------------ #
    # Page 3 — save
    # ------------------------------------------------------------------ #
    def _build_save_page(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(QLabel("<b>Step 3 — Save</b>"))
        self.save_label = QLabel(
            "The profile is saved in FreeCCR's field-correction library, where "
            "it survives updates. Pick it any time in Settings ▸ Color "
            "Management.")
        self.save_label.setWordWrap(True)
        lay.addWidget(self.save_label)
        self.path_label = QLabel("")
        self.path_label.setWordWrap(True)
        self.path_label.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(self.path_label)
        self.apply_check = QCheckBox("Use this profile now")
        self.apply_check.setChecked(True)
        lay.addWidget(self.apply_check)
        hint = QLabel(
            "Images already loaded keep the decode they were graded with. They "
            "are flagged ⚠ in the thumbnail list — right-click ▸ <i>Replace with "
            "current camera profile</i> to re-grade the ones you want.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(hint)
        lay.addStretch(1)
        return w

    def _target_path(self) -> str:
        name = self.name_edit.text().strip() or "Field correction"
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in name)
        return os.path.join(ccr_backend.field_profiles_dir(), safe + ff.EXTENSION)

    def _do_save(self) -> bool:
        path = self._target_path()
        if os.path.exists(path) and QMessageBox.question(
                self, "Replace profile?",
                f"“{os.path.basename(path)}” already exists in your library.\n"
                "Replace it?") != QMessageBox.Yes:
            return False
        self._profile.name = self.name_edit.text().strip() or self._profile.name
        try:
            ff.save_profile(self._profile, path)
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
        if i == self.PAGE_SOURCE:
            if not self._source_path:
                QMessageBox.information(self, "Calibration shot",
                                        "Select a calibration shot.")
                return
            if not self._decode_source() or not self._build_and_show():
                return
        elif i == self.PAGE_REVIEW:
            if self._profile is None:
                return
            self.path_label.setText(f"Saving to: {self._target_path()}")
        elif i == self.PAGE_SAVE:
            if self._do_save():
                self.accept()
            return
        self.stack.setCurrentIndex(i + 1)
        self._update_nav()

    def _update_nav(self):
        i = self.stack.currentIndex()
        self.back_btn.setEnabled(i > 0)
        self.next_btn.setText("Finish" if i == self.PAGE_SAVE else "Next")

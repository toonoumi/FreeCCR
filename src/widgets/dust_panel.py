"""
Dust Removal panel — covers the right-hand sliders panel while in dust mode.

Two sections sharing one non-destructive model (spots stored on the image):
  - Manual: a sized brush; paint over dust on the canvas to inpaint it.
  - AI: an ONNX detector (downloaded on first use) finds dust automatically;
    the same cv2.inpaint fills it.

All ONNX work happens off the GUI thread; the panel imports `dust_detect`
(which never imports onnxruntime at module level), so this widget is safe to
build even when onnxruntime is absent — the AI section then shows an
unavailable/needs-download state and the manual brush is unaffected.
See spec/dust-removal.md.
"""
import math

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                QPushButton, QSlider, QFrame, QProgressBar,
                                QMessageBox)
from PySide6.QtCore import Qt, QObject, QThread, QTimer, Signal

from core import dust_detect
from ui import theme

# Brush radius limits as a fraction of image width (the canvas clamps to the
# same range). The slider maps its integer steps onto this range LOG-scaled:
# dust spotting needs fine steps at the small end (0.05% is ~3 px radius on a
# 6000 px scan) without giving up the big scratch-covering sizes at the top —
# linearly, everything below 1% would sit in the first pixels of travel.
DUST_BRUSH_R_MIN = 0.0005
DUST_BRUSH_R_MAX = 0.2
BRUSH_STEPS = 300
_BRUSH_LOG_SPAN = math.log(DUST_BRUSH_R_MAX / DUST_BRUSH_R_MIN)


def slider_to_brush_r(value: int) -> float:
    """Slider step (0..BRUSH_STEPS) -> normalized brush radius, log-scaled."""
    v = max(0, min(BRUSH_STEPS, int(value)))
    return DUST_BRUSH_R_MIN * math.exp(_BRUSH_LOG_SPAN * v / BRUSH_STEPS)


def brush_r_to_slider(r_norm: float) -> int:
    """Normalized brush radius -> nearest slider step (inverse of the above)."""
    r = max(DUST_BRUSH_R_MIN, min(DUST_BRUSH_R_MAX, float(r_norm)))
    return int(round(BRUSH_STEPS * math.log(r / DUST_BRUSH_R_MIN)
                     / _BRUSH_LOG_SPAN))


class _DetectWorker(QObject):
    done = Signal(object)   # (prob, luma) numpy arrays
    failed = Signal(str)

    def __init__(self, source):
        super().__init__()
        self.source = source

    def run(self):
        try:
            prob, luma = dust_detect.detect(self.source)
            self.done.emit((prob, luma))
        except Exception as e:  # noqa: BLE001 — surfaced to the user
            self.failed.emit(str(e))


class _DetectAllWorker(QObject):
    """Runs AI detection on EVERY target image off the GUI thread. Detection is
    per-image (each scan has its own dust), at the current sensitivity. Emits a
    (img, spots) pair per image so the GUI thread applies them; reads only stable
    per-image state (resized_raw + manual spots) off-thread."""
    detected = Signal(object, object)   # img, spots
    progress = Signal(int, int)         # done, total
    done = Signal(int)                  # processed count
    failed = Signal(str)

    def __init__(self, targets, sensitivity):
        super().__init__()
        self._targets = targets
        self._sensitivity = sensitivity
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from widgets.image_preview import ImagePreview
            total = len(self._targets)
            processed = 0
            for img in self._targets:
                if self._cancel:
                    break
                source = ImagePreview.dust_detect_source_for(img)
                if source is None:
                    self.progress.emit(processed, total)
                    continue
                prob, luma = dust_detect.detect(source)
                spots = dust_detect.prob_to_spots(prob, luma, self._sensitivity)
                self.detected.emit(img, spots)
                processed += 1
                self.progress.emit(processed, total)
            self.done.emit(processed)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class _DownloadWorker(QObject):
    progress = Signal(int, int)   # done_bytes, total_bytes
    done = Signal()
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            dust_detect.download_model(
                progress_cb=lambda d, t: self.progress.emit(d, t),
                should_cancel=lambda: self._cancel)
            self.done.emit()
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class DustRemovalPanel(QWidget):
    def __init__(self, main_window, image_preview, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.image_preview = image_preview
        self._downloading = False
        self._detecting = False
        self._detecting_all = False
        self._download_thread = None
        self._download_worker = None
        self._detect_thread = None
        self._detect_worker = None
        self._detect_all_thread = None
        self._detect_all_worker = None
        # Cached detector outputs (probability map + luma) for the current
        # image so the Sensitivity slider re-thresholds without re-running the net.
        self._prob = None
        self._luma = None
        self._prob_image_ref = None
        # The image a running detection was started on — results are discarded
        # if the user switches images before it finishes.
        self._detect_image_ref = None
        # Debounce for the Feather slider: re-healing every spot on each drag
        # tick would thrash; apply shortly after the value settles.
        self._feather_timer = QTimer(self)
        self._feather_timer.setSingleShot(True)
        self._feather_timer.setInterval(200)
        self._feather_timer.timeout.connect(self._apply_feather)

        self._build_ui()
        self._refresh_ai_section()

    # --- UI ---------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        theme.apply_panel_spacing(layout)

        header = QLabel("Dust Removal")
        header.setAlignment(Qt.AlignCenter)
        header.setStyleSheet("font-size: 14px; font-weight: bold; margin: 4px;")
        layout.addWidget(header)

        # --- Manual section ---
        layout.addWidget(self._section_label("Manual"))
        brush_row = QHBoxLayout()
        brush_row.setSpacing(theme.GAP_TIGHT)
        brush_lbl = QLabel("Brush size")
        brush_lbl.setFixedWidth(theme.LABEL_COL_W)
        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(0)     # log scale: see slider_to_brush_r
        self.brush_slider.setMaximum(BRUSH_STEPS)
        self.brush_slider.setValue(brush_r_to_slider(0.012))  # 1.2% default
        self.brush_slider.setFixedHeight(theme.CONTROL_H)
        self.brush_value = QLabel("1.20%")
        self.brush_value.setFixedWidth(theme.VALUE_COL_W)
        self.brush_slider.valueChanged.connect(self._on_brush_changed)
        brush_row.addWidget(brush_lbl)
        brush_row.addWidget(self.brush_slider)
        brush_row.addWidget(self.brush_value)
        layout.addLayout(brush_row)

        # Edge feather: per-image render parameter applied to ALL of the
        # image's spots — manual and AI — live. Relative to the STROKE: the
        # value is the fraction of each stroke's edge-to-center distance the
        # fade spans, so a big dab feathers proportionally wide while a tight
        # trace stays near-hard.
        feather_row = QHBoxLayout()
        feather_row.setSpacing(theme.GAP_TIGHT)
        feather_lbl = QLabel("Feather")
        feather_lbl.setFixedWidth(theme.LABEL_COL_W)
        self.feather_slider = QSlider(Qt.Horizontal)
        self.feather_slider.setMinimum(0)    # f = value/100 -> 0..100% of stroke
        self.feather_slider.setMaximum(100)
        self.feather_slider.setValue(35)     # 35% default
        self.feather_slider.setFixedHeight(theme.CONTROL_H)
        self.feather_value = QLabel("35%")
        self.feather_value.setFixedWidth(theme.VALUE_COL_W)
        self.feather_slider.valueChanged.connect(self._on_feather_changed)
        feather_row.addWidget(feather_lbl)
        feather_row.addWidget(self.feather_slider)
        feather_row.addWidget(self.feather_value)
        layout.addLayout(feather_row)

        hint = QLabel("Click or drag over dust to remove it.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(hint)

        manual_btns = QHBoxLayout()
        theme.apply_button_row(manual_btns)
        self.undo_btn = QPushButton("Undo last spot")
        self.undo_btn.clicked.connect(self._on_undo_last)
        self.clear_btn = QPushButton("Clear all")
        self.clear_btn.clicked.connect(self._on_clear_all)
        theme.style_button(self.clear_btn, "danger")  # removes all spots
        manual_btns.addWidget(self.undo_btn)
        manual_btns.addWidget(self.clear_btn)
        layout.addLayout(manual_btns)

        layout.addWidget(self._separator())

        # --- AI section ---
        layout.addWidget(self._section_label("AI Dust Detection"))

        self.ai_unavailable_label = QLabel(
            "AI detection unavailable in this build.")
        self.ai_unavailable_label.setWordWrap(True)
        self.ai_unavailable_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self.ai_unavailable_label)

        self.download_label = QLabel(
            "Download the local AI model (~150 MB) to detect dust "
            "automatically. One-time download.")
        self.download_label.setWordWrap(True)
        self.download_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self.download_label)
        self.download_btn = QPushButton("Download AI model (~150 MB)")
        self.download_btn.setFixedHeight(theme.CONTROL_H)
        self.download_btn.clicked.connect(self._on_download)
        layout.addWidget(self.download_btn)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        sens_row = QHBoxLayout()
        sens_row.setSpacing(theme.GAP_TIGHT)
        self.sensitivity_label = QLabel("Sensitivity")
        self.sensitivity_label.setFixedWidth(theme.LABEL_COL_W)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(0)
        self.sensitivity_slider.setMaximum(100)
        self.sensitivity_slider.setValue(35)  # conservative default — fewer false positives
        self.sensitivity_slider.setFixedHeight(theme.CONTROL_H)
        self.sensitivity_value = QLabel("35")
        self.sensitivity_value.setFixedWidth(theme.VALUE_COL_W)
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        sens_row.addWidget(self.sensitivity_label)
        sens_row.addWidget(self.sensitivity_slider)
        sens_row.addWidget(self.sensitivity_value)
        self._sens_row = sens_row
        layout.addLayout(sens_row)

        self.detect_btn = QPushButton("Detect && Remove")
        self.detect_btn.setFixedHeight(theme.CONTROL_H)
        self.detect_btn.clicked.connect(self._on_detect)
        layout.addWidget(self.detect_btn)

        self.detect_all_btn = QPushButton("Detect && Remove — All Images")
        self.detect_all_btn.setFixedHeight(theme.CONTROL_H)
        self.detect_all_btn.setToolTip(
            "Run AI dust detection on every converted image — each scan is "
            "detected on its own (not by copying spots), at the current "
            "sensitivity.")
        self.detect_all_btn.clicked.connect(self._on_detect_all)
        layout.addWidget(self.detect_all_btn)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {theme.TEXT_MUTED}; font-size: 11px;")
        layout.addWidget(self.status_label)

        layout.addStretch(1)

        self.done_btn = QPushButton("✓  Done")
        self.done_btn.setMinimumHeight(theme.CONTROL_H_LG)
        self.done_btn.setToolTip("Close dust removal and return to the adjustment sliders.")
        # The panel's commit action → neutral primary (no success-green chrome).
        theme.style_button(self.done_btn, "primary")
        self.done_btn.clicked.connect(self._on_done)
        layout.addWidget(self.done_btn)

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
        """Refresh per-image state when the panel is shown. The detector prob
        cache is per image, so invalidate it on a different image."""
        img = self._current_image()
        if img is not self._prob_image_ref:
            self._prob = None
            self._luma = None
            self._prob_image_ref = None
        # Push the current brush size to the canvas so they agree on entry.
        self.image_preview.set_dust_brush_size(
            slider_to_brush_r(self.brush_slider.value()))
        # Reflect this image's feather without re-triggering a re-render.
        from core.ccr_processor import DUST_FEATHER_DEFAULT
        f = (getattr(img, "dust_feather", DUST_FEATHER_DEFAULT)
             if img is not None else DUST_FEATHER_DEFAULT)
        self.feather_slider.blockSignals(True)
        self.feather_slider.setValue(int(round(f * 100)))
        self.feather_slider.blockSignals(False)
        self.feather_value.setText(f"{f * 100.0:.0f}%")
        self._refresh_ai_section()
        self.status_label.setText("")

    def sync_brush_size(self, r_norm: float):
        """Reflect a wheel-driven brush change from the canvas without
        re-emitting back to the canvas. The label shows the canvas's exact
        radius (the slider is quantized to its nearest log step)."""
        self.brush_slider.blockSignals(True)
        self.brush_slider.setValue(brush_r_to_slider(r_norm))
        self.brush_slider.blockSignals(False)
        r = max(DUST_BRUSH_R_MIN, min(DUST_BRUSH_R_MAX, float(r_norm)))
        self.brush_value.setText(f"{r * 100.0:.2f}%")

    # --- Manual handlers --------------------------------------------------
    def _on_brush_changed(self, value):
        r = slider_to_brush_r(value)
        self.brush_value.setText(f"{r * 100.0:.2f}%")
        self.image_preview.set_dust_brush_size(r)

    def _on_feather_changed(self, value):
        self.feather_value.setText(f"{value}%")
        self._feather_timer.start()

    def _apply_feather(self):
        """Store the slider's feather on the current image and re-heal its
        spots so the edge softness updates live."""
        img = self._current_image()
        if img is None:
            return
        f = self.feather_slider.value() / 100.0
        if abs(getattr(img, "dust_feather", -1.0) - f) < 1e-9:
            return
        img.dust_feather = f
        if getattr(img, "dust_spots", None):
            self.image_preview._commit_dust_change(img)

    def _on_undo_last(self):
        if not self.image_preview.dust_undo_last():
            self.status_label.setText("Nothing to undo.")

    def _on_clear_all(self):
        from core.ccr_backend import ccr_backend
        img = ccr_backend.get_image_by_index(self.image_preview.current_idx)
        n = len(getattr(img, "dust_spots", []) or []) if img is not None else 0
        if n == 0:
            self.status_label.setText("No dust spots to clear.")
            return
        if QMessageBox.question(
                self, "Clear All Dust Spots",
                f"Remove all {n} dust spot{'s' if n != 1 else ''} from this image?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        if self.image_preview.dust_clear_all():
            self._prob = None  # spots gone; a re-detect should start fresh
            self.status_label.setText("Cleared all dust spots.")
        else:
            self.status_label.setText("No dust spots to clear.")

    def _on_done(self):
        self.main_window.toggle_dust_removal(False)

    # --- AI handlers ------------------------------------------------------
    def _on_sensitivity_changed(self, value):
        self.sensitivity_value.setText(str(value))
        # Cheap re-threshold of the cached prob map (no net re-run).
        if self._prob is not None and self._current_image() is self._prob_image_ref:
            spots = dust_detect.prob_to_spots(self._prob, self._luma, float(value))
            n = self.image_preview.apply_detected_spots(spots)
            self.status_label.setText(
                f"Removed {n} spot{'s' if n != 1 else ''}." if n
                else "No dust found at this sensitivity.")

    def _on_detect(self):
        if self._detecting or self._detecting_all or self._downloading:
            return
        source = self.image_preview.dust_detect_source()
        if source is None:
            self.status_label.setText("No image to detect on.")
            return
        self._detecting = True
        self._detect_image_ref = self._current_image()
        self.detect_btn.setEnabled(False)
        self.detect_btn.setText("Detecting…")
        self.status_label.setText("Running AI detection…")
        self._detect_thread = QThread()
        self._detect_worker = _DetectWorker(source)
        self._detect_worker.moveToThread(self._detect_thread)
        self._detect_thread.started.connect(self._detect_worker.run)
        self._detect_worker.done.connect(self._on_detect_done)
        self._detect_worker.failed.connect(self._on_detect_failed)
        self._detect_worker.done.connect(self._detect_thread.quit)
        self._detect_worker.failed.connect(self._detect_thread.quit)
        self._detect_worker.done.connect(self._detect_worker.deleteLater)
        self._detect_worker.failed.connect(self._detect_worker.deleteLater)
        self._detect_thread.finished.connect(self._clear_detect_thread)
        self._detect_thread.start()

    def _on_detect_done(self, result):
        # Discard results if the user navigated to a different image (or out of
        # dust mode) while detection was running — the prob map is for the old
        # image and must not be applied to the current one.
        if (self._current_image() is not self._detect_image_ref
                or not self.image_preview.dust_mode):
            self._finish_detect()
            return
        prob, luma = result
        self._prob = prob
        self._luma = luma
        self._prob_image_ref = self._current_image()
        spots = dust_detect.prob_to_spots(
            prob, luma, float(self.sensitivity_slider.value()))
        n = self.image_preview.apply_detected_spots(spots)
        self.status_label.setText(
            f"Removed {n} spot{'s' if n != 1 else ''}." if n
            else "No dust found.")
        self._finish_detect()

    def _on_detect_failed(self, msg):
        self.status_label.setText(f"AI detection failed: {msg}")
        self._finish_detect()

    def _finish_detect(self):
        self._detecting = False
        self.detect_btn.setEnabled(True)
        self.detect_btn.setText("Detect && Remove")

    def _clear_detect_thread(self):
        self._detect_thread = None
        self._detect_worker = None

    # --- AI: detect on ALL images ----------------------------------------
    def _on_detect_all(self):
        if self._detecting or self._detecting_all or self._downloading:
            return
        from core.ccr_backend import ccr_backend
        targets = [im for im in ccr_backend.images
                   if (getattr(im, "converted", False) or ccr_backend.positive_mode)
                   and getattr(im, "resized_raw", None) is not None]
        if not targets:
            self.status_label.setText("No converted images to detect on.")
            return
        n = len(targets)
        if QMessageBox.question(
                self, "Detect on All Images",
                f"Run AI dust detection on all {n} image{'s' if n != 1 else ''}? "
                "This may take a while.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        self._detecting_all = True
        self.detect_btn.setEnabled(False)
        self.detect_all_btn.setEnabled(False)
        self.detect_all_btn.setText("Detecting all…")
        self.status_label.setText(f"AI dust detection on {len(targets)} images…")
        try:
            self._detect_all_thread = QThread()
            self._detect_all_worker = _DetectAllWorker(
                targets, float(self.sensitivity_slider.value()))
            self._detect_all_worker.moveToThread(self._detect_all_thread)
            self._detect_all_thread.started.connect(self._detect_all_worker.run)
            self._detect_all_worker.detected.connect(self._on_detect_all_one)
            self._detect_all_worker.progress.connect(self._on_detect_all_progress)
            self._detect_all_worker.done.connect(self._on_detect_all_done)
            self._detect_all_worker.failed.connect(self._on_detect_all_failed)
            self._detect_all_worker.done.connect(self._detect_all_thread.quit)
            self._detect_all_worker.failed.connect(self._detect_all_thread.quit)
            self._detect_all_worker.done.connect(self._detect_all_worker.deleteLater)
            self._detect_all_worker.failed.connect(self._detect_all_worker.deleteLater)
            self._detect_all_thread.finished.connect(self._clear_detect_all_thread)
            self._detect_all_thread.start()
        except Exception as e:  # noqa: BLE001 — never leave the flag/buttons stuck
            self._detect_all_thread = None
            self._detect_all_worker = None
            self._finish_detect_all()
            self.status_label.setText(f"Could not start batch detection: {e}")

    def _finish_detect_all(self):
        self._detecting_all = False
        self.detect_btn.setEnabled(True)
        self.detect_all_btn.setEnabled(True)
        self.detect_all_btn.setText("Detect && Remove — All Images")

    def _on_detect_all_one(self, img, spots):
        # Runs on the GUI thread (queued from the worker). Discard if the user
        # left dust mode while the batch was running (Done/Esc cancels it, but a
        # signal may already be queued); apply_auto_spots_to_image also ignores
        # an image no longer loaded.
        if not self.image_preview.dust_mode:
            return
        self.image_preview.apply_auto_spots_to_image(img, spots)
        if img is self._prob_image_ref:
            self._prob = None
            self._luma = None
            self._prob_image_ref = None

    def _on_detect_all_progress(self, done, total):
        self.status_label.setText(f"AI dust detection… {done}/{total}")

    def _on_detect_all_done(self, count):
        self._finish_detect_all()
        self.status_label.setText(
            f"AI dust removal applied to {count} image{'s' if count != 1 else ''}.")
        try:
            from core.ccr_backend import ccr_backend
            ccr_backend.save_catalog()
        except Exception:
            pass

    def _on_detect_all_failed(self, msg):
        self._finish_detect_all()
        self.status_label.setText(f"Batch AI detection failed: {msg}")

    def _clear_detect_all_thread(self):
        self._detect_all_thread = None
        self._detect_all_worker = None

    def _on_download(self):
        if self._downloading:
            return
        self._downloading = True
        self.progress_bar.setValue(0)
        self.status_label.setText("Downloading AI model…")
        self._refresh_ai_section()
        self._download_thread = QThread()
        self._download_worker = _DownloadWorker()
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(self._on_download_progress)
        self._download_worker.done.connect(self._on_download_done)
        self._download_worker.failed.connect(self._on_download_failed)
        self._download_worker.done.connect(self._download_thread.quit)
        self._download_worker.failed.connect(self._download_thread.quit)
        self._download_worker.done.connect(self._download_worker.deleteLater)
        self._download_worker.failed.connect(self._download_worker.deleteLater)
        self._download_thread.finished.connect(self._clear_download_thread)
        self._download_thread.start()

    def _on_download_progress(self, done, total):
        if total > 0:
            self.progress_bar.setValue(int(done * 100 / total))

    def _on_download_done(self):
        self._downloading = False
        self.status_label.setText("AI model ready.")
        self._refresh_ai_section()

    def _on_download_failed(self, msg):
        self._downloading = False
        self.status_label.setText(f"Download failed: {msg}")
        self._refresh_ai_section()

    def _clear_download_thread(self):
        self._download_thread = None
        self._download_worker = None

    # --- helpers ----------------------------------------------------------
    def _refresh_ai_section(self):
        avail = dust_detect.is_available()
        present = avail and dust_detect.is_model_present()
        need_dl = avail and not present and not self._downloading

        if not avail:
            self.ai_unavailable_label.setText(
                dust_detect.availability_reason()
                or "AI detection unavailable in this build.")
        self.ai_unavailable_label.setVisible(not avail)
        self.download_label.setVisible(need_dl)
        self.download_btn.setVisible(need_dl)
        self.progress_bar.setVisible(self._downloading)
        self.sensitivity_label.setVisible(present)
        self.sensitivity_slider.setVisible(present)
        self.sensitivity_value.setVisible(present)
        self.detect_btn.setVisible(present)
        self.detect_all_btn.setVisible(present)

    def _current_image(self):
        from core.ccr_backend import ccr_backend
        idx = self.image_preview.current_idx
        return ccr_backend.get_image_by_index(idx) if idx is not None else None

    def cancel_jobs(self):
        """Cancel in-flight model download / batch detection when leaving dust
        mode (Done/Esc). Those workers honour cancel(); the single short ONNX
        detection has no interrupt, but its result is discarded if the image
        changed."""
        for wkr in (self._download_worker, self._detect_all_worker):
            if wkr is not None:
                try:
                    wkr.cancel()
                except Exception:
                    pass

    def shutdown(self):
        """Stop any in-flight download/detection thread (called on app close).
        Mirrors MainWindow._stop_loader_if_running: cancel, quit, wait, then
        drop the refs so isRunning() is never called on a freed C++ object."""
        for wkr in (self._download_worker, self._detect_all_worker):
            if wkr is not None:
                try:
                    wkr.cancel()
                except Exception:
                    pass
        for attr_thr, attr_wkr in (("_download_thread", "_download_worker"),
                                   ("_detect_thread", "_detect_worker"),
                                   ("_detect_all_thread", "_detect_all_worker")):
            thr = getattr(self, attr_thr, None)
            if thr is not None:
                try:
                    if thr.isRunning():
                        thr.quit()
                        thr.wait(3000)
                except RuntimeError:
                    pass
            setattr(self, attr_thr, None)
            setattr(self, attr_wkr, None)

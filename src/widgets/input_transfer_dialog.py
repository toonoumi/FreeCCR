"""Import-time TIFF colour-space dialog.

FreeCCR's film conversion measures optical density, which assumes LINEAR data,
but a scanner TIFF is often gamma-encoded. This asks — once per import, before
anything is loaded — how to read the TIFFs in the batch: from each file's own
metadata, as linear (the historical behaviour), or with one space forced for the
whole batch. See spec/input-transfer-function.md §4.1.
"""
from typing import List, Optional, Sequence

from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QVBoxLayout,
)

from core import input_transfer
from ui import theme

MODE_EMBEDDED = "embedded"
MODE_LINEAR = "linear"
MODE_MANUAL = "manual"


class InputTransferDialog(QDialog):
    """Three-way choice for the TIFFs of one import.

    Cancel aborts the import outright: the answer changes every pixel the
    conversion sees, so "no answer" must not silently mean linear."""

    def __init__(self, tiff_paths: Sequence[str], parent=None,
                 manual_default: str = input_transfer.SRGB):
        super().__init__(parent)
        self.setWindowTitle("TIFF colour space")
        self.setModal(True)
        self._paths: List[str] = list(tiff_paths)

        tagged, parts = input_transfer.summarize_batch(self._paths)

        root = QVBoxLayout(self)
        theme.apply_panel_spacing(root, spacing=theme.GAP_ROW)

        n = len(self._paths)
        head = QLabel(f"<b>This import contains {n} TIFF "
                      f"file{'s' if n != 1 else ''}.</b>")
        head.setWordWrap(True)
        root.addWidget(head)
        if parts:
            # Described with the SAME resolver the decode uses, so the dialog can
            # never claim something different from how the file will be read.
            root.addWidget(self._muted("  " + " · ".join(parts)))
        root.addWidget(self._muted(
            "FreeCCR's film conversion measures optical density, which assumes "
            "linear data. How should these files be read?"))
        root.addWidget(theme.section_separator())

        self._group = QButtonGroup(self)
        self._rb_embedded = QRadioButton("Use each file's own metadata")
        self._rb_linear = QRadioButton("Read as linear (no conversion)")
        self._rb_manual = QRadioButton("Specify manually:")
        for rb, mode in ((self._rb_embedded, MODE_EMBEDDED),
                         (self._rb_linear, MODE_LINEAR),
                         (self._rb_manual, MODE_MANUAL)):
            self._group.addButton(rb)
            rb.setProperty("transfer_mode", mode)
            rb.toggled.connect(self._sync_enabled)

        root.addWidget(self._rb_embedded)
        root.addWidget(self._muted("      Untagged files are read as linear."))
        root.addWidget(self._rb_linear)
        root.addWidget(self._muted("      What FreeCCR has always done."))

        manual_row = QHBoxLayout()
        manual_row.setSpacing(theme.GAP_BTN)
        manual_row.addWidget(self._rb_manual)
        self._combo = QComboBox()
        for value, label in input_transfer.TRANSFER_CHOICES:
            self._combo.addItem(label, value)
        idx = self._combo.findData(manual_default)
        self._combo.setCurrentIndex(idx if idx >= 0 else 0)
        manual_row.addWidget(self._combo, 1)
        root.addLayout(manual_row)
        root.addWidget(self._muted(
            "      Applied to every TIFF in this import, overriding its metadata."))

        root.addWidget(theme.section_separator())
        self._cb_remember = QCheckBox("Don't ask again — remember this choice")
        root.addWidget(self._cb_remember)

        footer = QHBoxLayout()
        theme.apply_button_row(footer)
        footer.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        footer.addWidget(cancel)
        ok = QPushButton("Import")
        theme.style_button(ok, "primary", default=True)
        ok.clicked.connect(self.accept)
        footer.addWidget(ok)
        root.addLayout(footer)

        # Default to the files' own metadata only when some file actually has
        # any; otherwise the honest default is today's behaviour.
        (self._rb_embedded if tagged else self._rb_linear).setChecked(True)
        self._sync_enabled()
        theme.apply_windows_dark_titlebar(self)

    # --- results ------------------------------------------------------- #
    def mode(self) -> str:
        btn = self._group.checkedButton()
        return (btn.property("transfer_mode") if btn is not None else MODE_LINEAR)

    def manual(self) -> str:
        return self._combo.currentData() or input_transfer.SRGB

    def remember(self) -> bool:
        return bool(self._cb_remember.isChecked())

    def decision(self) -> Optional[str]:
        """The transfer token to tag this import's images with."""
        return decision_for(self.mode(), self.manual())

    # --- internals ----------------------------------------------------- #
    def _sync_enabled(self, *_):
        self._combo.setEnabled(self._rb_manual.isChecked())

    def _muted(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {theme.TEXT_MUTED};")
        return lbl


def decision_for(mode: str, manual: str) -> str:
    """(mode, manual) -> the transfer token images get tagged with. Shared by the
    dialog and the suppressed path, so a remembered answer and a freshly-given
    one can never diverge."""
    if mode == MODE_EMBEDDED:
        return input_transfer.EMBEDDED
    if mode == MODE_MANUAL:
        return manual or input_transfer.SRGB
    return input_transfer.LINEAR

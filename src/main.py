import sys
import os

# FIRST: make print() unable to crash the app (a macOS .app launches with an
# ASCII stdout; any logged non-ASCII character raised UnicodeEncodeError mid-
# operation and aborted it — GitHub issue #86). Everything below prints.
from utils.stream_hardening import harden_std_streams
harden_std_streams()

# Make onnxruntime usable for AI dust detection, then pre-load it BEFORE Qt.
# Two Windows gotchas this handles (see spec/dust-removal.md §8):
#  1) Nuitka standalone build: the bundled onnxruntime .pyd is loaded without
#     its own folder on the DLL search path, so its onnxruntime.dll dependency
#     isn't found -> "DLL initialization routine failed". Adding the capi dir
#     (next to the exe) before import fixes it. (No-op when running from source.)
#  2) onnxruntime's DLLs fail to init if first loaded AFTER Qt, so import it
#     before any PySide6 import.
# Guarded so the app still runs when onnxruntime is absent (manual dust removal
# is unaffected, and the AI panel reports the reason).
try:
    import ctypes
    _bases = []
    for _b in (getattr(sys, "executable", ""), (sys.argv[0] if sys.argv else "")):
        try:
            _b = os.path.dirname(os.path.abspath(_b))
        except Exception:
            continue
        if _b and _b not in _bases:
            _bases.append(_b)
    for _b in _bases:
        _ort_capi = os.path.join(_b, "onnxruntime", "capi")
        if not os.path.isdir(_ort_capi):
            continue
        try:
            os.add_dll_directory(_ort_capi)
        except Exception:
            pass
        # add_dll_directory alone is NOT enough in the Nuitka build: it loads the
        # .pyd without searching capi/ for its onnxruntime.dll dependency. Load
        # the native runtime explicitly (by full path) so the .pyd finds it
        # already in-process. (ctypes-proven; see spec/dust-removal.md §8.)
        for _dll in ("onnxruntime.dll", "onnxruntime_providers_shared.dll"):
            _dp = os.path.join(_ort_capi, _dll)
            if os.path.exists(_dp):
                try:
                    ctypes.WinDLL(_dp)
                except Exception:
                    pass
        break
except Exception:
    pass
try:
    import onnxruntime as _ort  # noqa: F401
    print(f"AI dust detection: onnxruntime {_ort.__version__} ready", flush=True)
    del _ort
except Exception as _e:  # noqa: BLE001
    print(f"AI dust detection: onnxruntime unavailable ({_e})", flush=True)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtCore import QTimer

def main():
    # Command-line file/folder arguments (spec/cli-file-args.md). --help and
    # --version answer and exit before any Qt object exists; the remaining
    # positional paths seed the startup import further down. Qt's own switches
    # stay in sys.argv for QApplication to consume.
    from utils import cli_args
    if cli_args.wants_help(sys.argv[1:]):
        print(cli_args.USAGE)
        return
    if cli_args.wants_version(sys.argv[1:]):
        from version import VERSION
        print(f"FreeCCR {VERSION}")
        return
    startup_paths = cli_args.strip_qt_options(sys.argv[1:])

    from ui.main_window import MainWindow

    app = QApplication(sys.argv)
    from ui.theme import (apply_theme, apply_windows_dark_titlebar,
                          install_dark_titlebar_filter, install_macos_srgb_filter)
    apply_theme(app)  # neutral dark theme (Fusion + palette + global QSS)
    install_dark_titlebar_filter(app)  # dark title bars for dialogs / message boxes
    install_macos_srgb_filter(app)  # color-managed previews on wide-gamut Macs
    _icon_path = os.path.join(os.path.dirname(__file__), 'icons', 'freeccr_logo.png')
    app.setWindowIcon(QIcon(_icon_path))
    print("Starting FreeCCR...")
    window = MainWindow()
    print("MainWindow created, setting up UI...")

    apply_windows_dark_titlebar(window)  # dark native title bar (Win10/11)
    window.show()
    if startup_paths:
        # Queued, not called inline: the import puts a modal-ish loading dialog
        # up and pumps events, so the window must be shown and the event loop
        # running first — otherwise the user stares at an unpainted frame.
        QTimer.singleShot(0, lambda: window.load_paths(startup_paths))
    sys.exit(app.exec())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("Exception occurred:", e)
        traceback.print_exc()
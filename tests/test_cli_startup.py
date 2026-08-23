#!/usr/bin/env python3
"""MainWindow.load_paths — the command-line import seam (spec/cli-file-args.md §6).

The loader itself is stubbed everywhere: these tests assert WHICH importer runs
with WHICH paths (and that warnings replace a load when nothing is openable),
not that images decode. Dialog stubs prove the refactor left Open Files / Open
Folder driving the same helpers.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402


@pytest.fixture
def win(monkeypatch):
    """A MainWindow whose loader and message boxes are stubbed out. Yields
    (window, launched, warnings): `launched` records every _launch_loader call
    as {"files": ..., "folder": ...}; `warnings` records message-box titles."""
    from ui.main_window import MainWindow
    ccr_backend.images.clear()
    ccr_backend.file_paths.clear()
    ccr_backend.rgb_merge_mode = False

    w = MainWindow()
    launched = []
    warnings = []
    monkeypatch.setattr(
        type(w), "_launch_loader",
        lambda self, files=None, folder=None, force_no_merge=False:
            launched.append({"files": files, "folder": folder}))
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warnings.append(a[1])))
    monkeypatch.setattr(QMessageBox, "critical",
                        staticmethod(lambda *a, **k: warnings.append(a[1])))
    # save_catalog() would rewrite the user's real catalog during a test run.
    monkeypatch.setattr(ccr_backend, "save_catalog", lambda *a, **k: None)
    yield w, launched, warnings
    ccr_backend.rgb_merge_mode = False


def _touch(folder, *names):
    made = []
    for n in names:
        p = os.path.join(str(folder), n)
        with open(p, "wb") as f:
            f.write(b"x")
        made.append(p)
    return made


def test_single_folder_argument_imports_the_folder(win, tmp_path):
    w, launched, warnings = win
    d = tmp_path / "roll"
    d.mkdir()
    _touch(d, "a.nef")
    w.load_paths([str(d)])
    assert launched == [{"files": None, "folder": str(d)}]
    assert warnings == []


def test_file_arguments_import_as_a_file_list(win, tmp_path):
    w, launched, warnings = win
    a, b = _touch(tmp_path, "a.nef", "b.nef")
    w.load_paths([a, b])
    assert len(launched) == 1
    assert launched[0]["folder"] is None
    assert launched[0]["files"] == [a, b]
    assert warnings == []


def test_folder_in_a_list_is_expanded_to_its_images(win, tmp_path):
    w, launched, warnings = win
    d = tmp_path / "roll"
    d.mkdir()
    inside, = _touch(d, "inside.nef")
    _touch(d, "notes.txt")            # unsupported: not imported
    loose, = _touch(tmp_path, "loose.nef")
    w.load_paths([loose, str(d)])
    assert launched[0]["files"] == [loose, inside]
    assert launched[0]["folder"] is None


def test_missing_path_warns_and_loads_nothing(win, tmp_path):
    w, launched, warnings = win
    w.load_paths([str(tmp_path / "gone.nef")])
    assert launched == []
    assert warnings == ["Nothing to open"]


def test_missing_path_alongside_a_real_one_still_loads_the_real_one(win, tmp_path,
                                                                    monkeypatch):
    w, launched, warnings = win
    good, = _touch(tmp_path, "a.nef")
    # The report is modal, so it must come AFTER the good paths are underway:
    # record how much had launched by the time the box went up.
    at_warn = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: (warnings.append(a[1]),
                                                      at_warn.append(len(launched)))))
    w.load_paths([good, str(tmp_path / "gone.nef")])
    assert launched[0]["files"] == [good]
    assert warnings == ["Nothing to open"]
    assert at_warn == [1]          # the load had already started


def test_merge_mode_rejects_a_non_triplet_from_the_command_line(win, tmp_path):
    w, launched, warnings = win
    ccr_backend.rgb_merge_mode = True
    a, b = _touch(tmp_path, "a.nef", "b.nef")   # 2 files: not a multiple of 3
    w.load_paths([a, b])
    assert launched == []
    assert warnings == ["3-way RGB merge"]


def test_merge_mode_accepts_a_triplet_from_the_command_line(win, tmp_path):
    w, launched, warnings = win
    ccr_backend.rgb_merge_mode = True
    files = _touch(tmp_path, "r.nef", "g.nef", "b.nef")
    w.load_paths(files)
    assert len(launched) == 1 and launched[0]["files"] == files
    assert warnings == []


def test_open_files_dialog_still_drives_the_shared_importer(win, tmp_path,
                                                            monkeypatch):
    w, launched, warnings = win
    a, = _touch(tmp_path, "a.nef")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a_, **k: ([a], "")))
    w.open_files()
    assert launched == [{"files": [a], "folder": None}]


def test_open_folder_dialog_still_drives_the_shared_importer(win, tmp_path,
                                                             monkeypatch):
    w, launched, warnings = win
    d = tmp_path / "roll"
    d.mkdir()
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a_, **k: str(d)))
    w.open_folder()
    assert launched == [{"files": None, "folder": str(d)}]


def test_cancelled_dialogs_load_nothing(win, monkeypatch):
    w, launched, warnings = win
    monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                        staticmethod(lambda *a_, **k: ([], "")))
    monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                        staticmethod(lambda *a_, **k: ""))
    w.open_files()
    w.open_folder()
    assert launched == [] and warnings == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

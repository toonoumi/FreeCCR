# Command-line File / Folder Arguments

## 1. Goals / Non-goals

### Goals
- `freeccr [PATH ...]` opens the app with those images already importing, so a
  roll can be opened from a shell, a script, or a file manager's "Open With"
  without walking the Open Files / Open Folder dialogs.
- Accept **files**, **folders**, or a mix, in any number:
  - `freeccr /rolls/roll-07` — one folder: identical to **Open Folder**.
  - `freeccr a.nef b.nef c.nef` — a file list: identical to **Open Files**.
  - `freeccr /rolls/a /rolls/b *.nef` — several paths: every folder is expanded
    to its supported images and the whole thing loads as one file list.
- Reuse the existing import paths exactly — Unicode validation, 3-way-merge
  validation, catalog restore, the loading dialog, and the failure report all
  behave as they do for the dialogs. The CLI is a new *entry point*, not a new
  loader.
- `--help` / `-h` prints usage and exits; `--version` / `-V` prints the version
  and exits. Neither starts a GUI.
- Qt's own switches (`-platform offscreen`, `-style Fusion`, …) keep working and
  are never mistaken for paths.

### Non-goals
- No batch/headless mode: the arguments only seed an import, they do not export,
  convert, or otherwise script the app. Every argument-driven action ends in the
  normal GUI with images loaded.
- No recursive folder walk. A folder contributes the images *directly inside*
  it, matching Open Folder.
- No OS integration in this change — no `CFBundleDocumentTypes`, no Windows file
  association, no `QFileOpenEvent` handler, no drag-and-drop. macOS delivers
  double-clicked documents via `QFileOpenEvent`, **not** argv, so associating
  file types is a separate piece of work that can build on `load_paths()`.
- No new persisted setting.

## 2. UX / interaction

```
Usage: freeccr [OPTIONS] [PATH ...]

  PATH   image files and/or folders to open at startup.
         One folder alone loads like Open Folder; anything else loads as a
         file list, with each folder expanded to the images inside it.

Options:
  -h, --help      show this message and exit
  -V, --version   show the version and exit

Qt options (-platform, -style, …) are also accepted.
```

- The window appears first, then the import starts on the next event-loop turn,
  so the user sees the app (and the loading dialog on top of it) rather than a
  frozen splash. Startup with no arguments is byte-for-byte unchanged.
- Paths that do not exist are collected and reported in **one** warning box,
  titled "Nothing to open" only when nothing was importable and "Some paths
  could not be opened" when an import is running behind it (the box goes up
  after the dispatch, so the first title would contradict the load the user can
  see starting). It lists at most 5 with an "… and N more" tail, matching
  the Unicode-warning style already used by `open_files`. The box goes up
  **after** the good paths are dispatched — it is modal, and one mistyped
  argument must not hold up the rest of the import. If *every* argument is
  missing, the app still opens — empty — with that warning.
- A folder argument holding no supported images contributes nothing and is
  reported in the same box as "no supported images".
- An explicitly named file is passed through **whatever its extension** (the
  Open Files dialog has an "All Files" filter, so the CLI is no stricter); files
  found by expanding a folder are filtered to the supported extension set. An
  unopenable file then surfaces through the existing post-load failure report.
- 3-way merge mode applies exactly as in the dialogs: the resulting file list is
  validated up front (all RAW, a multiple of 3, FreeCCR-baked merge TIFFs
  exempt) and a rejection shows the same warning and loads nothing.

## 3. Data model

No persistent state. One transient value object:

```python
OpenPlan(folder: str | None,      # set only for the single-folder case
         files: list[str],        # ordered, de-duplicated
         problems: list[str])     # human-readable, one per bad argument
```

Exactly one of `folder` / `files` is non-empty; both may be empty (nothing
loadable), in which case only `problems` is reported.

`files/last_open_dir` (QSettings) is written by the **dialogs only**, never by a
command-line import. It records where the user *browsed*, and CLI paths are not
browsing: they arrive from a shell, are frequently relative, and
`os.path.dirname("a.nef")` is `""` — so letting the shared helpers write it
would silently wipe the setting and drop both dialogs back to the process cwd.
`freeccr .` would likewise persist a literal `"."`.

## 4. Processing

1. **Strip Qt switches.** Drop any argument starting with `-`, and additionally
   drop the *value* token following a Qt option known to take one
   (`-platform`, `-platformpluginpath`, `-plugin`, `-style`, `-stylesheet`,
   `-session`, `-graphicssystem`, `-display`, `-geometry`, `-qwindowgeometry`,
   `-qwindowtitle`, `-qwindowicon`, and their `--` spellings). `-opt=value`
   carries its value inline and consumes nothing. `--` ends option processing:
   everything after it is a path, even a leading-dash one.
2. **Expand wildcards.** An argument containing `*`, `?` or `[` that does not
   already name an existing path is replaced in place by its `glob` matches,
   sorted case-insensitively (`recursive=True`, so an explicit `**` walks). A
   pattern matching nothing is left literal for step 2 to report. Unix shells
   do this before `argv` reaches us; cmd.exe and PowerShell do not, so doing it
   here is what makes `freeccr *.tif` behave the same on every host. Checking
   for an existing path first keeps a file literally named `shot[1].tif`
   openable. Because expansion happens before the rules below, a pattern
   behaves exactly like the paths it matched — one matching a single folder
   takes the single-folder rule.
3. **Classify** each remaining argument: existing directory, existing file, or
   missing.
4. **Plan.** One argument, a directory ⇒ `folder=` that path. Otherwise expand
   every directory (non-recursive, `SUPPORTED_EXTS`, sorted case-insensitively
   by name) and concatenate with the named files in argument order,
   de-duplicated by normalised absolute path, first occurrence winning.
5. **Dispatch** on the GUI thread via `QTimer.singleShot(0, …)` after
   `window.show()`: `MainWindow.load_paths(paths)` runs the plan, calls the same
   helpers the dialogs call, then reports `problems`.

Steps 1–4 are pure (no Qt, no `MainWindow`) and live in `src/utils/cli_args.py`;
only step 4 touches the GUI.

## 5. Integration points

- `src/utils/cli_args.py` **(new)** — `USAGE`, `wants_help`, `wants_version`,
  `strip_qt_options`, `expand_folder`, `plan_open`. `expand_folder` imports
  `core.tether_watcher.SUPPORTED_EXTS` lazily (and accepts an injected predicate)
  so the module stays importable without Qt.
- `src/main.py` — handle `--help` / `--version` before `QApplication` is built;
  after `window.show()`, schedule `window.load_paths(paths)` when there are
  arguments. Nothing else in startup moves.
- `src/ui/main_window.py`:
  - `_launch_loader(files=None, folder=None, force_no_merge=False)` — the loader
    thread boilerplate, generalised from (and replacing) `_launch_file_loader`
    so the folder import stops duplicating it.
  - `_import_file_list(files)` / `_import_folder(folder)` — the post-dialog tail
    of `open_files` / `open_folder` (Unicode validation and warning, merge
    validation, loader launch), now callable without a dialog. They deliberately
    do NOT write `last_open_dir` — that stays in the dialog halves.
  - `open_files` / `open_folder` — reduced to `save_catalog()` + dialog + helper.
    `save_catalog()` stays *before* the dialog so a cancelled dialog still
    persists pending edits.
  - `load_paths(paths)` **(new, public)** — `save_catalog()`, `plan_open`,
    `_import_folder` or `_import_file_list`, then report problems. This is the seam any
    later "Open With" / drag-and-drop support hangs off.

## 6. Test plan

`tests/test_cli_args.py` (pure, no `MainWindow`):

- **Qt switch stripping**: `-platform offscreen` consumes its value;
  `--platform=offscreen` consumes nothing extra; a bare `-style` at the end does
  not eat past the list; `-stylesheet dark.qss` does not leak a real file into
  the paths; `--` passes a following `-weird-name.nef` through as a path.
- **Single folder** ⇒ `folder` set, `files` empty.
- **Single file / several files** ⇒ `files` in argument order, `folder` None.
- **Mixed** file + folder ⇒ folder expanded, order preserved, `folder` None.
- **Expansion filters** to supported extensions (a `.txt` in the folder is
  skipped) and sorts case-insensitively; an explicitly named `.txt` is kept.
- **De-duplication**: the same file named twice, and a file also reachable
  through a named folder, appears once.
- **Missing paths** land in `problems` and not in `files`; an empty folder is
  reported too; a plan of nothing but missing paths yields empty `folder`/`files`.
- **Nested folders are not walked** (a subdirectory contributes nothing).
- **Wildcards** expand to their matches in case-insensitive order and plan as a
  file list; `?` matches one character; an unmatched pattern is reported as
  *no files match this pattern* rather than *not found*; a real file named
  `shot[1].nef` wins over character-class interpretation; a pattern matching
  one folder takes the single-folder rule and several folders expand in a list;
  `**` recurses while a single `*` stays at one level.

`tests/test_cli_startup.py` (GUI, offscreen, mirrors `test_tether_watcher.py`'s
`MainWindow()` usage):

- `load_paths([folder])` calls `_import_folder` with that folder;
  `load_paths([f1, f2])` calls `_import_file_list` with both (loader stubbed).
- `load_paths([missing])` loads nothing and reports one warning; a missing path
  next to a good one still launches the load *before* the (modal) report.
- With 3-way merge mode on, `load_paths` of two RAWs is rejected by the existing
  merge validation and launches no loader.
- `open_files` / `open_folder` still drive the same helpers (dialog stubbed), so
  the refactor is behaviour-preserving.

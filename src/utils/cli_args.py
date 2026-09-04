"""
Command-line file / folder arguments.

`freeccr [PATH ...]` seeds the startup import with files, folders, wildcard
patterns, or a mix, so a roll can be opened from a shell or a script without
walking the Open Files / Open Folder dialogs. This module is the PURE half —
argument parsing, glob and folder expansion, planning, no Qt and no MainWindow
— so the whole decision is unit-testable; `MainWindow.load_paths` performs the
plan. See spec/cli-file-args.md.
"""
import glob
import os
from typing import Callable, List, NamedTuple, Optional, Sequence

USAGE = """Usage: freeccr [OPTIONS] [PATH ...]

  PATH   image files and/or folders to open at startup.
         One folder alone loads like Open Folder; anything else loads as a
         file list, with each folder expanded to the images inside it.
         Wildcards (*.tif, roll-?.nef) are expanded by FreeCCR itself, so
         they work in cmd.exe and PowerShell as well as in a Unix shell.

Options:
  -h, --help      show this message and exit
  -V, --version   show the version and exit

Qt options (-platform, -style, ...) are also accepted."""

# Characters that make an argument a wildcard pattern rather than a literal
# path. `[` is included because glob honours character classes, but an argument
# naming a real file always wins over pattern interpretation (see expand_globs),
# so a photo literally called `shot[1].tif` still opens.
WILDCARD_CHARS = "*?["

HELP_FLAGS = frozenset({"-h", "--help"})
VERSION_FLAGS = frozenset({"-V", "--version"})

# Qt/QApplication switches that take a SEPARATE value token. The value must be
# skipped too, or `-stylesheet dark.qss` would offer dark.qss as an image to
# open. Listed without the leading dash; both the `-opt` and `--opt` spellings
# are recognised. An inline `-opt=value` carries its value and consumes nothing.
# Value-LESS switches (-reverse, -widgetcount, -nograb, ...) are not listed:
# they consume nothing, and listing one would make it swallow a real path.
QT_VALUE_OPTIONS = frozenset({
    "platform", "platformpluginpath", "platformtheme", "plugin", "style",
    "stylesheet", "session", "graphicssystem", "display", "geometry",
    "qwindowgeometry", "qwindowtitle", "qwindowicon", "qmljsdebugger",
    "title", "name", "visual", "ncols", "cmap", "font", "fn",
})


class OpenPlan(NamedTuple):
    """What a set of command-line paths should open.

    folder:   set ONLY for the single-folder case (loads like Open Folder).
    files:    ordered, de-duplicated file list (folders already expanded).
    problems: one human-readable line per argument that contributed nothing.

    At most one of folder/files is non-empty; both are empty when nothing
    loadable was named (problems then says why).
    """
    folder: Optional[str]
    files: List[str]
    problems: List[str]


def wants_help(argv: Sequence[str]) -> bool:
    return any(a in HELP_FLAGS for a in _before_ddash(argv))


def wants_version(argv: Sequence[str]) -> bool:
    return any(a in VERSION_FLAGS for a in _before_ddash(argv))


def _before_ddash(argv: Sequence[str]) -> List[str]:
    """The arguments up to a bare `--` (after it, everything is a path)."""
    out: List[str] = []
    for a in argv:
        if a == "--":
            break
        out.append(a)
    return out


def strip_qt_options(argv: Sequence[str]) -> List[str]:
    """Return the positional (path) arguments of `argv`, dropping option
    switches and the value token of any Qt option known to take one. A bare
    `--` ends option processing: everything after it is a path, even one that
    starts with a dash."""
    paths: List[str] = []
    skip_next = False
    literal = False
    for arg in argv:
        if literal:
            paths.append(arg)
            continue
        if skip_next:
            skip_next = False
            continue
        if arg == "--":
            literal = True
            continue
        if arg.startswith("-") and arg != "-":
            name = arg.lstrip("-").split("=", 1)[0].lower()
            # An inline `-opt=value` already carries its value; a separate one
            # is the NEXT token and must not be read as a path.
            skip_next = "=" not in arg and name in QT_VALUE_OPTIONS
            continue
        paths.append(arg)
    return paths


def _has_wildcard(arg: str) -> bool:
    return any(ch in arg for ch in WILDCARD_CHARS)


def expand_globs(args: Sequence[str]) -> List[str]:
    """Expand wildcard arguments (`*.tif`, `roll-?.nef`, `**/*.dng`) in place,
    each to its matches sorted case-insensitively.

    Unix shells expand these before the program starts; cmd.exe and PowerShell
    hand the pattern over untouched, so `freeccr *.tif` used to reach us as the
    literal string and be reported as not found. Doing it here makes both hosts
    behave the same. It stays a no-op under a Unix shell: a pattern the shell
    matched arrives already expanded, and one it could not match is passed
    through literally and matches nothing here either.

    An argument naming an existing path is never treated as a pattern, so a
    file called `shot[1].tif` opens as itself. A pattern matching nothing is
    passed through unchanged for plan_open to report."""
    out: List[str] = []
    for arg in args:
        if not _has_wildcard(arg) or os.path.exists(arg):
            out.append(arg)
            continue
        # recursive=True only gives `**` its meaning; other patterns are
        # unaffected, and `**` is a depth the user asked for explicitly.
        matches = glob.glob(arg, recursive=True)
        matches.sort(key=lambda pth: (pth.lower(), pth))
        out.extend(matches or [arg])
    return out


def expand_folder(folder: str,
                  is_supported: Optional[Callable[[str], bool]] = None
                  ) -> List[str]:
    """The supported images directly inside `folder`, sorted case-insensitively
    by name. NOT recursive — a folder argument contributes what Open Folder
    would see. `is_supported` defaults to the app's shared extension test
    (imported lazily so this module needs no Qt); an unreadable folder yields
    an empty list."""
    supported = is_supported
    if supported is None:
        from core.tether_watcher import is_supported as supported
    try:
        names = [e.name for e in os.scandir(folder) if e.is_file()]
    except OSError:
        return []
    names.sort(key=lambda n: (n.lower(), n))
    return [os.path.join(folder, n) for n in names if supported(n)]


def plan_open(args: Sequence[str],
              is_supported: Optional[Callable[[str], bool]] = None) -> OpenPlan:
    """Turn already-stripped path arguments into an OpenPlan.

    One argument naming a folder loads as a folder (Open Folder). Anything else
    becomes one file list: each folder is expanded to the images inside it and
    concatenated with the named files IN ARGUMENT ORDER, de-duplicated by
    normalised absolute path (first occurrence wins). A named file is kept
    whatever its extension — the Open Files dialog has an All Files filter, so
    the CLI is no stricter — while folder-expanded files are filtered.
    Arguments that contribute nothing are reported, never silently dropped.

    Wildcard arguments are expanded first (see expand_globs), so a pattern
    behaves exactly like the paths it matches — including the single-folder
    rule, which a pattern matching one folder therefore takes."""
    args = expand_globs([a for a in args if a])
    problems: List[str] = []
    if not args:
        return OpenPlan(None, [], problems)

    if len(args) == 1 and os.path.isdir(args[0]):
        return OpenPlan(args[0], [], problems)

    files: List[str] = []
    seen = set()

    def _add(path: str) -> None:
        key = os.path.normcase(os.path.normpath(os.path.abspath(path)))
        if key not in seen:
            seen.add(key)
            files.append(path)

    for arg in args:
        if os.path.isdir(arg):
            found = expand_folder(arg, is_supported)
            if not found:
                problems.append(f"{arg} — no supported images in this folder")
            for path in found:
                _add(path)
        elif os.path.isfile(arg):
            _add(arg)
        elif _has_wildcard(arg):
            problems.append(f"{arg} — no files match this pattern")
        else:
            problems.append(f"{arg} — not found")
    return OpenPlan(None, files, problems)

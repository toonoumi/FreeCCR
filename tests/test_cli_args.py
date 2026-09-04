"""
Command-line file/folder argument parsing (spec/cli-file-args.md §6).

Pure tests: no Qt, no MainWindow. The supported-extension predicate is injected
so the plan can be exercised without importing the app's Qt-bound modules.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.cli_args import (OpenPlan, expand_folder, expand_globs,  # noqa: E402
                            plan_open, strip_qt_options, wants_help,
                            wants_version)

# Stand-in for core.tether_watcher.is_supported (same rule, no Qt import).
SUPPORTED = {".nef", ".dng", ".tif", ".jpg"}


def _supported(name):
    return os.path.splitext(name)[1].lower() in SUPPORTED


def _touch(folder, *names):
    made = []
    for n in names:
        p = os.path.join(str(folder), n)
        with open(p, "wb") as f:
            f.write(b"x")
        made.append(p)
    return made


# --- option stripping ----------------------------------------------------

def test_qt_option_with_separate_value_consumes_it():
    assert strip_qt_options(["-platform", "offscreen", "a.nef"]) == ["a.nef"]


def test_inline_qt_option_value_consumes_nothing_extra():
    assert strip_qt_options(["--platform=offscreen", "a.nef"]) == ["a.nef"]


def test_stylesheet_value_is_not_offered_as_a_path():
    assert strip_qt_options(["-stylesheet", "dark.qss", "a.nef"]) == ["a.nef"]


def test_trailing_value_option_does_not_read_past_the_list():
    assert strip_qt_options(["a.nef", "-style"]) == ["a.nef"]


def test_valueless_switch_does_not_swallow_the_next_path():
    # -reverse / -widgetcount take no value; the path after them must survive.
    assert strip_qt_options(["-reverse", "a.nef"]) == ["a.nef"]
    assert strip_qt_options(["-widgetcount", "a.nef", "b.nef"]) == ["a.nef",
                                                                    "b.nef"]


def test_double_dash_passes_a_dashed_path_through():
    assert strip_qt_options(["--", "-odd-name.nef"]) == ["-odd-name.nef"]


def test_help_and_version_flags_detected_before_double_dash_only():
    assert wants_help(["--help"]) and wants_help(["-h", "a.nef"])
    assert wants_version(["-V"]) and wants_version(["--version"])
    assert not wants_help(["a.nef"])
    # After `--` they are paths, not flags.
    assert not wants_help(["--", "--help"])
    assert not wants_version(["--", "--version"])


# --- planning ------------------------------------------------------------

def test_single_folder_loads_as_a_folder(tmp_path):
    d = tmp_path / "roll"
    d.mkdir()
    _touch(d, "a.nef")
    plan = plan_open([str(d)], _supported)
    assert plan == OpenPlan(str(d), [], [])


def test_single_file_and_file_list_keep_argument_order(tmp_path):
    a, b = _touch(tmp_path, "b.nef", "a.nef")
    assert plan_open([a], _supported).files == [a]
    plan = plan_open([a, b], _supported)
    assert plan.folder is None and plan.files == [a, b]


def test_mixed_file_and_folder_expands_the_folder_in_place(tmp_path):
    d = tmp_path / "roll"
    d.mkdir()
    inner = _touch(d, "B.nef", "a.nef")
    named, = _touch(tmp_path, "loose.nef")
    plan = plan_open([named, str(d)], _supported)
    assert plan.folder is None
    # Case-insensitive name sort inside the folder: a.nef before B.nef.
    assert plan.files == [named, inner[1], inner[0]]


def test_folder_expansion_filters_unsupported_but_named_files_are_kept(tmp_path):
    d = tmp_path / "roll"
    d.mkdir()
    _touch(d, "notes.txt")
    keep, = _touch(d, "keep.nef")
    txt, = _touch(tmp_path, "readme.txt")
    plan = plan_open([str(d), txt], _supported)
    assert plan.files == [keep, txt]


def test_duplicates_collapse_across_arguments(tmp_path):
    d = tmp_path / "roll"
    d.mkdir()
    inside, = _touch(d, "a.nef")
    plan = plan_open([inside, str(d), inside], _supported)
    assert plan.files == [inside]


def test_missing_paths_are_reported_not_loaded(tmp_path):
    good, = _touch(tmp_path, "a.nef")
    missing = str(tmp_path / "gone.nef")
    plan = plan_open([good, missing], _supported)
    assert plan.files == [good]
    assert len(plan.problems) == 1 and "gone.nef" in plan.problems[0]


def test_all_missing_yields_an_empty_plan(tmp_path):
    plan = plan_open([str(tmp_path / "x.nef"), str(tmp_path / "y.nef")],
                     _supported)
    assert plan.folder is None and plan.files == []
    assert len(plan.problems) == 2


def test_empty_folder_in_a_list_is_reported(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    good, = _touch(tmp_path, "a.nef")
    plan = plan_open([good, str(d)], _supported)
    assert plan.files == [good]
    assert len(plan.problems) == 1 and "no supported images" in plan.problems[0]


def test_no_arguments_is_an_empty_plan():
    assert plan_open([], _supported) == OpenPlan(None, [], [])


def test_expansion_is_not_recursive(tmp_path):
    d = tmp_path / "roll"
    (d / "sub").mkdir(parents=True)
    _touch(d / "sub", "deep.nef")
    top, = _touch(d, "top.nef")
    assert expand_folder(str(d), _supported) == [top]


def test_unreadable_folder_expands_to_nothing(tmp_path):
    assert expand_folder(str(tmp_path / "does-not-exist"), _supported) == []


# --- wildcard expansion --------------------------------------------------
# Unix shells expand these before argv reaches us; cmd.exe and PowerShell do
# not, so FreeCCR expands them itself and both hosts behave alike.

def test_star_pattern_expands_to_its_matches_sorted(tmp_path):
    _touch(tmp_path, "B.nef", "a.nef", "c.jpg")
    got = expand_globs([str(tmp_path / "*.nef")])
    assert got == [str(tmp_path / "a.nef"), str(tmp_path / "B.nef")]


def test_question_mark_pattern_matches_one_character(tmp_path):
    _touch(tmp_path, "roll-1.nef", "roll-12.nef")
    got = expand_globs([str(tmp_path / "roll-?.nef")])
    assert got == [str(tmp_path / "roll-1.nef")]


def test_pattern_plans_as_a_file_list(tmp_path):
    a, b = _touch(tmp_path, "a.nef", "b.nef")
    plan = plan_open([str(tmp_path / "*.nef")], _supported)
    assert plan.folder is None and plan.files == [a, b]
    assert plan.problems == []


def test_pattern_and_named_file_keep_argument_order_and_dedupe(tmp_path):
    a, b = _touch(tmp_path, "a.nef", "b.nef")
    plan = plan_open([b, str(tmp_path / "*.nef")], _supported)
    # The named file comes first; the pattern re-offers it and it collapses.
    assert plan.files == [b, a]


def test_unmatched_pattern_is_reported_as_a_pattern(tmp_path):
    plan = plan_open([str(tmp_path / "*.nef")], _supported)
    assert plan.files == []
    assert len(plan.problems) == 1
    assert "no files match this pattern" in plan.problems[0]


def test_a_real_file_named_like_a_pattern_wins(tmp_path):
    # glob would read `shot[1].nef` as a character class; the file exists, so
    # it must open as itself.
    real, = _touch(tmp_path, "shot[1].nef")
    assert expand_globs([real]) == [real]
    assert plan_open([real], _supported).files == [real]


def test_pattern_matching_one_folder_takes_the_single_folder_rule(tmp_path):
    d = tmp_path / "roll-1"
    d.mkdir()
    _touch(d, "a.nef")
    plan = plan_open([str(tmp_path / "roll-*")], _supported)
    assert plan == OpenPlan(str(d), [], [])


def test_pattern_matching_folders_expands_each_in_a_list(tmp_path):
    for name in ("roll-1", "roll-2"):
        (tmp_path / name).mkdir()
    one, = _touch(tmp_path / "roll-1", "a.nef")
    two, = _touch(tmp_path / "roll-2", "b.nef")
    plan = plan_open([str(tmp_path / "roll-*")], _supported)
    assert plan.folder is None and plan.files == [one, two]


def test_double_star_recurses_only_when_asked(tmp_path):
    (tmp_path / "sub").mkdir()
    deep, = _touch(tmp_path / "sub", "deep.nef")
    top, = _touch(tmp_path, "top.nef")
    assert expand_globs([str(tmp_path / "**" / "*.nef")]) == [deep, top]
    # A single star stays at one level.
    assert expand_globs([str(tmp_path / "*.nef")]) == [top]


def test_non_wildcard_arguments_pass_through_untouched():
    args = ["/a/b.nef", "relative.nef", "-"]
    assert expand_globs(args) == args


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

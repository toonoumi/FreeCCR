#!/usr/bin/env python3
"""White Balance restored to the panel, Channel Balance demoted to a collapsible
(spec/white-balance-restore.md).

Two things are checked here. First the PANEL: Temperature/Tint are back above
Brightness, the Balance trio moved into a collapsed section under Channel Levels
and Master Gain, and the positional ADJUSTMENT_KEYS zip survived the move — a
mis-zip routes every slider below the change to the wrong key without erroring
anywhere. Second the SOLVE: the WB picker and AWB drive Temperature/Tint again,
still by closed loop on the real render, so a picked spot renders neutral under
every stage combination rather than only the ones an analytic inverse models.
"""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2  # noqa: E402
from PySide6.QtWidgets import (QApplication, QWidget,  # noqa: E402
                               QVBoxLayout, QScrollArea)

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from widgets.sliders_panel import (SlidersPanel, SYNC_GROUPS,  # noqa: E402
                                   CollapsibleSection)

WB_KEYS = ("temperature", "tint")
BALANCE_KEYS = ("balance_r", "balance_g", "balance_b")


class _ImagePreviewStub:
    def update_preview(self, idx):
        pass


class _Host(QWidget):
    def __init__(self):
        super().__init__()
        self.image_preview = _ImagePreviewStub()
        self.mid = QWidget(self)


_HOSTS = []


@pytest.fixture
def panel(tmp_path):
    host = _Host()
    _HOSTS.append(host)                 # keep the parent alive for the test
    layout = QVBoxLayout(host.mid)
    p = SlidersPanel(host.mid)
    layout.addWidget(p)

    path = str(tmp_path / "scan.png")
    base = np.full((40, 60, 3), 20000, np.uint16)
    cv2.imwrite(path, cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
    img = CCRImage(path)
    saved = (ccr_backend.images, ccr_backend.file_paths)
    ccr_backend.images, ccr_backend.file_paths = [img], [img.file_path]
    p.current_idx = 0
    yield p
    ccr_backend.images, ccr_backend.file_paths = saved


def _scroll_layout(panel):
    """The panel's scrolling content layout — what the rows are placed in."""
    return panel.findChild(QScrollArea).widget().layout()


def _index_of(parent_layout, item):
    """Position of a child row (a nested layout) or widget, or -1."""
    for i in range(parent_layout.count()):
        it = parent_layout.itemAt(i)
        if it.layout() is item or it.widget() is item:
            return i
    return -1


# --- panel wiring -----------------------------------------------------------

def test_temperature_and_tint_are_back():
    assert "temperature" in SlidersPanel.ADJUSTMENT_KEYS
    assert "tint" in SlidersPanel.ADJUSTMENT_KEYS
    assert list(SlidersPanel.ADJUSTMENT_KEYS[:2]) == list(WB_KEYS)


def test_every_key_has_its_own_slider(panel):
    """ADJUSTMENT_KEYS is zipped positionally against create_slider() call
    order, and this change reordered both."""
    assert len(panel.sliders) == len(panel.adjustment_keys)
    assert len(set(panel.adjustment_keys)) == len(panel.adjustment_keys)


@pytest.mark.parametrize("key,other", [
    ("temperature", "tint"),            # the restored pair
    ("brightness", "gamma"),            # below them
    ("ch_master_gain", "ch_r_shift"),   # Channel Levels, below the tone block
    ("balance_g", "balance_b"),         # the moved trio
    ("band_feather", "saturation"),     # the very last key
])
def test_each_slider_writes_its_own_key(panel, key, other):
    """A mis-zip shows up as one slider driving its neighbour's key, with no
    error anywhere — so check across the whole reordered list."""
    img = ccr_backend.get_image_by_index(0)
    before_other = img.adjustment_settings.get(other, 0)
    panel.sliders[panel.adjustment_keys.index(key)].setValue(37)
    assert img.adjustment_settings[key] == 37
    assert img.adjustment_settings.get(other, 0) == before_other


def test_wb_sliders_sit_directly_above_brightness(panel):
    sl = _scroll_layout(panel)
    temp = _index_of(sl, panel.temperature_slider_layout)
    tint = _index_of(sl, panel.tint_slider_layout)
    bright = _index_of(sl, panel.brightness_slider_layout)
    assert temp >= 0 and (tint, bright) == (temp + 1, temp + 2)


def test_balance_section_sits_under_channel_levels_and_master_gain(panel):
    sl = _scroll_layout(panel)
    levels = _index_of(sl, panel.od_section)
    gain = _index_of(sl, panel.master_gain_slider_layout)
    balance = _index_of(sl, panel.balance_section)
    assert levels >= 0 and (gain, balance) == (levels + 1, levels + 2)
    # ...and above the White Balance sliders, which is pipeline order.
    assert balance < _index_of(sl, panel.temperature_slider_layout)


def test_balance_section_is_collapsed_by_default(panel):
    assert isinstance(panel.balance_section, CollapsibleSection)
    assert not panel.balance_section._content.isVisible()
    panel.balance_section._toggle_btn.click()
    assert panel.balance_section._content.isVisibleTo(panel.balance_section)


def test_balance_sliders_live_inside_the_section(panel):
    """Not in the scrolling column: the section owns them, which is what makes
    the default-collapsed state hide them."""
    sl = _scroll_layout(panel)
    for layout in (panel.balance_r_slider_layout, panel.balance_g_slider_layout,
                   panel.balance_b_slider_layout):
        assert _index_of(sl, layout) == -1
        assert _index_of(panel.balance_section._content.layout(), layout) >= 0


def test_sliders_carry_their_gradients(panel):
    from PySide6.QtGui import QColor
    from ui import theme
    pairs = [("temperature", theme.TEMP_GRADIENT), ("tint", theme.TINT_GRADIENT),
             ("balance_r", theme.BALANCE_R_GRADIENT),
             ("balance_g", theme.BALANCE_G_GRADIENT),
             ("balance_b", theme.BALANCE_B_GRADIENT)]
    for key, grad in pairs:
        s = panel.sliders[panel.adjustment_keys.index(key)]
        assert (s._lo, s._hi) == (QColor(grad[0]), QColor(grad[1])), key


# --- sync groups ------------------------------------------------------------

def test_wb_group_carries_temperature_and_tint():
    groups = {gid: keys for gid, _label, keys in SYNC_GROUPS}
    assert tuple(groups["wb"]) == WB_KEYS
    assert tuple(groups["balance"]) == BALANCE_KEYS


def test_groups_still_partition_adjustment_keys():
    grouped = [k for _gid, _label, keys in SYNC_GROUPS for k in keys]
    grouped = [k for k in grouped if k != "cineon_log"]   # a flag, not a slider
    assert sorted(grouped) == sorted(SlidersPanel.ADJUSTMENT_KEYS)
    assert len(grouped) == len(set(grouped))


# --- the closed-loop neutral solve ------------------------------------------
#
# Same contract, and same reason for measuring rather than inverting: clicking a
# neutral spot makes THAT SPOT render neutral, under stage combinations no
# analytic inverse of any one stage can account for. These assert on rendered
# output only.

def _cast_scene(tmp_path, name="scene.png"):
    """A converted, windowed base: a tonal ramp under a yellow cast so Auto Gain
    has real highlights to normalise against, plus a mid-grey patch to pick."""
    from core.ccr_processor import encode_window
    path = str(tmp_path / name)
    cv2.imwrite(path, np.full((60, 90, 3), 20000, np.uint16))
    img = CCRImage(path)
    img.converted = True
    img._ws_windowed = True
    h, w = 300, 450
    ramp = np.linspace(0.02, 1.0, w, dtype=np.float32)[None, :, None].repeat(h, 0)
    cast = np.array([1.0, 0.88, 0.55], np.float32)
    disp = np.clip(ramp * cast, 0, 1.2)
    disp[140:160, 200:220] = np.array([0.35, 0.35, 0.35], np.float32) * cast
    img.resized_raw = encode_window(disp.copy())
    return img


def _patch(img):
    """The sampled AREA the eyedropper hands the solver (rad=3 -> 7x7)."""
    return img.resized_raw[147:154, 207:214].copy()


def _spot(img, settings):
    """Mean RGB of the grey patch after a FULL-FRAME render."""
    out = img.apply_adjustments(img.resized_raw.copy(), settings=settings)
    return out[140:160, 200:220].reshape(-1, 3).mean(axis=0)


def _spread(rgb):
    return float(rgb.max() - rgb.min()) / max(float(rgb.max()), 1.0)


PRESETS = [
    ("nothing set", {}),
    ("channel levels", {"ch_r_shift": 15, "ch_g_gain": 12, "ch_b_blackpoint": -8}),
    ("master gain", {"ch_master_gain": 35}),
    ("input gain", {"ch_input_gain": -20}),
    ("cineon log", {"cineon_log": True}),
    ("tone + saturation", {"gamma": 40, "contrast": 30, "saturation": 25}),
    ("per-channel curves", {"curves": {"r": [[0, 0], [128, 150], [255, 255]]}}),
    ("channel balance set", {"balance_g": 20, "balance_b": 35}),
    ("everything", {"ch_r_shift": 10, "ch_master_gain": 25, "gamma": 30,
                    "contrast": 20, "saturation": 20, "cineon_log": True}),
]


@pytest.mark.parametrize("label,preset", PRESETS, ids=[p[0] for p in PRESETS])
def test_picked_spot_renders_neutral(tmp_path, label, preset):
    """The whole point: after the pick the sampled spot is grey IN THE RENDER —
    including on top of a Channel Balance the user already set, which is the
    combination the two controls now have to survive together."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = dict(preset)
    temp, tint = img.solve_neutral_wb(_patch(img))
    before = _spot(img, dict(preset))
    after = _spot(img, dict(preset, temperature=temp, tint=tint))
    assert _spread(before) > 0.2, "test scene should start visibly cast"
    assert _spread(after) < 0.02, f"{label}: {after} spread {_spread(after):.3f}"


def test_a_warm_cast_is_cooled(tmp_path):
    """Direction check: the scene is red-heavy/blue-poor, so temperature goes
    negative (R *= 1+s, B *= 1-s)."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    assert img.solve_neutral_wb(_patch(img))[0] < 0


def test_more_passes_never_get_worse(tmp_path):
    """The loop keeps the best result it has seen, so adding passes cannot
    regress. Unlike the Balance solve — where red is pinned and each channel
    chases the mean of the other two, so it takes several passes to settle —
    the two white balance objectives are independent and one pass already
    lands it, which is what the first entry asserts."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    spreads = []
    for n in (1, 2, 3, 4):
        temp, tint = img.solve_neutral_wb(_patch(img), passes=n)
        spreads.append(_spread(_spot(img, {"temperature": temp, "tint": tint})))
    assert all(b <= a + 1e-6 for a, b in zip(spreads, spreads[1:])), spreads
    assert spreads[0] < 0.02, spreads


def test_solve_replaces_the_existing_white_balance(tmp_path):
    """Both knobs are solved over the full range from a fixed (0, 0) start, so a
    pick replaces the white balance instead of accumulating onto it — picking
    twice lands in the same place."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    first = img.solve_neutral_wb(_patch(img))
    img.adjustment_settings = {"temperature": first[0], "tint": first[1]}
    assert img.solve_neutral_wb(_patch(img)) == first


def test_solve_uses_the_area_not_one_pixel(tmp_path):
    """Sampling by AREA: a single blown pixel inside the patch must not steer
    the result, because the loop drives the patch MEAN."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    clean = img.solve_neutral_wb(_patch(img))
    noisy = _patch(img)
    noisy[3, 3] = 0                       # one dead pixel in a 7x7 area
    assert img.solve_neutral_wb(noisy) == pytest.approx(clean, abs=3)


def test_black_and_white_profile_leaves_the_sliders_alone(tmp_path):
    """A B&W render collapses to luminance, so no white balance can change the
    output spread. The solver must leave the sliders at 0 rather than driving
    them to an endpoint."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    img.color_profile = "bw"
    assert img.solve_neutral_wb(_patch(img)) == (0, 0)


def test_solve_survives_an_empty_patch(tmp_path):
    img = _cast_scene(tmp_path)
    assert img.solve_neutral_wb(np.zeros((0, 0, 3), np.uint16)) == (0, 0)


def test_auto_gain_is_measured_from_the_base_not_the_patch(tmp_path):
    """The patch carries no highlights, so measuring Auto Gain from it would give
    a wildly different gain than the render uses. Passing the base-measured value
    is what keeps the loop's render equal to the real one."""
    from core.ccr_processor import compute_auto_gain_offset
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    ag_base = compute_auto_gain_offset(img.resized_raw, True)
    ag_patch = compute_auto_gain_offset(_patch(img), True)
    assert abs(ag_base - ag_patch) > 1.0        # they really do differ
    temp, tint = img.solve_neutral_wb(_patch(img))
    assert _spread(_spot(img, {"temperature": temp, "tint": tint})) < 0.02


# --- AWB ---------------------------------------------------------------------

@pytest.mark.parametrize("algorithm",
                         ["gray_world", "white_patch", "shades_of_gray", "gray_edge"])
def test_awb_corrects_the_frame_under_every_algorithm(tmp_path, algorithm):
    """AWB is the same loop as a whole-frame regression: it renders a downscaled
    copy every iteration and drives the estimator's reading of THAT to grey."""
    from core.awb import compute_awb_wb
    img = _cast_scene(tmp_path, name=f"awb_{algorithm}.png")
    img.adjustment_settings = {}
    res = compute_awb_wb(img, algorithm=algorithm)
    assert res is not None
    temp, tint = res
    before = _spot(img, {})
    after = _spot(img, {"temperature": temp, "tint": tint})
    assert _spread(after) < _spread(before)
    assert all(-100 <= v <= 100 for v in res)


def test_awb_without_a_base_is_none(tmp_path):
    from core.awb import compute_awb_wb
    img = _cast_scene(tmp_path)
    img.resized_raw = None
    assert compute_awb_wb(img, algorithm="gray_world") is None


def test_awb_hook_writes_temperature_and_tint(tmp_path):
    """The post-conversion Auto WB hook lands on the white balance sliders."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    saved = (ccr_backend.auto_awb, ccr_backend.awb_algorithm)
    try:
        ccr_backend.auto_awb = True
        ccr_backend.awb_algorithm = "gray_world"
        ccr_backend.maybe_auto_awb(img)
        assert any(img.adjustment_settings.get(k, 0) for k in WB_KEYS)
        assert not any(img.adjustment_settings.get(k, 0) for k in BALANCE_KEYS)
    finally:
        ccr_backend.auto_awb, ccr_backend.awb_algorithm = saved


# --- the apply path ----------------------------------------------------------

def test_balance_hotkeys_are_off_by_default():
    """Opt-in: the sliders they move live in a collapsed section, and they claim
    six single letters."""
    from core.ccr_backend import CCRBackend
    assert CCRBackend().balance_hotkeys is False


class _FakeShortcut:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, value):
        self.enabled = bool(value)


class _FakeSettings:
    def __init__(self):
        self.stored = {}

    def setValue(self, key, value):
        self.stored[key] = value


class _FakePanel:
    def __init__(self):
        self.hints = []

    def set_temporary_hint(self, text, duration=0):
        self.hints.append(text)


class _FakeWindow:
    """Enough of MainWindow to exercise the two hotkey methods unbound — a real
    window would read and write the user's actual QSettings."""
    def __init__(self):
        from ui.main_window import MainWindow
        self._balance_shortcuts = [_FakeShortcut() for _ in range(6)]
        self._settings = _FakeSettings()
        self.sliders_panel = _FakePanel()
        # The real method, bound to the fake — on_balance_hotkeys_toggled calls
        # it through self, so a stub here would hide the wiring under test.
        self._apply_balance_hotkey_state = (
            lambda: MainWindow._apply_balance_hotkey_state(self))


@pytest.fixture
def hotkey_window():
    from ui.main_window import MainWindow
    saved = ccr_backend.balance_hotkeys
    yield MainWindow, _FakeWindow()
    ccr_backend.balance_hotkeys = saved


@pytest.mark.parametrize("enabled", [True, False])
def test_shortcuts_follow_the_setting(hotkey_window, enabled):
    """Gated by setEnabled, not by a check in the handler: a disabled QShortcut
    does not consume its key, so U/I/O/J/K/L stay free while the setting is
    off."""
    MainWindow, win = hotkey_window
    ccr_backend.balance_hotkeys = enabled
    MainWindow._apply_balance_hotkey_state(win)
    assert all(sc.enabled is enabled for sc in win._balance_shortcuts)


def test_toggle_sets_persists_and_applies(hotkey_window):
    MainWindow, win = hotkey_window
    ccr_backend.balance_hotkeys = False
    MainWindow._apply_balance_hotkey_state(win)

    MainWindow.on_balance_hotkeys_toggled(win, True)
    assert ccr_backend.balance_hotkeys is True
    assert win._settings.stored["adjust/balance_hotkeys"] is True
    assert all(sc.enabled for sc in win._balance_shortcuts)

    MainWindow.on_balance_hotkeys_toggled(win, False)
    assert ccr_backend.balance_hotkeys is False
    assert win._settings.stored["adjust/balance_hotkeys"] is False
    assert not any(sc.enabled for sc in win._balance_shortcuts)


def test_nudge_itself_is_unchanged(panel):
    """The setting gates the SHORTCUTS, not the panel method they call — so the
    Balance control keeps working from the sliders regardless."""
    panel.nudge_balance("g", +1)
    img = ccr_backend.get_image_by_index(0)
    assert img.adjustment_settings["balance_g"] == SlidersPanel.BALANCE_HOTKEY_STEP


def test_on_wb_sampled_writes_both_sliders(panel):
    panel.on_wb_sampled(-30, 12)
    img = ccr_backend.get_image_by_index(0)
    assert img.adjustment_settings["temperature"] == -30
    assert img.adjustment_settings["tint"] == 12
    assert panel.sliders[panel.adjustment_keys.index("temperature")].value() == -30
    assert panel.sliders[panel.adjustment_keys.index("tint")].value() == 12


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

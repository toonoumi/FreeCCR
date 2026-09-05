#!/usr/bin/env python3
"""Channel Balance panel wiring and nudge hotkeys (spec/channel-balance.md).

The three sliders live in their own collapsible under Channel Levels, so the
positional ADJUSTMENT_KEYS zip is the thing most likely to break silently — a
mis-zip would route every slider below them to the wrong key without erroring.
The hotkeys (U/I/O raise R/G/B, J/K/L lower) are checked through the panel
method the shortcuts call.
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
                               QVBoxLayout)

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from widgets.sliders_panel import SlidersPanel, SYNC_GROUPS  # noqa: E402

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
    _HOSTS.append(host)
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


# --- panel wiring -----------------------------------------------------------

def test_balance_keys_follow_the_channel_levels_keys(panel):
    """ADJUSTMENT_KEYS is zipped positionally against create_slider() call
    order. The Balance sliders live in their own collapsible under Channel
    Levels, and are created right after the Channel Levels sliders — if that
    zip is off, sliders silently write the wrong keys."""
    keys = panel.adjustment_keys
    assert len(panel.sliders) == len(keys)
    last_channel = max(keys.index(k) for k in keys if k.startswith("ch_"))
    assert keys[last_channel + 1:last_channel + 4] == list(BALANCE_KEYS)


def test_keys_around_balance_still_line_up(panel):
    """Spot-check keys either side of the Balance block: a mis-zip would show up
    as Brightness driving Gamma's slider, with no error anywhere."""
    img = ccr_backend.get_image_by_index(0)
    for key, other in (("brightness", "gamma"), ("band_feather", "saturation")):
        panel.sliders[panel.adjustment_keys.index(key)].setValue(33)
        assert img.adjustment_settings[key] == 33
        assert img.adjustment_settings.get(other, 0) == 0


def test_balance_sliders_have_full_range_and_zero_default(panel):
    for key in BALANCE_KEYS:
        s = panel.sliders[panel.adjustment_keys.index(key)]
        assert (s.minimum(), s.maximum(), s.value()) == (-100, 100, 0)


def test_sliders_carry_their_channel_gradient(panel):
    """Each groove runs from the channel's complement to the channel colour, so
    the direction of a move reads without the label."""
    from PySide6.QtGui import QColor
    from ui import theme
    for key, grad in zip(BALANCE_KEYS, (theme.BALANCE_R_GRADIENT,
                                        theme.BALANCE_G_GRADIENT,
                                        theme.BALANCE_B_GRADIENT)):
        s = panel.sliders[panel.adjustment_keys.index(key)]
        assert (s._lo, s._hi) == (QColor(grad[0]), QColor(grad[1]))
    # ...and the high end is the channel's own colour, matching the Channel
    # Levels group headings.
    assert theme.BALANCE_R_GRADIENT[1] == theme.CH_R
    assert theme.BALANCE_G_GRADIENT[1] == theme.CH_G
    assert theme.BALANCE_B_GRADIENT[1] == theme.CH_B


def test_balance_has_its_own_sync_group():
    """Balance syncs independently of White Balance: they are different stages,
    and "wb" carries Temperature/Tint again."""
    groups = {gid: keys for gid, _label, keys in SYNC_GROUPS}
    assert tuple(groups["balance"]) == BALANCE_KEYS
    assert tuple(groups["wb"]) == ("temperature", "tint")


def test_sync_groups_still_partition_adjustment_keys():
    """The invariant SYNC_GROUPS documents: the adjustment-key groups partition
    ADJUSTMENT_KEYS exactly. Adding a key without a group would leave it
    unsyncable and unnoticed."""
    grouped = [k for _gid, _label, keys in SYNC_GROUPS for k in keys]
    grouped = [k for k in grouped if k != "cineon_log"]   # a flag, not a slider
    assert sorted(grouped) == sorted(SlidersPanel.ADJUSTMENT_KEYS)


# --- nudge hotkeys ----------------------------------------------------------

@pytest.mark.parametrize("channel,direction", [
    ("r", +1), ("g", +1), ("b", +1), ("r", -1), ("g", -1), ("b", -1),
])
def test_nudge_moves_only_its_own_channel(panel, channel, direction):
    panel.nudge_balance(channel, direction)
    step = SlidersPanel.BALANCE_HOTKEY_STEP
    for key in BALANCE_KEYS:
        val = panel.sliders[panel.adjustment_keys.index(key)].value()
        assert val == (direction * step if key == f"balance_{channel}" else 0)


def test_nudge_writes_through_to_the_image(panel):
    panel.nudge_balance("g", +1)
    img = ccr_backend.get_image_by_index(0)
    assert img.adjustment_settings["balance_g"] == SlidersPanel.BALANCE_HOTKEY_STEP


def test_nudge_accumulates(panel):
    for _ in range(3):
        panel.nudge_balance("b", +1)
    idx = panel.adjustment_keys.index("balance_b")
    assert panel.sliders[idx].value() == 3 * SlidersPanel.BALANCE_HOTKEY_STEP


def test_nudge_clamps_at_the_slider_ends(panel):
    idx = panel.adjustment_keys.index("balance_r")
    panel.sliders[idx].setValue(98)
    panel.nudge_balance("r", +1)
    assert panel.sliders[idx].value() == 100
    panel.nudge_balance("r", +1)          # already pegged — must not error
    assert panel.sliders[idx].value() == 100
    panel.sliders[idx].setValue(-98)
    panel.nudge_balance("r", -1)
    assert panel.sliders[idx].value() == -100


def test_nudge_is_a_noop_without_an_image(panel):
    panel.current_idx = None
    panel.nudge_balance("r", +1)
    assert panel.sliders[panel.adjustment_keys.index("balance_r")].value() == 0


def test_nudge_is_a_noop_while_sliders_are_disabled(panel):
    """Mirrors the other shortcut handlers: an unconverted image has its sliders
    disabled, and a hotkey must not edit through that."""
    for s in panel.sliders:
        s.setEnabled(False)
    panel.nudge_balance("r", +1)
    assert panel.sliders[panel.adjustment_keys.index("balance_r")].value() == 0


def test_nudge_run_collapses_into_one_undo_step(panel):
    """Holding a key auto-repeats; the burst timer must fold the whole run into
    a single undo state rather than twenty."""
    img = ccr_backend.get_image_by_index(0)
    before = len(img.undo_stack)
    for _ in range(6):
        panel.nudge_balance("g", +1)
    assert len(img.undo_stack) == before + 1


def test_unknown_channel_is_ignored(panel):
    panel.nudge_balance("x", +1)          # must not raise
    assert all(panel.sliders[panel.adjustment_keys.index(k)].value() == 0
               for k in BALANCE_KEYS)


# --- WB picker / AWB: the closed-loop neutral solve ---------------------------
#
# The contract: clicking a neutral spot makes THAT SPOT render neutral. This is
# solved by MEASURING the real render, not by inverting the Balance curve --
# every attempt to model the intervening stages (Channel Levels, the hidden Auto
# Gain offset, gamma, curves, Cineon) was wrong in some configuration. These
# tests therefore assert on rendered output only.

def _cast_scene(tmp_path, name="scene.png", cast=(1.0, 0.88, 0.55)):
    """A converted, windowed base: a tonal ramp under a yellow cast so Auto Gain
    has real highlights to normalise against, plus a mid-grey patch to pick.

    The default cast is deliberately extreme, to prove the solve's reach. It is
    extreme in DENSITY terms, so a Cineon decode expands it toward the end of
    what the Balance node can reach - those tests pass a milder one."""
    from core.ccr_processor import encode_window
    path = str(tmp_path / name)
    cv2.imwrite(path, np.full((60, 90, 3), 20000, np.uint16))
    img = CCRImage(path)
    img.converted = True
    img._ws_windowed = True
    h, w = 300, 450
    ramp = np.linspace(0.02, 1.0, w, dtype=np.float32)[None, :, None].repeat(h, 0)
    cast = np.asarray(cast, np.float32)
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
    ("tone + saturation", {"gamma": 40, "contrast": 30, "saturation": 25}),
    ("per-channel curves", {"curves": {"r": [[0, 0], [128, 150], [255, 255]]}}),
    ("everything", {"ch_r_shift": 10, "ch_master_gain": 25, "gamma": 30,
                    "contrast": 20, "saturation": 20}),
]

# Cineon gets its own scene. The decode sits between Channel Levels and
# Balance, and it turns a cast from a log OFFSET into a linear RATIO -- the
# extreme cast above then needs more than the node can deliver. See
# test_the_decoded_base_can_outrun_the_node.
CINEON_PRESETS = [
    ("cineon log", {"cineon_log": True}),
    ("cineon + everything", {"cineon_log": True, "ch_master_gain": 25,
                             "gamma": 30, "contrast": 20, "saturation": 20}),
]


@pytest.mark.parametrize("label,preset", PRESETS, ids=[p[0] for p in PRESETS])
def test_picked_spot_renders_neutral(tmp_path, label, preset):
    """The whole point: after the pick the sampled spot is grey IN THE RENDER,
    under every stage combination -- including per-channel Curves and the Cineon
    transform, which no analytic inverse of the Balance curve could account for."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = dict(preset)
    r, g, b = img.solve_neutral_balance(_patch(img))
    before = _spot(img, dict(preset))
    after = _spot(img, dict(preset, balance_r=r, balance_g=g, balance_b=b))
    assert _spread(before) > 0.2, "test scene should start visibly cast"
    assert _spread(after) < 0.02, f"{label}: {after} spread {_spread(after):.3f}"


@pytest.mark.parametrize("label,preset", CINEON_PRESETS,
                         ids=[p[0] for p in CINEON_PRESETS])
def test_picked_spot_renders_neutral_on_a_decoded_base(tmp_path, label, preset):
    """The pick still lands on grey when Cineon has decoded out of log ahead of
    the Balance node — the position that lets the node grade the tone it names."""
    img = _cast_scene(tmp_path, cast=(1.0, 0.95, 0.86))
    img.adjustment_settings = dict(preset)
    r, g, b = img.solve_neutral_balance(_patch(img))
    before = _spot(img, dict(preset))
    after = _spot(img, dict(preset, balance_r=r, balance_g=g, balance_b=b))
    assert _spread(before) > 0.1, "test scene should start visibly cast"
    assert _spread(after) < 0.02, f"{label}: {after} spread {_spread(after):.3f}"


def test_the_decoded_base_can_outrun_the_node(tmp_path):
    """The flip side, pinned so it is a known property: the decode turns a cast
    from a log OFFSET into a linear RATIO, and on the extreme scene (with a gain
    and the tone chain piled on) blue pegs at +100 with ~3% spread left. The
    solve is doing the right thing — that cast belongs in Channel Levels BEFORE
    the decode, where a per-channel shift is a density offset."""
    img = _cast_scene(tmp_path)
    preset = {"cineon_log": True, "ch_r_shift": 10, "ch_master_gain": 25,
              "gamma": 30, "contrast": 20, "saturation": 20}
    img.adjustment_settings = dict(preset)
    r, g, b = img.solve_neutral_balance(_patch(img))
    assert b == 100, "expected the blue node to peg, not to converge"
    after = _spot(img, dict(preset, balance_r=r, balance_g=g, balance_b=b))
    assert _spread(after) < 0.05          # ...and it still gets most of the way
    assert _spread(after) < _spread(_spot(img, dict(preset))) / 4


def test_red_slider_is_never_moved(tmp_path):
    """Red is the anchor: the solve reports 0 for it in every configuration."""
    img = _cast_scene(tmp_path)
    for _label, preset in PRESETS:
        img.adjustment_settings = dict(preset)
        assert img.solve_neutral_balance(_patch(img))[0] == 0


def test_first_pass_drives_blue_to_the_mean_of_red_and_green(tmp_path):
    """One pass of the loop is the described step: with R=134 and G=120, blue
    lands on 127. Iterating that (red pinned) is what converges to grey."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    start = _spot(img, {})
    target = (start[0] + start[1]) / 2.0
    r, g, b = img.solve_neutral_balance(_patch(img), passes=1)
    after = _spot(img, {"balance_r": r, "balance_g": g, "balance_b": b})
    assert after[2] == pytest.approx(target, rel=0.02)


def test_more_passes_never_get_worse(tmp_path):
    """The loop keeps the best result it has seen, so adding passes cannot
    regress -- it must converge monotonically toward grey."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    spreads = []
    for n in (1, 2, 3, 4):
        r, g, b = img.solve_neutral_balance(_patch(img), passes=n)
        spreads.append(_spread(_spot(img, {"balance_r": r, "balance_g": g,
                                           "balance_b": b})))
    assert all(b <= a + 1e-6 for a, b in zip(spreads, spreads[1:])), spreads
    assert spreads[-1] < spreads[0]


def test_solve_uses_the_area_not_one_pixel(tmp_path):
    """Sampling by AREA: a single blown pixel inside the patch must not steer
    the result, because the loop drives the patch MEAN."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    clean = img.solve_neutral_balance(_patch(img))
    noisy = _patch(img)
    noisy[3, 3] = 0                       # one dead pixel in a 7x7 area
    assert img.solve_neutral_balance(noisy) == pytest.approx(clean, abs=3)


def test_black_and_white_profile_leaves_the_sliders_alone(tmp_path):
    """A B&W render collapses to luminance, so no Balance value can change the
    output spread. The solver must detect the flat response and leave the
    sliders at 0 rather than driving them to an endpoint."""
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {}
    img.color_profile = "bw"
    assert img.solve_neutral_balance(_patch(img)) == (0, 0, 0)


def test_solve_survives_an_empty_patch(tmp_path):
    img = _cast_scene(tmp_path)
    assert img.solve_neutral_balance(np.zeros((0, 0, 3), np.uint16)) == (0, 0, 0)


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
    r, g, b = img.solve_neutral_balance(_patch(img))
    assert _spread(_spot(img, {"balance_r": r, "balance_g": g, "balance_b": b})) < 0.02


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

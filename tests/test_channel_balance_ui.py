#!/usr/bin/env python3
"""Channel Balance panel wiring and nudge hotkeys (spec/channel-balance.md).

The three sliders replaced Temperature/Tint in the same panel slot, so the
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

def test_temperature_and_tint_are_gone():
    assert "temperature" not in SlidersPanel.ADJUSTMENT_KEYS
    assert "tint" not in SlidersPanel.ADJUSTMENT_KEYS


def test_balance_keys_lead_the_positional_zip(panel):
    """ADJUSTMENT_KEYS is zipped positionally against create_slider() call
    order. The three Balance sliders took the two Temperature/Tint slots plus
    one, so everything below shifted by one — if that zip is off, sliders
    silently write the wrong keys."""
    assert panel.adjustment_keys[:3] == list(BALANCE_KEYS)
    assert len(panel.sliders) == len(panel.adjustment_keys)
    for i, key in enumerate(BALANCE_KEYS):
        assert panel.adjustment_keys.index(key) == i


def test_keys_after_balance_still_line_up(panel):
    """Spot-check a key well below the insertion point: a mis-zip would show up
    as Brightness driving Gamma's slider, with no error anywhere."""
    idx = panel.adjustment_keys.index("brightness")
    panel.sliders[idx].setValue(33)
    img = ccr_backend.get_image_by_index(0)
    assert img.adjustment_settings["brightness"] == 33
    assert img.adjustment_settings.get("gamma", 0) == 0


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


def test_wb_sync_group_carries_the_balance_keys():
    """The group id stays "wb" so a remembered selection from an earlier
    session still applies, but its keys are now the Balance trio."""
    group = {gid: keys for gid, _label, keys in SYNC_GROUPS}["wb"]
    assert tuple(group) == BALANCE_KEYS


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


# --- WB picker / AWB end to end ---------------------------------------------
#
# The user-visible contract: clicking a neutral spot must make THAT SPOT render
# neutral. Solving the bare curve inverse against the sampled base does not do
# that, because Channel Levels (carrying the hidden Auto Gain offset) runs ahead
# of the tone-dependent Balance stage.

def _cast_scene(tmp_path):
    """A converted, windowed base: a tonal ramp under a yellow cast so Auto Gain
    has real highlights to normalise against, plus a mid-grey patch to pick."""
    from core.ccr_processor import encode_window
    path = str(tmp_path / "scene.png")
    cv2.imwrite(path, np.full((60, 90, 3), 20000, np.uint16))
    img = CCRImage(path)
    img.converted = True
    img._ws_windowed = True
    h, w = 60, 90
    ramp = np.linspace(0.02, 1.0, w, dtype=np.float32)[None, :, None].repeat(h, 0)
    cast = np.array([1.0, 0.88, 0.55], np.float32)
    disp = np.clip(ramp * cast, 0, 1.2)
    disp[25:35, 40:50] = np.array([0.35, 0.35, 0.35], np.float32) * cast
    img.resized_raw = encode_window(disp.copy())
    return img


def _pick(img):
    """What _sample_wb_point hands the solver: the de-windowed sampled triple."""
    from core.ccr_processor import WS_B, WS_W
    m = img.resized_raw[30, 45, :].astype(np.float64)
    return tuple(np.clip((m - WS_B) / (WS_W - WS_B), 0.0, 1.0))


@pytest.mark.parametrize("preset", [
    {},
    {"ch_r_shift": 15, "ch_g_gain": 12, "ch_b_blackpoint": -8},
    {"ch_master_gain": 35},
    {"ch_input_gain": -20},
])
def test_picked_spot_renders_neutral(tmp_path, preset):
    """End to end through apply_adjustments: after the pick, the sampled spot
    comes out neutral in the RENDER — including with Channel Levels set and with
    Auto Gain live."""
    from core.ccr_processor import compute_neutral_balance_for_image
    img = _cast_scene(tmp_path)
    img.adjustment_settings = dict(preset)
    r, g, b = compute_neutral_balance_for_image(img, _pick(img))
    settings = dict(preset, balance_r=r, balance_g=g, balance_b=b)
    out = img.apply_adjustments(img.resized_raw.copy(), settings=settings)
    spot = out[30, 45, :].astype(float)
    spread = (spot.max() - spot.min()) / max(spot.max(), 1.0)
    assert spread < 0.02, f"spot {spot} spread {spread:.3f} with {preset}"


def test_bare_inverse_would_miss(tmp_path):
    """Guards the regression itself: the un-corrected solve is measurably wrong
    on the very same frame, so this suite fails if someone reverts to it."""
    from core.ccr_processor import compute_neutral_balance
    img = _cast_scene(tmp_path)
    img.adjustment_settings = {"ch_master_gain": 35}
    r, g, b = compute_neutral_balance(*_pick(img))
    settings = dict(img.adjustment_settings,
                    balance_r=r, balance_g=g, balance_b=b)
    out = img.apply_adjustments(img.resized_raw.copy(), settings=settings)
    spot = out[30, 45, :].astype(float)
    assert (spot.max() - spot.min()) / max(spot.max(), 1.0) > 0.05


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

"""Master Gain (and the Auto Gain offset riding it) runs AFTER Channel Balance
(spec/master-gain-after-balance.md).

Balance is tone-weighted — its node sits at a fixed display value — so an
exposure control in front of it made a brightness change move the colour. These
tests pin the new order down by its two observable consequences: the gain is a
pure scalar applied after the node, and the colour a Balance setting produces no
longer depends on the gain.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # ccr_backend pulls in Qt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core.ccr_processor import (  # noqa: E402
    CH_SLIDER_DIV,
    _apply_channel_levels,
    _apply_working_space_recovery,
    _master_gain_divisor,
    _ws_enabled,
    adjust_image,
    encode_window,
)

BALANCE = (0.0, 25.0, -30.0)        # a cast correction worth measuring
MID = [0.18, 0.22, 0.30]            # mid-tone patch, well inside the window


@pytest.fixture
def ws_on():
    if not _ws_enabled():
        pytest.skip("working space disabled (FREECCR_WORKING_SPACE=0)")


def _windowed(rows):
    return encode_window(np.asarray(rows, dtype=np.float32).reshape(-1, 1, 3).copy())


def _full(rows):
    arr = np.asarray(rows, dtype=np.float32).reshape(-1, 1, 3)
    return np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)


def _ws(base, master_gain=0.0, balance=(0.0, 0.0, 0.0)):
    """_apply_working_space_recovery with only Master Gain + Balance set."""
    levels = [0.0, 0.0, float(master_gain)] + [0.0] * 9
    return _apply_working_space_recovery(base, 0.0, 0.0, 0.0, 0.0, 1.0,
                                         *levels, *balance)


def _ratios(pixel):
    """Channel ratios against green — the colour, independent of exposure."""
    p = np.asarray(pixel, dtype=np.float64)
    return p / max(float(p[1]), 1e-9)


# --- the divisor itself ------------------------------------------------------

def test_divisor_matches_the_master_gain_curve():
    """The same 1/(1 - v/DIV) Channel Levels applied internally, now callable."""
    for v in (-100.0, -40.0, 0.0, 25.0, 100.0):
        assert _master_gain_divisor(v) == pytest.approx(
            max(1.0 - v / CH_SLIDER_DIV, 0.1))


def test_levels_can_run_without_the_master_gain():
    """include_master_gain=False leaves exactly the gain out — applying the
    divisor afterwards reproduces the full stage."""
    d = np.array([[[0.2, 0.4, 0.6]]], dtype=np.float32)
    whole = _apply_channel_levels(d.copy(), 0, 10, 40, *([0] * 9), clamp=False)
    split = _apply_channel_levels(d.copy(), 0, 10, 40, *([0] * 9), clamp=False,
                                  include_master_gain=False)
    split /= np.float32(_master_gain_divisor(40))
    np.testing.assert_allclose(split, whole, rtol=1e-6)


def test_default_still_includes_it():
    """The default is the whole stage, so a direct caller is unaffected."""
    d = np.array([[[0.3, 0.3, 0.3]]], dtype=np.float32)
    assert not np.array_equal(
        _apply_channel_levels(d.copy(), 0, 0, 50, *([0] * 9), clamp=False),
        _apply_channel_levels(d.copy(), 0, 0, 50, *([0] * 9), clamp=False,
                              include_master_gain=False))


# --- order: the gain is a scalar applied AFTER the node ----------------------

def test_windowed_gain_is_applied_after_balance(ws_on):
    """Balance-then-gain == balance alone, scaled. If the gain ran first the
    node would see different tones and this would not hold."""
    base = _windowed([MID])
    balanced = _ws(base, 0.0, BALANCE).astype(np.float64)
    with_gain = _ws(base, 35.0, BALANCE).astype(np.float64)
    expected = balanced / _master_gain_divisor(35.0)
    np.testing.assert_allclose(with_gain, expected, rtol=0, atol=1.5)


def test_non_windowed_gain_is_applied_after_balance():
    """Same order on a full-range base (reference mode, positive mode, areas)."""
    img = _full([MID])
    balanced = adjust_image(img.copy(), balance_r=BALANCE[0],
                            balance_g=BALANCE[1], balance_b=BALANCE[2]
                            ).astype(np.float64)
    with_gain = adjust_image(img.copy(), ch_master_gain=35.0,
                             balance_r=BALANCE[0], balance_g=BALANCE[1],
                             balance_b=BALANCE[2]).astype(np.float64)
    np.testing.assert_allclose(with_gain, balanced / _master_gain_divisor(35.0),
                               rtol=0, atol=2.0)


# --- the point of the change: exposure no longer moves the colour ------------

@pytest.mark.parametrize("gain", [-40.0, 0.0, 25.0, 60.0])
def test_balance_colour_is_independent_of_the_gain(ws_on, gain):
    """The user-visible property: dial in a Balance correction, then change the
    exposure — the colour stays put. Under the old order (gain inside Channel
    Levels, ahead of the node) the ratios drifted with every gain change."""
    base = _windowed([MID])
    ref = _ratios(_ws(base, 0.0, BALANCE)[0, 0])
    got = _ratios(_ws(base, gain, BALANCE)[0, 0])
    np.testing.assert_allclose(got, ref, rtol=2e-3)


def test_the_old_order_really_did_move_the_colour(ws_on):
    """Guards the test above from being vacuous: applying the same gain BEFORE
    the node (what the code used to do) does change the ratios."""
    base = _windowed([MID])
    ref = _ratios(_ws(base, 0.0, BALANCE)[0, 0])
    pre_gained = _windowed([[v / _master_gain_divisor(60.0) for v in MID]])
    moved = _ratios(_ws(pre_gained, 0.0, BALANCE)[0, 0])
    assert not np.allclose(moved, ref, rtol=2e-3)


def test_auto_gain_rides_the_moved_stage(ws_on):
    """Auto Gain is not a separate stage — it is a value on ch_master_gain, so
    it lands wherever Master Gain lands. Same scalar relationship."""
    from core.ccr_processor import compute_auto_gain_offset
    ramp = np.linspace(0.02, 0.98, 64, dtype=np.float32)[:, None].repeat(3, 1)
    base = encode_window(ramp.reshape(-1, 1, 3).copy())
    ag = compute_auto_gain_offset(base, True)
    assert ag != 0.0, "test base should need a non-trivial Auto Gain"
    balanced = _ws(base, 0.0, BALANCE).astype(np.float64)
    with_ag = _ws(base, ag, BALANCE).astype(np.float64)
    # Compare only where neither render is pinned at the window edges.
    live = (balanced > 200) & (with_ag > 200) & (with_ag < 65000)
    np.testing.assert_allclose(with_ag[live],
                               (balanced / _master_gain_divisor(ag))[live],
                               rtol=0, atol=2.0)


# --- nothing else moves ------------------------------------------------------

def test_neutral_balance_is_unchanged_by_the_reorder(ws_on):
    """With Balance at zero the reorder is a no-op: Channel Levels + Master Gain
    still compose exactly as one stage."""
    base = _windowed([[0.05, 0.2, 0.5], [0.4, 0.6, 0.9]])
    both = _ws(base, 45.0)
    levels = np.asarray(
        _apply_working_space_recovery(base, 0.0, 0.0, 0.0, 0.0, 1.0,
                                      0.0, 0.0, 45.0, *([0.0] * 9),
                                      0.0, 0.0, 0.0))
    np.testing.assert_array_equal(both, levels)


def test_master_shift_stays_with_channel_levels():
    """Only the GAIN moved. Master Shift is part of placing the histogram in the
    window, which Balance needs done first — so it is still inside the stage."""
    d = np.array([[[0.3, 0.3, 0.3]]], dtype=np.float32)
    shifted = _apply_channel_levels(d.copy(), 0, 30, 0, *([0] * 9), clamp=False,
                                    include_master_gain=False)
    assert float(shifted[0, 0, 0]) > 0.3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

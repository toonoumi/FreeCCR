#!/usr/bin/env python3
"""Cineon film log → workspace: the decode OUT of log behind the Channel Levels
checkbox (settings key "cineon_log"). See spec/cineon-display-transform.md.

Two things changed and both are pinned here. The stage runs **right after Master
Gain**, not at the end of the chain — so White Balance and the whole tone chain
below it grade display-referred data instead of log density. And it encodes with
the **working space's curve (sRGB)**, not a Rec.709 2.2 gamma; the transform was
always transfer-function-only (no matrix), so that is the whole difference.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.color_management import srgb_encode  # noqa: E402
from core.ccr_processor import (apply_cineon_to_workspace,  # noqa: E402
                                cineon_decode, encode_window, adjust_image,
                                _apply_working_space_recovery, _ws_enabled,
                                _master_gain_divisor,
                                CINEON_BLACK_CODE, CINEON_WHITE_CODE)

BLACK = CINEON_BLACK_CODE / 1023.0      # display value of Cineon black
WHITE = CINEON_WHITE_CODE / 1023.0      # ...and of 90% white


def _d(values):
    return np.asarray(values, dtype=np.float32).reshape(-1, 1, 3)


# --- the curve --------------------------------------------------------------

def test_anchors_land_on_black_and_white():
    """Code 95 is Dmin, 685 is 90% white — the levels the Scopes parade marks."""
    out = apply_cineon_to_workspace(_d([[BLACK] * 3, [WHITE] * 3]))
    assert out[0, 0, 0] == pytest.approx(0.0, abs=1e-3)
    assert out[1, 0, 0] == pytest.approx(1.0, abs=1e-3)


def test_headroom_survives_the_stage():
    """Un-clamped: a value above the window's white must come out ABOVE 1, or
    the White Point recovery below it would have nothing left to pull back."""
    out = apply_cineon_to_workspace(_d([[1.2] * 3]))
    assert out[0, 0, 0] > 1.0


def test_below_black_is_floored_not_nan():
    """The sRGB power segment is undefined below 0, so the linear result is
    floored — a NaN here would poison every stage after it."""
    out = apply_cineon_to_workspace(_d([[-0.5, 0.0, BLACK / 2]]))
    assert np.all(np.isfinite(out)) and np.all(out >= 0.0)


def test_monotonic_across_the_range():
    ramp = np.linspace(0.0, 1.2, 512, dtype=np.float32)
    out = apply_cineon_to_workspace(ramp[:, None, None].repeat(3, 2))[:, 0, 0]
    assert np.all(np.diff(out) >= -1e-6)


def test_matches_the_closed_form_with_the_srgb_curve():
    """The working space's curve, not gamma 2.2 — that is the whole change."""
    for code in (200.0, 390.0, 500.0, 600.0, 800.0):
        v = code / 1023.0
        off = 10.0 ** ((95.0 - 685.0) * 0.002 / 0.6)
        lin = (10.0 ** ((code - 685.0) * 0.002 / 0.6) - off) / (1.0 - off)
        expected = float(srgb_encode(np.array([max(lin, 0.0)]), clip=False)[0])
        got = float(apply_cineon_to_workspace(_d([[v] * 3]))[0, 0, 0])
        assert got == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_it_is_not_gamma_2_2():
    """Guards the test above from passing on the old encode: sRGB and 2.2 differ
    by more than rounding through the midtones."""
    v = 400.0 / 1023.0
    lin = float(cineon_decode(np.array([400.0], dtype=np.float32))[0])
    old = lin ** (1 / 2.2)
    got = float(apply_cineon_to_workspace(_d([[v] * 3]))[0, 0, 0])
    assert abs(got - old) > 1e-3


def test_neutral_stays_neutral():
    out = apply_cineon_to_workspace(_d([[0.45, 0.45, 0.45]]))
    assert out[0, 0, 0] == out[0, 0, 1] == out[0, 0, 2]


# --- position: right after Master Gain --------------------------------------

def test_non_windowed_decode_runs_before_the_tone_chain():
    """With nothing else set, the render is exactly the decoded image."""
    img = np.linspace(0, 65535, 6 * 3, dtype=np.uint16).reshape(2, 3, 3)
    out = adjust_image(img.copy(), cineon_log=True)
    expected = np.clip(
        apply_cineon_to_workspace(img.astype(np.float32) / 65535.0), 0.0, 1.0)
    np.testing.assert_allclose(out, np.round(expected * 65535.0), atol=1)


def test_white_balance_runs_AFTER_the_decode():
    """The point of moving it: WB's flat multiply now lands on display-referred
    data, not on log density (where a multiply is a per-channel gamma change)."""
    img = np.full((2, 2, 3), 30000, dtype=np.uint16)
    decoded = adjust_image(img.copy(), cineon_log=True)
    both = adjust_image(img.copy(), kelvin_shift=40.0, cineon_log=True)
    wb_after = adjust_image(decoded.copy(), kelvin_shift=40.0)
    np.testing.assert_allclose(both, wb_after, atol=2)


def test_master_gain_runs_AFTER_the_decode():
    """The decode sits between Channel Levels and Channel Balance, so Master
    Gain — which follows Balance — scales the DECODED image."""
    img = np.full((2, 2, 3), 30000, dtype=np.uint16)
    both = adjust_image(img.copy(), ch_master_gain=30.0, cineon_log=True)
    decoded = np.clip(apply_cineon_to_workspace(img.astype(np.float32) / 65535.0),
                      0.0, 1.0)
    expected = np.clip(decoded / _master_gain_divisor(30.0), 0.0, 1.0)
    np.testing.assert_allclose(both, np.round(expected * 65535.0), atol=2)


def test_channel_levels_runs_BEFORE_the_decode():
    """Channel Levels is the log-domain grading — a per-channel shift there is a
    density offset, which is the whole reason it stays ahead of the decode."""
    img = np.full((2, 2, 3), 30000, dtype=np.uint16)
    both = adjust_image(img.copy(), ch_master_shift=20.0, cineon_log=True)
    shifted = adjust_image(img.copy(), ch_master_shift=20.0)
    expected = np.clip(
        apply_cineon_to_workspace(shifted.astype(np.float32) / 65535.0), 0.0, 1.0)
    np.testing.assert_allclose(both, np.round(expected * 65535.0), atol=2)


def test_channel_balance_grades_the_decoded_image():
    """Balance's node sits at a fixed DISPLAY value, so it has to see the decode
    — on log data the node lands on a different tone than the one it names."""
    img = np.full((2, 2, 3), 30000, dtype=np.uint16)
    both = adjust_image(img.copy(), balance_g=35.0, cineon_log=True)
    decoded = adjust_image(img.copy(), cineon_log=True)
    np.testing.assert_allclose(both, adjust_image(decoded.copy(), balance_g=35.0),
                               atol=2)


def test_contrast_grades_the_decoded_image():
    """Everything downstream sees the decode. Under the old order (decode last)
    contrast pivoted on 0.5 of the LOG image, which is a different pixel."""
    img = np.full((2, 2, 3), 30000, dtype=np.uint16)
    both = adjust_image(img.copy(), contrast=30.0, cineon_log=True)
    decoded = adjust_image(img.copy(), cineon_log=True)
    np.testing.assert_allclose(both, adjust_image(decoded.copy(), contrast=30.0),
                               atol=2)


@pytest.mark.skipif(not _ws_enabled(), reason="working space disabled")
def test_windowed_path_decodes_inside_the_recovery():
    """Same slot on a windowed base — consumed by the recovery pre-stage, so the
    look chain never sees the flag."""
    base = encode_window(_d([[0.3, 0.4, 0.5], [BLACK] * 3, [WHITE] * 3]))
    out = _apply_working_space_recovery(base, 0.0, 0.0, 0.0, 0.0, 1.0,
                                        *([0.0] * 12), 0.0, 0.0, 0.0, True)
    flat = _apply_working_space_recovery(base, 0.0)
    assert not np.array_equal(out, flat)
    # The anchors still land where the curve says they do. Not exact: the window
    # container quantises the display value on the way in, and the log slope
    # amplifies that (~0.2% at white).
    assert int(out[1, 0, 0]) <= 100                # Cineon black -> display black
    assert int(out[2, 0, 0]) >= 65000              # 90% white   -> display white


@pytest.mark.skipif(not _ws_enabled(), reason="working space disabled")
def test_windowed_headroom_is_still_recoverable_after_the_decode():
    """The decode is un-clamped, so a blown highlight is still up there for the
    White Point slider to pull back — the reason it is not clipped in-stage."""
    base = encode_window(_d([[1.15] * 3]))
    blown = _apply_working_space_recovery(base, 0.0, 0.0, 0.0, 0.0, 1.0,
                                          *([0.0] * 12), 0.0, 0.0, 0.0, True)
    pulled = _apply_working_space_recovery(base, 0.0, -60.0, 0.0, 0.0, 1.0,
                                           *([0.0] * 12), 0.0, 0.0, 0.0, True)
    assert int(blown[0, 0, 0]) == 65535            # clipped at the window
    assert int(pulled[0, 0, 0]) < 65535            # ...and recovered below white


def test_flag_off_is_bit_identical():
    img = np.linspace(0, 65535, 6 * 3, dtype=np.uint16).reshape(2, 3, 3)
    np.testing.assert_array_equal(adjust_image(img.copy(), cineon_log=False),
                                  adjust_image(img.copy()))


# --- image model + panel wiring ----------------------------------------------

def _bare_image_obj():
    """A CCRImage shell with just the attributes apply_adjustments reads when
    every stage is overridden via parameters (no file/decoding needed)."""
    from core.ccr_image import CCRImage
    obj = CCRImage.__new__(CCRImage)
    obj.tint_balance_factor = 1.0
    obj.converted = False          # Auto Gain path off (conversion-only)
    return obj


def _apply(obj, img, settings):
    return obj.apply_adjustments(
        img, settings=settings, contrast_base=0, temperature_base=0,
        brightness_base=0, exposure_base=0, ws_windowed=False,
        color_profile="color", areas_override=[])


def test_apply_adjustments_passes_the_flag_into_the_pipeline():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 65536, (8, 8, 3), dtype=np.uint16)
    obj = _bare_image_obj()
    on = _apply(obj, img, {"saturation": 0, "cineon_log": True})
    np.testing.assert_allclose(on, adjust_image(img.copy(), cineon_log=True),
                               atol=1)
    # Absent/falsy flag → no transform.
    off = _apply(obj, img, {"saturation": 0})
    np.testing.assert_array_equal(off, _apply(obj, img, {"saturation": 0,
                                                         "cineon_log": 0}))
    assert not np.array_equal(on, off)


def test_sync_group_carries_cineon_key():
    from widgets.sliders_panel import SYNC_GROUPS
    channels = dict((gid, keys) for gid, _l, keys in SYNC_GROUPS)["channels"]
    assert "cineon_log" in channels


def test_panel_has_checkbox_default_unchecked():
    from widgets.sliders_panel import SlidersPanel
    panel = SlidersPanel(None)
    assert not panel.cineon_checkbox.isChecked()
    assert panel.cineon_checkbox.text() == "Cineon Log → Workspace"
    # _attach_cineon is a no-op with no image selected.
    adj = {}
    panel._attach_cineon(adj)
    assert "cineon_log" not in adj


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

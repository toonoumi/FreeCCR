#!/usr/bin/env python3
"""Positive-mode base exposure (EXPERIMENT).

The positive decode maps sensor saturation to white with no auto-exposure
anywhere, so a shot that leaves highlight headroom renders dark. This adds the
stage-1 grey-placement gain in linear light (libraw exp_shift, before the gamma
encode), with highlight preservation so the lift does not simply clip the top.

Tunable via FREECCR_POSITIVE_EV while we evaluate it; 0 restores the exact
pre-experiment decode.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import rawpy  # noqa: E402
from core import ccr_image as ccr_image_mod  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402

ARW = os.path.join(os.path.dirname(__file__), "..", "example_raw", "DSC07096.ARW")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("FREECCR_POSITIVE_EV", raising=False)


def _pos(**kw):
    return CCRImage._raw_color_postprocess_kwargs(positive=True, preview=True, **kw)


# --- the gain itself -------------------------------------------------------- #

def test_positive_decode_carries_the_base_exposure():
    kw = _pos()
    # exp_shift is a LINEAR multiplier, not EV: +2 EV == 4.0.
    assert kw["exp_shift"] == pytest.approx(2.0 ** ccr_image_mod.POSITIVE_BASE_EV)
    assert kw["exp_shift"] == pytest.approx(4.0)
    # Without this the lift would just clip the top off (measured: 29% of pixels
    # clipped at 0.0 vs 10% at 1.0).
    assert kw["exp_preserve_highlights"] == 1.0


def test_the_rest_of_the_positive_decode_is_untouched():
    kw = _pos()
    assert kw["output_color"] == rawpy.ColorSpace.sRGB
    assert kw["gamma"] == (2.222, 4.5)
    assert kw["use_camera_wb"] is True
    assert kw["no_auto_bright"] is True
    assert "no_auto_scale" not in kw


@pytest.mark.parametrize("no_icc_default", [False, True])
def test_negative_decode_never_gets_an_exposure_shift(no_icc_default):
    """Regression: the negative path feeds the density math and must not move."""
    kw = CCRImage._raw_color_postprocess_kwargs(
        positive=False, preview=False, no_icc_default=no_icc_default)
    assert "exp_shift" not in kw
    assert "exp_preserve_highlights" not in kw
    assert kw["gamma"] == (1, 1)


# --- the env override ------------------------------------------------------- #

@pytest.mark.parametrize("ev, expected", [
    ("1", 2.0),
    ("1.5", 2.0 ** 1.5),
    ("3", 8.0),
    ("-1", 0.5),
    ("99", 8.0),      # libraw clamps at 8.0 and SILENTLY ignores more
    ("-99", 0.25),
])
def test_env_override_sets_and_clamps_the_shift(monkeypatch, ev, expected):
    monkeypatch.setenv("FREECCR_POSITIVE_EV", ev)
    assert _pos()["exp_shift"] == pytest.approx(expected)


def test_zero_ev_restores_the_exact_previous_decode(monkeypatch):
    monkeypatch.setenv("FREECCR_POSITIVE_EV", "0")
    kw = _pos()
    assert "exp_shift" not in kw and "exp_preserve_highlights" not in kw


def test_garbage_env_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("FREECCR_POSITIVE_EV", "not-a-number")
    assert _pos()["exp_shift"] == pytest.approx(
        2.0 ** ccr_image_mod.POSITIVE_BASE_EV)


# --- it actually brightens a real decode ------------------------------------ #

@pytest.mark.skipif(not os.path.exists(ARW), reason="example ARW not present")
def test_real_decode_is_brighter_and_does_not_clip_everything(monkeypatch):
    def decode():
        with rawpy.imread(ARW) as raw:
            return raw.postprocess(**_pos()).astype(np.float32) / 65535.0

    monkeypatch.setenv("FREECCR_POSITIVE_EV", "0")
    before = decode()
    monkeypatch.setenv("FREECCR_POSITIVE_EV", "2")
    after = decode()

    assert np.median(after) > np.median(before) * 2.0     # a real lift
    # Highlight preservation keeps the shift from simply clipping the frame.
    assert (after >= 0.999).mean() < 0.25

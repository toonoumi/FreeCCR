#!/usr/bin/env python3
"""
Tests for the Curves (tone curve) feature.

Curves are stored inside adjustment_settings under the "curves" key as a dict of
channel -> list of [x, y] control points in the 0..255 domain. apply_curves
builds composed per-channel 16-bit LUTs (composite "rgb" curve first, then the
per-channel curve, Photoshop-style) and applies them. See
spec/curves-tone-control.md.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.ccr_processor import (apply_curves, build_channel_lut,  # noqa: E402
                                _is_identity_curves)


def _ramp_image():
    """A small image spanning the full 16-bit range across both channels."""
    vals = np.array([0, 8192, 16384, 32768, 49152, 65535], dtype=np.uint16)
    img = np.stack([vals, vals, vals], axis=-1).reshape(2, 3, 3)
    return img


class TestIdentity:
    def test_none_is_identity(self):
        assert _is_identity_curves(None)
        assert _is_identity_curves({})

    def test_explicit_identity_points(self):
        curves = {"rgb": [[0, 0], [255, 255]], "r": [[0, 0], [255, 255]]}
        assert _is_identity_curves(curves)

    def test_apply_identity_returns_input_unchanged(self):
        img = _ramp_image()
        out = apply_curves(img, None)
        assert out is img  # fast path returns the same object
        out2 = apply_curves(img, {"rgb": [[0, 0], [255, 255]]})
        assert np.array_equal(out2, img)

    def test_identity_lut_is_ramp(self):
        lut = build_channel_lut([[0, 0], [255, 255]])
        assert np.allclose(lut, np.arange(256), atol=1e-3)


class TestApply:
    def test_lift_black_point_raises_shadows(self):
        # Move bottom-left endpoint up: (0,0) -> (0,64). Output min must rise.
        curves = {"rgb": [[0, 64], [255, 255]]}
        img = _ramp_image()
        out = apply_curves(img, curves)
        # Pure black input is lifted toward ~64/255 of full scale.
        assert out[0, 0, 0] > 12000
        # Output stays within range and is monotonic non-decreasing along ramp.
        flat_in = img[..., 0].ravel()
        flat_out = out[..., 0].ravel()
        order = np.argsort(flat_in)
        assert np.all(np.diff(flat_out[order].astype(np.int64)) >= 0)
        assert out.max() <= 65535

    def test_pull_white_point_lowers_highlights(self):
        curves = {"rgb": [[0, 0], [255, 191]]}
        img = _ramp_image()
        out = apply_curves(img, curves)
        # Pure white is pulled down to ~191/255.
        assert out[1, 2, 0] < 52000
        assert out[1, 2, 0] > 45000

    def test_per_channel_only_affects_that_channel(self):
        # Lift red only; green and blue stay identity.
        curves = {"r": [[0, 0], [128, 200], [255, 255]]}
        img = _ramp_image()
        out = apply_curves(img, curves)
        assert np.array_equal(out[..., 1], img[..., 1])
        assert np.array_equal(out[..., 2], img[..., 2])
        # Red midtones are lifted above the input.
        assert out[1, 0, 0] > img[1, 0, 0]

    def test_composite_then_channel_compose_order(self):
        # Composite halves everything; red curve then doubles back.
        # Net effect on red should be close to identity; green/blue halved.
        curves = {
            "rgb": [[0, 0], [255, 128]],   # scale ~0.5
            "r":   [[0, 0], [128, 255]],   # scale ~2.0 over composite output
        }
        img = _ramp_image()
        out = apply_curves(img, curves)
        # Green is roughly halved at full white.
        assert abs(int(out[1, 2, 1]) - 32768) < 4000
        # Red recovers toward the original (much brighter than green).
        assert out[1, 2, 0] > out[1, 2, 1]

    def test_output_dtype_and_shape(self):
        curves = {"rgb": [[0, 0], [128, 160], [255, 255]]}
        out = apply_curves(_ramp_image(), curves)
        assert out.dtype == np.uint16
        assert out.shape == (2, 3, 3)


class TestRobustness:
    def test_unsorted_points_are_handled(self):
        curves = {"rgb": [[255, 255], [0, 0], [128, 128]]}
        out = apply_curves(_ramp_image(), curves)
        # Sorted identity-ish -> unchanged ramp.
        assert np.array_equal(out, _ramp_image())

    def test_malformed_channel_treated_as_identity(self):
        curves = {"rgb": [[0, 0]]}  # < 2 valid points
        assert _is_identity_curves(curves)
        out = apply_curves(_ramp_image(), curves)
        assert np.array_equal(out, _ramp_image())

    def test_out_of_range_points_clamped(self):
        curves = {"rgb": [[0, -50], [255, 999]]}
        out = apply_curves(_ramp_image(), curves)
        assert out.min() >= 0 and out.max() <= 65535


class TestEndpointHorizontal:
    """Endpoints move in X as well as Y. Dragging one inward sets the black /
    white INPUT point: tones beyond it clip to that endpoint's output level,
    rather than the curve extrapolating past its last knot."""

    def test_black_point_moved_right_clips_shadows(self):
        # Black endpoint dragged right to x=64 (= input 16448 of 65535).
        curves = {"rgb": [[64, 0], [255, 255]]}
        img = _ramp_image()
        out = apply_curves(img, curves)
        assert out[0, 0, 0] == 0        # input 0
        assert out[0, 1, 0] == 0        # input 8192  -> below the black point
        assert out[0, 2, 0] == 0        # input 16384 -> still below it
        assert out[1, 0, 0] > 0         # input 32768 -> above it, climbing
        assert out[1, 2, 0] == 65535    # white end untouched

    def test_white_point_moved_left_clips_highlights(self):
        # White endpoint dragged left to x=191 (= input 49087).
        curves = {"rgb": [[0, 0], [191, 255]]}
        img = _ramp_image()
        out = apply_curves(img, curves)
        assert out[1, 1, 0] == 65535    # input 49152 -> past the white point
        assert out[1, 2, 0] == 65535    # input 65535 -> clipped too
        assert out[0, 0, 0] == 0        # black end untouched

    def test_flat_hold_outside_endpoints_with_interior_point(self):
        # With >=3 points the Hermite path runs; outside the endpoint range it
        # must HOLD, not extrapolate the polynomial (which would overshoot and
        # could turn over, breaking monotonicity).
        lut = build_channel_lut([[64, 32], [128, 200], [192, 220]])
        assert np.all(lut[:64] == lut[64])
        assert np.all(lut[192:] == lut[192])
        assert lut.min() >= 0.0 and lut.max() <= 255.0
        # Still monotone non-decreasing across the whole 0..255 domain.
        assert np.all(np.diff(lut) >= -1e-5)

    def test_both_endpoints_moved_inward(self):
        lut = build_channel_lut([[32, 0], [224, 255]])
        assert np.all(lut[:32] == 0.0)
        assert np.all(lut[224:] == 255.0)
        assert lut[128] > 0.0 and lut[128] < 255.0

    def test_horizontally_moved_endpoints_are_not_identity(self):
        assert not _is_identity_curves({"rgb": [[64, 0], [255, 255]]})
        assert not _is_identity_curves({"rgb": [[0, 0], [191, 255]]})


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

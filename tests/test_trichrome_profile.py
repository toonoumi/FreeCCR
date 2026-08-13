#!/usr/bin/env python3
"""
Tests for camera profiles on trichrome (3-way RGB merge) captures.
See spec/trichrome-camera-profile.md §8.

Follows the merge tests' convention: merge_raw_channels is the only
rawpy-touching function, so it is monkeypatched and everything around it is
exercised for real.
"""
import logging
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import ccr_merge  # noqa: E402
from core import color_management as cm  # noqa: E402
from core import dcp_profile as dcp  # noqa: E402
from core import it8_profile as it8  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402

D50 = np.array(cm.D50_XYZ)
# sort_for_merge orders by basename, so a real trichrome set is named so that
# the sorted order IS (red, green, blue) — sequential frame numbers.
SOURCES = ["/chart/01_red.arw", "/chart/02_green.arw", "/chart/03_blue.arw"]


class _Fit:
    """Minimal CameraFit stand-in for the profile builders."""
    wb_id = "A1"                                 # anchor patch (cLUT builder)
    used_ids = list(it8.COLOR_IDS[:40])          # patches the cLUT residual bends on

    def __init__(self, wb=(1.8, 1.0, 1.3)):
        M = np.array([[0.46, 0.31, 0.17],
                      [0.23, 0.70, 0.07],
                      [0.02, 0.12, 0.93]])
        self.matrix = np.diag(D50 / (M @ np.ones(3))) @ M
        self.wb_mult = np.asarray(wb, dtype=np.float64)


@pytest.fixture
def merged(monkeypatch):
    """Patch merge_raw_channels and record how it was called."""
    rng = np.random.default_rng(5)
    arr = (rng.uniform(0.05, 0.5, (8, 10, 3)) * 65535).astype(np.uint16)
    calls = {}

    def fake(sources, preview=False, demosaic=False):
        calls["sources"] = list(sources)
        calls["preview"] = preview
        calls["demosaic"] = demosaic
        return arr.copy(), arr.shape[:2]

    monkeypatch.setattr(ccr_merge, "merge_raw_channels", fake)
    return arr, calls


@pytest.fixture(autouse=True)
def _clear_profiles():
    """No profile leaks between tests (the actives are module globals)."""
    cm.set_active_input_profile(None)
    cm.set_active_dcp_profile(None)
    cm.set_input_profile_disabled(False)
    CCRImage._kind_warned.clear()
    yield
    cm.set_active_input_profile(None)
    cm.set_active_dcp_profile(None)
    CCRImage._kind_warned.clear()


def _merged_image(demosaic=True):
    img = CCRImage.__new__(CCRImage)
    img.source_ops = []
    img.is_merged = True
    img.merge_sources = list(SOURCES)
    img.merge_demosaic = demosaic
    return img


# --------------------------------------------------------------------------- #
# 1-2. Profiling decode
# --------------------------------------------------------------------------- #

def test_decode_target_merged_uses_the_triplet_bare(merged, monkeypatch):
    """The profiling decode merges the triplet and measures UNTOUCHED device
    data — no field correction, no camera profile — even with both active."""
    arr, calls = merged
    cm.set_active_dcp_profile(dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(), "x")))
    called = []
    monkeypatch.setattr(CCRImage, "_apply_field_correction",
                        staticmethod(lambda a, **k: called.append("field") or a))
    out = it8.decode_target_merged(SOURCES, demosaic=True)
    np.testing.assert_array_equal(out, arr)
    assert calls["sources"] == SOURCES and calls["demosaic"] is True
    assert not called, "a profiling decode must not see field correction"


def test_decode_target_merged_requires_three_sources(merged):
    with pytest.raises(it8.IT8ReferenceError):
        it8.decode_target_merged(SOURCES[:2])
    with pytest.raises(it8.IT8ReferenceError):
        it8.decode_target_merged(SOURCES + ["/chart/extra.arw"])


def test_triplet_order_matches_an_import(merged):
    """Whatever order the user picks the files in, sort_for_merge decides which
    frame is red/green/blue — so the wizard and a 3-way import of the same three
    files build the same merge."""
    picked = ["/chart/03_blue.arw", "/chart/01_red.arw", "/chart/02_green.arw"]
    assert ccr_merge.sort_for_merge(picked) == SOURCES
    it8.decode_target_merged(ccr_merge.sort_for_merge(picked))
    _arr, calls = merged
    assert calls["sources"] == SOURCES


# --------------------------------------------------------------------------- #
# 3-5. Applying a profile to a merged image
# --------------------------------------------------------------------------- #

def test_merged_read_applies_the_dcp(merged):
    arr, _ = merged
    prof = dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(), "cal", trichrome=True))
    cm.set_active_dcp_profile(prof)
    out = _merged_image().read_image(SOURCES[0])
    np.testing.assert_array_equal(out, dcp.apply_dcp(prof, arr, as_shot_wb=None))
    assert not np.array_equal(out, arr)          # it really did something


def test_merged_read_applies_the_icc(merged):
    arr, _ = merged
    prof = cm.InputProfile.from_bytes(
        it8.build_camera_icc(_Fit(), "cal", trichrome=True))
    cm.set_active_input_profile(prof)
    out = _merged_image().read_image(SOURCES[0])
    np.testing.assert_array_equal(out, prof.apply(arr, as_shot_wb=None))


def test_merged_read_bare_device_skips_the_profile(merged):
    """apply_input_icc=False is the bare-device contract the profiling decode
    relies on — it must hold on the merged path too."""
    arr, _ = merged
    cm.set_active_dcp_profile(
        dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(), "cal")))
    out = _merged_image().read_image(SOURCES[0], apply_input_icc=False)
    np.testing.assert_array_equal(out, arr)


def test_merged_read_disabled_toggle_skips_the_profile(merged):
    arr, _ = merged
    cm.set_active_dcp_profile(
        dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(), "cal")))
    cm.set_input_profile_disabled(True)
    np.testing.assert_array_equal(_merged_image().read_image(SOURCES[0]), arr)


def test_profile_applies_after_field_correction(merged, monkeypatch):
    """Pipeline position: field correction, then the slice chain, then the
    profile — the same order the RAW branch uses, so the two compose alike."""
    order = []
    monkeypatch.setattr(CCRImage, "_apply_field_correction",
                        staticmethod(lambda a, **k: order.append("field") or a))
    monkeypatch.setattr(CCRImage, "_apply_source_ops",
                        lambda self, a: order.append("ops") or a)
    monkeypatch.setattr(CCRImage, "_apply_input_dcp",
                        lambda self, a, wb: order.append("profile") or a)
    cm.set_active_dcp_profile(
        dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(), "cal")))
    _merged_image().read_image(SOURCES[0])
    assert order == ["field", "ops", "profile"]


def test_baked_neutral_makes_the_merge_metadata_free(merged):
    """A merge has three source frames with three different camera_whitebalance
    values and none describes the merged balance, so the read passes None. A
    profile carrying its calibration neutral renders it correctly anyway."""
    arr, _ = merged
    fit = _Fit(wb=(1.9, 1.0, 1.4))
    prof = dcp.parse_dcp_bytes(dcp.build_camera_dcp(fit, "cal", trichrome=True))
    assert prof.as_shot_neutral is not None
    cm.set_active_dcp_profile(prof)
    out = _merged_image().read_image(SOURCES[0])
    # Identical to feeding the calibration WB by hand, and NOT to the unbalanced
    # path a profile without a neutral would have taken.
    np.testing.assert_array_equal(out, dcp.apply_dcp(prof, arr, as_shot_wb=None))
    plain = dcp.parse_dcp_bytes(
        dcp.build_camera_dcp(fit, "portable", bake_neutral=False))
    assert not np.array_equal(out, dcp.apply_dcp(plain, arr, as_shot_wb=None))


# --------------------------------------------------------------------------- #
# 6-7. Device-kind tag
# --------------------------------------------------------------------------- #

def test_kind_round_trips_in_both_containers():
    fit = _Fit()
    assert dcp.parse_dcp_bytes(
        dcp.build_camera_dcp(fit, "t", trichrome=True)).is_trichrome
    assert not dcp.parse_dcp_bytes(dcp.build_camera_dcp(fit, "n")).is_trichrome
    for mode_kw in ({}, {"trichrome": True}):
        prof = cm.InputProfile.from_bytes(it8.build_camera_icc(fit, "x", **mode_kw))
        assert prof.is_trichrome is bool(mode_kw)


def test_kind_round_trips_in_the_clut_container():
    fit = _Fit()
    samples, ref = _clut_inputs()
    prof = cm.InputProfile.from_bytes(it8.build_camera_icc(
        fit, "t", mode="clut", grid=9, samples=samples, ref=ref, trichrome=True))
    assert prof.is_trichrome


def _clut_inputs():
    """Enough sampled patches + reference for the cLUT builder."""
    rng = np.random.default_rng(4)
    fit = _Fit()
    Minv = np.linalg.inv(fit.matrix)
    ref = it8.IT8Reference(chart_type="IT8.7/1", batch="T")
    samples = {}
    for i, cid in enumerate(it8.COLOR_IDS[:40]):
        xyz = np.array([rng.uniform(5, 80), rng.uniform(5, 80), rng.uniform(5, 80)])
        lab = it8.xyz_to_lab(xyz)
        ref.patches[cid] = {"XYZ_X": xyz[0], "XYZ_Y": xyz[1], "XYZ_Z": xyz[2],
                            "LAB_L": lab[0], "LAB_A": lab[1], "LAB_B": lab[2]}
        raw = np.clip(Minv @ (xyz / 100.0) / fit.wb_mult, 0, 1)
        samples[cid] = it8.PatchSample(rgb=raw * 65535.0, valid=True, n_pix=900)
    return samples, ref


def test_kind_tag_does_not_disturb_the_rest():
    """A trichrome DCP is still a well-formed TIFF with ascending tags, and the
    calibration neutral still round-trips alongside it."""
    import struct
    fit = _Fit(wb=(1.9, 1.0, 1.4))
    data = dcp.build_camera_dcp(fit, "t", trichrome=True)
    p = dcp.parse_dcp_bytes(data)
    np.testing.assert_allclose(p.as_shot_neutral, 1.0 / fit.wb_mult, atol=1e-6)
    n = struct.unpack('<H', data[8:10])[0]
    tags = []
    for i in range(n):
        tag, typ, cnt = struct.unpack('<HHI', data[10 + i * 12:10 + i * 12 + 8])
        tags.append(tag)
        if tag == 52525:
            assert (typ, cnt) == (4, 1)          # LONG x1
    assert tags == sorted(tags)
    assert 52525 in tags
    # ICC: the neutral survives next to the kind tag.
    prof = cm.InputProfile.from_bytes(it8.build_camera_icc(fit, "t", trichrome=True))
    np.testing.assert_allclose(prof.calibration_neutral, 1.0 / fit.wb_mult, atol=1e-3)
    assert prof.is_trichrome


# --------------------------------------------------------------------------- #
# 8. Mismatch warning
# --------------------------------------------------------------------------- #

def test_kind_mismatch_warns_once_and_still_applies(merged, caplog):
    arr, _ = merged
    prof = dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(), "normal"))
    cm.set_active_dcp_profile(prof)
    with caplog.at_level(logging.WARNING):
        out1 = _merged_image().read_image(SOURCES[0])
        out2 = _merged_image().read_image(SOURCES[0])
    hits = [r for r in caplog.records if "device-space mismatch" in r.getMessage()]
    assert len(hits) == 1                        # once per profile+kind, not per frame
    np.testing.assert_array_equal(out1, out2)    # and it still applied
    assert not np.array_equal(out1, arr)


def test_matching_kinds_do_not_warn(merged, caplog):
    cm.set_active_dcp_profile(
        dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(), "t", trichrome=True)))
    with caplog.at_level(logging.WARNING):
        _merged_image().read_image(SOURCES[0])
    assert not [r for r in caplog.records if "device-space mismatch" in r.getMessage()]

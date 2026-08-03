#!/usr/bin/env python3
"""
Tests for the field-correction (flat-field) core (src/core/flat_field.py).
See spec/flat-field-correction.md §8. Pure numpy/cv2 — except the final
end-to-end test, which builds a real CCRImage and so needs an (offscreen) Qt
application for the preview QPixmap.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import flat_field as ff  # noqa: E402

FULL = 65535.0


# --------------------------------------------------------------------------- #
# Synthetic fields
# --------------------------------------------------------------------------- #

def make_field(h, w, *, strength=(0.55, 0.50, 0.45), center=(0.5, 0.5),
               spread=0.75):
    """A smooth per-channel falloff peaking at 1.0 at `center` — stands in for
    lens vignetting + colour shading + an (optionally off-centre) hot spot."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx + 0.5) / w - center[0]
    y = (yy + 0.5) / h - center[1]
    r2 = (x * x + y * y) / (spread ** 2)
    return np.stack([np.exp(-float(s) * r2) for s in strength], axis=2)


def calib_frame(h=768, w=1152, level=0.6, **kw):
    """A calibration shot of an evenly lit surface seen through `make_field`."""
    fld = make_field(h, w, **kw)
    return np.clip(fld * level * FULL, 0, FULL).astype(np.uint16), fld


def build(arr, **kw):
    return ff.build_profile(arr, "test", **kw)


# --------------------------------------------------------------------------- #
# 1. Building the map
# --------------------------------------------------------------------------- #

def test_recovers_the_synthetic_field():
    """The measured map is the inverse of the field it was shot through: gain x
    field is constant across the frame (its value is the central reference level,
    which is what makes gain == 1 at the centre)."""
    arr, fld = calib_frame()
    p = build(arr)
    gain = ff.gain_for_size(p, fld.shape[1], fld.shape[0])
    prod = gain * fld
    for c in range(3):
        ch = prod[..., c]
        # Achieved: cv ~0.0005, worst ~0.7% on the outermost row (0.01 EV).
        assert ch.std() / ch.mean() < 0.002
        assert np.abs(ch / ch.mean() - 1.0).max() < 0.015


def test_correction_flattens_a_vignetted_scene():
    """Apply to a scene shot through the same field: what comes back is the
    scene again, uniformly scaled — that scale is the only residual."""
    h, w = 768, 1152
    arr, fld = calib_frame(h, w)
    p = build(arr)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    scene = 0.35 + 0.15 * np.sin(xx / 90.0) * np.cos(yy / 70.0)
    scene = np.repeat(scene[:, :, None], 3, axis=2)
    shot = np.clip(scene * fld * FULL, 0, FULL).astype(np.uint16)

    out = ff.apply_field(shot, p).astype(np.float32) / FULL
    ratio = out / scene
    # Uniform across the frame (that is the whole point) to well under 1%.
    assert ratio.std() / ratio.mean() < 0.01
    # ...and much flatter than the uncorrected shot.
    raw_ratio = (shot.astype(np.float32) / FULL) / scene
    assert ratio.std() / ratio.mean() < 0.1 * (raw_ratio.std() / raw_ratio.mean())


def test_neutral_on_an_already_flat_frame():
    """A perfectly even calibration shot yields a unity map: activating such a
    profile must change neither exposure nor white balance."""
    flat = np.full((400, 600, 3), int(0.5 * FULL), dtype=np.uint16)
    flat[..., 0] = int(0.55 * FULL)          # a global cast must be preserved
    p = build(flat)
    assert np.allclose(p.gain, 1.0, atol=1e-3)
    out = ff.apply_field(flat, p)
    assert np.abs(out.astype(int) - flat.astype(int)).max() <= 1


def test_gain_is_unity_at_the_normalisation_reference():
    """Gain == 1 over the central window in every channel, so the profile is
    photometrically and chromatically neutral at the centre."""
    arr, _ = calib_frame()
    p = build(arr)
    h, w = p.gain.shape[:2]
    ch, cw = int(h * 0.316), int(w * 0.316)
    central = p.gain[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]
    assert abs(float(central.mean()) - 1.0) < 0.02
    # Channel ratios at the centre are untouched (no white-balance shift).
    assert float(np.ptp(central.reshape(-1, 3).mean(axis=0))) < 0.02


def test_colour_shading_is_corrected_per_channel():
    """Different per-channel falloff -> neutral corners, and cast_stops reports
    the spread."""
    arr, fld = calib_frame(strength=(0.9, 0.6, 0.3))
    p = build(arr)
    out = ff.apply_field(
        np.clip(fld * 0.6 * FULL, 0, FULL).astype(np.uint16), p
    ).astype(np.float32)
    corner = out[:40, :40].reshape(-1, 3).mean(axis=0)
    assert np.ptp(corner) / corner.mean() < 0.02        # corner is neutral again
    assert p.stats["cast_stops"] > 0.3                  # ...and it wasn't before


def test_off_centre_hot_spot_is_darkened():
    """A region brighter than the centre correctly gets gain < 1."""
    arr, _ = calib_frame(center=(0.25, 0.3), spread=0.6)
    p = build(arr)
    assert float(p.gain.min()) < 0.99
    assert float(p.gain.max()) > 1.5


def test_dust_and_hot_pixels_do_not_print_into_the_map():
    """Specks, hot pixels and smudges up to ~2.5% of the frame across are
    rejected. (Measured limit of the 13%-of-frame median window: a 2.6%-wide
    smudge leaves <2.5% gain error, a 5%-wide one ~7%. Beyond that a dark patch
    is no longer distinguishable from genuine light-source unevenness.)"""
    arr, _ = calib_frame()
    clean = build(arr).gain
    dirty = arr.copy()
    rng = np.random.default_rng(7)
    for _ in range(40):                                  # dust specks
        y, x = rng.integers(20, 700), rng.integers(20, 1100)
        dirty[y:y + 3, x:x + 3] = (dirty[y:y + 3, x:x + 3] * 0.45).astype(np.uint16)
    # Fibres / smudges, spaced out on purpose: two overlapping ones would form a
    # single blemish wider than the window is meant to handle.
    for y in (90, 420, 660):
        for x in (120, 520, 900):
            dirty[y:y + 30, x:x + 30] = (dirty[y:y + 30, x:x + 30] * 0.6).astype(np.uint16)
    for _ in range(40):                                  # hot pixels
        y, x = rng.integers(0, 768), rng.integers(0, 1152)
        dirty[y, x] = 65535
    got = build(dirty).gain
    assert np.abs(got - clean).max() < 0.025
    # ...and without rejection those blemishes would be far worse than that.
    raw = ff._resize_long_side(dirty.astype(np.float32) / FULL, ff.GRID_LONG_SIDE)
    naive = ff._central_level(raw)[None, None, :] / np.maximum(raw, 1e-4)
    assert np.abs(naive - clean).max() > 0.2


def test_a_sharp_mechanical_vignette_survives_blemish_rejection():
    """A fast corner cut is real shading, not a blemish: the median-based
    rejector must pass it through (an earlier depth-thresholded estimator
    smeared it, rewriting half the grid)."""
    h, w = 768, 1152
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx + .5) / w - .5) ** 2 + ((yy + .5) / h - .5) ** 2) / 0.5
    fld = 1.0 - 0.5 * np.clip((r - 0.72) / 0.28, 0, 1) ** 2   # 1 EV corner cut
    fld = np.repeat(fld[:, :, None], 3, axis=2).astype(np.float32)
    arr = np.clip(fld * 0.6 * FULL, 0, FULL).astype(np.uint16)

    p = build(arr)
    prod = ff.gain_for_size(p, w, h) * fld
    assert np.abs(prod[..., 0] / prod[..., 0].mean() - 1.0).max() < 0.08
    assert p.stats["corner_stops"][0] == pytest.approx(1.0, abs=0.1)


def test_gain_is_clamped_on_a_black_corner():
    """A vignette that reaches ~0 must clamp, not divide by zero."""
    arr, _ = calib_frame(strength=(9.0, 9.0, 9.0), spread=0.45)
    p = build(arr)
    assert np.isfinite(p.gain).all()
    assert float(p.gain.max()) == pytest.approx(ff.DEFAULT_MAX_GAIN, abs=1e-5)
    assert p.stats["clamped"] is True
    assert p.stats["max_gain_uncapped"] > ff.DEFAULT_MAX_GAIN


def test_max_gain_is_configurable():
    arr, _ = calib_frame(strength=(9.0, 9.0, 9.0), spread=0.45)
    p = build(arr, max_gain=2.0)
    assert float(p.gain.max()) == pytest.approx(2.0, abs=1e-5)


def test_corner_stops_report_the_falloff():
    arr, fld = calib_frame(strength=(0.55, 0.55, 0.55))
    p = build(arr)
    true_ev = -np.log2(fld[:40, :40].mean())
    assert p.stats["corner_stops"][0] == pytest.approx(true_ev, abs=0.1)


# --------------------------------------------------------------------------- #
# 2. Validation of the calibration shot
# --------------------------------------------------------------------------- #

def test_a_good_frame_raises_no_warnings():
    arr, _ = calib_frame()
    rng = np.random.default_rng(1)
    noisy = np.clip(arr.astype(np.float32)
                    + rng.normal(0, 0.002 * FULL, arr.shape), 0, FULL).astype(np.uint16)
    a = ff.analyze_field(noisy)
    assert a.warnings == []
    assert a.usable


def test_clipped_frame_warns():
    arr, _ = calib_frame(level=0.9)
    arr[100:400, 100:600] = 65535
    a = ff.analyze_field(arr)
    assert any("clipped" in w for w in a.warnings)
    assert a.clipped_fraction > 0.001


def test_dark_frame_warns_and_a_black_one_refuses():
    dark, _ = calib_frame(level=0.02)
    a = ff.analyze_field(dark)
    assert any("dark" in w for w in a.warnings)
    assert a.usable                                    # advisory, still buildable

    black = np.zeros((400, 600, 3), dtype=np.uint16)
    assert not ff.analyze_field(black).usable
    with pytest.raises(ff.FieldProfileError):
        build(black)


def test_a_photograph_of_a_scene_is_rejected_as_not_flat():
    h, w = 768, 1152
    scene = np.zeros((h, w, 3), dtype=np.uint16)
    for i in range(0, h, 96):                          # coarse structure
        for j in range(0, w, 96):
            scene[i:i + 96, j:j + 96] = int(FULL * (0.15 if (i + j) % 192 else 0.75))
    a = ff.analyze_field(scene)
    assert any("evenly lit" in w for w in a.warnings)
    assert a.flatness_residual > 0.05


# --------------------------------------------------------------------------- #
# 3. Storage
# --------------------------------------------------------------------------- #

def test_save_load_round_trip(tmp_path):
    arr, _ = calib_frame()
    p = build(arr, meta={"source": "cal.CR3",
                         "camera": {"model": "Z 8", "aperture": 8.0}})
    p.name = "Copy stand 105mm f/8"
    path = str(tmp_path / "profile.ffc")
    ff.save_profile(p, path)

    q = ff.load_profile(path)
    assert q.name == p.name
    assert np.array_equal(q.gain, p.gain)              # base64 float32 is exact
    assert q.aspect == pytest.approx(p.aspect)
    assert q.meta["camera"]["model"] == "Z 8"
    assert q.meta["stats"]["max_gain"] == p.stats["max_gain"]
    assert q.content_id and len(q.content_id) == 12
    assert q.path == path


def test_saved_file_is_readable_json_with_metadata(tmp_path):
    arr, _ = calib_frame()
    path = str(tmp_path / "p.ffc")
    ff.save_profile(build(arr), path)
    doc = json.loads(open(path, "rb").read().decode("utf-8"))
    assert doc["format"] == ff.FORMAT and doc["version"] == ff.FORMAT_VERSION
    assert doc["grid"]["channels"] == 3 and doc["grid"]["dtype"] == "float32"
    assert max(doc["grid"]["w"], doc["grid"]["h"]) == ff.GRID_LONG_SIDE


@pytest.mark.parametrize("mangle", [
    lambda d: {"format": "something-else", "version": 1},
    lambda d: dict(d, version=ff.FORMAT_VERSION + 1),
    lambda d: {k: v for k, v in d.items() if k != "grid"},
    lambda d: dict(d, grid=dict(d["grid"], data="!!!not base64!!!")),
    lambda d: dict(d, grid=dict(d["grid"], w=d["grid"]["w"] + 3)),
])
def test_bad_files_raise_field_profile_error(tmp_path, mangle):
    arr, _ = calib_frame()
    good = str(tmp_path / "good.ffc")
    ff.save_profile(build(arr), good)
    doc = json.loads(open(good, "rb").read().decode("utf-8"))
    bad = str(tmp_path / "bad.ffc")
    with open(bad, "wb") as f:
        f.write(json.dumps(mangle(doc)).encode("utf-8"))
    with pytest.raises(ff.FieldProfileError):
        ff.load_profile(bad)


def test_not_json_and_missing_file_raise(tmp_path):
    junk = str(tmp_path / "junk.ffc")
    with open(junk, "wb") as f:
        f.write(b"\x00\x01 not json at all")
    with pytest.raises(ff.FieldProfileError):
        ff.load_profile(junk)
    with pytest.raises(ff.FieldProfileError):
        ff.load_profile(str(tmp_path / "nope.ffc"))


# --------------------------------------------------------------------------- #
# 4. Applying
# --------------------------------------------------------------------------- #

def test_apply_is_resolution_independent():
    """A preview and a full-resolution export get the same correction at the
    same relative position."""
    import cv2
    arr, _ = calib_frame()
    p = build(arr)
    small = ff.gain_for_size(p, 270, 180)
    big = ff.gain_for_size(p, 2700, 1800)
    big_down = cv2.resize(big, (270, 180), interpolation=cv2.INTER_AREA)
    assert np.abs(small - big_down).max() < 0.01


def test_banded_apply_matches_a_single_shot_multiply():
    arr, _ = calib_frame(h=1600, w=1200)               # > _BAND_ROWS tall
    p = build(arr)
    img = (np.random.default_rng(3).random((1600, 1200, 3)) * 0.5 * FULL).astype(np.uint16)
    got = ff.apply_field(img, p)
    gain = ff.gain_for_size(p, 1200, 1600)
    want = np.clip(img.astype(np.float32) * gain, 0, FULL).astype(np.uint16)
    assert np.array_equal(got, want)


def test_apply_preserves_dtype_and_shape_and_clips():
    arr, _ = calib_frame()
    p = build(arr)
    img = np.full((300, 400, 3), 65000, dtype=np.uint16)
    out = ff.apply_field(img, p)
    assert out.dtype == np.uint16 and out.shape == img.shape
    assert out.max() <= 65535                          # gain > 1 must not wrap


def test_no_profile_is_a_passthrough():
    img = np.full((32, 32, 3), 1234, dtype=np.uint16)
    assert ff.apply_field(img, None) is img


def test_mono_keeps_the_frame_neutral():
    arr, _ = calib_frame(strength=(0.9, 0.6, 0.3))     # strong colour shading
    p = build(arr)
    gray = np.full((400, 600, 3), int(0.4 * FULL), dtype=np.uint16)
    out = ff.apply_field(gray, p, mono=True)
    assert np.array_equal(out[..., 0], out[..., 1])
    assert np.array_equal(out[..., 1], out[..., 2])
    colour = ff.apply_field(gray, p)                   # ...unlike the RGB path
    assert not np.array_equal(colour[..., 0], colour[..., 2])


def test_encoded_apply_corrects_in_linear_light():
    """A gamma-encoded (Positive-mode) frame is linearised before the multiply,
    so the correction is the same *light* ratio a linear frame would get."""
    arr, _ = calib_frame()
    p = build(arr)
    lin = np.full((400, 600, 3), 0.25, dtype=np.float32)
    enc = (ff._linear_to_srgb(lin) * FULL).astype(np.uint16)
    out = ff.apply_field(enc, p, encoded=True).astype(np.float32) / FULL
    got = ff._srgb_to_linear(out)
    want = lin * ff.gain_for_size(p, 600, 400)
    assert np.abs(got - want).max() < 2e-3
    # Applying it as if the data were linear OVER-corrects (multiplying an
    # encoded value by g scales the light it stands for by ~g^2.4) — that is
    # exactly what encoded=True exists to avoid.
    naive = ff.apply_field(enc, p).astype(np.float32) / FULL
    assert ff.gain_for_size(p, 600, 400)[0, 0, 1] > 1.2        # a real correction
    assert ff._srgb_to_linear(naive)[0, 0, 1] > want[0, 0, 1] * 1.1


def test_gain_cache_is_bounded_and_correct():
    arr, _ = calib_frame()
    p = build(arr)
    first = ff.gain_for_size(p, 100, 80)
    assert ff.gain_for_size(p, 100, 80) is first        # memoised
    for i in range(6):
        ff.gain_for_size(p, 100 + i, 80 + i)
    assert len(p._cache) <= 4


# --------------------------------------------------------------------------- #
# 5. Active-profile state and the grading signature
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _clear_active_profile():
    yield
    ff.set_active_profile(None)


def test_active_profile_state_and_signature(tmp_path):
    assert ff.active_field_signature() == "none"
    assert not ff.field_profile_active()

    arr, _ = calib_frame()
    other, _ = calib_frame(strength=(0.9, 0.8, 0.7))
    a, b = str(tmp_path / "a.ffc"), str(tmp_path / "b.ffc")
    ff.save_profile(build(arr), a)
    ff.save_profile(build(other), b)

    pa = ff.load_profile(a)
    ff.set_active_profile(pa)
    assert ff.field_profile_active()
    sig_a = ff.active_field_signature()
    assert sig_a.startswith("ff:")
    assert ff.active_field_signature() == sig_a         # stable while unchanged

    ff.set_active_profile(ff.load_profile(b))
    assert ff.active_field_signature() != sig_a         # a different profile...
    ff.set_active_profile(None)
    assert ff.active_field_signature() == "none"        # ...and cleared


def test_backend_signature_composes_the_field_profile(tmp_path):
    from core.ccr_backend import ccr_backend
    base = ccr_backend.active_profile_signature()
    arr, _ = calib_frame()
    path = str(tmp_path / "p.ffc")
    ff.save_profile(build(arr), path)
    ff.set_active_profile(ff.load_profile(path))
    composed = ccr_backend.active_profile_signature()
    assert composed != base
    assert composed.startswith(base + "+ff:")
    ff.set_active_profile(None)
    assert ccr_backend.active_profile_signature() == base


# --------------------------------------------------------------------------- #
# 6. The decode hook (spec §6.1)
# --------------------------------------------------------------------------- #

def test_decode_hook_corrects_and_bare_decodes_do_not(tmp_path, monkeypatch):
    """read_image applies the active profile; a bare-device profiling decode
    (apply_input_icc=False) must NOT, or profiles would compound."""
    import cv2
    from core.ccr_image import CCRImage

    arr, fld = calib_frame(h=200, w=300)
    p = build(arr)
    ff.set_active_profile(p)

    src = np.full((200, 300, 3), 20000, dtype=np.uint16)
    path = str(tmp_path / "scan.tif")
    assert cv2.imwrite(path, src)

    img = CCRImage.__new__(CCRImage)
    img.source_ops = []
    corrected = img.read_image(path)
    bare = img.read_image(path, apply_input_icc=False)

    assert not np.array_equal(corrected, bare)
    assert np.array_equal(bare, src)
    gain = ff.gain_for_size(p, 300, 200)
    want = np.clip(src.astype(np.float32) * gain, 0, FULL).astype(np.uint16)
    assert np.array_equal(corrected, want)

    ff.set_active_profile(None)
    assert np.array_equal(img.read_image(path), src)     # no profile -> untouched


def test_a_loaded_image_is_corrected_and_stamped(tmp_path):
    """End to end through the real CCRImage constructor: a vignetted scan comes
    out flat, and the image records that it was graded with this profile (which
    is what raises the thumbnail ⚠ when the profile later changes)."""
    import cv2
    # CCRImage.__init__ builds a preview QPixmap, which aborts the process
    # without a QApplication. Created here (not at module scope) so the rest of
    # this file stays Qt-free.
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv[:1])

    from core.ccr_image import CCRImage
    from core.ccr_backend import ccr_backend

    h, w = 400, 600
    cal, fld = calib_frame(h, w)
    p = build(cal)

    scan = np.clip(0.4 * fld * FULL, 0, FULL).astype(np.uint16)   # vignetted scan
    path = str(tmp_path / "scan.tif")
    assert cv2.imwrite(path, scan[..., ::-1])                     # cv2 wants BGR

    ff.set_active_profile(p)
    img = CCRImage(path)
    out = img.resized_raw.astype(np.float32)
    corner = out[:20, :20].mean()
    centre = out[h // 2 - 20:h // 2 + 20, w // 2 - 20:w // 2 + 20].mean()
    assert abs(corner / centre - 1.0) < 0.03          # the vignette is gone...
    raw_corner = scan[:20, :20].mean() / scan[h // 2 - 20:h // 2 + 20,
                                              w // 2 - 20:w // 2 + 20].mean()
    assert raw_corner < 0.75                          # ...and it was real
    assert "+ff:" in (img.profile_signature or "")
    assert img.profile_signature == ccr_backend.active_profile_signature()

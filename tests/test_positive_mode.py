#!/usr/bin/env python3
"""
Tests for global Positive mode (spec/positive-mode.md).

Positive mode decodes RAWs as normal sRGB positives and bypasses the film
negative pipeline: no conversion is needed, every image is editable/exportable
directly, and the export applies adjustments WITHOUT inversion.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import cv2  # noqa: E402
import rawpy  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# Full CCRImage construction builds QPixmaps, which needs a QApplication.
_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_image import CCRImage  # noqa: E402
from core.ccr_processor import adjust_image, ccr_export_positive  # noqa: E402


def _scan_png(tmp_path, name="pos.png", w=96, h=64):
    """A small 16-bit positive-ish image written to disk (non-RAW decode path)."""
    yy, xx = np.mgrid[0:h, 0:w]
    base = 8000 + 40000 * (xx / w) + 8000 * (yy / h)
    img = np.clip(np.stack([base, base * 0.95, base * 0.9], axis=-1),
                  500, 64000).astype(np.uint16)
    path = str(tmp_path / name)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return path


# --------------------------------------------------------------------------- #
# RAW decode kwargs (pure; the negative path must stay byte-for-byte unchanged)
# --------------------------------------------------------------------------- #
class TestRawPostprocessKwargs:
    def test_positive_decodes_as_srgb_photo(self):
        kw = CCRImage._raw_color_postprocess_kwargs(positive=True, preview=True)
        assert kw["output_color"] == rawpy.ColorSpace.sRGB
        assert kw["gamma"] == (2.222, 4.5)
        assert kw["use_camera_wb"] is True
        # Auto-brightness OFF so highlights are not clipped to white; rawpy's
        # auto-scale (no_auto_scale left absent/off) still gives a full-range
        # exposure.
        assert kw["no_auto_bright"] is True
        assert kw["demosaic_algorithm"] == rawpy.DemosaicAlgorithm.AHD
        assert "no_auto_scale" not in kw

    def test_negative_is_the_original_raw_readout(self):
        kw = CCRImage._raw_color_postprocess_kwargs(positive=False, preview=False)
        # Regression guard: the greenish raw-sensor decode the inverter expects.
        # Default (no_icc_default=False) = the ICC / bare-device decode that keeps
        # absolute sensor values (no_auto_scale=True + manual white-level scaling).
        assert kw["output_color"] == rawpy.ColorSpace.raw
        assert kw["gamma"] == (1, 1)
        assert kw["no_auto_bright"] is True
        assert kw["use_camera_wb"] is False
        assert kw["use_auto_wb"] is False
        assert kw["no_auto_scale"] is True

    def test_negative_no_icc_decode_is_adobe_autoscaled(self):
        # no_icc_default=True (no input ICC will correct the decode): Adobe RGB
        # with rawpy auto-scale (no_auto_scale=False); gamma/WB/auto-bright stay
        # as the negative readout.
        kw = CCRImage._raw_color_postprocess_kwargs(
            positive=False, preview=False, no_icc_default=True)
        assert kw["output_color"] == rawpy.ColorSpace.Adobe
        assert kw["gamma"] == (1, 1)
        assert kw["no_auto_bright"] is True
        assert kw["use_camera_wb"] is False
        assert kw["use_auto_wb"] is False
        assert kw["no_auto_scale"] is False

    def test_positive_ignores_no_icc_default_flag(self):
        # The positive path always decodes as an sRGB photo; no_icc_default is a
        # negative-only knob and must not leak into it.
        assert (CCRImage._raw_color_postprocess_kwargs(
            positive=True, preview=True, no_icc_default=True)["output_color"]
            == rawpy.ColorSpace.sRGB)

    def test_preview_flag_threads_to_half_size(self):
        assert CCRImage._raw_color_postprocess_kwargs(True, True)["half_size"] is True
        assert CCRImage._raw_color_postprocess_kwargs(True, False)["half_size"] is False
        assert CCRImage._raw_color_postprocess_kwargs(False, True)["half_size"] is True


# --------------------------------------------------------------------------- #
# read_image decode WIRING (RAW branch). The pure kwargs builder is tested above;
# these pin read_image's behaviour: a plain unprofiled negative decodes to Adobe
# RGB + rawpy auto-scale (thr=0), while ANY ICC path — an active input ICC, or the
# bare-device decode used to FIT one (apply_input_icc=False) — uses the canonical
# camera-profiling space: raw camera-native primaries + no_auto_scale + manual
# white-level scaling. So fit-space == apply-space EXACTLY. Verified by stubbing
# the heavy rawpy decode. (DNG is not special-cased: it applies the input ICC
# like any RAW.)
# --------------------------------------------------------------------------- #
from core import color_management as cm  # noqa: E402


def _matrix_shaper_profile():
    """A synthetic Adobe-RGB-like matrix-shaper InputProfile (any non-None
    profile flips the decode to the raw+ICC path)."""
    m = np.array([[0.5767309, 0.1855540, 0.1881852],
                  [0.2973769, 0.6273491, 0.0752741],
                  [0.0270343, 0.0706872, 0.9911085]])
    adapted = cm.M_BRADFORD_D65_D50 @ m
    icc = cm.build_matrix_shaper_icc(
        "Adobe-like", tuple(adapted[:, 0]), tuple(adapted[:, 1]),
        tuple(adapted[:, 2]), (2.19921875, 1.0, 0.0, 0.0, 0.0))
    return cm.InputProfile.from_bytes(icc)


class _FakeSizes:
    height, width = 64, 96


class _FakeRaw:
    """Minimal rawpy RawPy context-manager stand-in: records the postprocess
    kwargs and returns a small non-monochrome 16-bit RGB array."""
    def __init__(self, rec):
        self._rec = rec
        self.white_level = 16383
        self.sizes = _FakeSizes()
        self.num_colors = 3
        self.color_desc = b'RGGB'
        self.raw_pattern = np.array([[0, 1], [1, 2]])
        self.camera_whitebalance = [2.0, 1.0, 1.5, 0.0]   # as-shot WB (for DCP)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def postprocess(self, **kw):
        self._rec['kwargs'] = kw
        return np.full((32, 48, 3), 10000, dtype=np.uint16)


def _decode_raw(monkeypatch, path, *, profile, apply_input_icc=True,
                camera_matrix=False):
    """Run read_image's RAW branch with rawpy stubbed; return
    (postprocess kwargs, decoded array). positive_override=False pins negative
    mode (no backend dependency); the path need not exist on disk. _FakeRaw
    reports white_level=16383 and a flat 10000-valued decode, so the manual
    white-level scaling (when applied) multiplies by 65535/16383 ≈ 4.
    camera_matrix=True selects the Camera-Matrix decode (Adobe + auto-scale)."""
    rec = {}
    monkeypatch.setattr(rawpy, "imread", lambda p: _FakeRaw(rec))
    cm.set_active_input_profile(profile)
    cm.set_camera_matrix_mode(camera_matrix)
    try:
        img = CCRImage.__new__(CCRImage)
        img.source_ops = []
        img.file_path = path
        out = img.read_image(path, preview=True, positive_override=False,
                             apply_input_icc=apply_input_icc)
    finally:
        cm.set_active_input_profile(None)
        cm.set_camera_matrix_mode(False)
    return rec["kwargs"], out


class TestInputIccWillApply:
    def _img(self, name):
        img = CCRImage.__new__(CCRImage)
        img.file_path = name
        return img

    def test_true_when_profile_active(self):
        cm.set_active_input_profile(_matrix_shaper_profile())
        try:
            assert self._img("x.ARW")._input_icc_will_apply() is True
        finally:
            cm.set_active_input_profile(None)

    def test_false_when_no_profile(self):
        cm.set_active_input_profile(None)
        assert self._img("x.ARW")._input_icc_will_apply() is False

    def test_dng_is_no_longer_carved_out(self):
        # Regression: DNG used to return False unconditionally; it now applies
        # the input ICC like any other RAW.
        cm.set_active_input_profile(_matrix_shaper_profile())
        try:
            assert self._img("x.dng")._input_icc_will_apply() is True
        finally:
            cm.set_active_input_profile(None)

    def test_true_when_dcp_active(self):
        # A DCP (no ICC) also flips the decode to camera-native.
        from core import dcp_profile

        class _F:
            matrix = np.array([[0.46, 0.31, 0.17], [0.23, 0.70, 0.07], [0.02, 0.12, 0.93]])
            wb_mult = np.ones(3)
        cm.set_active_input_profile(None)
        cm.set_active_dcp_profile(dcp_profile.parse_dcp_bytes(
            dcp_profile.build_camera_dcp(_F(), "c")))
        try:
            assert self._img("x.arw")._input_icc_will_apply() is True
        finally:
            cm.set_active_dcp_profile(None)


def _camera_dcp_bytes():
    from core import dcp_profile

    class _F:
        matrix = np.array([[0.46, 0.31, 0.17], [0.23, 0.70, 0.07], [0.02, 0.12, 0.93]])
        wb_mult = np.ones(3)
    return dcp_profile.build_camera_dcp(_F(), "c")


class _FakeRawMono(_FakeRaw):
    def __init__(self, rec):
        super().__init__(rec)
        self.num_colors = 1

    def postprocess(self, **kw):
        self._rec['kwargs'] = kw
        return np.full((32, 48), 12000, dtype=np.uint16)     # single channel


class TestDcpDecodeWiring:
    def _decode(self, monkeypatch, fake_cls=_FakeRaw, dcp_active=True, apply_icc=True):
        from core import dcp_profile
        rec = {}
        monkeypatch.setattr(rawpy, "imread", lambda p: fake_cls(rec))
        cm.set_active_input_profile(None)
        cm.set_active_dcp_profile(dcp_profile.parse_dcp_bytes(_camera_dcp_bytes())
                                  if dcp_active else None)
        try:
            img = CCRImage.__new__(CCRImage)
            img.source_ops = []
            img.file_path = "scan.arw"
            out = img.read_image("scan.arw", preview=True, positive_override=False,
                                 apply_input_icc=apply_icc)
        finally:
            cm.set_active_dcp_profile(None)
        return rec.get("kwargs"), out

    def test_dcp_uses_camera_native_and_applies(self, monkeypatch):
        kw, out = self._decode(monkeypatch, dcp_active=True, apply_icc=True)
        assert kw["output_color"] == rawpy.ColorSpace.raw     # camera-native for DCP
        assert kw["no_auto_scale"] is True
        _, bare = self._decode(monkeypatch, dcp_active=True, apply_icc=False)  # bare device
        assert not np.array_equal(out, bare)                  # DCP was applied

    def test_dcp_skipped_for_monochrome(self, monkeypatch):
        _, with_dcp = self._decode(monkeypatch, fake_cls=_FakeRawMono, dcp_active=True)
        _, without = self._decode(monkeypatch, fake_cls=_FakeRawMono, dcp_active=False)
        assert np.array_equal(with_dcp, without)              # no profile burned into mono


class TestRawDecodeSpaceWiring:
    # No profile + normal decode → Adobe RGB + auto-scale (the camera-independent
    # default). ANY ICC path — active ICC, or the bare-device decode used to FIT
    # one (apply_input_icc=False) — → canonical camera-native raw + no_auto_scale +
    # manual white-level scaling, so the profile is fitted in and applied to the
    # SAME device space. _FakeRaw returns a flat 10000 decode, white_level=16383
    # (manual scaling ×65535/16383 ≈ 4 when it applies).
    def test_none_is_camera_native_raw(self, monkeypatch):
        # "None" now decodes bare camera-native RAW (linear, no auto-scale, manual
        # white-level scaling) — NOT Adobe.
        kw, out = _decode_raw(monkeypatch, "scan.arw", profile=None)
        assert kw["output_color"] == rawpy.ColorSpace.raw
        assert kw["no_auto_scale"] is True
        assert 39000 < int(out.max()) < 41000   # manual white-level scaling applied

    def test_camera_matrix_is_adobe_autoscaled(self, monkeypatch):
        # "Camera Matrix" reproduces the prior default: Adobe RGB + rawpy auto-scale.
        kw, out = _decode_raw(monkeypatch, "scan.arw", profile=None, camera_matrix=True)
        assert kw["output_color"] == rawpy.ColorSpace.Adobe
        assert kw["no_auto_scale"] is False
        assert int(out.max()) == 10000          # manual white-level scaling skipped

    def test_profile_active_is_camera_native(self, monkeypatch):
        # Active ICC → camera-native raw + no_auto_scale device space, ICC on top.
        kw, out = _decode_raw(monkeypatch, "scan.arw",
                              profile=_matrix_shaper_profile())
        assert kw["output_color"] == rawpy.ColorSpace.raw
        assert kw["no_auto_scale"] is True
        _, bare = _decode_raw(monkeypatch, "scan.arw", profile=None)
        assert not np.array_equal(out, bare)    # different space + ICC applied

    def test_dng_matches_other_raws(self, monkeypatch):
        kw, _ = _decode_raw(monkeypatch, "scan.dng",
                            profile=_matrix_shaper_profile())
        assert kw["output_color"] == rawpy.ColorSpace.raw
        assert kw["no_auto_scale"] is True

    def test_profiling_decode_is_camera_native_without_icc(self, monkeypatch):
        # IT8 profiling: apply_input_icc=False decodes the bare camera-native raw
        # device RGB (no ICC), WITH the manual white-level scaling applied so the
        # values are absolute — the exact space the ICC is later applied in.
        kw, out = _decode_raw(monkeypatch, "scan.arw",
                              profile=_matrix_shaper_profile(),
                              apply_input_icc=False)
        assert kw["output_color"] == rawpy.ColorSpace.raw
        assert kw["no_auto_scale"] is True
        # 10000 × 65535/16383 ≈ 40009 — manual white-level scaling applied.
        assert 39000 < int(out.max()) < 41000

    def test_fit_and_apply_decode_spaces_are_identical(self, monkeypatch):
        # The crux of a correct camera profile: the kwargs used to FIT the profile
        # (apply_input_icc=False) must equal those used to APPLY it (active ICC).
        fit_kw, _ = _decode_raw(monkeypatch, "scan.arw",
                                profile=_matrix_shaper_profile(),
                                apply_input_icc=False)
        apply_kw, _ = _decode_raw(monkeypatch, "scan.arw",
                                  profile=_matrix_shaper_profile())
        for key in ("output_color", "no_auto_scale", "gamma", "no_auto_bright",
                    "use_camera_wb", "use_auto_wb", "output_bps"):
            assert fit_kw[key] == apply_kw[key], f"fit/apply diverge on {key}"


# --------------------------------------------------------------------------- #
# Positive export — no inversion
# --------------------------------------------------------------------------- #
def _stub_image(resized_raw, color_profile="color", adjustment_settings=None):
    """A CCRImage shell (no file load) wired for apply_adjustments + export."""
    img = CCRImage.__new__(CCRImage)
    img.resized_raw = resized_raw
    img.adjustment_settings = adjustment_settings or {}
    img.color_profile = color_profile
    img.contrast_base = 0
    img.temperature_base = 0
    img.brightness_base = 0
    img.tint_balance_factor = 1.0
    img.area_layers = []
    img.fine_rotation_angle = 0
    img.rotation_angle = 0
    img.horizontal_mirrored = False
    img.vertical_mirrored = False
    img.crop_rect = None
    img.crop_angle = 0.0
    # Export re-reads the source via _load_export_source; for the stub (no file
    # on disk) serve the in-memory pixels instead.
    img.file_path = "stub.tiff"
    img.original_full_size = resized_raw.shape[:2]
    img.read_image = lambda *a, **k: img.resized_raw
    return img


def _asymmetric_image():
    img = np.zeros((2, 3, 3), dtype=np.uint16)
    img[0, 0] = (60000, 30000, 10000)
    img[0, 1] = (10000, 50000, 20000)
    img[0, 2] = (30000, 25000, 20000)
    img[1, 0] = (1000, 2000, 3000)
    img[1, 1] = (65000, 64000, 63000)
    img[1, 2] = (200, 100, 50)
    return img


class TestPositiveExport:
    def test_processing_path_returns_adjusted_not_inverted(self):
        src = _asymmetric_image()
        out = ccr_export_positive(_stub_image(src), output_path=None)
        # Identity adjustments → the positive is returned unchanged.
        np.testing.assert_array_equal(out, src)
        # And it is emphatically NOT the negative inversion.
        assert not np.array_equal(out, (65535 - src).astype(np.uint16))

    def test_processing_path_honors_bw_profile(self):
        src = _asymmetric_image()
        out = ccr_export_positive(_stub_image(src, color_profile="bw"),
                                  output_path=None)
        # B&W collapses to a single luminance channel (R==G==B), still no invert.
        assert np.array_equal(out[..., 0], out[..., 1])
        assert np.array_equal(out[..., 1], out[..., 2])
        assert out.max() < 65535  # not an inverted near-black image

    def test_processing_path_applies_real_adjustment(self):
        src = _asymmetric_image()
        out = ccr_export_positive(
            _stub_image(src, adjustment_settings={"exposure": 40}),
            output_path=None)
        assert not np.array_equal(out, src)  # exposure actually changed pixels

    def test_writes_a_file(self, tmp_path):
        src = _asymmetric_image()
        dest = str(tmp_path / "out.tiff")
        result = ccr_export_positive(_stub_image(src), output_path=dest)
        assert result is None
        assert os.path.exists(dest)


# --------------------------------------------------------------------------- #
# Backend: flag default, export routing, mode-toggle reprocess
# --------------------------------------------------------------------------- #
class _StubBackendImage:
    def __init__(self, converted=False, adjustment_settings=None,
                 source_ops=None, is_duplicate=False, tbf=1.0,
                 conversion_inputs=None, reference_frame=None):
        self.file_path = "stub.ARW"
        self.converted = converted
        self.conversion_inputs = conversion_inputs
        self.adjustment_settings = adjustment_settings or {}
        self.source_ops = source_ops or []
        self.is_duplicate = is_duplicate
        self.tint_balance_factor = tbf
        self.reference_frame = reference_frame
        self.reload_called = False
        self.preview_refreshed = False

    def reload_image(self):
        self.reload_called = True
        # Simulate read_image recomputing the tint factor from new pixels —
        # the reprocess must restore the inherited value for slices/duplicates.
        self.tint_balance_factor = 99.0

    def update_thumbnail_and_preview(self):
        self.preview_refreshed = True


@pytest.fixture
def backend():
    from core.ccr_backend import ccr_backend
    saved_images = ccr_backend.images
    saved_positive = ccr_backend.positive_mode
    saved_bp = ccr_backend.black_point_bgr
    saved_wp = ccr_backend.white_point_bgr
    yield ccr_backend
    ccr_backend.images = saved_images
    ccr_backend.positive_mode = saved_positive
    ccr_backend.black_point_bgr = saved_bp
    ccr_backend.white_point_bgr = saved_wp


class TestBackendPositiveMode:
    def test_default_is_false(self):
        from core.ccr_backend import CCRBackend
        # Bypass the singleton __new__ to check the _init default in isolation.
        fresh = object.__new__(CCRBackend)
        fresh._init()
        assert fresh.positive_mode is False

    def test_export_routes_to_positive(self, backend, monkeypatch):
        calls = {}
        monkeypatch.setattr("core.ccr_backend.ccr_export_positive",
                            lambda *a, **k: calls.setdefault("positive", k))
        monkeypatch.setattr("core.ccr_backend.ccr_normalize_with_reference",
                            lambda *a, **k: calls.setdefault("reference", k))
        backend.images = [_StubBackendImage(reference_frame=(0, 0, 10, 10))]
        backend.positive_mode = True
        assert backend.export_image_by_index(0, "out.tiff") is True
        assert "positive" in calls and "reference" not in calls

    def test_export_routes_to_negative_when_off(self, backend, monkeypatch):
        calls = {}
        monkeypatch.setattr("core.ccr_backend.ccr_export_positive",
                            lambda *a, **k: calls.setdefault("positive", k))
        monkeypatch.setattr("core.ccr_backend.ccr_normalize_with_reference",
                            lambda *a, **k: calls.setdefault("reference", k))
        backend.images = [_StubBackendImage(reference_frame=(0, 0, 10, 10))]
        backend.positive_mode = False
        backend.black_point_bgr = None
        backend.white_point_bgr = None
        assert backend.export_image_by_index(0, "out.tiff") is True
        assert "reference" in calls and "positive" not in calls

    def test_reprocess_drops_conversion_keeps_adjustments(self, backend):
        img = _StubBackendImage(converted=True,
                                adjustment_settings={"exposure": 20},
                                conversion_inputs={"mode": "ref"})
        backend.images = [img]
        backend.positive_mode = True
        backend.reprocess_all_for_positive_mode_change()
        assert img.reload_called
        assert img.converted is False
        assert img.conversion_inputs is None
        assert img.adjustment_settings == {"exposure": 20}  # kept

    def test_reprocess_preserves_inherited_tint_factor_for_slices(self, backend):
        slice_img = _StubBackendImage(source_ops=[(0, (0, 0, 1, 1))], tbf=1.234)
        backend.images = [slice_img]
        backend.positive_mode = True
        backend.reprocess_all_for_positive_mode_change()
        # reload_image set it to 99.0; the inherited value must be restored.
        assert slice_img.tint_balance_factor == pytest.approx(1.234)

    def test_reprocess_recomputes_tint_factor_for_normal_images(self, backend):
        plain = _StubBackendImage(tbf=1.0)  # no source_ops, not a duplicate
        backend.images = [plain]
        backend.positive_mode = True
        backend.reprocess_all_for_positive_mode_change()
        # A normal image keeps whatever reload_image computed (no restore).
        assert plain.tint_balance_factor == pytest.approx(99.0)


class TestBrightnessBaseline:
    """Positives must go decode -> user adjustments -> output with NO negative
    look baseline. The -8 brightness_base (a film-negative default) applies a
    gamma-1.3 darkening that crushes shadows; positives must use a neutral 0."""

    def test_negative_brightness_base_crushes_shadows_positive_is_identity(self):
        shadow = np.full((4, 4, 3), 8000, dtype=np.uint16)
        identity = adjust_image(shadow, brightness=0)
        darkened = adjust_image(shadow, brightness=-8)
        np.testing.assert_array_equal(identity, shadow)        # positive baseline: untouched
        assert darkened.mean() < shadow.mean() * 0.85          # negative baseline: darkened

    def test_fresh_positive_image_uses_neutral_brightness_base(self, tmp_path, backend):
        path = _scan_png(tmp_path)
        backend.positive_mode = True
        assert CCRImage(path).brightness_base == 0
        backend.positive_mode = False
        assert CCRImage(path).brightness_base == -8            # negatives keep the look

    def test_fresh_positive_preview_pipeline_is_identity(self, tmp_path, backend):
        # With the neutral baseline and no user sliders, the preview pipeline is
        # an identity on the decoded positive — no clipping/darkening step.
        path = _scan_png(tmp_path)
        backend.positive_mode = True
        pos = CCRImage(path)
        out = pos.apply_adjustments(pos.resized_raw)
        np.testing.assert_array_equal(out, pos.resized_raw)


class TestHiResSignature:
    """The zoom hi-res cache keys off _hires_signature, including base reuse. An
    un-converted image decodes to completely different pixels in positive (sRGB
    photo) vs negative (raw green scan) mode, so the signature MUST differ —
    otherwise toggling the mode reuses the stale decode base and re-adjusts the
    wrong pixels (the "green then dark non-green" hi-res glitch)."""

    def test_unconverted_signature_encodes_decode_mode(self, backend):
        from widgets.image_preview import ImagePreview
        # _hires_signature uses only its img arg and the global flag — no self
        # state — so a bare instance is enough (avoids the heavy Qt toolbar).
        ip = ImagePreview.__new__(ImagePreview)
        img = type("Img", (), {"converted": False, "conversion_inputs": None})()
        backend.positive_mode = True
        sig_positive = ip._hires_signature(img)
        backend.positive_mode = False
        sig_negative = ip._hires_signature(img)
        assert sig_positive != sig_negative


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

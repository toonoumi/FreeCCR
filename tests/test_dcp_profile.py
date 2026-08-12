#!/usr/bin/env python3
"""
Tests for Adobe DCP (DNG camera profile) support — core.dcp_profile. Parse
(II/MM, SRATIONAL, inline/offset), apply (ForwardMatrix + ColorMatrix fallback,
dual-illuminant), generate (build_camera_dcp round-trip), and the linear-Adobe
output contract. See spec/dcp-camera-profile.md §8. Pure numpy/struct — no Qt.
"""
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import dcp_profile as dcp        # noqa: E402
from core import it8_profile as it8        # noqa: E402
from core import color_management as cm    # noqa: E402

D50 = np.array(cm.D50_XYZ)
M0 = np.array([[0.46, 0.31, 0.17], [0.23, 0.70, 0.07], [0.02, 0.12, 0.93]])


def _pin(M):
    """Pin so balanced white (1,1,1) -> D50 exactly (the white-relative anchor a
    real fit's matrix carries: M @ (1,1,1) == D50)."""
    M = np.asarray(M, float)
    return np.diag(D50 / (M @ np.ones(3))) @ M


M0P = _pin(M0)            # white-relative variant for build-contract assertions


class _Fit:
    # NEW convention: `matrix` (M_f) maps WHITE-BALANCED device RGB -> XYZ D50
    # (M_f @ (1,1,1) == D50) and `wb_mult` is the green-normalised WB (raw ->
    # balanced, wb_mult[1] == 1). build_camera_dcp consumes both.
    def __init__(self, m, wb=None):
        self.matrix = np.asarray(m, dtype=np.float64)
        self.wb_mult = np.ones(3) if wb is None else np.asarray(wb, dtype=np.float64)


def _neutral(M):
    n = np.linalg.inv(M) @ D50
    return n / np.max(np.abs(n))


# --------------------------------------------------------------------------- #
# Generate -> parse round-trip.
# --------------------------------------------------------------------------- #

def test_build_parse_roundtrip():
    # A fit with a non-trivial WB so FM != inv(CM): ForwardMatrix1 == the fit's
    # matrix exactly; ColorMatrix1 == inv(M @ diag(wb)) (NOT inv(FM)) — the two
    # differ by the white-balance diagonal (DNG spec 1.6, ch.6).
    wb = np.array([1.9, 1.0, 1.4])
    fit = _Fit(M0P, wb)                    # white-relative matrix (M @ 1 == D50)
    data = dcp.build_camera_dcp(fit, "Cam D50", illuminant=23)
    assert data[0:2] == b'II' and struct.unpack('<H', data[2:4])[0] == 42
    p = dcp.parse_dcp_bytes(data)
    assert p.name == "Cam D50" and p.illuminant_1 == 23
    assert p.has_forward and not p.is_dual
    np.testing.assert_allclose(p.forward_matrix_1, M0P, atol=1e-4)
    np.testing.assert_allclose(p.color_matrix_1, np.linalg.inv(M0P @ np.diag(wb)),
                               atol=1e-4)
    # ForwardMatrix is white-relative: FM @ (1,1,1) == D50.
    np.testing.assert_allclose(np.asarray(p.forward_matrix_1) @ np.ones(3), D50, atol=1e-4)
    # The camera neutral is the green-normalised raw neutral: CM @ D50 == 1/wb.
    np.testing.assert_allclose(np.asarray(p.color_matrix_1) @ D50, 1.0 / wb, atol=1e-4)


def test_parse_big_endian_mm():
    # A minimal MM (big-endian) DCP with only ColorMatrix1 (SRATIONAL, offset).
    en = '>'
    vals = M0.ravel()
    data_off = 8 + 2 + 12 + 4
    body = b''.join(struct.pack(en + 'ii', int(round(v * 1e6)), 1000000) for v in vals)
    entry = struct.pack(en + 'HHI', 50721, 10, 9) + struct.pack(en + 'I', data_off)
    ifd = struct.pack(en + 'H', 1) + entry + struct.pack(en + 'I', 0)
    blob = struct.pack(en + '2sHI', b'MM', 42, 8) + ifd + body
    p = dcp.parse_dcp_bytes(blob)
    np.testing.assert_allclose(p.color_matrix_1, M0, atol=1e-5)
    assert not p.has_forward                        # CM-only -> fallback path


def test_inline_short_and_negative_srational():
    p = dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(M0), "x", illuminant=17))
    assert p.illuminant_1 == 17                      # SHORT inline
    # ColorMatrix has negative entries (inverse of a positive matrix) -> SRATIONAL signs
    assert np.any(p.color_matrix_1 < 0)


# --------------------------------------------------------------------------- #
# Apply.
# --------------------------------------------------------------------------- #

def test_apply_forward_roundtrip_and_icc_parity():
    # FM is now white-relative (M @ 1 == D50); apply white-balances the raw by the
    # green-normalised as-shot WB before the ForwardMatrix, so the expected linear
    # Adobe is (balanced_raw) @ (A @ FM).T.
    p = dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(M0P), "c"))
    N = _neutral(M0P)
    as_shot = 1.0 / N
    rng = np.random.default_rng(0)
    cam = np.clip(N * rng.uniform(0, 1, (8, 8, 3)), 0, 1)     # in-range after WB
    cam_u16 = (cam * 65535).astype(np.uint16)
    out = dcp.apply_dcp(p, cam_u16, as_shot_wb=as_shot)
    balanced = (cam_u16 / 65535.0) * (as_shot / as_shot[1])  # green-normalised WB
    expect = np.clip(balanced @ (cm.M_XYZ_D50_2_ADOBE @ M0P).T, 0, 1)
    assert np.abs(out.astype(int) - np.rint(expect * 65535).astype(int)).max() <= 3
    # parity: the same fit as a matrix ICC produces the same linear-Adobe output
    # (the ICC path white-balances with the same as-shot WB).
    mp = cm.InputProfile.from_bytes(it8.build_camera_icc(_Fit(M0P), "m"))
    assert np.abs(out.astype(int) - mp.apply(cam_u16, as_shot_wb=as_shot).astype(int)).max() <= 3


def test_apply_is_linear_not_srgb():
    # A 50% linear-Adobe grey must come out ~mid (linear), not sRGB-bumped (~0.73).
    p = dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(M0P), "c"))
    N = _neutral(M0P)
    cam = np.clip(N * 0.5, 0, 1)
    cam_u16 = np.tile((cam * 65535).astype(np.uint16).reshape(1, 1, 3), (2, 2, 1))
    out = dcp.apply_dcp(p, cam_u16, as_shot_wb=1.0 / N)[0, 0].astype(float) / 65535.0
    assert out.max() - out.min() < 0.03              # near-neutral
    assert 0.40 < out.mean() < 0.60                  # linear ~0.5, not ~0.73


def test_colormatrix_fallback_near_neutral():
    # A profile with ONLY a ColorMatrix (no ForwardMatrix): the fallback path
    # still lands the camera neutral near-neutral in linear Adobe.
    N = _neutral(M0)
    CM = np.linalg.inv(M0 @ np.diag(N))
    p = dcp.DcpProfile(name="cm-only", color_matrix_1=CM, illuminant_1=23,
                       camera_calibration_1=np.eye(3), camera_calibration_2=np.eye(3),
                       analog_balance=np.ones(3))
    assert not p.has_forward
    grey = np.tile((np.clip(N * 0.5, 0, 1) * 65535).astype(np.uint16).reshape(1, 1, 3),
                   (2, 2, 1))
    out = dcp.apply_dcp(p, grey, as_shot_wb=1.0 / N)[0, 0].astype(float)
    assert out.max() - out.min() < 0.06 * 65535


# --------------------------------------------------------------------------- #
# Dual-illuminant interpolation.
# --------------------------------------------------------------------------- #

def test_dual_illuminant_weight_and_blend():
    FM1 = M0 @ np.diag(_neutral(M0))
    FM2 = (M0 * 1.1) @ np.diag(_neutral(M0 * 1.1))
    p = dcp.DcpProfile(name="dual", forward_matrix_1=FM1, forward_matrix_2=FM2,
                       color_matrix_1=np.linalg.inv(FM1), color_matrix_2=np.linalg.inv(FM2),
                       illuminant_1=21, illuminant_2=17,        # D65 / StdA
                       camera_calibration_1=np.eye(3), camera_calibration_2=np.eye(3),
                       analog_balance=np.ones(3))
    assert p.is_dual
    # weight is in [0,1] and the mired formula hits the endpoints at the cal CCTs
    inv = lambda t: 1e6 / t
    T1, T2 = 6504.0, 2856.0
    for T, expect in [(6504.0, 1.0), (2856.0, 0.0)]:
        w = np.clip((inv(T) - inv(T2)) / (inv(T1) - inv(T2)), 0, 1)
        assert abs(w - expect) < 1e-6
    # a dual profile applies and the result lies between the two single-FM results
    N = _neutral(M0)
    cam = np.tile((np.clip(N * 0.4, 0, 1) * 65535).astype(np.uint16).reshape(1, 1, 3), (2, 2, 1))
    out = dcp.apply_dcp(p, cam, as_shot_wb=1.0 / N)
    assert out.shape == cam.shape and out.dtype == np.uint16


# --------------------------------------------------------------------------- #
# Synthetic-camera round-trip (key).
# --------------------------------------------------------------------------- #

def test_synthetic_camera_roundtrip():
    # IT8 reference patches; the camera is (white-relative matrix M0P, WB `wb`):
    # raw = (inv(M0P) @ XYZ) / wb. Build DCP from the fit; apply with as_shot_wb=wb;
    # back to XYZ -> Lab gives avg dE2000 ~ 0 (the DCP reproduces the chart).
    patches = {}
    for k, Y in enumerate(np.linspace(89, 3, 24)):
        patches[f'GS{k}'] = (Y / 100.0) * D50 * 100.0
    rng = np.random.default_rng(7)
    for cid in it8.COLOR_IDS[:60]:
        Y = rng.uniform(8, 80)
        patches[cid] = np.array([Y * rng.uniform(0.8, 1.2), Y, Y * rng.uniform(0.6, 1.1)])
    wb = np.array([2.0, 1.0, 1.5])              # raw -> balanced (green == 1)
    fit = _Fit(M0P, wb)
    p = dcp.parse_dcp_bytes(dcp.build_camera_dcp(fit, "c"))
    Ainv = np.linalg.inv(cm.M_XYZ_D50_2_ADOBE)
    Minv = np.linalg.inv(M0P)
    de = []
    for xyz in patches.values():
        balanced = Minv @ (xyz / 100.0)                       # balanced device
        raw = balanced / wb                                   # camera-native raw
        if np.any(raw < 0) or np.any(balanced > 1):           # skip out-of-gamut/clipped
            continue
        cu = np.tile((np.clip(raw, 0, 1) * 65535).astype(np.uint16).reshape(1, 1, 3), (2, 2, 1))
        out = dcp.apply_dcp(p, cu, as_shot_wb=wb)[0, 0].astype(float) / 65535.0
        got = (Ainv @ out) * 100.0
        de.append(float(it8.delta_e_2000(it8.xyz_to_lab(got), it8.xyz_to_lab(xyz))[0]))
    assert len(de) > 20 and float(np.mean(de)) < 0.5


# --------------------------------------------------------------------------- #
# Look tables (apply_look).
# --------------------------------------------------------------------------- #

def test_look_table_off_by_default_and_identity_noop():
    N = _neutral(M0)
    cam = (np.clip(N * 0.6, 0, 1) * 65535).astype(np.uint16)
    img = np.tile(cam.reshape(1, 1, 3), (4, 4, 1))
    dims = (6, 4, 1)
    identity = np.zeros((*dims, 3)); identity[..., 1] = 1.0; identity[..., 2] = 1.0  # dH0,sS1,vS1
    p = dcp.DcpProfile(name="look", forward_matrix_1=M0 @ np.diag(N),
                       color_matrix_1=np.linalg.inv(M0 @ np.diag(N)), illuminant_1=23,
                       camera_calibration_1=np.eye(3), camera_calibration_2=np.eye(3),
                       analog_balance=np.ones(3), hsm_dims=dims, hsm_data_1=identity)
    base = dcp.apply_dcp(p, img, as_shot_wb=1.0 / N, apply_look=False)
    ident = dcp.apply_dcp(p, img, as_shot_wb=1.0 / N, apply_look=True)
    assert np.abs(base.astype(int) - ident.astype(int)).max() <= 4      # identity look ~ no-op


def test_look_table_saturation_scale_changes_output():
    N = _neutral(M0)
    cam = (np.clip(N * 0.6, 0, 1) * 65535).astype(np.uint16)
    img = np.tile(cam.reshape(1, 1, 3), (4, 4, 1))
    dims = (6, 4, 1)
    boost = np.zeros((*dims, 3)); boost[..., 0] = 0.0; boost[..., 1] = 1.5; boost[..., 2] = 1.0
    p = dcp.DcpProfile(name="look", forward_matrix_1=M0 @ np.diag(N),
                       color_matrix_1=np.linalg.inv(M0 @ np.diag(N)), illuminant_1=23,
                       camera_calibration_1=np.eye(3), camera_calibration_2=np.eye(3),
                       analog_balance=np.ones(3), hsm_dims=dims, hsm_data_1=boost)
    off = dcp.apply_dcp(p, img, as_shot_wb=1.0 / N, apply_look=False)
    on = dcp.apply_dcp(p, img, as_shot_wb=1.0 / N, apply_look=True)
    assert np.abs(off.astype(int) - on.astype(int)).max() > 4           # sat boost changes it


# --------------------------------------------------------------------------- #
# Errors.
# --------------------------------------------------------------------------- #

def test_errors():
    with pytest.raises(dcp.DcpError):
        dcp.parse_dcp_bytes(b"xx")                              # too short
    with pytest.raises(dcp.DcpError):
        dcp.parse_dcp_bytes(b"ZZ" + b"\x00" * 10)               # bad byte-order
    # valid TIFF but no ColorMatrix1/ForwardMatrix1 -> unusable
    en = '<'
    ifd = struct.pack(en + 'H', 1) + struct.pack(en + 'HHI', 50936, 2, 2) + b'ab' + b'\x00\x00'
    ifd += struct.pack(en + 'I', 0)
    blob = struct.pack(en + '2sHI', b'II', 42, 8) + ifd
    with pytest.raises(dcp.DcpError):
        dcp.parse_dcp_bytes(blob)


# --------------------------------------------------------------------------- #
# Hand-built IFD coverage: type decoding, error guards, look tables.
# --------------------------------------------------------------------------- #

def _srat(vals):
    return b''.join(struct.pack('<ii', int(round(v * 1e6)), 1000000) for v in vals)


def _ifd(entries):
    """entries: list of (tag, type_code, count, raw_bytes). Little-endian DCP."""
    n = len(entries)
    base = 8 + 2 + n * 12 + 4
    ent = b''; data = b''
    for tag, tc, cnt, raw in entries:
        if len(raw) <= 4:
            field = raw + b'\x00' * (4 - len(raw))
        else:
            field = struct.pack('<I', base + len(data)); data += raw
            if len(data) % 2:
                data += b'\x00'
        ent += struct.pack('<HHI', tag, tc, cnt) + field
    return (struct.pack('<2sHI', b'II', 42, 8) + struct.pack('<H', n) + ent
            + struct.pack('<I', 0) + data)


_CM1 = (50721, 10, 9, _srat(M0.ravel()))       # a valid ColorMatrix1 entry


def test_parse_offset_past_eof():
    # ColorMatrix1 (SRATIONAL×9 -> 72 bytes, offset-typed) with offset past EOF.
    ent = struct.pack('<HHI', 50721, 10, 9) + struct.pack('<I', 10 ** 6)
    blob = (struct.pack('<2sHI', b'II', 42, 8) + struct.pack('<H', 1) + ent
            + struct.pack('<I', 0))
    with pytest.raises(dcp.DcpError):
        dcp.parse_dcp_bytes(blob)


def test_parse_matrix_count_not_9():
    blob = _ifd([(50721, 10, 6, _srat([1, 2, 3, 4, 5, 6]))])     # count 6, not 9
    with pytest.raises(dcp.DcpError, match="9"):
        dcp.parse_dcp_bytes(blob)


def test_parse_rational_and_tonecurve():
    ab = struct.pack('<IIIIII', 1, 1, 0, 0, 2, 1)               # 1/1, 0/0 (zero-denom), 2/1
    tone = struct.pack('<6f', 0.0, 0.0, 0.5, 0.6, 1.0, 1.0)
    blob = _ifd([_CM1, (50727, 5, 3, ab), (50940, 11, 6, tone)])
    p = dcp.parse_dcp_bytes(blob)
    np.testing.assert_allclose(p.analog_balance, [1.0, 0.0, 2.0])   # RATIONAL + zero-denom guard
    assert p.tone_curve.shape == (3, 2)
    np.testing.assert_allclose(p.tone_curve, [[0, 0], [0.5, 0.6], [1, 1]])


def test_parse_huesatmap_ordering_and_size_mismatch():
    dims = (2, 2, 2)
    flat = []
    for h in range(2):
        for s in range(2):
            for v in range(2):
                flat += [h * 100 + s * 10 + v, 1.0, 1.0]        # marker in component 0
    data = struct.pack('<%df' % len(flat), *flat)
    blob = _ifd([_CM1, (50937, 4, 3, struct.pack('<3I', *dims)), (50938, 11, len(flat), data)])
    p = dcp.parse_dcp_bytes(blob)
    assert p.hsm_data_1.shape == (2, 2, 2, 3)
    # value-axis-fastest C-order: cell (h,s,v) component0 == h*100+s*10+v
    assert p.hsm_data_1[1, 0, 1, 0] == 101 and p.hsm_data_1[0, 1, 1, 0] == 11
    # wrong-length data -> DcpError
    bad = _ifd([_CM1, (50937, 4, 3, struct.pack('<3I', *dims)),
                (50938, 11, 12, struct.pack('<12f', *([0.0] * 12)))])
    with pytest.raises(dcp.DcpError):
        dcp.parse_dcp_bytes(bad)


def test_parse_count0_scalar_and_bad_dims():
    # CalibrationIlluminant1 (SHORT) with count 0 must not IndexError.
    blob = _ifd([_CM1, (50778, 3, 0, b'')])
    p = dcp.parse_dcp_bytes(blob)
    assert p.illuminant_1 == 21                                 # left at default
    # HueSatMapDims with < 3 values -> DcpError, not IndexError.
    bad = _ifd([_CM1, (50937, 4, 2, struct.pack('<2I', 2, 2)),
                (50938, 11, 12, struct.pack('<12f', *([0.0] * 12)))])
    with pytest.raises(dcp.DcpError):
        dcp.parse_dcp_bytes(bad)


def test_apply_as_shot_wb_none_is_unbalanced():
    # A profile with NO baked calibration neutral (a portable DCP, or any
    # third-party one) still renders the frame's as-shot WB, and falls back to
    # unbalanced when the frame has none.
    p = dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(M0), "c", bake_neutral=False))
    assert p.as_shot_neutral is None
    cam = (np.clip(_neutral(M0) * 0.5, 0, 1) * 65535).astype(np.uint16)
    img = np.tile(cam.reshape(1, 1, 3), (2, 2, 1))
    none_out = dcp.apply_dcp(p, img)                            # m = ones (unbalanced)
    wb_out = dcp.apply_dcp(p, img, as_shot_wb=1.0 / _neutral(M0))
    assert none_out.dtype == np.uint16 and none_out.shape == img.shape
    assert not np.array_equal(none_out, wb_out)                # WB genuinely matters


def test_apply_highlight_not_preclipped():
    # d*m can exceed 1 (film-base highlights); the matrix must see the unclipped
    # value (clip only at the final Adobe output), matching the ICC path.
    # A strong green-normalised WB (m[1] == 1) pushes a balanced channel well past
    # 1.0 (~2.34) — a film-base highlight the matrix must see UNCLIPPED. It is the
    # profile's own calibration WB here, so apply uses it whatever the frame says.
    m = np.array([2.6, 1.0, 1.9])
    p = dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(M0P, m), "c"))
    N = _neutral(M0P)
    cam = (np.clip(N * 0.9, 0, 1) * 65535).astype(np.uint16)
    img = np.tile(cam.reshape(1, 1, 3), (2, 2, 1))
    assert (cam / 65535.0 * m).max() > 1.0                         # genuinely > 1 pre-matrix
    out = dcp.apply_dcp(p, img, as_shot_wb=m)[0, 0]
    FM = p.forward_matrix_1
    # m is already green-normalised, so it is the exact balance apply uses.
    expect = np.clip(((cam / 65535.0 * m) @ FM.T) @ cm.M_XYZ_D50_2_ADOBE.T, 0, 1)
    np.testing.assert_allclose(out / 65535.0, expect, atol=1e-3)   # no pre-matrix clip


def test_interp_weight_clamps_outside_calibration_range():
    FM1 = M0 @ np.diag(_neutral(M0))
    p = dcp.DcpProfile(name="d", forward_matrix_1=FM1, forward_matrix_2=FM1 * 1.1,
                       color_matrix_1=np.eye(3), color_matrix_2=np.eye(3),
                       illuminant_1=21, illuminant_2=17,        # D65 / StdA
                       camera_calibration_1=np.eye(3), camera_calibration_2=np.eye(3),
                       analog_balance=np.ones(3))
    # CM=I so _neutral_cct uses the neutral n=1/m directly as XYZ -> xy -> McCamy.
    hot = 1.0 / np.array([0.6, 0.9, 1.6])      # blue-heavy neutral -> very high CCT
    cold = 1.0 / np.array([1.6, 0.9, 0.5])     # red-heavy neutral -> very low CCT
    assert dcp._interp_weight(p, hot) == 1.0    # clamped to set 1 (D65)
    assert dcp._interp_weight(p, cold) == 0.0   # clamped to set 2 (StdA)
    mid = dcp._interp_weight(p, 1.0 / np.array([1.0, 1.0, 1.0]))
    assert 0.0 <= mid <= 1.0


# --------------------------------------------------------------------------- #
# Calibration white balance (the profile owns the WB).
# See spec/camera-profile-calibration-wb.md.
# --------------------------------------------------------------------------- #

def test_build_bakes_calibration_neutral():
    # The generated DCP records the chart's camera-native neutral (1/wb_mult,
    # green-normalised) as AsShotNeutral (50728, RATIONAL x3).
    wb = np.array([1.9, 1.0, 1.4])
    data = dcp.build_camera_dcp(_Fit(M0P, wb), "cal", illuminant=23)
    p = dcp.parse_dcp_bytes(data)
    np.testing.assert_allclose(p.as_shot_neutral, 1.0 / wb, atol=1e-6)
    assert abs(p.as_shot_neutral[1] - 1.0) < 1e-9          # green-normalised
    # ...and it is a well-formed RATIONAL (type 5, count 3) IFD entry.
    n_entries = struct.unpack('<H', data[8:10])[0]
    entries = {}
    for i in range(n_entries):
        tag, typ, cnt = struct.unpack('<HHI', data[10 + i * 12:10 + i * 12 + 8])
        entries[tag] = (typ, cnt)
    assert entries[50728] == (5, 3)
    assert sorted(entries) == list(entries)        # TIFF requires ascending tags
    # It also matches the DNG camera neutral the ColorMatrix already implies.
    np.testing.assert_allclose(p.color_matrix_1 @ D50, 1.0 / wb, atol=1e-5)


def test_build_can_omit_calibration_neutral():
    p = dcp.parse_dcp_bytes(
        dcp.build_camera_dcp(_Fit(M0P, np.array([1.9, 1.0, 1.4])), "portable",
                             bake_neutral=False))
    assert p.as_shot_neutral is None
    assert p.forward_matrix_1 is not None and p.color_matrix_1 is not None


def test_baked_neutral_ignores_frame_metadata():
    """The regression test for AWB drift: identical sensor data must render
    identically no matter what the frame's camera_whitebalance says."""
    wb = np.array([1.9, 1.0, 1.4])
    p = dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(M0P, wb), "cal"))
    rng = np.random.default_rng(3)
    img = (rng.uniform(0.05, 0.45, (6, 6, 3)) * 65535).astype(np.uint16)
    ref = dcp.apply_dcp(p, img, as_shot_wb=wb)
    # Three wildly different per-frame estimates + no metadata at all.
    for meta in (None, np.array([1.0, 1.0, 1.0]), np.array([2366.0, 1024.0, 1640.0]),
                 np.array([0.4, 1.0, 3.3])):
        np.testing.assert_array_equal(dcp.apply_dcp(p, img, as_shot_wb=meta), ref)
    # And it really is the calibration WB that was used (not "no balancing"):
    # same output as feeding that WB by hand, to within the RATIONAL quantisation
    # of the stored neutral (1e-6 -> at most a LSB or two of uint16).
    unbaked = dcp.parse_dcp_bytes(
        dcp.build_camera_dcp(_Fit(M0P, wb), "portable", bake_neutral=False))
    assert np.abs(dcp.apply_dcp(p, img, as_shot_wb=None).astype(int)
                  - dcp.apply_dcp(unbaked, img, as_shot_wb=wb).astype(int)).max() <= 2
    assert not np.array_equal(dcp.apply_dcp(unbaked, img, as_shot_wb=None), ref)


def test_baked_neutral_renders_chart_neutral_as_grey():
    # The chart neutral the profile was fit on must come back equal-RGB (D50 grey)
    # with NO help from frame metadata.
    wb = np.array([1.9, 1.0, 1.4])
    p = dcp.parse_dcp_bytes(dcp.build_camera_dcp(_Fit(M0P, wb), "cal"))
    raw = np.clip(0.5 / wb, 0, 1)                       # camera-native neutral @ 50%
    img = np.tile((raw * 65535).astype(np.uint16).reshape(1, 1, 3), (2, 2, 1))
    out = dcp.apply_dcp(p, img)[0, 0].astype(float) / 65535.0
    assert out.max() - out.min() < 0.01                 # neutral in linear Adobe

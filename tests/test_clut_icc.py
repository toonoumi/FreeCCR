#!/usr/bin/env python3
"""
Tests for LUT-based (cLUT / A2B) ICC support in core.color_management — parse
(mft1/mft2/mAB), tetrahedral interpolation, PCS XYZ/Lab, reject unsupported, and
the generate (residual cLUT) round-trip. See spec/clut-icc-support.md §8.
Pure numpy/struct — no Qt.
"""
import os
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import color_management as cm        # noqa: E402
from core import it8_profile as it8            # noqa: E402

D50 = np.array(cm.D50_XYZ)


# --------------------------------------------------------------------------- #
# Helpers: wrap an A2B element in a minimal valid RGB ICC; build mft1/mft2/mAB.
# --------------------------------------------------------------------------- #

def _wrap(element: bytes, pcs=b'XYZ ', version=0x02400000) -> bytes:
    tags = {'desc': cm._desc_type('t'), 'A2B0': element,
            'wtpt': cm._xyz_type(*cm.D50_XYZ), 'cprt': cm._text_type('x')}
    order = ['desc', 'A2B0', 'wtpt', 'cprt']
    n = len(order); base = 128 + 4 + n * 12
    data = b''; entries = []; seen = {}
    for name in order:
        payload = tags[name]; key = bytes(payload)
        if key in seen:
            off, ln = seen[key]
        else:
            if len(data) % 4:
                data += b'\x00' * (4 - len(data) % 4)
            off = base + len(data); ln = len(payload); data += payload
            seen[key] = (off, ln)
        entries.append((name.encode(), off, ln))
    table = struct.pack('>I', n)
    for tag, off, ln in entries:
        table += struct.pack('>4sII', tag, off, ln)
    h = bytearray(128)
    struct.pack_into('>I', h, 0, 128 + len(table) + len(data))
    struct.pack_into('>I', h, 8, version)
    struct.pack_into('>4s', h, 12, b'mntr')
    struct.pack_into('>4s', h, 16, b'RGB ')
    struct.pack_into('>4s', h, 20, pcs)
    struct.pack_into('>4s', h, 36, b'acsp')
    return bytes(h) + table + data


def _grid_vals(grid, fn):
    ax = np.linspace(0.0, 1.0, grid)
    out = np.empty((grid, grid, grid, 3))
    for ir in range(grid):
        for ig in range(grid):
            for ib in range(grid):
                out[ir, ig, ib] = fn(ax[ir], ax[ig], ax[ib])
    return out


def _mft1_element(grid, vals01):
    body = struct.pack('>4sI', b'mft1', 0) + struct.pack('>BBBx', 3, 3, grid)
    for v in (1, 0, 0, 0, 1, 0, 0, 0, 1):
        body += struct.pack('>i', cm._s15f16(v))
    ramp = np.rint(np.linspace(0, 255, 256)).astype(np.uint8).tobytes()
    body += ramp * 3
    body += np.rint(np.clip(vals01, 0, 1) * 255).astype(np.uint8).tobytes()
    body += ramp * 3
    return body


def _mft2_element(grid, vals01):
    body = struct.pack('>4sI', b'mft2', 0) + struct.pack('>BBBx', 3, 3, grid)
    for v in (1, 0, 0, 0, 1, 0, 0, 0, 1):
        body += struct.pack('>i', cm._s15f16(v))
    body += struct.pack('>HH', 2, 2)
    for _ in range(3):
        body += struct.pack('>HH', 0, 65535)
    body += np.rint(np.clip(vals01, 0, 1) * 65535).astype('>u2').tobytes()
    for _ in range(3):
        body += struct.pack('>HH', 0, 65535)
    return body


def _mAB_element(grid, vals01, *, matrix=None):
    """mAB with identity A-curves + CLUT (+ optional 3x4 matrix on the output)."""
    curv0 = struct.pack('>4sII', b'curv', 0, 0)          # identity 'curv', 12 bytes
    A = curv0 * 3
    off_A = 32
    off_clut = off_A + len(A)
    gp = bytes([grid] * 3) + bytes(13)                   # gridPoints[16]
    clut = gp + bytes([2, 0, 0, 0])                      # u16 precision + pad
    clut += np.rint(np.clip(vals01, 0, 1) * 65535).astype('>u2').tobytes()
    off_mat = 0
    mat_bytes = b''
    if matrix is not None:
        off_mat = off_clut + len(clut)
        m = np.asarray(matrix, float).ravel()            # 12 (3x3 + offset)
        mat_bytes = b''.join(struct.pack('>i', cm._s15f16(v)) for v in m)
    body = struct.pack('>4sI', b'mAB ', 0) + struct.pack('>BB2x', 3, 3)
    body += struct.pack('>5I', 0, off_mat, 0, off_clut, off_A)   # B,mat,M,CLUT,A
    body += A + clut + mat_bytes
    return body


# --------------------------------------------------------------------------- #
# Parse + interpolation.
# --------------------------------------------------------------------------- #

M0 = np.array([[0.46, 0.31, 0.17], [0.23, 0.70, 0.07], [0.02, 0.12, 0.93]])


def _affine_xyz(r, g, b):
    return M0 @ np.array([r, g, b])              # XYZ D50, Y=1, in range for grid<=1


def _parse(element, pcs=b'XYZ ', version=0x02400000):
    return cm.InputProfile.from_bytes(_wrap(element, pcs, version))


def test_parse_mft2_affine_roundtrip():
    g = 9
    vals = _grid_vals(g, lambda r, gg, b: _affine_xyz(r, gg, b) / cm._XYZ_ENC)
    prof = _parse(_mft2_element(g, vals))
    assert prof._kind == 'clut' and prof._clut.grid == (g, g, g)
    rng = np.random.default_rng(1)
    dev = (rng.uniform(0, 1, (8, 8, 3)) * 65535).astype(np.uint16)
    out = prof.apply(dev)
    d01 = dev.astype(np.float64) / 65535.0
    expect = np.clip(d01 @ (cm.M_XYZ_D50_2_ADOBE @ M0).T, 0, 1)
    err = np.abs(out.astype(int) - np.rint(expect * 65535).astype(int)).max()
    assert err < 12                              # tetra affine-exact; err = u16 quantisation


def test_parse_mft1_8bit():
    g = 7
    vals = _grid_vals(g, lambda r, gg, b: _affine_xyz(r, gg, b) / cm._XYZ_ENC)
    prof = _parse(_mft1_element(g, vals))
    assert prof._kind == 'clut'
    # 8-bit storage is coarse; the neutral axis should still be near-neutral XYZ.
    xyz = prof._clut.table[g // 2, g // 2, g // 2]
    assert xyz[1] > 0                            # non-zero Y at mid-grey


def test_parse_mAB_with_matrix():
    g = 5
    # CLUT outputs 0.5*the value; the mAB matrix (2*I) doubles it -> affine map.
    vals = _grid_vals(g, lambda r, gg, b: 0.5 * (M0 @ np.array([r, gg, b])) / cm._XYZ_ENC)
    # ICC mAB matrix element = 9 row-major 3x3 values THEN 3 offset values.
    mat12 = np.concatenate([np.diag([2.0, 2.0, 2.0]).ravel(), [0.0, 0.0, 0.0]])
    prof = _parse(_mAB_element(g, vals, matrix=mat12))
    # corner (1,1,1): CLUT 0.5*M0@1 /_XYZ_ENC, *2 (matrix), *_XYZ_ENC (decode) = M0@1
    got = prof._clut.table[-1, -1, -1]
    np.testing.assert_allclose(got, M0 @ np.ones(3), atol=2e-3)


def test_tetra_affine_exact_and_neutral():
    g = 9
    vals = _grid_vals(g, lambda r, gg, b: _affine_xyz(r, gg, b) / cm._XYZ_ENC)
    clut = _parse(_mft2_element(g, vals))._clut
    q = np.random.default_rng(2).uniform(0, 1, (200, 3)).astype(np.float32)
    tetra = cm._clut_interp_tetra(q, clut)
    tri = cm._clut_interp_trilinear(q, clut)
    # affine -> both exact; tetra ~ trilinear
    np.testing.assert_allclose(tetra, q @ M0.T, atol=2e-3)
    np.testing.assert_allclose(tetra, tri, atol=2e-3)


def test_pcs_lab_v4_decodes_to_xyz():
    g = 6
    def lab01(r, gg, b):
        xyz = _affine_xyz(r, gg, b)                       # Y=1
        L, a, bb = it8.xyz_to_lab(xyz * 100.0)
        return np.array([L / 100.0, (a + 128) / 255.0, (bb + 128) / 255.0])
    vals = _grid_vals(g, lab01)
    prof = _parse(_mft2_element(g, vals), pcs=b'Lab ', version=0x04200000)
    # table is decoded XYZ(D50,Y=1); compare to the affine map at a node
    node = prof._clut.table[-1, -1, -1]
    np.testing.assert_allclose(node, _affine_xyz(1, 1, 1), atol=2e-3)


# --------------------------------------------------------------------------- #
# Rejection.
# --------------------------------------------------------------------------- #

def test_reject_cmyk():
    g = 5
    vals = _grid_vals(g, lambda r, gg, b: _affine_xyz(r, gg, b) / cm._XYZ_ENC)
    icc = bytearray(_wrap(_mft2_element(g, vals)))
    icc[16:20] = b'CMYK'
    with pytest.raises(cm.UnsupportedICCError):
        cm.InputProfile.from_bytes(bytes(icc))


def test_reject_unsupported_pcs():
    g = 5
    vals = _grid_vals(g, lambda r, gg, b: _affine_xyz(r, gg, b) / cm._XYZ_ENC)
    with pytest.raises(cm.UnsupportedICCError):
        cm.InputProfile.from_bytes(_wrap(_mft2_element(g, vals), pcs=b'FOO '))


def test_reject_degenerate_grid_mft():
    # g < 2 in mft1/mft2 must be rejected at PARSE time (was a deferred IndexError).
    for builder in (_mft1_element, _mft2_element):
        vals = np.zeros((1, 1, 1, 3))
        el = builder(1, vals)                      # g = 1
        with pytest.raises(cm.UnsupportedICCError):
            cm.InputProfile.from_bytes(_wrap(el))


def test_reject_non_3channel_and_mAB_guards():
    # i != 3 (e.g. CMYK-style 4-in) rejected by the parser (distinct from the
    # header RGB guard).
    el = bytearray(_mft2_element(4, _grid_vals(4, lambda r, g, b: _affine_xyz(r, g, b) / cm._XYZ_ENC)))
    el[8] = 4                                       # i = 4
    with pytest.raises(cm.UnsupportedICCError):
        cm.InputProfile.from_bytes(_wrap(bytes(el)))
    # mAB without a CLUT (off_clut == 0) rejected.
    g = 4
    vals = _grid_vals(g, lambda r, gg, b: _affine_xyz(r, gg, b) / cm._XYZ_ENC)
    mab = bytearray(_mAB_element(g, vals))
    struct.pack_into('>5I', mab, 12, 0, 0, 0, 0, 32)      # off_clut = 0
    with pytest.raises(cm.UnsupportedICCError):
        cm.InputProfile.from_bytes(_wrap(bytes(mab)))


def test_pcs_lab_v2_16bit_and_mft1_8bit():
    g = 6
    def lab01(r, gg, b, scale):
        xyz = _affine_xyz(r, gg, b)
        L, a, bb = it8.xyz_to_lab(xyz * 100.0)
        return np.array([L / 100.0, (a + 128) / 255.0, (bb + 128) / 255.0]) * scale
    # v2 16-bit legacy: stored so L*=100 -> 0xFF00, i.e. divide by 65535/65280.
    s = 65280.0 / 65535.0
    prof = _parse(_mft2_element(g, _grid_vals(g, lambda r, gg, b: lab01(r, gg, b, s))),
                  pcs=b'Lab ', version=0x02400000)
    np.testing.assert_allclose(prof._clut.table[-1, -1, -1], _affine_xyz(1, 1, 1), atol=3e-3)
    # mft1 8-bit Lab (scale 1.0): coarse but the mid node should be plausible XYZ.
    p8 = _parse(_mft1_element(g, _grid_vals(g, lambda r, gg, b: lab01(r, gg, b, 1.0))),
                pcs=b'Lab ', version=0x02400000)
    assert p8._clut.table[g // 2, g // 2, g // 2][1] > 0


def test_tetra_keeps_neutral_where_trilinear_desaturates():
    # A table neutral ON the diagonal but chroma-shifted OFF it. A diagonal query
    # d=(t,t,t): tetra interpolates only the two neutral diagonal corners (stays
    # neutral); trilinear weights all 8 corners incl. the chroma off-diagonal ones
    # (desaturates). This is the property the tetra path exists for (spec §5.2).
    g = 5
    ax = np.linspace(0, 1, g)
    tbl = np.empty((g, g, g, 3))
    for i in range(g):
        for j in range(g):
            for k in range(g):
                v = np.array([ax[i], ax[j], ax[k]])
                m = v.mean(); spread = v.max() - v.min()
                tbl[i, j, k] = m + 0.3 * spread * np.array([1.0, -0.5, -0.5])
    clut = cm._CLUT(grid=(g, g, g), table=tbl.astype(np.float32),
                    in_luts=[np.linspace(0, 1, 65536).astype(np.float32)] * 3)
    diag = np.stack([np.linspace(0.05, 0.95, 20)] * 3, axis=1).astype(np.float32)
    te = cm._clut_interp_tetra(diag, clut)
    tr = cm._clut_interp_trilinear(diag, clut)
    assert float(np.abs(te.max(1) - te.min(1)).max()) < 1e-4    # tetra: neutral preserved
    assert float(np.abs(tr.max(1) - tr.min(1)).max()) > 1e-2    # trilinear: desaturates


def test_tetra_chunk_boundary(monkeypatch):
    # Exercise the multi-chunk loop cheaply by shrinking the chunk size.
    g = 5
    vals = _grid_vals(g, lambda r, gg, b: _affine_xyz(r, gg, b) / cm._XYZ_ENC)
    clut = _parse(_mft2_element(g, vals))._clut
    monkeypatch.setattr(cm, "_TETRA_CHUNK", 50)
    q = np.random.default_rng(5).uniform(0, 1, (137, 3)).astype(np.float32)   # > 2 chunks
    out = cm._clut_interp_tetra(q, clut)
    np.testing.assert_allclose(out, q @ M0.T, atol=2e-3)        # boundary rows correct


def test_residual_clut_bounded_ringing():
    samples, ref = _nonaffine_camera()
    fit = it8.fit_camera_matrix(samples, ref)
    tbl = it8.build_residual_clut(fit, samples, ref, grid=17)
    ax = np.linspace(0, 1, 17)
    base = (np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), -1).reshape(-1, 3)
            @ fit.matrix.T).reshape(17, 17, 17, 3)
    assert it8._residual_monotone(base, tbl)                    # no reversal > RINGING_TOL


def test_reject_b2a_only():
    # An RGB profile with neither matrix-shaper tags nor an A2B tag.
    h = bytearray(128)
    struct.pack_into('>4s', h, 16, b'RGB ')
    struct.pack_into('>4s', h, 20, b'XYZ ')
    struct.pack_into('>4s', h, 36, b'acsp')
    table = struct.pack('>I', 1) + struct.pack('>4sII', b'B2A0', 128 + 16, 4) + b'\x00' * 4
    with pytest.raises(cm.UnsupportedICCError):
        cm.InputProfile.from_bytes(bytes(h) + table)


# --------------------------------------------------------------------------- #
# Matrix-shaper path unchanged (regression).
# --------------------------------------------------------------------------- #

def test_matrix_shaper_path_still_matrix():
    adapted = cm.M_BRADFORD_D65_D50 @ cm.M_SRGB2XYZ
    icc = cm.build_matrix_shaper_icc(
        'm', tuple(adapted[:, 0]), tuple(adapted[:, 1]), tuple(adapted[:, 2]),
        (1.0, 1.0, 0.0, 1.0, 0.0))
    prof = cm.InputProfile.from_bytes(icc)
    assert prof._kind == 'matrix' and prof._clut is None
    img = (np.random.default_rng(3).uniform(0, 1, (4, 4, 3)) * 65535).astype(np.uint16)
    assert prof.apply(img).shape == img.shape


# --------------------------------------------------------------------------- #
# Generate: build_clut_icc round-trip + residual cLUT beats the 3x3 matrix.
# --------------------------------------------------------------------------- #

def test_build_clut_icc_reparses():
    g = 9
    clut = _grid_vals(g, _affine_xyz)
    icc = cm.build_clut_icc('gen', clut, g)
    assert icc[36:40] == b'acsp' and icc[16:20] == b'RGB ' and icc[20:24] == b'XYZ '
    prof = cm.InputProfile.from_bytes(icc)
    assert prof._kind == 'clut' and prof._clut.grid == (g, g, g)


def _nonaffine_camera():
    """Synthetic patches whose device->XYZ is non-affine, so a 3x3 leaves
    residuals a cLUT can remove."""
    patches = {}
    for k, Y in enumerate(np.linspace(89, 3, 24)):
        patches[f'GS{k}'] = (Y / 100.0) * D50 * 100.0
    rng = np.random.default_rng(7)
    for cid in it8.COLOR_IDS:
        Y = rng.uniform(5, 85)
        patches[cid] = np.array([Y * rng.uniform(0.7, 1.3), Y, Y * rng.uniform(0.4, 1.2)])
    Minv = np.linalg.inv(M0)
    samples = {}
    for sid, xyz in patches.items():
        d = np.clip(Minv @ (xyz / 100.0), 0, None)
        d = d + 0.06 * (d ** 2 - d)                       # non-affine distortion
        samples[sid] = it8.PatchSample(np.clip(d, 0, 1) * 65535.0, True, 900)
    ref = it8.IT8Reference(batch='S')
    for sid, xyz in patches.items():
        lab = it8.xyz_to_lab(xyz)
        ref.patches[sid] = {'XYZ_X': xyz[0], 'XYZ_Y': xyz[1], 'XYZ_Z': xyz[2],
                            'LAB_L': lab[0], 'LAB_A': lab[1], 'LAB_B': lab[2]}
    return samples, ref


def _patch_des(prof, fit, samples, ref):
    # NEW convention: the profile (matrix or cLUT) consumes WHITE-BALANCED device,
    # so the raw samples must be balanced by the fit's green-normalised WB before
    # the profile — otherwise both dEs are computed in the wrong space. Passing
    # as_shot_wb=fit.wb_mult reproduces what apply does on a real frame.
    Ainv = np.linalg.inv(cm.M_XYZ_D50_2_ADOBE)
    de = []
    for sid in fit.used_ids:
        dev = np.clip(samples[sid].rgb, 0, 65535).astype(np.uint16).reshape(1, 1, 3)
        out = prof.apply(np.tile(dev, (2, 2, 1)),
                         as_shot_wb=fit.wb_mult)[0, 0].astype(np.float64) / 65535.0
        xyz = (Ainv @ out) * 100.0
        de.append(float(it8.delta_e_2000(it8.xyz_to_lab(xyz), ref.lab(sid))[0]))
    return float(np.mean(de)), float(np.max(de))


def test_clut_beats_matrix_on_nonaffine_camera():
    samples, ref = _nonaffine_camera()
    fit = it8.fit_camera_matrix(samples, ref)
    mp = cm.InputProfile.from_bytes(it8.build_camera_icc(fit, 'm', mode='matrix'))
    cp = cm.InputProfile.from_bytes(
        it8.build_camera_icc(fit, 'c', mode='clut', grid=17, samples=samples, ref=ref))
    m_avg, m_max = _patch_des(mp, fit, samples, ref)
    c_avg, c_max = _patch_des(cp, fit, samples, ref)
    # The residual cLUT still markedly beats the bare 3x3 (clut avg ~1.7 vs
    # matrix ~2.3 dE on this synthetic). Threshold relaxed from 0.6x to 0.8x:
    # in the standards-compliant WHITE-BALANCED space the 3x3 already fits better
    # than it did against un-balanced raw, so the cLUT's relative gain is smaller
    # (~25%) but still clearly present and never worse on the max.
    assert c_avg < 0.8 * m_avg                            # markedly lower residual
    assert c_max <= m_max


def test_clut_brightness_tracks_matrix_not_blown():
    """Regression: the cLUT must reproduce at the SAME exposure as the 3x3 (cLUT
    = matrix + small colour residual). A prior bug scaled the cLUT base by 1/n1
    (~6.6x) -> blown highlights / lost colour. The neutral patch must come out at
    matrix brightness, never multiples of it."""
    samples, ref = _nonaffine_camera()
    fit = it8.fit_camera_matrix(samples, ref)
    mp = cm.InputProfile.from_bytes(it8.build_camera_icc(fit, 'm', mode='matrix'))
    cp = cm.InputProfile.from_bytes(
        it8.build_camera_icc(fit, 'c', mode='clut', grid=17, samples=samples, ref=ref))
    # The lightest neutral (the fit's wb_id) — cLUT vs matrix output, balanced.
    dev = np.clip(samples[fit.wb_id].rgb, 0, 65535).astype(np.uint16).reshape(1, 1, 3)
    m_out = mp.apply(dev, as_shot_wb=fit.wb_mult)[0, 0].astype(np.float64)
    c_out = cp.apply(dev, as_shot_wb=fit.wb_mult)[0, 0].astype(np.float64)
    ratio = c_out / np.maximum(m_out, 1.0)
    assert np.allclose(ratio, 1.0, atol=0.1), f"cLUT neutral not at matrix brightness: {ratio}"


def test_build_camera_icc_clut_requires_samples():
    samples, ref = _nonaffine_camera()
    fit = it8.fit_camera_matrix(samples, ref)
    with pytest.raises(ValueError):
        it8.build_camera_icc(fit, 'c', mode='clut')      # no samples/ref


def test_clut_icc_bakes_calibration_neutral():
    """The cLUT container records the calibration neutral too, so a cLUT profile
    is as WB-stable as the matrix one. See spec/camera-profile-calibration-wb.md."""
    samples, ref = _nonaffine_camera()
    fit = it8.fit_camera_matrix(samples, ref)
    cp = cm.InputProfile.from_bytes(
        it8.build_camera_icc(fit, 'c', mode='clut', grid=17, samples=samples, ref=ref))
    assert cp.calibration_neutral is not None
    np.testing.assert_allclose(cp.calibration_neutral, 1.0 / fit.wb_mult, atol=1e-3)
    dev = np.tile(np.clip(samples[fit.wb_id].rgb, 0, 65535)
                  .astype(np.uint16).reshape(1, 1, 3), (3, 3, 1))
    base = cp.apply(dev, as_shot_wb=fit.wb_mult)
    for meta in (None, np.array([2366.0, 1024.0, 1640.0]), np.array([0.4, 1.0, 3.3])):
        np.testing.assert_array_equal(cp.apply(dev, as_shot_wb=meta), base)

#!/usr/bin/env python3
"""
Tests for the IT8 camera-profile core (src/core/it8_profile.py).
See spec/it8-camera-profile.md §8. Pure numpy/cv2 — no Qt required.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core import it8_profile as it8  # noqa: E402
from core import color_management as cm  # noqa: E402

D50 = np.array(cm.D50_XYZ)                 # [0,1] scale, Y=1


# --------------------------------------------------------------------------- #
# Reference-file fixtures (CGATS.17), built programmatically.
# --------------------------------------------------------------------------- #

def _make_cgats(fields, rows, *, chart="IT8.7/1", serial="E111007 Batch",
                eol="\n", bom=False, comment=True):
    """Build a minimal CGATS file string. `rows` = list of (sample_id, {field:val})."""
    lines = [chart,
             'ORIGINATOR "Wolf Faust"',
             'DESCRIPTOR "IT8.7/1 light D50, 2 degree"',
             f'MANUFACTURER "Wolf Faust - http://www.coloraid.de"',
             f'SERIAL "{serial}"',
             f'NUMBER_OF_FIELDS {len(fields)}',
             f'NUMBER_OF_SETS {len(rows)}']
    if comment:
        lines.append('KEYWORD "MEAN_DE" # a custom column name')
    lines.append('BEGIN_DATA_FORMAT')
    lines.append(' '.join(fields))
    lines.append('END_DATA_FORMAT')
    lines.append('BEGIN_DATA')
    for sid, vals in rows:
        toks = [sid] + [f"{vals.get(f, 0.0):.4f}" for f in fields[1:]]
        lines.append('   '.join(toks))
    lines.append('END_DATA')
    text = eol.join(lines) + eol
    if bom:
        text = '﻿' + text
    return text


def _synthetic_patches():
    """288 patches: GS0..GS23 neutral (Y ramp * D50), colour patches varied.
    Returns dict id -> XYZ (Y=100 scale)."""
    rng = np.random.default_rng(7)
    patches = {}
    # Grayscale: neutral, Y from ~89 down to ~1.
    ys = np.linspace(89.0, 1.0, 24)
    for k, y in enumerate(ys):
        patches[f"GS{k}"] = (y / 100.0) * D50 * 100.0   # exactly D50 chromaticity
    # Colour: random but plausible XYZ (Y 5..85), kept in a sane range.
    for cid in it8.COLOR_IDS:
        Y = rng.uniform(5, 85)
        x = Y * rng.uniform(0.7, 1.3)
        z = Y * rng.uniform(0.4, 1.2)
        patches[cid] = np.array([x, Y, z])
    return patches


def _write(tmp_path, name, text):
    p = tmp_path / name
    # utf-8 so a BOM ('﻿') round-trips to the EF BB BF the parser strips;
    # the rest of the fixture is ASCII.
    p.write_bytes(text.encode('utf-8'))
    return str(p)


def test_parse_18_field(tmp_path):
    fields = ["SAMPLE_ID", "XYZ_X", "XYZ_Y", "XYZ_Z", "LAB_L", "LAB_A", "LAB_B",
              "D_RED", "D_GREEN", "D_BLUE", "D_VIS", "STDEV_X", "STDEV_Y",
              "STDEV_Z", "MEAN_DE", "STDEV_DE"]
    rows = [("A1", {"XYZ_X": 2.26, "XYZ_Y": 1.91, "XYZ_Z": 1.32,
                    "LAB_L": 15.02, "LAB_A": 9.49, "LAB_B": 3.12}),
            ("GS0", {"XYZ_X": 79.8, "XYZ_Y": 82.7, "XYZ_Z": 63.6,
                     "LAB_L": 92.9, "LAB_A": 0.1, "LAB_B": 0.2}),
            ("L22", {"XYZ_X": 5.0, "XYZ_Y": 4.0, "XYZ_Z": 3.0,
                     "LAB_L": 23.0, "LAB_A": 1.0, "LAB_B": -2.0})]
    ref = it8.parse_it8_reference(_write(tmp_path, "ref18.it8",
                                         _make_cgats(fields, rows)))
    assert ref.chart_type == "IT8.7/1"
    assert ref.batch.startswith("E111007")
    assert ref.fields[0] == "SAMPLE_ID"
    assert "A1" in ref.patches and "GS0" in ref.patches and "L22" in ref.patches
    np.testing.assert_allclose(ref.xyz("A1"), [2.26, 1.91, 1.32])
    np.testing.assert_allclose(ref.lab("GS0"), [92.9, 0.1, 0.2])


def test_parse_17_field_no_xyz_derives_from_lab(tmp_path):
    # Older layout: Lab present, plus STDEV_LAB, no density columns.
    fields = ["SAMPLE_ID", "XYZ_X", "XYZ_Y", "XYZ_Z", "LAB_L", "LAB_A", "LAB_B",
              "LAB_C", "LAB_H", "STDEV_L", "STDEV_A", "STDEV_B"]
    rows = [("A1", {"XYZ_X": 20.0, "XYZ_Y": 18.0, "XYZ_Z": 10.0,
                    "LAB_L": 49.5, "LAB_A": 8.0, "LAB_B": 12.0})]
    ref = it8.parse_it8_reference(_write(tmp_path, "ref17.it8",
                                         _make_cgats(fields, rows)))
    assert len(ref.fields) == 12
    np.testing.assert_allclose(ref.lab("A1"), [49.5, 8.0, 12.0])


def test_parse_id_normalization_and_crlf_bom(tmp_path):
    fields = ["SAMPLE_ID", "XYZ_X", "XYZ_Y", "XYZ_Z"]
    rows = [("A01", {"XYZ_X": 1, "XYZ_Y": 2, "XYZ_Z": 3}),
            ("GS00", {"XYZ_X": 80, "XYZ_Y": 82, "XYZ_Z": 64}),
            ("gs23", {"XYZ_X": 0.1, "XYZ_Y": 0.1, "XYZ_Z": 0.1})]
    ref = it8.parse_it8_reference(_write(
        tmp_path, "refn.it8", _make_cgats(fields, rows, eol="\r\n", bom=True)))
    assert "A1" in ref.patches          # zero-padded -> normalized
    assert "GS0" in ref.patches and "GS23" in ref.patches


def test_parse_errors(tmp_path):
    with pytest.raises(it8.IT8ReferenceError):
        it8.parse_it8_reference(_write(tmp_path, "empty.it8", "IT8.7/1\n"))
    # No colorimetry columns at all.
    bad = _make_cgats(["SAMPLE_ID", "D_VIS"], [("A1", {"D_VIS": 1.0})])
    with pytest.raises(it8.IT8ReferenceError):
        it8.parse_it8_reference(_write(tmp_path, "nocol.it8", bad))


def test_parse_skips_count_mismatched_rows(tmp_path):
    # A row truncated mid-way must NOT become a partial record (which would
    # later KeyError in lab()); it is skipped, valid rows still parse.
    fields = ["SAMPLE_ID", "XYZ_X", "XYZ_Y", "XYZ_Z", "LAB_L", "LAB_A", "LAB_B"]
    good = _make_cgats(fields, [
        ("A1", {"XYZ_X": 2.0, "XYZ_Y": 1.9, "XYZ_Z": 1.3,
                "LAB_L": 15.0, "LAB_A": 9.0, "LAB_B": 3.0}),
        ("GS0", {"XYZ_X": 80.0, "XYZ_Y": 82.0, "XYZ_Z": 64.0,
                 "LAB_L": 92.0, "LAB_A": 0.0, "LAB_B": 0.0})])
    # Splice in a truncated A2 row (only 3 tokens) before the DATA terminator.
    # Match "END_DATA\n" (not the "END_DATA_FORMAT" header) so only the data
    # block's closing line is targeted.
    text = good.replace("END_DATA\n", "A2   5.00   4.00\nEND_DATA\n")
    ref = it8.parse_it8_reference(_write(tmp_path, "trunc.it8", text))
    assert "A1" in ref.patches and "GS0" in ref.patches
    assert "A2" not in ref.patches            # malformed row skipped
    # No partial record => xyz()/lab() never KeyError.
    assert ref.xyz("A1") is not None and ref.lab("GS0") is not None


def test_decode_target_smoke(tmp_path):
    # decode_target builds a bare CCRImage via __new__ and calls read_image —
    # verify that path returns a uint16 RGB array (guards the __init__ bypass).
    import cv2
    bgr = np.zeros((40, 60, 3), dtype=np.uint8)
    bgr[..., 2] = 200          # R via BGR
    path = str(tmp_path / "target.png")
    cv2.imwrite(path, bgr)
    arr = it8.decode_target(path, sample_max=2048)
    assert arr is not None and arr.dtype == np.uint16
    assert arr.ndim == 3 and arr.shape[2] == 3
    # RGB order: red channel dominant (input ICC bypassed, no colour transform).
    assert arr[0, 0, 0] > arr[0, 0, 1] and arr[0, 0, 0] > arr[0, 0, 2]


def test_partial_record_does_not_crash():
    # Even if a partial record somehow exists, xyz()/lab() return None, not raise.
    ref = it8.IT8Reference()
    ref.patches["A1"] = {"XYZ_X": 5.0, "XYZ_Y": 4.0}   # missing XYZ_Z
    assert ref.xyz("A1") is None
    assert ref.lab("A1") is None


# --------------------------------------------------------------------------- #
# CxF reference parsing.
# --------------------------------------------------------------------------- #

def _make_cxf(rows, *, prefix="", british=True):
    """Build a minimal CxF3 file. prefix '' = default namespace, 'cc:' =
    prefixed. british=True uses LaserSoft spelling (Colour*/ColourIEXYZ)."""
    def t(name):
        return f"{prefix}{name}"
    XYZ = "ColourIEXYZ" if british else "ColorCIEXYZ"
    LAB = "ColourCIELab" if british else "ColorCIELab"
    CV = "ColourValues" if british else "ColorValues"
    SREF = "ColourSpecification" if british else "ColorSpecification"
    SCOL = "ColourSpecificationCollection" if british else "ColorSpecificationCollection"
    SPEC = "ColourSpecification" if british else "ColorSpecification"
    nsdecl = (f'xmlns:{prefix[:-1]}="http://colorexchangeformat.com/CxF3-core"'
              if prefix else 'xmlns="http://colorexchangeformat.com/CxF3-core"')
    objs = []
    for name, (X, Y, Z), (L, A, B) in rows:
        objs.append(
            f'<{t("Object")} ObjectType="Target" Name="{name}" Id="{name}">'
            f'<{t(CV)}>'
            f'<{t(XYZ)} {SREF}="CS1"><{t("X")}>{X}</{t("X")}>'
            f'<{t("Y")}>{Y}</{t("Y")}><{t("Z")}>{Z}</{t("Z")}></{t(XYZ)}>'
            f'<{t(LAB)} {SREF}="CS1"><{t("L")}>{L}</{t("L")}>'
            f'<{t("A")}>{A}</{t("A")}><{t("B")}>{B}</{t("B")}></{t(LAB)}>'
            f'</{t(CV)}></{t("Object")}>')
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<{t("CxF")} {nsdecl}>'
        f'<{t("FileInformation")}>'
        f'<{t("Creator")}>LaserSoft Imaging AG</{t("Creator")}>'
        f'<{t("Tag")} Name="28178_TargetType" Value="ISO 12641-2 Candidate"/>'
        f'<{t("Tag")} Name="Serial" Value="E210616"/>'
        f'</{t("FileInformation")}>'
        f'<{t("Resources")}><{t("ObjectCollection")}>{"".join(objs)}'
        f'</{t("ObjectCollection")}>'
        f'<{t(SCOL)}><{t(SPEC)} Id="CS1"><{t("TristimulusSpec")}>'
        f'<{t("Illuminant")}>D50</{t("Illuminant")}>'
        f'<{t("Observer")}>2_Degree</{t("Observer")}>'
        f'</{t("TristimulusSpec")}></{t(SPEC)}></{t(SCOL)}>'
        f'</{t("Resources")}></{t("CxF")}>')


def test_parse_cxf_british_default_ns(tmp_path):
    # Matches the user's LaserSoft file: default namespace + British spelling.
    rows = [("A1", (2.819, 2.434, 1.559), (17.62, 9.14, 4.69)),
            ("L72", (5.627, 12.305, 1.759), (41.6, -38.0, 35.0))]
    p = tmp_path / "E210616.cxf"
    p.write_text(_make_cxf(rows, prefix="", british=True), encoding="utf-8")
    ref = it8.parse_reference(str(p))
    assert ref.batch == "E210616"
    assert "ISO 12641-2" in ref.chart_type
    assert len(ref.patches) == 2
    np.testing.assert_allclose(ref.xyz("A1"), [2.819, 2.434, 1.559])
    np.testing.assert_allclose(ref.lab("L72"), [41.6, -38.0, 35.0])


def test_parse_cxf_prefixed_american(tmp_path):
    # X-Rite style: cc: prefix + American spelling. Tolerated by local-name match.
    rows = [("R1", (9.243, 9.910, 5.366), (37.99, 13.56, 14.06))]
    p = tmp_path / "x.cxf"
    p.write_text(_make_cxf(rows, prefix="cc:", british=False), encoding="utf-8")
    ref = it8.parse_reference(str(p))
    assert len(ref.patches) == 1
    np.testing.assert_allclose(ref.xyz("R1"), [9.243, 9.910, 5.366])


def test_parse_reference_dispatch(tmp_path):
    cxf = tmp_path / "ref.cxf"
    cxf.write_text(_make_cxf([("A1", (2, 2, 1), (17, 9, 4))]), encoding="utf-8")
    assert it8.parse_reference(str(cxf)).originator == "LaserSoft Imaging AG"
    # CGATS text still routes to the CGATS parser.
    fields = ["SAMPLE_ID", "XYZ_X", "XYZ_Y", "XYZ_Z"]
    txt = _make_cgats(fields, [("A1", {"XYZ_X": 2, "XYZ_Y": 2, "XYZ_Z": 1})])
    cg = _write(tmp_path, "ref.txt", txt)
    assert "A1" in it8.parse_reference(cg).patches


def test_parse_cxf_errors(tmp_path):
    p = tmp_path / "bad.cxf"
    p.write_text("<NotCxF><x/></NotCxF>", encoding="utf-8")
    with pytest.raises(it8.IT8ReferenceError):
        it8.parse_reference(str(p))


# --------------------------------------------------------------------------- #
# Generic block geometry + fit (non-classic targets, e.g. ISO 12641-2).
# --------------------------------------------------------------------------- #

def test_parse_block():
    rows, cols = it8.parse_block("A49", "L72")
    assert rows == list("ABCDEFGHIJKL")
    assert cols == list(range(49, 73))
    # order-insensitive
    assert it8.parse_block("L72", "A49") == (rows, cols)


def test_next_block_ids_advances_columns():
    # The classic 3-card split: each card shifts by its own column width.
    assert it8.next_block_ids("A1", "L24") == ("A25", "L48")
    assert it8.next_block_ids("A25", "L48") == ("A49", "L72")
    # Rows are preserved; order of the input ids doesn't matter (parse_block sorts).
    assert it8.next_block_ids("L48", "A25") == ("A49", "L72")
    # Clamp the right column to the reference's extent on an uneven last card.
    assert it8.next_block_ids("A49", "L72", max_col=72) == ("A72", "L72")
    assert it8.next_block_ids("A40", "L60", max_col=72) == ("A61", "L72")
    with pytest.raises(ValueError):
        it8.next_block_ids("nope", "L24")


def test_merge_samples_prefers_valid():
    def ps(v, valid):
        return it8.PatchSample(rgb=np.array([float(v)] * 3), valid=valid, n_pix=9)
    card_a = {"A1": ps(1, True), "A2": ps(2, False)}
    card_b = {"A2": ps(3, True), "A3": ps(4, True)}
    merged = it8.merge_samples([card_a, card_b])
    assert set(merged) == {"A1", "A2", "A3"}
    # The valid A2 from card_b overrides the invalid A2 from card_a...
    assert merged["A2"].valid and merged["A2"].rgb[0] == 3.0
    # ...and a later *invalid* sample must not clobber an existing valid one.
    merged2 = it8.merge_samples([card_b, card_a])
    assert merged2["A2"].valid and merged2["A2"].rgb[0] == 3.0


def test_merge_samples_union_covers_all_cards():
    def ps(v):
        return it8.PatchSample(rgb=np.array([float(v)] * 3), valid=True, n_pix=9)
    c1 = {f"A{c}": ps(c) for c in range(1, 25)}      # columns 1-24
    c2 = {f"A{c}": ps(c) for c in range(25, 49)}     # columns 25-48
    c3 = {f"A{c}": ps(c) for c in range(49, 73)}     # columns 49-72
    merged = it8.merge_samples([c1, c2, c3])
    assert len(merged) == 72 and merged["A50"].rgb[0] == 50.0


def test_block_sample_points_unit_quad():
    quad = [(0, 0), (240, 0), (240, 120), (0, 120)]   # TL,TR,BR,BL
    rows, cols = it8.parse_block("A49", "L72")          # 12 rows x 24 cols
    pts = it8.block_sample_points(quad, rows, cols)
    assert len(pts) == 288
    # A49 (row0,col0) centre = (0.5/24*240, 0.5/12*120) = (5,5).
    np.testing.assert_allclose(pts["A49"], (5.0, 5.0), atol=1e-6)
    # L72 (row11,col23) centre = (23.5/24*240, 11.5/12*120) = (235,115).
    np.testing.assert_allclose(pts["L72"], (235.0, 115.0), atol=1e-6)


def test_fit_block_generic_autoneutral():
    # A 12x24 block with in-grid neutrals (no GS strip): auto-neutral WB must
    # find a gray and the fit recover the generator.
    M0 = _pin(np.array([[0.46, 0.31, 0.17],
                        [0.23, 0.70, 0.07],
                        [0.02, 0.12, 0.93]]))
    wb = np.array([2.1, 1.0, 1.5])
    rng = np.random.default_rng(11)
    patches = {}
    rows, cols = it8.parse_block("A49", "L72")
    for ri, r in enumerate(rows):
        for c in cols:
            if r in "GHIJ":                 # neutral rows (D50 chromaticity)
                Y = rng.uniform(3, 88)
                patches[f"{r}{c}"] = (Y / 100.0) * D50 * 100.0
            else:
                Y = rng.uniform(5, 85)
                patches[f"{r}{c}"] = np.array([Y * rng.uniform(0.7, 1.3), Y,
                                               Y * rng.uniform(0.4, 1.2)])
    samples = _balanced_samples(M0, patches, wb)
    ref = _ref_from_patches(patches)
    fit = it8.fit_camera_matrix(samples, ref)
    assert fit.wb_id[0] in "GHIJ"           # auto-picked an in-grid neutral
    # Recovers the balanced->XYZ matrix (white-relative) and the wb multipliers.
    np.testing.assert_allclose(fit.matrix, M0, atol=1e-6)
    np.testing.assert_allclose(fit.matrix @ np.ones(3), D50, atol=1e-9)
    np.testing.assert_allclose(fit.wb_mult, wb, atol=1e-6)


# --------------------------------------------------------------------------- #
# Geometry.
# --------------------------------------------------------------------------- #

def test_grid_points_unit_quad():
    quad = [(0, 0), (220, 0), (220, 120), (0, 120)]   # TL,TR,BR,BL
    pts = it8.grid_sample_points(quad)
    assert len(pts) == 288
    # A1 (row0,col0) centre = (0.5/22*220, 0.5/12*120) = (5,5).
    np.testing.assert_allclose(pts["A1"], (5.0, 5.0), atol=1e-6)
    # L22 (row11,col21) centre = (21.5/22*220, 11.5/12*120) = (215,115).
    np.testing.assert_allclose(pts["L22"], (215.0, 115.0), atol=1e-6)
    # GS strip lies below the colour block (v > 1).
    assert pts["GS0"][1] > 120 and pts["GS23"][1] > 120
    assert pts["GS0"][0] < pts["GS23"][0]             # GS0 left of GS23


def test_flip_quad():
    quad = [(0, 0), (220, 0), (220, 120), (0, 120)]
    flipped = it8.flip_quad(quad)
    pts0 = it8.grid_sample_points(quad)
    ptsf = it8.grid_sample_points(flipped)
    # After a 180 flip, A1 lands where L22 was (and vice versa).
    np.testing.assert_allclose(ptsf["A1"], pts0["L22"], atol=1e-6)
    np.testing.assert_allclose(ptsf["L22"], pts0["A1"], atol=1e-6)


# --------------------------------------------------------------------------- #
# Sampling.
# --------------------------------------------------------------------------- #

def test_sample_patches_flat_chart():
    # Build a synthetic warped chart: each colour cell a flat known RGB.
    quad = [(0, 0), (220, 0), (220, 120), (0, 120)]
    img = np.zeros((130, 230, 3), dtype=np.uint16)
    pts = it8.grid_sample_points(quad)
    expected = {}
    for r in range(12):
        for c in range(22):
            sid = it8.COLOR_IDS[r * 22 + c]
            val = np.array([1000 + r * 50, 2000 + c * 40, 3000], dtype=np.uint16)
            x0, x1 = c * 10, (c + 1) * 10
            y0, y1 = r * 10, (r + 1) * 10
            img[y0:y1, x0:x1] = val
            expected[sid] = val
    samples = it8.sample_patches(img, pts, quad, frac=0.5)
    for sid, val in expected.items():
        np.testing.assert_allclose(samples[sid].rgb, val, atol=1.0)
        assert samples[sid].valid


def test_sample_clip_rejection():
    quad = [(0, 0), (220, 0), (220, 120), (0, 120)]
    img = np.full((130, 230, 3), 65535, dtype=np.uint16)   # fully clipped white
    pts = it8.grid_sample_points(quad)
    samples = it8.sample_patches(img, pts, quad)
    assert not samples["A1"].valid                          # clipped -> invalid


def test_sample_keeps_saturated_and_dark():
    # Raw-linear: a saturated hue reads near-zero in its complementary channels;
    # that is real signal, not clipping, so the patch must stay VALID. Only a
    # blown highlight or a truly-black patch is dropped.
    quad = [(0, 0), (220, 0), (220, 120), (0, 120)]
    img = np.zeros((130, 230, 3), dtype=np.uint16)

    def fill(sid, val):
        i = it8.COLOR_IDS.index(sid)
        r, c = divmod(i, 22)
        img[r * 10:(r + 1) * 10, c * 10:(c + 1) * 10] = val

    fill("A1", [60000, 180, 120])      # saturated red (G/B ~ 0) -> still valid
    fill("A2", [120, 150, 58000])      # saturated blue (R ~ 0) -> still valid
    fill("A3", [4200, 3900, 3600])     # ordinary mid-tone -> valid
    fill("A4", [65535, 65535, 65535])  # blown highlight -> invalid
    fill("A5", [9, 7, 5])              # essentially black -> invalid
    pts = it8.grid_sample_points(quad)
    samples = it8.sample_patches(img, pts, quad, frac=0.5)
    assert samples["A1"].valid and samples["A2"].valid and samples["A3"].valid
    assert not samples["A4"].valid     # highlight-clipped dropped
    assert not samples["A5"].valid     # black-crush dropped


# --------------------------------------------------------------------------- #
# Matrix fit — synthetic round-trip (the key test).
# --------------------------------------------------------------------------- #

def _consistent_samples(M0, patches):
    """Device samples consistent with XYZ = M0 @ (rgb/65535), i.e. exact data."""
    Minv = np.linalg.inv(M0)
    samples = {}
    for sid, xyz in patches.items():
        d = Minv @ (xyz / 100.0)            # normalised device RGB in [0,1]
        rgb = np.clip(d, 0, None) * 65535.0
        samples[sid] = it8.PatchSample(rgb=rgb, valid=True, n_pix=900)
    return samples


def _pin(M0):
    """Pin a matrix so its balanced white (1,1,1) maps EXACTLY to D50 — the
    white-relative anchor the fit produces (M @ (1,1,1) == D50)."""
    return np.diag(D50 / (np.asarray(M0, float) @ np.ones(3))) @ np.asarray(M0, float)


def _balanced_samples(M0, patches, wb):
    """Device samples for the NEW (standards-compliant) convention: M0 maps
    WHITE-BALANCED device RGB -> XYZ D50, and the raw device is the balanced
    value divided by the green-normalised WB multipliers (raw = balanced / wb,
    wb[1] == 1). The fit must recover (pinned M0, wb) exactly. Returns the
    samples; pass `_pin(M0)` as M0 so balanced white -> D50."""
    Minv = np.linalg.inv(np.asarray(M0, float))
    wb = np.asarray(wb, float)
    samples = {}
    for sid, xyz in patches.items():
        balanced = Minv @ (xyz / 100.0)         # balanced device RGB in [0,1]
        raw = np.clip(balanced / wb, 0, None)   # un-balance: raw -> *wb -> balanced
        samples[sid] = it8.PatchSample(rgb=raw * 65535.0, valid=True, n_pix=900)
    return samples


def _ref_from_patches(patches):
    ref = it8.IT8Reference(chart_type="IT8.7/1", batch="TEST")
    for sid, xyz in patches.items():
        lab = it8.xyz_to_lab(xyz)
        ref.patches[sid] = {"XYZ_X": xyz[0], "XYZ_Y": xyz[1], "XYZ_Z": xyz[2],
                            "LAB_L": lab[0], "LAB_A": lab[1], "LAB_B": lab[2]}
    return ref


def test_fit_synthetic_roundtrip():
    # NEW convention: M0 maps WHITE-BALANCED device -> XYZ D50 (pinned so
    # M0 @ (1,1,1) == D50); the raw device carries an explicit green-normalised
    # white balance. The fit must recover BOTH the balanced matrix and the wb
    # multipliers exactly.
    M0 = _pin(np.array([[0.46, 0.31, 0.17],
                        [0.23, 0.70, 0.07],
                        [0.02, 0.12, 0.93]]))
    wb = np.array([2.1, 1.0, 1.5])             # raw -> balanced (green == 1)
    patches = _synthetic_patches()
    samples = _balanced_samples(M0, patches, wb)
    ref = _ref_from_patches(patches)
    fit = it8.fit_camera_matrix(samples, ref)
    # Balanced->XYZ matrix recovered, white-relative (M @ (1,1,1) == D50)...
    np.testing.assert_allclose(fit.matrix, M0, atol=1e-6)
    np.testing.assert_allclose(fit.matrix @ np.ones(3), D50, atol=1e-9)
    # ...and the green-normalised WB multipliers recovered (wb[1] == 1).
    np.testing.assert_allclose(fit.wb_mult, wb, atol=1e-6)
    assert abs(fit.wb_mult[1] - 1.0) < 1e-9
    assert len(fit.used_ids) == 288 and not fit.dropped_ids


def test_fit_neutral_axis_is_neutral():
    M0 = _pin(np.array([[0.50, 0.30, 0.18],
                        [0.25, 0.68, 0.07],
                        [0.03, 0.10, 0.90]]))
    wb = np.array([1.8, 1.0, 1.4])
    patches = _synthetic_patches()
    samples = _balanced_samples(M0, patches, wb)
    ref = _ref_from_patches(patches)
    fit = it8.fit_camera_matrix(samples, ref)
    # fit.matrix now consumes WHITE-BALANCED device, so balance each grey by the
    # recovered wb first; the balanced neutral must map to D50 chromaticity
    # (predicted Lab a*,b* ~ 0).
    for k in it8.GRAY_IDS:
        balanced = (samples[k].rgb / 65535.0) * fit.wb_mult
        lab = it8.xyz_to_lab((fit.matrix @ balanced) * 100.0)
        assert abs(lab[1]) < 0.3 and abs(lab[2]) < 0.3
    # The camera's own raw neutral is 1/wb: balancing it (raw * wb) gives (1,1,1),
    # which the white-relative matrix maps EXACTLY to D50.
    cam_neutral = 1.0 / fit.wb_mult
    white_xyz = fit.matrix @ (cam_neutral * fit.wb_mult)
    np.testing.assert_allclose(white_xyz, D50, atol=1e-9)


def test_fit_drops_invalid_and_missing():
    M0 = np.eye(3) * 0.8 + 0.05
    patches = _synthetic_patches()
    samples = _consistent_samples(M0, patches)
    samples["A5"] = it8.PatchSample(samples["A5"].rgb, valid=False, n_pix=10)
    ref = _ref_from_patches(patches)
    del ref.patches["B7"]                   # missing in reference
    fit = it8.fit_camera_matrix(samples, ref)
    assert "A5" in fit.dropped_ids
    assert "A5" not in fit.used_ids and "B7" not in fit.used_ids


def test_fit_wb_fallback_when_gs0_invalid():
    M0 = _pin(np.array([[0.46, 0.31, 0.17],
                        [0.23, 0.70, 0.07],
                        [0.02, 0.12, 0.93]]))
    wb = np.array([2.1, 1.0, 1.5])
    patches = _synthetic_patches()
    samples = _balanced_samples(M0, patches, wb)
    samples["GS0"] = it8.PatchSample(samples["GS0"].rgb, valid=False, n_pix=1)
    ref = _ref_from_patches(patches)
    fit = it8.fit_camera_matrix(samples, ref)
    # Falls back to another neutral GS patch; the green-normalised wb is the same
    # regardless of which neutral is chosen (all GS are D50-chromatic), so the
    # matrix + wb still come out white-relative and correct.
    assert fit.wb_id != "GS0" and fit.wb_id.startswith("GS")
    np.testing.assert_allclose(fit.matrix @ np.ones(3), D50, atol=1e-9)
    np.testing.assert_allclose(fit.wb_mult, wb, atol=1e-6)


# --------------------------------------------------------------------------- #
# Colour-science helpers.
# --------------------------------------------------------------------------- #

def test_xyz_lab_roundtrip():
    xyz = np.array([[20.0, 18.0, 10.0], [80.0, 82.0, 64.0], [1.0, 1.0, 1.0]])
    lab = it8.xyz_to_lab(xyz)
    back = it8.lab_to_xyz(lab)
    np.testing.assert_allclose(back, xyz, atol=1e-6)


def test_delta_e_2000_sharma_pairs():
    # Sharma et al. CIEDE2000 test data (subset).
    cases = [
        ((50, 2.6772, -79.7751), (50, 0, -82.7485), 2.0425),
        ((50, 3.1571, -77.2803), (50, 0, -82.7485), 2.8615),
        ((50, -1.3802, -84.2814), (50, 0, -82.7485), 1.0000),
        ((50, 0, 0), (50, -1, 2), 2.3669),
        ((50, 2.49, -0.001), (50, -2.49, 0.0009), 7.1792),
        ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ]
    for lab1, lab2, expected in cases:
        got = float(it8.delta_e_2000(np.array(lab1), np.array(lab2))[0])
        assert abs(got - expected) < 1e-3, f"{lab1}->{lab2}: {got} != {expected}"


# --------------------------------------------------------------------------- #
# ICC synthesis round-trips through the existing InputProfile engine.
# --------------------------------------------------------------------------- #

def test_build_icc_reparses_as_input_profile():
    M0 = _pin(np.array([[0.46, 0.31, 0.17],
                        [0.23, 0.70, 0.07],
                        [0.02, 0.12, 0.93]]))
    wb = np.array([2.1, 1.0, 1.5])
    patches = _synthetic_patches()
    samples = _balanced_samples(M0, patches, wb)
    fit = it8.fit_camera_matrix(samples, _ref_from_patches(patches))
    icc = it8.build_camera_icc(fit, "FreeCCR Camera — Test")
    # Valid ICC header, RGB matrix-shaper, parses without UnsupportedICCError.
    assert icc[36:40] == b"acsp" and icc[16:20] == b"RGB "
    prof = cm.InputProfile.from_bytes(icc)
    assert "Test" in prof.description
    # Applying to a RAW mid-gray device patch WITH its as-shot WB yields a
    # near-neutral linear-Adobe result (R~=G~=B). The ICC white-balances the raw
    # (green-normalised wb) before the matrix — without as_shot_wb the un-balanced
    # raw would NOT come out neutral.
    d_gray = samples["GS8"].rgb.astype(np.uint16)
    out = prof.apply(np.tile(d_gray, (2, 2, 1)).astype(np.uint16), as_shot_wb=wb)
    px = out[0, 0].astype(float)
    assert px.max() - px.min() < 0.04 * 65535      # near-neutral output


def test_end_to_end_render_sample_fit():
    # Render a synthetic chart from a known (balanced) matrix + WB, then locate ->
    # sample -> fit and recover both (exercises geometry + sampling + fit composed).
    M0 = _pin(np.array([[0.46, 0.31, 0.17],
                        [0.23, 0.70, 0.07],
                        [0.02, 0.12, 0.93]]))
    wb = np.array([2.1, 1.0, 1.5])
    patches = _synthetic_patches()
    devs = _balanced_samples(M0, patches, wb)      # id -> PatchSample(raw rgb)
    S = 14
    ox = oy = 30
    quad = [(ox, oy), (ox + 22 * S, oy),
            (ox + 22 * S, oy + 12 * S), (ox, oy + 12 * S)]
    pts = it8.grid_sample_points(quad)
    H = int(max(y for _, y in pts.values())) + 2 * S
    W = int(max(x for x, _ in pts.values())) + 2 * S
    img = np.zeros((H, W, 3), dtype=np.uint16)
    half = int(S * 0.45)
    for sid, (cx, cy) in pts.items():
        rgb = np.clip(devs[sid].rgb, 0, 65535).astype(np.uint16)
        x, y = int(round(cx)), int(round(cy))
        img[max(0, y - half):y + half, max(0, x - half):x + half] = rgb
    samples = it8.sample_patches(img, pts, quad, frac=0.5)
    fit = it8.fit_camera_matrix(samples, _ref_from_patches(patches))
    # The balanced->XYZ matrix and the WB multipliers are both recovered through
    # the full render/sample pipeline (residual is u16-quantisation only).
    np.testing.assert_allclose(fit.matrix, M0, atol=2e-3)
    np.testing.assert_allclose(fit.wb_mult, wb, atol=2e-3)
    np.testing.assert_allclose(fit.matrix @ np.ones(3), D50, atol=1e-9)


def test_identity_trc_is_identity():
    # The camera profile's TRC must be the identity (raw-linear device space):
    # build a profile and check its parsed TRC LUT is linear via InputProfile.
    M0 = np.eye(3) * 0.8
    patches = _synthetic_patches()
    fit = it8.fit_camera_matrix(_consistent_samples(M0, patches),
                                _ref_from_patches(patches))
    prof = cm.InputProfile.from_bytes(it8.build_camera_icc(fit, "lin"))
    lut0 = prof._luts[0]
    xs = np.linspace(0, 1, len(lut0))
    np.testing.assert_allclose(lut0, xs, atol=1e-4)


# --------------------------------------------------------------------------- #
# Calibration white balance (the profile owns the WB).
# See spec/camera-profile-calibration-wb.md.
# --------------------------------------------------------------------------- #

def _fit_known_camera():
    M0 = _pin(np.array([[0.46, 0.31, 0.17],
                        [0.23, 0.70, 0.07],
                        [0.02, 0.12, 0.93]]))
    wb = np.array([2.1, 1.0, 1.5])
    patches = _synthetic_patches()
    samples = _balanced_samples(M0, patches, wb)
    return it8.fit_camera_matrix(samples, _ref_from_patches(patches)), samples, wb


def test_icc_bakes_calibration_neutral():
    # The generated ICC records the chart's camera-native neutral (1/wb_mult) in
    # the private 'CCRn' tag, green-normalised, and reparses it.
    fit, _, wb = _fit_known_camera()
    prof = cm.InputProfile.from_bytes(it8.build_camera_icc(fit, "cal"))
    assert prof.calibration_neutral is not None
    np.testing.assert_allclose(prof.calibration_neutral, 1.0 / wb, atol=1e-3)
    assert abs(prof.calibration_neutral[1] - 1.0) < 1e-6      # green-normalised


def test_icc_without_calibration_neutral_uses_frame_wb():
    # A plain matrix-shaper ICC (no 'CCRn') keeps the per-frame behaviour: the
    # as-shot WB is honoured, and None still means unbalanced.
    icc = cm.build_matrix_shaper_icc(
        "plain", (0.4, 0.2, 0.0), (0.3, 0.7, 0.1), (0.2, 0.1, 0.7),
        (1.0, 1.0, 0.0, 1.0, 0.0))
    prof = cm.InputProfile.from_bytes(icc)
    assert prof.calibration_neutral is None
    img = np.full((2, 2, 3), 20000, dtype=np.uint16)
    assert not np.array_equal(prof.apply(img, as_shot_wb=None),
                              prof.apply(img, as_shot_wb=np.array([1.8, 1.0, 1.3])))


def test_icc_calibration_neutral_ignores_frame_metadata():
    """Regression: with a calibration neutral, identical device data renders
    identically whatever the camera's WB mode wrote into the frame."""
    fit, samples, wb = _fit_known_camera()
    prof = cm.InputProfile.from_bytes(it8.build_camera_icc(fit, "cal"))
    dev = np.tile(samples["GS8"].rgb.astype(np.uint16).reshape(1, 1, 3), (3, 3, 1))
    ref = prof.apply(dev, as_shot_wb=wb)
    for meta in (None, np.array([1.0, 1.0, 1.0]), np.array([2366.0, 1024.0, 1640.0]),
                 np.array([0.4, 1.0, 3.3])):
        np.testing.assert_array_equal(prof.apply(dev, as_shot_wb=meta), ref)
    # ...and the chart's own neutral still renders neutral with no metadata help.
    px = prof.apply(dev)[0, 0].astype(float)
    assert px.max() - px.min() < 0.04 * 65535


def test_resolve_wb_gains_precedence():
    n = np.array([0.5, 1.0, 0.7])                       # calibration neutral
    np.testing.assert_allclose(cm.resolve_wb_gains(n, np.array([9.0, 1.0, 4.0])),
                               1.0 / n)                 # profile wins
    np.testing.assert_allclose(cm.resolve_wb_gains(None, np.array([2366.0, 1024.0, 1640.0])),
                               np.array([2366.0, 1024.0, 1640.0]) / 1024.0)
    assert cm.resolve_wb_gains(None, None) is None      # unbalanced degraded path
    # Un-normalised / degenerate inputs are sanitised, never propagated.
    np.testing.assert_allclose(cm.resolve_wb_gains(np.array([1.0, 2.0, 3.0]), None),
                               1.0 / np.array([0.5, 1.0, 1.5]))
    np.testing.assert_allclose(cm.resolve_wb_gains(np.array([0.0, 1.0, -2.0]), None),
                               np.array([1.0, 1.0, 1.0]))

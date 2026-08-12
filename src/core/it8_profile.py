"""
IT8 camera-profile core (pure numpy + OpenCV, no Qt).

Builds a camera *input* ICC profile from a photograph of an IT8 calibration
target. See spec/it8-camera-profile.md. The pipeline is:

  1. parse_it8_reference()   -- read the batch CGATS.17 reference file
  2. decode_target()         -- raw-linear device RGB of the target shot
  3. grid_sample_points()    -- 4-corner homography -> 288 patch centres
  4. sample_patches()        -- robust per-patch device RGB (clip-rejected)
  5. fit_camera_matrix()     -- least-squares 3x3 device->XYZ(D50), D50-pinned
  6. build_camera_icc()      -- valid ICC v2.4 matrix-shaper bytes

The fit reduces to a 3x3 matrix + identity tone curve, exactly what an ICC
matrix-shaper profile (and FreeCCR's InputProfile apply path) can represent.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
    _CV2 = True
except ImportError:                       # pragma: no cover - cv2 is a hard dep
    _CV2 = False

from core import color_management


class IT8ReferenceError(Exception):
    """Raised when an IT8 reference (CGATS) file can't be parsed or is invalid.
    Surfaced to the user instead of a raw traceback."""


# --------------------------------------------------------------------------- #
# Patch identifiers (the geometry the dialog overlays and the fit consumes).
# --------------------------------------------------------------------------- #

_ROWS = "ABCDEFGHIJKL"                     # 12 rows, top -> bottom
# Colour grid IDs, column-fastest (A1..A22, B1.., L22) = 264.
COLOR_IDS: List[str] = [f"{r}{c}" for r in _ROWS for c in range(1, 23)]
# Grayscale strip GS0..GS23 = 24, left (light/Dmin) -> right (dark/Dmax).
GRAY_IDS: List[str] = [f"GS{k}" for k in range(24)]
ALL_IDS: List[str] = COLOR_IDS + GRAY_IDS

# Chart proportions from ArgyllCMS it8.cht (units are template-internal but
# proportionally exact). The colour block is the user-placed quad; the gray
# strip is positioned relative to it. See spec §3.2 / §5.2.
_CELL = 25.625
_CB_X0, _CB_W = 26.625, 22 * _CELL         # colour block x-origin / width (563.75)
_CB_Y0, _CB_H = 26.625, 12 * _CELL         # colour block y-origin / height (307.5)
_GS_CY = 358.75 + 51.25 / 2.0              # gray strip centre y in cht units (384.375)
GRAY_V0 = (_GS_CY - _CB_Y0) / _CB_H        # gray strip v in colour-block unit space (~1.1634)
DEFAULT_GRAY_OFFSET = 0.0


def _gray_u(k: int) -> float:
    """Unit-space x (colour-block normalised) of grayscale cell k (0..23)."""
    return ((k + 0.5) * _CELL - _CB_X0) / _CB_W


# --------------------------------------------------------------------------- #
# 1. Reference file (CGATS.17).
# --------------------------------------------------------------------------- #

def _normalize_sample_id(sid: str) -> str:
    """Map a reference SAMPLE_ID onto our canonical ids.

    Colour patches: a letter A-L + column number, with optional zero padding
    ('A01' -> 'A1'). Grayscale: 'GS'/'GS0n' -> 'GS<int>'. Anything else is
    upper-cased and returned as-is (ignored downstream if not in ALL_IDS).
    """
    s = sid.strip().upper()
    if not s:
        return s
    if s.startswith("GS"):
        num = s[2:].lstrip("0") or "0"
        return f"GS{int(num)}" if num.isdigit() else s
    if s[0] in _ROWS and s[1:].isdigit():
        return f"{s[0]}{int(s[1:])}"
    return s


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    if raw[:3] == b"\xef\xbb\xbf":         # strip UTF-8 BOM
        raw = raw[3:]
    for enc in ("utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def _strip_comment(line: str) -> str:
    """Remove a '#' comment (to end of line) but not inside a quoted value."""
    out, in_q = [], False
    for ch in line:
        if ch == '"':
            in_q = not in_q
        if ch == "#" and not in_q:
            break
        out.append(ch)
    return "".join(out)


@dataclass
class IT8Reference:
    chart_type: str = ""                   # 'IT8.7/1' | 'IT8.7/2' | ''
    batch: str = ""                        # SERIAL value (for display)
    descriptor: str = ""
    originator: str = ""
    fields: List[str] = field(default_factory=list)
    patches: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def xyz(self, sample_id: str) -> Optional[np.ndarray]:
        """XYZ (D50, 2deg, Y~=100 scale) for a patch, or None.

        Uses XYZ_* columns when present, else derives from LAB_* via the D50
        white point.
        """
        p = self.patches.get(sample_id)
        if p is None:
            return None
        if "XYZ_X" in p and "XYZ_Y" in p and "XYZ_Z" in p:
            return np.array([p["XYZ_X"], p["XYZ_Y"], p["XYZ_Z"]], dtype=np.float64)
        lab = self.lab(sample_id)
        return None if lab is None else lab_to_xyz(lab)

    def lab(self, sample_id: str) -> Optional[np.ndarray]:
        """CIE Lab (D50) for a patch, or None. Uses LAB_* columns when present,
        else derives from XYZ_*."""
        p = self.patches.get(sample_id)
        if p is None:
            return None
        if "LAB_L" in p and "LAB_A" in p and "LAB_B" in p:
            return np.array([p["LAB_L"], p["LAB_A"], p["LAB_B"]], dtype=np.float64)
        if "XYZ_X" in p and "XYZ_Y" in p and "XYZ_Z" in p:
            return xyz_to_lab(np.array([p["XYZ_X"], p["XYZ_Y"], p["XYZ_Z"]]))
        return None

    @property
    def matched_ids(self) -> List[str]:
        """Canonical ids present in the file (intersection with ALL_IDS)."""
        return [i for i in ALL_IDS if i in self.patches]


def parse_it8_reference(path: str) -> IT8Reference:
    """Parse a CGATS.17 IT8 reference file. Reads columns dynamically from the
    DATA_FORMAT block (the field set varies by batch). Raises IT8ReferenceError
    on a malformed/empty file."""
    text = _read_text(path)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    ref = IT8Reference()
    state = "PRE"                          # PRE | FORMAT | DATA
    fmt: List[str] = []
    n_fields: Optional[int] = None
    n_sets: Optional[int] = None
    first_token_seen = False
    bad_rows = 0

    for raw_line in lines:
        line = _strip_comment(raw_line).strip()
        if not line:
            continue

        if not first_token_seen:
            first_token_seen = True
            tok = line.split()[0]
            if tok.upper().startswith("IT8"):
                ref.chart_type = tok
                continue                   # consume the bare type token

        upper = line.upper()
        if state == "PRE":
            if upper.startswith("BEGIN_DATA_FORMAT"):
                state = "FORMAT"
                continue
            if upper.startswith("BEGIN_DATA"):
                state = "DATA"
                continue
            parts = line.split(None, 1)
            kw = parts[0].upper()
            val = parts[1].strip().strip('"').strip() if len(parts) > 1 else ""
            if kw == "NUMBER_OF_FIELDS":
                try:
                    n_fields = int(val.split()[0])
                except (ValueError, IndexError):
                    pass
            elif kw == "NUMBER_OF_SETS":
                try:
                    n_sets = int(val.split()[0])
                except (ValueError, IndexError):
                    pass
            elif kw == "SERIAL":
                ref.batch = val
            elif kw == "DESCRIPTOR":
                ref.descriptor = val
            elif kw == "ORIGINATOR":
                ref.originator = val
            # KEYWORD "NAME" / other preamble keywords are ignored — the
            # DATA_FORMAT block is the authority on column names.
        elif state == "FORMAT":
            if upper.startswith("END_DATA_FORMAT"):
                state = "PRE"
                continue
            fmt.extend(line.split())
        elif state == "DATA":
            if upper.startswith("END_DATA"):
                state = "PRE"
                continue
            toks = line.split()
            if not fmt or len(toks) != len(fmt):
                # Skip rows whose token count doesn't match DATA_FORMAT — never
                # zip-truncate them into partial records (a truncated row would
                # otherwise store e.g. XYZ_X/Y without Z). Gross truncation is
                # then caught by the patch-count guard below.
                bad_rows += 1
                continue
            rec: Dict[str, float] = {}
            sid = None
            for name, tok in zip(fmt, toks):
                if name.upper() in ("SAMPLE_ID", "SAMPLE_NAME") and sid is None:
                    sid = tok
                    continue
                try:
                    rec[name.upper()] = float(tok)
                except ValueError:
                    pass
            if sid is not None:
                ref.patches[_normalize_sample_id(sid)] = rec

    ref.fields = fmt
    if not fmt:
        raise IT8ReferenceError(
            "No DATA_FORMAT block found — this does not look like a CGATS/IT8 "
            "reference file.")
    if not ref.patches:
        extra = (f" ({bad_rows} row(s) had a token count != NUMBER_OF_FIELDS.)"
                 if bad_rows else "")
        raise IT8ReferenceError("No data patches found in the reference file."
                                + extra)
    # The reliably-present colorimetry must be there for at least the neutrals.
    if not any(("XYZ_X" in p or "LAB_L" in p) for p in ref.patches.values()):
        raise IT8ReferenceError(
            "Reference file has no XYZ_* or LAB_* columns — cannot build a "
            "profile from it.")
    if n_sets is not None and len(ref.patches) < min(n_sets, 200) // 2:
        extra = f" ({bad_rows} malformed row(s) skipped.)" if bad_rows else ""
        raise IT8ReferenceError(
            f"Reference file declares {n_sets} patches but only "
            f"{len(ref.patches)} were parsed — the file looks truncated." + extra)
    return ref


# --------------------------------------------------------------------------- #
# 1b. CxF (Color Exchange Format) reference files — XML, ISO 17972.
# --------------------------------------------------------------------------- #

def _strip_ns(root) -> None:
    """Rewrite every tag/attribute to its local name (drop the namespace).
    CxF producers vary between a default and a 'cc:' namespace; some (LaserSoft)
    also use British spelling (Colour*). Local-name matching tolerates all."""
    for el in root.iter():
        el.tag = el.tag.rpartition('}')[2]
        if el.attrib:
            el.attrib = {k.rpartition('}')[2]: v for k, v in el.attrib.items()}


def _cxf_spec_ref(el) -> Optional[str]:
    """The Colo[u]rSpecification IDREF attribute of a colour-value element."""
    for k, v in el.attrib.items():
        if k.endswith("Specification"):
            return v
    return None


def _cxf_triplet(obj, children: Tuple[str, str, str], specs: dict):
    """Among an Object's descendants, find the element that has exactly the
    given direct child tags (e.g. L/A/B or X/Y/Z) — matched by child structure,
    so ColorCIELab vs ColourCIELab vs ColourIEXYZ spelling doesn't matter — and
    return its 3 floats, preferring a value under a D50/2° ColorSpecification."""
    best, best_score = None, -1
    for el in obj.iter():
        vals = []
        for ct in children:
            sub = el.find(ct)
            if sub is None or sub.text is None:
                vals = None
                break
            try:
                vals.append(float(sub.text.strip()))
            except ValueError:
                vals = None
                break
        if not vals or len(vals) != 3:
            continue
        ill, obs = specs.get(_cxf_spec_ref(el), ("", ""))
        score = (2 if ill.upper() == "D50" else 0) + (1 if obs.startswith("2") else 0)
        if score > best_score:
            best, best_score = vals, score
    return best


_CXF_SKIP_TYPES = {"substrate", "colorant", "trick", "measurements", "device"}


def parse_cxf_reference(path: str) -> IT8Reference:
    """Parse a CxF3 (ISO 17972) reference file. Namespace- and spelling-tolerant
    (default or cc: namespace; American 'Color' or British 'Colour'). Reads each
    patch's D50/2° CIEXYZ (and/or CIELab). Raises IT8ReferenceError on a file
    that isn't parseable CxF or has no usable patches."""
    try:
        with open(path, "rb") as f:
            root = ET.fromstring(f.read())     # bytes → honours XML encoding/BOM
    except ET.ParseError as e:
        raise IT8ReferenceError(f"Not a valid CxF/XML file: {e}")
    _strip_ns(root)
    if not root.tag.endswith("CxF"):
        raise IT8ReferenceError("XML root is not <CxF> — not a CxF colour file.")

    ref = IT8Reference()
    fi = root.find("FileInformation")
    if fi is not None:
        for tag in fi.findall("Tag"):
            nm = (tag.get("Name") or "").lower()
            if nm == "serial":
                ref.batch = tag.get("Value") or ""
            elif nm.endswith("targettype"):
                ref.chart_type = tag.get("Value") or ""
        ref.descriptor = fi.findtext("Description") or ""
        ref.originator = fi.findtext("Creator") or ""

    # spec Id -> (illuminant, observer), for D50/2° selection.
    specs: Dict[str, Tuple[str, str]] = {}
    for cs in root.iter():
        if cs.tag.endswith("Specification") and cs.get("Id") is not None:
            ill = obs = ""
            for sub in cs.iter():
                if sub.tag == "Illuminant" and sub.text:
                    ill = sub.text.strip()
                elif sub.tag == "Observer" and sub.text:
                    obs = sub.text.strip()
            specs[cs.get("Id")] = (ill, obs)

    for obj in root.iter("Object"):
        otype = (obj.get("ObjectType") or "").lower()
        if otype in _CXF_SKIP_TYPES:
            continue
        name = obj.get("Name") or obj.get("Id")
        if not name:
            continue
        rec: Dict[str, float] = {}
        xyz = _cxf_triplet(obj, ("X", "Y", "Z"), specs)
        if xyz is not None:
            rec["XYZ_X"], rec["XYZ_Y"], rec["XYZ_Z"] = xyz
        lab = _cxf_triplet(obj, ("L", "A", "B"), specs)
        if lab is not None:
            rec["LAB_L"], rec["LAB_A"], rec["LAB_B"] = lab
        if rec:
            ref.patches[_normalize_sample_id(name)] = rec

    ref.fields = ["SAMPLE_ID", "XYZ_X", "XYZ_Y", "XYZ_Z", "LAB_L", "LAB_A", "LAB_B"]
    if not ref.patches:
        raise IT8ReferenceError(
            "No usable patches found in the CxF file (no D50 CIEXYZ/CIELab "
            "values were parsed).")
    return ref


def parse_reference(path: str) -> IT8Reference:
    """Parse an IT8 reference file, dispatching by content: CxF (XML) vs CGATS
    (text). This is the entry point the UI should call."""
    ext = os.path.splitext(path)[1].lower()
    try:
        with open(path, "rb") as f:
            head = f.read(256).lstrip()
    except OSError as e:
        raise IT8ReferenceError(f"Could not read the reference file:\n{e}")
    if ext == ".cxf" or head[:1] == b"<":
        return parse_cxf_reference(path)
    return parse_it8_reference(path)


# --------------------------------------------------------------------------- #
# 2. Target decode (raw-linear device RGB, no input ICC).
# --------------------------------------------------------------------------- #

SAMPLE_MAX = 2048                          # long-side cap for the sampling decode


def decode_target(path: str, sample_max: int = SAMPLE_MAX) -> Optional[np.ndarray]:
    """Decode the IT8 target shot to raw-linear device RGB (uint16, HxWx3, RGB
    order), bypassing Positive mode and any active input ICC — the bare device
    space the profile is fitted on. See spec §5.1 / §6.1.

    read_image is called on a bare CCRImage created via __new__ so the
    heavyweight __init__ (a redundant 1080px decode + lens correction +
    thumbnail) is skipped; read_image only needs `source_ops` on the instance.
    A half-size (preview) decode capped at sample_max gives ample pixels per
    patch. apply_input_icc=False also pins no_auto_scale=True, so this matches
    the device space the profiled/ICC preview decode uses (raw + absolute sensor
    values); the default unprofiled preview is Adobe RGB + rawpy auto-scale, but
    that is irrelevant to fitting — the fit and its later application both operate
    in raw device RGB."""
    from core.ccr_image import CCRImage
    img = CCRImage.__new__(CCRImage)
    img.source_ops = []
    return img.read_image(path, preview=True, max_long_side=sample_max,
                          positive_override=False, apply_input_icc=False)


# --------------------------------------------------------------------------- #
# 3. Patch grid geometry (4-corner homography -> sample centres).
# --------------------------------------------------------------------------- #

def _homography(quad: np.ndarray) -> np.ndarray:
    """Perspective transform mapping the unit square (TL,TR,BR,BL) -> quad."""
    src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    dst = np.asarray(quad, dtype=np.float32)
    if _CV2:
        return cv2.getPerspectiveTransform(src, dst)
    return _perspective_transform_np(src, dst)


def _perspective_transform_np(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Solve the 8-DOF homography (fallback when cv2 is unavailable)."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        b.extend([u, v])
    h = np.linalg.solve(np.asarray(A, float), np.asarray(b, float))
    return np.array([[h[0], h[1], h[2]],
                     [h[3], h[4], h[5]],
                     [h[6], h[7], 1.0]], dtype=np.float64)


def _apply_h(H: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Map (N,2) unit-space points through homography H -> (N,2) array coords."""
    pts = np.hstack([uv, np.ones((len(uv), 1))])
    out = pts @ np.asarray(H, float).T
    return out[:, :2] / out[:, 2:3]


def flip_quad(quad):
    """180-degree relabel of a (TL, TR, BR, BL) quad — for a chart placed
    upside-down. new TL=old BR, TR=old BL, BR=old TL, BL=old TR."""
    q = list(quad)
    return [q[2], q[3], q[0], q[1]]


def grid_sample_points(quad, gray_offset: float = DEFAULT_GRAY_OFFSET
                       ) -> Dict[str, Tuple[float, float]]:
    """Sample centres (array coords) for all 288 patches.

    quad: the 4 corners (TL, TR, BR, BL) of the COLOUR block in array space.
    gray_offset nudges the grayscale row vertically (batch variation)."""
    H = _homography(np.asarray(quad, dtype=np.float64))
    uv = []
    for r in range(12):
        for c in range(22):
            uv.append(((c + 0.5) / 22.0, (r + 0.5) / 12.0))
    gv = GRAY_V0 + gray_offset
    for k in range(24):
        uv.append((_gray_u(k), gv))
    pts = _apply_h(H, np.asarray(uv, dtype=np.float64))
    return {i: (float(x), float(y)) for i, (x, y) in zip(ALL_IDS, pts)}


def parse_block(tl_id: str, br_id: str):
    """Row letters + column numbers for a rectangular patch block from its
    top-left and bottom-right ids, e.g. ('A49','L72') -> (['A'..'L'],[49..72]).
    Used for non-classic targets (e.g. the ISO 12641-2 advanced grid) that are a
    plain rows×cols grid with no separate gray strip."""
    def split(s):
        m = re.match(r'^\s*([A-Za-z])(\d+)\s*$', s or "")
        if not m:
            raise ValueError(f"Patch id must be a letter + number (e.g. A49), got {s!r}")
        return m.group(1).upper(), int(m.group(2))
    (r0, c0), (r1, c1) = split(tl_id), split(br_id)
    rows = [chr(x) for x in range(min(ord(r0), ord(r1)), max(ord(r0), ord(r1)) + 1)]
    cols = list(range(min(c0, c1), max(c0, c1) + 1))
    return rows, cols


def next_block_ids(tl_id: str, br_id: str, max_col: Optional[int] = None
                   ) -> Tuple[str, str]:
    """Advance a block's column range by its own width for the next card of a
    multi-card target (e.g. ('A1','L24') -> ('A25','L48') -> ('A49','L72')).

    Rows are preserved; only the columns shift. If `max_col` is given the new
    range is clamped so the right column never exceeds it (the last card of a
    target that doesn't divide evenly). Raises ValueError on an unparseable id.
    """
    rows, cols = parse_block(tl_id, br_id)
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    width = c1 - c0 + 1
    nc0, nc1 = c0 + width, c1 + width
    if max_col is not None and nc1 > max_col:
        nc1 = max_col
        nc0 = min(nc0, nc1)
    return f"{r0}{nc0}", f"{r1}{nc1}"


def block_sample_points(quad, rows, cols) -> Dict[str, Tuple[float, float]]:
    """Sample centres (array coords) for a regular len(rows)×len(cols) grid laid
    on the 4-corner quad (TL,TR,BR,BL). IDs are '<row><col>' (e.g. 'A49')."""
    H = _homography(np.asarray(quad, dtype=np.float64))
    nr, nc = len(rows), len(cols)
    uv, ids = [], []
    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            uv.append(((ci + 0.5) / nc, (ri + 0.5) / nr))
            ids.append(f"{r}{c}")
    pts = _apply_h(H, np.asarray(uv, dtype=np.float64))
    return {i: (float(x), float(y)) for i, (x, y) in zip(ids, pts)}


# --------------------------------------------------------------------------- #
# 4. Patch sampling (robust central window, clip rejection).
# --------------------------------------------------------------------------- #

@dataclass
class PatchSample:
    rgb: np.ndarray                        # (3,) float, device RGB in [0, 65535]
    valid: bool                            # False if clipped / out of frame
    n_pix: int


def _quad_cell_halfsize(quad: np.ndarray, frac: float,
                        ncols: int = 22, nrows: int = 12) -> Tuple[float, float]:
    """Half-window (px) for sampling, derived from the block's mean cell size so
    the window scales with the chart's size in the frame and its grid density."""
    q = np.asarray(quad, dtype=np.float64)
    top = np.linalg.norm(q[1] - q[0]); bot = np.linalg.norm(q[2] - q[3])
    left = np.linalg.norm(q[3] - q[0]); right = np.linalg.norm(q[2] - q[1])
    cell_w = 0.5 * (top + bot) / max(1, ncols)
    cell_h = 0.5 * (left + right) / max(1, nrows)
    return max(1.0, 0.5 * frac * cell_w), max(1.0, 0.5 * frac * cell_h)


def sample_patches(img_u16: np.ndarray, points: Dict[str, Tuple[float, float]],
                   quad, frac: float = 0.5, clip_hi: float = 0.995,
                   black_floor: float = 0.0008, ncols: int = 22,
                   nrows: int = 12) -> Dict[str, PatchSample]:
    """Sample each patch's device RGB from a central window via a per-channel
    trimmed mean and flag only the genuinely unusable ones. ncols/nrows set the
    grid density so the sampling window scales to the cell size.

    A patch is invalid only when its colour can't be trusted:
      * **highlight-clipped** — any channel has >2% of pixels pinned at the
        sensor ceiling (`clip_hi`), so information above it is lost; or
      * **black-crushed** — even its brightest channel sits at/below the black
        floor (`black_floor`), i.e. no signal at all; or
      * partly **out of frame**.

    A merely *low* channel is NOT a defect on raw-linear device data: a saturated
    hue legitimately reads near-zero in one or two complementary channels while
    carrying full signal in the rest (a vivid red has green/blue ≈ 0), and dense
    film patches read low — yet these saturated/dark patches are the most
    informative for the matrix fit. The previous "any pixel below 0.5 % of full
    scale counts as clipped" test wrongly rejected all of them."""
    h, w = img_u16.shape[:2]
    full = 65535.0
    hi = clip_hi * full
    floor = black_floor * full
    half_w, half_h = _quad_cell_halfsize(quad, frac, ncols, nrows)
    out: Dict[str, PatchSample] = {}
    for sid, (cx, cy) in points.items():
        x0 = int(round(cx - half_w)); x1 = int(round(cx + half_w)) + 1
        y0 = int(round(cy - half_h)); y1 = int(round(cy + half_h)) + 1
        x0c, x1c = max(0, x0), min(w, x1)
        y0c, y1c = max(0, y0), min(h, y1)
        if x1c - x0c < 2 or y1c - y0c < 2:
            out[sid] = PatchSample(np.zeros(3), False, 0)
            continue
        win = img_u16[y0c:y1c, x0c:x1c, :3].reshape(-1, 3).astype(np.float64)
        # Per-channel trimmed mean (drop 10/90 tails) — robust to dust/edges.
        rgb = np.empty(3)
        for ch in range(3):
            col = win[:, ch]
            p10, p90 = np.percentile(col, [10, 90])
            keep = col[(col >= p10) & (col <= p90)]
            rgb[ch] = keep.mean() if keep.size else col.mean()
        # Highlight clipping is judged per-channel (a blown channel loses info);
        # black-crush on the brightest channel so saturated hues survive.
        hi_frac = np.mean(win >= hi, axis=0)
        highlight_clipped = bool(np.any(hi_frac >= 0.02))
        black_crushed = bool(rgb.max() <= floor)
        in_frame = (x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h)
        valid = (not highlight_clipped) and (not black_crushed) and in_frame
        out[sid] = PatchSample(rgb, bool(valid), win.shape[0])
    return out


def merge_samples(cards: List[Dict[str, PatchSample]]) -> Dict[str, PatchSample]:
    """Combine the per-card sample dicts of a multi-card target into one set.

    Each card photographs a different block of patch ids, so the union covers
    the whole chart. On the rare id collision (overlapping blocks) a *valid*
    sample always wins over an invalid one; on equal validity the first (earlier)
    card in the list is kept. The merged dict feeds straight into
    `fit_camera_matrix`.
    """
    merged: Dict[str, PatchSample] = {}
    for samples in cards:
        for sid, ps in samples.items():
            old = merged.get(sid)
            if old is None or (ps.valid and not old.valid):
                merged[sid] = ps
    return merged


# --------------------------------------------------------------------------- #
# 5. Colour-science helpers (XYZ<->Lab, deltaE). XYZ on the Y~=100 scale.
# --------------------------------------------------------------------------- #

_D50_W100 = np.array([96.422, 100.0, 82.521])   # D50 white, Y=100 scale
_EPS = 216.0 / 24389.0
_KAPPA = 24389.0 / 27.0


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray = _D50_W100) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    xr = xyz / white
    f = np.where(xr > _EPS, np.cbrt(np.clip(xr, 0, None)), (_KAPPA * xr + 16) / 116)
    if xyz.ndim == 1:
        L = 116 * f[1] - 16; a = 500 * (f[0] - f[1]); b = 200 * (f[1] - f[2])
        return np.array([L, a, b])
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def lab_to_xyz(lab: np.ndarray, white: np.ndarray = _D50_W100) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    def inv(t):
        t3 = t ** 3
        return np.where(t3 > _EPS, t3, (116 * t - 16) / _KAPPA)
    xyz = np.stack([inv(fx), inv(fy), inv(fz)], axis=-1) * white
    return xyz


def delta_e_76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(lab1) - np.asarray(lab2), axis=-1)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 (kL=kC=kH=1), Sharma formulation. Accepts (N,3) or (3,)."""
    lab1 = np.atleast_2d(np.asarray(lab1, dtype=np.float64))
    lab2 = np.atleast_2d(np.asarray(lab2, dtype=np.float64))
    L1, a1, b1 = lab1.T; L2, a2, b2 = lab2.T
    C1 = np.hypot(a1, b1); C2 = np.hypot(a2, b2)
    Cbar = 0.5 * (C1 + C2)
    G = 0.5 * (1 - np.sqrt(Cbar ** 7 / (Cbar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)
    dhp = np.where((C1p * C2p) == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2)
    Lbp = 0.5 * (L1 + L2)
    Cbp = 0.5 * (C1p + C2p)
    hsum = h1p + h2p
    hbp = np.where(np.abs(h1p - h2p) > 180, (hsum + 360) / 2, hsum / 2)
    hbp = np.where((C1p * C2p) == 0, hsum, hbp)
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30))
         + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6))
         - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dTheta = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + (0.015 * (Lbp - 50) ** 2) / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc = 1 + 0.045 * Cbp
    Sh = 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dTheta)) * Rc
    de = np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                 + Rt * (dCp / Sc) * (dHp / Sh))
    return de


# --------------------------------------------------------------------------- #
# 6. Matrix fit (device-linear RGB -> XYZ D50), D50 neutral-pinned.
# --------------------------------------------------------------------------- #

D50_XYZ = np.array(color_management.D50_XYZ, dtype=np.float64)   # [0,1] scale, Y=1


@dataclass
class CameraFit:
    matrix: np.ndarray                     # (3,3) WHITE-BALANCED device RGB -> XYZ D50;
                                           #       matrix @ (1,1,1) == D50 (white-relative)
    wb_mult: np.ndarray                    # (3,) green-normalised WB multipliers used to
                                           #      balance the raw device before the matrix
    avg_de: float
    med_de: float
    p95_de: float
    max_de: float
    per_patch: List[Tuple[str, float]]     # (id, dE2000), worst-first
    used_ids: List[str]
    dropped_ids: List[str]
    wb_id: str


_NEUTRAL_CHROMA = 6.0   # max CIELab chroma for a patch to count as "neutral"


def _pick_wb_id(samples: Dict[str, PatchSample], ref: IT8Reference,
                preferred: Optional[str] = None) -> Optional[str]:
    """The lightest valid neutral to white-balance on: `preferred` if given and
    valid, else the lightest valid near-neutral patch (low Lab chroma, highest
    L*) — works for any target (classic GS strip or in-grid grays), else the
    highest-Y valid patch."""
    if (preferred and preferred in samples and samples[preferred].valid
            and ref.lab(preferred) is not None):
        return preferred
    best, best_L = None, -1.0
    for sid, ps in samples.items():
        if not ps.valid:
            continue
        lab = ref.lab(sid)
        if lab is None:
            continue
        if float(np.hypot(lab[1], lab[2])) < _NEUTRAL_CHROMA and lab[0] > best_L:
            best, best_L = sid, lab[0]
    if best is not None:
        return best
    best, best_Y = None, -1.0
    for sid, ps in samples.items():
        xyz = ref.xyz(sid)
        if ps.valid and xyz is not None and xyz[1] > best_Y:
            best, best_Y = sid, xyz[1]
    return best


def fit_camera_matrix(samples: Dict[str, PatchSample], ref: IT8Reference, *,
                      ids: Optional[List[str]] = None, weight: str = "none",
                      wb_id: Optional[str] = None) -> CameraFit:
    """Fit the 3x3 camera matrix from sampled device RGB + reference XYZ.

    ids: patch ids to use (default: every sampled id present in the reference).
    weight: 'none' (plain XYZ least squares, the documented baseline) or '1/Y'
    (down-weight bright patches). wb_id: force the white-balance patch (default:
    auto-pick the lightest neutral). See spec §5.4."""
    chosen_wb = _pick_wb_id(samples, ref, wb_id)
    if chosen_wb is None:
        raise IT8ReferenceError("No valid neutral patch to white-balance on — "
                                "check the chart placement and exposure.")

    iter_ids = ids if ids is not None else [s for s in samples
                                            if ref.xyz(s) is not None]
    used, dropped = [], []
    d_list, x_list = [], []
    for sid in iter_ids:
        ps = samples.get(sid)
        xyz = ref.xyz(sid)
        if ps is None or xyz is None:
            continue
        if not ps.valid:
            dropped.append(sid)
            continue
        used.append(sid)
        d_list.append(ps.rgb / 65535.0)
        x_list.append(xyz / 100.0)
    if len(used) < 6:
        raise IT8ReferenceError(
            f"Only {len(used)} usable patches — too few to fit a profile. "
            "Re-check corner placement, exposure (clipping), and the reference "
            "file.")

    d = np.asarray(d_list)                  # (N,3) normalised raw device RGB (un-WB)
    X = np.asarray(x_list)                  # (N,3) reference XYZ, Y in [0,~1]

    # Relative-colorimetric: map the chart's white patch to the PCS white (D50) by
    # normalising the reference so the lightest neutral reads D50 (Y=1) — the
    # ArgyllCMS "brightest patch -> media white" rule. The matrix already pins
    # white->D50; scoring it against a <100%-reflectance reference white would
    # inflate every ΔE by the white patch's reflectance deficit (~10%) and falsely
    # flag a good fit as "Check placement".
    white_Y = float(ref.xyz(chosen_wb)[1]) / 100.0
    if white_Y > 1e-6:
        X = X / white_Y

    # --- White-balance the device on the neutral (standard convention) --------
    # ICC matrix and DNG ForwardMatrix profiles operate on WHITE-BALANCED data,
    # so the fitted matrix consumes balanced device RGB. The multipliers are
    # green-normalised (green channel == 1), matching the DNG AsShotNeutral
    # convention; the same balance is rebuilt per-image at apply time from the
    # frame's own as-shot neutral.
    n_raw = np.clip(samples[chosen_wb].rgb / 65535.0, 1e-6, None)
    wb = n_raw[1] / n_raw                    # neutral -> equal RGB; wb[1] == 1
    # Balanced device, normalised so the white/neutral patch reads ~ (1,1,1):
    # makes the matrix WHITE-RELATIVE so the chart's absolute exposure drops OUT
    # of the gain (ArgyllCMS / colprof default), not baked into it.
    bN = (d * wb) / float(n_raw[1])

    # --- (Weighted) least squares  X ~= bN @ A ;  M = A.T  => XYZ = M @ balanced
    if weight == "1/Y":
        sw = np.sqrt(1.0 / np.clip(X[:, 1], 0.02, None))[:, None]
        A, *_ = np.linalg.lstsq(bN * sw, X * sw, rcond=None)
    else:
        A, *_ = np.linalg.lstsq(bN, X, rcond=None)
    M = A.T                                  # balanced device RGB -> XYZ D50

    # Pin balanced white (1,1,1) EXACTLY to D50 (Y=1). A row (output) scale fixes
    # both the neutral chromaticity and the residual exposure; for a good fit it
    # is ~uniform. This is the white-relative anchor every standard CMM expects.
    white = M @ np.ones(3)
    white = np.where(np.abs(white) < 1e-9, 1e-9, white)
    M = np.diag(D50_XYZ / white) @ M         # M @ (1,1,1) == D50

    # Quality: predicted Lab vs the white-normalised reference Lab (same relative-
    # colorimetric space the matrix maps into, so ΔE measures COLOUR error only).
    pred_xyz = (bN @ M.T) * 100.0
    pred_lab = xyz_to_lab(pred_xyz)
    ref_lab = xyz_to_lab(X * 100.0)
    de = delta_e_2000(ref_lab, pred_lab)
    order = np.argsort(-de)
    per_patch = [(used[i], float(de[i])) for i in order]

    return CameraFit(
        matrix=M,
        wb_mult=wb,
        avg_de=float(de.mean()),
        med_de=float(np.median(de)),
        p95_de=float(np.percentile(de, 95)),
        max_de=float(de.max()),
        per_patch=per_patch,
        used_ids=used,
        dropped_ids=dropped,
        wb_id=chosen_wb,
    )


# --------------------------------------------------------------------------- #
# 7. ICC synthesis (matrix-shaper, identity TRC).
# --------------------------------------------------------------------------- #

# Identity parametric (type-3) TRC: y = x for x>=0  (g=1, a=1, b=0, c=1, d=0).
_IDENTITY_TRC = (1.0, 1.0, 0.0, 1.0, 0.0)


def build_camera_icc(fit: CameraFit, desc: str, *,
                     mode: str = "matrix", grid: int = 17,
                     samples: Optional[Dict[str, "PatchSample"]] = None,
                     ref: Optional[IT8Reference] = None,
                     copyright_text: str = "Public Domain. No rights reserved."
                     ) -> bytes:
    """Synthesise camera-profile ICC bytes from the fit. Round-trips through
    InputProfile.from_bytes.

    mode='matrix' (default) writes a 3x3 matrix-shaper (M's columns are the XYZ
    D50 colorants, identity TRC). mode='clut' writes a higher-accuracy lut16 cLUT
    (the 3x3 base + a smooth RBF residual; needs `samples`+`ref`).

    Both record the chart's camera-native neutral (`1/wb_mult`) in the private
    'CCRn' tag, so apply balances every frame on the setup the profile was
    calibrated on instead of the frame's as-shot metadata (see
    spec/camera-profile-calibration-wb.md)."""
    desc = str(desc).encode("ascii", "replace").decode("ascii")
    copyright_text = str(copyright_text).encode("ascii", "replace").decode("ascii")
    neutral = 1.0 / np.asarray(fit.wb_mult, dtype=np.float64)[:3]
    if mode == "clut":
        if samples is None or ref is None:
            raise ValueError("cLUT mode requires the sampled patches and reference")
        clut_xyz = build_residual_clut(fit, samples, ref, grid=grid)
        return color_management.build_clut_icc(
            desc, clut_xyz, grid, copyright_text=copyright_text, neutral=neutral)
    M = np.asarray(fit.matrix, dtype=np.float64)
    return color_management.build_matrix_shaper_icc(
        desc,
        tuple(M[:, 0]), tuple(M[:, 1]), tuple(M[:, 2]),
        _IDENTITY_TRC,
        copyright_text=copyright_text,
        neutral=neutral,
    )


def build_residual_clut(fit: CameraFit, samples: Dict[str, "PatchSample"],
                        ref: IT8Reference, grid: int = 17) -> np.ndarray:
    """A (grid,grid,grid,3) XYZ D50 (Y=1) cLUT = the 3x3 matrix base + a smooth
    scattered-data residual that bends the saturated corners the 3x3 can't reach.

    The residual is a broad Gaussian-RBF-with-polynomial fit of the per-patch
    errors `X_i - M·d_i` over the device-RGB sample points, regularised and faded
    to zero outside the patch hull so the LUT degrades to the safe 3x3 in unsampled
    regions. Gross per-axis reversals (the chroma-noise/ringing failure mode) are
    bounded to ≤ RINGING_TOL by shrinking the correction toward the base if needed;
    the LUT is NOT guaranteed strictly monotone — small reversals are the cost of
    the bias correction, and the 3x3 matrix mode remains the strictly-safe default.
    Indexed [R][G][B]."""
    # The CLUT lives in the SAME white-balanced device space the matrix path
    # consumes (raw * as-shot WB, in [0,1]) and uses the SAME matrix M (fit.matrix)
    # as its linear base, so the cLUT and the 3x3 AGREE (cLUT = matrix + a small
    # colour residual) — apply must not produce a different brightness than the
    # matrix. The chart is shot below white, so its balanced device values sit at
    # ~n1 (the neutral's green level); the reference is scaled by n1 to the matrix's
    # exposure so the residual is a pure colour correction, NOT a 1/n1 brightness
    # blow-up. The residual fades to zero outside the sampled hull.
    M = np.asarray(fit.matrix, dtype=np.float64)         # balanced device -> XYZ (matches apply)
    wb = np.asarray(fit.wb_mult, dtype=np.float64)[:3]
    # Anchor the reference's exposure on the white patch so the white neutral's
    # residual is zero — the cLUT then matches the 3x3 EXACTLY on the neutral axis
    # (corrects colour, not brightness); toggling matrix<->cLUT shifts nothing.
    xw = ref.xyz(fit.wb_id)
    if fit.wb_id in samples and xw is not None and float(xw[1]) > 1e-6:
        di_w = np.clip(samples[fit.wb_id].rgb / 65535.0 * wb, 0.0, 1.0)
        s = float((M @ di_w)[1]) / (float(xw[1]) / 100.0)
    else:
        s = 1.0
    d_list, r_list = [], []
    for sid in fit.used_ids:
        ps = samples.get(sid)
        xyz = ref.xyz(sid)
        if ps is None or xyz is None:
            continue
        di = np.clip(ps.rgb / 65535.0 * wb, 0.0, 1.0)    # white-balanced, clamped to grid
        d_list.append(di)
        r_list.append(s * xyz / 100.0 - M @ di)          # residual at the matrix's exposure scale
    d = np.asarray(d_list, dtype=np.float64)             # (N,3) balanced device
    r = np.asarray(r_list, dtype=np.float64)             # (N,3) XYZ residual
    n = len(d)

    ax = np.linspace(0.0, 1.0, grid)
    Rg, Gg, Bg = np.meshgrid(ax, ax, ax, indexing="ij")
    nodes = np.stack([Rg, Gg, Bg], axis=-1).reshape(-1, 3)
    base = nodes @ M.T                                    # M_f — same scale as the matrix path

    if n < 5:                                             # too few patches to bend safely
        return np.clip(base, 0.0, 2.0).reshape(grid, grid, grid, 3)

    # Gaussian RBF. A BROAD kernel (≈3× the mean nearest-neighbour device spacing)
    # plus a firm smoothness regularisation captures the systematic saturated-corner
    # bias without the high-frequency overshoot that a tight kernel rings with — it
    # gives both lower ΔE and far less non-monotonicity than a narrow fit.
    dd = np.sqrt(np.maximum(((d[:, None, :] - d[None, :, :]) ** 2).sum(-1), 0.0))
    nn = dd + np.eye(n) * 1e9
    sigma = max(float(nn.min(1).mean()) * 3.0, 1e-3)
    Phi = np.exp(-(dd / sigma) ** 2)
    Phi = Phi + (1e-2 * np.trace(Phi) / n) * np.eye(n)   # smoothness regularisation
    P = np.concatenate([np.ones((n, 1)), d], axis=1)     # affine polynomial term
    A = np.zeros((n + 4, n + 4))
    A[:n, :n] = Phi; A[:n, n:] = P; A[n:, :n] = P.T
    rhs = np.zeros((n + 4, 3)); rhs[:n] = r
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)

    dn = np.sqrt(np.maximum(((nodes[:, None, :] - d[None, :, :]) ** 2).sum(-1), 0.0))
    resid = np.exp(-(dn / sigma) ** 2) @ sol[:n] + np.concatenate(
        [np.ones((len(nodes), 1)), nodes], axis=1) @ sol[n:]
    # Fade the residual to zero beyond ~one kernel width from the patch hull, so
    # unsampled corners fall back to the pure matrix instead of ringing.
    fade = np.exp(-(np.maximum(dn.min(1) - sigma, 0.0) / (2.0 * sigma)) ** 2)
    corr = (resid * fade[:, None])

    # Ringing safeguard: the correction must not introduce a *gross* per-axis
    # reversal the linear base lacks (the parked density experiment's failure
    # mode). If it does, shrink the whole correction toward the base and recheck —
    # strength 0 is the pure 3×3 (always safe), so this terminates. Small reversals
    # are tolerated: they are the cost of the bias correction (see RINGING_TOL).
    base_grid = base.reshape(grid, grid, grid, 3)
    table = base_grid
    for strength in (1.0, 0.7, 0.5, 0.35, 0.2, 0.1, 0.0):
        table = np.clip(base + corr * strength, 0.0, 2.0).reshape(grid, grid, grid, 3)
        if _residual_monotone(base_grid, table):
            break
    return table


RINGING_TOL = 0.12   # max per-axis reversal (fraction of white) the cLUT may add


def _residual_monotone(base_grid: np.ndarray, table: np.ndarray,
                       tol: float = RINGING_TOL) -> bool:
    """True if `table` introduces no per-axis reversal larger than `tol` where the
    linear `base_grid` is non-decreasing. NOT strict monotonicity — small reversals
    are allowed (the bias correction's cost); only gross ringing is rejected."""
    for axis in range(3):
        db = np.diff(base_grid, axis=axis)
        dt = np.diff(table, axis=axis)
        if np.any((db >= 0) & (dt < -tol)):
            return False
    return True

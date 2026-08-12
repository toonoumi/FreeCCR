"""
Color management for FreeCCR.

Two responsibilities, both pure-numpy (no Pillow/lcms runtime dependency — see
spec/color-management.md for why Pillow's ImageCms is unusable for our 16-bit
pipeline):

1. EXPORT color space. The internal pipeline produces an uncalibrated result that
   every viewer interprets as sRGB-encoded display RGB. ``apply_export_colorspace``
   either passes that through (sRGB, pixel-identical to before) or re-encodes it
   into ProPhoto RGB (ROMM), returning the matching ICC profile bytes to embed so
   a colour-managed viewer shows the same colours.

2. INPUT ICC profile. ``InputProfile`` parses a *matrix-shaper* ICC profile and
   converts decoded pixels from that profile's space into the working sRGB
   encoding, applied at decode time before conversion/adjustments. LUT-based and
   CMYK profiles are rejected with ``UnsupportedICCError``.

ICC profiles are synthesised in-process (``build_matrix_shaper_icc``) so nothing
needs to be bundled and there is no profile-licensing/trademark concern. The
synthesised bytes are valid ICC v2.4 matrix-shaper profiles (verified to embed
via tifffile tag 34675 and to parse + build a transform under littleCMS).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


class UnsupportedICCError(Exception):
    """Raised when an input ICC profile is not a simple RGB matrix-shaper
    profile (e.g. LUT/cLUT-based, CMYK, or otherwise unparseable). The caller
    surfaces this to the user instead of silently mis-converting."""


# Global active input profile (a single app-wide setting, by design). Decode
# (CCRImage.read_image) reads it via get_active_input_profile() so it does not
# need to import the backend singleton — avoiding an import cycle.
_active_input_profile: "Optional[InputProfile]" = None


def set_active_input_profile(profile) -> None:
    global _active_input_profile
    _active_input_profile = profile


def get_active_input_profile():
    return _active_input_profile


# Global active DCP (DNG camera profile). Mutually exclusive with the input ICC
# (a single "input colour transform" slot with two possible kinds). Stored
# opaquely here so ccr_image can read it without importing dcp_profile/backend.
_active_dcp_profile = None


def set_active_dcp_profile(profile) -> None:
    global _active_dcp_profile
    _active_dcp_profile = profile


def get_active_dcp_profile():
    return _active_dcp_profile


# Temporary "disable camera profile" toggle (persisted by the UI). When set, the
# active ICC/DCP is NOT applied at decode (the decode reverts to the unprofiled
# path) without clearing the profile.
_input_profile_disabled = False


def set_input_profile_disabled(disabled: bool) -> None:
    global _input_profile_disabled
    _input_profile_disabled = bool(disabled)


def input_profile_disabled() -> bool:
    return _input_profile_disabled


def camera_profile_active() -> bool:
    """Whether a camera profile (ICC or DCP) will actually be applied: one is set
    AND the disable toggle is off."""
    return (not _input_profile_disabled
            and (_active_input_profile is not None or _active_dcp_profile is not None))


# "Camera Matrix" decode mode (a non-removable picker entry, no profile): decode
# in Adobe RGB with rawpy auto-scale (linear) — the camera's built-in matrix. When
# OFF and no profile is set ("None"), the decode is bare camera-native RAW (linear,
# no auto-scale). Mutually exclusive with an active ICC/DCP. Read by ccr_image to
# pick the no-profile decode space. See spec/camera-profile-library.md.
_camera_matrix_mode = False


def set_camera_matrix_mode(on: bool) -> None:
    global _camera_matrix_mode
    _camera_matrix_mode = bool(on)


def camera_matrix_mode() -> bool:
    return _camera_matrix_mode


def active_profile_signature() -> str:
    """Stable id of the profile that would be applied now — what an image's decode
    is 'graded under'. 'none' when disabled or unset. (Positive mode is folded in
    by ccr_backend.active_profile_signature, which this does not know about.)"""
    if _input_profile_disabled:
        return "none"
    if _active_dcp_profile is not None:
        cid = getattr(_active_dcp_profile, "content_id", None)
        return "dcp:" + (cid or getattr(_active_dcp_profile, "name", "") or "?")
    if _active_input_profile is not None:
        cid = getattr(_active_input_profile, "content_id", None)
        return "icc:" + (cid or getattr(_active_input_profile, "description", "") or "?")
    return "none"


def green_normalised(v) -> "Optional[np.ndarray]":
    """Sanitised, green-normalised copy of a 3-vector (a camera neutral or a set
    of WB gains), or None. Non-positive channels fall back to 1.0 — a zero would
    otherwise blow up the reciprocal."""
    if v is None:
        return None
    m = np.asarray(v, dtype=np.float64)[:3].copy()
    m[~np.isfinite(m) | (m <= 0)] = 1.0
    if m[1] > 0:
        m = m / m[1]
    return m


def resolve_wb_gains(calibration_neutral, as_shot_wb) -> "Optional[np.ndarray]":
    """The green-normalised WB gains a camera profile should be applied with.

    A profile built from an IT8 chart is calibrated on ONE physical setup (fixed
    light, fixed camera), and records that setup's camera-native neutral. When it
    carries one, that neutral OWNS the white balance — `1/n` — and the frame's
    as-shot metadata is ignored: the profiled decode is unbalanced
    (`use_camera_wb=False`), so `raw.camera_whitebalance` only reflects the
    camera's WB *setting*, which on AWB is re-estimated per frame from negative
    content and drifts colour across a roll under an unchanging light. See
    spec/camera-profile-calibration-wb.md.

    Profiles without a calibration neutral (imported third-party DCPs, and
    profiles generated before that was recorded) keep the per-frame behaviour.
    Returns None only when neither is available (unbalanced degraded path)."""
    n = green_normalised(calibration_neutral)
    if n is not None:
        return 1.0 / n                       # n[1] == 1, so the gains are too
    return green_normalised(as_shot_wb)


def load_input_profile(path: str) -> "InputProfile":
    """Read and parse a matrix-shaper ICC file. Raises UnsupportedICCError for
    LUT/CMYK/unparseable profiles, or OSError if the file can't be read."""
    with open(path, "rb") as f:
        data = f.read()
    return InputProfile.from_bytes(data)


# --------------------------------------------------------------------------- #
# Reference matrices (float64). Sources: Lindbloom / ninedegreesbelow /
# colour-science. See spec/color-management.md §5.1.
# --------------------------------------------------------------------------- #

# linear sRGB (D65) -> XYZ (D65)
M_SRGB2XYZ = np.array([
    [0.4123908, 0.3575843, 0.1804808],
    [0.2126390, 0.7151687, 0.0721923],
    [0.0193308, 0.1191948, 0.9505322],
], dtype=np.float64)

# Bradford chromatic adaptation D65 -> D50
M_BRADFORD_D65_D50 = np.array([
    [1.0478112, 0.0228866, -0.0501270],
    [0.0295424, 0.9904844, -0.0170491],
    [-0.0092345, 0.0150436, 0.7521316],
], dtype=np.float64)

# linear ProPhoto/ROMM (D50) -> XYZ (D50); columns are the ProPhoto primaries.
M_PROPHOTO2XYZ = np.array([
    [0.7976749, 0.1351917, 0.0313534],
    [0.2880402, 0.7118741, 0.0000857],
    [0.0000000, 0.0000000, 0.8252100],
], dtype=np.float64)

# Derived inverses / adaptations.
M_XYZ2SRGB = np.linalg.inv(M_SRGB2XYZ)                       # XYZ(D65) -> linear sRGB
M_BRADFORD_D50_D65 = np.linalg.inv(M_BRADFORD_D65_D50)       # D50 -> D65
M_XYZ2PROPHOTO = np.linalg.inv(M_PROPHOTO2XYZ)               # XYZ(D50) -> linear ProPhoto

# Combined linear-sRGB(D65) -> linear-ProPhoto(D50) (one matmul on export).
M_SRGB2PROPHOTO = M_XYZ2PROPHOTO @ M_BRADFORD_D65_D50 @ M_SRGB2XYZ
_M_SRGB2PROPHOTO_F32 = M_SRGB2PROPHOTO.astype(np.float32)   # export math runs in float32
# Combined XYZ(D50) -> linear sRGB(D65) (used by InputProfile to reach working space).
M_XYZ_D50_2_SRGB = M_XYZ2SRGB @ M_BRADFORD_D50_D65

# Adobe RGB (1998) primaries -> XYZ (D65); columns are the Adobe primaries.
M_ADOBE2XYZ_D65 = np.array([
    [0.5767309, 0.1855540, 0.1881852],
    [0.2973769, 0.6273491, 0.0752741],
    [0.0270343, 0.0706872, 0.9911085],
], dtype=np.float64)
M_XYZ2ADOBE_D65 = np.linalg.inv(M_ADOBE2XYZ_D65)            # XYZ(D65) -> linear Adobe
# Combined XYZ(D50) -> LINEAR Adobe RGB. The negative pipeline's no-ICC decode is
# linear Adobe RGB (output_color=Adobe, gamma=(1,1)) and the density-based
# inversion reads -log10(value/full) assuming LINEAR input, so an input ICC must
# land an image in this SAME linear-Adobe space (NOT sRGB-gamma-encoded) — else the
# optical-density cast balance mis-reads every channel and casts the result.
M_XYZ_D50_2_ADOBE = M_XYZ2ADOBE_D65 @ M_BRADFORD_D50_D65

# D50 PCS white point (ICC reference illuminant).
D50_XYZ = (0.9642, 1.0000, 0.8249)


# --------------------------------------------------------------------------- #
# Transfer functions (operate on float arrays normalised to [0, 1]).
# --------------------------------------------------------------------------- #

def srgb_decode(v: np.ndarray) -> np.ndarray:
    """sRGB EOTF: encoded -> linear."""
    a = 0.055
    return np.where(v <= 0.04045, v / 12.92, np.power((v + a) / (1 + a), 2.4))


def srgb_encode(x: np.ndarray) -> np.ndarray:
    """sRGB inverse EOTF: linear -> encoded."""
    a = 0.055
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * np.power(x, 1 / 2.4) - a)


def romm_encode(x: np.ndarray) -> np.ndarray:
    """ProPhoto/ROMM gamma encode (gamma 1.8 with the 1/512 linear toe, slope 16).
    Continuous at x = 1/512 because 16*(1/512) == (1/512)**(1/1.8) == 1/32."""
    Et = 1.0 / 512.0
    x = np.clip(x, 0.0, 1.0)
    return np.where(x < Et, 16.0 * x, np.power(x, 1.0 / 1.8))


# --------------------------------------------------------------------------- #
# ICC profile synthesis (matrix-shaper, ICC v2.4 mntr/RGB/XYZ).
# --------------------------------------------------------------------------- #

def _s15f16(x: float) -> int:
    """Encode a float as an ICC s15Fixed16Number (signed 16.16)."""
    return int(round(x * 65536.0))


def _xyz_type(x: float, y: float, z: float) -> bytes:
    return struct.pack('>4sI3i', b'XYZ ', 0, _s15f16(x), _s15f16(y), _s15f16(z))


def _para_type3(g: float, a: float, b: float, c: float, d: float) -> bytes:
    """parametricCurveType, function type 3:  Y=(aX+b)^g for X>=d ; Y=cX for X<d."""
    body = struct.pack('>4sIH2x', b'para', 0, 3)
    for v in (g, a, b, c, d):
        body += struct.pack('>i', _s15f16(v))
    return body


def _desc_type(text: str) -> bytes:
    """textDescriptionType (ICC v2)."""
    b = text.encode('ascii') + b'\x00'
    out = struct.pack('>4sII', b'desc', 0, len(b)) + b
    out += struct.pack('>II', 0, 0)          # unicode language + count
    out += struct.pack('>HB', 0, 0)          # scriptcode code + count
    out += b'\x00' * 67                      # macintosh description
    return out


def _text_type(text: str) -> bytes:
    return struct.pack('>4sI', b'text', 0) + text.encode('ascii') + b'\x00'


# Private (vendor) ICC tag holding the camera-native calibration neutral of a
# FreeCCR-generated camera profile — the neutral of the chart shot the profile was
# fit from, green-normalised. ICC has no standard slot for a device neutral; the
# payload reuses the XYZType container (three s15Fixed16) so the tag is
# well-formed for any CMM, which then ignores the unknown signature.
# See spec/camera-profile-calibration-wb.md §3.2.
_CCR_NEUTRAL_SIG = b'CCRn'


def build_matrix_shaper_icc(desc: str,
                            r_xyz: Tuple[float, float, float],
                            g_xyz: Tuple[float, float, float],
                            b_xyz: Tuple[float, float, float],
                            trc_para: Tuple[float, float, float, float, float],
                            wtpt: Tuple[float, float, float] = D50_XYZ,
                            copyright_text: str = "Public Domain. No rights reserved.",
                            neutral=None) -> bytes:
    """Build a valid ICC v2.4 RGB matrix-shaper profile from D50 colorants and a
    shared parametric (type-3) TRC. Returns the raw profile bytes.

    neutral: optional camera-native calibration neutral (3 values) recorded in the
    private 'CCRn' tag — see resolve_wb_gains."""
    tags = {
        'desc': _desc_type(desc),
        'wtpt': _xyz_type(*wtpt),
        'rXYZ': _xyz_type(*r_xyz),
        'gXYZ': _xyz_type(*g_xyz),
        'bXYZ': _xyz_type(*b_xyz),
        'cprt': _text_type(copyright_text),
    }
    trc = _para_type3(*trc_para)
    tags['rTRC'] = trc
    tags['gTRC'] = trc
    tags['bTRC'] = trc

    order = ['desc', 'rXYZ', 'gXYZ', 'bXYZ', 'wtpt', 'rTRC', 'gTRC', 'bTRC', 'cprt']
    nv = green_normalised(neutral)
    if nv is not None:
        tags[_CCR_NEUTRAL_SIG.decode('ascii')] = _xyz_type(*nv)
        order.append(_CCR_NEUTRAL_SIG.decode('ascii'))
    n = len(order)
    header_size = 128
    table_size = 4 + n * 12
    base = header_size + table_size

    data = b''
    entries = []
    seen: dict = {}
    for name in order:
        payload = tags[name]
        key = bytes(payload)
        if key in seen:
            off, ln = seen[key]
        else:
            if len(data) % 4:
                data += b'\x00' * (4 - len(data) % 4)
            off = base + len(data)
            ln = len(payload)
            data += payload
            seen[key] = (off, ln)
        entries.append((name.encode('ascii'), off, ln))

    table = struct.pack('>I', n)
    for tag, off, ln in entries:
        table += struct.pack('>4sII', tag, off, ln)

    total = header_size + len(table) + len(data)
    header = bytearray(128)
    struct.pack_into('>I', header, 0, total)            # profile size
    struct.pack_into('>4s', header, 4, b'lcms')         # preferred CMM (cosmetic)
    struct.pack_into('>I', header, 8, 0x02400000)       # version 2.4.0
    struct.pack_into('>4s', header, 12, b'mntr')        # device class: display
    struct.pack_into('>4s', header, 16, b'RGB ')        # data colour space
    struct.pack_into('>4s', header, 20, b'XYZ ')        # PCS
    struct.pack_into('>4s', header, 36, b'acsp')        # signature
    struct.pack_into('>i', header, 68, _s15f16(wtpt[0]))  # PCS illuminant (D50)
    struct.pack_into('>i', header, 72, _s15f16(wtpt[1]))
    struct.pack_into('>i', header, 76, _s15f16(wtpt[2]))
    return bytes(header) + table + data


def _adapt_columns_d65_to_d50(m_rgb2xyz_d65: np.ndarray):
    """Adapt the primary XYZ columns of a D65 RGB->XYZ matrix to D50, returning
    (r_xyz, g_xyz, b_xyz) tuples suitable for ICC colorant tags."""
    adapted = M_BRADFORD_D65_D50 @ m_rgb2xyz_d65
    return (tuple(adapted[:, 0]), tuple(adapted[:, 1]), tuple(adapted[:, 2]))


# sRGB working-space profile (D50-adapted sRGB colorants; sRGB piecewise TRC as a
# parametric type-3 curve). Pixel-identical export to before, just now tagged.
_SR, _SG, _SB = _adapt_columns_d65_to_d50(M_SRGB2XYZ)
SRGB_ICC_BYTES = build_matrix_shaper_icc(
    "FreeCCR sRGB",
    _SR, _SG, _SB,
    # sRGB EOTF as Y=(aX+b)^g for X>=d else cX : g=2.4, a=1/1.055, b=0.055/1.055,
    # c=1/12.92, d=0.04045.
    (2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045),
)

# ProPhoto/ROMM profile (colorants already D50; gamma 1.8 + 1/512 toe as type-3).
PROPHOTO_ICC_BYTES = build_matrix_shaper_icc(
    "FreeCCR ROMM (ProPhoto)",
    tuple(M_PROPHOTO2XYZ[:, 0]),
    tuple(M_PROPHOTO2XYZ[:, 1]),
    tuple(M_PROPHOTO2XYZ[:, 2]),
    (1.8, 1.0, 0.0, 0.0625, 0.03125),
)


# --------------------------------------------------------------------------- #
# Export: working(sRGB-encoded) -> target colour space.
# --------------------------------------------------------------------------- #

def export_icc_bytes(target: str) -> bytes:
    return PROPHOTO_ICC_BYTES if target == "prophoto" else SRGB_ICC_BYTES


def apply_export_colorspace(rgb_u16: np.ndarray, target: str) -> Tuple[np.ndarray, bytes]:
    """Map an export array (HxWx3 uint16 RGB, treated as sRGB-encoded) to the
    target colour space. Returns (out_uint16, icc_bytes).

    - 'srgb'      : pixels returned unchanged (the file is merely tagged sRGB).
    - 'prophoto'  : re-encode into ProPhoto/ROMM at full precision.
    """
    if target != "prophoto":
        return rgb_u16, SRGB_ICC_BYTES
    # float32 keeps the memory footprint in line with the rest of the export
    # pipeline; precision stays well under 1 LSB at 16-bit.
    lin = srgb_decode(rgb_u16.astype(np.float32) / np.float32(65535.0))
    pro = lin @ _M_SRGB2PROPHOTO_F32.T
    np.clip(pro, 0.0, 1.0, out=pro)       # clip imaginary/overflow before encode
    enc = romm_encode(pro)
    out = np.rint(enc * 65535.0).astype(np.uint16)
    return out, PROPHOTO_ICC_BYTES


# --------------------------------------------------------------------------- #
# JPEG ICC embedding (cv2 cannot embed; inject APP2 ICC_PROFILE segments).
# --------------------------------------------------------------------------- #

_ICC_APP2_MAXCHUNK = 65519   # 65533 marker-payload limit - 12-byte ICC header - 2


def inject_jpeg_icc(jpeg_bytes: bytes, icc: bytes) -> bytes:
    """Insert an ICC profile into an encoded JPEG byte string as one or more
    APP2 'ICC_PROFILE' marker segments (right after SOI). Returns new bytes.
    No-op (returns input) if the stream does not start with SOI."""
    if not icc or jpeg_bytes[:2] != b'\xff\xd8':
        return jpeg_bytes
    chunks = [icc[i:i + _ICC_APP2_MAXCHUNK]
              for i in range(0, len(icc), _ICC_APP2_MAXCHUNK)] or [b'']
    n = len(chunks)
    segs = b''
    for i, ch in enumerate(chunks, start=1):
        payload = b'ICC_PROFILE\x00' + bytes([i, n]) + ch
        segs += b'\xff\xe2' + struct.pack('>H', len(payload) + 2) + payload
    return jpeg_bytes[:2] + segs + jpeg_bytes[2:]


# --------------------------------------------------------------------------- #
# Input ICC profile -> working sRGB encoding (matrix-shaper only).
# --------------------------------------------------------------------------- #

def _read_tag_table(icc: bytes) -> dict:
    if len(icc) < 132 or icc[36:40] != b'acsp':
        raise UnsupportedICCError("not a valid ICC profile")
    count = struct.unpack('>I', icc[128:132])[0]
    tags = {}
    pos = 132
    for _ in range(count):
        if pos + 12 > len(icc):
            break
        sig = icc[pos:pos + 4]
        off, size = struct.unpack('>II', icc[pos + 4:pos + 12])
        tags[sig] = (off, size)
        pos += 12
    return tags


def _parse_xyz(icc: bytes, off: int) -> np.ndarray:
    x, y, z = struct.unpack('>3i', icc[off + 8:off + 20])
    return np.array([x, y, z], dtype=np.float64) / 65536.0


def _parse_trc_to_lut(icc: bytes, off: int, n: int = 65536) -> np.ndarray:
    """Return an n-entry float64 LUT mapping device value [0,1] -> linear [0,1].
    n == 65536 so a 16-bit device value indexes the LUT exactly (no quantisation)."""
    tag_type = icc[off:off + 4]
    xs = np.linspace(0.0, 1.0, n)
    if tag_type == b'curv':
        count = struct.unpack('>I', icc[off + 8:off + 12])[0]
        if count == 0:
            return xs.copy()                          # identity (gamma 1.0)
        if count == 1:
            gamma = struct.unpack('>H', icc[off + 12:off + 14])[0] / 256.0  # u8Fixed8
            return np.power(xs, gamma)
        samples = np.frombuffer(icc[off + 12:off + 12 + 2 * count], dtype='>u2').astype(np.float64) / 65535.0
        src = np.linspace(0.0, 1.0, count)
        return np.interp(xs, src, samples)
    if tag_type == b'para':
        func = struct.unpack('>H', icc[off + 8:off + 10])[0]
        nparams = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(func)
        if nparams is None:
            raise UnsupportedICCError(f"unsupported parametric curve type {func}")
        raw = struct.unpack('>%di' % nparams, icc[off + 12:off + 12 + 4 * nparams])
        p = [v / 65536.0 for v in raw]
        return _eval_para(func, p, xs)
    raise UnsupportedICCError(f"unsupported TRC tag type {tag_type!r}")


def _eval_para(func: int, p: list, x: np.ndarray) -> np.ndarray:
    g = p[0]
    if func == 0:
        return np.power(np.clip(x, 0, 1), g)
    if func == 1:
        a, b = p[1], p[2]
        thr = -b / a if a != 0 else 0.0
        return np.where(x >= thr, np.power(np.clip(a * x + b, 0, None), g), 0.0)
    if func == 2:
        a, b, c = p[1], p[2], p[3]
        thr = -b / a if a != 0 else 0.0
        return np.where(x >= thr, np.power(np.clip(a * x + b, 0, None), g) + c, c)
    if func == 3:
        a, b, c, d = p[1], p[2], p[3], p[4]
        return np.where(x >= d, np.power(np.clip(a * x + b, 0, None), g), c * x)
    if func == 4:
        a, b, c, d, e, f = p[1], p[2], p[3], p[4], p[5], p[6]
        return np.where(x >= d, np.power(np.clip(a * x + b, 0, None), g) + e, c * x + f)
    raise UnsupportedICCError(f"unsupported parametric curve type {func}")


class InputProfile:
    """A parsed RGB matrix-shaper input profile that converts decoded device
    pixels into the working sRGB encoding."""

    def __init__(self, combined_matrix: np.ndarray, luts: list, desc: str):
        self._kind = "matrix"                               # "matrix" | "clut"
        self._matrix = combined_matrix.astype(np.float32)   # device-linRGB -> linear Adobe RGB
        # 3 device->linear LUTs, length 65536 so a uint16 value indexes directly.
        self._luts = [np.ascontiguousarray(l, dtype=np.float32) for l in luts]
        self._clut: Optional["_CLUT"] = None
        self.description = desc
        # Camera-native neutral this profile was calibrated on ('CCRn'), or None
        # for a profile that carries none. When set it OWNS the white balance.
        self.calibration_neutral: Optional[np.ndarray] = None

    @classmethod
    def _from_clut(cls, clut: "_CLUT", desc: str) -> "InputProfile":
        """Build a LUT-based (cLUT / A2B) input profile — same apply contract
        (device RGB -> linear Adobe RGB), interpolated through the CLUT."""
        self = cls.__new__(cls)
        self._kind = "clut"
        self._matrix = None
        self._luts = None
        self._clut = clut
        self.description = desc
        self.calibration_neutral = None
        return self

    @staticmethod
    def _read_calibration_neutral(icc: bytes, tags: dict):
        """The camera-native calibration neutral from the private 'CCRn' tag
        (XYZType payload), green-normalised, or None. A malformed tag is ignored
        rather than failing the whole profile."""
        if _CCR_NEUTRAL_SIG not in tags:
            return None
        try:
            return green_normalised(_parse_xyz(icc, tags[_CCR_NEUTRAL_SIG][0]))
        except Exception:
            return None

    @classmethod
    def from_bytes(cls, icc: bytes) -> "InputProfile":
        tags = _read_tag_table(icc)
        if icc[16:20] != b'RGB ':
            raise UnsupportedICCError("input profile is not an RGB profile")
        # Matrix-shaper: colorant + TRC tags. Kept verbatim (byte-identical path).
        needed = [b'rXYZ', b'gXYZ', b'bXYZ', b'rTRC', b'gTRC', b'bTRC']
        if all(t in tags for t in needed):
            r = _parse_xyz(icc, tags[b'rXYZ'][0])
            g = _parse_xyz(icc, tags[b'gXYZ'][0])
            b = _parse_xyz(icc, tags[b'bXYZ'][0])
            m_colorants = np.stack([r, g, b], axis=1)    # columns = primaries (XYZ D50)
            combined = M_XYZ_D50_2_ADOBE @ m_colorants   # device-linRGB -> linear Adobe RGB
            luts = [_parse_trc_to_lut(icc, tags[t][0])
                    for t in (b'rTRC', b'gTRC', b'bTRC')]
            prof = cls(combined, luts, cls._read_desc(icc, tags))
            prof.calibration_neutral = cls._read_calibration_neutral(icc, tags)
            return prof
        # LUT-based: an A2B0 (fall back A2B1) device->PCS cLUT in mft1/mft2/mAB.
        a2b = next((t for t in (b'A2B0', b'A2B1') if t in tags), None)
        if a2b is not None:
            pcs = icc[20:24]
            if pcs not in (b'XYZ ', b'Lab '):
                raise UnsupportedICCError(f"unsupported PCS {pcs!r:s} (only XYZ/Lab)")
            is_v4 = icc[8] >= 4                           # major version byte
            clut = _parse_a2b(icc, tags[a2b][0], pcs, is_v4)
            prof = cls._from_clut(clut, cls._read_desc(icc, tags))
            prof.calibration_neutral = cls._read_calibration_neutral(icc, tags)
            return prof
        raise UnsupportedICCError(
            "only RGB matrix-shaper or A2B cLUT ICC profiles are supported "
            "(CMYK / B2A-only profiles are not)")

    @staticmethod
    def _read_desc(icc: bytes, tags: dict) -> str:
        if b'desc' not in tags:
            return ""
        off, size = tags[b'desc']
        try:
            if icc[off:off + 4] == b'desc':
                n = struct.unpack('>I', icc[off + 8:off + 12])[0]
                return icc[off + 12:off + 12 + n].split(b'\x00')[0].decode('ascii', 'replace')
        except Exception:
            pass
        return ""

    def _wb_gains(self, as_shot_wb):
        """Green-normalised white-balance multipliers to apply before the matrix,
        or None to skip balancing. The profile's own calibration neutral wins when
        it has one; otherwise the frame's as-shot neutral
        (raw.camera_whitebalance). See resolve_wb_gains."""
        m = resolve_wb_gains(getattr(self, "calibration_neutral", None), as_shot_wb)
        return None if m is None else m.astype(np.float32)

    def apply(self, rgb_u16: np.ndarray, as_shot_wb=None) -> np.ndarray:
        """Convert HxWx3 uint16 camera device RGB into uint16 **linear Adobe RGB**
        — the exact working space the no-ICC negative decode produces, so the
        density-based inversion downstream (-log10(value/full)) reads consistent
        LINEAR data.

        The matrix is a standard camera-profile matrix, so it consumes
        WHITE-BALANCED data, so the raw is balanced before the matrix — exactly
        what a DNG ForwardMatrix or a RawTherapee input ICC expects. A profile
        that records the neutral it was CALIBRATED on balances every frame with
        that (fixed setup ⇒ fixed WB); otherwise the frame's as-shot neutral
        (as_shot_wb, green-normalised) is used, and as_shot_wb=None (e.g. a
        non-RAW input) skips balancing (degraded path)."""
        if rgb_u16.ndim != 3 or rgb_u16.shape[2] != 3 or rgb_u16.dtype != np.uint16:
            return rgb_u16
        m = self._wb_gains(as_shot_wb)
        if self._kind == "clut":
            return self._apply_clut(rgb_u16, m)
        # LUTs are length 65536, so a uint16 value indexes them directly.
        lin = np.empty(rgb_u16.shape, dtype=np.float32)
        for c in range(3):
            lin[..., c] = self._luts[c][rgb_u16[..., c]]
        if m is not None:
            lin = lin * m                                # white balance (raw -> balanced)
        adobe_lin = lin @ self._matrix.T                 # balanced -> linear Adobe RGB
        out = np.clip(adobe_lin, 0.0, 1.0)               # stay linear (no sRGB OETF)
        return np.rint(out * 65535.0).astype(np.uint16)

    def _apply_clut(self, rgb_u16: np.ndarray, m=None) -> np.ndarray:
        """cLUT apply: white-balance -> tetrahedral CLUT interpolation -> XYZ(D50)
        -> linear Adobe RGB. The CLUT is built in balanced device space, so the
        raw is balanced (and clamped into the [0,1] node grid) before the lookup
        — same uint16 linear-Adobe output as the matrix path."""
        clut = self._clut
        lin = np.empty(rgb_u16.shape, dtype=np.float32)
        for c in range(3):                               # input curves (65536 LUTs)
            lin[..., c] = clut.in_luts[c][rgb_u16[..., c]]
        if m is not None:
            lin = np.clip(lin * m, 0.0, 1.0)             # balance into the [0,1] grid
        xyz = _clut_interp_tetra(lin, clut)              # (...,3) XYZ D50 (Y=1)
        adobe_lin = xyz @ M_XYZ_D50_2_ADOBE.T            # -> linear Adobe RGB
        out = np.clip(adobe_lin, 0.0, 1.0)               # stay linear (no sRGB OETF)
        return np.rint(out * 65535.0).astype(np.uint16)


# --------------------------------------------------------------------------- #
# LUT-based (cLUT / A2B) ICC support — parse + interpolate. See
# spec/clut-icc-support.md. Device->PCS only (A2B), PCS decoded to XYZ(D50, Y=1)
# at parse time so apply() is a uniform gather+interpolate+matmul.
# --------------------------------------------------------------------------- #

_XYZ_ENC = 1.0 + 32767.0 / 32768.0          # ICC lut8/lut16 XYZ-number max (~1.99997)


@dataclass
class _CLUT:
    grid: Tuple[int, ...]                    # per-axis grid points (len = n_in, == 3)
    table: np.ndarray                        # (g0,g1,g2,3) XYZ D50 (Y=1), float32
    in_luts: List[np.ndarray]                # 3 device->[0,1] LUTs, len 65536 (uint16 index)


def _pcs_decode(v01: np.ndarray, pcs: bytes, is_v4: bool, eight_bit: bool) -> np.ndarray:
    """Decode normalised [0,1] PCS grid values to XYZ(D50, Y=1)."""
    v01 = np.asarray(v01, dtype=np.float64)
    if pcs == b'XYZ ':
        return v01 * _XYZ_ENC
    # PCS Lab. v4 and 8-bit legacy use the plain scale; v2 16-bit legacy maps
    # L*=100 to 0xFF00 (not 0xFFFF), hence the 65535/65280 correction.
    scale = 1.0 if (is_v4 or eight_bit) else (65535.0 / 65280.0)
    L = v01[..., 0] * 100.0 * scale
    a = v01[..., 1] * 255.0 * scale - 128.0
    b = v01[..., 2] * 255.0 * scale - 128.0
    lab = np.stack([L, a, b], axis=-1)
    from core.it8_profile import lab_to_xyz       # lazy: it8_profile imports us
    return lab_to_xyz(lab) / 100.0                # D50 white, Y=1


def _clut_interp_trilinear(d01: np.ndarray, clut: "_CLUT") -> np.ndarray:
    """Trilinear interpolation over the 3-D CLUT grid (reference / fallback)."""
    table = clut.table
    dims = np.array(table.shape[:3])
    shape = d01.shape[:-1]
    pts = np.asarray(d01, dtype=np.float32).reshape(-1, 3)
    t = pts * (dims - 1)
    i0 = np.clip(np.floor(t).astype(np.intp), 0, dims - 2)
    f = (t - i0).astype(np.float32)
    ir, ig, ib = i0[:, 0], i0[:, 1], i0[:, 2]
    out = np.zeros((pts.shape[0], 3), dtype=np.float32)
    for cx in (0, 1):
        for cy in (0, 1):
            for cz in (0, 1):
                w = ((f[:, 0] if cx else 1 - f[:, 0])
                     * (f[:, 1] if cy else 1 - f[:, 1])
                     * (f[:, 2] if cz else 1 - f[:, 2]))
                out += w[:, None] * table[ir + cx, ig + cy, ib + cz]
    return out.reshape(*shape, 3)


def _tetra_chunk(pts: np.ndarray, table: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Tetrahedral interpolation for one flat chunk of (N,3) device points."""
    t = pts * (dims - 1).astype(np.float32)
    i0 = np.clip(np.floor(t).astype(np.intp), 0, dims - 2)
    f = (t - i0).astype(np.float32)
    fr, fg, fb = f[:, 0], f[:, 1], f[:, 2]
    ir, ig, ib = i0[:, 0], i0[:, 1], i0[:, 2]

    def C(dx, dy, dz):
        return table[ir + dx, ig + dy, ib + dz]
    c000 = C(0, 0, 0); c100 = C(1, 0, 0); c010 = C(0, 1, 0); c001 = C(0, 0, 1)
    c110 = C(1, 1, 0); c101 = C(1, 0, 1); c011 = C(0, 1, 1); c111 = C(1, 1, 1)
    R, G, B = fr[:, None], fg[:, None], fb[:, None]
    # The six tetrahedra of the unit cube, by the ordering of (fr,fg,fb).
    v1 = c000 + R * (c100 - c000) + G * (c110 - c100) + B * (c111 - c110)  # r>=g>=b
    v2 = c000 + R * (c100 - c000) + B * (c101 - c100) + G * (c111 - c101)  # r>=b>=g
    v3 = c000 + B * (c001 - c000) + R * (c101 - c001) + G * (c111 - c101)  # b>=r>=g
    v4 = c000 + B * (c001 - c000) + G * (c011 - c001) + R * (c111 - c011)  # b>=g>=r
    v5 = c000 + G * (c010 - c000) + B * (c011 - c010) + R * (c111 - c011)  # g>=b>=r
    v6 = c000 + G * (c010 - c000) + R * (c110 - c010) + B * (c111 - c110)  # g>=r>=b
    conds = [(fr >= fg) & (fg >= fb), (fr >= fb) & (fb >= fg),
             (fb >= fr) & (fr >= fg), (fb >= fg) & (fg >= fr),
             (fg >= fb) & (fb >= fr), (fg >= fr) & (fr >= fb)]
    return np.select([c[:, None] for c in conds], [v1, v2, v3, v4, v5, v6],
                     default=c000)


def _clut_interp_tetra(d01: np.ndarray, clut: "_CLUT") -> np.ndarray:
    """Tetrahedral interpolation (ICC-standard; affine-exact, no neutral-axis
    desaturation). Chunked to bound memory at full-resolution export."""
    table = clut.table
    dims = np.array(table.shape[:3])
    if np.any(dims < 2):
        return _clut_interp_trilinear(d01, clut)
    shape = d01.shape[:-1]
    pts = np.ascontiguousarray(np.asarray(d01, dtype=np.float32).reshape(-1, 3))
    n = pts.shape[0]
    out = np.empty((n, 3), dtype=np.float32)
    for s in range(0, n, _TETRA_CHUNK):
        out[s:s + _TETRA_CHUNK] = _tetra_chunk(pts[s:s + _TETRA_CHUNK], table, dims)
    return out.reshape(*shape, 3)


_TETRA_CHUNK = 1 << 22                                    # rows per pass (bounds memory)


# --- A2B element parsers --------------------------------------------------- #

def _assemble_clut(in_tab: np.ndarray, clut_grid: np.ndarray, out_tab: np.ndarray,
                   grid: Tuple[int, ...], pcs: bytes, is_v4: bool,
                   eight_bit: bool) -> "_CLUT":
    """Common tail for mft1/mft2: input curves -> 65536 LUTs; output curves +
    PCS decode folded into the table."""
    xs = np.linspace(0.0, 1.0, 65536)
    in_luts = []
    for c in range(in_tab.shape[0]):
        src = np.linspace(0.0, 1.0, in_tab.shape[1])
        in_luts.append(np.interp(xs, src, in_tab[c]).astype(np.float32))
    tbl = np.array(clut_grid, dtype=np.float64)
    for j in range(out_tab.shape[0]):
        src = np.linspace(0.0, 1.0, out_tab.shape[1])
        tbl[..., j] = np.interp(tbl[..., j], src, out_tab[j])
    table = _pcs_decode(tbl, pcs, is_v4, eight_bit).astype(np.float32)
    return _CLUT(grid=grid, table=table, in_luts=in_luts)


def _parse_lut8(icc: bytes, off: int, pcs: bytes, is_v4: bool) -> "_CLUT":
    i, o, g = icc[off + 8], icc[off + 9], icc[off + 10]
    if i != 3 or o != 3:
        raise UnsupportedICCError("only 3->3 cLUT input profiles are supported")
    if g < 2:
        raise UnsupportedICCError("degenerate CLUT grid")
    p = off + 48                                          # after 'mft1'+rsv+i/o/g/pad+3x3 matrix
    in_tab = (np.frombuffer(icc[p:p + i * 256], dtype=np.uint8)
              .astype(np.float64).reshape(i, 256) / 255.0)
    p += i * 256
    n = g ** i * o
    clut = np.frombuffer(icc[p:p + n], dtype=np.uint8).astype(np.float64) / 255.0
    p += n
    out_tab = (np.frombuffer(icc[p:p + o * 256], dtype=np.uint8)
               .astype(np.float64).reshape(o, 256) / 255.0)
    return _assemble_clut(in_tab, clut.reshape(g, g, g, o), out_tab,
                          (g, g, g), pcs, is_v4, eight_bit=True)


def _parse_lut16(icc: bytes, off: int, pcs: bytes, is_v4: bool) -> "_CLUT":
    i, o, g = icc[off + 8], icc[off + 9], icc[off + 10]
    if i != 3 or o != 3:
        raise UnsupportedICCError("only 3->3 cLUT input profiles are supported")
    if g < 2:
        raise UnsupportedICCError("degenerate CLUT grid")
    n = struct.unpack('>H', icc[off + 48:off + 50])[0]    # input table entries
    m = struct.unpack('>H', icc[off + 50:off + 52])[0]    # output table entries
    p = off + 52
    in_tab = (np.frombuffer(icc[p:p + i * n * 2], dtype='>u2')
              .astype(np.float64).reshape(i, n) / 65535.0)
    p += i * n * 2
    cn = g ** i * o
    clut = np.frombuffer(icc[p:p + cn * 2], dtype='>u2').astype(np.float64) / 65535.0
    p += cn * 2
    out_tab = (np.frombuffer(icc[p:p + o * m * 2], dtype='>u2')
               .astype(np.float64).reshape(o, m) / 65535.0)
    return _assemble_clut(in_tab, clut.reshape(g, g, g, o), out_tab,
                          (g, g, g), pcs, is_v4, eight_bit=False)


def _curve_len(icc: bytes, off: int) -> int:
    """Byte length (padded to 4) of a standalone 'curv'/'para' element."""
    sig = icc[off:off + 4]
    if sig == b'curv':
        count = struct.unpack('>I', icc[off + 8:off + 12])[0]
        ln = 12 + 2 * count
    elif sig == b'para':
        func = struct.unpack('>H', icc[off + 8:off + 10])[0]
        nparams = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}.get(func)
        if nparams is None:
            raise UnsupportedICCError(f"unsupported parametric curve type {func}")
        ln = 12 + 4 * nparams
    else:
        raise UnsupportedICCError(f"unsupported curve element {sig!r:s}")
    return ln + (-ln % 4)


def _read_curve_set(icc: bytes, off: int, count: int) -> List[np.ndarray]:
    """`count` consecutive curve elements -> list of 65536-entry [0,1]->[0,1] LUTs."""
    luts, p = [], off
    for _ in range(count):
        luts.append(_parse_trc_to_lut(icc, p))
        p += _curve_len(icc, p)
    return luts


def _parse_lutAToB(icc: bytes, off: int, pcs: bytes, is_v4: bool) -> "_CLUT":
    i, o = icc[off + 8], icc[off + 9]
    if i != 3 or o != 3:
        raise UnsupportedICCError("only 3->3 cLUT input profiles are supported")
    off_B, off_mat, off_M, off_clut, off_A = struct.unpack(
        '>5I', icc[off + 12:off + 32])
    xs = np.linspace(0.0, 1.0, 65536)
    in_luts = (_read_curve_set(icc, off + off_A, i) if off_A
               else [xs.astype(np.float32) for _ in range(i)])
    in_luts = [np.asarray(l, dtype=np.float32) for l in in_luts]
    if not off_clut:
        raise UnsupportedICCError("lutAToB without a CLUT is not supported")
    grid, clut_grid = _read_clut_element(icc, off + off_clut, i, o)
    tbl = np.array(clut_grid, dtype=np.float64)
    if off_M:
        for j, lut in enumerate(_read_curve_set(icc, off + off_M, o)):
            tbl[..., j] = np.interp(tbl[..., j], xs, lut)
    if off_mat:
        vals = np.array(struct.unpack('>12i', icc[off + off_mat:off + off_mat + 48]),
                        dtype=np.float64) / 65536.0
        tbl = tbl @ vals[:9].reshape(3, 3).T + vals[9:12]
    if off_B:
        for j, lut in enumerate(_read_curve_set(icc, off + off_B, o)):
            tbl[..., j] = np.interp(tbl[..., j], xs, lut)
    table = _pcs_decode(tbl, pcs, is_v4, eight_bit=False).astype(np.float32)
    return _CLUT(grid=grid, table=table, in_luts=in_luts)


def _read_clut_element(icc: bytes, off: int, i: int, o: int):
    """Parse an mAB CLUT element -> (grid dims, (g0,..,o) [0,1] float grid)."""
    gp = [icc[off + k] for k in range(i)]                 # gridPoints[16], first i used
    if any(x < 2 for x in gp):
        raise UnsupportedICCError("degenerate CLUT grid")
    prec = icc[off + 16]                                  # 1=u8, 2=u16
    p = off + 20
    count = o
    for x in gp:
        count *= x
    if prec == 1:
        raw = np.frombuffer(icc[p:p + count], dtype=np.uint8).astype(np.float64) / 255.0
    elif prec == 2:
        raw = np.frombuffer(icc[p:p + count * 2], dtype='>u2').astype(np.float64) / 65535.0
    else:
        raise UnsupportedICCError(f"unsupported CLUT precision {prec}")
    return tuple(gp), raw.reshape(*gp, o)


def _parse_a2b(icc: bytes, off: int, pcs: bytes, is_v4: bool) -> "_CLUT":
    sig = icc[off:off + 4]
    if sig == b'mft1':
        return _parse_lut8(icc, off, pcs, is_v4)
    if sig == b'mft2':
        return _parse_lut16(icc, off, pcs, is_v4)
    if sig == b'mAB ':
        return _parse_lutAToB(icc, off, pcs, is_v4)
    raise UnsupportedICCError(f"unsupported A2B element type {sig!r:s}")


# --------------------------------------------------------------------------- #
# cLUT ICC synthesis (lut16 / 'mft2', RGB device -> XYZ PCS). Round-trips through
# _parse_lut16 above. See spec/clut-icc-support.md §5.8.
# --------------------------------------------------------------------------- #

def _lut16_type(clut_xyz: np.ndarray, grid: int) -> bytes:
    """lut16Type ('mft2') element: identity 3x3 + identity 2-entry input/output
    tables; the 3-D CLUT carries the whole RGB->XYZ(D50) transform. `clut_xyz` is
    (grid,grid,grid,3) XYZ D50 (Y=1), indexed [R][G][B][channel]."""
    body = struct.pack('>4sI', b'mft2', 0)
    body += struct.pack('>BBBx', 3, 3, grid)              # i=3, o=3, g=grid, pad
    for v in (1, 0, 0, 0, 1, 0, 0, 0, 1):                 # identity 3x3 (s15Fixed16)
        body += struct.pack('>i', _s15f16(v))
    body += struct.pack('>HH', 2, 2)                      # n=2 in, m=2 out table entries
    for _ in range(3):                                    # identity input tables
        body += struct.pack('>HH', 0, 65535)
    enc = np.clip(np.asarray(clut_xyz, dtype=np.float64) / _XYZ_ENC, 0.0, 1.0)
    body += np.rint(enc * 65535.0).astype('>u2').tobytes()  # R slowest, B fastest, ch fastest
    for _ in range(3):                                    # identity output tables
        body += struct.pack('>HH', 0, 65535)
    return body


def build_clut_icc(desc: str, clut_xyz: np.ndarray, grid: int,
                   wtpt: Tuple[float, float, float] = D50_XYZ,
                   copyright_text: str = "Public Domain. No rights reserved.",
                   neutral=None) -> bytes:
    """Build a valid ICC v2.4 RGB->XYZ cLUT (lut16) profile. `clut_xyz` is a
    (grid,grid,grid,3) XYZ D50 (Y=1) table indexed [R][G][B]. Parses back via
    InputProfile.from_bytes (cLUT path).

    neutral: optional camera-native calibration neutral recorded in 'CCRn', as in
    build_matrix_shaper_icc."""
    desc = str(desc).encode('ascii', 'replace').decode('ascii')
    copyright_text = str(copyright_text).encode('ascii', 'replace').decode('ascii')
    tags = {
        'desc': _desc_type(desc),
        'wtpt': _xyz_type(*wtpt),
        'A2B0': _lut16_type(clut_xyz, grid),
        'cprt': _text_type(copyright_text),
    }
    order = ['desc', 'A2B0', 'wtpt', 'cprt']
    nv = green_normalised(neutral)
    if nv is not None:
        tags[_CCR_NEUTRAL_SIG.decode('ascii')] = _xyz_type(*nv)
        order.append(_CCR_NEUTRAL_SIG.decode('ascii'))
    n = len(order)
    base = 128 + 4 + n * 12
    data = b''
    entries = []
    seen: dict = {}
    for name in order:
        payload = tags[name]
        key = bytes(payload)
        if key in seen:
            off, ln = seen[key]
        else:
            if len(data) % 4:
                data += b'\x00' * (4 - len(data) % 4)
            off = base + len(data)
            ln = len(payload)
            data += payload
            seen[key] = (off, ln)
        entries.append((name.encode('ascii'), off, ln))
    table = struct.pack('>I', n)
    for tag, off, ln in entries:
        table += struct.pack('>4sII', tag, off, ln)
    total = 128 + len(table) + len(data)
    header = bytearray(128)
    struct.pack_into('>I', header, 0, total)
    struct.pack_into('>4s', header, 4, b'lcms')
    struct.pack_into('>I', header, 8, 0x02400000)         # v2.4.0
    struct.pack_into('>4s', header, 12, b'mntr')
    struct.pack_into('>4s', header, 16, b'RGB ')
    struct.pack_into('>4s', header, 20, b'XYZ ')
    struct.pack_into('>4s', header, 36, b'acsp')
    struct.pack_into('>i', header, 68, _s15f16(wtpt[0]))
    struct.pack_into('>i', header, 72, _s15f16(wtpt[1]))
    struct.pack_into('>i', header, 76, _s15f16(wtpt[2]))
    return bytes(header) + table + data

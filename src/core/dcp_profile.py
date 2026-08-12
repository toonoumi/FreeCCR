"""
Adobe DCP (DNG camera profile) support — parse, apply, generate. Pure numpy +
struct (no rawpy DNG API / DNG SDK). See spec/dcp-camera-profile.md.

A .dcp is a single-IFD TIFF stream carrying DNG camera-profile tags. The profile
operates on CAMERA-NATIVE raw (the space read_image already produces for the ICC
path) and, like InputProfile.apply, outputs LINEAR Adobe RGB so the negative
inversion downstream reads consistent linear data.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from core import color_management as cm


class DcpError(Exception):
    """Raised when a .dcp can't be parsed or is invalid (surfaced to the user)."""


# EXIF LightSource enum -> approximate correlated colour temperature (kelvin).
ILLUMINANT_CCT: Dict[int, float] = {
    0: 5500, 1: 5500, 2: 4150, 3: 2856, 4: 5500,    # default/daylight/fluor/tungsten/flash
    9: 5500, 10: 5500, 11: 5500,
    17: 2856, 18: 4874, 19: 6774, 20: 5503, 21: 6504, 22: 7504, 23: 5003, 24: 3200,
}

# DNG profile tag ids (decimal) — verified against the Adobe DNG Spec 1.4-1.6.
_T_COLORMATRIX1 = 50721; _T_COLORMATRIX2 = 50722
_T_CAMERACALIB1 = 50723; _T_CAMERACALIB2 = 50724
_T_ANALOGBAL = 50727; _T_ASSHOTNEUTRAL = 50728
_T_ILLUM1 = 50778; _T_ILLUM2 = 50779
_T_PROFILENAME = 50936
_T_HSMDIMS = 50937; _T_HSMDATA1 = 50938; _T_HSMDATA2 = 50939
_T_TONECURVE = 50940; _T_EMBEDPOLICY = 50941
_T_LOOKDIMS = 50981; _T_LOOKDATA = 50982
_T_HSMENC = 51107; _T_LOOKENC = 51108
_T_FORWARDMATRIX1 = 50964; _T_FORWARDMATRIX2 = 50965

_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}


@dataclass
class DcpProfile:
    name: str = ""
    color_matrix_1: Optional[np.ndarray] = None      # (3,3) XYZ(D50) -> camera, illum 1
    color_matrix_2: Optional[np.ndarray] = None
    forward_matrix_1: Optional[np.ndarray] = None    # (3,3) wb-camera -> XYZ D50, illum 1
    forward_matrix_2: Optional[np.ndarray] = None
    camera_calibration_1: Optional[np.ndarray] = None
    camera_calibration_2: Optional[np.ndarray] = None
    analog_balance: Optional[np.ndarray] = None
    illuminant_1: int = 21                            # default D65
    illuminant_2: Optional[int] = None
    hsm_dims: Optional[Tuple[int, int, int]] = None
    hsm_data_1: Optional[np.ndarray] = None           # (h,s,v,3): dHue, sScale, vScale
    hsm_data_2: Optional[np.ndarray] = None
    hsm_encoding: int = 0
    look_dims: Optional[Tuple[int, int, int]] = None
    look_data: Optional[np.ndarray] = None
    look_encoding: int = 0
    tone_curve: Optional[np.ndarray] = None           # (N,2) (x,y) in [0,1]
    embed_policy: int = 1
    # AsShotNeutral (50728): the camera-native neutral this profile was calibrated
    # on, green-normalised. FreeCCR writes it into the profiles it generates so the
    # WB is a property of the profile, not of each frame's metadata; third-party
    # DCPs carry no such tag and keep the per-frame path. See apply_dcp.
    as_shot_neutral: Optional[np.ndarray] = None

    @property
    def has_forward(self) -> bool:
        return self.forward_matrix_1 is not None

    @property
    def is_dual(self) -> bool:
        return self.color_matrix_2 is not None and self.illuminant_2 is not None


# --------------------------------------------------------------------------- #
# Parse.
# --------------------------------------------------------------------------- #

def parse_dcp(path: str) -> DcpProfile:
    with open(path, "rb") as f:
        return parse_dcp_bytes(f.read())


def _read_values(data: bytes, field: bytes, typ: int, cnt: int, en: str):
    size = _TYPE_SIZE.get(typ)
    if size is None:
        return None
    total = size * cnt
    if total <= 4:
        raw = field[:total]
    else:
        off = struct.unpack(en + 'I', field)[0]
        if off + total > len(data):
            raise DcpError("tag value offset past end of file")
        raw = data[off:off + total]
    if typ == 2:                                       # ASCII
        return raw.split(b'\x00')[0].decode('ascii', 'replace')
    if typ == 3:                                       # SHORT
        return list(struct.unpack(en + '%dH' % cnt, raw))
    if typ == 4:                                       # LONG
        return list(struct.unpack(en + '%dI' % cnt, raw))
    if typ == 5:                                       # RATIONAL
        nd = struct.unpack(en + '%dI' % (2 * cnt), raw)
        return [nd[2 * i] / nd[2 * i + 1] if nd[2 * i + 1] else 0.0 for i in range(cnt)]
    if typ == 10:                                      # SRATIONAL
        nd = struct.unpack(en + '%di' % (2 * cnt), raw)
        return [nd[2 * i] / nd[2 * i + 1] if nd[2 * i + 1] else 0.0 for i in range(cnt)]
    if typ == 11:                                      # FLOAT
        return list(struct.unpack(en + '%df' % cnt, raw))
    return None


def _read_ifd(data: bytes, off: int, en: str) -> Dict[int, object]:
    if off + 2 > len(data):
        raise DcpError("truncated IFD")
    count = struct.unpack(en + 'H', data[off:off + 2])[0]
    entries: Dict[int, object] = {}
    p = off + 2
    for _ in range(count):
        if p + 12 > len(data):
            raise DcpError("truncated IFD entry")
        tag, typ, cnt = struct.unpack(en + 'HHI', data[p:p + 8])
        entries[tag] = _read_values(data, data[p + 8:p + 12], typ, cnt, en)
        p += 12
    return entries


def parse_dcp_bytes(data: bytes) -> DcpProfile:
    if len(data) < 8:
        raise DcpError("file too short to be a DCP")
    bo = data[0:2]
    if bo == b'II':
        en = '<'
    elif bo == b'MM':
        en = '>'
    else:
        raise DcpError("not a TIFF/DCP (bad byte-order mark)")
    if struct.unpack(en + 'H', data[2:4])[0] != 42:
        raise DcpError("not a TIFF/DCP (bad magic)")
    ifd_off = struct.unpack(en + 'I', data[4:8])[0]
    e = _read_ifd(data, ifd_off, en)

    def mat(tag):
        v = e.get(tag)
        if v is None:
            return None
        a = np.asarray(v, dtype=np.float64)
        if a.size != 9:
            raise DcpError(f"matrix tag {tag} has {a.size} values (expected 9)")
        return a.reshape(3, 3)

    def vec(tag, n):
        v = e.get(tag)
        return np.asarray(v, dtype=np.float64) if v is not None and len(v) >= n else None

    p = DcpProfile()
    p.color_matrix_1 = mat(_T_COLORMATRIX1)
    p.color_matrix_2 = mat(_T_COLORMATRIX2)
    p.forward_matrix_1 = mat(_T_FORWARDMATRIX1)
    p.forward_matrix_2 = mat(_T_FORWARDMATRIX2)
    cc1 = mat(_T_CAMERACALIB1); cc2 = mat(_T_CAMERACALIB2)
    p.camera_calibration_1 = cc1 if cc1 is not None else np.eye(3)
    p.camera_calibration_2 = cc2 if cc2 is not None else np.eye(3)
    ab = vec(_T_ANALOGBAL, 3)
    p.analog_balance = ab[:3] if ab is not None else np.ones(3)
    p.as_shot_neutral = cm.green_normalised(vec(_T_ASSHOTNEUTRAL, 3))
    if e.get(_T_ILLUM1):                              # truthy guards count-0 tags ([] is falsy)
        p.illuminant_1 = int(e[_T_ILLUM1][0])
    if e.get(_T_ILLUM2):
        p.illuminant_2 = int(e[_T_ILLUM2][0])
    if isinstance(e.get(_T_PROFILENAME), str):
        p.name = e[_T_PROFILENAME]
    if e.get(_T_EMBEDPOLICY):
        p.embed_policy = int(e[_T_EMBEDPOLICY][0])

    def hsv_table(dims_tag, data_tag):
        dims = e.get(dims_tag)
        dat = e.get(data_tag)
        if not dims or dat is None:
            return None, None
        if len(dims) < 3:
            raise DcpError("HueSat/Look dims must have 3 values")
        dims = tuple(int(x) for x in dims[:3])
        if len(dat) != 3 * dims[0] * dims[1] * dims[2]:
            raise DcpError("HueSat/Look table size mismatch")
        return dims, np.asarray(dat, dtype=np.float64).reshape(*dims, 3)

    p.hsm_dims, p.hsm_data_1 = hsv_table(_T_HSMDIMS, _T_HSMDATA1)
    _, p.hsm_data_2 = hsv_table(_T_HSMDIMS, _T_HSMDATA2)
    p.look_dims, p.look_data = hsv_table(_T_LOOKDIMS, _T_LOOKDATA)
    if e.get(_T_HSMENC):
        p.hsm_encoding = int(e[_T_HSMENC][0])
    if e.get(_T_LOOKENC):
        p.look_encoding = int(e[_T_LOOKENC][0])
    tc = e.get(_T_TONECURVE)
    if tc is not None and len(tc) >= 4 and len(tc) % 2 == 0:
        p.tone_curve = np.asarray(tc, dtype=np.float64).reshape(-1, 2)

    if p.color_matrix_1 is None and p.forward_matrix_1 is None:
        raise DcpError("DCP has no ColorMatrix1 or ForwardMatrix1 — unusable")
    return p


# --------------------------------------------------------------------------- #
# Apply (camera-native unbalanced raw -> linear Adobe RGB).
# --------------------------------------------------------------------------- #

_D50 = np.array(cm.D50_XYZ)


def _xy(xyz: np.ndarray) -> Tuple[float, float]:
    s = float(np.sum(xyz))
    if s <= 1e-9:
        return 0.3457, 0.3585
    return float(xyz[0] / s), float(xyz[1] / s)


def _mccamy_cct(x: float, y: float) -> float:
    denom = (0.1858 - y)
    if abs(denom) < 1e-9:
        return 6504.0
    nn = (x - 0.3320) / denom
    return 449 * nn ** 3 + 3525 * nn ** 2 + 6823.3 * nn + 5520.33


def _neutral_cct(profile: DcpProfile, m: np.ndarray) -> float:
    """CCT of the as-shot neutral (camera RGB = 1/m) via the ColorMatrices."""
    n = 1.0 / np.clip(m, 1e-6, None)
    n = n / np.max(n)
    cms = [c for c in (profile.color_matrix_1, profile.color_matrix_2) if c is not None]
    ts = []
    for cmat in cms:
        try:
            xyz = np.linalg.inv(cmat) @ n
        except np.linalg.LinAlgError:
            continue
        ts.append(_mccamy_cct(*_xy(xyz)))
    return float(np.mean(ts)) if ts else 6504.0


def _interp_weight(profile: DcpProfile, m: np.ndarray) -> float:
    """Mired-space blend weight on illuminant-set 1 (1.0 if single-illuminant)."""
    if not profile.is_dual:
        return 1.0
    T = _neutral_cct(profile, m)
    T1 = ILLUMINANT_CCT.get(profile.illuminant_1, 6504.0)
    T2 = ILLUMINANT_CCT.get(profile.illuminant_2, 2856.0)
    inv = lambda t: 1e6 / t
    denom = inv(T1) - inv(T2)
    if abs(denom) < 1e-9:
        return 1.0
    return float(np.clip((inv(T) - inv(T2)) / denom, 0.0, 1.0))


def _blend(a, b, w):
    if a is None:
        return b
    if b is None:
        return a
    return w * a + (1.0 - w) * b


def apply_dcp(profile: DcpProfile, rgb_u16: np.ndarray, *,
              as_shot_wb: Optional[np.ndarray] = None,
              apply_look: bool = False) -> np.ndarray:
    """Camera-native unbalanced raw (uint16) -> uint16 LINEAR Adobe RGB.

    A profile that records the neutral it was CALIBRATED on (AsShotNeutral — every
    profile the IT8 wizard generates) balances every frame with that, so a fixed
    copy-stand setup renders identically across a roll whatever the camera's WB
    mode was. Otherwise the frame's as_shot_wb (length-3 camera-native WB
    multipliers, raw.camera_whitebalance) is used, and None -> unity (a degraded
    path — a DCP is a raw profile). apply_look enables the subjective
    HueSatMap/LookTable/ToneCurve (off by default)."""
    if rgb_u16.ndim != 3 or rgb_u16.shape[2] != 3 or rgb_u16.dtype != np.uint16:
        return rgb_u16
    d = rgb_u16.astype(np.float64) / 65535.0
    # Green-normalised WB diagonal. Normalising matters because raw as-shot values
    # are gains (e.g. [2366,1024,1640]); un-normalised they multiply the image by
    # ~1000x and blow it to white. This is the DNG reference-neutral diagonal.
    m = cm.resolve_wb_gains(profile.as_shot_neutral, as_shot_wb)
    if m is None:
        m = np.ones(3)

    w = _interp_weight(profile, m)
    FM = _blend(profile.forward_matrix_1, profile.forward_matrix_2, w)
    CM = _blend(profile.color_matrix_1, profile.color_matrix_2, w)
    CC = _blend(profile.camera_calibration_1, profile.camera_calibration_2, w)
    AB = np.diag(profile.analog_balance[:3])
    inv_cc_ab = np.linalg.inv(CC @ AB)

    d_wb = d * m                                      # as-shot white balance (unclipped)
    if FM is not None:
        xyz = d_wb @ (FM @ inv_cc_ab).T               # ForwardMatrix -> XYZ D50
    else:
        # ColorMatrix-only fallback: inv(CM) on the white-balanced camera RGB
        # (colorimetrically weaker than the ForwardMatrix path; CM is D50-referenced).
        xyz = d_wb @ np.linalg.inv(CM @ AB).T

    if apply_look:
        xyz = _apply_look(profile, xyz, w)

    adobe = xyz @ cm.M_XYZ_D50_2_ADOBE.T
    return np.rint(np.clip(adobe, 0.0, 1.0) * 65535.0).astype(np.uint16)


# --------------------------------------------------------------------------- #
# Optional "look" stage (HueSatMap / LookTable / ToneCurve), apply_look only.
# --------------------------------------------------------------------------- #

def _rgb_to_hsv(rgb: np.ndarray):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1); mn = np.min(rgb, axis=-1); df = mx - mn
    h = np.zeros_like(mx)
    nz = df > 1e-12
    im = (mx == r) & nz
    h[im] = (60.0 * ((g - b)[im] / df[im])) % 360.0
    im = (mx == g) & nz
    h[im] = 60.0 * ((b - r)[im] / df[im]) + 120.0
    im = (mx == b) & nz
    h[im] = 60.0 * ((r - g)[im] / df[im]) + 240.0
    s = np.where(mx > 1e-12, df / np.maximum(mx, 1e-12), 0.0)
    return h % 360.0, s, mx


def _hsv_to_rgb(h, s, v):
    h = h % 360.0
    c = v * s
    x = c * (1 - np.abs((h / 60.0) % 2 - 1))
    m = v - c
    z = np.zeros_like(h)
    seg = (h // 60).astype(int) % 6
    r = np.select([seg == 0, seg == 1, seg == 2, seg == 3, seg == 4, seg == 5],
                  [c, x, z, z, x, c])
    g = np.select([seg == 0, seg == 1, seg == 2, seg == 3, seg == 4, seg == 5],
                  [x, c, c, x, z, z])
    b = np.select([seg == 0, seg == 1, seg == 2, seg == 3, seg == 4, seg == 5],
                  [z, z, x, c, c, x])
    return np.stack([r + m, g + m, b + m], axis=-1)


def _hsv_map(xyz: np.ndarray, dims, data: np.ndarray, encoding: int) -> np.ndarray:
    """Apply a DNG HueSat/Look table (hue°,satScale,valScale) in linear ProPhoto."""
    pp = np.clip(xyz @ cm.M_XYZ2PROPHOTO.T, 0.0, None)
    h, s, v = _rgb_to_hsv(pp)
    hd, sd, vd = dims
    v_enc = cm.srgb_encode(np.clip(v, 0, 1)) if encoding == 1 else np.clip(v, 0, 1)
    # fractional table coordinates (hue wraps, sat/val clamp)
    hf = (h / 360.0) * hd
    sf = np.clip(s, 0, 1) * (sd - 1)
    vf = np.clip(v_enc, 0, 1) * (vd - 1) if vd > 1 else np.zeros_like(v)
    h0 = np.floor(hf).astype(int) % hd
    h1 = (h0 + 1) % hd
    s0 = np.clip(np.floor(sf).astype(int), 0, sd - 1); s1 = np.minimum(s0 + 1, sd - 1)
    v0 = np.clip(np.floor(vf).astype(int), 0, vd - 1); v1 = np.minimum(v0 + 1, vd - 1)
    fh = (hf - np.floor(hf))[..., None]
    fs = (sf - s0)[..., None]
    fv = (vf - v0)[..., None]

    def cell(hi, si, vi):
        return data[hi, si, vi]
    c000 = cell(h0, s0, v0); c100 = cell(h1, s0, v0)
    c010 = cell(h0, s1, v0); c110 = cell(h1, s1, v0)
    c001 = cell(h0, s0, v1); c101 = cell(h1, s0, v1)
    c011 = cell(h0, s1, v1); c111 = cell(h1, s1, v1)
    c00 = c000 * (1 - fh) + c100 * fh
    c10 = c010 * (1 - fh) + c110 * fh
    c01 = c001 * (1 - fh) + c101 * fh
    c11 = c011 * (1 - fh) + c111 * fh
    c0 = c00 * (1 - fs) + c10 * fs
    c1 = c01 * (1 - fs) + c11 * fs
    delta = c0 * (1 - fv) + c1 * fv                   # (...,3): dHue, sScale, vScale

    h2 = h + delta[..., 0]
    s2 = np.clip(s * delta[..., 1], 0.0, 1.0)
    v2 = v * delta[..., 2]
    rgb = _hsv_to_rgb(h2, s2, v2)
    return rgb @ np.linalg.inv(cm.M_XYZ2PROPHOTO).T


def _apply_tone(xyz: np.ndarray, tone: np.ndarray) -> np.ndarray:
    """Apply the ProfileToneCurve on the ProPhoto value channel (max RGB)."""
    pp = np.clip(xyz @ cm.M_XYZ2PROPHOTO.T, 0.0, None)
    v = np.max(pp, axis=-1, keepdims=True)
    xs = np.clip(tone[:, 0], 0, 1); ys = np.clip(tone[:, 1], 0, 1)
    order = np.argsort(xs)
    nv = np.interp(np.clip(v, 0, 1), xs[order], ys[order])
    scale = np.where(v > 1e-6, nv / np.maximum(v, 1e-6), 1.0)
    return (pp * scale) @ np.linalg.inv(cm.M_XYZ2PROPHOTO).T


def _apply_look(profile: DcpProfile, xyz: np.ndarray, w: float) -> np.ndarray:
    out = xyz
    if profile.hsm_data_1 is not None and profile.hsm_dims is not None:
        data = _blend(profile.hsm_data_1, profile.hsm_data_2, w)
        out = _hsv_map(out, profile.hsm_dims, data, profile.hsm_encoding)
    if profile.look_data is not None and profile.look_dims is not None:
        out = _hsv_map(out, profile.look_dims, profile.look_data, profile.look_encoding)
    if profile.tone_curve is not None:
        out = _apply_tone(out, profile.tone_curve)
    return out


# --------------------------------------------------------------------------- #
# Generate a minimal single-illuminant matrix DCP from an IT8 CameraFit.
# --------------------------------------------------------------------------- #

def _to_srational(x: float, den: int = 1000000) -> Tuple[int, int]:
    return int(round(x * den)), den


def _to_rational(x: float, den: int = 1000000) -> Tuple[int, int]:
    """Unsigned RATIONAL (TIFF type 5). Negatives are clamped to 0 — the only
    unsigned tag written is AsShotNeutral, whose values are positive by
    construction."""
    return max(int(round(x * den)), 0), den


def _write_ifd(tags: Dict[int, Tuple[str, object]]) -> bytes:
    """Build a little-endian single-IFD TIFF from {tag: (kind, value)}. Tags are
    emitted in ascending order; values >4 bytes go to a word-aligned data block."""
    items = sorted(tags.items())
    n = len(items)
    ifd_size = 2 + n * 12 + 4
    data_base = 8 + ifd_size
    entry_bytes = b''
    data_bytes = b''
    for tag, (kind, val) in items:
        if kind == 'ascii':
            raw = val.encode('ascii', 'replace') + b'\x00'
            tcode, cnt = 2, len(raw)
        elif kind == 'short':
            tcode, cnt = 3, len(val)
            raw = b''.join(struct.pack('<H', int(v) & 0xFFFF) for v in val)
        elif kind == 'long':
            tcode, cnt = 4, len(val)
            raw = b''.join(struct.pack('<I', int(v) & 0xFFFFFFFF) for v in val)
        elif kind == 'srational':
            tcode, cnt = 10, len(val)
            raw = b''.join(struct.pack('<ii', *_to_srational(v)) for v in val)
        elif kind == 'rational':
            tcode, cnt = 5, len(val)
            raw = b''.join(struct.pack('<II', *_to_rational(v)) for v in val)
        else:
            raise ValueError(kind)
        if len(raw) <= 4:
            field = raw + b'\x00' * (4 - len(raw))
        else:
            field = struct.pack('<I', data_base + len(data_bytes))
            data_bytes += raw
            if len(data_bytes) % 2:
                data_bytes += b'\x00'
        entry_bytes += struct.pack('<HHI', tag, tcode, cnt) + field
    header = struct.pack('<2sHI', b'II', 42, 8)
    ifd = struct.pack('<H', n) + entry_bytes + struct.pack('<I', 0)
    return header + ifd + data_bytes


def build_camera_dcp(fit, name: str, *, illuminant: int = 23,
                     bake_neutral: bool = True) -> bytes:
    """Synthesise a minimal single-illuminant matrix DCP from the IT8 fit.

    The fit's matrix M maps WHITE-BALANCED camera RGB -> XYZ D50 with M@(1,1,1)=D50,
    which is exactly the DNG ForwardMatrix. The ColorMatrix (XYZ D50 -> un-balanced
    reference camera RGB) is inv(M @ diag(wb)) — NOT inv(ForwardMatrix); the two
    differ by the white-balance diagonal (DNG spec 1.6, ch.6). Parses back via
    parse_dcp_bytes.

    bake_neutral (default) records the chart's camera-native neutral (1/wb) as
    AsShotNeutral, so FreeCCR balances every frame on the setup the profile was
    calibrated on rather than on each frame's as-shot metadata (see apply_dcp).
    The tag is inert in Adobe/RawTherapee, which read only profile tags; pass
    False for a portable DCP that defers to the host's per-frame WB."""
    M = np.asarray(fit.matrix, dtype=np.float64)
    wb = np.asarray(fit.wb_mult, dtype=np.float64)[:3]  # raw -> balanced (green=1)
    FM = M                                             # ForwardMatrix: balanced cam -> XYZ D50
    # Un-balanced raw -> XYZ is M @ diag(wb) (balanced = raw*wb), so XYZ -> raw is
    # its inverse. CameraNeutral = CM @ D50 = 1/wb (the green-normalised raw neutral).
    CM = np.linalg.inv(M @ np.diag(wb))                # XYZ D50 -> un-balanced camera
    name = str(name).encode('ascii', 'replace').decode('ascii')
    tags = {
        _T_PROFILENAME: ('ascii', name),
        _T_EMBEDPOLICY: ('long', [1]),                # allow copying
        _T_ILLUM1: ('short', [int(illuminant)]),
        _T_COLORMATRIX1: ('srational', list(CM.ravel())),
        _T_FORWARDMATRIX1: ('srational', list(FM.ravel())),
    }
    if bake_neutral:
        tags[_T_ASSHOTNEUTRAL] = ('rational', list(1.0 / wb))
    return _write_ifd(tags)

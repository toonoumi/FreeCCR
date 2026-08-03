"""
Field correction (flat-field) profiles — pure numpy/cv2, no Qt.

A field-correction profile is a smooth per-channel GAIN MAP measured from one
photograph of an evenly lit blank surface (ideally the light source the user
scans on, through the same lens at the same aperture). Multiplying a decoded
frame by that map removes everything multiplicative in the imaging chain at
once: lens vignetting, sensor colour shading, and light-source unevenness.

The map is measured in — and applied to — the decode that `read_image` produces,
so measurement space == application space (see spec/flat-field-correction.md).
Profiles are stored as `.ffc` (JSON + base64 float32 grid) in the per-user
library and activated globally, mirroring how `color_management` holds the
active camera profile so `ccr_image` can reach it without importing the backend.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# Stored grid long side. The measured field's real bandwidth is a few cycles
# across the frame, so 128 samples over-resolves it while keeping the file
# ~180 KB; it is upsampled bicubically at apply time.
GRID_LONG_SIDE = 128
# Working resolution for validation (§5.2) — high enough that scene content is
# unmistakable, low enough that sensor noise is already averaged out.
ANALYSIS_LONG_SIDE = 256

# Gain clamp. +2 EV is past any real camera-scan rig; beyond it the corner is
# mostly read noise and a botched calibration shot (lens cap, half-covered
# panel) could otherwise produce a map that destroys every image.
DEFAULT_MAX_GAIN = 4.0
MIN_GAIN = 0.25
# Smoothed-field floor (normalised units) — stops a black corner from dividing
# by ~0 before the clamp gets a chance to bite.
_FIELD_FLOOR = 1e-4

FORMAT = "freeccr-field-correction"
FORMAT_VERSION = 1
EXTENSION = ".ffc"

_FULL = 65535.0
# Rows per band in the apply loop: bounds the float32 temporary to a few MB
# even on a 6000x4000 export.
_BAND_ROWS = 512


class FieldProfileError(Exception):
    """Raised when a .ffc file is missing, malformed, or from a newer format
    version, or when a calibration frame cannot yield a usable profile. The
    caller surfaces the message to the user."""


# --------------------------------------------------------------------------- #
# 1. Analysis of a calibration frame
# --------------------------------------------------------------------------- #

@dataclass
class FieldAnalysis:
    """What the wizard's review page reports about a calibration frame."""
    level: float                  # robust central level, 0..1 (max over channels)
    clipped_fraction: float       # worst channel's fraction of pixels at the ceiling
    flatness_residual: float      # relative RMS of (frame - smoothed frame)
    warnings: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Whether a profile can be built at all (warnings are advisory; this is
        the hard floor — below 1% of full scale the map is pure noise gain)."""
        return self.level >= 0.01


def _to_float(arr_u16: np.ndarray) -> np.ndarray:
    """16-bit RGB -> float32 in [0,1], always 3-channel."""
    if arr_u16 is None:
        raise FieldProfileError("No image data.")
    a = np.asarray(arr_u16)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=2)
    if a.ndim != 3 or a.shape[2] < 3:
        raise FieldProfileError(
            f"Expected an RGB image, got shape {getattr(a, 'shape', None)}.")
    return a[..., :3].astype(np.float32) / _FULL


def _resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    """Aspect-preserving downsample with INTER_AREA (a box average — already a
    strong low-pass, which is most of the noise/dust rejection)."""
    h, w = img.shape[:2]
    if max(h, w) <= long_side:
        return img.astype(np.float32, copy=True)
    if h >= w:
        nh, nw = long_side, max(1, int(round(w * long_side / h)))
    else:
        nw, nh = long_side, max(1, int(round(h * long_side / w)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)


def _pad_odd(a: np.ndarray, p: int) -> np.ndarray:
    """Pad by ODD (antisymmetric) reflection: f(-k) = 2*f(0) - f(k).

    Every OpenCV border mode holds the field flat (REPLICATE) or folds it back
    on itself (REFLECT) outside the frame. Both understate a falloff that is
    still dropping at the frame edge, so any filter biases the border *upward* —
    which reads as less vignetting and under-corrects the corners, the exact
    region this feature exists to measure. Odd reflection continues the local
    slope instead (it is exact for a locally linear field), so filtering is
    unbiased at the border. Measured: it removes a ~1.2% corner error."""
    for axis in (0, 1):
        n = a.shape[axis]
        k = min(p, n - 1)
        if k < 1:
            continue
        first = np.take(a, [0], axis=axis)
        lead = 2.0 * first - np.flip(np.take(a, np.arange(1, k + 1), axis=axis),
                                     axis=axis)
        last = np.take(a, [-1], axis=axis)
        trail = 2.0 * last - np.flip(np.take(a, np.arange(-k - 1, -1), axis=axis),
                                     axis=axis)
        if k < p:                       # tiny input: repeat the outermost value
            lead = np.concatenate([np.repeat(np.take(lead, [0], axis=axis),
                                             p - k, axis=axis), lead], axis=axis)
            trail = np.concatenate([trail, np.repeat(np.take(trail, [-1], axis=axis),
                                                     p - k, axis=axis)], axis=axis)
        a = np.concatenate([lead, a, trail], axis=axis)
    return np.ascontiguousarray(a, dtype=np.float32)


def _smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that does not bias the frame border (see _pad_odd)."""
    p = max(2, int(np.ceil(3.0 * float(sigma))))
    padded = cv2.GaussianBlur(_pad_odd(grid, p), (0, 0), sigmaX=float(sigma),
                              sigmaY=float(sigma), borderType=cv2.BORDER_REPLICATE)
    return padded[p:p + grid.shape[0], p:p + grid.shape[1]]


# Blemish-rejection window, as a fraction of the grid's long side. 0.13 removes
# dust/fibres/smudges up to ~6% of the frame across; measured fidelity cost on a
# clean field is ~0.1%. Bigger windows reject bigger smudges but a blemish that
# large is no longer distinguishable from real shading.
BLEMISH_WINDOW_FRAC = 0.13


def _median_window(grid: np.ndarray, k: int) -> np.ndarray:
    """k x k median over the grid (build-time only, so a strided view is fine).
    Run per channel to keep the temporary small."""
    p = k // 2
    padded = _pad_odd(grid, p)
    out = np.empty_like(grid, dtype=np.float32)
    for c in range(grid.shape[2]):
        win = np.lib.stride_tricks.sliding_window_view(padded[..., c], (k, k))
        out[..., c] = np.median(win, axis=(-2, -1))
    return out


def _reject_blemishes(grid: np.ndarray) -> np.ndarray:
    """Remove dust, fibres and hot pixels from the measured field without
    touching the falloff underneath: a 5x5 median for isolated hot/dead pixels,
    then a LARGE median for everything bigger.

    A median is the right robust estimator here because it is exact on monotone
    data: a genuine shading edge — the fast corner cut of a mechanical vignette —
    passes through untouched, while an isolated blob smaller than the window is
    replaced by the surrounding field. The alternatives were measured and
    rejected: a Gaussian wide enough to hide a smudge flattens the corners it is
    meant to measure, morphological closing only reaches blemishes smaller than
    its own kernel, and iterated clipped smoothing (reject by DEPTH rather than
    size) mistook a sharp vignette knee for a blemish and smeared it — it
    rewrote 54% of the grid on such a field.

    Baking a blemish into a profile would put a permanent BRIGHT spot in every
    corrected scan, which is much worse than leaving one dark speck alone."""
    g = cv2.medianBlur(_pad_odd(grid, 2), 5)[2:2 + grid.shape[0],
                                             2:2 + grid.shape[1]]
    k = max(3, int(round(GRID_LONG_SIDE * BLEMISH_WINDOW_FRAC)) | 1)
    return _median_window(np.ascontiguousarray(g, dtype=np.float32), k)


def _central_level(grid: np.ndarray) -> np.ndarray:
    """Per-channel median over the central window (middle 10% of frame area).
    This is the normalisation reference: gain == 1 here, in every channel, so a
    profile changes neither exposure nor white balance — only spatial variation."""
    h, w = grid.shape[:2]
    # sqrt(0.10) ~= 0.316 of each dimension -> 10% of the area.
    ch, cw = max(1, int(round(h * 0.316))), max(1, int(round(w * 0.316)))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    window = grid[y0:y0 + ch, x0:x0 + cw]
    return np.median(window.reshape(-1, window.shape[2]), axis=0).astype(np.float32)


def analyze_field(arr_u16: np.ndarray) -> FieldAnalysis:
    """Validate a decoded calibration frame: is it bright enough, unclipped, and
    actually a flat field rather than a photograph of a scene? Warnings are
    advisory (the user's rig, the user's call); `usable` is the hard floor."""
    f = _to_float(arr_u16)
    work = _resize_long_side(f, ANALYSIS_LONG_SIDE)
    smooth = _smooth(work, max(1.0, max(work.shape[:2]) / 12.0))

    level = float(np.max(_central_level(smooth)))
    clipped = float(np.max(np.mean(f >= 0.99, axis=(0, 1))))
    # Relative to the global level rather than per-pixel (a per-pixel ratio
    # explodes on a near-black frame and would report noise as structure).
    residual = float(np.sqrt(np.mean((work - smooth) ** 2)) / max(level, 1e-6))

    warnings: List[str] = []
    if level < 0.05:
        warnings.append(
            "The calibration shot is very dark — the corrected corners will be "
            "noisy. Re-shoot it brighter (around half to three-quarters of full "
            "exposure).")
    elif level > 0.95:
        warnings.append(
            "The calibration shot is over-exposed. Re-shoot it darker so no "
            "area reaches pure white.")
    if clipped > 0.001:
        warnings.append(
            f"{clipped * 100:.1f}% of pixels are clipped to white — the falloff "
            "cannot be measured there. Re-shoot with less exposure.")
    if residual > 0.05:
        warnings.append(
            "This doesn't look like an evenly lit blank surface — it has too "
            "much detail. Shoot the bare light source (slightly defocused), not "
            "a scene or a frame of film.")
    return FieldAnalysis(level=level, clipped_fraction=clipped,
                         flatness_residual=residual, warnings=warnings)


# --------------------------------------------------------------------------- #
# 2. Building the profile
# --------------------------------------------------------------------------- #

@dataclass
class FieldProfile:
    """A measured gain map plus the metadata that says what it was measured on."""
    name: str
    gain: np.ndarray                       # (gh, gw, 3) float32
    aspect: float                          # source frame w/h
    meta: Dict[str, Any] = field(default_factory=dict)
    path: Optional[str] = None
    content_id: Optional[str] = None
    # Resized-map memo, keyed by (w, h). A pure function of (gain, w, h), so a
    # race between decode worker threads only ever duplicates work.
    _cache: Dict[tuple, np.ndarray] = field(default_factory=dict, repr=False,
                                            compare=False)

    @property
    def stats(self) -> Dict[str, Any]:
        return self.meta.get("stats", {})

    def summary(self) -> str:
        """One-line description for the Settings library row."""
        st = self.stats
        cs = st.get("corner_stops") or [0.0, 0.0, 0.0]
        cam = self.meta.get("camera", {}) or {}
        bits = [b for b in (cam.get("model") or cam.get("make"),
                            _lens_label(cam)) if b]
        head = " · ".join(bits)
        tail = (f"max +{max(cs):.1f} EV · colour spread "
                f"{st.get('cast_stops', 0.0):.2f} EV")
        return f"{head} · {tail}" if head else tail


def _lens_label(cam: Dict[str, Any]) -> str:
    """'105 mm f/8' / the lens model, whichever the EXIF actually had."""
    parts = []
    if cam.get("lens_model"):
        parts.append(str(cam["lens_model"]))
    if cam.get("focal_length"):
        parts.append(f"{float(cam['focal_length']):g} mm")
    if cam.get("aperture"):
        parts.append(f"f/{float(cam['aperture']):g}")
    return " ".join(parts)


def _corner_stops(gain: np.ndarray) -> List[float]:
    """Per-channel mean gain over the four corner regions (outer 5% x 5%), in
    EV — 'how many stops of falloff this profile lifts out of the corners'."""
    h, w = gain.shape[:2]
    ch, cw = max(2, int(round(h * 0.05))), max(2, int(round(w * 0.05)))
    corners = np.concatenate([
        gain[:ch, :cw].reshape(-1, 3), gain[:ch, -cw:].reshape(-1, 3),
        gain[-ch:, :cw].reshape(-1, 3), gain[-ch:, -cw:].reshape(-1, 3)])
    return [float(np.log2(max(v, 1e-6))) for v in corners.mean(axis=0)]


def build_profile(arr_u16: np.ndarray, name: str, *,
                  meta: Optional[Dict[str, Any]] = None,
                  max_gain: float = DEFAULT_MAX_GAIN) -> FieldProfile:
    """Measure the gain map of a decoded calibration frame.

    INTER_AREA downsample -> 3x3 median (hot pixels, surviving dust specks) ->
    Gaussian blur (the field's true bandwidth) -> divide the central reference
    level by it. See spec/flat-field-correction.md §5.3."""
    f = _to_float(arr_u16)
    src_h, src_w = f.shape[:2]
    analysis = analyze_field(arr_u16)
    if not analysis.usable:
        raise FieldProfileError(
            "The calibration shot is essentially black — nothing to measure. "
            "Shoot the lit surface itself, exposed to about half to "
            "three-quarters of full brightness.")

    # Blemish rejection does the heavy lifting; the Gaussian that follows only
    # de-staircases the median, so it stays MINIMAL. A blur wide enough to hide
    # dust on its own would measurably flatten the corners it is supposed to
    # measure (blurring a curved field biases it by ~sigma^2/2 * laplacian, worst
    # exactly at the corners): at sigma=2 a sharp mechanical-vignette knee was
    # reproduced to 10%, at sigma=1 to 6%.
    grid = _reject_blemishes(_resize_long_side(f, GRID_LONG_SIDE))
    smooth = _smooth(grid, max(1.0, GRID_LONG_SIDE / 128.0))

    ref = _central_level(smooth)                       # (3,) gain == 1 here
    gain = ref[None, None, :] / np.maximum(smooth, _FIELD_FLOOR)
    reached = float(np.max(gain))
    gain = np.clip(gain, MIN_GAIN, float(max_gain)).astype(np.float32)
    gain = np.ascontiguousarray(gain)

    corner = _corner_stops(gain)
    stats = {
        "max_gain": round(float(np.max(gain)), 4),
        "max_gain_uncapped": round(reached, 4),
        "clamped": bool(reached > max_gain + 1e-6),
        "corner_stops": [round(c, 4) for c in corner],
        "cast_stops": round(float(max(corner) - min(corner)), 4),
        "level": round(analysis.level, 4),
        "clipped_fraction": round(analysis.clipped_fraction, 6),
        "flatness_residual": round(analysis.flatness_residual, 6),
    }
    full_meta: Dict[str, Any] = dict(meta or {})
    full_meta["stats"] = stats
    full_meta["max_gain_cap"] = float(max_gain)
    full_meta.setdefault("created", datetime.now().isoformat(timespec="seconds"))
    full_meta.setdefault("warnings", analysis.warnings)
    return FieldProfile(name=name or "Field correction", gain=gain,
                        aspect=float(src_w) / float(max(1, src_h)),
                        meta=full_meta)


# --------------------------------------------------------------------------- #
# 3. Storage (.ffc = JSON + base64 float32 grid)
# --------------------------------------------------------------------------- #

def _profile_dict(p: FieldProfile) -> Dict[str, Any]:
    g = np.ascontiguousarray(p.gain, dtype="<f4")
    gh, gw = g.shape[:2]
    doc = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "name": p.name,
        "aspect": round(float(p.aspect), 6),
        "grid": {"w": int(gw), "h": int(gh), "channels": 3, "dtype": "float32",
                 "data": base64.b64encode(g.tobytes()).decode("ascii")},
    }
    doc.update({k: v for k, v in p.meta.items() if k not in doc})
    return doc


def save_profile(profile: FieldProfile, path: str) -> str:
    """Write a .ffc. Returns the path actually written."""
    path = os.path.normpath(path)
    folder = os.path.dirname(os.path.abspath(path))
    if folder:
        os.makedirs(folder, exist_ok=True)
    blob = json.dumps(_profile_dict(profile), indent=1, ensure_ascii=False)
    with open(path, "wb") as f:
        f.write(blob.encode("utf-8"))
    profile.path = path
    profile.content_id = _content_id(path)
    return path


def _content_id(path: str) -> Optional[str]:
    """Short content hash — makes the grading signature distinguish two profiles
    that share a name."""
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:12]
    except OSError:
        return None


def load_profile(path: str) -> FieldProfile:
    """Read and validate a .ffc file. Raises FieldProfileError with a message
    meant for the user."""
    try:
        with open(os.path.normpath(path), "rb") as f:
            raw = f.read()
    except OSError as e:
        raise FieldProfileError(f"Could not read the profile: {e}") from e
    try:
        doc = json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise FieldProfileError(
            f"This is not a valid field-correction profile: {e}") from e
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise FieldProfileError(
            "This file is not a FreeCCR field-correction profile.")
    version = doc.get("version")
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise FieldProfileError(
            f"This profile was written by a newer version of FreeCCR "
            f"(format {version}). Update FreeCCR to use it.")

    grid = doc.get("grid")
    if not isinstance(grid, dict):
        raise FieldProfileError("The profile has no gain map.")
    try:
        gw, gh = int(grid["w"]), int(grid["h"])
        data = base64.b64decode(grid["data"], validate=True)
        gain = np.frombuffer(data, dtype="<f4")
    except Exception as e:
        raise FieldProfileError(f"The profile's gain map is corrupt: {e}") from e
    if gw < 2 or gh < 2 or gain.size != gw * gh * 3:
        raise FieldProfileError("The profile's gain map is the wrong size.")
    gain = np.ascontiguousarray(gain.reshape(gh, gw, 3).astype(np.float32))
    if not np.all(np.isfinite(gain)) or float(np.min(gain)) <= 0.0:
        raise FieldProfileError("The profile's gain map has invalid values.")

    meta = {k: v for k, v in doc.items()
            if k not in ("format", "version", "name", "aspect", "grid")}
    aspect = doc.get("aspect")
    try:
        aspect = float(aspect)
    except (TypeError, ValueError):
        aspect = float(gw) / float(gh)
    return FieldProfile(
        name=str(doc.get("name") or os.path.splitext(os.path.basename(path))[0]),
        gain=gain, aspect=aspect, meta=meta, path=path,
        content_id=_content_id(path))


# --------------------------------------------------------------------------- #
# 4. Applying a profile
# --------------------------------------------------------------------------- #

def gain_for_size(profile: FieldProfile, w: int, h: int) -> np.ndarray:
    """The gain map resized to (w, h), memoised per size. Bicubic: the map is
    smooth, and bicubic avoids the faint slope-discontinuity facets bilinear can
    leave when a 128-px grid is stretched to 6000 px."""
    key = (int(w), int(h))
    cached = profile._cache.get(key)
    if cached is not None:
        return cached
    g = profile.gain
    out = (g if g.shape[:2] == (h, w) else
           cv2.resize(g, (int(w), int(h)), interpolation=cv2.INTER_CUBIC))
    out = np.ascontiguousarray(out, dtype=np.float32)
    if len(profile._cache) >= 4:                # preview / thumb / zoom / export
        profile._cache.pop(next(iter(profile._cache)), None)
    profile._cache[key] = out
    return out


def _srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92,
                    np.power((v + 0.055) / 1.055, 2.4))


def _linear_to_srgb(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.0031308, v * 12.92,
                    1.055 * np.power(np.maximum(v, 0.0), 1.0 / 2.4) - 0.055)


def apply_field(arr_u16: np.ndarray, profile: Optional[FieldProfile], *,
                encoded: bool = False, mono: bool = False) -> np.ndarray:
    """Multiply a decoded 16-bit RGB frame by the profile's gain map.

    `encoded=True` for an sRGB-gamma-encoded decode (Positive mode): linearise,
    multiply, re-encode — a gain measured in linear light and applied straight to
    gamma-encoded data OVER-corrects, by roughly the gamma exponent (multiplying
    an encoded value by g scales the light it stands for by g^2.4).
    `mono=True` for monochrome/grayscale sources: the three channel gains are
    averaged so the frame stays neutral.

    Runs in horizontal bands so a full-resolution export never materialises a
    float32 copy of the whole frame."""
    if profile is None or arr_u16 is None:
        return arr_u16
    a = np.asarray(arr_u16)
    if a.ndim != 3 or a.shape[2] != 3:
        return arr_u16                        # not an RGB frame — leave it alone
    h, w = a.shape[:2]
    gain = gain_for_size(profile, w, h)
    if mono:
        gain = gain.mean(axis=2, keepdims=True)

    out = np.empty_like(a, dtype=np.uint16)
    for y in range(0, h, _BAND_ROWS):
        y1 = min(h, y + _BAND_ROWS)
        band = a[y:y1].astype(np.float32)
        g = gain[y:y1]
        if encoded:
            # Only this path needs normalised units (the sRGB transfer functions).
            band /= _FULL
            band = _linear_to_srgb(_srgb_to_linear(band) * g)
            band *= _FULL
        else:
            # Straight multiply in the frame's own units — no /65535 round trip,
            # which would cost a rounding step for nothing.
            band *= g
        np.clip(band, 0.0, _FULL, out=band)
        out[y:y1] = band.astype(np.uint16)
    return out


# --------------------------------------------------------------------------- #
# 5. Global active profile (mirrors color_management's module globals, so
#    ccr_image can reach it without importing the backend singleton).
# --------------------------------------------------------------------------- #

_active_profile: Optional[FieldProfile] = None


def set_active_profile(profile: Optional[FieldProfile]) -> None:
    global _active_profile
    _active_profile = profile


def get_active_profile() -> Optional[FieldProfile]:
    return _active_profile


def field_profile_active() -> bool:
    return _active_profile is not None


def active_field_signature() -> str:
    """Stable id of the field correction a decode would get now — composed into
    the per-image grading signature so a change flags the affected thumbnails."""
    if _active_profile is None:
        return "none"
    return "ff:" + (_active_profile.content_id or _active_profile.name or "?")


# --------------------------------------------------------------------------- #
# 6. Capture-side helpers (used by the wizard)
# --------------------------------------------------------------------------- #

def decode_calibration(path: str, sample_max: int = 1024) -> Optional[np.ndarray]:
    """Decode a calibration shot to BARE device RGB — camera-native, raw-linear,
    unbalanced, no camera profile and (critically) no field correction, so
    profiles can never compound. Same decode contract as the IT8 profiling path
    (spec/it8-camera-profile.md §5.1); `apply_input_icc=False` is what pins it.

    Built on a bare CCRImage (__new__) so the heavyweight __init__ decode +
    thumbnail work is skipped."""
    from core.ccr_image import CCRImage
    img = CCRImage.__new__(CCRImage)
    img.source_ops = []
    return img.read_image(path, preview=True, max_long_side=sample_max,
                          positive_override=False, apply_input_icc=False)


def source_metadata(path: str) -> Dict[str, Any]:
    """Camera/lens EXIF for the profile's metadata block (best effort — a
    calibration shot with no readable EXIF still makes a valid profile)."""
    cam: Dict[str, Any] = {}
    try:
        from core.ccr_image import CCRImage
        info = CCRImage.get_camera_and_lens_for_lensfun(path) or {}
        for key in ("camera_make", "camera_model", "lens_make", "lens_model",
                    "focal_length", "aperture"):
            val = info.get(key)
            if val not in (None, ""):
                cam[key.replace("camera_", "")] = val
    except Exception as e:                     # EXIF is a nicety, never a blocker
        logging.debug(f"No EXIF for calibration shot {path}: {e}")
    return {"source": os.path.basename(path), "camera": cam}


def suggested_name(meta: Dict[str, Any]) -> str:
    """Default profile name from what we know about the shot."""
    cam = (meta or {}).get("camera", {}) or {}
    lens = _lens_label(cam)
    body = cam.get("model") or cam.get("make") or ""
    head = " ".join(b for b in (body, lens) if b).strip()
    return f"Field {head} {datetime.now():%Y-%m-%d}".replace("  ", " ").strip() \
        if head else f"Field correction {datetime.now():%Y-%m-%d}"

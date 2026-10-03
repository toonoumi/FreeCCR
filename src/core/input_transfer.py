"""Input transfer function for non-RAW decodes.

FreeCCR's density math assumes LINEAR data: the black-point-only inversion takes
`log10(base/v)` and the no-anchor inversion `-log10(16*p)` with fixed constants
calibrated against linear values. The non-RAW reader historically interpreted no
transfer function at all, so a gamma-encoded TIFF was inverted as though it were
linear transmission.

This module resolves what a file's own metadata says about its encoding and
applies the inverse, inside the reader — so preview, hi-res zoom, export, slice,
B/W sampling and the flat-field build all inherit the same decode.

TRANSFER-FUNCTION ONLY: no primaries/matrix conversion is ever done here (an
Adobe RGB-tagged TIFF is linearised but keeps its primaries), matching the
Cineon → workspace stage. A full colorimetric conversion is the input-ICC /
camera-profile path's job. See spec/input-transfer-function.md.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple
import logging
import os

import numpy as np

from core import color_management

# --- transfer tokens -------------------------------------------------------- #
# A token is what gets stored on the image and in the catalog. "embedded" means
# "resolve from this file's metadata on every read" — reproducible because the
# catalog record is invalidated when the file's size/mtime changes, so the bytes
# the tag is read from cannot drift underneath it.
LINEAR = "linear"
SRGB = "srgb"
REC709 = "rec709"
EMBEDDED = "embedded"
GAMMA_PREFIX = "gamma:"
LUT = "lut"          # resolved-only: a table TRC, never stored as a choice

# Adobe RGB (1998) transfer: 563/256 exactly.
ADOBE_RGB_GAMMA = 563.0 / 256.0

# The manual combo's offering (value, label), in display order.
TRANSFER_CHOICES: List[Tuple[str, str]] = [
    (SRGB, "sRGB"),
    (GAMMA_PREFIX + "2.2", "Gamma 2.2"),
    (GAMMA_PREFIX + "1.8", "Gamma 1.8"),
    (GAMMA_PREFIX + "2.4", "Gamma 2.4"),
    (REC709, "Rec.709"),
    (LINEAR, "Linear (no conversion)"),
]

TIFF_EXTS = (".tif", ".tiff")

# TIFF/EXIF tags consulted, in priority order (see resolve_file_transfer).
_TAG_ICC = 34675          # InterColorProfile
_TAG_TRANSFERFUNCTION = 301


@dataclass
class ResolvedTransfer:
    """What a file's metadata says, reduced to something applicable."""
    token: str = LINEAR
    luts: Optional[List[np.ndarray]] = field(default=None, repr=False)
    label: str = "linear"
    tagged: bool = False      # did metadata actually say anything?

    @property
    def identity(self) -> bool:
        return is_identity(self.token)


def is_identity(transfer: Optional[str]) -> bool:
    """True when the transfer needs no work at all (so the reader can return the
    decoded array untouched — the structural guarantee that an untagged file and
    every pre-existing catalog decode byte-identically to before this feature)."""
    if not transfer or transfer == LINEAR:
        return True
    g = gamma_of(transfer)
    return g is not None and abs(g - 1.0) < 1e-6


def gamma_of(transfer: Optional[str]) -> Optional[float]:
    """The exponent of a "gamma:<g>" token, or None for any other token."""
    if not transfer or not transfer.startswith(GAMMA_PREFIX):
        return None
    try:
        return float(transfer[len(GAMMA_PREFIX):])
    except ValueError:
        return None


def describe(transfer: Optional[str]) -> str:
    """Short human label for a token (dialog summaries, log lines)."""
    if not transfer or transfer == LINEAR:
        return "linear"
    if transfer == SRGB:
        return "sRGB"
    if transfer == REC709:
        return "Rec.709"
    if transfer == EMBEDDED:
        return "from metadata"
    if transfer == LUT:
        return "embedded curve"
    g = gamma_of(transfer)
    if g is not None:
        if abs(g - ADOBE_RGB_GAMMA) < 1e-4:
            return "Adobe RGB (gamma 2.2)"
        return f"gamma {g:g}"
    return transfer


def is_tiff(path) -> bool:
    return os.path.splitext(str(path))[1].lower() in TIFF_EXTS


# --- resolving a file's own encoding ---------------------------------------- #

def resolve_file_transfer(path: str) -> ResolvedTransfer:
    """What `path`'s metadata says its encoding is.

    Priority: embedded ICC (authoritative — real TRC curves) > TIFF
    TransferFunction LUT > EXIF ColorSpace/InteropIndex > EXIF Gamma > linear.
    Reads only headers, never pixel data. Any failure degrades to linear with a
    warning: a metadata quirk must never be able to fail an import."""
    path = os.path.normpath(str(path))
    for probe in (_from_icc, _from_transferfunction, _from_exif):
        try:
            got = probe(path)
        except Exception as e:
            logging.warning(f"Input transfer probe {probe.__name__} failed for "
                            f"{os.path.basename(path)}: {e}")
            continue
        if got is not None:
            return got
    return ResolvedTransfer()


def _tiff_tag(path: str, tag):
    """One TIFF tag's value from the first page, or None. Header-only read."""
    try:
        import tifffile
    except ImportError:
        return None
    with tifffile.TiffFile(path) as tf:
        t = tf.pages[0].tags.get(tag)
        return None if t is None else t.value


def _from_icc(path: str) -> Optional[ResolvedTransfer]:
    """An embedded ICC's TRCs, as three device->linear LUTs."""
    if not is_tiff(path):
        return None
    icc = _tiff_tag(path, _TAG_ICC)
    if not icc:
        return None
    if isinstance(icc, (tuple, list)):
        icc = bytes(bytearray(icc))
    luts = color_management.transfer_luts_from_icc(bytes(icc))
    if luts is None:
        return None
    return ResolvedTransfer(token=LUT, luts=luts,
                            label=color_management.icc_trc_label(bytes(icc)),
                            tagged=True)


def _from_transferfunction(path: str) -> Optional[ResolvedTransfer]:
    """TIFF tag 301: an explicit per-channel encoded->linear table. Stored as
    1 or 3 channels of 2**BitsPerSample uint16 entries (0..65535)."""
    if not is_tiff(path):
        return None
    raw = _tiff_tag(path, _TAG_TRANSFERFUNCTION)
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64).ravel()
    if arr.size == 0:
        return None
    n = arr.size // 3 if arr.size % 3 == 0 and arr.size >= 3 else arr.size
    chans = ([arr[0:n], arr[n:2 * n], arr[2 * n:3 * n]]
             if arr.size >= 3 * n else [arr[:n]] * 3)
    luts = [_resample_lut(c / 65535.0) for c in chans]
    return ResolvedTransfer(token=LUT, luts=luts,
                            label="embedded curve (TransferFunction)",
                            tagged=True)


def _from_exif(path: str) -> Optional[ResolvedTransfer]:
    """EXIF ColorSpace (1 = sRGB; Uncalibrated + InteropIndex R03 = Adobe RGB),
    then EXIF Gamma. exifread is already a dependency and reads TIFF natively."""
    try:
        import exifread
    except ImportError:
        return None
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=False)
    if not tags:
        return None

    def _first_int(name):
        t = tags.get(name)
        vals = getattr(t, "values", None) if t is not None else None
        if isinstance(vals, (list, tuple)) and vals:
            try:
                return int(vals[0])
            except (TypeError, ValueError):
                return None
        return None

    cs = _first_int("EXIF ColorSpace")
    if cs == 1:
        return ResolvedTransfer(token=SRGB, label="sRGB", tagged=True)
    if cs == 65535:
        interop = str(tags.get("Interoperability InteropIndex", "")).strip()
        if interop.upper() == "R03":
            return ResolvedTransfer(token=GAMMA_PREFIX + repr(ADOBE_RGB_GAMMA),
                                    label="Adobe RGB (gamma 2.2)", tagged=True)
    g = tags.get("EXIF Gamma")
    vals = getattr(g, "values", None) if g is not None else None
    if isinstance(vals, (list, tuple)) and vals:
        try:
            gv = float(vals[0].num) / float(vals[0].den)
        except AttributeError:
            try:
                gv = float(vals[0])
            except (TypeError, ValueError):
                gv = 0.0
        except ZeroDivisionError:
            gv = 0.0
        if gv > 0:
            token = GAMMA_PREFIX + f"{gv:g}"
            return ResolvedTransfer(token=token, label=describe(token),
                                    tagged=True)
    return None


def _resample_lut(samples: np.ndarray, n: int = 65536) -> np.ndarray:
    """Stretch a short encoded->linear table to n entries, so a uint16 value
    indexes it directly (same contract as color_management's TRC LUTs)."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == n:
        return samples
    src = np.linspace(0.0, 1.0, samples.size)
    return np.interp(np.linspace(0.0, 1.0, n), src, samples)


# --- applying --------------------------------------------------------------- #

def apply_transfer(arr: Optional[np.ndarray], transfer: Optional[str], *,
                   file_path: Optional[str] = None,
                   positive: bool = False) -> Optional[np.ndarray]:
    """Convert a decoded uint16 frame out of `transfer` into the space the
    pipeline expects, and return uint16 full-range (the reader's contract).

    Negative mode -> LINEAR (what the density math needs). Positive mode ->
    sRGB DISPLAY encoding, because that path returns display-referred pixels and
    the adjustments own the look; a file already tagged sRGB is then an exact
    identity rather than a lossy round trip.

    An identity transfer returns the SAME object — the no-op is structural, so
    untagged files and every pre-existing catalog are bit-for-bit unaffected."""
    if arr is None or arr.dtype != np.uint16:
        return arr
    token, luts = _resolve(transfer, file_path)
    if is_identity(token):
        return arr
    if positive and token == SRGB:
        return arr                      # already in the positive path's space
    lin = _to_linear(arr, token, luts)
    if lin is None:
        return arr
    if positive:
        lin = color_management.srgb_encode(lin)
    np.clip(lin, 0.0, 1.0, out=lin)
    lin *= np.float32(65535.0)
    return (lin + np.float32(0.5)).astype(np.uint16)


def _resolve(transfer: Optional[str], file_path: Optional[str]):
    """(token, luts) for a stored transfer — resolving "embedded" against the
    file, which is why the reader passes its path in."""
    if transfer == EMBEDDED:
        if not file_path:
            return LINEAR, None
        r = resolve_file_transfer(file_path)
        return r.token, r.luts
    return (transfer or LINEAR), None


def _to_linear(arr: np.ndarray, token: str,
               luts: Optional[List[np.ndarray]]) -> Optional[np.ndarray]:
    """Decode uint16 encoded values to float32 linear [0,1] (in a fresh array)."""
    if token == LUT:
        if not luts:
            return None
        out = np.empty(arr.shape, dtype=np.float32)
        if arr.ndim == 3:
            for c in range(arr.shape[2]):
                lut = luts[c if c < len(luts) else -1].astype(np.float32)
                out[..., c] = lut[arr[..., c]]
        else:
            out[...] = luts[0].astype(np.float32)[arr]
        return out
    v = arr.astype(np.float32) / np.float32(65535.0)
    if token == SRGB:
        return color_management.srgb_decode(v).astype(np.float32)
    if token == REC709:
        return color_management.rec709_decode(v).astype(np.float32)
    g = gamma_of(token)
    if g is None:
        logging.warning(f"Unknown input transfer {token!r} — read as linear")
        return None
    np.power(v, np.float32(g), out=v)
    return v


# --- batch summary (for the import dialog) ---------------------------------- #

def tiffs_in(paths: Sequence[str]) -> List[str]:
    """The TIFFs in a batch that this feature may reinterpret — FreeCCR's OWN
    baked linear merge TIFFs are excluded. Those are genuinely linear and
    deliberately untagged (spec/trichrome-linear-tiff-export.md), so offering to
    re-interpret them is the one way this could corrupt data the app owns."""
    from core import ccr_merge
    out = []
    for p in paths:
        if not is_tiff(p):
            continue
        try:
            if ccr_merge.is_freeccr_merge_tiff(p):
                continue
        except Exception:
            pass
        out.append(p)
    return out


def summarize_batch(paths: Sequence[str]) -> Tuple[int, List[str]]:
    """(tagged count, human summary parts) for the TIFFs in `paths` — the dialog
    describes files with the SAME resolver the decode uses, so it can never
    claim something different from how the file will actually be read."""
    labels = {}
    tagged = 0
    for p in paths:
        r = resolve_file_transfer(p)
        if r.tagged:
            tagged += 1
        key = r.label if r.tagged else "untagged"
        labels[key] = labels.get(key, 0) + 1
    parts = [f"{n} {name}" for name, n in
             sorted(labels.items(), key=lambda kv: (-kv[1], kv[0]))]
    return tagged, parts

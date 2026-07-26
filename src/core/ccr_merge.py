"""
3-way RGB-light merge — trichrome (three-colour) capture.

The user shoots one static scene three times under a single pure light each —
red, then green, then blue — and every consecutive triplet of source RAWs is
merged into one full-colour image by taking each frame's OWN colour channel and
discarding the other two, WITHOUT demosaicing.

Two sensor kinds are supported:

* **Bayer (RGGB)** — read the RAW Bayer mosaic directly and take ONLY the
  wanted colour's photosites, with NO demosaic and NO libraw colour pipeline, and
  never mixing in the OTHER colours — so there is zero inter-channel crosstalk.
  R and B use their single site per quad; green averages its two same-colour sites
  (not crosstalk — both are green — and it preserves the green SNR). A Bayer merge
  is half-sensor resolution (one site per quad = its full resolution). NB: libraw
  `half_size=True` gives a numerically similar result, but it goes through the
  postprocess pipeline; reading the mosaic directly keeps full control and
  guarantees no hidden colour operation.

* **Monochrome** — no CFA at all, so there is nothing to demosaic: every
  photosite measured the (single) light's intensity. The whole grayscale frame
  IS that frame's channel, at FULL sensor resolution. A monochrome sensor is in
  fact the ideal trichrome sensor (no wasted photosites, full resolution).

Either way each frame contributes exactly one channel (R from the red-light
frame, G from green, B from blue), scaled to 16-bit by 65535/white_level.

This module is split so everything except `merge_raw_channels` is pure and
unit-testable without rawpy/Qt. See spec/three-way-rgb-merge.md.
"""
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Exactly read_image's rawpy-Bayer-decodable extension set (ccr_image.py:434,
# == export_estimator.RAW_EXTS). Deliberately NOT the broader folder glob, and
# excludes .fff (read_image treats it as TIFF, not a CFA). Only files this set
# covers can be channel-extracted, so anything else is rejected in merge mode.
RAW_EXTENSIONS = frozenset({
    ".cr3", ".cr2", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf",
    ".srw", ".pef", ".3fr",
})

# Number of source frames per merged image (red, green, blue).
MERGE_GROUP_SIZE = 3

# Marker stamped into the Software tag of a FreeCCR-generated 3-way-merge linear
# TIFF (see ccr_backend.bake_merged_to_linear_tiff). It lets such a file be
# re-opened while 3-way merge mode is ON without toggling the mode off: the
# loader recognises a marked TIFF and loads it as a normal image instead of
# treating it as a (RAW) merge input. See spec/merge-linear-tiff-replace.md.
FREECCR_MERGE_TIFF_MARKER = "FreeCCR:3-way-RGB-merge-linear-v1"


def is_freeccr_merge_tiff(path) -> bool:
    """True when `path` is a TIFF FreeCCR wrote as a baked 3-way-merge result
    (carries FREECCR_MERGE_TIFF_MARKER in its Software tag). Reads only the TIFF
    header; a non-TIFF, unreadable, or unmarked file returns False."""
    if os.path.splitext(str(path))[1].lower() not in (".tif", ".tiff"):
        return False
    try:
        import tifffile
        from core.ccr_processor import safe_unicode_path
        with tifffile.TiffFile(safe_unicode_path(str(path))) as tf:
            tag = tf.pages[0].tags.get("Software")
            value = tag.value if tag is not None else ""
        return isinstance(value, str) and FREECCR_MERGE_TIFF_MARKER in value
    except Exception:
        return False


def is_raw_path(path: str) -> bool:
    """True when the file extension is a RAW format this module can merge."""
    return os.path.splitext(path)[1].lower() in RAW_EXTENSIONS


def sort_for_merge(paths: Sequence[str]) -> List[str]:
    """Order paths by case-insensitive basename (directory ignored), so the
    triplet ordering is determined purely by filename, as the feature requires.
    Ties (same basename in different dirs) keep a stable order via the full
    normalized path as a secondary key."""
    return sorted(paths, key=lambda p: (os.path.basename(p).lower(),
                                        os.path.normcase(p)))


def group_into_triplets(sorted_paths: Sequence[str]) -> List[Tuple[str, str, str]]:
    """Group an already-sorted path list into consecutive (R, G, B) triplets.
    The caller is responsible for validating the count is a multiple of 3 first;
    any trailing remainder (< 3) is dropped here."""
    n = (len(sorted_paths) // MERGE_GROUP_SIZE) * MERGE_GROUP_SIZE
    return [tuple(sorted_paths[i:i + MERGE_GROUP_SIZE])  # type: ignore[misc]
            for i in range(0, n, MERGE_GROUP_SIZE)]


def validate_merge_inputs(paths: Sequence[str]) -> Tuple[bool, Optional[str]]:
    """Pre-decode validation for a merge import. Returns (ok, error_message).

    Checks, in order: non-empty, every file is a supported RAW, count is a
    multiple of 3. The sensor check (Bayer vs monochrome vs unsupported) can only
    happen at decode time (see merge_raw_channels), so it is NOT done here."""
    if not paths:
        return False, "No files selected for 3-way RGB merge."
    non_raw = [p for p in paths if not is_raw_path(p)]
    if non_raw:
        shown = "\n".join(os.path.basename(p) for p in non_raw[:8])
        if len(non_raw) > 8:
            shown += f"\n… and {len(non_raw) - 8} more"
        return False, ("3-way RGB merge requires RAW files. "
                       f"These are not supported RAW:\n\n{shown}")
    if len(paths) % MERGE_GROUP_SIZE != 0:
        return False, ("3-way RGB merge needs a multiple of 3 images "
                       f"(got {len(paths)}). Select 3 frames per shot: "
                       "red, green, blue.")
    return True, None


def bayer_channel_indices(color_desc) -> Tuple[int, int, int]:
    """Map (R, G, B) to the output-channel indices a camera-native (output_color
    =raw) decode emits, which follow libraw's internal colour order == the
    `color_desc` string. For the canonical b'RGBG' this is (0, 1, 2); a permuted
    desc is honoured. Raises ValueError when the sensor is not an R/G/B Bayer
    (e.g. monochrome b'G', or 4-colour b'RGBE'/CYGM), which must be rejected."""
    if isinstance(color_desc, (bytes, bytearray)):
        s = bytes(color_desc).decode("ascii", "ignore").upper()
    else:
        s = str(color_desc).upper()
    r, g, b = s.find("R"), s.find("G"), s.find("B")
    if -1 in (r, g, b):
        raise ValueError(
            f"3-way merge requires a Bayer (RGGB) sensor; color_desc "
            f"{color_desc!r} is not R/G/B (X-Trans, monochrome or 4-colour).")
    return r, g, b


def combine_channels(plane_r: np.ndarray, plane_g: np.ndarray, plane_b: np.ndarray,
                     white_levels: Sequence[float]) -> np.ndarray:
    """Pure merge core: take the red frame's R-plane, the green frame's G-plane,
    and the blue frame's B-plane (each a 2-D array, already the correct channel),
    scale each by 65535/white_level (matching read_image's white-level scaling),
    and stack into one (H, W, 3) uint16 RGB image.

    Planes may differ slightly in size; all are cropped to the common (min H,
    min W). Returns linear RGB in [0, 65535]."""
    planes = [plane_r, plane_g, plane_b]
    if len(white_levels) != 3:
        raise ValueError("white_levels must have 3 entries (R, G, B)")
    h = min(p.shape[0] for p in planes)
    w = min(p.shape[1] for p in planes)
    out = np.empty((h, w, 3), dtype=np.uint16)
    for i, (plane, wl) in enumerate(zip(planes, white_levels)):
        cropped = plane[:h, :w].astype(np.float32)
        scale = 65535.0 / wl if wl and wl > 0 else 1.0
        out[..., i] = np.clip(cropped * scale, 0, 65535).astype(np.uint16)
    return out


def _desc_bytes(color_desc) -> bytes:
    if isinstance(color_desc, (bytes, bytearray)):
        return bytes(color_desc)
    return str(color_desc).encode("ascii", "ignore")


def is_monochrome_sensor(num_colors, color_desc, raw_pattern=None) -> bool:
    """Whether a RAW comes from a monochrome (no-CFA) sensor. Mirrors
    read_image's detection (ccr_image.py): num_colors == 1, a grey color_desc,
    or an RGBG desc whose CFA pattern is all one colour index. Pure (takes the
    rawpy primitives, not a raw object), so it is unit-testable."""
    if num_colors == 1:
        return True
    cd = _desc_bytes(color_desc)
    if cd in (b"G", b"GRAY", b"GREY"):
        return True
    if cd == b"RGBG" and raw_pattern is not None:
        try:
            if int(np.asarray(raw_pattern).max()) == 0:
                return True
        except Exception:
            pass
    return False


def extract_cfa_channel(mosaic: np.ndarray, colors: np.ndarray, color_desc,
                        letter: str, black_levels=None) -> np.ndarray:
    """From a raw Bayer mosaic (2-D sensor read-out) and its per-pixel CFA colour
    indices, return the half-resolution plane for ONE colour `letter` (R/G/B) by
    phase-slicing the 2x2 lattice — NO demosaic, NO colour matrix, and NO mixing
    with the OTHER colours, so there is zero inter-channel crosstalk.

    R and B have a single site per quad → that bare site. Green has two sites per
    quad (same colour, not crosstalk) → their per-site-black-subtracted AVERAGE,
    which keeps the green SNR. Each contributing site is matched by its CFA letter
    via `color_desc` (so it works whether the two greens share one colour index or
    use indices 1 and 3), located by its phase in the tile read from the actual
    `colors` at the visible origin (offset-safe; the CFA is period-2 so
    `mosaic[dy::2, dx::2]` is exactly that phase's sites), and black-subtracted by
    its own colour index when `black_levels` is given. Pure — unit-testable
    without rawpy."""
    eff = np.asarray(colors)[:2, :2]
    desc = _desc_bytes(color_desc).decode("ascii", "ignore").upper()
    letter = letter.upper()
    phases = [(dy, dx, int(eff[dy, dx]))
              for dy in range(eff.shape[0]) for dx in range(eff.shape[1])
              if 0 <= int(eff[dy, dx]) < len(desc) and desc[int(eff[dy, dx])] == letter]
    if not phases:
        raise ValueError(f"CFA colour {letter!r} not present in the 2x2 tile "
                         f"{eff.tolist()} (color_desc {desc!r})")
    m = np.asarray(mosaic)
    subs = [m[dy::2, dx::2].astype(np.float32) for dy, dx, _ in phases]
    h = min(s.shape[0] for s in subs)
    w = min(s.shape[1] for s in subs)
    acc = np.zeros((h, w), dtype=np.float32)
    for (dy, dx, idx), s in zip(phases, subs):
        black = 0.0
        if black_levels is not None and idx < len(black_levels):
            black = float(black_levels[idx])
        acc += s[:h, :w] - black
    return acc / len(phases)            # 1 site for R/B; average of 2 for green


def _decode_frame_plane(path: str, frame_pos: int, preview: bool = False,
                        demosaic: bool = False):
    """Decode one source RAW and return (plane_2d, white_level, is_mono,
    sensor_full) for the single colour this frame contributes (frame_pos
    0=R/1=G/2=B).

    `demosaic` (Bayer only; monochrome has no CFA and ignores it) switches the
    channel extraction from the raw-mosaic phase slice to a full-resolution
    LINEAR (bilinear) demosaic that the frame's channel is then taken from.
    Bilinear interpolates each colour plane only from its OWN photosites, so
    the "each frame contributes only its own channel, no inter-channel
    crosstalk" guarantee still holds — now at full sensor resolution. See
    spec/trichrome-demosaic-mode.md.

    * Bayer (RGGB): read the RAW Bayer mosaic directly (`raw.raw_image_visible`)
      and take ONLY this frame's colour photosites — no demosaic, no libraw
      colour pipeline, and never mixing in the OTHER colours, so there is zero
      inter-channel crosstalk. R/B use their single site per quad; green averages
      its two same-colour sites (not crosstalk — both are green — and it keeps the
      green SNR). The black pedestal is subtracted per site manually (raw_image
      carries it). Half-sensor resolution (one site per quad) is the Bayer merge's
      full resolution; `preview` is irrelevant (the read is already cheap).
    * Monochrome: no CFA — the whole grayscale frame IS the channel, at FULL
      sensor resolution (nothing to demosaic, no quad to mix). `preview` may
      decode at half size for a fast preview/zoom tile, but the canonical full
      resolution reported is always the full sensor.

    Raises ValueError on an unsupported sensor (X-Trans, 4-colour)."""
    import rawpy

    with rawpy.imread(path) as raw:
        white_level = float(raw.white_level)
        sensor_full = (int(raw.sizes.height), int(raw.sizes.width))
        num_colors = int(getattr(raw, "num_colors", 0) or 0)
        color_desc = getattr(raw, "color_desc", b"")
        pattern = getattr(raw, "raw_pattern", None)

        if is_monochrome_sensor(num_colors, color_desc, pattern):
            rgb = raw.postprocess(
                output_bps=16,
                no_auto_bright=True,
                gamma=(1, 1),                                  # linear
                user_flip=0,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                half_size=preview,         # full res unless a fast preview decode
                use_camera_wb=False,
                use_auto_wb=False,
                output_color=rawpy.ColorSpace.raw,
                no_auto_scale=True,        # absolute sensor values; scale manually
                adjust_maximum_thr=0.0,
                four_color_rgb=False,
            )
            plane = rgb if rgb.ndim == 2 else rgb[..., 0]   # channels are equal
            return np.ascontiguousarray(plane), white_level, True, sensor_full

        # Bayer (RGGB): require a 3-colour 2x2 R/G/B mosaic.
        if num_colors != 3:
            raise ValueError(
                f"3-way merge requires a Bayer (RGGB) or monochrome sensor; "
                f"{os.path.basename(path)} reports {num_colors} colours "
                f"(e.g. 4-colour CYGM/RGBE).")
        if pattern is not None and tuple(np.asarray(pattern).shape) != (2, 2):
            raise ValueError(
                f"3-way merge requires a 2x2 Bayer mosaic or a monochrome "
                f"sensor; {os.path.basename(path)} is non-Bayer (e.g. X-Trans).")
        channel_indices = bayer_channel_indices(color_desc)  # raises if not R/G/B
        letter = "RGB"[frame_pos]                 # this frame's colour

        if demosaic:
            # Full-resolution LINEAR demosaic, then take this frame's channel.
            # Identical decode kwargs to the monochrome path above: linear,
            # absolute values (no_auto_scale; combine scales by white_level),
            # black-subtracted by libraw, camera-native. Bilinear interpolates
            # each plane from its own sites only — no inter-channel mixing.
            rgb = raw.postprocess(
                output_bps=16,
                no_auto_bright=True,
                gamma=(1, 1),                                  # linear
                user_flip=0,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                half_size=preview,     # full res unless a fast preview decode
                use_camera_wb=False,
                use_auto_wb=False,
                output_color=rawpy.ColorSpace.raw,
                no_auto_scale=True,    # absolute sensor values; scale manually
                adjust_maximum_thr=0.0,
                four_color_rgb=False,
            )
            plane = rgb[..., channel_indices[frame_pos]]
            return (np.ascontiguousarray(plane), white_level, False,
                    sensor_full)

        # Read the RAW mosaic and pull only this colour's sites (R/B single,
        # green = average of its two sites) with per-site black subtraction;
        # combine then scales 65535/white_level, matching read_image's decode.
        mosaic = np.asarray(raw.raw_image_visible)
        colors = np.asarray(raw.raw_colors_visible)
        try:
            black_levels = list(raw.black_level_per_channel)
        except Exception:
            black_levels = None
        plane = extract_cfa_channel(mosaic, colors, color_desc, letter,
                                    black_levels=black_levels)
        plane = np.clip(plane, 0, None)
        return (np.ascontiguousarray(plane), white_level, False,
                (plane.shape[0], plane.shape[1]))


def merge_raw_channels(sources: Sequence[str], preview: bool = False,
                       demosaic: bool = False) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Merge a (red, green, blue) triplet of RAW files into one (H, W, 3) uint16
    linear-RGB image, taking only each frame's own colour channel.

    `demosaic=False` (single photosite): Bayer frames are mosaic-phase-sliced —
    no demosaic at all — at half-sensor resolution (2x2 bin). `demosaic=True`:
    Bayer frames go through a LINEAR (bilinear, per-channel — still no
    inter-channel mixing) demosaic at FULL sensor resolution. Monochrome
    sensors have no CFA and decode identically in both modes (full sensor).
    `preview` lets a monochrome or demosaic decode run at half size for fast
    preview/zoom (the photosite read is already cheap and ignores it).

    Returns (merged_rgb, full_size=(H, W)) where full_size is the merged image's
    canonical FULL (export) resolution. Raises ValueError on an unsupported
    sensor or a decode failure (the caller surfaces it)."""
    if len(sources) != MERGE_GROUP_SIZE:
        raise ValueError(f"merge_raw_channels needs exactly {MERGE_GROUP_SIZE} "
                         f"sources, got {len(sources)}")
    for p in sources:
        if not os.path.exists(p):
            raise ValueError(f"3-way merge source missing: {p}")

    planes: List[np.ndarray] = []
    white_levels: List[float] = []
    monos: List[bool] = []
    sensor_full: Optional[Tuple[int, int]] = None
    for frame_pos, path in enumerate(sources):       # 0=red, 1=green, 2=blue
        plane, white_level, mono, sfull = _decode_frame_plane(
            path, frame_pos, preview, demosaic=demosaic)
        planes.append(plane)
        white_levels.append(white_level)
        monos.append(mono)
        if sensor_full is None:
            sensor_full = sfull

    # All three frames must be the same sensor type — mixing a monochrome
    # (full-res) frame with Bayer (half-res) ones would silently min-crop into a
    # misaligned merge. A real trichrome set is always one body, so reject.
    any_mono = any(monos)
    if any_mono and not all(monos):
        raise ValueError("3-way merge sources must all be the same sensor type "
                         "(all Bayer, or all monochrome).")

    merged = combine_channels(planes[0], planes[1], planes[2], white_levels)
    # Canonical FULL (export) resolution: full sensor for monochrome and for a
    # demosaiced Bayer merge (both may decode a half-size preview while their
    # export resolution is the full sensor); the 2x2-binned size for a
    # photosite Bayer merge (its only resolution).
    full_size = (sensor_full if (any_mono or demosaic)
                 else (merged.shape[0], merged.shape[1]))
    return merged, full_size

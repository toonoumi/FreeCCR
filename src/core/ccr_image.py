from typing import Any, Dict, Optional, List
import numpy as np
import os
import copy
import rawpy
import exifread
import cv2
import logging
import time
from PySide6.QtCore import QCoreApplication, QThread
from PySide6.QtGui import QImage, QPixmap  # or from PySide6.QtGui import QImage, QPixmap if you use PySide
#import lensfunpy  # Make sure lensfunpy is installed
from core.ccr_processor import (adjust_image, adjust_image_opencl,
                                BAND_ADJUSTMENT_KEYS, apply_curves,
                                apply_gamma_curve, apply_cineon_to_rec709,
                                apply_area_layers, apply_crop_to_image,
                                apply_dust_removal, DUST_FEATHER_DEFAULT,
                                _DUST_PLAN_LONG,
                                _apply_working_space_recovery,
                                compute_auto_gain_offset)
from core import color_management

# Import optional libraries with fallbacks
try:
    import tifffile
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False
    logging.warning("tifffile not available, TIFF reading may be limited")

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logging.warning("PIL/Pillow not available, some image formats may not be supported")


def _read_tiff_bgr(file_path: str):
    """Decode a TIFF with tifffile and return it in OpenCV's BGR channel
    order (any dtype). Handles the layouts OpenCV chokes on with scanner
    TIFFs (Pakon / Nikon Coolscan): planar storage (PlanarConfiguration=2,
    which tifffile returns as (C, H, W)), alpha channels (dropped), and
    exotic sample formats; compressed files decode via imagecodecs."""
    arr = tifffile.imread(file_path)
    if arr is None:
        return None
    arr = np.asarray(arr)
    if (arr.ndim == 3 and arr.shape[0] in (3, 4)
            and arr.shape[2] not in (3, 4)):
        arr = np.moveaxis(arr, 0, -1)   # planar (C, H, W) -> (H, W, C)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[..., :3]              # drop alpha
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = arr[..., ::-1]            # RGB -> BGR, dtype-agnostic
    return np.ascontiguousarray(arr)


def _to_uint16_full_range(img: np.ndarray) -> np.ndarray:
    """Normalize a decoded image to the pipeline's uint16 full-range
    contract. Scanner TIFFs arrive in many sample formats; float (0..1) and
    wide/signed integer data previously passed through untouched and
    rendered black or half-range downstream."""
    if img.dtype == np.uint16:
        return img
    if img.dtype == np.uint8:
        return img.astype(np.uint16) * 257
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img.astype(np.float32) * 65535.0 + 0.5,
                       0, 65535).astype(np.uint16)
    info = np.iinfo(img.dtype)
    scale = 65535.0 / float(max(int(info.max), 1))
    return np.clip(img.astype(np.float64) * scale, 0, 65535).astype(np.uint16)


class CCRImage:
    def __init__(
        self,
        file_path: str,
        thumbnail: Optional[np.ndarray] = None,         # 8-bit RGB (H, W, 3), dtype=np.uint8
        resized_raw: Optional[np.ndarray] = None,       # 16-bit RGB (H, W, 3), dtype=np.uint16
        # Coordinates as 4 uint32 integers: (x1, y1, x2, y2)
        reference_frame: Optional[tuple[int, int, int, int]] = None,
        adjustment_settings: Optional[Dict[str, Any]] = None,
        rotation_angle: int = 0,
        fine_rotation_angle: int = 0, ##remember to divide by 100 to get the actual angle
        horizontal_mirrored: bool = False,
        vertical_mirrored: bool = False,
        converted: bool = False,
        source_ops: Optional[list] = None,
        preloaded_img: Optional[np.ndarray] = None,
        preloaded_full_size: Optional[tuple[int, int]] = None,
        display_name: Optional[str] = None,
        slice_group: Optional[str] = None,
        slice_parent: Optional[Dict[str, Any]] = None,
        color_profile: str = "color",
        areas: Optional[List[Dict[str, Any]]] = None,
        is_merged: bool = False,
        merge_sources: Optional[list] = None,
        merge_demosaic: bool = True,
        ws_windowed: bool = False,
        ):
        # Normalize file path to handle Unicode characters properly
        self.file_path = os.path.normpath(file_path)
        # 3-way RGB-light merge (trichrome): this image is synthesized from three
        # source RAWs (red, green, blue exposures) by taking each frame's own
        # channel without demosaicing. file_path is the RED source (a real RAW,
        # so EXIF/lensfun work); read_image dispatches to a re-merge whenever the
        # pixels are needed again (export, zoom, slice, duplicate). Session-only:
        # excluded from the per-file edit catalog. See spec/three-way-rgb-merge.md.
        self.is_merged = bool(is_merged)
        self.merge_sources = list(merge_sources) if merge_sources else None
        # How this merge extracts each frame's channel — captured at import
        # (like merge_sources) so every re-read (zoom, export, linear TIFF,
        # slice, duplicate) reproduces the same decode at the same canonical
        # resolution regardless of later Settings changes. True = LINEAR
        # demosaic at full sensor resolution (default); False = raw-mosaic
        # single-photosite read at half resolution. Monochrome ignores it.
        # See spec/trichrome-demosaic-mode.md.
        self.merge_demosaic = bool(merge_demosaic)
        # Sliced images own a chain of slice operations applied to the source
        # file by read_image in every path (preview load, hi-res zoom,
        # full-res export, B/W sampling). Each op is
        #   (rotation_hundredths_deg, (x1, y1, x2, y2))
        # — rotate the current frame about its center (the fine rotation
        # baked at slice time), then crop to the fractional region of that
        # frame. Nested slices simply append ops. [] = whole file.
        self.source_ops: list = list(source_ops) if source_ops else []
        # Display/export name override (e.g. "scan_s2.ARW" for slice #2);
        # None = use the file's basename. A merged image with no explicit name
        # reads as "<redStem>_RGB<ext>" so the list/export filename is sensible.
        if display_name is None and self.is_merged:
            _stem, _ext = os.path.splitext(os.path.basename(self.file_path))
            display_name = f"{_stem}_RGB{_ext}"
        self.display_name = display_name
        # True for working copies made via Duplicate. Duplicates are session
        # artifacts: removing one from the list also removes its catalog
        # entry, whereas removing an ACTUAL image keeps its stored edits.
        self.is_duplicate = False
        # Slice lineage. All slices produced by one slicing operation share
        # the same slice_group id, and carry slice_parent — a snapshot of the
        # canvas they were cut from ({display_name, is_duplicate,
        # slice_group}). "Reset Slice" uses the group to find exactly the
        # round's members (never conflating independent rounds of the same
        # file, e.g. a duplicate sliced separately) and slice_parent to
        # restore the original canvas in their place. None for non-slices.
        self.slice_group: Optional[str] = slice_group
        self.slice_parent: Optional[Dict[str, Any]] = slice_parent
        self.thumbnail = thumbnail
        self.resized_raw = resized_raw
        # Underlying exception from the last failed decode, kept because
        # read_image reports failure as None and would otherwise throw the
        # cause away. The load path chains it onto the ValueError it raises so
        # the loader can explain WHY a file was dropped (see core/load_errors).
        self.last_read_error: Optional[BaseException] = None
        # Camera profile (ICC/DCP/none) this image's decode was graded under;
        # stamped on every working decode so the thumbnail can flag a mismatch
        # when the active profile changes. Not persisted (a reload re-stamps).
        self.profile_signature = None
        self.reference_frame = reference_frame
        self.resized_preview = None  # Placeholder for resized preview, if needed later
        self.adjustment_settings = adjustment_settings if adjustment_settings is not None else {}
        # Color profile: "color" = full RGB (the original behaviour), "bw" =
        # map the adjusted result to a single luminance channel. Affects both
        # the preview/thumbnail and the exported file.
        self.color_profile = color_profile if color_profile in ("color", "bw") else "color"
        self.rotation_angle = rotation_angle
        self.fine_rotation_angle = fine_rotation_angle
        self.horizontal_mirrored = horizontal_mirrored
        self.vertical_mirrored = vertical_mirrored
        self.converted = converted  # Indicates if the image has been converted to CCR format
        # Snapshot of the inputs the CURRENT conversion was baked with —
        # set by the converters, cleared on reload/unconvert. The zoom
        # hi-res replay must use this (not live editable state) so the
        # detail layer always matches the conversion the preview shows.
        # {"mode": "ref", "ref": (x1,y1,x2,y2), "fine_rot": int} or
        # {"mode": "bw", "bw": ((B,G,R),(B,G,R)|None), "fine_rot": int,
        #  "density": bool}   # density: two-point log inversion (see spec)
        self.conversion_inputs: Optional[Dict[str, Any]] = None
        # Feathered white-mask alpha (uint8 H×W, 0..255) marking clear-film
        # (sprocket / rebate) regions, at the preview resolution. Set on every
        # B/W-point conversion; composited to white as the last preview step when
        # the global reversal-look toggle is on. None otherwise (not persisted —
        # a reconvert repopulates it). See spec/sprocket-hole-mask.md.
        self.sprocket_alpha: Optional[np.ndarray] = None
        # User crop as normalized (x1, y1, x2, y2) fractions of the un-rotated/
        # un-flipped image; None = no crop. Display-level only — resized_raw is
        # never modified, so clearing the crop needs no re-conversion.
        # crop_angle rotates the box about its center (degrees, positive =
        # clockwise on screen, Qt convention).
        self.crop_rect: Optional[tuple[float, float, float, float]] = None
        self.crop_angle: float = 0.0
        # Area editing: local masked adjustment layers. Each area is a dict
        # {id, kind ("circle"|"gradient"), enabled, feather, angle, geometry
        # (normalized fractions), settings (a full adjustment_settings dict)}.
        # The global adjustment_settings is the implicit "whole image" layer;
        # areas composite additively on top of it (see spec/area-editing.md).
        self.area_layers: List[Dict[str, Any]] = list(areas) if areas else []
        # Dust removal: non-destructive spot inpainting. Each spot is
        # {kind ("brush"|"auto"), pts ([[x,y],...] normalized over W/H),
        # r (radius as a fraction of WIDTH)}. Rasterized + inpainted at render
        # time (resolution-independent). [] = no dust removal. See
        # spec/dust-removal.md.
        self.dust_spots: List[Dict[str, Any]] = []
        # Edge fade width for the dust heal, as a fraction of each hole's own
        # half-thickness so it scales with the brush (user-set via the dust
        # panel's Feather slider; render parameter, deliberately NOT in undo
        # snapshots — like brush size).
        self.dust_feather: float = DUST_FEATHER_DEFAULT
        # (spots_key, plan) captured from the last PREVIEW-scale heal, replayed
        # by hi-res/export renders so they sample exactly the patches shown on
        # screen. Session cache only — rebuilt by the next preview render.
        self._dust_plan_cache = None
        # Which layer the adjustment panel currently edits: None = global
        # (whole image); otherwise the id of an area in area_layers. Session
        # state only — never persisted; always defaults to global on load.
        self.active_area_id: Optional[str] = None
        self.undo_stack: list = []  # Snapshots for Ctrl+Z, most recent last
        self.contrast_base: int = 0      # Non-destructive base contrast added internally; slider shows 0
        self.temperature_base: int = 0   # Non-destructive base temperature offset; slider shows 0
        # Non-destructive base brightness offset (slider shows 0). The -8 is part
        # of the film-NEGATIVE look; positives go straight to user adjustments
        # from a neutral baseline (no darkening), so 0 there. See spec/positive-mode.md.
        self.brightness_base: int = 0 if self._positive_mode_active() else -8
        # Non-destructive auto-exposure (default-slope mode). Rides the Gain/
        # Exposure argument (NOT ch_input_gain, despite what this comment used
        # to say) → applied as a uniform gain. See spec/auto-exposure-default-slope.md.
        self.exposure_base: float = 0.0
        # True when the current converted base is a WINDOWED working-space buffer
        # (highlight headroom preserved); apply_adjustments de-windows it. Set by
        # the conversion that produced the base, or passed in for a preloaded
        # converted base (slice/duplicate) so the first preview render at the end
        # of __init__ de-windows correctly. See spec/working-space-headroom.md.
        self._ws_windowed: bool = ws_windowed
        self.histogram_data = None   # raw per-channel counts (3, 256) R/G/B, or None
        self.original_full_size: Optional[tuple[int, int]] = None  # (height, width) of the full-res source, set by read_image

        self.info = self.get_camera_and_lens_for_lensfun(self.file_path)  # Extract camera and lens info for lensfun

        # Read image from file and populate resized_raw (downsized to 1080
        # long side inside the reader, before white-level scaling). Slicing
        # passes preloaded_img (the shared parent decode, already cropped to
        # this slice's region) so N slices don't re-decode the file N times.
        if preloaded_img is not None:
            self.original_full_size = preloaded_full_size
            img = self.resize_image_to_max_pixel(preloaded_img, 1080)
            if img is preloaded_img or img.base is not None:
                # Never retain a view of the shared parent decode
                img = img.copy()
        else:
            img = self.read_image(self.file_path, max_long_side=1080)
        if img is not None:
            self.resized_raw = img
            #correct lens distortion and vignetting if possible
            corrected = self.correct_lens_distortion_and_vignette()
            if corrected is not None:
                self.resized_raw = corrected
            else:
                logging.warning(f"Could not correct lens distortion for {self.file_path}, using original resized image.")
            
            # Calculate tint balance factor once during loading
            self.tint_balance_factor = self._calculate_tint_balance_factor()
            self._stamp_profile_signature()

            # Populate thumbnail and preview
            self.update_thumbnail_and_preview()
        else:
            # Chained, not swallowed: the loader classifies the ROOT cause (a
            # LibRaw "unsupported format" vs a missing file) to tell the user
            # why this file was dropped from the import.
            raise ValueError(
                f"Could not read image from file: {self.file_path}"
            ) from self.last_read_error
        print(f"CCRImage initialized: {self.file_path}, info: {self.info}")


    def _calculate_tint_balance_factor(self) -> float:
        """
        Calculate the tint balance factor based on the R/B channel ratio of the original image.
        This only needs to be calculated once during image loading.
        """
        if self.resized_raw is None:
            return 1.0
        
        img_norm = self.resized_raw.astype(np.float32) / 65535.0
        # Calculate R and B channel means in one operation
        rb_means = np.mean(img_norm[..., [0, 2]], axis=(0, 1))  # [r_mean, b_mean]
        current_rb_ratio = rb_means[0] / (rb_means[1] + 1e-8)
        balance_factor = 1.0 + 0.2 * np.tanh((current_rb_ratio - 1.0) * 2)
        
        return balance_factor

    def reload_image_decode_only(self) -> bool:
        """Re-decode resized_raw + tint balance and reset the base offsets,
        WITHOUT building the QPixmap thumbnail/preview (QPixmap must be created
        on the GUI thread). Returns True on success. This is the thread-safe,
        slow part of a reload — the bulk-reset path runs it concurrently across
        a thread pool, then builds previews on the main thread via
        update_thumbnail_and_preview()."""
        self.contrast_base = 0      # Clear base offsets when reverting to original scan
        self.temperature_base = 0
        # -8 is the negative-look baseline; positives reset to a neutral 0 so the
        # decode goes straight to user adjustments (no darkening / shadow crush).
        self.brightness_base = 0 if self._positive_mode_active() else -8
        self.exposure_base = 0.0    # Clear auto-exposure when reverting to original scan
        self._ws_windowed = False   # raw scan is full-range, not a windowed base
        self.conversion_inputs = None
        img = self.read_image(self.file_path, max_long_side=1080)
        if img is None:
            logging.error(f"Failed to reload image: {self.file_path}")
            return False
        self.resized_raw = img
        # Recalculate tint balance factor for the new image
        self.tint_balance_factor = self._calculate_tint_balance_factor()
        self._stamp_profile_signature()
        return True

    def reload_image(self) -> None:
        """
        Reload the image from the file path and update resized_raw, thumbnail, and preview.
        This is useful if the image file has been modified externally.
        """
        if self.reload_image_decode_only():
            self.update_thumbnail_and_preview()

    def _apply_source_ops(self, img: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Apply this image's slice chain to a decoded image: each op rotates
        the current frame about its center (the fine rotation that was baked
        at slice time — the same warp the preview displayed) and then crops
        to a fractional region of that frame. Identity when no ops are set."""
        if not self.source_ops or img is None:
            return img
        for rotation, region in self.source_ops:
            if rotation:
                h, w = img.shape[:2]
                matrix = cv2.getRotationMatrix2D((w // 2, h // 2), -rotation / 100.0, 1.0)
                img = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            h, w = img.shape[:2]
            fx1, fy1, fx2, fy2 = region
            x1 = max(0, min(w - 1, int(round(fx1 * w))))
            y1 = max(0, min(h - 1, int(round(fy1 * h))))
            x2 = max(x1 + 1, min(w, int(round(fx2 * w))))
            y2 = max(y1 + 1, min(h, int(round(fy2 * h))))
            img = img[y1:y2, x1:x2]
        # Materialize: returning a view would pin the entire full-frame
        # decode in long-lived holders (hi-res cache, resized_raw).
        return np.ascontiguousarray(img)

    def _ops_full_size(self, full_hw: tuple) -> tuple:
        """Full-resolution (height, width) after this image's slice chain
        (rotations keep the frame size; regions scale it)."""
        h, w = full_hw
        for _rotation, region in self.source_ops:
            fx1, fy1, fx2, fy2 = region
            h = max(1, int(round((fy2 - fy1) * h)))
            w = max(1, int(round((fx2 - fx1) * w)))
        return (h, w)

    def resize_image_to_max_pixel(self, image: np.ndarray, max_long_side: int) -> np.ndarray:
        """
        Resize the image so that its longest side is equal to max_long_side pixels,
        preserving aspect ratio. Returns the resized image.
        This function does not copy if resizing is not needed; otherwise, returns a new resized reference.
        """
        h, w = image.shape[:2]
        if max(h, w) <= max_long_side:
            return image
        if h > w:
            new_h = max_long_side
            new_w = int(w * max_long_side / h)
        else:
            new_w = max_long_side
            new_h = int(h * max_long_side / w)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Profile+kind pairs already warned about, so a mismatched profile logs once
    # per session instead of once per decoded frame.
    _kind_warned: set = set()

    def _warn_kind_mismatch(self, profile) -> None:
        """Warn when a profile built for one device space is applied to the other.

        A trichrome merge and a normal capture are NOT the same device space (each
        trichrome channel is the sensor response times its own light), so swapping
        the profiles produces badly wrong colour with no symptom beyond "the colour
        looks off". This warns rather than refuses: colour management is the user's
        call, and a hard block would be worse than a preview they can see is wrong.
        See spec/trichrome-camera-profile.md section 6."""
        if profile is None:
            return
        prof_tri = bool(getattr(profile, "is_trichrome", False))
        img_tri = bool(getattr(self, "is_merged", False))
        if prof_tri == img_tri:
            return
        key = (id(profile), img_tri)
        if key in CCRImage._kind_warned:
            return
        CCRImage._kind_warned.add(key)
        logging.warning(
            "Camera profile device-space mismatch: a %s profile is being applied "
            "to a %s image. Build the profile from the same kind of capture.",
            "trichrome" if prof_tri else "normal",
            "trichrome (3-way merge)" if img_tri else "normal")

    def _apply_input_icc(self, arr: Optional[np.ndarray],
                         as_shot_wb=None) -> Optional[np.ndarray]:
        """Convert a freshly-decoded scan from the globally-assigned input ICC
        profile into the working LINEAR Adobe RGB space (the same space the
        no-ICC decode produces, so the density-based inversion sees consistent
        linear data), before any conversion or adjustment. No-op when no input
        profile is set. Applied inside read_image so preview, hi-res zoom, and
        export all inherit it identically (resolution-independent point op).

        The camera matrix consumes WHITE-BALANCED data, so the frame's as-shot
        neutral is threaded through (mirrors _apply_input_dcp)."""
        if arr is None:
            return arr
        profile = color_management.get_active_input_profile()
        if profile is None or color_management.input_profile_disabled():
            return arr
        self._warn_kind_mismatch(profile)
        try:
            return profile.apply(arr, as_shot_wb=as_shot_wb)
        except Exception as e:
            logging.warning(f"Input ICC profile could not be applied: {e}")
            return arr

    @staticmethod
    def _apply_field_correction(arr: Optional[np.ndarray], *,
                                encoded: bool = False,
                                mono: bool = False) -> Optional[np.ndarray]:
        """Multiply a freshly-decoded frame by the active field-correction gain
        map (lens vignetting + sensor colour shading + light-source unevenness),
        before slicing, the camera profile, and the negative conversion — the one
        place every consumer (preview, zoom, slice, merge, export) inherits it
        from. No-op when no field profile is active.

        Called at the FULL-FRAME point of each decode branch (before
        _apply_source_ops) so the map always lines up with the whole sensor frame
        whatever the slice/crop state. See spec/flat-field-correction.md §6.1."""
        if arr is None:
            return arr
        from core import flat_field
        profile = flat_field.get_active_profile()
        if profile is None:
            return arr
        try:
            return flat_field.apply_field(arr, profile, encoded=encoded, mono=mono)
        except Exception as e:
            logging.warning(f"Field correction could not be applied: {e}")
            return arr

    def _input_icc_will_apply(self) -> bool:
        """Whether read_image will burn an external camera profile into this scan
        — an input ICC **or** a DCP is active (mirrors the apply guard). Drives the
        negative RAW decode: a profiled decode is camera-native raw with absolute
        sensor values (no_auto_scale=True + manual white-level scaling); the
        unprofiled default decode is Adobe RGB, rawpy-auto-scaled. False when the
        profile is temporarily disabled."""
        return color_management.camera_profile_active()

    def _apply_input_dcp(self, arr, as_shot_wb) -> Optional[np.ndarray]:
        """Burn the globally-active DCP into a freshly-decoded camera-native scan
        (-> linear Adobe RGB), threading the as-shot WB. No-op when no DCP is set.
        Mirrors _apply_input_icc; failures log and pass the input through."""
        if arr is None:
            return arr
        profile = color_management.get_active_dcp_profile()
        if profile is None or color_management.input_profile_disabled():
            return arr
        self._warn_kind_mismatch(profile)
        try:
            from core import dcp_profile
            return dcp_profile.apply_dcp(profile, arr, as_shot_wb=as_shot_wb)
        except Exception as e:
            logging.warning(f"DCP profile could not be applied: {e}")
            return arr

    def _stamp_profile_signature(self):
        """Record which camera profile this image's working decode was graded
        under, so the thumbnail can flag a mismatch when the active profile (or
        the disable toggle / Positive mode) later changes."""
        try:
            from core.ccr_backend import ccr_backend
            self.profile_signature = ccr_backend.active_profile_signature()
        except Exception:
            self.profile_signature = None

    @staticmethod
    def _positive_mode_active() -> bool:
        """Whether the app is in global Positive mode (RAWs decode as normal
        sRGB positives, no negative conversion). Read lazily from the backend
        singleton so decode/display stay in sync with a live toggle without a
        per-image copy. See spec/positive-mode.md."""
        try:
            from core.ccr_backend import ccr_backend
            return bool(ccr_backend.positive_mode)
        except Exception:
            return False

    @staticmethod
    def _raw_color_postprocess_kwargs(positive: bool, preview: bool,
                                      no_icc_default: bool = False) -> dict:
        """rawpy.postprocess kwargs for a (non-monochrome) RAW decode.

        Negative (positive=False) is the scan the negative pipeline inverts:
        linear gamma, no explicit white balance. no_icc_default picks the output
        space and scaling. When False (an input ICC will correct the decode, or a
        caller wants bare device RGB): raw camera primaries with no_auto_scale=True
        — absolute sensor values (read_image's manual *65535/white_level then
        brings them to full range, so the ICC + inversion see consistent values).
        When True (unprofiled scan): Adobe RGB (camera-independent working space)
        with no_auto_scale=False — rawpy auto-scales the decode to full range
        (read_image then skips its manual white-level scaling for this path).
        Positive (positive=True) decodes a normal photo: sRGB color space + gamma,
        camera white balance, AHD demosaic, rawpy auto-brightness (no_icc_default
        is ignored on this path). Kept pure so the choice is unit-testable."""
        if positive:
            return dict(
                output_bps=16,
                # Auto-brightness OFF: rawpy's auto-bright scales until ~1% of
                # the brightest pixels saturate, CLIPPING highlights to white.
                # rawpy's auto-scale (no_auto_scale left at its default, off) still
                # maps the sensor white level to full range — a proper, non-clipping
                # exposure that preserves highlight headroom (raise Gain to taste).
                no_auto_bright=True,
                gamma=(2.222, 4.5),                               # sRGB-ish TRC
                user_flip=0,
                demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
                half_size=preview,
                use_camera_wb=True,                               # photographer's WB
                use_auto_wb=False,
                output_color=rawpy.ColorSpace.sRGB,
                four_color_rgb=False,
            )
        return dict(
            output_bps=16,
            no_auto_bright=True,      # Consistent absolute sensor values across all frames
            gamma=(1, 1),            # Linear gamma (no gamma correction)
            user_flip=0,              # No rotation
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # Simple linear demosaic
            half_size=preview,        # Process at half resolution - much faster!
            use_camera_wb=False,      # No camera white balance
            use_auto_wb=False,        # No auto white balance
            # No-ICC default decode: Adobe RGB (camera-independent working space)
            # + rawpy auto-scale to full range (read_image skips the manual
            # white-level scaling for it). ICC / bare-device decode: raw camera
            # primaries with scaling OFF, so absolute sensor values stay
            # consistent for the ICC + inversion.
            output_color=(rawpy.ColorSpace.Adobe if no_icc_default
                          else rawpy.ColorSpace.raw),
            no_auto_scale=(not no_icc_default),
            adjust_maximum_thr=0.0,   # Don't auto-lower maximum from frame data
            four_color_rgb=False,     # Standard 3-color processing
        )

    def _read_merged(self, preview: bool = True,
                     max_long_side: Optional[int] = None,
                     apply_input_icc: bool = True) -> Optional[np.ndarray]:
        """Decode a 3-way RGB-merged image from its source RAWs. Mirrors the tail
        of read_image's RAW branch (slice ops, full-size capture, optional
        downsize) so export / zoom / slice all compose. The merge's native
        resolution follows the captured merge_demosaic mode: full-sensor for a
        demosaiced Bayer merge and for monochrome (no CFA), half-sensor for the
        single-photosite Bayer read (the 2x2 bin IS its full resolution).
        `preview` speeds up monochrome/demosaic decodes (half size);
        full_decode_size always reports the canonical full resolution, and
        max_long_side does the rest."""
        from core import ccr_merge
        rgb, full_decode_size = ccr_merge.merge_raw_channels(
            self.merge_sources, preview=preview,
            demosaic=getattr(self, "merge_demosaic", True))
        # Field correction on the full merged frame (camera-native linear, like
        # the RAW branch). The linear-TIFF bake writes merge_raw_channels output
        # directly, NOT through read_image, so a baked replacement stays
        # uncorrected on disk and is corrected on reload — never twice.
        if apply_input_icc:
            rgb = self._apply_field_correction(rgb)
        # Sliced merged children read only their region of the source.
        rgb = self._apply_source_ops(rgb)
        self.original_full_size = self._ops_full_size(full_decode_size)
        if max_long_side:
            rgb = self.resize_image_to_max_pixel(rgb, max_long_side)
        # Burn in the camera profile, in the same pipeline position as the RAW
        # branch (field correction -> slice ops -> downsize -> profile), so a
        # trichrome capture is colour-managed like any other camera-native scan.
        #
        # as_shot_wb is None on purpose: a merge has THREE source RAWs with three
        # different camera_whitebalance values, and none of them describes the
        # merged channel balance — that is a property of the three light sources
        # and their exposures. A FreeCCR-generated profile carries its own
        # calibration neutral and ignores the as-shot value anyway
        # (spec/camera-profile-calibration-wb.md), which is exactly right here.
        # See spec/trichrome-camera-profile.md §5.
        if not apply_input_icc:
            return rgb
        if color_management.get_active_dcp_profile() is not None:
            return self._apply_input_dcp(rgb, None)
        return self._apply_input_icc(rgb, None)

    def read_image(self, file_path: str, preview = True, max_long_side: Optional[int] = None,
                   positive_override: Optional[bool] = None, apply_input_icc: bool = True) -> Optional[np.ndarray]:
        """
        Read an image file. preview=True decodes RAW at half size.
        max_long_side, when given, downsizes to that size INSIDE the reader —
        for RAW this happens before white-level scaling (both steps are
        linear, so the order only changes values by <=3 LSB, and scaling at
        preview size instead of decode size saves ~100 ms per image).

        positive_override / apply_input_icc let a caller pin the decode space
        without mutating global state (used by the IT8 camera-profiling decode,
        see spec/it8-camera-profile.md §6.1): positive_override=False forces the
        raw-linear negative decode regardless of the live Positive-mode toggle,
        and apply_input_icc=False is the BARE DEVICE DECODE — it forces the
        camera-native decode space and skips BOTH the active camera profile
        (ICC/DCP) and the active field correction, so a profiling decode measures
        untouched device data and profiles can never compound
        (spec/flat-field-correction.md §10.1). Both default to current behaviour,
        so existing callers are unaffected.
        """
        # 3-way RGB merge: this image has no single backing file — it is
        # re-synthesized from its three source RAWs. Dispatch BEFORE the
        # extension branch and BEFORE positive_mode is read (a merged frame is
        # always a camera-native negative). All re-read call sites (export,
        # zoom, slice, duplicate) flow through here, so they re-merge for free.
        if getattr(self, "is_merged", False) and self.merge_sources:
            return self._read_merged(preview=preview, max_long_side=max_long_side,
                                     apply_input_icc=apply_input_icc)

        # Ensure file path is properly encoded for Unicode support
        file_path = os.path.normpath(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        # Treat FFF files as TIFF files
        if ext == ".fff":
            ext = ".tiff"

        # Read the global Positive-mode flag once per call so every branch of
        # this decode (RAW vs non-RAW, white-level, input ICC) uses one
        # consistent value (spec/positive-mode.md §3 "read once"). A caller may
        # override it (e.g. force the raw-linear decode for IT8 profiling).
        positive_mode = (self._positive_mode_active() if positive_override is None
                         else bool(positive_override))

        if ext in [".cr3", ".cr2", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf", ".srw", ".pef", ".3fr"]:
            try:
                print(f"Starting RAW processing for: {os.path.basename(file_path)}")
                start_time = time.time()
                
                with rawpy.imread(file_path) as raw:
                    # Capture sensor ceiling before postprocess (e.g. 16383 for 14-bit)
                    white_level = raw.white_level
                    # As-shot white-balance multipliers (for the DCP apply path,
                    # which renders WB at apply time). RGB channels only.
                    try:
                        as_shot_wb = np.asarray(raw.camera_whitebalance[:3], dtype=float)
                    except Exception:
                        as_shot_wb = None

                    # Full processed output size, valid even when half_size=True.
                    # Kept LOCAL until after the (slow) postprocess: with a
                    # source_ops chain the final value differs, and read_image
                    # runs on worker threads while the GUI reads
                    # original_full_size — it must be assigned exactly once.
                    full_decode_size = (raw.sizes.height, raw.sizes.width)

                    # Check if this is a monochrome sensor
                    is_monochrome = False
                    try:
                        # Primary check: number of colors
                        if hasattr(raw, 'num_colors') and raw.num_colors == 1:
                            is_monochrome = True
                        # Secondary check: raw pattern (with error handling)
                        elif hasattr(raw, 'raw_pattern') and hasattr(raw, 'color_desc'):
                            try:
                                if raw.color_desc == b'RGBG' and raw.raw_pattern.max() == 0:
                                    is_monochrome = True
                            except (AttributeError, ValueError):
                                pass
                        # Tertiary check: color description indicates monochrome
                        elif hasattr(raw, 'color_desc') and raw.color_desc in [b'G', b'GRAY', b'GREY']:
                            is_monochrome = True
                    except Exception as e:
                        logging.warning(f"Error detecting monochrome sensor: {e}")
                        is_monochrome = False

                    # Global Positive mode decodes color RAWs as normal sRGB
                    # photos (monochrome sensors are left on their own path).
                    positive_decode = positive_mode and not is_monochrome

                    # Set in the colour branch below; pre-seeded so the
                    # white-level-scaling guard is valid on the monochrome path.
                    no_icc_default = False

                    if is_monochrome:
                        print(f"Detected monochrome sensor for: {os.path.basename(file_path)}")
                        # Fixed absolute sensor values, like the colour negative
                        # decode (and ccr_merge's mono path): linear gamma, no
                        # auto-bright, no_auto_scale — the single manual
                        # white-level scale below owns the range mapping.
                        # Without no_auto_scale libraw already stretches to full
                        # 16-bit and the scale below multiplies AGAIN, clipping
                        # everything above white_level; auto-bright also varies
                        # per frame, breaking the "B/W points are absolute
                        # anchors, constant across the roll" contract.
                        rgb = raw.postprocess(
                            output_bps=16,
                            no_auto_bright=True,
                            no_auto_scale=True,
                            gamma=(1, 1),
                            user_flip=0,
                            half_size=preview,
                            use_camera_wb=False,  # Disable WB for monochrome
                            four_color_rgb=False,
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR                        )
                        # Convert single channel to RGB by duplicating the channel
                        if len(rgb.shape) == 2:
                            rgb = np.stack([rgb, rgb, rgb], axis=2)
                        elif rgb.shape[2] == 1:
                            rgb = np.repeat(rgb, 3, axis=2)
                    else:
                        # Device space for the negative decode. Whenever an input
                        # ICC is in play —
                        #   • it is active and will be burned in now
                        #     (_input_icc_will_apply), or
                        #   • the caller wants the bare device RGB to FIT one on
                        #     (IT8 profiling, apply_input_icc=False)
                        # — decode in the CANONICAL camera-profiling space: raw
                        # camera-native primaries (output_color=raw, no camera
                        # matrix / no working-space conversion), linear gamma,
                        # no_auto_bright, no white balance, no_auto_scale + manual
                        # white-level scaling for fixed absolute sensor values.
                        # This is the standard input for matrix/cLUT ICC and DCP
                        # (`dcraw -o 0 -g 1 1 -W -r 1 1 1 1`, see
                        # spec/it8-camera-profile.md §12.5), and makes fit-space ==
                        # apply-space EXACTLY.
                        #
                        # No-profile decode space is the picker's choice: "Camera
                        # Matrix" -> Adobe RGB + rawpy auto-scale (the camera's
                        # built-in matrix); "None" -> bare camera-native RAW (same
                        # space a profile is applied on, just no matrix). A profile
                        # always decodes camera-native. The IT8 profiling decode
                        # (apply_input_icc=False) is forced camera-native too.
                        icc_device_space = (not apply_input_icc
                                            or not color_management.camera_matrix_mode())
                        no_icc_default = not icc_device_space
                        rgb = raw.postprocess(
                            **self._raw_color_postprocess_kwargs(
                                positive_decode, preview, no_icc_default))

                    # Field correction (flat-field) on the FULL frame, before the
                    # slice chain crops it — a monochrome decode gets the neutral
                    # (channel-averaged) gain so it stays colourless, and a
                    # Positive decode is sRGB-encoded, not linear.
                    if apply_input_icc:
                        rgb = self._apply_field_correction(
                            rgb, encoded=positive_decode, mono=is_monochrome)

                    # Sliced images read only their region of the source.
                    # Single atomic assignment of the final size (see above).
                    rgb = self._apply_source_ops(rgb)
                    self.original_full_size = self._ops_full_size(full_decode_size)

                    # Downsize before white-level scaling when a target size is
                    # known (see docstring — measured ~100 ms saved per image).
                    if max_long_side:
                        rgb = self.resize_image_to_max_pixel(rgb, max_long_side)

                    # Scale native bit depth to full 16-bit range so images display at
                    # correct brightness (e.g. 14-bit data sits in [0,16383] without this).
                    # Skipped in positive mode AND on the no-ICC default decode:
                    # both are already rawpy-auto-scaled to full range, so re-scaling
                    # here would blow out the highlights.
                    if (not positive_decode and not no_icc_default
                            and white_level > 0 and white_level < 65535):
                        print(f"Scaling RAW from {white_level}-ceiling to 16-bit (factor {65535.0/white_level:.4f})")
                        rgb = np.clip(
                            rgb.astype(np.float32) * (65535.0 / white_level),
                            0, 65535
                        ).astype(np.uint16)
                
                elapsed_time = time.time() - start_time
                print(f"RAW processing completed in {elapsed_time:.3f} seconds")
                # Burn in the global camera profile (ICC or DCP, if any) on the
                # decoded camera-native scan, before negative conversion. Skipped
                # in positive mode (the decode is already a ready sRGB positive)
                # and when the caller opted out (apply_input_icc=False, e.g. IT8
                # profiling wants bare device RGB). A monochrome sensor has no
                # colour to profile and is NOT decoded camera-native, so skip it
                # too. A DCP (mutually exclusive with the ICC) renders the as-shot
                # WB at apply time.
                if positive_decode or not apply_input_icc or is_monochrome:
                    return rgb
                if color_management.get_active_dcp_profile() is not None:
                    return self._apply_input_dcp(rgb, as_shot_wb)
                return self._apply_input_icc(rgb, as_shot_wb)
            except Exception as e:
                logging.exception(f"Failed to read RAW image: {file_path}")
                # Keep the LibRaw error itself — "unsupported compression" and
                # "damaged file" are indistinguishable once it becomes None.
                self.last_read_error = e
                return None
        else:
            # Handle Unicode file paths and OpenCV TIFF issues properly
            img = None
            is_tiff = file_path.lower().endswith(('.tif', '.tiff'))

            # Scanner TIFFs (Pakon, Nikon Coolscan) often use PLANAR storage
            # (PlanarConfiguration=2, RRR..GGG..BBB). OpenCV misreads those
            # into scrambled channels WITHOUT failing, so sniff the tag first
            # and route them straight to tifffile (issues #86/#87). The sniff
            # reads only the TIFF header, not the pixel data.
            if is_tiff and TIFFFILE_AVAILABLE:
                try:
                    with tifffile.TiffFile(file_path) as tf:
                        planar = getattr(tf.pages[0], "planarconfig", 1)
                        planar = int(getattr(planar, "value", planar) or 1)
                    if planar == 2:
                        img = _read_tiff_bgr(file_path)
                        print(f"Planar TIFF read via tifffile: "
                              f"{os.path.basename(file_path)}")
                except Exception as e:
                    logging.warning(
                        f"TIFF planar sniff failed for {file_path}: {e}")
                    img = None

            # Try multiple reading methods for better compatibility
            if img is None:
                try:
                    # Method 1: Try OpenCV imread first (handles most formats)
                    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

                except Exception as e:
                    logging.warning(f"OpenCV imread failed for {file_path}: {e}")
                    img = None

            # Method 2: If OpenCV fails or returns None, try alternative methods
            if img is None:
                try:
                    # For TIFF files, try using tifffile library which is more robust
                    if is_tiff and TIFFFILE_AVAILABLE:
                        img = _read_tiff_bgr(file_path)
                        print(f"Successfully read TIFF using tifffile: {os.path.basename(file_path)}")
                    else:
                        # For other formats, try binary reading with cv2.imdecode
                        with open(file_path, 'rb') as f:
                            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
                        
                except Exception as e:
                    logging.warning(f"Alternative reading method failed for {file_path}: {e}")
                    img = None
            
            # Method 3: Final fallback - try PIL/Pillow for maximum compatibility
            if img is None and PIL_AVAILABLE:
                try:
                    pil_img = PILImage.open(file_path)
                    # Convert PIL image to numpy array
                    img_array = np.array(pil_img)
                    
                    # Handle different PIL modes
                    if pil_img.mode == 'RGB':
                        # PIL uses RGB, OpenCV uses BGR
                        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    elif pil_img.mode == 'RGBA':
                        img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGRA)
                    elif pil_img.mode == 'L':
                        img = img_array
                    elif pil_img.mode == 'P':
                        # Convert palette mode to RGB first
                        pil_img = pil_img.convert('RGB')
                        img_array = np.array(pil_img)
                        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    else:
                        img = img_array
                    
                    print(f"Successfully read using PIL: {os.path.basename(file_path)}")
                    
                except Exception as e:
                    logging.error(f"PIL reading also failed for {file_path}: {e}")
                    img = None
            
            if img is None:
                logging.error(f"All reading methods failed for: {file_path}")
                self.last_read_error = RuntimeError(
                    "no decoder (OpenCV, tifffile, PIL) could read this file")
                return None
            # Normalize the sample format FIRST (cvtColor can't take
            # float64/int16 input, and float 0..1 data used to slip through
            # unconverted and render black).
            img = _to_uint16_full_range(img)
            # Convert grayscale to RGB
            is_gray = len(img.shape) == 2
            if is_gray:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Field correction on the full frame, before the slice chain. The
            # gain is applied in the file's own space (encoded=False): a
            # multiplicative falloff stays multiplicative through a power-law
            # encoding, so a profile measured and applied in the same space is
            # self-consistent. See spec/flat-field-correction.md §10.4.
            if apply_input_icc:
                img = self._apply_field_correction(img, mono=is_gray)
            # Sliced images read only their region of the source
            img = self._apply_source_ops(img)
            # This branch always reads at full resolution regardless of `preview`
            self.original_full_size = (img.shape[0], img.shape[1])
            if max_long_side:
                img = self.resize_image_to_max_pixel(img, max_long_side)
            # Burn in the global camera profile (ICC or DCP) before conversion.
            # Skipped in positive mode and when the caller opted out. A non-RAW
            # file has no as-shot WB, so a DCP applies unbalanced (warned in the UI).
            if positive_mode or not apply_input_icc:
                return img
            if color_management.get_active_dcp_profile() is not None:
                return self._apply_input_dcp(img, None)
            return self._apply_input_icc(img, None)
        
    def update_thumbnail_and_preview(self, thumbnail_size: int = 156, preview_size: int = 1080) -> None:
        """
        Populates/updates the thumbnail and resized_preview attributes using the 16-bit resized_raw image.
        Both outputs are 8-bit RGB np.ndarray.
        Applies adjustments before resizing.
        """
        if self.resized_raw is None:
            return

        def to_8bit(img16: np.ndarray) -> np.ndarray:
            # Saturating SIMD 16->8 bit conversion (~4x faster than the
            # previous float divide + astype)
            return cv2.convertScaleAbs(img16, alpha=255.0 / 65535.0)

        # Apply adjustments first
        adjusted_img = self.apply_adjustments(self.resized_raw)

        # Display-only auto-brightness: the un-converted negative scan is very dark
        # (linear-gamma data sitting low in the 16-bit range). Stretch it so it's
        # legible in the preview/thumbnail. This is purely cosmetic — it operates on
        # a copy and never touches resized_raw, so conversion, export, and B/W-point
        # sampling (which read resized_raw or the original file) are unaffected.
        # Once converted, the positive is properly exposed and the slider adjustments
        # own the look, so the auto-brightness is skipped. Positive mode decodes a
        # correctly-exposed positive too, so it skips the negative auto-brightness
        # as well (the adjustments own the look).
        display_img = (adjusted_img
                       if (self.converted or self._positive_mode_active())
                       else self._auto_brightness_for_preview(adjusted_img))

        # Sprocket-hole / clear-film white mask (reversal look) — the LAST look
        # step, so no adjustment can tint the whitened regions. B/W-point
        # conversions only (sprocket_alpha is set only there); gated on the live
        # global toggle. Applied to the DISPLAY pixels (thumbnail + preview
        # pixmap) only — the histogram reads the PRE-mask image below, so a
        # whitened border doesn't add a misleading spike at white. display_img
        # shares resized_raw's dims, as does the alpha. See spec §4.2.
        from core.ccr_backend import ccr_backend
        _mask_on = (self.converted and not self._positive_mode_active()
                    and self.sprocket_alpha is not None
                    and ccr_backend.sprocket_mask_white)
        if _mask_on:
            from core.ccr_processor import apply_sprocket_mask
            display_masked = apply_sprocket_mask(display_img, self.sprocket_alpha)
        else:
            display_masked = display_img

        # Create thumbnail + preview pixels (8-bit RGB numpy — safe on any thread)
        thumb_img_8 = to_8bit(self.resize_image_to_max_pixel(display_masked, thumbnail_size))
        preview_img = self.resize_image_to_max_pixel(display_masked, preview_size)
        preview_img_8bit = to_8bit(preview_img)

        # QPixmap may only be created on the GUI thread, but the batch paths
        # (initial load, auto-frame-all, B/W convert-all) run this method on
        # pool/QThread workers. Off the GUI thread, stash the pixels and let
        # the first GUI-thread read of .thumbnail/.resized_preview build the
        # pixmaps (see the property getters).
        if self._on_gui_thread():
            self.thumbnail = QPixmap.fromImage(
                self.generate_qimage_from_np_array_8(thumb_img_8))
            self.resized_preview = QPixmap.fromImage(
                self.generate_qimage_from_np_array_8(preview_img_8bit))
        else:
            self._thumb_np8 = thumb_img_8
            self._preview_np8 = preview_img_8bit
        # Compute the per-channel histogram over the 8-bit preview (RGB). When a
        # crop is set, it's computed over only the kept (cropped) region so it
        # matches what the canvas shows. Same normalized-rect + angle contract as
        # the display/export crop path; apply_crop_to_image returns the input
        # unchanged when crop_rect is None.
        #
        # We store only the raw counts here (shape (3, 256), R/G/B order). All
        # presentation — the percentile-clipped vertical scale, smoothing,
        # filled curves and clip markers — lives in HistogramWidget, which paints
        # at the widget's real resolution. See widgets/histogram_widget.py.
        # Pre-mask preview for the histogram (the tones the user edits). Identical
        # to preview_img_8bit when the sprocket mask is off, so only pay the extra
        # resize when it is on. See spec/sprocket-hole-mask.md §7.
        hist_base_8bit = (to_8bit(self.resize_image_to_max_pixel(display_img, preview_size))
                          if _mask_on else preview_img_8bit)
        hist_source = apply_crop_to_image(
            hist_base_8bit, self.crop_rect, getattr(self, "crop_angle", 0.0) or 0.0)
        counts = np.empty((3, 256), dtype=np.float32)
        for i in range(3):   # 0=R, 1=G, 2=B
            counts[i] = cv2.calcHist([hist_source], [i], None, [256], [0, 256]).flatten()
        self.histogram_data = counts

    def generate_qimage_from_np_array_8(self, thumb_img_8):
        h, w, ch = thumb_img_8.shape
        bytes_per_line = ch * w
        qimage = QImage(
            thumb_img_8.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        return qimage

    @staticmethod
    def _on_gui_thread() -> bool:
        """True on the Qt GUI thread — or with no app at all, where eager
        QPixmap creation matches the historical (pre-deferral) behaviour."""
        app = QCoreApplication.instance()
        return app is None or QThread.currentThread() is app.thread()

    # QPixmap is GUI-thread-only. When update_thumbnail_and_preview runs on a
    # worker (batch load / auto-frame-all / B/W convert-all pools), it stashes
    # the rendered 8-bit pixels in _thumb_np8/_preview_np8 instead of building
    # pixmaps; the first GUI-thread read below materializes them. A read from
    # a worker thread returns the previous pixmap untouched.
    @property
    def thumbnail(self):
        if self._thumb_np8 is not None and self._on_gui_thread():
            self._thumbnail_pix = QPixmap.fromImage(
                self.generate_qimage_from_np_array_8(self._thumb_np8))
            self._thumb_np8 = None
        return self._thumbnail_pix

    @thumbnail.setter
    def thumbnail(self, value):
        self._thumbnail_pix = value
        self._thumb_np8 = None

    @property
    def resized_preview(self):
        if self._preview_np8 is not None and self._on_gui_thread():
            self._preview_pix = QPixmap.fromImage(
                self.generate_qimage_from_np_array_8(self._preview_np8))
            self._preview_np8 = None
        return self._preview_pix

    @resized_preview.setter
    def resized_preview(self, value):
        self._preview_pix = value
        self._preview_np8 = None

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Map an RGB image to a single luminance channel (Rec.601 weights:
        0.299R + 0.587G + 0.114B) replicated across all three channels, so it
        renders/exports as black & white while keeping the RGB shape and dtype
        every downstream stage expects. Returns a new array."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def _apply_dust_removal(self, image: np.ndarray,
                            ws_windowed: bool = False) -> np.ndarray:
        """Inpaint stored dust spots out of the working image (no-op when there
        are none). Spots are normalized, so this scales to whatever resolution
        is being processed — preview, hi-res zoom, or full-res export.

        WYSIWYG plan sharing: a preview-scale heal caches its plan (segments +
        chosen source patches) keyed by the spots' content; hi-res and export
        renders replay THAT plan, so they sample exactly the patches the user
        approved on screen — a fresh plan from the export's own re-decoded
        pixels can flip near-tie source choices. Feather is excluded from the
        key (it shapes the blend, not the plan).

        Sticky sources: the cached plan — even one keyed to a PREVIOUS spot
        set — seeds every preview-scale re-heal as `prior_plan`, so once a
        segment's source patch is chosen, later strokes/edits reuse it
        verbatim instead of re-searching (the plan is serialized in the
        catalog, so this holds across sessions too). See
        spec/dust-removal.md."""
        spots = getattr(self, "dust_spots", None)
        if not spots:
            return image
        t0 = time.time()
        feather = getattr(self, "dust_feather", DUST_FEATHER_DEFAULT)
        # Sources must come from INSIDE the confirmed crop — outside is the
        # holder/rebate/junk the user cut away, not scene content.
        rect = getattr(self, "crop_rect", None)
        crop = ((tuple(rect), float(getattr(self, "crop_angle", 0.0) or 0.0))
                if rect else None)
        key = repr(spots)
        h, w = image.shape[:2]
        if max(h, w) <= _DUST_PLAN_LONG:
            plan_out = []
            cached = getattr(self, "_dust_plan_cache", None)
            prior = cached[1] if cached is not None else None
            # A prior record may only rebind to a spot that STILL EXISTS.
            # The cache deliberately outlives spot edits (sticky sources),
            # but a record whose spot was deleted/replaced must never seed
            # a NEW spot painted or re-detected at the same place — that
            # resurrected old sources verbatim and the sampling rule never
            # ran for the new spot (stale catalog plans made the touch rule
            # look permanently broken).
            snap = getattr(self, "_dust_plan_spots", None)
            if prior and snap is not None:
                gone = [s for s in snap if s not in spots]
                if gone:
                    from core.ccr_processor import rasterize_dust_mask
                    gm = rasterize_dust_mask(gone, h, w) > 0
                    prior = [r for r in prior
                             if not gm[min(h - 1, max(0, int(round(r[0] * h)))),
                                       min(w - 1, max(0, int(round(r[1] * w))))]]
            out = apply_dust_removal(image, spots, feather=feather,
                                     collect_plan=plan_out, prior_plan=prior,
                                     crop=crop, ws_windowed=ws_windowed)
            self._dust_plan_cache = (key, plan_out)
            # Shallow copy ON PURPOSE: a right-drag stroke MOVE mutates the
            # spot dict in place, so the snapshot entry stays identical and
            # the moved stroke keeps its pinned source.
            self._dust_plan_spots = list(spots)
            print(f"Dust heal total ({len(spots)} spots, preview scale, "
                  f"plan cached): {time.time() - t0:.3f}s")
            return out
        cached = getattr(self, "_dust_plan_cache", None)
        plan = cached[1] if (cached is not None and cached[0] == key) else None
        out = apply_dust_removal(image, spots, feather=feather, plan=plan,
                                 crop=crop, ws_windowed=ws_windowed)
        print(f"Dust heal total ({len(spots)} spots, "
              f"preview plan {'reused' if plan is not None else 'MISSING'}): "
              f"{time.time() - t0:.3f}s")
        return out

    def apply_adjustments(self, image: np.ndarray, settings=None, contrast_base=None,
                          temperature_base=None, brightness_base=None,
                          color_profile=None, areas_override=None,
                          exposure_base=None, ws_windowed=None) -> np.ndarray:
        """Apply the slider adjustments. The optional overrides let the zoom
        hi-res worker render from a snapshot taken at request time instead of
        live state the GUI thread may be mutating concurrently — and let the
        export pipeline describe the buffer it just produced (ws_windowed)
        without mutating this image's live state."""
        # Dust removal runs FIRST so the inpainted positive flows through the
        # rest of the adjustment stage (and so a dust-only image is still
        # cleaned even when the early-return guard below would otherwise skip).
        # It runs on the (possibly WINDOWED) base, so the heal must know — the
        # sampling rule's hue/sat/value deltas are computed on display values.
        ws = self._ws_windowed if ws_windowed is None else ws_windowed
        image = self._apply_dust_removal(image, ws_windowed=ws)
        s = self.adjustment_settings if settings is None else settings
        cb = self.contrast_base if contrast_base is None else contrast_base
        tb = self.temperature_base if temperature_base is None else temperature_base
        bb = self.brightness_base if brightness_base is None else brightness_base
        eb = self.exposure_base if exposure_base is None else exposure_base
        profile = self.color_profile if color_profile is None else color_profile
        areas = (getattr(self, "area_layers", []) if areas_override is None
                 else areas_override)
        has_areas = bool(areas) and any(a.get("enabled") for a in areas)
        # Auto Gain (spec/auto-gain.md): a hidden, live offset on Channel Levels'
        # MASTER GAIN that places the top-2% in-bound highlight at 95% of the
        # working-space window. Master Gain is now the app's one gain control (the
        # general-adjustments "Gain" slider was the same math at a different scale
        # and was removed), so Auto Gain and the user's manual gain ride the same
        # stage. Film CONVERSIONS only — its reference is the sampled clear/dense
        # range, which positive mode has no concept of (positive previews stay
        # identity; adjustments own the look). When ON it SUPERSEDES the legacy
        # baked auto-exposure (eb) so they don't double-apply. Deferred import —
        # ccr_backend imports CCRImage at load.
        from core.ccr_backend import ccr_backend
        auto_on = getattr(ccr_backend, "auto_gain", True) and self.converted
        ag = compute_auto_gain_offset(image, ws) if auto_on else 0.0
        eb_eff = 0.0 if auto_on else eb        # suppress-overlap with the baked eb
        if (not s and cb == 0 and tb == 0 and bb == 0 and eb_eff == 0 and ag == 0
                and not has_areas):
            # No slider/base/area adjustments. A windowed working-space base still
            # has to be de-windowed + window-clamped to a normal full-range image
            # before display/export (the base itself is not directly renderable).
            if ws:
                image = _apply_working_space_recovery(image, 0.0)
            # Black & White still has to map the image to a single luminance channel.
            return self._to_grayscale(image) if profile == "bw" else image
        adjusted = adjust_image_opencl(image,
                     s.get('temperature', 0) + tb,
                     s.get('tint', 0),
                     # The Gain/Exposure stage has no slider any more. It still
                     # carries the legacy baked auto-exposure (eb, default-slope
                     # mode) and whatever an AREA layer sets programmatically;
                     # applied un-clamped before the window clamp so a lift lands
                     # in highlight headroom. Auto Gain (ag) is the live,
                     # toggleable generalization and now rides MASTER GAIN below;
                     # when it is on it replaces eb (eb_eff == 0) so the two can't
                     # double-apply.
                     s.get('exposure', 0) + eb_eff,
                     # Brightness slider is half-strength per click; the
                     # always-on base offset (bb) keeps full weight so the
                     # default look is unchanged.
                     0.5 * s.get('brightness', 0) + bb,
                     s.get('black_point', 0),
                     s.get('white_point', 0),
                     s.get('contrast', 0) + cb,
                     s.get('saturation', 0),
                     self.tint_balance_factor,
                     highlights=s.get('highlights', 0),
                     shadows=s.get('shadows', 0),
                     ch_input_gain=s.get('ch_input_gain', 0),
                     ch_master_shift=s.get('ch_master_shift', 0),
                     # Auto Gain rides Master Gain: ADDED to the user's value, so
                     # the slider still reads what the user set while the render
                     # carries the automatic normalization on top.
                     ch_master_gain=s.get('ch_master_gain', 0) + ag,
                     ch_r_shift=s.get('ch_r_shift', 0),
                     ch_r_gain=s.get('ch_r_gain', 0),
                     ch_r_blackpoint=s.get('ch_r_blackpoint', 0),
                     ch_g_shift=s.get('ch_g_shift', 0),
                     ch_g_gain=s.get('ch_g_gain', 0),
                     ch_g_blackpoint=s.get('ch_g_blackpoint', 0),
                     ch_b_shift=s.get('ch_b_shift', 0),
                     ch_b_gain=s.get('ch_b_gain', 0),
                     ch_b_blackpoint=s.get('ch_b_blackpoint', 0),
                     sub_saturation=s.get('sub_saturation', 0),
                     # Per-color-band sliders ride the same GPU pass (or the
                     # CPU fallback); None keeps the inactive case free.
                     band_settings=(s if any(s.get(k, 0)
                                             for k in BAND_ADJUSTMENT_KEYS)
                                    else None),
                     # Windowed working-space base → de-window + Gain/Exposure
                     # recovery happens inside the adjustment call.
                     ws_windowed=ws)
        # Gamma slider: a center-point tone curve driven through the SAME
        # monotone-cubic path as the Curves editor (a single 'rgb' point moving
        # diagonally from center). Applied before the user's manual curves so the
        # two compose predictably. No-op at 0. Per-channel by default; the global
        # gamma_luminance flag switches it to hue-preserving (luminance) mode.
        gamma = s.get('gamma', 0)
        if gamma:
            adjusted = apply_gamma_curve(adjusted, gamma,
                                         luminance=ccr_backend.gamma_luminance)
        # Tone curves run after the slider pass, in RGB, before any B&W
        # luminance collapse (Photoshop-like). No-op for identity curves.
        curves = s.get('curves')
        if curves:
            adjusted = apply_curves(adjusted, curves)
        # Area editing: composite each enabled local layer additively on top of
        # the globally-adjusted ("whole image") result. Runs before the B&W
        # collapse so per-area color adjustments apply in RGB, like curves.
        if has_areas:
            adjusted = apply_area_layers(adjusted, areas, self._adjust_for_area)
        if profile == "bw":
            adjusted = self._to_grayscale(adjusted)
        # Cineon film log → Rec.709 (γ 2.2): optional FINAL stage after every
        # other adjustment (Channel Levels checkbox), so preview, hi-res zoom
        # and export transform identically. Whole-image only — area layers
        # never carry the key. See spec/cineon-display-transform.md.
        if s.get("cineon_log"):
            adjusted = apply_cineon_to_rec709(adjusted)
        return adjusted

    def _adjust_for_area(self, base_u16: np.ndarray, settings: dict) -> np.ndarray:
        """One area's full per-pixel adjustment layer, computed against the
        globally-adjusted base. Reuses the exact slider + curve math, but with
        ZEROED base offsets (contrast_base/temperature_base/brightness_base):
        those are global-look offsets already baked into the base, so an area
        must not re-apply them. Never touches negative inversion (whole-image)."""
        s = settings or {}
        if not s:
            return base_u16
        adjusted = adjust_image_opencl(base_u16,
                     s.get('temperature', 0),
                     s.get('tint', 0),
                     s.get('exposure', 0),
                     0.5 * s.get('brightness', 0),
                     s.get('black_point', 0),
                     s.get('white_point', 0),
                     s.get('contrast', 0),
                     s.get('saturation', 0),
                     self.tint_balance_factor,
                     highlights=s.get('highlights', 0),
                     shadows=s.get('shadows', 0),
                     ch_input_gain=s.get('ch_input_gain', 0),
                     ch_master_shift=s.get('ch_master_shift', 0),
                     ch_master_gain=s.get('ch_master_gain', 0),
                     ch_r_shift=s.get('ch_r_shift', 0),
                     ch_r_gain=s.get('ch_r_gain', 0),
                     ch_r_blackpoint=s.get('ch_r_blackpoint', 0),
                     ch_g_shift=s.get('ch_g_shift', 0),
                     ch_g_gain=s.get('ch_g_gain', 0),
                     ch_g_blackpoint=s.get('ch_g_blackpoint', 0),
                     ch_b_shift=s.get('ch_b_shift', 0),
                     ch_b_gain=s.get('ch_b_gain', 0),
                     ch_b_blackpoint=s.get('ch_b_blackpoint', 0),
                     sub_saturation=s.get('sub_saturation', 0),
                     band_settings=(s if any(s.get(k, 0)
                                             for k in BAND_ADJUSTMENT_KEYS)
                                    else None))
        gamma = s.get('gamma', 0)
        if gamma:
            from core.ccr_backend import ccr_backend
            adjusted = apply_gamma_curve(adjusted, gamma,
                                         luminance=ccr_backend.gamma_luminance)
        curves = s.get('curves')
        if curves:
            adjusted = apply_curves(adjusted, curves)
        return adjusted

    def render_hires_base(self, max_long_side: Optional[int] = None,
                          conversion_inputs=None):
        """
        Re-decode this image and reproduce its conversion at the RAW
        half-size resolution (capped at max_long_side — important for
        non-RAW sources, which otherwise decode at FULL resolution),
        color-matched to the 1080 preview: the conversion is replayed from
        the snapshot captured at convert time (conversion_inputs), never
        from live editable state, so it always matches what the preview
        shows. Runs on a worker thread for the zoom detail view; never
        touches resized_raw or any preview state.

        Returns `(base, sprocket_alpha)`: `base` is the converted, PRE-
        adjustment uint16 array (or the raw scan for un-converted images, or
        None when no color-matched replay is possible); `sprocket_alpha` is
        the hi-res clear-film white-mask alpha for B/W-point conversions
        (None otherwise). See spec/sprocket-hole-mask.md §4.2.
        """
        ci = conversion_inputs if conversion_inputs is not None else self.conversion_inputs
        if self.converted and ci is None:
            return None, None  # converted through an unknown path — no replay possible
        t0 = time.time()
        img = self.read_image(self.file_path, preview=True, max_long_side=max_long_side)
        if img is None:
            return None, None
        print(f"Hi-res decode: {time.time() - t0:.2f}s, shape {img.shape}")
        # Decide replay from the (snapshotted) conversion_inputs, NOT the live
        # self.converted flag: a positive-mode toggle can clear self.converted on
        # the GUI thread mid-render, and this worker must reproduce exactly the
        # conversion its snapshot describes. ci is None for un-converted scans
        # (incl. positive mode) -> return the decoded image as-is.
        if not ci:
            return img, None

        from core.ccr_processor import (compute_reference_norm_params,
                                        apply_reference_normalization,
                                        apply_bwpoint_normalization,
                                        compute_sprocket_alpha)
        sprocket_alpha = None
        t0 = time.time()
        if ci.get("mode") == "ref":
            ref_small = self.resize_image_to_max_pixel(img, 1080)
            p_lo, p_hi, od_factors = compute_reference_norm_params(
                ref_small, ci["ref"], ci["fine_rot"])
            out = apply_reference_normalization(img, p_lo, p_hi, od_factors)
        elif ci.get("mode") == "ref_params":
            # Sliced children of a reference-converted parent carry the
            # parent's precomputed conversion constants directly (the
            # parent's frame coordinates would be meaningless here).
            out = apply_reference_normalization(img, ci["p_lo"], ci["p_hi"], ci["od"])
        elif ci.get("mode") == "bw":
            # The bwpoint pipeline keeps the preview un-rotated (display
            # semantics — the canvas item applies the fine rotation), so the
            # replay must NOT warp either; ci["fine_rot"] is an informational
            # snapshot only.
            black_point, white_point = ci["bw"]
            out = apply_bwpoint_normalization(img, black_point, white_point,
                                              density=ci.get("density", False),
                                              slopes_bgr=ci.get("slopes"))
            # Hi-res clear-film mask from the re-decoded raw (same threshold as
            # the preview; morphology/feather scale with this resolution).
            sprocket_alpha = compute_sprocket_alpha(img, black_point)
        else:
            return None, None
        print(f"Hi-res convert: {time.time() - t0:.2f}s")
        return out, sprocket_alpha

    # --- Undo support (Ctrl+Z) ---------------------------------------------
    UNDO_STACK_LIMIT = 50

    def capture_undo_state(self) -> dict:
        """Snapshot of every user-editable, non-destructive setting."""
        return {
            "adjustment_settings": dict(self.adjustment_settings),
            "color_profile": self.color_profile,
            "crop_rect": self.crop_rect,
            "crop_angle": self.crop_angle,
            "rotation_angle": self.rotation_angle,
            "fine_rotation_angle": self.fine_rotation_angle,
            "horizontal_mirrored": self.horizontal_mirrored,
            "vertical_mirrored": self.vertical_mirrored,
            # Deep copy: each area nests a settings dict (with a curves
            # sub-dict) and a geometry dict — a shallow copy would alias the
            # live structure and a later edit would corrupt the snapshot.
            "area_layers": copy.deepcopy(getattr(self, "area_layers", [])),
            # Deep copy: each spot nests a point list, so a shallow copy would
            # alias live state and a later edit would corrupt the snapshot.
            "dust_spots": copy.deepcopy(getattr(self, "dust_spots", [])),
        }

    def push_undo_state(self) -> None:
        """Push the current state onto the undo stack. Call BEFORE mutating
        settings. Consecutive identical snapshots are collapsed so every
        Ctrl+Z press changes something."""
        state = self.capture_undo_state()
        if self.undo_stack and self.undo_stack[-1] == state:
            return
        self.undo_stack.append(state)
        if len(self.undo_stack) > self.UNDO_STACK_LIMIT:
            del self.undo_stack[0]

    def pop_undo_state(self) -> bool:
        """Restore the most recent snapshot. Returns False when there is
        nothing to undo. Callers own refreshing previews/UI afterwards."""
        if not self.undo_stack:
            return False
        state = self.undo_stack.pop()
        self.adjustment_settings = state["adjustment_settings"]
        self.color_profile = state.get("color_profile", "color")
        self.crop_rect = state["crop_rect"]
        self.crop_angle = state.get("crop_angle", 0.0)
        self.rotation_angle = state["rotation_angle"]
        self.fine_rotation_angle = state["fine_rotation_angle"]
        self.horizontal_mirrored = state["horizontal_mirrored"]
        self.vertical_mirrored = state["vertical_mirrored"]
        self.area_layers = copy.deepcopy(state.get("area_layers", []))
        self.dust_spots = copy.deepcopy(state.get("dust_spots", []))
        # The previously-active area may have been added back/removed by this
        # undo; the panel re-resolves None -> global if the id is now stale.
        active_id = getattr(self, "active_area_id", None)
        if active_id is not None and not any(
                a.get("id") == active_id for a in self.area_layers):
            self.active_area_id = None
        return True

    # --- Area editing helpers ----------------------------------------------
    def get_area(self, area_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """The area dict with this id, or None."""
        if area_id is None:
            return None
        for a in self.area_layers:
            if a.get("id") == area_id:
                return a
        return None

    def active_settings(self) -> Dict[str, Any]:
        """The adjustment_settings dict the panel currently edits: the global
        (whole-image) dict when active_area_id is None, else the active area's
        settings. Falls back to the global dict if the active id is stale."""
        a = self.get_area(self.active_area_id)
        return a["settings"] if a is not None else self.adjustment_settings

    def _auto_brightness_for_preview(self, image: np.ndarray) -> np.ndarray:
        """
        Display-only auto-brightness for the un-converted negative preview.

        Applies a pure GAIN that places the 95th-percentile value at full
        scale. Gain-only on purpose: the previous version also subtracted a
        per-frame black offset, and subtracting a constant from RGB shifts
        hue — mostly-blank frames (lots of film base) rendered their base
        orange/red while exposed frames stayed pink, which read as the app
        corrupting scans on import (GitHub issue #86). A gain preserves
        channel ratios, so the film base looks the same on every frame.

        Returns a new array; the input (resized_raw) is never modified, so
        this has no effect on conversion, export, or any other processing.
        """
        if image is None:
            return image

        # Estimate the anchor on a 4x4-subsampled view (16x fewer pixels,
        # visually identical) — this runs on every preview refresh of an
        # un-converted image.
        sample = image[::4, ::4]
        hi = float(np.percentile(sample, 95.0))
        if hi <= 0:
            # Degenerate (black) image — nothing meaningful to brighten.
            return image

        img = image.astype(np.float32)
        np.multiply(img, 65535.0 / hi, out=img)
        np.clip(img, 0, 65535, out=img)
        return img.astype(np.uint16)

    def __repr__(self):
        return (
            f"CCRImage(file_path={self.file_path!r}, "
            f"thumbnail={'set' if self.thumbnail is not None else 'None'}, "
            f"resized_raw={'set' if self.resized_raw is not None else 'None'}, "
            f"resized_preview={'set' if self.resized_preview is not None else 'None'}, "
            f"reference_frame={self.reference_frame!r}, "
            f"adjustment_settings={self.adjustment_settings!r}, "
            f"tint_balance_factor={getattr(self, 'tint_balance_factor', 1.0):.6f}, "
            f"rotation_angle={self.rotation_angle}, "
            f"fine_rotation_angle={self.fine_rotation_angle}, "
            f"horizontal_mirrored={self.horizontal_mirrored}, "
            f"vertical_mirrored={self.vertical_mirrored})"
        )
    
    @staticmethod
    def get_camera_and_lens_for_lensfun(raw_path: str) -> Dict[str, Optional[Any]]:
        """
        Extract camera and lens info from a raw file and parse for lensfun.

        Args:
            raw_path (str): Path to the raw image file.

        Returns:
            dict: Dictionary with keys suitable for lensfunpy:
                - camera_make
                - camera_model
                - lens_make
                - lens_model
                - focal_length (float)
                - aperture (float)
                - distance (float, meters)
        """
        info: Dict[str, Optional[Any]] = {}
        try:
            # Normalize path to handle Unicode characters
            raw_path = os.path.normpath(raw_path)
            with open(raw_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
                info['camera_make'] = str(tags.get('Image Make', '')).strip()
                info['camera_model'] = str(tags.get('Image Model', '')).strip()
                info['lens_make'] = str(tags.get('EXIF LensMake', '')).strip()
                info['lens_model'] = str(tags.get('EXIF LensModel', '')).strip()
                if len(info['lens_make']) == 0:
                    lens_model = info['lens_model'].upper()
                    if "DG DN" in lens_model or "DC DN" in lens_model:
                        info['lens_make'] = "Sigma"
                # FocalLength and FNumber may be Ratio objects, convert to float
                focal = tags.get('EXIF FocalLength')
                if focal:
                    try:
                        val = focal.values[0]
                        if hasattr(val, 'num') and hasattr(val, 'den') and val.den != 0:
                            info['focal_length'] = float(val.num) / float(val.den)
                        else:
                            info['focal_length'] = float(val)
                    except Exception as e:
                        logging.warning(f"Error parsing focal length from EXIF: {e}")
                        try:
                            info['focal_length'] = float(str(focal))
                        except Exception as e2:
                            logging.warning(f"Error converting focal length to float: {e2}")
                            info['focal_length'] = None
                else:
                    info['focal_length'] = None
                fnum = tags.get('EXIF FNumber')
                if fnum:
                    try:
                        val = fnum.values[0]
                        if hasattr(val, 'num') and hasattr(val, 'den') and val.den != 0:
                            info['aperture'] = float(val.num) / float(val.den)
                        else:
                            info['aperture'] = float(val)
                    except Exception as e:
                        logging.warning(f"Error parsing aperture from EXIF: {e}")
                        try:
                            info['aperture'] = float(str(fnum))
                        except Exception as e2:
                            logging.warning(f"Error converting aperture to float: {e2}")
                            info['aperture'] = None
                else:
                    info['aperture'] = None
                # Fetch focus distance (in meters)
                dist = tags.get('EXIF SubjectDistance')
                if dist:
                    try:
                        val = dist.values[0]
                        if hasattr(val, 'num') and hasattr(val, 'den') and val.den != 0:
                            info['distance'] = float(val.num) / float(val.den)
                        else:
                            info['distance'] = float(val)
                    except Exception as e:
                        logging.warning(f"Error parsing subject distance from EXIF: {e}")
                        try:
                            info['distance'] = float(str(dist))
                        except Exception as e2:
                            logging.warning(f"Error converting subject distance to float: {e2}")
                            info['distance'] = None
                else:
                    info['distance'] = None
        except Exception as e:
            logging.error(f"Failed to extract EXIF info from {raw_path}: {e}")
            info = {
                'camera_make': None,
                'camera_model': None,
                'lens_make': None,
                'lens_model': None,
                'focal_length': None,
                'aperture': None,
                'distance': None
            }
        return info

    def correct_lens_distortion_and_vignette(self) -> Optional[np.ndarray]:
        """
        Correct lens distortion and vignetting on self.resized_raw (16-bit RGB) using lensfunpy,
        preserving 16-bit data by remapping with OpenCV.
        Returns a new np.ndarray with corrections applied, or raises an exception if correction is not possible.
        """
        return self.resized_raw
        # if self.resized_raw is None or not self.info:
        #     return None  # No image or no lens/camera info available
        # try:
        #     print("here 0")
        #     db = lensfunpy.Database()
        #     cam = db.find_cameras(
        #         self.info.get('camera_make', ''),
        #         self.info.get('camera_model', '')
        #     )[0]
        #     print(cam)
        #     lens = db.find_lenses(cam, self.info.get('lens_make', ''), self.info.get('lens_model', ''))[0]
        #     print(lens)
        # except Exception as e:
        #     logging.error(f"Failed to access lensfun database: {e}")
        #     return None

        # return self.resized_raw


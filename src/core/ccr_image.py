from typing import Any, Dict, Optional, List
import numpy as np
import os
import copy
import rawpy
import exifread
import cv2
import logging
import time
from PySide6.QtGui import QImage, QPixmap  # or from PySide6.QtGui import QImage, QPixmap if you use PySide
#import lensfunpy  # Make sure lensfunpy is installed
from core.ccr_processor import (adjust_image, adjust_image_opencl,
                                BAND_ADJUSTMENT_KEYS, apply_curves,
                                apply_area_layers, apply_crop_to_image)
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
        ):
        # Normalize file path to handle Unicode characters properly
        self.file_path = os.path.normpath(file_path)
        # Sliced images own a chain of slice operations applied to the source
        # file by read_image in every path (preview load, hi-res zoom,
        # full-res export, B/W sampling). Each op is
        #   (rotation_hundredths_deg, (x1, y1, x2, y2))
        # — rotate the current frame about its center (the fine rotation
        # baked at slice time), then crop to the fractional region of that
        # frame. Nested slices simply append ops. [] = whole file.
        self.source_ops: list = list(source_ops) if source_ops else []
        # Display/export name override (e.g. "scan_s2.ARW" for slice #2);
        # None = use the file's basename.
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
        # {"mode": "bw", "bw": ((B,G,R),(B,G,R)), "fine_rot": int}
        self.conversion_inputs: Optional[Dict[str, Any]] = None
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
        # Dust healing: a list of resolution-independent vector "strokes" (see
        # core/dust_removal.py and spec/dust-healing.md). Normalized coords in
        # the same un-rotated/un-cropped space as crop_rect; replayed in
        # apply_adjustments so healing shows at every resolution. Per-image
        # only (dust is physical to one frame) — excluded from copy/paste/sync.
        self.dust_heal_strokes: List[Dict[str, Any]] = []
        # Which layer the adjustment panel currently edits: None = global
        # (whole image); otherwise the id of an area in area_layers. Session
        # state only — never persisted; always defaults to global on load.
        self.active_area_id: Optional[str] = None
        self.undo_stack: list = []  # Snapshots for Ctrl+Z, most recent last
        self.contrast_base: int = 0      # Non-destructive base contrast added internally; slider shows 0
        self.temperature_base: int = 0   # Non-destructive base temperature offset; slider shows 0
        self.brightness_base: int = -8   # Non-destructive base brightness offset; slider shows 0
        self.histogram_image = None
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
            
            # Populate thumbnail and preview
            self.update_thumbnail_and_preview()
        else:
            raise ValueError(f"Could not read image from file: {self.file_path}")
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

    def reload_image(self) -> None:
        """
        Reload the image from the file path and update resized_raw, thumbnail, and preview.
        This is useful if the image file has been modified externally.
        """
        self.contrast_base = 0      # Clear base offsets when reverting to original scan
        self.temperature_base = 0
        self.brightness_base = -8   # Always applied; not tied to conversion state
        self.conversion_inputs = None
        img = self.read_image(self.file_path, max_long_side=1080)
        if img is not None:
            self.resized_raw = img
            # Recalculate tint balance factor for the new image
            self.tint_balance_factor = self._calculate_tint_balance_factor()
            self.update_thumbnail_and_preview()
        else:
            logging.error(f"Failed to reload image: {self.file_path}")

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

    def _apply_input_icc(self, arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Convert a freshly-decoded scan from the globally-assigned input ICC
        profile into the working sRGB encoding, before any conversion or
        adjustment. No-op when no input profile is set. Applied inside
        read_image so preview, hi-res zoom, and export all inherit it
        identically (resolution-independent point operation)."""
        if arr is None:
            return arr
        profile = color_management.get_active_input_profile()
        if profile is None:
            return arr
        try:
            return profile.apply(arr)
        except Exception as e:
            logging.warning(f"Input ICC profile could not be applied: {e}")
            return arr

    def read_image(self, file_path: str, preview = True, max_long_side: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Read an image file. preview=True decodes RAW at half size.
        max_long_side, when given, downsizes to that size INSIDE the reader —
        for RAW this happens before white-level scaling (both steps are
        linear, so the order only changes values by <=3 LSB, and scaling at
        preview size instead of decode size saves ~100 ms per image).
        """
        # Ensure file path is properly encoded for Unicode support
        file_path = os.path.normpath(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        
        # Treat FFF files as TIFF files
        if ext == ".fff":
            ext = ".tiff"
        
        if ext in [".cr3", ".cr2", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf", ".srw", ".pef", ".3fr"]:
            try:
                print(f"Starting RAW processing for: {os.path.basename(file_path)}")
                start_time = time.time()
                
                with rawpy.imread(file_path) as raw:
                    # Capture sensor ceiling before postprocess (e.g. 16383 for 14-bit)
                    white_level = raw.white_level

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

                    if is_monochrome:
                        print(f"Detected monochrome sensor for: {os.path.basename(file_path)}")
                        # For monochrome sensors, use different processing
                        rgb = raw.postprocess(
                            output_bps=16,
                            no_auto_bright=False,
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
                        # Pure/raw sensor readout with minimal processing (greenish result)
                        rgb = raw.postprocess(
                            output_bps=16,
                            no_auto_bright=True,      # Consistent absolute sensor values across all frames
                            gamma=(1, 1),            # Linear gamma (no gamma correction)
                            user_flip=0,              # No rotation
                            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,  # Simple linear demosaic
                            half_size=preview,        # Process at half resolution - much faster!
                            use_camera_wb=False,      # No camera white balance
                            use_auto_wb=False,        # No auto white balance
                            output_color=rawpy.ColorSpace.raw,  # Raw color space (no color correction)
                            no_auto_scale=True,       # No automatic scaling
                            four_color_rgb=False,     # Standard 3-color processing
                        )

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
                    if white_level > 0 and white_level < 65535:
                        print(f"Scaling RAW from {white_level}-ceiling to 16-bit (factor {65535.0/white_level:.4f})")
                        rgb = np.clip(
                            rgb.astype(np.float32) * (65535.0 / white_level),
                            0, 65535
                        ).astype(np.uint16)
                
                elapsed_time = time.time() - start_time
                print(f"RAW processing completed in {elapsed_time:.3f} seconds")
                # Burn in the global input ICC (if any) on the decoded scan,
                # before negative conversion / adjustments.
                return self._apply_input_icc(rgb)
            except Exception as e:
                logging.exception(f"Failed to read RAW image: {file_path}")
                return None
        else:
            # Handle Unicode file paths and OpenCV TIFF issues properly
            img = None
            
            # Try multiple reading methods for better compatibility
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
                    if file_path.lower().endswith(('.tif', '.tiff')) and TIFFFILE_AVAILABLE:
                        img = tifffile.imread(file_path)
                        # Convert to OpenCV format (BGR if color, grayscale if mono)
                        if len(img.shape) == 3 and img.shape[2] == 3:
                            # RGB to BGR conversion for OpenCV compatibility
                            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
                return None
            # Convert grayscale to RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert to 16-bit if needed
            if img.dtype != np.uint16:
                img = img.astype(np.uint16) * 257 if img.dtype == np.uint8 else img
            # Sliced images read only their region of the source
            img = self._apply_source_ops(img)
            # This branch always reads at full resolution regardless of `preview`
            self.original_full_size = (img.shape[0], img.shape[1])
            if max_long_side:
                img = self.resize_image_to_max_pixel(img, max_long_side)
            # Burn in the global input ICC (if any) before conversion/adjustments.
            return self._apply_input_icc(img)
        
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
        # own the look, so the auto-brightness is skipped.
        display_img = adjusted_img if self.converted else self._auto_brightness_for_preview(adjusted_img)

        # Create thumbnail
        thumb_img = self.resize_image_to_max_pixel(display_img, thumbnail_size)
        thumb_img_8 = to_8bit(thumb_img)
        qimage = self.generate_qimage_from_np_array_8(thumb_img_8)
        self.thumbnail = QPixmap.fromImage(qimage)

        # Create preview
        preview_img = self.resize_image_to_max_pixel(display_img, preview_size)
        qimage = self.generate_qimage_from_np_array_8(to_8bit(preview_img))
        self.resized_preview = QPixmap.fromImage(qimage)
        # Calculate histogram for the 8-bit preview (RGB). When a crop is set,
        # the histogram is computed over only the kept (cropped) region so it
        # matches what the canvas shows. Same normalized-rect + angle contract
        # as the display/export crop path; apply_crop_to_image returns the
        # input unchanged when crop_rect is None.
        hist = {}
        preview_img_8bit = to_8bit(preview_img)
        hist_source = apply_crop_to_image(
            preview_img_8bit, self.crop_rect, getattr(self, "crop_angle", 0.0) or 0.0)
        for i, color in enumerate(['r', 'g', 'b']):
            hist[color] = cv2.calcHist([hist_source], [i], None, [256], [0, 256]).flatten()

        # Generate a histogram image (RGB channels overlaid). Fully vectorized:
        # the previous per-column cv2.addWeighted loop took ~55 ms per call
        # (every load AND every slider tick); this renders in ~2 ms.
        hist_height = 150
        hist_width = 256
        alpha = 0.33  # transparency for histogram curves
        hist_img_f = np.full((hist_height, hist_width, 3), 180.0, dtype=np.float32)

        # Find the global max value across all channels for adaptive scaling
        max_val = max([hist[c].mean() for c in hist]) * 6
        scale = max(max_val, 0.1)  # prevent division by zero / too-flat lines

        rows = np.arange(hist_height, dtype=np.int32)[:, None]  # (150, 1)
        masks = []
        # RGB order: hist_img is rendered via QImage Format_RGB888, not cv2.imshow,
        # so colors must be RGB — (255,0,0) draws the R-data curve in red.
        for channel, color in zip(('r', 'g', 'b'),
                                  ((255, 0, 0), (0, 255, 0), (0, 0, 255))):
            h_scaled = np.clip((hist[channel] / scale) * (hist_height - 1),
                               0, hist_height - 1)
            col_tops = hist_height - h_scaled.astype(np.int32)  # (256,)
            mask = rows >= col_tops[None, :]                    # (150, 256) bool
            masks.append(mask)
            hist_img_f[mask] = (hist_img_f[mask] * (1.0 - alpha)
                                + np.array(color, dtype=np.float32) * alpha)

        hist_img = hist_img_f.astype(np.uint8)
        # Where all three channels overlap, set to white
        hist_img[masks[0] & masks[1] & masks[2]] = (235, 235, 235)
        hist_img = np.ascontiguousarray(hist_img)

        self.histogram_image = QPixmap.fromImage(self.generate_qimage_from_np_array_8(hist_img))

    def generate_qimage_from_np_array_8(self, thumb_img_8):
        h, w, ch = thumb_img_8.shape
        bytes_per_line = ch * w
        qimage = QImage(
            thumb_img_8.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        return qimage

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Map an RGB image to a single luminance channel (Rec.601 weights:
        0.299R + 0.587G + 0.114B) replicated across all three channels, so it
        renders/exports as black & white while keeping the RGB shape and dtype
        every downstream stage expects. Returns a new array."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def apply_adjustments(self, image: np.ndarray, settings=None, contrast_base=None,
                          temperature_base=None, brightness_base=None,
                          color_profile=None, areas_override=None,
                          dust_strokes_override=None) -> np.ndarray:
        """Apply the slider adjustments. The optional overrides let the zoom
        hi-res worker render from a snapshot taken at request time instead of
        live state the GUI thread may be mutating concurrently."""
        s = self.adjustment_settings if settings is None else settings
        cb = self.contrast_base if contrast_base is None else contrast_base
        tb = self.temperature_base if temperature_base is None else temperature_base
        bb = self.brightness_base if brightness_base is None else brightness_base
        profile = self.color_profile if color_profile is None else color_profile
        areas = (getattr(self, "area_layers", []) if areas_override is None
                 else areas_override)
        has_areas = bool(areas) and any(a.get("enabled") for a in areas)
        # Dust healing runs FIRST (on the converted positive), before the
        # early-return, so a heal-only edit still applies. Resolution
        # independent: strokes are normalized to this array's dimensions.
        strokes = (getattr(self, "dust_heal_strokes", None)
                   if dust_strokes_override is None else dust_strokes_override)
        if strokes:
            from core.dust_removal import inpaint_dust
            image = inpaint_dust(image, strokes)
        if not s and cb == 0 and tb == 0 and bb == 0 and not has_areas:
            # No slider/base/area adjustments — but Black & White still has to
            # map the image to a single luminance channel.
            return self._to_grayscale(image) if profile == "bw" else image
        adjusted = adjust_image_opencl(image,
                     s.get('temperature', 0) + tb,
                     s.get('tint', 0),
                     s.get('exposure', 0),
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
                     # Per-color-band sliders ride the same GPU pass (or the
                     # CPU fallback); None keeps the inactive case free.
                     band_settings=(s if any(s.get(k, 0)
                                             for k in BAND_ADJUSTMENT_KEYS)
                                    else None))
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
        curves = s.get('curves')
        if curves:
            adjusted = apply_curves(adjusted, curves)
        return adjusted

    def render_hires_base(self, max_long_side: Optional[int] = None,
                          conversion_inputs=None) -> Optional[np.ndarray]:
        """
        Re-decode this image and reproduce its conversion at the RAW
        half-size resolution (capped at max_long_side — important for
        non-RAW sources, which otherwise decode at FULL resolution),
        color-matched to the 1080 preview: the conversion is replayed from
        the snapshot captured at convert time (conversion_inputs), never
        from live editable state, so it always matches what the preview
        shows. Returns the converted, PRE-adjustment uint16 array (or the
        raw scan for un-converted images). Runs on a worker thread for the
        zoom detail view; never touches resized_raw or any preview state.
        """
        ci = conversion_inputs if conversion_inputs is not None else self.conversion_inputs
        if self.converted and ci is None:
            return None  # converted through an unknown path — no replay possible
        t0 = time.time()
        img = self.read_image(self.file_path, preview=True, max_long_side=max_long_side)
        if img is None:
            return None
        print(f"Hi-res decode: {time.time() - t0:.2f}s, shape {img.shape}")
        if not self.converted:
            return img

        from core.ccr_processor import (compute_reference_norm_params,
                                        apply_reference_normalization,
                                        apply_bwpoint_normalization)
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
            # The bwpoint pipeline bakes the fine rotation into the preview
            # pixels BEFORE normalization — replicate that here so the
            # detail layer lines up with the preview content.
            fine_angle = ci.get("fine_rot", 0) / 100.0
            if fine_angle != 0:
                h0, w0 = img.shape[:2]
                rot = cv2.getRotationMatrix2D((w0 // 2, h0 // 2), -fine_angle, 1.0)
                img = cv2.warpAffine(img, rot, (w0, h0), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            black_point, white_point = ci["bw"]
            out = apply_bwpoint_normalization(img, black_point, white_point)
        else:
            return None
        print(f"Hi-res convert: {time.time() - t0:.2f}s")
        return out

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
            # Each stroke nests a points list — deep-copy so a later edit
            # can't corrupt the snapshot (same reasoning as area_layers).
            "dust_heal_strokes": copy.deepcopy(getattr(self, "dust_heal_strokes", [])),
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
        self.dust_heal_strokes = copy.deepcopy(state.get("dust_heal_strokes", []))
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

        Linearly stretches the tonal range so the central 90% of pixel values
        (5th-95th percentile) is spread across the full 0-65535 range. A single
        global map is applied to all channels so the colour balance of the scan
        is preserved (no per-channel white balancing).

        Returns a new array; the input (resized_raw) is never modified, so this
        has no effect on conversion, export, or any other processing.
        """
        if image is None:
            return image

        # Estimate percentiles on a 4x4-subsampled view (16x fewer pixels,
        # visually identical anchors) and stretch in place — this runs on
        # every preview refresh of an un-converted image.
        sample = image[::4, ::4]
        lo, hi = (float(v) for v in np.percentile(sample, (5.0, 95.0)))
        if hi <= lo:
            # Degenerate (flat) image — nothing meaningful to stretch.
            return image

        img = image.astype(np.float32)
        np.subtract(img, lo, out=img)
        np.multiply(img, 65535.0 / (hi - lo), out=img)
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


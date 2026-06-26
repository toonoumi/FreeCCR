from typing import List, Optional
import cv2
from core.ccr_image import CCRImage
from core.ccr_processor import (ccr_normalize_with_reference, ccr_normalize_with_bwpoint,
                                ccr_normalize_with_refparams, ccr_export_positive,
                                auto_fine_angle, auto_frame,
                                auto_frame_v2, log_bwpoint_slopes)
import os
import glob
import concurrent.futures
import time
import uuid
import copy


def _new_slice_group() -> str:
    """A globally-unique id shared by all slices of one slicing operation.
    Used to identify a slice round for 'Reset Slice' without relying on
    geometry or display names (both of which collide across duplicated
    lineages)."""
    return uuid.uuid4().hex

class CCRBackend:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CCRBackend, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.images: List[CCRImage] = []
        self.file_paths: List[str] = []
        # Global Positive mode: when True, RAWs decode as normal sRGB positives
        # and the negative pipeline (conversion / B-W point / reference frame) is
        # bypassed — every image is editable/exportable directly. App-wide, like
        # the input ICC profile; persisted by MainWindow. See spec/positive-mode.md.
        self.positive_mode: bool = False
        self.white_point_bgr = None  # (B, G, R) of dense/exposed area
        self.black_point_bgr = None  # (B, G, R) of transparent/clear area
        # Global input ICC profile (one app-wide setting; applied to every
        # decode before conversion/adjustments). The parsed profile lives in
        # color_management's module-level holder; these mirror it for the UI.
        self.input_icc_path: Optional[str] = None
        self.input_icc_name: Optional[str] = None
        # Active DCP (DNG camera profile) — mutually exclusive with the input ICC.
        self.input_dcp_path: Optional[str] = None
        self.input_dcp_name: Optional[str] = None
        # The active profile's path within the camera-profile library (or None).
        # The library (camera_profiles/) keeps every imported/generated profile;
        # the active one is just a selection, not a copy. See spec/camera-profile-library.md.
        self.active_profile_path: Optional[str] = None
        # Catalog entries of ACTUAL images removed from the list this
        # session: {file_path: {"signature": sig, "entries": {name: state}}}.
        # Removal must not lose their stored edits, so saves merge these
        # back into the records (duplicates, by contrast, are deleted).
        self._catalog_preserved = {}

    def load_images_from_files(self, file_paths: List[str], cancel_flag=None) -> int:
        """Load files (with catalog restore). Returns the number of FILES
        that produced at least one image (a sliced file yields several)."""
        self.images.clear()
        self.file_paths = file_paths
        # A fresh batch is a new roll: drop any B/W point sampled from the
        # previous roll so its film base / dense-area density can't carry over
        # (different stocks differ). Only the in-memory fields are cleared — the
        # QSettings copy is intact and still restores at next app start.
        self.black_point_bgr = None
        self.white_point_bgr = None
        # Preserved removal states belong to the previous batch (the open
        # flows save the catalog before loading a new one)
        self._catalog_preserved = {}

        # Read the catalog ONCE for the whole batch — per-file reads would
        # re-parse the entire JSON N times across the loader threads.
        from core.catalog import create_images_for_path, load_catalog
        try:
            catalog_data = load_catalog()
        except Exception:
            catalog_data = None

        def load_single_image(path):
            try:
                if cancel_flag and cancel_flag():
                    return path, None
                print(f"Loading image: {os.path.basename(path)}")
                # Restores cataloged state (slices, conversion, adjustments)
                # when this file was processed before; plain load otherwise.
                imgs = create_images_for_path(path, catalog_data=catalog_data)
                for order, img in enumerate(imgs):
                    img._catalog_order = order  # keep slice order within a file
                print(f"Successfully loaded: {os.path.basename(path)}")
                return path, imgs
            except Exception as e:
                print(f"Failed to load {os.path.basename(path)}: {e}")
                return path, None
        
        # Use parallel loading with ThreadPoolExecutor
        max_workers = min(8, os.cpu_count() or 1)
        
        loaded_file_count = 0
        if max_workers == 1:
            # Sequential fallback
            for path in file_paths:
                if cancel_flag and cancel_flag():
                    break
                _path, imgs = load_single_image(path)
                if imgs:
                    self.images.extend(imgs)
                    loaded_file_count += 1
        else:
            # Parallel loading - collect into local list to avoid concurrent modification
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {executor.submit(load_single_image, path): path for path in file_paths}

                for future in concurrent.futures.as_completed(future_to_path):
                    if cancel_flag and cancel_flag():
                        break
                    path, imgs = future.result()
                    if imgs:
                        results.extend(imgs)
                        loaded_file_count += 1

            # Slices of one file keep their catalog order within the file
            results.sort(key=lambda img: (os.path.basename(img.file_path),
                                          getattr(img, "_catalog_order", 0)))
            self.images = results

        # Keep file_paths derived from actually-loaded images so the two lists stay in sync
        self.file_paths = [img.file_path for img in self.images]
        return loaded_file_count

    def clear(self):
        """
        Clear all images and file paths from the backend.
        This is useful for resetting the state of the backend.
        """
        self.images.clear()
        self.file_paths.clear()

    def load_images_from_folder(self, folder_path: str, extensions: Optional[List[str]] = None, cancel_flag=None):
        if extensions is None:
            extensions = [
                "*.dng", "*.tif", "*.tiff", "*.arw", "*.nef", "*.cr2", "*.cr3",
                "*.raf", "*.png", "*.jpg", "*.jpeg", "*.rw2", "*.3fr", "*.sr2", "*.orf", "*.pef", "*.heic", "*.heif",
                "*.fff", "*.dcr", "*.kdc", "*.x3f", "*.srw", "*.erf", "*.nrw", "*.ptx", "*.r3d", "*.raf"
            ]
        
        # Normalize folder path to handle Unicode characters
        folder_path = os.path.normpath(folder_path)
        file_paths = []
        print(f"Loading from folder: {folder_path}")
        print(f"Folder exists: {os.path.exists(folder_path)}")
        
        # Check what files are actually in the folder
        if os.path.exists(folder_path):
            try:
                all_files = os.listdir(folder_path)
                print(f"All files in folder: {all_files[:10]}...")  # Show first 10 files
            except UnicodeDecodeError as e:
                print(f"Unicode error reading folder contents: {e}")
                # Try with different encoding on Windows
                try:
                    import sys
                    if sys.platform == "win32":
                        # Use os.scandir for better Unicode support on Windows
                        all_files = [entry.name for entry in os.scandir(folder_path)]
                        print(f"All files in folder (via scandir): {all_files[:10]}...")
                    else:
                        raise e
                except Exception as e2:
                    print(f"Failed to read folder contents: {e2}")
                    return
        
        # Try both lowercase and uppercase extensions
        for ext in extensions:
            try:
                # Original extension
                pattern1 = os.path.join(folder_path, ext)
                matches1 = glob.glob(pattern1)
                file_paths.extend(matches1)
                
                # Uppercase extension
                pattern2 = os.path.join(folder_path, ext.upper())
                matches2 = glob.glob(pattern2)
                file_paths.extend(matches2)
                
                print(f"Pattern {ext}: {len(matches1)} files, {ext.upper()}: {len(matches2)} files")
            except Exception as e:
                print(f"Error processing pattern {ext}: {e}")
                # Fallback: manually check files
                try:
                    if 'all_files' in locals():
                        ext_no_star = ext[2:] if ext.startswith('*.') else ext  # Remove '*.'
                        matching_files = [f for f in all_files if f.lower().endswith(ext_no_star.lower())]
                        for f in matching_files:
                            full_path = os.path.join(folder_path, f)
                            if os.path.isfile(full_path):
                                file_paths.append(full_path)
                        print(f"Manual pattern {ext}: {len(matching_files)} files")
                except Exception as e2:
                    print(f"Fallback pattern matching failed for {ext}: {e2}")
        
        # Remove duplicates and normalize paths
        file_paths = list(set(os.path.normpath(path) for path in file_paths))
        
        print(f"Found {len(file_paths)} files total: {file_paths[:5]}...")  # Show first 5 files
        
        # Load images and track success/failure (counted in FILES — a sliced
        # file restores as several images)
        loaded_files = self.load_images_from_files(sorted(file_paths), cancel_flag=cancel_flag) or 0
        failed_count = len(file_paths) - loaded_files

        print(f"Loading complete: {loaded_files} files loaded successfully "
              f"({len(self.images)} images), {failed_count} failed")

    def get_image_by_index(self, idx: int) -> Optional[CCRImage]:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx]
        return None
    
    def set_reference_frame_by_index(self, idx: int, reference_frame: Optional[tuple[int, int, int, int]]):
        if idx is not None and 0 <= idx < len(self.images):
            self.images[idx].reference_frame = reference_frame

    def get_reference_frame_by_index(self, idx: int) -> Optional[tuple[int, int, int, int]]:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx].reference_frame
        return None

    def get_preview_w_ref_frame_by_index(self, idx: int) -> Optional[CCRImage]:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx]
        return None

    def get_image_horizontal_flip_by_index(self, idx: int) -> bool:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx].horizontal_mirrored
        return False
    def set_image_horizontal_flip_by_index(self, idx: int, flip: bool):
        if idx is not None and 0 <= idx < len(self.images):
            self.images[idx].horizontal_mirrored = flip

    def get_image_vertical_flip_by_index(self, idx: int) -> bool:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx].vertical_mirrored
        return False
    
    def set_image_vertical_flip_by_index(self, idx: int, flip: bool):
        if idx is not None and 0 <= idx < len(self.images):
            self.images[idx].vertical_mirrored = flip
    
    def get_image_fine_rotation_by_index(self, idx: int) -> float:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx].fine_rotation_angle
        return 0.0
    
    def set_image_fine_rotation_by_index(self, idx: int, angle: float):
        if idx is not None and 0 <= idx < len(self.images):
            self.images[idx].fine_rotation_angle = angle
    
    def get_image_rotation_by_index(self, idx: int) -> float:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx].rotation_angle
        return 0.0
    
    def set_image_rotation_by_index(self, idx: int, angle: int):
        if idx is not None and 0 <= idx < len(self.images):
            self.images[idx].rotation_angle = angle
    
    def get_converted_state_by_index(self, idx: int) -> bool:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx].converted
        return False

    def get_thumbnail_by_index(self, idx: int) -> Optional[CCRImage]:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx].thumbnail
        return None
    
    def get_preview_by_index(self, idx: int) -> Optional[CCRImage]:
        if idx is not None and 0 <= idx < len(self.images):
            image = self.images[idx]
            preview = image.resized_preview.copy() if image.resized_preview is not None else None
            return preview
        return None

    def get_histogram_image_by_index(self, idx: int) -> Optional[CCRImage]:
        if idx is not None and 0 <= idx < len(self.images):
            image = self.images[idx]
            histogram_image = image.histogram_image if image.histogram_image is not None else None
            return histogram_image
        return None
    
    def set_adjustment_by_index(self, idx: int, adjustment: dict):
        """
        Sets the adjustment parameters for the image at the given index.
        The adjustment dictionary can contain keys like 'fine_rotation_angle', 'rotation_angle',
        'horizontal_mirrored', 'vertical_mirrored', etc.
        """
        if idx is not None and 0 <= idx < len(self.images):
            self.images[idx].adjustment_settings = adjustment
            self.images[idx].update_thumbnail_and_preview()

    def get_adjustment_by_index(self, idx: int) -> Optional[dict]:
        """
        Returns the adjustment parameters for the image at the given index.
        The returned dictionary can contain keys like 'fine_rotation_angle', 'rotation_angle',
        'horizontal_mirrored', 'vertical_mirrored', etc.
        """
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx].adjustment_settings
        return None       

    def apply_adjustment_by_index(self, idx: int):
        """
        Applies the adjustment settings to the image at the given index.
        This method will update the resized_raw and thumbnail of the image.
        """
        if idx is not None and 0 <= idx < len(self.images):
            image_obj = self.images[idx]
            image_obj.update_thumbnail_and_preview()

    # --- Area editing (local masked adjustment layers) ---------------------
    # The SlidersPanel edits whichever LAYER is active: the global
    # (whole-image) adjustment_settings when active_area_id is None, else the
    # selected area's own settings dict. These accessors route there so the
    # panel needs no special-casing.

    def _image_at(self, idx: int) -> Optional[CCRImage]:
        if idx is not None and 0 <= idx < len(self.images):
            return self.images[idx]
        return None

    def get_active_settings_by_index(self, idx: int) -> Optional[dict]:
        """The adjustment_settings dict of the active layer (global or area)."""
        img = self._image_at(idx)
        return img.active_settings() if img is not None else None

    def set_active_settings_by_index(self, idx: int, settings: dict,
                                     reprocess: bool = True):
        """Write settings to the active layer (global dict or the active
        area's settings), then optionally reprocess."""
        img = self._image_at(idx)
        if img is None:
            return
        area = img.get_area(img.active_area_id)
        if area is not None:
            area["settings"] = settings
        else:
            img.adjustment_settings = settings
        if reprocess:
            img.update_thumbnail_and_preview()

    def get_areas_by_index(self, idx: int) -> list:
        img = self._image_at(idx)
        return img.area_layers if img is not None else []

    def get_active_area_id_by_index(self, idx: int) -> Optional[str]:
        img = self._image_at(idx)
        return img.active_area_id if img is not None else None

    def set_active_area_by_index(self, idx: int, area_id: Optional[str]):
        """Select the layer the panel edits: None = global (whole image),
        else an area id. A stale id falls back to global."""
        img = self._image_at(idx)
        if img is None:
            return
        img.active_area_id = area_id if (area_id is None
                                         or img.get_area(area_id)) else None

    def _default_area(self, img: CCRImage, kind: str) -> dict:
        """A new area centered on the image. The circle default is on-screen
        circular (radius = 25% of the shorter side) by expressing rx/ry as
        fractions of width/height respectively."""
        h, w = (img.resized_raw.shape[:2]
                if img.resized_raw is not None else (1, 1))
        if kind == "gradient":
            geom = {"x0": 0.30, "y0": 0.50, "x1": 0.70, "y1": 0.50}
        else:
            kind = "circle"
            rpx = 0.25 * min(w, h)
            geom = {"cx": 0.50, "cy": 0.50,
                    "rx": rpx / max(w, 1), "ry": rpx / max(h, 1)}
        return {
            "id": uuid.uuid4().hex,
            "kind": kind,
            "enabled": True,
            "feather": 0.25,
            "angle": 0.0,
            "geometry": geom,
            "settings": {},
        }

    def add_area_by_index(self, idx: int, kind: str) -> Optional[str]:
        """Append a new area, make it the active layer, and return its id."""
        img = self._image_at(idx)
        if img is None:
            return None
        area = self._default_area(img, kind)
        img.area_layers.append(area)
        img.active_area_id = area["id"]
        img.update_thumbnail_and_preview()
        return area["id"]

    def remove_area_by_index(self, idx: int, area_id: str):
        img = self._image_at(idx)
        if img is None:
            return
        img.area_layers = [a for a in img.area_layers
                           if a.get("id") != area_id]
        if img.active_area_id == area_id:
            img.active_area_id = None
        img.update_thumbnail_and_preview()

    def set_area_enabled_by_index(self, idx: int, area_id: str, enabled: bool):
        img = self._image_at(idx)
        if img is None:
            return
        area = img.get_area(area_id)
        if area is not None:
            area["enabled"] = bool(enabled)
            img.update_thumbnail_and_preview()

    def update_area_geometry_by_index(self, idx: int, area_id: str,
                                      geometry=None, angle=None, feather=None,
                                      reprocess: bool = True):
        """Update an area's mask geometry/rotation/feather (used by the canvas
        overlay drag and the feather slider)."""
        img = self._image_at(idx)
        if img is None:
            return
        area = img.get_area(area_id)
        if area is None:
            return
        if geometry is not None:
            area["geometry"] = dict(geometry)
        if angle is not None:
            area["angle"] = float(angle)
        if feather is not None:
            area["feather"] = float(feather)
        if reprocess:
            img.update_thumbnail_and_preview()

    def get_image_count(self) -> int:
        return len(self.images)
    
    def get_total_file_paths(self) -> int:
        return len(self.file_paths)

    def get_image_by_path(self, file_path: str) -> Optional[CCRImage]:
        for img in self.images:
            if img.file_path == file_path:
                return img
        return None

    def get_all_file_paths(self) -> List[str]:
        return self.file_paths
    
    def unconvert_negative_by_index(self, idx: int):
        """
        Unconverts the negative image at the given index by resetting the converted state
        and reloading the original image.
        """
        if idx is not None and 0 <= idx < len(self.images):
            image_obj = self.images[idx]
            if image_obj.converted:
                # Reset conversion state
                image_obj.converted = False
                # Reload the original image
                image_obj.reload_image()
                # Update thumbnail and preview
                image_obj.update_thumbnail_and_preview()
            else:
                print(f"Image at index {idx} is not converted.")

    @staticmethod
    def _clear_edit_state(image_obj) -> None:
        """Clear every non-destructive edit (conversion, adjustments, reference
        frame, crop, orientation, colour profile, undo history) so the image
        matches a just-imported one. Slicing (source_ops/lineage) is preserved —
        a slice resets to its un-edited region, not back to the whole file."""
        image_obj.converted = False
        image_obj.adjustment_settings = {}
        image_obj.reference_frame = None
        image_obj.crop_rect = None
        image_obj.crop_angle = 0.0
        image_obj.rotation_angle = 0
        image_obj.fine_rotation_angle = 0
        image_obj.horizontal_mirrored = False
        image_obj.vertical_mirrored = False
        image_obj.color_profile = "color"
        image_obj.awb_enabled = False
        image_obj.awb_gains = None
        image_obj._awb_src_id = None
        # The conversion is gone, so the undo history (which never captured
        # the conversion anyway) would be inconsistent — drop it.
        image_obj.undo_stack = []

    def reset_image_by_index(self, idx: int) -> bool:
        """Reset the image at idx to its freshly-loaded state: undo the
        conversion and clear every non-destructive edit (adjustments,
        reference frame, crop, orientation, colour profile) so it matches a
        just-imported image. Slicing (source_ops/lineage) is preserved — a
        slice resets to its un-edited region, not back to the whole file
        (use "Reset Slice" for that). Returns False if idx is out of range."""
        if idx is None or not (0 <= idx < len(self.images)):
            return False
        image_obj = self.images[idx]
        # Clear edit state BEFORE reloading so the rebuilt preview reflects
        # the pristine settings. reload_image() re-decodes resized_raw and
        # resets the base offsets + conversion_inputs.
        self._clear_edit_state(image_obj)
        image_obj.reload_image()
        return True

    def _reload_decode_parallel(self, imgs, progress_callback=None) -> None:
        """Re-decode resized_raw for each image concurrently. read_image (RAW
        decode / file IO) is the slow part and releases the GIL, so a thread
        pool gives a near-linear speed-up. QPixmap building is deliberately NOT
        done here — it must run on the GUI thread, so the caller follows up with
        update_thumbnail_and_preview() per image on the main thread."""
        if not imgs:
            return
        total = len(imgs)
        max_workers = min(8, os.cpu_count() or 1)
        done = 0
        if max_workers <= 1 or total <= 1:
            for img in imgs:
                try:
                    img.reload_image_decode_only()
                except Exception as e:
                    print(f"Reset decode failed for {img.file_path}: {e}")
                done += 1
                if progress_callback:
                    progress_callback(done, total)
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(img.reload_image_decode_only): img for img in imgs}
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Reset decode failed for {futures[future].file_path}: {e}")
                done += 1
                if progress_callback:
                    progress_callback(done, total)

    def reset_images_by_indices(self, indices, progress_callback=None) -> bool:
        """Reset each image in indices to its freshly-loaded state. The slow
        per-image re-decode runs in parallel; previews are then rebuilt on the
        calling (GUI) thread. Returns True if at least one image was reset."""
        idxs = [i for i in sorted(set(indices))
                if i is not None and 0 <= i < len(self.images)]
        if not idxs:
            return False
        imgs = [self.images[i] for i in idxs]
        for img in imgs:
            self._clear_edit_state(img)
        self._reload_decode_parallel(imgs, progress_callback)
        # QPixmap thumbnails/previews on the caller's thread (GUI-thread safe).
        for img in imgs:
            img.update_thumbnail_and_preview()
        return True

    def _reconvert_in_place(self, image_obj) -> None:
        """Replay the stored negative conversion (bwpoint / reference / ref_params)
        on the freshly re-decoded resized_raw, in place — so a profile re-grade
        keeps the conversion. No-op if there is no recipe to replay."""
        ci = getattr(image_obj, "conversion_inputs", None)
        try:
            if ci is not None and ci.get("mode") == "bw":
                black_point, white_point = ci["bw"]
                processed = ccr_normalize_with_bwpoint(
                    image_obj, black_point, white_point)
            elif ci is not None and ci.get("mode") == "ref_params":
                processed = ccr_normalize_with_refparams(
                    image_obj, ci["p_lo"], ci["p_hi"], ci["od"])
            elif image_obj.reference_frame is not None:
                processed = ccr_normalize_with_reference(image_obj)
            else:
                return
            if processed is not None:
                image_obj.resized_raw = processed
                image_obj.converted = True
        except Exception as e:
            print(f"Re-convert failed for {getattr(image_obj, 'file_path', '?')}: {e}")

    def regrade_images_by_indices(self, indices, progress_callback=None) -> bool:
        """Re-decode each image under the CURRENT camera profile, KEEPING all of the
        user's work — the negative conversion (bwpoint / reference), colour
        adjustments, crop, flips, rotation and slice. Only the camera-profile decode
        changes; the conversion is replayed on the new decode. Backs 'Replace with
        current camera profile' (distinct from Reset, which drops everything).
        Returns True if at least one image was re-graded."""
        idxs = [i for i in sorted(set(indices))
                if i is not None and 0 <= i < len(self.images)]
        if not idxs:
            return False
        imgs = [self.images[i] for i in idxs]
        # The re-decode clears conversion_inputs + base offsets — snapshot the
        # recipe + converted flag so they can be restored and replayed after.
        saved = [(img.conversion_inputs, img.converted) for img in imgs]
        self._reload_decode_parallel(imgs, progress_callback)   # re-decode base + re-stamp signature
        for img, (ci, was_converted) in zip(imgs, saved):
            img.conversion_inputs = ci
            if was_converted:
                self._reconvert_in_place(img)
            img.update_thumbnail_and_preview()
        return True

    def convert_negative_by_index(self, idx: int):
        """
        Converts the negative image at the given index using CCR normalization with reference.
        Updates the resized_raw in the CCRImage object in-place.
        """
        if idx is not None and 0 <= idx < len(self.images):
            image_obj = self.images[idx]
            if image_obj.converted:
                #reload the image if it has already been converted
                image_obj.reload_image()
            try:
                processed = ccr_normalize_with_reference(image_obj)
                image_obj.resized_raw = processed
                image_obj.converted = True
                # Snapshot what this conversion was baked with (the zoom
                # hi-res replay must use these, not later edited values)
                image_obj.conversion_inputs = {
                    "mode": "ref",
                    "ref": tuple(image_obj.reference_frame),
                    "fine_rot": image_obj.fine_rotation_angle,
                }
                image_obj.update_thumbnail_and_preview()
            except Exception as e:
                print(f"Failed to convert image at index {idx}: {e}")

    def active_profile_signature(self) -> str:
        """The camera-profile signature an image decoded NOW would be graded
        under: 'none' in Positive mode (the profile isn't applied) or when the
        profile is disabled/unset, else 'icc:<desc>' / 'dcp:<name>'. Drives the
        per-image thumbnail mismatch warning."""
        from core import color_management
        if self.positive_mode:
            return "none"
        if color_management.camera_matrix_mode():
            return "camera_matrix"          # Adobe+autoscale decode, distinct from RAW "none"
        return color_management.active_profile_signature()

    @staticmethod
    def _profile_content_id(path):
        """A short content hash of a profile file, attached to the active profile
        so the per-image grading signature distinguishes two different profiles
        that happen to share a description/name (or both lack one)."""
        import hashlib
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:12]
        except OSError:
            return None

    # --- Camera-profile library (persistent in the app-data workspace) --------
    def camera_profiles_dir(self) -> str:
        """The library folder under %APPDATA%/FreeCCR that keeps every imported
        and IT8-generated camera profile (survives app updates/reinstalls — it is
        outside the install dir)."""
        from core.catalog import default_catalog_path
        d = os.path.join(os.path.dirname(default_catalog_path()), "camera_profiles")
        os.makedirs(d, exist_ok=True)
        return d

    def list_camera_profiles(self) -> list:
        """All profiles in the library: [{name, path, kind('icc'|'dcp')}], by name."""
        import glob
        out = []
        for ext, kind in ((".icc", "icc"), (".icm", "icc"), (".dcp", "dcp")):
            for p in glob.glob(os.path.join(self.camera_profiles_dir(), "*" + ext)):
                out.append({"name": os.path.splitext(os.path.basename(p))[0],
                            "path": p, "kind": kind})
        return sorted(out, key=lambda x: x["name"].lower())

    def import_camera_profile(self, src_path: str) -> str:
        """Validate then copy an external .icc/.icm/.dcp into the library, keeping
        it for re-selection. Returns the new library path. Raises (UnsupportedICCError
        / DcpError / ValueError) on an unparseable or unsupported file."""
        import shutil
        ext = os.path.splitext(src_path)[1].lower()
        if ext == ".dcp":
            from core import dcp_profile
            dcp_profile.parse_dcp(src_path)                  # validate
        elif ext in (".icc", ".icm"):
            from core import color_management
            color_management.load_input_profile(src_path)    # validate
        else:
            raise ValueError(f"Unsupported profile type: {ext}")
        base, e = os.path.splitext(os.path.basename(src_path))
        dst = os.path.join(self.camera_profiles_dir(), base + e)
        n = 1
        srcn = os.path.normcase(os.path.abspath(src_path))
        while os.path.exists(dst) and os.path.normcase(os.path.abspath(dst)) != srcn:
            dst = os.path.join(self.camera_profiles_dir(), f"{base} ({n}){e}")
            n += 1
        if os.path.normcase(os.path.abspath(dst)) != srcn:
            shutil.copyfile(src_path, dst)
        return dst

    # Non-removable picker sentinel: no profile, decode in Adobe RGB + auto-scale
    # (the camera's built-in matrix) instead of bare camera-native RAW ("None").
    CAMERA_MATRIX = "__camera_matrix__"

    def set_active_profile(self, path: Optional[str]) -> Optional[str]:
        """Activate the library profile at `path` (.icc/.icm/.dcp), the special
        CAMERA_MATRIX mode, or None (bare RAW). Never copies or deletes — the
        library file is the source. Returns the display name, or None. Raises on a
        bad file."""
        from core import color_management
        if path == self.CAMERA_MATRIX:
            color_management.set_active_input_profile(None)
            color_management.set_active_dcp_profile(None)
            color_management.set_camera_matrix_mode(True)
            self.input_icc_path = self.input_icc_name = None
            self.input_dcp_path = self.input_dcp_name = None
            self.active_profile_path = self.CAMERA_MATRIX
            return "Camera Matrix"
        color_management.set_camera_matrix_mode(False)   # any real selection clears it
        if not path:
            color_management.set_active_input_profile(None)
            color_management.set_active_dcp_profile(None)
            self.input_icc_path = self.input_icc_name = None
            self.input_dcp_path = self.input_dcp_name = None
            self.active_profile_path = None
            return None
        ext = os.path.splitext(path)[1].lower()
        if ext == ".dcp":
            from core import dcp_profile
            prof = dcp_profile.parse_dcp(path)
            prof.content_id = self._profile_content_id(path)
            color_management.set_active_dcp_profile(prof)
            color_management.set_active_input_profile(None)
            self.input_dcp_path = path
            self.input_dcp_name = prof.name or os.path.basename(path)
            self.input_icc_path = self.input_icc_name = None
            name = self.input_dcp_name
        else:
            prof = color_management.load_input_profile(path)
            prof.content_id = self._profile_content_id(path)
            color_management.set_active_input_profile(prof)
            color_management.set_active_dcp_profile(None)
            self.input_icc_path = path
            self.input_icc_name = prof.description or os.path.basename(path)
            self.input_dcp_path = self.input_dcp_name = None
            name = self.input_icc_name
        self.active_profile_path = path
        return name

    def delete_camera_profile(self, path: str) -> bool:
        """Remove a profile file from the library. Deactivates it first if active."""
        if (self.active_profile_path and os.path.normcase(os.path.abspath(path))
                == os.path.normcase(os.path.abspath(self.active_profile_path))):
            self.set_active_profile(None)
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    # --- Global input ICC profile -----------------------------------------
    def _input_icc_storage_path(self) -> str:
        """Persistent working-copy path inside the app-data folder (next to the
        edit catalog), so the profile survives the user moving/deleting the
        original .icc."""
        from core.catalog import default_catalog_path
        return os.path.join(os.path.dirname(default_catalog_path()), "input_profile.icc")

    def set_input_icc(self, src_path: str) -> str:
        """Assign a global input ICC profile from src_path. Parses it first
        (raising UnsupportedICCError / OSError on failure, leaving the current
        profile untouched), copies it into app data as the persistent working
        copy, activates it, and returns its description name. The caller persists
        the storage path (self.input_icc_path) and triggers reprocessing."""
        from core import color_management
        import shutil
        profile = color_management.load_input_profile(src_path)  # parse before mutating
        storage = self._input_icc_storage_path()
        if (os.path.normcase(os.path.abspath(src_path))
                != os.path.normcase(os.path.abspath(storage))):
            shutil.copyfile(src_path, storage)
        profile.content_id = self._profile_content_id(src_path)
        color_management.set_active_input_profile(profile)
        self.clear_input_dcp()                       # ICC and DCP are exclusive
        self.input_icc_path = storage
        self.input_icc_name = profile.description or os.path.basename(src_path)
        return self.input_icc_name

    def load_input_icc_from_storage(self, storage_path: str) -> Optional[str]:
        """Startup restore: activate a previously-saved working copy. Returns the
        profile name, or None if it could not be loaded (e.g. file gone)."""
        from core import color_management
        try:
            profile = color_management.load_input_profile(storage_path)
        except Exception as e:
            print(f"Could not load saved input ICC {storage_path}: {e}")
            return None
        profile.content_id = self._profile_content_id(storage_path)
        color_management.set_active_input_profile(profile)
        self.input_icc_path = storage_path
        self.input_icc_name = profile.description or os.path.basename(storage_path)
        return self.input_icc_name

    def clear_input_icc(self) -> None:
        """Remove the global input ICC profile and its working copy."""
        from core import color_management
        color_management.set_active_input_profile(None)
        self.input_icc_path = None
        self.input_icc_name = None
        try:
            storage = self._input_icc_storage_path()
            if os.path.exists(storage):
                os.remove(storage)
        except OSError:
            pass

    # --- DCP (DNG camera profile) — mirrors the input-ICC trio, exclusive ----
    def _input_dcp_storage_path(self) -> str:
        from core.catalog import default_catalog_path
        return os.path.join(os.path.dirname(default_catalog_path()), "input_profile.dcp")

    def set_input_dcp(self, src_path: str) -> str:
        """Assign a global DCP from src_path (parsed first, raising DcpError on
        failure). Clears any active ICC (exclusive slot), copies to the working
        copy, activates it. Returns the profile name."""
        from core import color_management, dcp_profile
        import shutil
        profile = dcp_profile.parse_dcp(src_path)    # parse before mutating
        storage = self._input_dcp_storage_path()
        if (os.path.normcase(os.path.abspath(src_path))
                != os.path.normcase(os.path.abspath(storage))):
            shutil.copyfile(src_path, storage)
        profile.content_id = self._profile_content_id(src_path)
        color_management.set_active_dcp_profile(profile)
        self.clear_input_icc()                       # DCP and ICC are exclusive
        self.input_dcp_path = storage
        self.input_dcp_name = profile.name or os.path.basename(src_path)
        return self.input_dcp_name

    def load_input_dcp_from_storage(self, storage_path: str) -> Optional[str]:
        from core import color_management, dcp_profile
        try:
            profile = dcp_profile.parse_dcp(storage_path)
        except Exception as e:
            print(f"Could not load saved DCP {storage_path}: {e}")
            return None
        profile.content_id = self._profile_content_id(storage_path)
        color_management.set_active_dcp_profile(profile)
        self.input_dcp_path = storage_path
        self.input_dcp_name = profile.name or os.path.basename(storage_path)
        return self.input_dcp_name

    def clear_input_dcp(self) -> None:
        """Remove the global DCP and its working copy."""
        from core import color_management
        color_management.set_active_dcp_profile(None)
        self.input_dcp_path = None
        self.input_dcp_name = None
        try:
            storage = self._input_dcp_storage_path()
            if os.path.exists(storage):
                os.remove(storage)
        except OSError:
            pass

    def reprocess_all_for_input_icc_change(self, progress_callback=None) -> None:
        """A change to the global input ICC changes the decode itself, so every
        loaded image is FULLY reset: the conversion and all non-destructive edits
        (adjustments, reference frame, crop, orientation, colour profile) are
        dropped and the scan is re-decoded fresh (in parallel). Replaying a
        conversion that was computed against the old decode would mis-colour the
        result, so we start from a clean slate. Slicing/lineage is preserved (a
        slice resets to its un-edited region)."""
        self.reset_images_by_indices(range(len(self.images)), progress_callback)

    def reprocess_all_for_positive_mode_change(self, progress_callback=None) -> None:
        """Re-decode every loaded image after the global Positive-mode toggle.

        Product decision (spec/positive-mode.md): KEEP adjustments, DROP
        conversion. reload_image re-decodes through read_image (which now reads
        the new global mode) and preserves adjustment_settings / crop / areas /
        orientation; we explicitly drop the conversion so a re-decoded scan is
        never left flagged converted. The reference_frame is kept so toggling
        back to negative restores the user's frame."""
        total = len(self.images)
        for i, img in enumerate(self.images):
            # Slices/duplicates INHERIT the parent's tint balance factor; their
            # own region would derive a different one, and reload_image always
            # recomputes it — so preserve the inherited value for those images.
            inherited_tbf = bool(img.source_ops) or bool(getattr(img, "is_duplicate", False))
            tbf = getattr(img, "tint_balance_factor", 1.0)
            try:
                img.converted = False
                img.conversion_inputs = None
                img.reload_image()                 # re-decode in the new global mode
                if inherited_tbf:
                    img.tint_balance_factor = tbf
                img.update_thumbnail_and_preview()
            except Exception as e:
                print(f"Reprocess after positive-mode change failed for {img.file_path}: {e}")
            if progress_callback:
                progress_callback(i + 1, total)

    def update_thumbnail_by_index(self, idx: int):
        """
        Updates the thumbnail for the image at the given index.
        This method is useful when the image has been modified and needs a new thumbnail.
        """
        if idx is not None and 0 <= idx < len(self.images):
            image_obj = self.images[idx]
            image_obj.update_thumbnail_and_preview()

    def auto_frame_all_images(self, progress_callback=None):
        """
        Automatically sets the reference frame for all images based on their content using parallel processing.
        """
        # Filter images that need processing
        images_to_process = [(idx, img) for idx, img in enumerate(self.images) if not img.converted]
        
        if not images_to_process:
            return
        
        total_images = len(images_to_process)
        completed_count = 0
        
        if progress_callback:
            progress_callback(0, total_images)
        
        def process_single_image(idx_img_tuple):
            idx, img = idx_img_tuple
            try:
                if img.fine_rotation_angle == 0 and img.reference_frame is None:              
                    img.fine_rotation_angle = int(auto_fine_angle(img.resized_raw, debug=False) * 100)  # Store as integer for fine rotation
                if img.reference_frame is None:
                    img.reference_frame = auto_frame_v2(img.resized_raw, img.fine_rotation_angle, debug=False)
                    self.convert_negative_by_index(idx)
                img.update_thumbnail_and_preview()
                return idx, True, None
            except Exception as e:
                return idx, False, str(e)
        
        # Use parallel processing
        max_workers = min(8, os.cpu_count() or 1)
        
        if max_workers == 1:
            # Sequential fallback
            for idx, img in images_to_process:
                try:
                    if img.fine_rotation_angle == 0 and img.reference_frame is None:              
                        img.fine_rotation_angle = int(auto_fine_angle(img.resized_raw, debug=False) * 100)  # Store as integer for fine rotation
                    if img.reference_frame is None:
                        img.reference_frame = auto_frame_v2(img.resized_raw, img.fine_rotation_angle, debug=False)
                        self.convert_negative_by_index(idx)
                    img.update_thumbnail_and_preview()
                except Exception as e:
                    print(f"Failed to auto-frame image at index {idx}: {e}")
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total_images)
        else:
            # Parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(process_single_image, (idx, img)): idx 
                               for idx, img in images_to_process}
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx, success, error = future.result()
                    if not success:
                        print(f"Failed to auto-frame image at index {idx}: {error}")
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_images)

    def convert_all_images(self):
        """
        Converts all images in the backend using CCR normalization with reference.
        Updates the resized_raw in each CCRImage object in-place.
        """
        for idx, img in enumerate(self.images):
            if img.converted:
                continue
            try:
                self.convert_negative_by_index(idx)
            except Exception as e:
                print(f"Failed to convert image at index {idx}: {e}")

    def export_image_by_index(self, idx: int, output_path: str, jpg_output: bool = False,
                              jpg_quality: int = 95, max_long_side: int = None,
                              output_colorspace: str = "srgb") -> bool:
        """
        Exports the processed image at the given index to the specified output path.
        Routes to bwpoint or reference-frame pipeline depending on how the image was converted.
        Returns True on success.
        """
        if idx is not None and 0 <= idx < len(self.images):
            image_obj = self.images[idx]
            try:
                if self.positive_mode:
                    # Positive mode: export the adjusted positive directly — no
                    # inversion, regardless of any leftover reference_frame.
                    ccr_export_positive(image_obj, output_path=output_path,
                                        jpg_out=jpg_output, jpg_quality=jpg_quality,
                                        max_long_side=max_long_side,
                                        output_colorspace=output_colorspace)
                    return True
                ci = getattr(image_obj, "conversion_inputs", None)
                if ci is not None and ci.get("mode") == "ref_params":
                    # Sliced child of a reference-converted parent: replay the
                    # stored conversion constants at full resolution.
                    ccr_normalize_with_refparams(image_obj, ci["p_lo"], ci["p_hi"], ci["od"],
                                                 output_path=output_path,
                                                 jpg_out=jpg_output, jpg_quality=jpg_quality,
                                                 max_long_side=max_long_side,
                                                 output_colorspace=output_colorspace)
                elif ci is not None and ci.get("mode") == "bw":
                    # Use the anchors BAKED at convert time — resampling the
                    # global points later must not change this image's export.
                    black_point, white_point = ci["bw"]
                    ccr_normalize_with_bwpoint(image_obj, black_point, white_point,
                                               output_path=output_path,
                                               jpg_out=jpg_output, jpg_quality=jpg_quality,
                                               max_long_side=max_long_side,
                                               output_colorspace=output_colorspace)
                elif image_obj.reference_frame is None and self.black_point_bgr is not None:
                    # Legacy/un-snapshotted B/W point conversion — global anchors.
                    # white_point_bgr may be None → default-slope mode.
                    ccr_normalize_with_bwpoint(image_obj, self.black_point_bgr, self.white_point_bgr,
                                               output_path=output_path,
                                               jpg_out=jpg_output, jpg_quality=jpg_quality,
                                               max_long_side=max_long_side,
                                               output_colorspace=output_colorspace)
                else:
                    ccr_normalize_with_reference(image_obj, output_path=output_path,
                                                 jpg_out=jpg_output,
                                                 jpg_quality=jpg_quality, max_long_side=max_long_side,
                                                 output_colorspace=output_colorspace)
                return True
            except Exception as e:
                print(f"Failed to export image at index {idx}: {e}")
        return False

    def export_items(self, items, jpg_output: bool = False, jpg_quality: int = 95,
                     max_long_side: int = None, output_colorspace: str = "srgb",
                     progress_callback=None, cancel_check=None) -> dict:
        """
        Exports specific images to explicit output paths using parallel processing.

        Args:
            items: list of (idx, absolute_output_path) tuples; filename building is
                   owned by the caller
            progress_callback: optional callable(current, total)
            cancel_check: optional zero-arg callable; once it returns True, queued
                          exports are abandoned (in-flight ones finish)

        Returns:
            dict with "exported", "failed", "cancelled" counts/flag and
            "failures" as a list of (idx, message) tuples.
        """
        import time
        total_start_time = time.time()

        result = {"exported": 0, "failed": 0, "cancelled": False, "failures": []}
        total_items = len(items)
        if total_items == 0:
            return result

        print(f"Starting export of {total_items} images...")

        def cancelled():
            return cancel_check is not None and cancel_check()

        def export_single_image(idx, output_path):
            if cancelled():
                return idx, None, None  # None success = not attempted
            start_time = time.time()
            success = self.export_image_by_index(idx, output_path, jpg_output=jpg_output,
                                                 jpg_quality=jpg_quality, max_long_side=max_long_side,
                                                 output_colorspace=output_colorspace)
            elapsed = time.time() - start_time
            base_name = os.path.basename(output_path)
            if success:
                print(f"Export completed for {base_name} in {elapsed:.2f}s")
                return idx, True, None
            print(f"Export failed for image {idx} after {elapsed:.2f}s")
            return idx, False, "export failed (see log)"

        # Use parallel processing for exports
        max_workers = min(4, os.cpu_count() or 1)  # Use very few workers for export to avoid I/O contention
        completed_count = 0

        if progress_callback:
            progress_callback(0, total_items)

        if max_workers == 1:
            # Sequential fallback
            for idx, output_path in items:
                if cancelled():
                    result["cancelled"] = True
                    break
                idx, success, error = export_single_image(idx, output_path)
                if success:
                    result["exported"] += 1
                else:
                    result["failed"] += 1
                    result["failures"].append((idx, error))
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total_items)
        else:
            # Parallel export
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {executor.submit(export_single_image, idx, path): idx
                                 for idx, path in items}

                for future in concurrent.futures.as_completed(future_to_idx):
                    try:
                        idx, success, error = future.result()
                    except concurrent.futures.CancelledError:
                        result["cancelled"] = True
                        continue
                    if success is None:
                        result["cancelled"] = True
                    elif success:
                        result["exported"] += 1
                    else:
                        result["failed"] += 1
                        result["failures"].append((idx, error))
                    completed_count += 1
                    if progress_callback:
                        progress_callback(completed_count, total_items)
                    if cancelled() and not result["cancelled"]:
                        result["cancelled"] = True
                        # Cancel anything not yet started; in-flight exports finish
                        for f in future_to_idx:
                            f.cancel()

        total_elapsed = time.time() - total_start_time
        print(f"Export finished in {total_elapsed:.2f}s: {result['exported']} exported, "
              f"{result['failed']} failed{', cancelled' if result['cancelled'] else ''}")
        return result
    
    def remove_image_by_index(self, idx: int):
        """
        Remove the image and its file path at the given index.
        """
        if idx is not None and 0 <= idx < len(self.images):
            del self.images[idx]
            self.file_paths = [img.file_path for img in self.images]

    def remove_images_by_indices(self, indices) -> int:
        """Remove several images at once. Returns how many were removed.

        Catalog semantics: a removed DUPLICATE also loses its catalog entry
        (a deliberately discarded copy must not resurrect on the next open),
        while a removed ACTUAL image keeps its stored edits — its latest
        state is preserved so later saves don't silently drop it."""
        removed_images = []
        for idx in sorted(set(indices), reverse=True):
            if 0 <= idx < len(self.images):
                removed_images.append(self.images[idx])
                del self.images[idx]
        self.file_paths = [img.file_path for img in self.images]
        if not removed_images:
            return 0
        try:
            from core.catalog import serialize_image, remove_duplicate_entries
            duplicate_removals = {}
            for img in removed_images:
                if getattr(img, "is_duplicate", False):
                    if img.display_name:
                        duplicate_removals.setdefault(img.file_path,
                                                      set()).add(img.display_name)
                else:
                    record = self._catalog_preserved.setdefault(
                        img.file_path,
                        {"signature": getattr(img, "_catalog_signature", None),
                         "entries": {}})
                    record["entries"][img.display_name] = serialize_image(img)
            if duplicate_removals:
                remove_duplicate_entries(duplicate_removals)
        except Exception as e:
            print(f"Catalog removal bookkeeping failed: {e}")
        return len(removed_images)

    def duplicate_images_by_indices(self, indices) -> int:
        """Insert a working copy right after each selected image. Copies are
        instant (no file re-read — they reuse the in-memory pixels) and carry
        the full edit state (conversion, crop, adjustments, orientation), so
        each copy can then be edited independently, e.g. to crop the same
        frame two different ways. Returns the number of copies made."""
        created = 0
        for idx in sorted({i for i in indices if 0 <= i < len(self.images)},
                          reverse=True):
            img = self.images[idx]
            if img.resized_raw is None:
                continue
            stem, ext = os.path.splitext(img.display_name
                                         or os.path.basename(img.file_path))
            existing = {im.display_name for im in self.images if im.display_name}
            n = 1
            while f"{stem}_copy{n}{ext}" in existing:
                n += 1
            dup = CCRImage(
                img.file_path,
                adjustment_settings=dict(img.adjustment_settings),
                rotation_angle=img.rotation_angle,
                fine_rotation_angle=img.fine_rotation_angle,
                horizontal_mirrored=img.horizontal_mirrored,
                vertical_mirrored=img.vertical_mirrored,
                converted=img.converted,
                source_ops=list(img.source_ops),
                preloaded_img=img.resized_raw.copy(),
                preloaded_full_size=img.original_full_size,
                display_name=f"{stem}_copy{n}{ext}",
                color_profile=img.color_profile,
                awb_enabled=img.awb_enabled,
            )
            dup.reference_frame = img.reference_frame
            dup.conversion_inputs = (dict(img.conversion_inputs)
                                     if img.conversion_inputs else None)
            dup.crop_rect = img.crop_rect
            dup.crop_angle = img.crop_angle
            dup.contrast_base = img.contrast_base
            dup.temperature_base = img.temperature_base
            dup.brightness_base = img.brightness_base
            dup.exposure_base = getattr(img, "exposure_base", 0.0)
            dup.tint_balance_factor = getattr(img, "tint_balance_factor", 1.0)
            dup._catalog_signature = getattr(img, "_catalog_signature", None)
            # Clone area layers (deep — each nests settings + geometry) so the
            # copy can diverge without aliasing the source. Areas are cloned on
            # duplicate (whole-image copy), not transferred cross-image.
            dup.area_layers = copy.deepcopy(img.area_layers)
            dup.is_duplicate = True
            # A duplicated slice becomes its OWN one-image round: a fresh
            # group means resetting the source's round never reaches into
            # this copy. Resetting the copy itself restores its parent
            # canvas (as a duplicate). Non-slice copies carry no lineage.
            if img.source_ops:
                dup.slice_group = _new_slice_group()
                parent_snap = dict(img.slice_parent) if img.slice_parent else {}
                parent_snap["is_duplicate"] = True
                dup.slice_parent = parent_snap
            else:
                dup.slice_group = None
                dup.slice_parent = None
            if ((dup.contrast_base, dup.temperature_base, dup.brightness_base)
                    != (0, 0, -8) or img.adjustment_settings.get("tint")
                    or dup.crop_rect is not None or dup.crop_angle
                    or any(a.get("enabled") for a in dup.area_layers)):
                # crop_rect/crop_angle were assigned just above, after the
                # constructor already built the histogram over the FULL image;
                # the histogram is crop-aware now, so a crop needs a regen too.
                dup.update_thumbnail_and_preview()
            self.images.insert(idx + 1, dup)
            created += 1
        self.file_paths = [im.file_path for im in self.images]
        return created

    def save_catalog(self):
        """Persist the edit state (conversion, slices, crop, adjustments) of
        all loaded images so reopening the files restores it. Cheap; called
        after significant operations and on app close."""
        if not self.images and not self._catalog_preserved:
            return
        try:
            from core.catalog import update_for_images
            update_for_images(self.images, preserved=self._catalog_preserved)
        except Exception as e:
            print(f"Catalog save failed: {e}")

    @staticmethod
    def _clean_slice_cuts(cuts) -> list:
        """Sorted cut fractions with 0/1 boundaries; drops cuts within 1% of
        an edge or of each other."""
        bounds = [0.0]
        for value in sorted(c for c in cuts if 0.01 <= c <= 0.99):
            if value - bounds[-1] >= 0.01:
                bounds.append(value)
        bounds.append(1.0)
        return bounds

    def slice_image_by_index(self, idx: int, x_cuts, y_cuts, progress_callback=None) -> int:
        """
        Split the image at idx into a grid of separate images along the given
        cut positions (fractions of the image's displayed frame; x_cuts are
        vertical lines, y_cuts horizontal). The slices replace the original
        in the list, in reading order (left-to-right, top-to-bottom).

        The parent's edits carry over: its fine rotation is BAKED into the
        slices (cuts are made on the rotated frame, exactly as displayed),
        its conversion is replayed on each slice with the parent's own
        constants so colors match, and adjustments/bases/orientation are
        inherited. Each slice's source_ops chain maps back to the ORIGINAL
        file, so zoom detail and full-res export read the correct region at
        full quality. The source is decoded only once.
        Returns the number of slices created (0 = nothing done).
        """
        img_obj = self.get_image_by_index(idx)
        if img_obj is None:
            return 0
        xs = self._clean_slice_cuts(x_cuts)
        ys = self._clean_slice_cuts(y_cuts)
        total = (len(xs) - 1) * (len(ys) - 1)
        if total <= 1:
            return 0

        # One shared decode (the parent's own slice chain is applied inside)
        full = img_obj.read_image(img_obj.file_path, preview=True)
        if full is None:
            return 0

        # Conversion replay: derive the parent's conversion constants so each
        # slice can be converted to look exactly like the parent did.
        parent_ci = img_obj.conversion_inputs if img_obj.converted else None
        norm_params = None
        if parent_ci is not None and parent_ci.get("mode") == "ref":
            from core.ccr_processor import compute_reference_norm_params
            ref_small = img_obj.resize_image_to_max_pixel(full, 1080)
            p_lo, p_hi, od = compute_reference_norm_params(
                ref_small, parent_ci["ref"], parent_ci["fine_rot"])
            norm_params = (tuple(float(v) for v in p_lo),
                           tuple(float(v) for v in p_hi),
                           tuple(float(v) for v in od))
        elif parent_ci is not None and parent_ci.get("mode") == "ref_params":
            norm_params = (parent_ci["p_lo"], parent_ci["p_hi"], parent_ci["od"])

        # Bake the parent's current fine rotation: the cuts were placed on
        # the rotated display, so the slices are cut from the rotated frame.
        baked_rotation = img_obj.fine_rotation_angle or 0
        if baked_rotation:
            h0, w0 = full.shape[:2]
            matrix = cv2.getRotationMatrix2D((w0 // 2, h0 // 2),
                                             -baked_rotation / 100.0, 1.0)
            full = cv2.warpAffine(full, matrix, (w0, h0), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        h, w = full.shape[:2]
        parent_full = img_obj.original_full_size or (h, w)
        # Nested slices must extend the PARENT's name (scan_s2 -> scan_s2_s1):
        # deriving from the file basename would make cousins collide and
        # exports could silently overwrite each other.
        stem, ext = os.path.splitext(img_obj.display_name
                                     or os.path.basename(img_obj.file_path))

        # One id for the whole round, plus a snapshot of the canvas being
        # sliced, so "Reset Slice" can collapse exactly these children back
        # into this parent — even after a catalog round trip, and without
        # confusing them with an independently-sliced duplicate of the file.
        slice_group = _new_slice_group()
        slice_parent = {
            "display_name": img_obj.display_name,
            "is_duplicate": bool(getattr(img_obj, "is_duplicate", False)),
            "slice_group": getattr(img_obj, "slice_group", None),
        }

        children = []
        if progress_callback:
            progress_callback(0, total)
        for yi in range(len(ys) - 1):
            for xi in range(len(xs) - 1):
                fx1, fx2 = xs[xi], xs[xi + 1]
                fy1, fy2 = ys[yi], ys[yi + 1]
                cx1 = max(0, min(w - 1, int(round(fx1 * w))))
                cy1 = max(0, min(h - 1, int(round(fy1 * h))))
                cx2 = max(cx1 + 1, min(w, int(round(fx2 * w))))
                cy2 = max(cy1 + 1, min(h, int(round(fy2 * h))))
                crop = full[cy1:cy2, cx1:cx2]
                child_full = (max(1, int(round((fy2 - fy1) * parent_full[0]))),
                              max(1, int(round((fx2 - fx1) * parent_full[1]))))

                # Replay the parent's conversion on this slice
                child_ci = None
                if norm_params is not None:
                    from core.ccr_processor import apply_reference_normalization
                    crop = apply_reference_normalization(crop, *norm_params)
                    child_ci = {"mode": "ref_params", "p_lo": norm_params[0],
                                "p_hi": norm_params[1], "od": norm_params[2]}
                elif parent_ci is not None and parent_ci.get("mode") == "bw":
                    from core.ccr_processor import apply_bwpoint_normalization
                    black_point, white_point = parent_ci["bw"]
                    crop = apply_bwpoint_normalization(crop, black_point, white_point)
                    child_ci = {"mode": "bw", "bw": parent_ci["bw"], "fine_rot": 0}

                index = len(children) + 1
                child = CCRImage(
                    img_obj.file_path,
                    adjustment_settings=dict(img_obj.adjustment_settings),
                    rotation_angle=img_obj.rotation_angle,
                    horizontal_mirrored=img_obj.horizontal_mirrored,
                    vertical_mirrored=img_obj.vertical_mirrored,
                    converted=child_ci is not None,
                    source_ops=img_obj.source_ops + [(baked_rotation, (fx1, fy1, fx2, fy2))],
                    preloaded_img=crop,
                    preloaded_full_size=child_full,
                    display_name=f"{stem}_s{index}{ext}",
                    slice_group=slice_group,
                    slice_parent=dict(slice_parent),
                    color_profile=img_obj.color_profile,
                    awb_enabled=img_obj.awb_enabled,
                )
                child.conversion_inputs = child_ci
                # Slices descend from the parent's loaded content — carry its
                # load-time file signature for the edit catalog.
                child._catalog_signature = getattr(img_obj, "_catalog_signature", None)
                # Inherit the parent's perceptual tint factor: the child's
                # ctor derived one from CONVERTED pixels, which would render
                # an inherited tint setting differently than the parent did.
                child.tint_balance_factor = img_obj.tint_balance_factor
                # Inherit the non-destructive base offsets; rebuild the
                # preview when the inherited state differs from what the
                # ctor already rendered with.
                child.contrast_base = img_obj.contrast_base
                child.temperature_base = img_obj.temperature_base
                child.brightness_base = img_obj.brightness_base
                child.exposure_base = getattr(img_obj, "exposure_base", 0.0)
                if ((child.contrast_base, child.temperature_base,
                        child.brightness_base) != (0, 0, -8)
                        or img_obj.adjustment_settings.get("tint")):
                    child.update_thumbnail_and_preview()
                children.append(child)
                if progress_callback:
                    progress_callback(len(children), total)

        self.images[idx:idx + 1] = children
        self.file_paths = [im.file_path for im in self.images]
        print(f"Sliced {os.path.basename(img_obj.file_path)} into {len(children)} images")
        return len(children)

    def reset_slice_by_indices(self, indices) -> Optional[int]:
        """Undo the slicing round each selected slice came from.

        For every selected sliced image, ALL slices produced by the same
        slicing operation (identified by a shared slice_group id, so an
        independently-sliced duplicate of the same file is never touched)
        are removed and the original canvas is restored in their place:
        re-decoded from the source file, with the selected slice's edits
        carried back — conversion replayed with the same constants,
        adjustments / orientation / bases inherited, and the fine rotation
        that slicing baked into the cuts restored as a live setting. The
        restored canvas re-joins its OWN round (if it was itself a slice),
        so a nested stack can be reset one level at a time. Returns the list
        index of the first restored parent, or None when nothing was reset.
        """
        selected = [self.images[i] for i in sorted(set(indices))
                    if 0 <= i < len(self.images)
                    and self.images[i].source_ops
                    and self.images[i].slice_group is not None]
        first_restored = None
        for template in selected:
            if not any(im is template for im in self.images):
                continue  # this round was already reset via a sibling
            restored_at = self._reset_one_slice_round(template)
            if restored_at is not None and first_restored is None:
                first_restored = restored_at
        if first_restored is not None:
            self.file_paths = [im.file_path for im in self.images]
            self.save_catalog()
        return first_restored

    def _reset_one_slice_round(self, template) -> Optional[int]:
        from core.ccr_processor import (apply_reference_normalization,
                                        apply_bwpoint_normalization,
                                        compute_reference_norm_params)
        group = template.slice_group
        parent_ops = list(template.source_ops[:-1])
        baked_rotation = template.source_ops[-1][0]
        # Membership is the shared round id — never geometry or names, both
        # of which collide across duplicated lineages. Collapsing a round
        # also collapses every round nested INSIDE its pieces: a nested
        # round's slice_parent points back (by group) to the member it was
        # cut from, so grow the set of groups to a fixpoint. A separately
        # sliced duplicate never links back in (its parent snapshot carries
        # a different group), so its lineage is left untouched.
        subtree = {group}
        changed = True
        while changed:
            changed = False
            for im in self.images:
                g = im.slice_group
                if g is not None and g not in subtree:
                    parent_g = (im.slice_parent or {}).get("slice_group")
                    if parent_g in subtree:
                        subtree.add(g)
                        changed = True
        members = [im for im in self.images
                   if im.slice_group is not None and im.slice_group in subtree]
        member_ids = {id(im) for im in members}
        insert_at = next(i for i, im in enumerate(self.images)
                         if id(im) in member_ids)

        # The canvas this round was cut from, captured at slice time.
        snapshot = template.slice_parent or {}
        parent_name = snapshot.get("display_name")
        parent_is_duplicate = bool(snapshot.get("is_duplicate", False))
        parent_group = snapshot.get("slice_group")
        # If the restored canvas was itself a slice, let it re-join its round
        # so it can be reset again — recover that round's parent snapshot
        # from a surviving sibling.
        parent_slice_parent = None
        if parent_group is not None:
            parent_slice_parent = next(
                (im.slice_parent for im in self.images
                 if im.slice_group == parent_group
                 and id(im) not in member_ids), None)

        # Conversion constants from the template slice, mirroring what
        # slicing does in the parent→child direction. "ref" coordinates
        # live in the slice's own frame, so derive transferable per-channel
        # params from the slice's unconverted pixels first.
        ci = template.conversion_inputs if template.converted else None
        norm_params = None
        bw_points = None
        if ci is not None and ci.get("mode") == "ref":
            child_full = template.read_image(template.file_path, preview=True)
            if child_full is not None:
                small = template.resize_image_to_max_pixel(child_full, 1080)
                p_lo, p_hi, od = compute_reference_norm_params(
                    small, ci["ref"], ci.get("fine_rot", 0))
                norm_params = (tuple(float(v) for v in p_lo),
                               tuple(float(v) for v in p_hi),
                               tuple(float(v) for v in od))
        elif ci is not None and ci.get("mode") == "ref_params":
            norm_params = (ci["p_lo"], ci["p_hi"], ci["od"])
        elif ci is not None and ci.get("mode") == "bw":
            bw_points = ci["bw"]

        # Re-decode the source canvas. The constructor raises if the file is
        # gone/unreadable; fail the reset gracefully and leave the list
        # untouched (nothing is removed until the decode succeeds).
        try:
            parent = CCRImage(
                template.file_path,
                adjustment_settings=dict(template.adjustment_settings),
                rotation_angle=template.rotation_angle,
                fine_rotation_angle=baked_rotation,
                horizontal_mirrored=template.horizontal_mirrored,
                vertical_mirrored=template.vertical_mirrored,
                converted=False,
                source_ops=parent_ops,
                display_name=parent_name,
                slice_group=parent_group,
                slice_parent=(dict(parent_slice_parent)
                              if parent_slice_parent else None),
                color_profile=template.color_profile,
                awb_enabled=template.awb_enabled,
            )
        except Exception as e:
            print(f"Reset slice failed: could not re-decode "
                  f"{template.file_path}: {e}")
            return None
        parent.is_duplicate = parent_is_duplicate
        if norm_params is not None:
            parent.resized_raw = apply_reference_normalization(
                parent.resized_raw, *norm_params)
            parent.converted = True
            parent.conversion_inputs = {
                "mode": "ref_params", "p_lo": norm_params[0],
                "p_hi": norm_params[1], "od": norm_params[2]}
        elif bw_points is not None:
            parent.resized_raw = apply_bwpoint_normalization(
                parent.resized_raw, *bw_points)
            parent.converted = True
            parent.conversion_inputs = {"mode": "bw", "bw": bw_points,
                                        "fine_rot": 0}
        parent.tint_balance_factor = template.tint_balance_factor
        parent.contrast_base = template.contrast_base
        parent.temperature_base = template.temperature_base
        parent.brightness_base = template.brightness_base
        parent.exposure_base = getattr(template, "exposure_base", 0.0)
        parent._catalog_signature = getattr(template, "_catalog_signature", None)
        parent.update_thumbnail_and_preview()

        self.images = [im for im in self.images if id(im) not in member_ids]
        self.images.insert(insert_at, parent)
        # Drop preserved catalog entries for members that were earlier
        # removed (as actuals): otherwise update_for_images would merge those
        # stale slices back in and resurrect them on the next open.
        self._purge_preserved_slice_round(template.file_path, subtree)
        print(f"Reset slice: {len(members)} slice(s) of "
              f"{os.path.basename(template.file_path)} replaced by "
              f"{parent_name or os.path.basename(template.file_path)}")
        return insert_at

    def _purge_preserved_slice_round(self, file_path, groups) -> None:
        record = self._catalog_preserved.get(file_path)
        if not record:
            return
        entries = record.get("entries", {})
        stale = [name for name, state in entries.items()
                 if state.get("slice_group") in groups]
        for name in stale:
            del entries[name]
        if not entries:
            self._catalog_preserved.pop(file_path, None)

    def set_white_point(self, bgr_tuple):
        self.white_point_bgr = bgr_tuple

    def clear_white_point(self):
        """Drop the sampled white point so B/W-point conversion falls back to the
        calibrated default slope (black point only). See
        spec/optional-white-point-default-slope.md."""
        self.white_point_bgr = None

    def clear_black_point(self):
        """Drop the sampled black point (film base). With no black point the
        B/W-point section needs a fresh Set Black Point before converting."""
        self.black_point_bgr = None

    def set_black_point(self, bgr_tuple):
        self.black_point_bgr = bgr_tuple

    def apply_bwpoint_to_all_images(self, progress_callback=None):
        """
        Apply B/W point film negative conversion to all loaded images using the same
        pipeline as ccr_normalize_with_reference, but with user-specified B/W points
        instead of auto-detected percentiles.  Always converts from original scan data.
        """
        if self.black_point_bgr is None:
            raise ValueError("A black point (film base) must be set before applying.")
        # Calibration: when both points are set, log the linear + density slopes
        # they imply so a fixed default slope can be measured and baked in. With
        # only a black point, conversion uses the baked default slope.
        if self.white_point_bgr is not None:
            log_bwpoint_slopes(self.black_point_bgr, self.white_point_bgr)
        total = len(self.images)
        if progress_callback:
            progress_callback(0, total)
        for i, img in enumerate(self.images):
            try:
                # Ensure we start from original (unprocessed) scan data
                if img.converted:
                    img.reload_image()
                processed = ccr_normalize_with_bwpoint(
                    img, self.black_point_bgr, self.white_point_bgr
                )
                if processed is not None:
                    img.resized_raw = processed
                img.converted = True
                img.conversion_inputs = {
                    "mode": "bw",
                    "bw": (tuple(self.black_point_bgr),
                           tuple(self.white_point_bgr) if self.white_point_bgr is not None else None),
                    "fine_rot": img.fine_rotation_angle,
                }
                img.update_thumbnail_and_preview()
            except Exception as e:
                print(f"B/W point conversion failed for image {i}: {e}")
            if progress_callback:
                progress_callback(i + 1, total)

# Singleton instance
ccr_backend = CCRBackend()
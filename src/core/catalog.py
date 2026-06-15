"""
Persistent per-file edit catalog.

Remembers how each processed file was converted, sliced, cropped and
adjusted, and restores that state when the file is opened again — including
re-running the conversion with the recorded inputs so the preview comes back
exactly as the user left it. Stored as JSON in the user's app-data folder;
entries are validated against the file's size/mtime so edits recorded for a
since-modified file are ignored rather than misapplied.
"""
import json
import logging
import os
import tempfile
import time
import copy
import uuid

CATALOG_VERSION = 1
MAX_CATALOG_ENTRIES = 2000  # bounds growth; oldest records pruned beyond this


def default_catalog_path() -> str:
    base = os.environ.get("APPDATA") or os.path.join(os.path.expanduser("~"), ".config")
    folder = os.path.join(base, "FreeCCR")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "catalog.json")


def load_catalog(path: str = None) -> dict:
    path = path or default_catalog_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Structural validation too: a parseable-but-malformed catalog must
        # degrade to a fresh one, never block image loading.
        if (isinstance(data, dict) and data.get("version") == CATALOG_VERSION
                and isinstance(data.get("files"), dict)):
            return data
    except FileNotFoundError:
        pass
    except Exception as e:
        logging.warning(f"Could not read catalog {path}: {e}")
    return {"version": CATALOG_VERSION, "files": {}}


def save_catalog(catalog: dict, path: str = None) -> None:
    path = path or default_catalog_path()
    try:
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(catalog, f)
        os.replace(tmp, path)  # atomic: a crash can't corrupt the catalog
    except Exception as e:
        logging.warning(f"Could not write catalog {path}: {e}")


def _file_key(file_path: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(file_path)))


def _file_signature(file_path: str) -> dict:
    st = os.stat(file_path)
    return {"size": st.st_size, "mtime": st.st_mtime}


def _ci_to_json(ci):
    if ci is None:
        return None
    out = dict(ci)
    if out.get("ref") is not None:
        out["ref"] = list(out["ref"])
    if out.get("bw") is not None:
        out["bw"] = [list(out["bw"][0]), list(out["bw"][1])]
    for key in ("p_lo", "p_hi", "od"):
        if out.get(key) is not None:
            out[key] = list(out[key])
    return out


def _ci_from_json(ci):
    if not ci:
        return None
    out = dict(ci)
    if out.get("ref") is not None:
        out["ref"] = tuple(out["ref"])
    if out.get("bw") is not None:
        out["bw"] = (tuple(out["bw"][0]), tuple(out["bw"][1]))
    for key in ("p_lo", "p_hi", "od"):
        if out.get(key) is not None:
            out[key] = tuple(out[key])
    return out


def _slice_parent_to_json(parent):
    """slice_parent is a plain {display_name, is_duplicate, slice_group} dict
    of JSON-safe scalars, but copy defensively so the stored catalog never
    aliases live state."""
    return dict(parent) if parent else None


def _areas_to_json(areas):
    """Serialize the per-image area-editing layers. Geometry is normalized
    floats and each area's settings reuses the adjustment-dict shape (all
    JSON-safe), but copy defensively and coerce types so the stored catalog
    never aliases live state."""
    out = []
    for a in (areas or []):
        out.append({
            "id": a.get("id"),
            "kind": a.get("kind", "circle"),
            "enabled": bool(a.get("enabled", True)),
            "feather": float(a.get("feather", 0.25)),
            "angle": float(a.get("angle", 0.0)),
            "geometry": {k: float(v)
                         for k, v in (a.get("geometry") or {}).items()},
            "settings": copy.deepcopy(a.get("settings") or {}),
        })
    return out


def _areas_from_json(areas):
    """Inverse of _areas_to_json; defensive against legacy/partial records."""
    out = []
    for a in (areas or []):
        if not isinstance(a, dict):
            continue
        out.append({
            "id": a.get("id") or uuid.uuid4().hex,
            "kind": a.get("kind", "circle"),
            "enabled": bool(a.get("enabled", True)),
            "feather": float(a.get("feather", 0.25)),
            "angle": float(a.get("angle", 0.0)),
            "geometry": {k: float(v)
                         for k, v in (a.get("geometry") or {}).items()},
            "settings": dict(a.get("settings") or {}),
        })
    return out


def serialize_image(img) -> dict:
    """Everything needed to bring a CCRImage back to its current state."""
    return {
        "display_name": img.display_name,
        "is_duplicate": bool(getattr(img, "is_duplicate", False)),
        "slice_group": getattr(img, "slice_group", None),
        "slice_parent": _slice_parent_to_json(getattr(img, "slice_parent", None)),
        "source_ops": [[int(rot), list(region)] for rot, region in img.source_ops],
        "converted": bool(img.converted),
        "conversion_inputs": _ci_to_json(img.conversion_inputs),
        "adjustment_settings": dict(img.adjustment_settings),
        "areas": _areas_to_json(getattr(img, "area_layers", None)),
        "color_profile": getattr(img, "color_profile", "color"),
        "crop_rect": list(img.crop_rect) if img.crop_rect else None,
        "crop_angle": float(img.crop_angle or 0.0),
        "rotation_angle": int(img.rotation_angle),
        "fine_rotation_angle": int(img.fine_rotation_angle),
        "horizontal_mirrored": bool(img.horizontal_mirrored),
        "vertical_mirrored": bool(img.vertical_mirrored),
        "contrast_base": int(img.contrast_base),
        "temperature_base": int(img.temperature_base),
        "brightness_base": int(img.brightness_base),
        "tint_balance_factor": float(getattr(img, "tint_balance_factor", 1.0)),
        "reference_frame": list(img.reference_frame) if img.reference_frame else None,
        "dust_heal_strokes": [dict(s) for s in
                              (getattr(img, "dust_heal_strokes", None) or [])],
    }


def _is_pristine(state: dict) -> bool:
    """True when a serialized state carries no user edits worth saving."""
    return (not state["converted"] and not state["source_ops"]
            and not state["adjustment_settings"]
            and not state.get("areas")
            and state.get("color_profile", "color") == "color"
            and state["crop_rect"] is None
            and state["rotation_angle"] == 0 and state["fine_rotation_angle"] == 0
            and not state["horizontal_mirrored"] and not state["vertical_mirrored"]
            and state["reference_frame"] is None
            and not state.get("dust_heal_strokes"))


def _prune(catalog: dict) -> None:
    files = catalog["files"]
    if len(files) <= MAX_CATALOG_ENTRIES:
        return
    oldest_first = sorted(files.items(), key=lambda kv: kv[1].get("saved_at", 0))
    for fkey, _record in oldest_first[:len(files) - MAX_CATALOG_ENTRIES]:
        del files[fkey]


def update_for_images(images, path: str = None, preserved: dict = None) -> None:
    """Write the current state of all loaded images into the catalog,
    grouped by source file (the slices of one file form one entry list, in
    list order). Entries for files not currently loaded are kept.

    preserved: states of ACTUAL images removed from the list this session,
    {file_path: {"signature": sig, "entries": {display_name: state}}} —
    removal must not lose their stored edits, so they are merged back into
    the records alongside the loaded images."""
    catalog = load_catalog(path)
    grouped = {}
    for img in images:
        grouped.setdefault(_file_key(img.file_path), []).append(img)
    preserved_by_key = {}
    for file_path, record in (preserved or {}).items():
        if record.get("entries"):
            preserved_by_key[_file_key(file_path)] = record
    now = time.time()
    for fkey, imgs in grouped.items():
        states = [serialize_image(im) for im in imgs]
        # A failed restore fell back to a plain load — never let that
        # OVERWRITE the stored record (which still holds the real slices and
        # edits) unless the user has since made real edits worth saving.
        if (any(getattr(im, "_catalog_restore_failed", False) for im in imgs)
                and all(_is_pristine(s) for s in states)):
            preserved_by_key.pop(fkey, None)
            continue
        # Merge in the states of this file's removed actual images
        kept_record = preserved_by_key.pop(fkey, None)
        if kept_record:
            loaded_names = {s.get("display_name") for s in states}
            states += [state for name, state in kept_record["entries"].items()
                       if name not in loaded_names]
        # Prefer the signature captured when the file was actually READ:
        # edits belong to the content as loaded, not as it is at save time.
        signature = next((getattr(im, "_catalog_signature", None) for im in imgs
                          if getattr(im, "_catalog_signature", None)), None)
        if signature is None:
            try:
                signature = _file_signature(imgs[0].file_path)
            except OSError:
                continue
        catalog["files"][fkey] = {
            "signature": signature,
            "saved_at": now,
            "images": states,
        }
    # Files whose images were ALL removed (as actuals): merge the preserved
    # states into the existing record so their edits survive.
    for fkey, kept_record in preserved_by_key.items():
        entries = kept_record["entries"]
        existing = catalog["files"].get(fkey)
        if isinstance(existing, dict) and existing.get("images"):
            removed_names = set(entries)
            merged = [s for s in existing["images"]
                      if s.get("display_name") not in removed_names]
            merged += list(entries.values())
            existing["images"] = merged
            existing["saved_at"] = now
        elif kept_record.get("signature"):
            catalog["files"][fkey] = {
                "signature": kept_record["signature"],
                "saved_at": now,
                "images": list(entries.values()),
            }
    _prune(catalog)
    save_catalog(catalog, path)


def remove_duplicate_entries(removals: dict, path: str = None) -> None:
    """Delete the catalog entries of removed DUPLICATES so a deliberately
    discarded copy does not resurrect on the next open. Entries of actual
    images are never touched here.

    removals: {file_path: set of duplicate display_names removed}"""
    catalog = load_catalog(path)
    changed = False
    for file_path, names in removals.items():
        fkey = _file_key(file_path)
        record = catalog["files"].get(fkey)
        if not isinstance(record, dict):
            continue
        entries = record.get("images") or []
        kept = [state for state in entries
                if not (state.get("is_duplicate")
                        and state.get("display_name") in names)]
        if len(kept) != len(entries):
            if kept:
                record["images"] = kept
            else:
                del catalog["files"][fkey]
            changed = True
    if changed:
        save_catalog(catalog, path)


def entries_for_path(file_path: str, path: str = None, catalog_data: dict = None):
    """Catalog entries for a file, or None when absent or when the file has
    changed since it was cataloged (stale edits must not be misapplied).
    Pass a preloaded catalog_data to avoid re-reading the catalog per file
    (batch loads)."""
    catalog = catalog_data if catalog_data is not None else load_catalog(path)
    record = catalog["files"].get(_file_key(file_path))
    if not isinstance(record, dict):
        return None
    if not record:
        return None
    try:
        signature = _file_signature(file_path)
    except OSError:
        return None
    stored = record.get("signature") or {}
    if (stored.get("size") != signature["size"]
            or abs(stored.get("mtime", 0) - signature["mtime"]) > 1.0):
        return None
    return record.get("images") or None


def create_images_for_path(file_path: str, path: str = None,
                           catalog_data: dict = None) -> list:
    """Create the CCRImage(s) for a file, restoring cataloged state when
    available (slices, conversion, adjustments, crop, orientation).

    Restore is ALL-OR-NOTHING per file: a partial restore would silently
    drop a photo from the session, and the next catalog save would then
    permanently erase its record. On any entry failure the whole file falls
    back to a plain load, flagged so the save path preserves the stored
    record until the user makes new edits."""
    from core.ccr_image import CCRImage
    signature = None
    try:
        signature = _file_signature(file_path)
        entries = entries_for_path(file_path, path, catalog_data)
    except Exception as e:
        logging.warning(f"Catalog lookup failed for {file_path}: {e}")
        entries = None

    def _plain(restore_failed=False):
        img = CCRImage(file_path)
        img._catalog_signature = signature
        if restore_failed:
            img._catalog_restore_failed = True
        return [img]

    if not entries:
        return _plain()
    images = []
    for state in entries:
        try:
            images.append(_restore_image(file_path, state))
        except Exception as e:
            logging.warning(f"Catalog restore failed for {file_path}: {e}")
            return _plain(restore_failed=True)
    for img in images:
        img._catalog_signature = signature
    return images


def _restore_image(file_path: str, state: dict):
    from core.ccr_image import CCRImage
    source_ops = [(int(rot), tuple(region))
                  for rot, region in (state.get("source_ops") or [])]
    img = CCRImage(
        file_path,
        adjustment_settings=dict(state.get("adjustment_settings") or {}),
        rotation_angle=state.get("rotation_angle", 0),
        fine_rotation_angle=state.get("fine_rotation_angle", 0),
        horizontal_mirrored=state.get("horizontal_mirrored", False),
        vertical_mirrored=state.get("vertical_mirrored", False),
        source_ops=source_ops,
        display_name=state.get("display_name"),
        slice_group=state.get("slice_group"),
        slice_parent=(dict(state["slice_parent"])
                      if state.get("slice_parent") else None),
        areas=_areas_from_json(state.get("areas")),
    )
    img.is_duplicate = bool(state.get("is_duplicate", False))
    img.color_profile = state.get("color_profile", "color")
    ref = state.get("reference_frame")
    img.reference_frame = tuple(ref) if ref else None
    crop = state.get("crop_rect")
    img.crop_rect = tuple(crop) if crop else None
    img.crop_angle = state.get("crop_angle", 0.0) or 0.0
    from core.dust_removal import clean_strokes
    img.dust_heal_strokes = clean_strokes(state.get("dust_heal_strokes"))
    img.tint_balance_factor = state.get("tint_balance_factor",
                                        getattr(img, "tint_balance_factor", 1.0))

    ci = _ci_from_json(state.get("conversion_inputs"))
    if state.get("converted") and ci is not None:
        _replay_conversion(img, ci)

    # Bases AFTER the replay (the bw pipeline writes its own defaults)
    img.contrast_base = state.get("contrast_base", img.contrast_base)
    img.temperature_base = state.get("temperature_base", img.temperature_base)
    img.brightness_base = state.get("brightness_base", img.brightness_base)
    img.update_thumbnail_and_preview()
    return img


def _replay_conversion(img, ci) -> None:
    """Re-run the recorded conversion so the preview comes back exactly as
    it was. Uses the inputs captured at convert time, not live state."""
    from core.ccr_processor import (ccr_normalize_with_reference,
                                    ccr_normalize_with_bwpoint,
                                    apply_reference_normalization)
    mode = ci.get("mode")
    if mode == "ref":
        saved_fine = img.fine_rotation_angle
        saved_ref = img.reference_frame
        img.fine_rotation_angle = ci.get("fine_rot", 0)
        img.reference_frame = ci["ref"]
        try:
            processed = ccr_normalize_with_reference(img)
        finally:
            img.fine_rotation_angle = saved_fine
            # Unconditionally: the user may have deleted the frame after
            # converting, and a restore must reproduce exactly that state.
            img.reference_frame = saved_ref
        img.resized_raw = processed
    elif mode == "ref_params":
        img.resized_raw = apply_reference_normalization(
            img.resized_raw, ci["p_lo"], ci["p_hi"], ci["od"])
    elif mode == "bw":
        saved_fine = img.fine_rotation_angle
        img.fine_rotation_angle = ci.get("fine_rot", 0)
        try:
            black_point, white_point = ci["bw"]
            processed = ccr_normalize_with_bwpoint(img, black_point, white_point)
        finally:
            img.fine_rotation_angle = saved_fine
        img.resized_raw = processed
    else:
        return
    img.converted = True
    img.conversion_inputs = ci

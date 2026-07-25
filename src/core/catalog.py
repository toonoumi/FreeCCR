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

# Bump whenever stored fields change meaning so legacy catalogs are discarded
# (load_catalog returns a fresh one on a version mismatch) rather than replayed
# with stale semantics. v2: the unprofiled negative RAW decode changed (the
# no-ICC default is now Adobe RGB + rawpy auto-scale instead of raw sensor
# primaries + uniform white-level scaling), so v1 conversion anchors (ref_params
# p_lo/p_hi/od and bw black/white points) were computed against a different
# decode and would recolour slices / B&W-point conversions if replayed.
CATALOG_VERSION = 2
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


# --- 3-way merge (trichrome) composite keys ------------------------------
# A merged image has no single backing file — it is synthesized from three
# source RAWs (red, green, blue). Its edits are cataloged under a COMPOSITE key
# derived from all three sources, validated against a COMPOSITE signature (each
# source's size+mtime). The "merge:" prefix keeps it disjoint from every
# single-file key (a real path never starts with it), so a merge and a plain
# open of the red RAW never share a record. See spec/merge-catalog-persistence.md.
MERGE_KEY_PREFIX = "merge:"


def _is_merge_key(key: str) -> bool:
    return isinstance(key, str) and key.startswith(MERGE_KEY_PREFIX)


def _merge_key(sources) -> str:
    """Composite catalog key for a merge: the ordered (R, G, B) source file keys
    joined under the merge prefix. Order matters; the merge always sorts inputs
    by filename, so the same three files always yield the same key."""
    return MERGE_KEY_PREFIX + "|".join(_file_key(p) for p in sources)


def _catalog_key_for_image(img) -> str:
    """The catalog key this image's edits belong under: the composite merge key
    for a trichrome-merged image (never a single source file's key, which a
    later plain open would then read as its own edits), else the file key."""
    if getattr(img, "is_merged", False) and getattr(img, "merge_sources", None):
        return _merge_key(img.merge_sources)
    return _file_key(img.file_path)


def _merge_signature(sources) -> dict:
    """Composite freshness signature: each source's size+mtime in (R, G, B)
    order. A change to ANY source invalidates the cataloged edits — they were
    computed against those exact three frames. Raises OSError if a source is
    unreadable (callers treat that as 'no valid entry')."""
    return {"sources": [_file_signature(p) for p in sources]}


def _ci_to_json(ci):
    if ci is None:
        return None
    out = dict(ci)
    if out.get("ref") is not None:
        out["ref"] = list(out["ref"])
    if out.get("bw") is not None:
        black, white = out["bw"]
        out["bw"] = [list(black), (list(white) if white is not None else None)]
    for key in ("p_lo", "p_hi", "od", "slopes"):
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
        black, white = out["bw"]
        out["bw"] = (tuple(black), (tuple(white) if white is not None else None))
    for key in ("p_lo", "p_hi", "od", "slopes"):
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
    is_merged = bool(getattr(img, "is_merged", False))
    return {
        "display_name": img.display_name,
        "is_duplicate": bool(getattr(img, "is_duplicate", False)),
        # 3-way merge (trichrome): a merged image restores by re-merging its
        # three sources, so its identity travels with the state. Harmless for a
        # normal image (is_merged False, merge_sources None). merge_demosaic is
        # a restore fallback — a re-import prefers the current global setting.
        "is_merged": is_merged,
        "merge_sources": (list(img.merge_sources)
                          if is_merged and getattr(img, "merge_sources", None)
                          else None),
        "merge_demosaic": bool(getattr(img, "merge_demosaic", True)),
        "slice_group": getattr(img, "slice_group", None),
        "slice_parent": _slice_parent_to_json(getattr(img, "slice_parent", None)),
        "source_ops": [[int(rot), list(region)] for rot, region in img.source_ops],
        "converted": bool(img.converted),
        "conversion_inputs": _ci_to_json(img.conversion_inputs),
        "adjustment_settings": dict(img.adjustment_settings),
        "areas": _areas_to_json(getattr(img, "area_layers", None)),
        "dust_spots": copy.deepcopy(getattr(img, "dust_spots", None) or []),
        # The heal plan (each segment's chosen source patch + verdicts):
        # once a source is set, nothing may move it — not even a reload, so
        # the plan persists next to the spots and reseeds the plan cache on
        # restore (a from-scratch replan of an incrementally-built plan can
        # pick different sources than the user saw). None when the cache is
        # stale or there are no spots.
        "dust_plan": _dust_plan_to_json(img),
        # Feather as a fraction of each hole's own radius. Written under a NEW
        # key: the legacy "dust_feather" key held a fraction of image width
        # (0..0.01) and is only ever read back (migrated), never written.
        "dust_feather_r": float(getattr(img, "dust_feather", 0.25)),
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
        "exposure_base": float(getattr(img, "exposure_base", 0.0)),
        "tint_balance_factor": float(getattr(img, "tint_balance_factor", 1.0)),
        "reference_frame": list(img.reference_frame) if img.reference_frame else None,
    }


def _dust_plan_to_json(img):
    """The image's cached heal plan as JSON-safe lists, or None when it does
    not match the current spots (never persist a stale plan). Records mirror
    _heal_impl's: [cy, cx, [oy, ox] | None, genuine, dlike_on, manual]."""
    cached = getattr(img, "_dust_plan_cache", None)
    spots = getattr(img, "dust_spots", None) or []
    if not cached or not spots or cached[0] != repr(spots):
        return None
    out = []
    for r in cached[1]:
        off = None if r[2] is None else [float(r[2][0]), float(r[2][1])]
        out.append([float(r[0]), float(r[1]), off] + list(r[3:6]))
    return out


def _dust_plan_from_json(plan):
    """Inverse of _dust_plan_to_json (tuples, offset as a tuple)."""
    return [(r[0], r[1], (tuple(r[2]) if r[2] is not None else None),
             *r[3:6]) for r in plan]


def _is_pristine(state: dict) -> bool:
    """True when a serialized state carries no user edits worth saving."""
    return (not state["converted"] and not state["source_ops"]
            and not state["adjustment_settings"]
            and not state.get("areas")
            and not state.get("dust_spots")
            and state.get("color_profile", "color") == "color"
            and state["crop_rect"] is None
            and state["rotation_angle"] == 0 and state["fine_rotation_angle"] == 0
            and not state["horizontal_mirrored"] and not state["vertical_mirrored"]
            and state["reference_frame"] is None)


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
    {catalog_key: {"signature": sig, "entries": {display_name: state}}} —
    removal must not lose their stored edits, so they are merged back into
    the records alongside the loaded images. Keys are FINAL catalog keys
    (a file key or a "merge:" composite key), used as-is here."""
    catalog = load_catalog(path)
    grouped = {}
    for img in images:
        # A trichrome-merged image is keyed by the composite of its three
        # sources, NOT by any one source RAW's file key (which a later plain
        # open would misread as its own edits). A merged image that somehow
        # lost its sources can't be keyed safely, so skip it rather than
        # corrupt a file record. See spec/merge-catalog-persistence.md.
        if getattr(img, "is_merged", False) and not getattr(img, "merge_sources", None):
            continue
        grouped.setdefault(_catalog_key_for_image(img), []).append(img)
    preserved_by_key = {}
    for key, record in (preserved or {}).items():
        if record.get("entries"):
            preserved_by_key[key] = record
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
                signature = (_merge_signature(imgs[0].merge_sources)
                             if _is_merge_key(fkey)
                             else _file_signature(imgs[0].file_path))
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

    removals: {catalog_key: set of duplicate display_names removed}, where the
    key is a raw source path (normalized here) or a "merge:" composite key
    (used as-is)."""
    catalog = load_catalog(path)
    changed = False
    for key, names in removals.items():
        fkey = key if _is_merge_key(key) else _file_key(key)
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


def entries_for_merge(sources, path: str = None, catalog_data: dict = None):
    """Catalog entries for a trichrome merge of `sources` (R, G, B order), or
    None when absent or when ANY source changed since it was cataloged (stale
    edits must not be misapplied). The merge analogue of entries_for_path:
    validates the composite signature source-by-source."""
    catalog = catalog_data if catalog_data is not None else load_catalog(path)
    record = catalog["files"].get(_merge_key(sources))
    if not isinstance(record, dict) or not record:
        return None
    try:
        current = _merge_signature(sources)["sources"]
    except OSError:
        return None
    stored = (record.get("signature") or {}).get("sources") or []
    if len(stored) != len(current):
        return None
    for s, c in zip(stored, current):
        if (s.get("size") != c["size"]
                or abs(s.get("mtime", 0) - c["mtime"]) > 1.0):
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


def create_images_for_merge(sources, merge_demosaic: bool = True,
                            path: str = None, catalog_data: dict = None) -> list:
    """Create the merged CCRImage(s) for a trichrome triplet, restoring
    cataloged state (conversion, adjustments, crop, dust, slices, duplicates)
    when the same three frames were merged and edited before.

    All-or-nothing per triplet, like create_images_for_path: any entry failure
    falls back to a fresh merge, flagged so the next save preserves the stored
    record. `sources` are the live (R, G, B) paths of THIS import (they equal
    the stored ones because the composite key matched); `merge_demosaic` is the
    current global setting the fresh/re-merge decode uses."""
    from core.ccr_image import CCRImage
    signature = None
    try:
        signature = _merge_signature(sources)
        entries = entries_for_merge(sources, path, catalog_data)
    except Exception as e:
        logging.warning(f"Merge catalog lookup failed for {list(sources)}: {e}")
        entries = None

    def _plain(restore_failed=False):
        img = CCRImage(sources[0], is_merged=True, merge_sources=list(sources),
                       merge_demosaic=merge_demosaic)
        img._catalog_signature = signature
        if restore_failed:
            img._catalog_restore_failed = True
        return [img]

    if not entries:
        return _plain()
    images = []
    for state in entries:
        try:
            images.append(_restore_image(sources[0], state,
                                         live_merge_sources=list(sources),
                                         live_merge_demosaic=merge_demosaic))
        except Exception as e:
            logging.warning(f"Merge catalog restore failed for "
                            f"{list(sources)}: {e}")
            return _plain(restore_failed=True)
    for img in images:
        img._catalog_signature = signature
    return images


def _restore_image(file_path: str, state: dict, live_merge_sources=None,
                   live_merge_demosaic=None):
    from core.ccr_image import CCRImage
    source_ops = [(int(rot), tuple(region))
                  for rot, region in (state.get("source_ops") or [])]
    # Rebuild a trichrome-merged image when the stored state is one: every
    # re-read (preview/zoom/export) then re-merges the three sources and applies
    # source_ops. Prefer the live sources/demosaic (current import + current
    # global setting) over the stored copies. See spec/merge-catalog-persistence.md.
    is_merged = bool(state.get("is_merged", False))
    if is_merged:
        merge_sources = (list(live_merge_sources) if live_merge_sources
                         else (state.get("merge_sources") or None))
        merge_demosaic = (live_merge_demosaic if live_merge_demosaic is not None
                          else bool(state.get("merge_demosaic", True)))
    else:
        merge_sources = None
        merge_demosaic = True
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
        is_merged=is_merged,
        merge_sources=merge_sources,
        merge_demosaic=merge_demosaic,
    )
    img.is_duplicate = bool(state.get("is_duplicate", False))
    img.dust_spots = copy.deepcopy(state.get("dust_spots") or [])
    plan = state.get("dust_plan")
    if plan and img.dust_spots:
        # Reseed the plan cache so the restored image keeps the exact
        # sources the user saw (sticky across sessions); a missing/legacy
        # plan just replans once and is sticky from then on. The snapshot
        # ties the plan to THESE spots — records of spots later deleted or
        # replaced are pruned instead of rebinding to new spots.
        img._dust_plan_cache = (repr(img.dust_spots),
                                _dust_plan_from_json(plan))
        img._dust_plan_spots = list(img.dust_spots)
    if "dust_feather_r" in state:
        img.dust_feather = float(state["dust_feather_r"])
    elif "dust_feather" in state:
        # Legacy width-fraction feather (slider 0..100 -> 0..1% of image
        # width): map the old slider position onto today's radius-fraction
        # scale (0..100 -> 0..100% of radius) so the user's relative setting
        # survives. Neither key present -> CCRImage's default stands.
        img.dust_feather = min(1.0, float(state["dust_feather"]) * 100.0)
    img.color_profile = state.get("color_profile", "color")
    ref = state.get("reference_frame")
    img.reference_frame = tuple(ref) if ref else None
    crop = state.get("crop_rect")
    img.crop_rect = tuple(crop) if crop else None
    img.crop_angle = state.get("crop_angle", 0.0) or 0.0
    img.tint_balance_factor = state.get("tint_balance_factor",
                                        getattr(img, "tint_balance_factor", 1.0))

    # Positive mode never replays a stored negative conversion onto the
    # (positive-decoded) scan — the image loads as an editable positive with its
    # stored adjustments. See spec/positive-mode.md.
    positive = False
    try:
        from core.ccr_backend import ccr_backend
        positive = bool(ccr_backend.positive_mode)
    except Exception:
        positive = False
    ci = _ci_from_json(state.get("conversion_inputs"))
    if state.get("converted") and ci is not None and not positive:
        _replay_conversion(img, ci)

    # Bases AFTER the replay (the bw pipeline writes its own defaults)
    img.contrast_base = state.get("contrast_base", img.contrast_base)
    img.temperature_base = state.get("temperature_base", img.temperature_base)
    img.brightness_base = state.get("brightness_base", img.brightness_base)
    img.exposure_base = state.get("exposure_base", getattr(img, "exposure_base", 0.0))
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
        # Snapshot passed as parameters — the live reference_frame /
        # fine_rotation_angle stay whatever the stored state restored them
        # to (the user may have deleted the frame after converting).
        img.resized_raw = ccr_normalize_with_reference(
            img, reference_rect=ci["ref"], fine_rot=ci.get("fine_rot", 0))
    elif mode == "ref_params":
        img.resized_raw = apply_reference_normalization(
            img.resized_raw, ci["p_lo"], ci["p_hi"], ci["od"])
        img._ws_windowed = False   # reference path is full-range, not windowed
    elif mode == "bw":
        saved_fine = img.fine_rotation_angle
        img.fine_rotation_angle = ci.get("fine_rot", 0)
        try:
            black_point, white_point = ci["bw"]
            processed = ccr_normalize_with_bwpoint(img, black_point, white_point,
                                                   density=ci.get("density", False),
                                                   slopes_bgr=ci.get("slopes"))
        finally:
            img.fine_rotation_angle = saved_fine
        img.resized_raw = processed
    else:
        return
    img.converted = True
    img.conversion_inputs = ci

# Trichrome Linear TIFF Export

> Status: refined (round 2 folded inline — see "Review resolutions" at the end;
> round 3 added crop support on request).

## Summary

In trichrome (3-way RGB merge) sessions, let the user export a merged image as a
**16-bit linear TIFF of the raw channel combination** — the exact array
`merge_raw_channels` produces (each frame's own channel, black-subtracted,
white-level-scaled, stacked) — plus **framing only**: the slice chain (for
sliced images) and the user's crop. No value processing of any kind: no negative
conversion, no adjustments, no orientation, no colour-space re-encode, no
resize, no ICC burn-in. Conversion is **not** required to export. The option
exists only for merged (trichrome) images.

Use case: archival / interchange — take the crosstalk-free trichrome combination
into another tool (darktable, Photoshop, a scientific pipeline) before FreeCCR's
inversion look is applied.

## Goals

- A new format choice in the existing Export dialog, present **only when merged
  images are loaded**: `TIFF 16-bit linear merge (unprocessed)`.
- Exportable set for this format = **all merged images**, converted or not
  (conversion status is irrelevant to the linear merge).
- Pixel contract: output file == `merge_raw_channels(merge_sources)` at full
  canonical resolution, byte-for-byte — `(H, W, 3)` uint16, linear,
  camera-native RGB, untagged (no ICC profile embedded).
- Reuse the whole existing export shell: scope radios, destination, naming
  macros + conflict policy, parallel worker, progress + cancel, failure list,
  size estimate.

## Non-Goals

- **Not a general "skip processing" export** for normal (non-merged) images —
  trichrome only, per the request.
- **Framing yes, orientation no** (round 3): the export applies the geometry
  that defines *which pixels the image is* — the slice chain (`source_ops`)
  and the user crop (`crop_rect` + `crop_angle`, via `apply_crop_to_image`) —
  but not the display orientation (flips / 90° rotation / fine rotation).
  An axis-aligned crop is a pure sub-array (values bit-exact); an **angled**
  crop (and a slice chain with a baked rotation) necessarily resamples with
  linear interpolation — inherent to the operation, values stay linear.
  Crop coordinates are defined relative to the (sliced) frame, which is why
  `source_ops` must apply whenever the crop does.
- **No resize** — the format forces "Original size" (the combo is disabled
  while this format is selected and its stored preference is not overwritten).
- **No ICC embedding, no colour-space option** — the data is camera-native
  linear; tagging it sRGB/ProPhoto would be a lie. The colour-space row is
  hidden for this format and a muted hint explains the file's nature.
- **No JPEG variant** — linear data in 8-bit gamma-less JPEG is useless.
- No change to any other export path or to the merge pipeline itself.

## UX / Interaction

### Format combo (`export_dialog.py`)

- When `any(getattr(img, "is_merged", False) for img in ccr_backend.images)`,
  `_build_ui` appends: `self.format_combo.addItem("TIFF 16-bit linear merge
  (unprocessed)", "linear")`. Otherwise the entry does not exist (QSettings
  restore of `"linear"` then falls back to the default TIFF entry via the
  existing `findData >= 0` guard).
- While `"linear"` is selected:
  - Quality row hidden (existing non-JPEG behaviour).
  - Colour-space row hidden; in its place the (re-used) `colorspace_hint` label
    shows: *"Raw trichrome combination: camera-native linear RGB, no profile
    embedded, no conversion or adjustments. Crop and slice framing apply;
    rotation and flips do not."*
  - Resize combo disabled and displayed at "Original size" (selection restored
    when switching back to another format; the persisted `export/resize`
    preference is not clobbered — `_save_settings` skips the resize key for
    linear, and skips the colorspace key too since none applies).
  - Scope radios re-labelled and re-counted for the merged set (see below).
  - Estimate uses the existing TIFF estimator with each image's merged
    canonical dims.

### Scope semantics

The dialog today computes one `_exportable` predicate at construction
(converted, or anything in Positive mode). The linear format needs a second
set. Both are computed in `__init__`:

- `self._converted_indices` — as today.
- `self._merged_indices` — `[i for i, img in enumerate(ccr_backend.images)
  if getattr(img, "is_merged", False)]`.

`_active_indices_all()`, `_current_exportable()`, `_selected_exportable()`
resolve from `format_combo.currentData() == "linear"`. `_on_format_changed`
updates the three radio labels/enablement in place:

- All: `All merged images (N)` vs the existing converted/positive label.
- Current: enabled iff the current image is merged (tooltip: *"The current
  image is not a merged trichrome image."*).
- Selected: `Selected images (K)` counting only merged selections.
- If the checked radio becomes disabled by the switch, selection falls back to
  the All radio (mirrors `_restore_settings`' fallback).

`_scope_indices`, `_update_example`, `_update_estimate`, and
`_on_export_clicked` all read through the format-aware helpers, so naming,
estimate and the empty-scope guard follow automatically. The empty-scope
message becomes format-aware: *"There are no merged images to export."*

### ExportPlan

New field `linear_merge: bool = False`. Set from
`format_combo.currentData() == "linear"`. `jpg_output` stays False,
`max_long_side` forced `None`, `output_colorspace` value irrelevant (worker
ignores it on the linear path).

## Data Model

No new persistent state. `export/format` may persist `"linear"`; restoring it
in a session without merged images falls back to TIFF (guard exists). No
catalog impact (merged images are already excluded from the catalog).

## Processing / Maths

None — that is the point. The write path is:

```python
# ccr_backend.py
def _export_merged_linear(self, image_obj, output_path: str) -> None:
    """Write the raw trichrome combination as an untagged 16-bit linear TIFF.
    Full-resolution re-merge + framing only (slice chain, user crop);
    NO orientation/conversion/adjustments/colour management."""
    from core import ccr_merge
    from core.ccr_processor import apply_crop_to_image, safe_tifffile_imwrite
    merged, _full = ccr_merge.merge_raw_channels(image_obj.merge_sources,
                                                 preview=False)
    if getattr(image_obj, "source_ops", None):
        merged = image_obj._apply_source_ops(merged)   # slice framing
    merged = apply_crop_to_image(merged, getattr(image_obj, "crop_rect", None),
                                 getattr(image_obj, "crop_angle", 0.0) or 0.0)
    out = os.path.splitext(output_path)[0] + ".tiff"
    if not safe_tifffile_imwrite(out, merged, photometric="rgb",
                                 compression="deflate"):
        raise IOError(f"Failed to save image to {out}")
```

`apply_crop_to_image` is a no-op for `crop_rect is None`, so the uncropped,
unsliced export remains byte-identical to round 2.

- `compression="deflate", predictor=True` is lossless (bit-exact values); the
  horizontal predictor (TIFF Predictor 2) decorrelates adjacent pixels so deflate
  compresses 16-bit continuous-tone data ~1.3–2× (plain deflate barely shrinks
  it). Universally readable (libtiff/OpenCV/tifffile). No `iccprofile` → untagged.
- The 3 rawpy decodes per image are the documented cost of any merged-image
  re-read (spec/three-way-rgb-merge.md, Performance note); the parallel export
  pool amortises across images exactly as it does for normal exports.

## Integration Points

| Where | Change |
|---|---|
| `export_dialog.py` | `_merged_indices`; conditional `"linear"` combo entry; format-aware scope helpers + radio relabeling; hide colour-space row / show linear hint; disable resize; `ExportPlan.linear_merge`; `_save_settings` skips resize/colorspace keys when linear. |
| `image_preview.py` `ExportItemsWorker.run` | Pass `linear_merge=self.plan.linear_merge` through to `export_items`. |
| `ccr_backend.py` `export_items` | New kwarg `linear_merge=False`, forwarded per item to `export_image_by_index`. |
| `ccr_backend.py` `export_image_by_index` | New kwarg `linear_merge=False`. When True: if the image is not merged, record a failure ("not a merged trichrome image") and return False; else `_export_merged_linear(image_obj, output_path)` and return True — **before** any conversion routing (no `ci` inspection, no reference/bwpoint fallback). |
| `ccr_backend.py` | New `_export_merged_linear` (above). |

Everything else (worker thread, progress dialog, cancel, failure counting,
conflict resolution, `{name}/{date}/{time}/{seq}` macros, open-folder-when-done)
is reused untouched.

## Edge Cases

- **Source RAW missing/moved since load** → `merge_raw_channels` raises → the
  existing per-item failure handling records it; other items continue.
- **Duplicate of a merged image in scope** → same sources; identical pixels
  unless their crops differ. Allowed; harmless.
- **Slice of a merged image in scope** → its slice chain (and any crop set on
  the slice) applies, so the file is the framed region the user sees — in raw
  linear merge values.
- **Degenerate/None crop** → `apply_crop_to_image` returns the input unchanged
  (existing guard).
- **Non-merged image reaches the linear path** (defensive; scope should
  prevent it) → recorded failure, not a crash.
- **Positive mode on** → irrelevant: the linear path never consults
  positive-mode state (it bypasses `read_image` entirely by calling
  `merge_raw_channels` directly).
- **`export/format` == "linear" persisted, next session has no merged images**
  → combo entry absent, `findData` misses, default TIFF stays selected.
- **Estimate** → existing `get_original_dims` is already `is_merged`-aware
  (never probes the red RAW at full sensor size); the TIFF estimator is used
  as-is. Deflate on linear negatives compresses differently than on converted
  positives — the estimate is approximate, same as today.

## Test Plan

`tests/test_linear_merge_export.py` (offscreen Qt, monkeypatched merge — no
real trichrome fixture exists in the repo):

1. **Byte-exact write**: monkeypatch `ccr_merge.merge_raw_channels` to return a
   known uint16 gradient; a stub merged CCRImage exports; read the TIFF back
   with `tifffile` → array-equal, dtype uint16, shape (H, W, 3); filename ends
   `.tiff`.
2. **Nothing else runs**: monkeypatch `apply_adjustments` /
   `ccr_normalize_with_bwpoint` / `ccr_normalize_with_reference` to raise;
   linear export of a *converted* merged image still succeeds (proves the
   routing bypasses conversion and adjustments entirely).
3. **Non-merged rejection**: `export_image_by_index(..., linear_merge=True)` on
   a normal image returns False and records a failure through `export_items`.
4. **Dialog gating**: with no merged images the combo has no "linear" entry;
   with a merged stub it does.
5. **Scope math**: mixed backend (merged + normal images) → linear scope counts
   only merged; converted scope unchanged; radio labels update on format
   switch; checked-but-disabled radio falls back to All.
6. **Plan resolution**: selecting linear yields `linear_merge=True`,
   `max_long_side=None`, `jpg_output=False`.
7. **Settings hygiene**: switching to linear and exporting does not overwrite
   the persisted `export/resize` / `export/colorspace` values.

Manual verification (needs a real triplet, not in CI): export a merged frame,
open the TIFF in a viewer that shows untagged data linearly, confirm it matches
the pre-conversion negative content and full merged resolution.

## Review resolutions (round 2)

- **Scope predicate collision**: the dialog's single `_exportable` snapshot
  can't serve two formats → resolved with two index sets and format-aware
  helpers; radios re-label in place on format change and fall back to All when
  the checked radio becomes disabled.
- **QSettings pollution**: persisting `resize`/`colorspace` while the controls
  are forced/hidden for linear would clobber the user's normal-export
  preferences → `_save_settings` skips those two keys when linear is selected
  (scope/destination/template/conflict still persist normally).
- **Slice ambiguity**: exporting a slice's region would require `source_ops`
  replay and re-open the "is geometry an operation?" question → v1 exports the
  full frame, stated in the dialog hint and Non-Goals (follow-up noted).
- **`.tif` vs `.tiff`**: `_selected_ext()` returns `.tiff` for every non-JPEG
  format, and `_export_merged_linear` re-normalises the suffix exactly like
  `write_export_image` — consistent with the normal TIFF export.
- **Estimator**: no new estimator work; `get_original_dims` already handles
  merged images and the TIFF heuristic is intentionally approximate.
- **Why backend-level bypass instead of a `write_export_image` flag**: the
  linear path must skip `_load_export_source` (which honours crops and resize)
  and `apply_export_colorspace`; entering the existing chokepoint and disabling
  most of it piecemeal is more fragile than one small, self-contained writer.

## Round 3 (crop support, on request)

The original non-goal "no geometry of any kind" is relaxed to **framing yes,
orientation no**: the slice chain and the user crop now apply (they define
*which pixels* the image is), while flips/rotation/fine rotation and every
value-processing stage remain excluded. Axis-aligned crops are bit-exact
sub-arrays; angled crops/slice rotations interpolate (inherent). `source_ops`
must accompany the crop because crop coordinates live in the sliced frame's
space. Dialog hint updated accordingly. Tests: axis-aligned crop equals the
`apply_crop_to_image` sub-array (and stays bit-exact vs the merge), angled
crop has the box's dimensions, slice chain applies, and the
uncropped/unsliced path stays byte-identical.

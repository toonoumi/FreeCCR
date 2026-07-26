# Replace trichrome originals with a linear TIFF

> Status: draft.

## Summary

An opt-in workflow for **3-way RGB merge (trichrome)** capture: after a merge
import succeeds, **bake each merged image to a full-resolution 16-bit linear
TIFF**, **permanently delete** its three source RAWs, and **reload the list from
the generated TIFFs**. This flattens a trichrome shoot (3 RAWs per frame) into
one archival file per frame, so future sessions load a single normal linear-TIFF
negative instead of re-merging three RAWs.

The linear TIFF is the exact `merge_raw_channels` output — each frame's own
channel, black-subtracted, white-level-scaled, stacked — written losslessly
(deflate), untagged, camera-native linear RGB. It is **not** a Bayer RAW, but for
a merged image the merge output *is* the final full-resolution pixel data, so no
information about the merged result is lost. The only thing not preserved is the
ability to re-merge differently later (e.g. change the demosaic mode), because
that needs the original mosaics — the current merge/demosaic choice is baked in.

Reuses the existing linear-TIFF writer (`_export_merged_linear`,
spec/trichrome-linear-tiff-export.md).

## Goals

- A persistent checkbox in the **Trichrome capture** group of Settings → Color
  Management: **"Replace originals with linear TIFF on import"**.
- While the checkbox and 3-way merge are both on, after a **successful** merge
  import (Open Files or Open Folder), FreeCCR:
  1. **confirms** once (count of TIFFs to create + RAWs to delete; states it is
     permanent and cannot be undone);
  2. writes one full-resolution linear TIFF per merged image, next to its red
     source, `<mergedDisplayNameStem>.tiff`;
  3. **verifies** each TIFF exists and reads back as a valid uint16 `(H, W, 3)`
     image **before** touching any original;
  4. **permanently deletes** the three source RAWs of every image whose TIFF was
     written and verified;
  5. **reloads** the list from the generated TIFFs as normal images (merge
     bypassed for this reload), and reports how many were replaced + any
     failures.
- Non-destructive on failure: a source RAW is deleted **only** after its
  replacement exists and verifies; a per-image write/verify failure leaves that
  image's originals intact.
- Cancellable: cancelling the bake deletes **nothing** and removes any TIFFs
  already written (leaves the session exactly as it was — merged, all RAWs
  present).

## Non-Goals

- Not for non-merged images (trichrome only).
- No Recycle Bin / trash: deletion is a permanent `os.remove` (user choice), so
  it is gated by an explicit confirmation and readback verification.
- No re-merge after baking (originals are gone); the demosaic mode is baked in.
- No new export format or change to the existing linear-TIFF export path.
- No framing at import time: a freshly imported merged image has no slice/crop,
  so the baked TIFF is the full canonical merge. (`_export_merged_linear` applies
  `source_ops`/crop, which are identity here.)

## UX / Interaction

### The checkbox (`settings_dialog.py`)

In the **Trichrome capture** group, after the "Merge detail" row, add
`self._cb_rgb_replace` — *"Replace originals with linear TIFF on import"* — with
muted help:

> After a successful 3-way merge import, each merged frame is written as a
> full-resolution 16-bit linear TIFF next to its red source, then the three
> original RAWs are **permanently deleted** and the list reloads from the TIFFs.
> You'll confirm before anything is deleted. The merge/demosaic choice is baked
> in and cannot be changed afterwards. Requires 3-way merge to be on.

- Staged like the other trichrome toggles: seeded in `_init_toggles` from
  `ccr_backend.rgb_merge_replace`; applied on Done in `_apply_pending` via
  `self._mw.on_rgb_merge_replace_toggled(checked)` when it differs.

### Trigger (`main_window.py`)

- Persistent flag `ccr_backend.rgb_merge_replace` (QSettings
  `import/rgb_merge_replace`), restored at startup next to `rgb_merge_mode`.
- After a load finishes, `_cleanup_loader` (GUI thread, after thumbnails +
  merge-error surfacing) calls `_maybe_replace_merged_with_tiff()`:
  - Runs only when `rgb_merge_mode and rgb_merge_replace`, there is **no**
    `last_merge_error`, at least one loaded image `is_merged`, and no replace is
    already in progress (re-entrancy guard).
  - Shows a modal confirm (`Yes`/`No`, default `No`) naming the counts. `No` →
    keep the merged session untouched.
  - `Yes` → start `MergeReplaceWorker` in a `QThread` with a `QProgressDialog`
    ("Writing linear TIFFs…", cancellable).
- On the worker's `finished(result)`:
  - Close the progress dialog; clear the in-progress guard.
  - If `result["cancelled"]` → toast "Replace cancelled — originals kept.",
    reload nothing (session stays merged).
  - Else reload from `result["tiff_paths"]` via `_launch_file_loader(tiffs,
    force_no_merge=True)`, and toast/warn a summary ("Replaced M frames with
    linear TIFFs; deleted N originals" + any failures list).
- The reload runs through the normal loader worker (thumbnails, progress). Its
  images are ordinary TIFFs (`is_merged=False`), so the reload's own
  `_cleanup_loader` never re-triggers the bake.

### Force-no-merge reload

`load_images_from_files(..., force_no_merge=True)` skips the merge branch so the
generated TIFFs load as normal negatives even though `rgb_merge_mode` is still
on (TIFFs aren't RAW; the merge validator would otherwise reject them). The
global merge toggle is **not** changed — the user's preference is preserved.

### Reopening a baked TIFF in merge mode (marker)

So the user never has to toggle 3-way merge off just to reopen a baked frame,
the generated TIFF is stamped with a marker in its **Software** tag
(`ccr_merge.FREECCR_MERGE_TIFF_MARKER`, written by `_export_merged_linear(...,
mark=True)`). `ccr_merge.is_freeccr_merge_tiff(path)` reads only the TIFF header
to detect it.

When 3-way merge is on, `load_images_from_files` (and `open_files`'
pre-validation) partition the selection:

- **marked TIFFs** → loaded as **normal images** (never merge inputs);
- **everything else** → the usual merge path (validate multiple-of-3 + RAW,
  group into triplets).

If every selected file is a marked TIFF, the whole import falls through to the
normal load (no merge, no "needs a multiple of 3" error). A mixed import (marked
TIFFs + RAW triplets) merges the RAWs and loads the marked TIFFs alongside them
(`_load_merged_triplets(..., passthrough_paths=…)`). The `force_no_merge` reload
path above still exists for the replace flow's own reload; the marker makes a
*manual* reopen work too. Only files FreeCCR baked carry the marker — an
arbitrary third-party TIFF in merge mode is still rejected as a non-RAW input.

## Data Model

- `ccr_backend.rgb_merge_replace: bool = False` (in `_init`, next to
  `rgb_merge_mode`/`rgb_merge_demosaic`).
- No catalog impact: replaced merged images simply leave the list on reload;
  their composite-key catalog entries (if any) become stale (sources deleted)
  and are ignored/pruned. The new TIFFs are normal cataloged files.

## Processing

### Backend `bake_merged_to_linear_tiff(images=None, cancel_flag=None, progress_cb=None)`

Returns `{"tiff_paths": [...], "deleted": [...], "failures": [(name, reason)],
"cancelled": bool}`.

1. `imgs` = the `is_merged` images (with `merge_sources`) from `images`
   (default `self.images`).
2. For each, in order (emitting `progress_cb(done, total, name)`):
   - `out = _linear_tiff_output_path(im)` — `<stem>.tiff` next to the red
     source, conflict-suffixed (`_2`, `_3`, …) if the name is taken.
   - `_export_merged_linear(im, out)` — full-res re-merge + framing (identity at
     import) → lossless untagged linear TIFF.
   - `_verify_linear_tiff(out, im)` — the file exists, is non-empty, and its
     TIFF series reads back as `uint16`, `ndim == 3`, 3 channels, H/W > 0.
     Raises otherwise.
   - Success → record `(im, out)`; failure → record `(name, reason)` and add the
     image's sources to a `bad_sources` set (never delete those).
   - `cancel_flag()` before an image → set `cancelled`, stop.
3. **Cancelled** → delete every TIFF already written (no orphans), delete **no**
   source, return `cancelled=True`, `tiff_paths=[]`.
4. Else → for each successfully baked image, mark its sources for deletion,
   **excluding** any source also referenced by a failed image (shared-source
   safety); delete each unique source with `os.remove` (per-file failures become
   `failures` entries but never abort the rest). Return the tiff paths (for
   reload), the deleted sources, and any failures.

Helpers:
- `_linear_tiff_output_path(im)` — dir of `merge_sources[0]`, stem of
  `display_name` (falls back to the red basename), `.tiff`, conflict-suffixed.
- `_verify_linear_tiff(out, im)` — the readback guard above (uses
  `safe_unicode_path` + `tifffile.TiffFile`).

### Worker (`main_window.py`)

`MergeReplaceWorker(QObject)`: `progress(int, int, str)`, `finished(object)`;
`run()` calls `ccr_backend.bake_merged_to_linear_tiff(self.images, cancel_flag,
progress_cb)` and emits the result; `cancel()` sets the flag.

## Integration Points

| Where | Change |
|---|---|
| `ccr_backend.py` `_init` | `rgb_merge_replace = False`. |
| `ccr_backend.py` `load_images_from_files` | new `force_no_merge=False`; merge branch guarded by `and not force_no_merge`; partitions marked TIFFs out of the merge set. |
| `ccr_backend.py` `_load_merged_triplets` | new `passthrough_paths` — marked TIFFs loaded as normal images alongside merged triplets. |
| `ccr_backend.py` `_export_merged_linear` | new `mark=False`; when True, stamps the Software-tag marker; the bake passes `mark=True`. |
| `ccr_merge.py` | `FREECCR_MERGE_TIFF_MARKER`; `is_freeccr_merge_tiff(path)`. |
| `main_window.py` `open_files` | merge pre-validation excludes marked TIFFs. |
| `ccr_backend.py` | `bake_merged_to_linear_tiff`, `_linear_tiff_output_path`, `_verify_linear_tiff`. |
| `settings_dialog.py` | `_cb_rgb_replace` checkbox + help; seed in `_init_toggles`; apply in `_apply_pending`. |
| `main_window.py` `__init__` | restore `import/rgb_merge_replace` → backend. |
| `main_window.py` | `on_rgb_merge_replace_toggled`; `ImageLoaderWorker(force_no_merge=…)`; `_launch_file_loader` helper (shared by `open_files` + reload); `MergeReplaceWorker`; `_maybe_replace_merged_with_tiff`, `_start_merge_replace`, `_on_replace_progress`, `_on_replace_finished`. |
| `main_window.py` `_cleanup_loader` | call `_maybe_replace_merged_with_tiff()` after the merge-error check. |

## Edge Cases

- **User declines the confirm** → nothing happens; session stays merged.
- **A source RAW missing at bake** → `merge_raw_channels` raises → that image
  fails, its (present) sources are **not** deleted, others proceed.
- **TIFF write returns False / truncated** → `_export_merged_linear` raises or
  `_verify_linear_tiff` rejects → failure, sources kept.
- **Cancel mid-bake** → no deletions, written TIFFs removed, session unchanged.
- **Output name already exists** → conflict-suffixed; never overwrites.
- **Duplicate/slice of a merged image present** (not at fresh import, but
  defensive) → images sharing a source: a source is deleted only if every image
  referencing it succeeded.
- **`force_no_merge` reload** → TIFFs load as normal negatives; merge toggle
  unchanged; reloaded images are not merged, so no re-trigger.
- **Replace on but merge off** → no trigger (guard requires `rgb_merge_mode`).
- **Partial failures** → the summary lists them; successfully replaced frames
  are reloaded, failed frames' RAWs remain on disk (re-import to retry).

## Test Plan

`tests/test_merge_replace.py` (offscreen Qt, `merge_raw_channels` monkeypatched —
no real triplet in the repo; stub merged images with real temp source files):

1. **Bake + delete**: two stub merged images → two valid uint16 `(H, W, 3)`
   TIFFs on disk; all six source files deleted; no failures; `cancelled` False.
2. **Write failure preserves sources**: make the merge raise for one image → it
   is a failure, its three sources survive; the other image bakes and its
   sources are deleted.
3. **Verify catches corruption**: monkeypatch `safe_tifffile_imwrite` to create
   an empty/invalid file → `_verify_linear_tiff` raises → failure, sources kept.
4. **Cancel aborts deletion**: `cancel_flag` → `cancelled` True, no deletions,
   no leftover TIFFs, all sources present.
5. **Conflict-safe naming**: a pre-existing `<stem>.tiff` → the bake writes
   `<stem>_2.tiff` and does not overwrite the existing file.
6. **force_no_merge load**: with `rgb_merge_mode` on, `load_images_from_files
   ([tiff], force_no_merge=True)` loads the TIFF as one non-merged image.
7. **Marker + reopen**: a baked TIFF is detected by `is_freeccr_merge_tiff`
   (a plain TIFF / RAW ext / missing file is not); with `rgb_merge_mode` on and
   NO force flag, `load_images_from_files([marked_tiff])` loads it as one normal
   image with no `last_merge_error` (no toggling needed).

Manual / launch verification (needs a real triplet): enable both toggles, Open
three RAWs, confirm → a linear TIFF appears, the RAWs are gone, the list shows
the TIFF; reopen it later as a normal negative.

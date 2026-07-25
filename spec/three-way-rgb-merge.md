# 3-Way RGB-Light Merge Mode

> Status: refined (post adversarial review). Round-2 decisions are folded inline;
> see the "Review resolutions" note at the end for the rationale trail.

## Summary

A capture/import mode for **trichrome (three-colour) photography**: the user
shoots one static scene three times, each under a single pure light — **red,
then green, then blue** — and FreeCCR merges every consecutive triplet of source
RAWs into one full-colour image by taking each frame's *own* colour channel and
discarding the other two, **without demosaicing**. The merged image then flows
into the existing negative-inversion pipeline unchanged.

The mode is a global toggle on the **Settings → Color Management** tab, mirroring
the existing Positive-mode toggle.

## Goals

- Add a persistent boolean setting **"3-way RGB merge mode"** in the Color
  Management tab of the Settings dialog.
- When ON, importing images (Open Files **and** Open Folder):
  1. requires the input count to be a **multiple of 3** (else a clear error,
     nothing loads);
  2. **sorts** the inputs by filename (basename, case-insensitive);
  3. groups every 3 consecutive files into one triplet `(R, G, B)`;
  4. merges each triplet into a single RGB image by **channel selection without
     demosaicing** — R-plane from file 1, G-plane from file 2, B-plane from file 3;
  5. presents the merged image as a normal loaded image that continues into the
     existing negative-inversion / adjustment / export pipeline.
- Merged images survive the full lifecycle: preview, convert (negative
  inversion), slider adjustments, zoom hi-res detail, export, **Duplicate, and
  Slice** (the latter two by re-merging on re-read — see Integration).
- Keep the merge maths pure and unit-testable, independent of rawpy/Qt.

## Non-Goals

- **No image registration / alignment.** The three frames are assumed already
  aligned (tripod, static scene). Misalignment is the user's responsibility.
- ~~**No cross-session catalog persistence of merged images.**~~ *(Implemented —
  see `spec/merge-catalog-persistence.md`.)* Originally a merged image was a
  session artifact excluded from the catalog on every serialization path. It is
  now persisted under a **composite key of the 3 source signatures**, so edits
  (reference frame, conversion, crop, sliders, dust) survive between sessions and
  are restored when the same 3 frames are merged again (and none has changed).
- **Bayer (RGGB) and monochrome sensors.** "Do not demosaic / take one channel"
  applies to a 2×2 Bayer CFA (collapse each tile → one pixel) and, even more
  naturally, to a monochrome sensor (no CFA at all — the whole grayscale frame is
  that light's channel, full resolution). X-Trans (Fujifilm `.raf`) and 4-colour
  (CYGM/RGBE) sensors are rejected at decode time with a clear message. All three
  frames of a triplet must be the same sensor type.
- **Non-RAW inputs are not supported.** Validated by extension up front and by a
  sensor-type guard at decode.
- **Resolution is sensor-native.** A Bayer merge is inherently a 2×2-binned,
  half-sensor-resolution image (that binned size *is* its full resolution); a
  monochrome merge is **full sensor resolution** (no binning needed). For a
  monochrome merge, `preview`/zoom decodes may run at half size for speed, but
  export delivers full sensor resolution.
- The mode is global; no per-image override. No change to the non-merge flow when
  the toggle is OFF.
- **Positive mode is assumed OFF for merge.** Trichrome frames are negatives; the
  request is to invert them. The two toggles are conceptually opposite (see Edge
  Cases) — we surface a hint rather than force an override.

## UX / Interaction

### The toggle

`src/widgets/settings_dialog.py` → `_build_color_management_page()`. Add a new
group box after the existing "Negative conversion" group, mirroring the
Positive-mode pattern (descriptive group title; behaviour in the checkbox +
muted text):

```
[ Negative inversion ]                 (existing group)
   ☐ Positive mode …

[ Trichrome capture ]                  (new group)
   ☐ 3-way RGB merge (combine red/green/blue-light exposures)
   <muted>  Shoot a static scene three times under pure red, then green, then
            blue light. On your NEXT import, every 3 RAWs (sorted by filename)
            are merged into one colour image — each frame contributes only its
            own channel, no demosaicing — then converted as a negative. RAW
            (Bayer) only; the selected count must be a multiple of 3. Applies to
            the next import only; merged-image edits are not saved between
            sessions.
```

- Checkbox `self._cb_rgb_merge`, `toggled` → `self._on_rgb_merge(checked)` →
  `self._mw.on_rgb_merge_mode_toggled(checked)` (same delegation as `_on_positive`).
- `refresh_color_management()` syncs the checkbox from
  `ccr_backend.rgb_merge_mode` with `blockSignals`, like `_cb_positive`.

### Behaviour on import

- **Toggle is sticky/global**, persisted in `QSettings` under
  `import/rgb_merge_mode`; restored at startup before any image loads.
- Toggling does **not** retroactively re-merge loaded images; it affects only the
  **next** import. Transient hints:
  - on: *"3-way RGB merge on — your next import merges every 3 RAWs (R,G,B by
    filename)."* If Positive mode is also on, append: *" Positive mode is on;
    merged frames are negatives — turn Positive off to invert them."*
  - off: *"3-way RGB merge off."*
- On import with the mode ON:
  - **Valid**: triplets merge in parallel; the list shows N/3 merged items, each
    named `<firstStem>_RGB<ext>` (de-duplicated within the batch).
  - **Invalid count** (not a multiple of 3): *"3-way RGB merge needs a multiple of
    3 images (got N). Select 3 frames per shot: red, green, blue."* — nothing
    loads.
  - **Non-RAW present**: *"3-way RGB merge requires RAW files. These are not
    supported RAW: …"* — nothing loads.
  - **Unsupported sensor** (X-Trans / 4-colour, or a triplet mixing monochrome
    and Bayer — only detectable at decode): the triplet is rejected at decode;
    the import reports it via `last_merge_error`. (Monochrome and Bayer are both
    supported; only X-Trans/4-colour and mixed-type triplets are rejected.)

### Error surfacing across both entry points

`load_images_from_folder` already delegates to `load_images_from_files` (it ends
with `load_images_from_files(sorted(file_paths), …)`), so **the merge branch lives
only in `load_images_from_files`** and both entry points inherit it.

- `open_files` holds the validated list, so it **pre-validates** with
  `validate_merge_inputs` and shows the `QMessageBox` *before* starting the loader
  (immediate feedback; returns without starting a loader — does not rely on
  `_cleanup_loader`).
- The backend also validates defensively (covers the folder path and decode-time
  Bayer rejection) and records `ccr_backend.last_merge_error`. The main thread
  surfaces it in `_cleanup_loader` (runs on the GUI thread via `thread.finished`,
  after `load_thumbnails` has rendered the now-empty panel) with a `QMessageBox`.
- `last_merge_error` is **reset to `None` at the very top of
  `load_images_from_files`** (next to the B/W-point clears) and **read-then-cleared**
  in `_cleanup_loader`, making it strictly per-load (no cross-load staleness).

## Data Model

### CCRImage (`src/core/ccr_image.py`)

New constructor kwargs + attributes:

- `is_merged: bool = False`
- `merge_sources: Optional[list[str]] = None` — the 3 absolute source paths in
  `(R, G, B)` order.

`self.file_path = merge_sources[0]` (the red exposure) so existing machinery
(lensfun EXIF read, lens correction, the nominal path) works on a real RAW of the
right camera/lens. `display_name` defaults to `<stem(file_path)>_RGB<ext>`.

`read_image()` gains a dispatch at the very top (after path normalisation, before
the extension branch and before `positive_mode` is read):

```python
if self.is_merged and self.merge_sources:
    return self._read_merged(preview=preview, max_long_side=max_long_side)
```

`_read_merged(preview, max_long_side)` mirrors the tail of the RAW branch:

```python
from core.ccr_merge import merge_raw_channels
rgb, full_decode_size = merge_raw_channels(self.merge_sources)  # half-sensor res
rgb = self._apply_source_ops(rgb)                # identity for a non-slice
self.original_full_size = self._ops_full_size(full_decode_size)
if max_long_side:
    rgb = self.resize_image_to_max_pixel(rgb, max_long_side)
return rgb
```

`preview` is forwarded to `merge_raw_channels`: for **Bayer** it is irrelevant
(the 2×2 bin is already the only resolution); for **monochrome** it lets the
decode run at half size for a fast preview/zoom tile. `merge_raw_channels` always
returns the *canonical full* size (full sensor for monochrome, binned for Bayer)
so `original_full_size` is correct even when a monochrome preview array is
half-sensor; `max_long_side` then does the final downsize.

Because the dispatch lives inside `read_image`, **export, zoom hi-res replay,
slicing, and duplication all compose for free** — each re-reads through
`read_image(self.file_path)` and transparently re-merges from `merge_sources`
(slice children and duplicates carry `is_merged`/`merge_sources`; see
Integration). `_stamp_profile_signature` is left as-is; merged images decode
camera-native and may show a profile-mismatch ⚠ if the active profile later
changes — acceptable for v1 (documented).

**`backend.file_paths`** after a merge load holds one path per merged image (the
red exposure of each triplet). The G/B sources do not appear in `file_paths`.

### Backend (`src/core/ccr_backend.py`)

- `self.rgb_merge_mode: bool = False` (in `_init()`, next to `positive_mode`).
- `self.last_merge_error: Optional[str] = None`.

### Catalog (`src/core/catalog.py` + backend serialization paths)

Merged images must never enter the per-file catalog. Guard **every** path:

- `update_for_images()` skips `getattr(img, "is_merged", False)`.
- `remove_images_by_indices()` (`ccr_backend.py:1177`) skips `is_merged` images —
  does **not** add them to `_catalog_preserved` (this was the review's blocker:
  removing a merged image would otherwise serialize its edits under the red RAW's
  key).
- `serialize_image()` is belt-and-suspenders: callers above already skip merged,
  but a one-line note documents the invariant. (No functional change required if
  the two callers are guarded; we still add the guard at the loop level.)

## Processing / Maths

### "No demosaic" for the two sensor kinds

**Bayer.** A Bayer sensor records one colour per photosite in a 2×2 tile (RGGB).
A normal decode *demosaics* — interpolates the two missing colours at every site.
The requirement is the opposite: take **only** the photosites that natively
measured the wanted colour, **never mixing in the other colours** (zero
inter-channel crosstalk). We read the RAW Bayer mosaic directly
(`raw.raw_image_visible` + `raw.raw_colors_visible`) and phase-slice out a single
colour's sites — `mosaic[dy::2, dx::2]`, where `(dy, dx)` is that colour's
position in the 2×2 tile (read from the actual `colors` at the visible origin, so
it is offset-safe). **No demosaic and no libraw colour pipeline whatsoever.** R
and B have one site per quad → that bare site; **green has two sites per quad,
which are averaged** (both are green, so this is not crosstalk, and it preserves
the green SNR). The black pedestal is subtracted per site manually (`raw_image`
carries it). This yields a half-width × half-height plane (one site per quad = the
Bayer merge's full resolution).

> `half_size=True` gives a numerically similar result (it bins each quad to
> `R = R-site, G = mean(two G-sites), B = B-site`), but it goes through libraw's
> postprocess pipeline. Reading the mosaic directly keeps full control and
> guarantees no hidden colour operation, which is why it is used instead.

**Monochrome.** A monochrome sensor has no CFA, so there is nothing to demosaic:
every photosite measured the (single) light's intensity. The whole grayscale
frame **is** that frame's channel, at **full sensor resolution**. This is in fact
the ideal trichrome sensor — no wasted photosites, no resolution loss.

### Per-frame extraction

- **Bayer:** read the raw mosaic; `extract_cfa_channel(mosaic, colors,
  color_desc, letter, black_levels)` returns that colour's plane — the single
  site for R/B, the per-site-black-subtracted **average of the two sites for
  green** — matching contributing sites by CFA letter (so it works whether the
  two greens share one colour index or use indices 1 and 3). No `raw.postprocess`
  at all. (`bayer_channel_indices` is still the guard that rejects a non-R/G/B
  `color_desc`.)
- **Monochrome:** `raw.postprocess(half_size=preview, output_color=raw,
  gamma=(1,1), no_auto_bright=True, no_auto_scale=True, use_camera_wb=False)`;
  the channel **is** the grayscale plane (`rgb` if 2-D, else `rgb[..., 0]` — the
  channels are equal). `preview` gives a fast half-size decode; the full sensor
  size is still reported as the canonical resolution.

Either way the plane is scaled by `65535/white_level` in `combine_channels`
(matching `read_image`'s black-subtracted negative decode).

Verified on `example_raw/DSC07096.ARW` (Bayer) against libraw's own
`half_size`+`output_color=raw` read: R and B are **byte-identical** (max abs diff
0), and green matches the two-green average to within float-vs-integer **rounding**
(max abs diff 6 / 65535 ≈ 0.01% of pixels) — confirming the phase, black-level,
and green-average handling. The direct read reaches the same values as `half_size`
without going through libraw's postprocess. (Monochrome cannot be CI-verified
here — the repo ships no monochrome RAW — so its decode mirrors `read_image`'s
existing monochrome path.)

### Sensor-type guard (decode-time)

After `rawpy.imread`, classify the sensor and reject the unsupported kinds, with a
clear error surfaced via `last_merge_error`:

- **Monochrome** (accepted): `num_colors == 1`, a grey `color_desc`
  (`b'G'`/`b'GRAY'`/`b'GREY'`), or a `b'RGBG'` desc whose CFA pattern is all one
  index — mirrors `read_image`'s monochrome detection.
- **Bayer** (accepted): `num_colors == 3`, `color_desc` a permutation containing
  `R`,`G`,`B`, and `raw_pattern` shape `(2, 2)`.
- **Rejected:** X-Trans (`.raf`, 6×6) and 4-colour (CYGM/RGBE) sensors, with a
  message naming the file and reason.

All three frames of a triplet must be the **same** type (`merge_raw_channels`
raises if a triplet mixes monochrome and Bayer — they have different
resolutions).

### Channel selection by `color_desc` (not hardcoded indices)

`output_color=raw` emits channels in libraw's internal colour-index order, which
is defined by `color_desc`. For the canonical `b'RGBG'`, index 0 = R, 1 = G,
2 = B — but this must be **derived**, not assumed, so a permuted desc is handled
and an invalid one is rejected:

```python
def bayer_channel_indices(color_desc: bytes) -> tuple[int, int, int]:
    s = color_desc.decode("ascii", "ignore").upper()
    r, g, b = s.find("R"), s.find("G"), s.find("B")
    if -1 in (r, g, b):
        raise ValueError(f"non-RGB Bayer color_desc {color_desc!r}")
    return r, g, b   # (RGBG) -> (0, 1, 2)
```

For the **red-light** frame pick channel `r`, the **green-light** frame channel
`g`, the **blue-light** frame channel `b`.

### White-level scaling

Scale each picked plane by `65535 / white_level` (clipped to `[0, 65535]`),
**matching `read_image` lines 546–552** for pipeline consistency. (Because the
data is already black-subtracted, the true ceiling is `white_level − black_level`,
so saturated pixels land slightly below 65535 — identical behaviour to every
other negative decode; intentional, not a merge-specific bug.) Per-frame scaling
means differing exposures normalise consistently. Output contract matches the
inversion pipeline: `(H, W, 3)` `uint16`, `[0, 65535]`, **linear**, RGB order.

### Performance note

`_read_merged` performs **3 rawpy decodes** per call. Export and every zoom
hi-res tile re-read via `read_image`, so a merged image costs ~3× a normal
image's re-decode. Acceptable for v1 (no session cache); documented so it isn't
filed as a regression.

### Module API (`src/core/ccr_merge.py`)

```python
# Exactly read_image's rawpy-Bayer-decodable set (== export_estimator.RAW_EXTS);
# deliberately NOT the broader folder glob, and excludes .fff (treated as TIFF).
RAW_EXTENSIONS = frozenset({
    ".cr3", ".cr2", ".nef", ".arw", ".dng", ".rw2", ".orf", ".raf",
    ".srw", ".pef", ".3fr",
})

def is_raw_path(path) -> bool
def sort_for_merge(paths) -> list[str]                 # basename, case-insensitive
def group_into_triplets(sorted_paths) -> list[tuple[str, str, str]]
def validate_merge_inputs(paths) -> tuple[bool, Optional[str]]
        # (ok, error). Checks: non-empty, len % 3 == 0, all is_raw_path.
def bayer_channel_indices(color_desc) -> tuple[int, int, int]   # pure
def is_monochrome_sensor(num_colors, color_desc, raw_pattern=None) -> bool  # pure
def extract_cfa_channel(mosaic, colors, color_desc, letter, black_levels=None) -> np.ndarray
        # pure: one CFA colour's plane from the raw mosaic — single site for R/B,
        # per-site-black-subtracted average of the two sites for green
def combine_channels(plane_r, plane_g, plane_b, white_levels) -> np.ndarray  # pure
        # crop to common (min H, min W); scale each by 65535/wl; clip; stack -> uint16
def merge_raw_channels(sources, preview=False) -> tuple[np.ndarray, tuple[int, int]]
        # decode 3 RAWs camera-native (Bayer half-size 2x2 bin / monochrome
        # full-size), sensor-type guard, take each frame's channel, combine;
        # returns (merged, full_size). preview only affects monochrome.
```

`merge_raw_channels` is the only rawpy-touching function; everything else
(`is_raw_path`, `sort_for_merge`, `group_into_triplets`, `validate_merge_inputs`,
`bayer_channel_indices`, `is_monochrome_sensor`, `extract_cfa_channel`,
`combine_channels`) is pure and unit-tested.

## Integration Points

| Where | Change |
|---|---|
| `src/core/ccr_merge.py` | **New module** (API above). |
| `ccr_image.py` `__init__` | Accept `is_merged`/`merge_sources`; store **before** the `read_image` call; default `display_name` to `<stem>_RGB<ext>` when merged. |
| `ccr_image.py` `read_image` | Top-of-method dispatch to `_read_merged`; add `_read_merged`. |
| `ccr_backend.py` `_init` | `rgb_merge_mode=False`, `last_merge_error=None`. |
| `ccr_backend.py` `load_images_from_files` | Reset `last_merge_error`; if `rgb_merge_mode`: `sort_for_merge` → `validate_merge_inputs` (fail ⇒ set `last_merge_error`, load nothing) → `group_into_triplets` → set `self.file_paths` to triplet reps (red paths) up front so the progress count tops out at N → build one merged `CCRImage` per triplet in the pool (`_load_merged_triplet`, catches decode errors → records `last_merge_error`, skips triplet) → de-dup display names → sort + assign. |
| `catalog.py` `update_for_images` | Skip `is_merged` images. |
| `ccr_backend.py` `remove_images_by_indices` | Skip `is_merged` (don't preserve into catalog). |
| `ccr_backend.py` `duplicate_images_by_indices` | Pass `is_merged`/`merge_sources` to the dup `CCRImage` so re-reads re-merge. |
| `ccr_backend.py` `slice_image_by_index` | Pass `is_merged`/`merge_sources` to each child so hi-res/export re-reads re-merge (parent's shared decode already re-merges). |
| `export_estimator.py` `get_original_dims` | If `is_merged`, never probe `file_path` (full-sensor RAW = 2× wrong); use `original_full_size` / `resized_raw` dims. |
| `settings_dialog.py` | New group box + `_cb_rgb_merge`; `_on_rgb_merge`; sync in `refresh_color_management`. |
| `main_window.py` `__init__` | Restore `import/rgb_merge_mode` → `ccr_backend.rgb_merge_mode` before first load. |
| `main_window.py` | `on_rgb_merge_mode_toggled(checked)`: set backend flag, persist QSettings, hint (incl. Positive-mode note). |
| `main_window.py` `open_files` | When merge mode on, pre-validate via `validate_merge_inputs`; on failure show `QMessageBox`, return without loading. |
| `main_window.py` `_cleanup_loader` | If `ccr_backend.last_merge_error`, show `QMessageBox.warning`, then clear it. |

### Loader detail

```python
def _load_merged_triplet(triplet, order):           # runs in the pool
    if cancel_flag and cancel_flag(): return None
    r, g, b = triplet
    try:
        img = CCRImage(r, is_merged=True, merge_sources=[r, g, b])
    except Exception as e:
        self.last_merge_error = str(e)               # e.g. non-Bayer reason
        return None
    img._catalog_order = order
    return img
```

Triplet display names are de-duplicated within the batch (append `_2`, `_3` on
collision), mirroring the existing duplicate/slice naming guards. The post-load
sort by `(basename, _catalog_order)` keys on the (unique) red basenames, keeping
triplet order stable.

## Edge Cases

- **Count not a multiple of 3** → rejected, nothing loads (both entry points).
- **Non-RAW file present** → rejected up front with the offending names.
- **Monochrome RAW** → supported: decoded full-sensor, the grayscale frame is the
  channel (no demosaic needed).
- **Unsupported sensor** (X-Trans / 4-colour) → rejected at decode via the
  sensor-type guard; reported through `last_merge_error`.
- **Mixed-type triplet** (monochrome + Bayer) → rejected at decode (different
  resolutions would silently misalign); reported through `last_merge_error`.
- **A source fails to decode** → `merge_raw_channels` raises; the pool task
  catches, records `last_merge_error`, skips that triplet; others still load.
- **Differing frame dimensions** within one sensor type → `combine_channels`
  crops to the common min H/W.
- **Duplicate of a merged image** → carries `is_merged`/`merge_sources`; initial
  preview reuses the copied `resized_raw`, and zoom/export re-merge correctly.
- **Slice of a merged image** → children carry `is_merged`/`merge_sources`; the
  parent's shared decode re-merges, children crop via `source_ops`, and their
  hi-res/export re-reads re-merge then crop. (Could not be end-to-end tested
  without a real triplet fixture; correct by the verified `read_image` dispatch
  composition. `_read_merged` raises if a source file is missing, failing loudly
  rather than silently wrong.)
- **Positive mode + merge** → conceptually opposite. v1 does **not** override:
  with Positive mode on, merged frames are decoded but treated as positives
  (not inverted), which is wrong for trichrome. We surface a hint when enabling
  merge while Positive is on. Toggling Positive after a merge import re-decodes
  (re-merges) correctly but still treats them as positives. Documented; the user
  should keep Positive off for trichrome.
- **Mode toggled while images are loaded** → no retroactive change; only the next
  import is affected.

## Test Plan

`tests/test_three_way_merge.py` — pure-function unit tests (no rawpy/Qt):

1. `validate_merge_inputs`:
   - empty → invalid; 3/6 RAW → valid; 4 RAW → invalid, **message contains the
     count "4"**; 3 with a `.jpg` → invalid, **message lists the `.jpg`**;
   - `.fff` and `.heic` are rejected by `is_raw_path` (not in `RAW_EXTENSIONS`).
2. `is_raw_path` accepts each of the 11 `RAW_EXTENSIONS` (case-insensitive),
   rejects `.jpg/.png/.tif/.tiff/.fff/.heic`.
3. `sort_for_merge` orders by case-insensitive basename, ignoring directory;
   **stable across files that collide on basename in different dirs**.
4. `group_into_triplets` of 6 sorted paths → 2 triplets, correct membership/order.
5. `bayer_channel_indices`: `b'RGBG'` → `(0,1,2)`; `b'GRBG'` → `(1,0,2)` (or the
   documented mapping); raises on `b'RGBE'`/`b'G'`.
6. `combine_channels`:
   - constructs three distinct constant-colour planes; asserts merged R == frame0
     R-plane, G == frame1 G-plane, B == frame2 B-plane (correct frame→channel);
   - white-level scaling: `white_level = 32767` ⇒ values ≈ double toward 65535;
   - size-mismatch inputs crop to common min, no raise;
   - output dtype `uint16`, shape `(H, W, 3)`, range `[0, 65535]`.

Manual / launch verification:
- App launches; Settings → Color Management shows the new toggle; toggling
  persists across restart (QSettings).
- `merge_raw_channels` and slice/duplicate-of-merged cannot be CI-tested here:
  `example_raw/` has only 2 ARWs (no aligned trichrome triplet). Note this; a
  small synthetic 3-RAW fixture or an env-gated manual test is a follow-up.

Existing suite: `pytest tests/ -v`; the known pre-existing failures
(`exposure_base`, occasional native crash) are unrelated.

## Review resolutions (round 2)

Folded from adversarial review: catalog isolation extended to
`remove_images_by_indices`/serialize paths (was a blocker); Duplicate & Slice now
thread `is_merged`/`merge_sources` (avoid silent wrong export); decode-time Bayer
guard + `color_desc`-derived channel mapping (portability); `RAW_EXTENSIONS`
pinned to read_image's exact Bayer set excluding `.fff`; `last_merge_error` reset
per-load; progress count fixed via early `file_paths`; `export_estimator`
`is_merged` guard; Positive-mode interaction documented with a hint; folder
loader needs no change (delegates to files); display-name de-dup; white-level
wording clarified.

Round 3 (implementation review): `_load_merged_triplets` clears
`last_merge_error` when the load was cancelled (the worker pool has joined, so
the write has landed) — a cancelled import never pops a stray decode-error
dialog. `load_images_from_folder`'s diagnostic failure count is computed in
triplet units when merge mode is on. The per-worker `last_merge_error` is a
single last-writer-wins slot (accepted: the message is deliberately generic and
the set of loaded images is deterministic regardless).

Round 4 (monochrome support): added monochrome sensors as a first-class merge
input (full-sensor resolution, no CFA — the grayscale frame is the channel). The
Bayer-only guard became a sensor-type classifier (`is_monochrome_sensor`); the
decode branches on it (`_decode_frame_plane`), with `merge_raw_channels(preview)`
running monochrome decodes at half size for fast previews while always reporting
the canonical full size so `original_full_size` stays correct. Mixed-type
triplets are rejected. Reviewed for the full → preview/export → slice/duplicate
resolution round-trip; the only items found were a stale `export_estimator`
comment (fixed) and the mixed-type guard (added). The monochrome rawpy decode
itself is not CI-verifiable here (no monochrome RAW in the repo); it mirrors
`read_image`'s existing monochrome path plus the Bayer absolute-value scaling.

Round 5 (crosstalk-free Bayer): replaced the Bayer `half_size=True` decode with a
direct raw-mosaic read (`extract_cfa_channel`) that pulls only the wanted colour's
photosites — no demosaic, no libraw colour pipeline, never mixing the other
colours — eliminating inter-channel crosstalk. R/B use their single site; green
**averages its two same-colour sites** (not crosstalk — both green — and it keeps
the green SNR). Per-site black subtraction. Validated on the example ARW: R and B
are byte-identical to libraw's single-site read and green matches the two-green
average within rounding. `half_size` (numerically similar, but via libraw's
postprocess) is no longer used for Bayer; monochrome still uses `postprocess` (no
CFA, so no crosstalk to remove).

(During review the green channel was first implemented as a single site — strict
"no merge" — then changed per request to the two-green average, since averaging
two same-colour sites is not crosstalk and preserves green resolution/SNR.)

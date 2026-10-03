# Monochrome RAW Interpretation + Single-Channel Export

> Status: refined (round 2 — open questions resolved inline; see "Decisions").
> Related: spec/trichrome-mono-read.md (the mono mosaic read this reuses),
> spec/positive-mode.md (the global decode-toggle pattern this mirrors),
> spec/color-management.md (export colour spaces / ICC embedding).

## Summary

Two halves of one feature:

1. **Interpret RAW as monochrome** — a new checkbox in **Settings → Color
   Management**. When on, every RAW decodes as a monochrome sensor: the whole
   visible mosaic is read, **one luminance sample per photosite**, at full
   sensor resolution — no demosaic, no phase slice, no interpretation of the
   colour-filter pattern the file declares. The image then renders **grey
   end-to-end** (preview, thumbnail, zoom, export).

2. **Single-channel output** — when an image is monochrome, an exported TIFF or
   JPEG carries **1 channel**, not three identical ones, with the metadata that
   tells other software it is greyscale (TIFF `PhotometricInterpretation = 1`
   (BlackIsZero) + `SamplesPerPixel = 1`; JPEG a 1-component JFIF; a **grey**
   ICC profile embedded in both).

"Monochrome" for (2) is any image whose render is monochrome **by declaration**:

- it was decoded under the new toggle, **or**
- it came from a true monochrome sensor (what `read_image` already auto-detects:
  `num_colors == 1`, a grey `color_desc`, or a single-index CFA), **or**
- its per-image **Color Profile** is **Black & White**.

Why the toggle exists: FreeCCR only treats a RAW as monochrome when its
metadata says so. A **mono-converted (debayered) camera** — the filter array
physically removed, but the RAW still claiming RGGB — fails that test, so it is
demosaiced: 3/4 of the real resolution is interpolated away and a false colour
cast is invented from four identical-filter sites. The same read the trichrome
Merge-detail → Monochrome mode already performs is the correct read for such a
body, and it is wanted for **single** RAWs, not just merges.

Why single-channel output: a greyscale file is half to a third the size, and
software that inspects `SamplesPerPixel`/JFIF components (Photoshop's Grayscale
mode, ImageMagick, `identify`, print shops, DAM ingest rules) currently sees
FreeCCR's black-and-white exports as RGB images that merely happen to be
neutral.

## Goals

- Global toggle, staged in the Settings dialog like its neighbours, persisted as
  QSettings `import/mono_raw`, **applied immediately**: loaded images re-decode
  in the new interpretation (adjustments kept, conversion dropped), exactly like
  Positive mode.
- Exactly **one** mono mosaic read in the codebase — reuse
  `ccr_merge.mono_plane_from_mosaic` / `bin2x2`.
- Monochrome images render grey at every resolution (preview, thumbnail, hi-res
  zoom tile, export), so the file always matches what was on screen.
- Single-channel TIFF (16-bit) and JPEG (8-bit) with correct metadata and a grey
  ICC profile whose tone curve matches the chosen export colour space.
- A FreeCCR-written 1-channel file **re-imports** correctly (the non-RAW reader
  already promotes a 2-D array with `GRAY2RGB`).
- Purely additive: with the toggle off and Color Profile = Color, every existing
  decode, render and exported file is bit-for-bit unchanged.

## Non-Goals

- **No per-image toggle and no catalog field.** This is a global interpretation
  of the decode, like Positive mode — not a per-image edit. (The per-image way
  to ask for black and white already exists: Color Profile → Black & White.)
- **No effect on non-RAW files.** A TIFF/JPEG/PNG is already whatever it is; its
  greyscale route is Color Profile → Black & White.
- **No auto-detection of mono-converted bodies.** The metadata does not say —
  that is the whole problem. It is the user's explicit choice.
- **No change to the trichrome merge modes.** `merge_mono` reads three source
  RAWs as mono planes that become **R, G and B of one colour image**; a merged
  image is colour by construction and is never single-channel. The two features
  share a helper, nothing else.
- **No choice of grey weighting.** Rec.601 (`0.299R + 0.587G + 0.114B`), the
  weights `CCRImage._to_grayscale` already uses, so the file matches the preview.
- No 16-bit JPEG (JPEG is 8-bit), no grey CMYK/Lab, no 1-channel linear-merge
  TIFF (that path is a colour trichrome bake).

## UX

Settings → Color Management, a new group between **Negative conversion** and
**Trichrome capture**:

```
  ┌ Monochrome ─────────────────────────────────────────────────────┐
  │ ☐ Interpret RAW as monochrome (no demosaic)                     │
  │   Read every photosite as one luminance sample at full sensor    │
  │   resolution, ignoring the colour-filter pattern the RAW         │
  │   declares — for monochrome sensors, including mono-converted     │
  │   cameras whose RAW still reports a colour filter. On a colour    │
  │   sensor it shows a checkerboard. Loaded RAWs are re-decoded      │
  │   (your adjustments are kept, the conversion is dropped).         │
  │   Monochrome images export as single-channel greyscale TIFF/JPEG. │
  └─────────────────────────────────────────────────────────────────┘
```

- Staged like the other toggles: `_cb_mono_raw`, seeded once in `_init_toggles`
  from `ccr_backend.mono_raw`, committed on **Done** by `_apply_pending` only
  when it differs from the live value → `MainWindow.on_mono_raw_toggled`.
- The handler mirrors `on_positive_mode_toggled`: set the flag, persist it,
  re-decode all loaded images under a wait cursor, release the hi-res zoom
  cache, refresh thumbnails + preview, save the catalog, show a hint
  ("Monochrome RAW interpretation on — RAWs read as greyscale, exports are
  single-channel." / "… off — RAWs read in colour again.").
- No new UI anywhere else. The export dialog is unchanged: the greyscale decision
  follows the image, not an export option.

## Data Model

| Where | What |
|---|---|
| `ccr_backend.mono_raw: bool = False` | The global toggle. Persisted by MainWindow as `import/mono_raw` (missing ⇒ False, so existing installs are unchanged), restored **before** the first load like `positive_mode`. |
| `CCRImage._mono_raw_active() -> bool` | Static, reads the backend singleton lazily inside a `try` — the exact shape of `_positive_mode_active()`, so a live toggle and the decode never disagree and no per-image copy exists. |
| `CCRImage.decoded_mono: bool` | Stamped by **every** decode path: True when that decode produced a monochrome frame (the toggle, or an auto-detected mono sensor), False otherwise (non-RAW, colour RAW, merged). Session state, **not persisted** — a reload re-stamps it, like `profile_signature`. Initialised False in `__init__`. |
| `ccr_processor.renders_monochrome(ccr_image) -> bool` | `bool(getattr(img, "decoded_mono", False)) or getattr(img, "color_profile", "color") == "bw"`. The single predicate behind both the forced-grey render and the single-channel write. It lives in `ccr_processor` because the export writer needs it and `ccr_image` already imports that module at load (so no new import direction and no cycle). **`getattr` with defaults is required, not cosmetic**: the export writer is called throughout the suite with lightweight stubs (`test_color_management._Dummy()`, `test_positive_mode._stub_image`) that have neither attribute, and they must keep reading as colour without being updated. |
| `CCRImage.renders_monochrome()` | Thin method delegating to the above, for readability at the render site. |

Deliberately **no** new ctor kwarg, no catalog key, no `conversion_inputs`
field: nothing here needs to survive a restart independently of the global flag
(`color_profile` already persists on its own).

## Processing

### 1. The monochrome RAW decode (`CCRImage.read_image`, RAW branch)

Read the global flag **once** per call, next to `positive_mode` (same "read
once" rule):

```python
mono_forced = self._mono_raw_active()
```

Inside `with rawpy.imread(file_path) as raw:`, fold it into the existing sensor
detection:

```python
is_monochrome = <existing metadata detection> or mono_forced
```

That one `or` gets three behaviours for free from code already present:

- `positive_decode = positive_mode and not is_monochrome` — the mono read wins
  over Positive mode, exactly as a true mono sensor already does.
- the camera profile (ICC/DCP) is skipped (`... or is_monochrome` in the return
  guard) — a frame with no colour has nothing to profile.
- `no_icc_default` stays False, so the manual `65535/white_level` scaling below
  still runs.

Then a **new branch ahead of** the existing `if is_monochrome:` postprocess:

```python
if mono_forced:
    mosaic = np.asarray(raw.raw_image_visible)
    if mosaic.ndim != 2:            # linear DNG etc. — no mosaic to read
        → fall through to the normal decode (warning logged); the forced-grey
          render still collapses the result, so the user still gets greyscale
    plane = ccr_merge.mono_plane_from_mosaic(mosaic, raw_colors_visible,
                                            black_level_per_channel)
    if preview:
        plane = ccr_merge.bin2x2(plane)
    rgb = np.repeat(np.clip(plane, 0, 65535).astype(np.uint16)[..., None], 3, axis=2)
```

- `raw_colors_visible` / `black_level_per_channel` are each read in their own
  `try` (absent ⇒ None), as `_decode_frame_plane` does.
- **`full_decode_size` must be re-assigned from the mosaic**, not left at the
  `(raw.sizes.height, raw.sizes.width)` captured above: `raw.sizes` describes the
  *postprocess output*, which can differ from the visible mosaic by a few pixels,
  and this branch never calls postprocess. Use the **unbinned** `plane.shape`
  (before `bin2x2`), exactly as `_decode_frame_plane`'s mono branch returns
  `full = plane.shape`. Getting this wrong silently corrupts
  `original_full_size`, which drives zoom and export resolution.
- Three identical channels, not a 2-D array: every downstream stage (field
  correction, slice ops, resize, conversion, adjustments, histogram, QImage
  `Format_RGB888`) expects `(H, W, 3)`. The collapse to one channel happens
  **only at the export writer**.
- Everything after the branch is the existing code, untouched: field correction
  with `mono=True` (the neutral, channel-averaged gain, so the frame stays
  colourless), `_apply_source_ops`, `original_full_size`, the optional
  `max_long_side` downsize, then the `65535/white_level` scale — which is the
  same absolute-value contract as the trichrome mono read (`combine_channels`
  scales by exactly that factor).
- `preview` bins 2×2 rather than skipping sites: all four samples contribute,
  and the canonical `original_full_size` stays the unbinned mosaic size, so
  zoom/export resolutions are reported correctly.

`self.decoded_mono = bool(is_monochrome)` is assigned on the RAW path, and
`False` on the non-RAW and merged paths, before each returns.

### 2. Forced-grey render (`CCRImage.apply_adjustments`)

One line where the profile is resolved:

```python
profile = self.color_profile if color_profile is None else color_profile
if self.renders_monochrome():
    profile = "bw"
```

This reuses **both** existing `_to_grayscale` call sites — the no-adjustments
early return and the tail after curves/areas — so there is no new render code,
and a mono image behaves identically to an explicit Black & White image. Colour
sliders keep working on the data but cannot tint the result (the collapse is
last), which is the point: preview, thumbnail, zoom tile and file agree.

The hi-res zoom cache signature (`image_preview._adjust_signature`) gains
`bool(ccr_backend.mono_raw)` beside `gamma_luminance` / `auto_gain`: it is a
live global read inside `apply_adjustments`, so a toggle must invalidate a baked
tile. (`color_profile` is already in that tuple; `decoded_mono` changes only via
a re-decode, which releases the cache outright.)

### 3. Single-channel write (`ccr_processor.write_export_image`)

The one export chokepoint all three conversion pipelines already share. After
the `max_long_side` resize, branch on `renders_monochrome(ccr_image)`:

```
gray_u16 = cv2.cvtColor(rgb_u16, cv2.COLOR_RGB2GRAY)        # Rec.601, (H, W)
gray_u16, icc = color_management.apply_export_colorspace_gray(gray_u16, target)
TIFF : safe_tifffile_imwrite(path, gray_u16, photometric="minisblack",
                             compression="deflate", iccprofile=icc)
JPEG : cv2.imencode(".jpg", to_8bit(gray_u16), [JPEG_QUALITY, q])  # 1 component
       → inject_jpeg_icc(..., icc) → write bytes
```

- The collapse is **Rec.601 on the encoded values** — identical weights, and the
  same position relative to the tone curve, as `_to_grayscale` in the render, so
  the exported luminance equals the previewed luminance. Because the render
  already forced grey, the three channels are equal and this is a no-op
  selection; it stays well-defined (never an arbitrary channel pick) for any
  path that reaches the writer with a residual tint.
- Verified against the installed cv2 / tifffile / Pillow before writing this
  round, so the implementation has no open unknowns: `cv2.cvtColor` accepts
  `CV_16U` for `RGB2GRAY` and is **exactly** identity on equal channels (the
  Rec.601 weights sum to 1.0); `cv2.imencode(".jpg", <2-D uint8>)` emits a
  **1-component** JFIF (SOF0 component count 1, PIL mode `L`);
  `tifffile.imwrite(..., photometric="minisblack", iccprofile=...)` yields
  `PhotometricInterpretation == 1`, `SamplesPerPixel == 1`, the ICC intact in
  tag 34675, and a 2-D array on read back (PIL mode `I;16`). No new dependency.
- **Only the file-writing path collapses.** The in-memory processing path
  (`output_path=None`, used by the preview/zoom and by
  `test_positive_mode.test_processing_path_honors_bw_profile`) never reaches
  this writer and keeps returning a 3-channel neutral array, so that existing
  contract is untouched.
- The RGB `apply_export_colorspace` is left exactly as it is; the mono branch
  calls `apply_export_colorspace_gray` instead.
- No BGR swap on the mono JPEG path (there are no channels to swap); the colour
  path keeps its `COLOR_RGB2BGR`.

### 4. Grey ICC profiles (`core/color_management.py`)

A monochrome file must not carry an RGB matrix-shaper profile: the data has one
channel, so `rXYZ/gXYZ/bXYZ` describe nothing and a strict CMM rejects the pair.

```python
build_gray_icc(desc, trc_para, wtpt=D50_XYZ, copyright_text=...) -> bytes
GRAY_SRGB_ICC_BYTES      = build_gray_icc("FreeCCR Gray (sRGB tone)",   <sRGB para>)
GRAY_PROPHOTO_ICC_BYTES  = build_gray_icc("FreeCCR Gray (ROMM tone)",   <ROMM para>)
gray_icc_bytes(target) -> bytes
apply_export_colorspace_gray(gray_u16, target) -> (gray_u16, icc_bytes)
```

- `build_gray_icc` reuses the existing `_desc_type` / `_xyz_type` /
  `_para_type3` / `_text_type` encoders and the same tag-table/offset assembly
  as `build_matrix_shaper_icc`. Header differences: data colour space `'GRAY'`
  instead of `'RGB '`. Tags: `desc`, `wtpt`, `kTRC`, `cprt` — `kTRC` is the
  single grey tone curve, carrying the **same** parametric curve the matching
  RGB profile uses, so tone is interpreted identically. (The input-ICC parser
  in this module already reads `kTRC`, so these profiles round-trip through
  FreeCCR's own reader.)
- `apply_export_colorspace_gray` deliberately applies **no matrix**: for
  `"srgb"` the samples pass through unchanged; for `"prophoto"` it re-encodes
  tone only — `romm_encode(srgb_decode(x))` — because a grey profile has no
  primaries and neutral data has no gamut. This sidesteps the question of
  whether the RGB sRGB→ProPhoto matrix preserves neutrality, and it is exactly
  the transform a grey profile can describe.

## Integration Points

| File | Change |
|---|---|
| `core/ccr_merge.py` | None — `mono_plane_from_mosaic` / `bin2x2` are imported as-is. |
| `core/ccr_image.py` | `_mono_raw_active()`; `decoded_mono` init + stamped on every decode path; `renders_monochrome()`; the mono mosaic branch in `read_image`; the `profile` override in `apply_adjustments`. |
| `core/color_management.py` | `build_gray_icc`, the two grey profile constants, `gray_icc_bytes`, `apply_export_colorspace_gray`. |
| `core/ccr_processor.py` | `renders_monochrome(ccr_image)`; `write_export_image` monochrome branch (TIFF minisblack / 1-component JPEG / grey ICC). |
| `core/ccr_backend.py` | `mono_raw` flag; `reprocess_all_for_mono_raw_change` — shares one private `_redecode_all_keep_adjustments` body extracted from `reprocess_all_for_positive_mode_change` (identical contract: keep adjustments, drop conversion, preserve inherited tint balance). |
| `widgets/settings_dialog.py` | The Monochrome group + checkbox; seed in `_init_toggles`; staged commit in `_apply_pending`. |
| `ui/main_window.py` | Restore `import/mono_raw`; `on_mono_raw_toggled`. |
| `widgets/image_preview.py` | `mono_raw` in the hi-res adjustment signature. |

## Edge Cases

- **Colour sensor with the toggle on** → each photosite's filtered response is
  read as luminance, printing a 2×2 checkerboard. Documented in the settings
  text; the user's explicit choice (same stance as the trichrome mono read).
- **No mosaic to read** (linear DNG, some converted files) → the decode falls
  back to the normal path with a logged warning instead of failing the import;
  the forced-grey render still delivers greyscale. No exception reaches the
  loader.
- **True monochrome sensor, toggle off** → now renders forced-grey and exports
  1 channel, where before it exported three identical channels (and could pick
  up a slight tint from the post-invert shadow warmth). Intended: the file now
  matches the sensor and the preview.
- **Color Profile = Black & White on a colour image** → 1-channel export too,
  per the same predicate. The previewed B&W rendition is unchanged.
- **Positive mode + toggle on** → the mono linear decode wins (as it already
  does for true mono sensors); brightness/gamma own the look.
- **Trichrome merged image** → `decoded_mono` False, colour export, regardless
  of `merge_mono` and of this toggle.
- **Linear-merge TIFF export** (`_export_merged_linear`) → untouched, still
  3-channel `photometric="rgb"`.
- **Re-importing a FreeCCR greyscale file** → the non-RAW reader's existing
  `is_gray → GRAY2RGB` promotion handles it; it loads as a normal (neutral)
  image.
- **Toggling with images loaded** → every image re-decodes; conversions are
  dropped (a conversion computed on a demosaiced colour base is meaningless on a
  mono base), adjustments and crop/orientation survive.

## Test Plan

`tests/test_monochrome_raw.py` (Qt offscreen, the file-layout conventions of
`tests/test_trichrome_mono.py`):

1. **Predicate** — `renders_monochrome` truth table: `decoded_mono` alone,
   `color_profile="bw"` alone, both, neither; a merged image stays False.
2. **Decode** (fake `rawpy.imread` object exposing `raw_image_visible`,
   `raw_colors_visible`, `black_level_per_channel`, `white_level`, `sizes`,
   `postprocess`): with the toggle on, the decode is the full mosaic at full
   resolution, three equal channels, `decoded_mono` True, values
   `(mosaic − black) · 65535/white`; `preview=True` is 2×2-binned with
   `original_full_size` still the unbinned size; with the toggle off the same
   file takes the colour path and `decoded_mono` is False. A real-ARW variant
   (skipped if absent) asserts the plane equals
   `ccr_merge.mono_plane_from_mosaic` on the same file.
3. **Fallback** — a raw whose `raw_image_visible` is 3-D does not raise and
   still returns an image.
4. **Forced grey** — `apply_adjustments` on a `decoded_mono=True` image returns
   equal channels even with `temperature`/`saturation`/`balance_*` set; the same
   image with `decoded_mono=False` does not.
5. **Export TIFF** — a mono image exports to `ndim == 2`, `uint16`,
   `PhotometricInterpretation == 1`, `SamplesPerPixel == 1`, ICC tag 34675 ==
   `gray_icc_bytes(target)`, for both `srgb` and `prophoto`; a colour image
   still writes `(H, W, 3)` with the RGB profile (regression).
6. **Export JPEG** — a mono image's JPEG has exactly one component (SOF0 parse,
   plus `PILImage.open(...).mode == "L"` when Pillow is present) and carries the
   grey ICC; a colour image still has three.
7. **Black & White profile** — `color_profile="bw"` on a colour image also
   produces a 1-channel file.
8. **Grey ICC validity** — `icc[36:40] == b"acsp"`, `icc[16:20] == b"GRAY"`,
   a `kTRC` tag is present, the profile parses via `InputProfile.from_bytes`,
   and via `ImageCms.getOpenProfile` when Pillow is available.
9. **`apply_export_colorspace_gray`** — `srgb` is identity; `prophoto` is
   monotone, maps 0→0 and 65535→65535, and matches
   `romm_encode(srgb_decode(x))` within 1 LSB.
10. **Round trip** — re-reading the exported 1-channel TIFF through
    `CCRImage.read_image` yields `(H, W, 3)` neutral data.
11. **Settings dialog** — the checkbox seeds from `ccr_backend.mono_raw`;
    `_apply_pending` calls `on_mono_raw_toggled` only when it changed.
12. **MainWindow handler** — sets the backend flag, persists `import/mono_raw`,
    and calls the re-decode when images are loaded (stub backend).

## Decisions

- **Global toggle, not per-image** — it is an interpretation of the decode, and
  Positive mode is the established precedent (immediate re-decode, adjustments
  kept, conversion dropped). A per-image greyscale *edit* already exists as
  Color Profile → Black & White, so the two do not compete.
- **Reuse the trichrome mono read** rather than a second implementation: the
  black-pedestal-per-site handling, the clip at 0 and the binned preview are
  already specified, tested and shipped there.
- **Fold into `is_monochrome`** rather than adding a parallel flag through the
  RAW branch: the three existing consequences (Positive mode yields, camera
  profile skipped, manual white-level scaling applies) are all correct for this
  mode, and inheriting them keeps the diff small and the behaviour consistent
  with true mono sensors.
- **Three channels in, one channel out** — the pipeline stays RGB-shaped
  (nothing downstream has to learn a new array rank) and the collapse happens at
  the single export writer. The render still forces grey, so the collapse is
  information-free rather than a late conversion.
- **Forced grey rather than "collapse at write only"** — otherwise a colour
  slider could tint a preview that the greyscale file cannot reproduce. Preview
  and file must agree.
- **Grey ICC with the matching TRC, and no matrix for ProPhoto** — an RGB
  matrix-shaper profile is invalid on single-channel data; a grey profile
  describes exactly what a neutral file has (a white point and a tone curve).
- **A `getattr`-based module-level predicate, not a required method** — the
  export writer is exercised all over the suite with partial stubs; demanding a
  real `CCRImage` there would force unrelated test churn and make the writer
  fragile for any future caller that hands it a shell object.
- **Rec.601 weights, applied to encoded values** — matches `_to_grayscale`, so
  the file equals the preview. Photometrically a luminance average belongs in
  linear light, but agreement with what the user approved on screen is worth
  more than that purity, and for already-neutral data the two are identical.

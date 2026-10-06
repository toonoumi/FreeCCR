# Trichrome Merge Detail: Monochrome Read

> Status: refined (open questions resolved inline — see "Decisions" at the end).
> Extends spec/trichrome-demosaic-mode.md and spec/three-way-rgb-merge.md.

## Summary

A third entry in the **Settings → Color Management → Trichrome capture → Merge
detail** dropdown:

- Demosaic (full resolution) — unchanged, default.
- Single photosite (half resolution) — unchanged.
- **Monochrome (full resolution, no demosaic)** — NEW: every source RAW is read
  as a monochrome sensor. The whole visible mosaic is taken as-is, one
  luminance sample per photosite, at full sensor resolution — no demosaic, no
  phase slice, no interpretation of the colour-filter pattern the file
  declares. The red-light frame's plane becomes R, green's G, blue's B.

Why: FreeCCR only treats a RAW as monochrome when its metadata says so
(`is_monochrome_sensor`: `num_colors == 1`, a grey `color_desc`, or a
single-index CFA). A **mono-converted (debayered) camera** — and any body whose
RAW still claims RGGB although the filter is gone — fails that test, so it goes
down the Bayer path: demosaic mode interpolates 3/4 of the data away from one
CFA phase, photosite mode throws 3/4 of it away. Under trichrome capture every
photosite of a monochrome sensor measured the single light, so the right read
is the whole mosaic.

## Goals

- Dropdown entry with the same staged-apply pattern as the other two.
- Captured **per merged image at import** (`CCRImage.merge_mono`), like
  `merge_demosaic`: every re-read (preview, zoom, export, linear TIFF, slice,
  duplicate, catalog restore, IT8 wizard target) reproduces the same decode.
- Full sensor resolution; `preview` decodes a 2×2-binned half-size frame, like
  the demosaic/monochrome previews, with the canonical full size unchanged.
- Absolute-value contract identical to the photosite read: per-site black
  subtraction, clip at 0, then `combine_channels` scales by `65535/white_level`.

## Non-Goals

- No auto-detection of mono-converted bodies (the metadata does not tell us —
  that is the whole problem). It is the user's explicit choice.
- No non-merge monochrome import (a single RAW read as greyscale). This is a
  Merge detail mode only.
- No change to true monochrome sensors under the other two modes (they already
  decode at full resolution via the existing mono path).
- No flat-field / per-site gain correction of residual CFA response. On a
  **colour** sensor this mode reads the colour filter's response under each
  light as luminance, which prints a 2×2 checkerboard — the settings text says
  it is for monochrome sensors only.

## UX

```
   Merge detail:  [ Demosaic (full resolution)                  ▾ ]
                  [ Single photosite (half resolution)            ]
                  [ Monochrome (full resolution, no demosaic)      ]
   <muted> …existing text… Monochrome reads every photosite as luminance with
           no demosaic — for monochrome sensors, including mono-converted
           cameras whose RAW still reports a colour filter. On a colour sensor
           it shows a checkerboard. Applies to the next import.
```

- Combo data becomes a string mode: `"demosaic"`, `"photosite"`, `"mono"`.
- Seeded from the backend: `"mono"` if `rgb_merge_mono`, else `"demosaic"` /
  `"photosite"` from `rgb_merge_demosaic`.
- On Done, a changed mode calls `MainWindow.on_rgb_merge_detail_changed(mode)`,
  which sets both backend flags, persists both QSettings keys, and shows the
  "applies to the next import" hint. `on_rgb_merge_demosaic_changed(bool)`
  stays as a thin wrapper (it is the old public handler).
- IT8 wizard target label shows `Merge mode: monochrome read`.

## Data Model

- `ccr_backend.rgb_merge_mono: bool = False`; QSettings `import/rgb_merge_mono`
  (missing ⇒ False, so existing installs are unchanged).
- Mode → flags: `demosaic` ⇒ (demosaic=True, mono=False); `photosite` ⇒
  (False, False); `mono` ⇒ (mono=True; the demosaic flag is left as it was —
  it is ignored while mono is set, and keeping it means switching back
  restores the previous Bayer choice).
- `CCRImage(..., merge_mono: bool = False)` → `self.merge_mono`. Threaded
  everywhere `merge_demosaic` is: loader → `create_images_for_merge`,
  duplicate, slice, `_export_merged_linear`, catalog serialise/restore/re-key,
  `it8_profile.decode_target_merged`, the IT8 dialog's current-merge tuple.
- Catalog record gains `"merge_mono"` (missing ⇒ False). Restore prefers the
  live global setting over the stored copy, exactly like `merge_demosaic`.
- **Additive, not a replacement**: `merge_demosaic` keeps its bool meaning and
  `mono` overrides it. Every old catalog, QSettings value and call site stays
  valid; new keyword params are appended at the END of each signature
  (`merge_raw_channels`, `create_images_for_merge`, `_restore_image`,
  `decode_target_merged`, `CCRImage.__init__`) so positional callers are safe.

## Processing

`ccr_merge.mono_plane_from_mosaic(mosaic, colors=None, black_levels=None)` —
pure:

```
plane = float32(mosaic)                     # 2-D only; a 3-D (linear DNG) raises
if black_levels:
    lut = float32(black_levels)
    plane -= lut[0]                if colors is None or all levels equal
    plane -= lut[clip(colors)]     otherwise   # per-site pedestal, no 2x2 print
plane = max(plane, 0)
```

`ccr_merge.bin2x2(plane)` — 2×2 box average, odd trailing row/col dropped.

`_decode_frame_plane(path, frame_pos, preview, demosaic=False, mono=False)`:
when `mono`, BEFORE any sensor detection or guard: read
`raw.raw_image_visible`, `raw.raw_colors_visible`,
`raw.black_level_per_channel`; `plane = mono_plane_from_mosaic(...)`;
`plane = normalize_cfa_phases(plane)`; `full = plane.shape`;
`plane = bin2x2(plane)` if preview; return
`(plane, white_level, True, full)`. Reporting `is_mono=True` makes the
existing full-size rule (`sensor_full if any_mono or demosaic`) and the
mixed-sensor guard (all frames mono) apply unchanged.

**Why `normalize_cfa_phases` belongs here** (it was missing until the fix that
added this paragraph, and the artefact was reported from the field): stripping a
CFA is never perfect, so each of the four photosite phases keeps a slightly
different gain — measured 1577.7 / 1357.6 / 1350.0 / 1496.8 on a converted body,
a 15.75 % spread. Monochrome is the **only** merge detail that makes every
photosite its own pixel, so adjacent pixels come from *different* phases and the
spread becomes spatial: one strongly bright site per 2×2 quad (R, +9.1 %) on a
2 px lattice, which a smooth zoom upscale renders as a mesh of light lines
around darker quads. Photosite (one phase per quad) and Demosaic (each plane
interpolated from its own phase) turn the same spread into a flat per-channel
gain that white balance absorbs, which is why only this mode shows it. All three
sources share the body, so untreated the mesh is **neutral** — a white lattice,
not a colour cast.

Normalisation is per frame, so the three sources may land on marginally
different corrections; that difference is a single scalar per frame, i.e. a flat
per-channel gain, absorbed by white balance. It must run **before** `bin2x2`:
binning averages the four phases, which is exactly why a half-size preview hid
this while full-resolution zoom and export showed it.

This presumes a **converted** body, which is what the mode is for. On a live CFA
the per-phase spread is real scene colour (the example `DSC07096.ARW` measures
47.2 %: R 1005 / G 1247 / G 1247 / B 746) and normalising it is meaningless —
but so is reading a live mosaic as luminance in the first place, so that
combination was already garbage and is not newly broken.

`merge_raw_channels(sources, preview=False, demosaic=False, mono=False)`
forwards the flag.

No sensor-type guards in this mode: a mono-converted X-Trans or 4-colour body
reads the same way (the CFA layout is irrelevant once it is ignored). A file
with no mosaic (linear DNG) raises a ValueError naming the file, which the
loader already surfaces as a merge failure.

## Integration Points

| Where | Change |
|---|---|
| `ccr_merge.py` | `mono_plane_from_mosaic`, `bin2x2`; `mono` on `_decode_frame_plane` + `merge_raw_channels`; docstrings. |
| `ccr_image.py` | Ctor kwarg + attribute; `_read_merged` forwards it. |
| `ccr_backend.py` | `rgb_merge_mono`; loader, duplicate, slice, `_export_merged_linear`. |
| `catalog.py` | Serialise/re-key/restore `merge_mono`; `create_images_for_merge` + `_restore_image` params. |
| `it8_profile.py` | `decode_target_merged(..., mono=False)`. |
| `it8_profile_dialog.py` | Current-merge tuple `(sources, demosaic, mono)`; target mono flag; label. |
| `settings_dialog.py` | Third combo entry, string data, seed/apply; muted text. |
| `main_window.py` | Restore `import/rgb_merge_mono`; `on_rgb_merge_detail_changed`; current-merge tuple. |

## Edge Cases

- **Toggling with merges loaded** → no effect on them (per-image capture).
- **Colour sensor in Monochrome mode** → checkerboard (documented, user choice).
- **True mono sensor in Monochrome mode** → same data as the existing mono
  path, minus libraw's pipeline; full resolution either way.
- **Linear-TIFF replace** → bakes the mono read; the TIFF is a plain RGB file.

## Test Plan

`tests/test_trichrome_mono.py`:

1. **Pure read**: `mono_plane_from_mosaic` keeps full resolution, subtracts a
   per-CFA-index pedestal exactly (a mosaic of `black[idx] + v` returns `v`
   everywhere), clips at 0, rejects a 3-D array; `bin2x2` averages and drops
   the odd edge. **Phase lattice**: a flat scene carrying the measured
   converted-body per-phase gains decodes through `_decode_frame_plane(mono=True)`
   to a plane whose four phase means agree to <1e-5 relative, with the frame's
   overall level unchanged (a redistribution, not a gain).
2. **Real RAW** (example ARW, skipped if absent): `merge_raw_channels([arw]*3,
   mono=True)` returns the full visible mosaic size, `full_size ==
   merged.shape`, all three channels identical (same file, same plane);
   `preview=True` is half size with the same `full_size`; the same pixel equals
   `(mosaic − black) · 65535/white`.
3. **Threading** (monkeypatched `merge_raw_channels`): the ctor captures and
   forwards `mono`; the default is False; duplicate inherits it; the loader
   stamps `rgb_merge_mono`; linear export forwards it; the catalog round-trips
   it and a record without the key restores as False; `decode_target_merged`
   forwards it.
4. **Settings dialog**: seeding picks "mono" from the backend; choosing a
   different entry calls `on_rgb_merge_detail_changed` with the mode string,
   and not at all when unchanged.

## Decisions

- **Additive flag, not an enum migration**: `merge_demosaic` already persists in
  catalogs and QSettings; a separate `merge_mono` that overrides it needs no
  migration and keeps every existing record meaning what it meant.
- **Ignore the declared CFA entirely** (no guards): the feature exists because
  the metadata lies about the sensor; trusting any part of it would defeat it.
- **Per-site black subtraction**: libraw reports per-CFA-index pedestals; they
  are normally equal, but subtracting one scalar would print a 2×2 pattern if
  they are not. Equal levels take the cheap scalar path.
- **Binned preview**: keeps the preview/zoom decode as cheap as the other
  full-resolution modes and uses all four sites rather than skipping three.

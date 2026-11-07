# Anamorphic De-squeeze

A new top-level **Edit** menu with **Anamorphic De-squeeze ▸**, a submenu of
common squeeze factors plus **Custom…**, so frames shot through an anamorphic
lens are displayed and exported at their intended (wider) aspect ratio.

## Goals

- Per-image factor `desqueeze` (float, `1.0` = off) on `CCRImage`.
- The **preview**, the **thumbnails** and **every export** show the frame
  stretched horizontally by that factor.
- Set from the menu on **all selected thumbnails** (falls back to the image in
  the preview when nothing is selected) — anamorphic is a per-roll property, so
  one click should cover a selection.
- Persisted in the catalog, captured by undo, carried by duplicates / slices /
  tether templates, and offered as its own **Copy / Sync** group.

## Non-goals

- No vertical-squeeze mode (shrinking height to keep the pixel count). Stretching
  width is what every de-squeeze tool does and loses no resolution.
- No pixel re-processing: like rotation/mirror it is a display-level geometric
  transform. The working image (`resized_raw`), histogram, conversion inputs and
  every sampling coordinate are untouched.
- Crop aspect presets are not de-squeeze-aware (a 2.39 preset constrains the
  *squeezed* box). Overlay glyphs (crop handles, dust circles) are drawn in the
  stretched view, so round glyphs read slightly elliptical. Both acceptable for v1.

## UX

```
Edit
└─ Anamorphic De-squeeze ▸
     ● None (1.0×)
     ○ 1.2×   ○ 1.25×  ○ 1.33×  ○ 1.4×  ○ 1.44×  ○ 1.5×
     ○ 1.55×  ○ 1.6×   ○ 1.65×  ○ 1.8×  ○ 2.0×
     ─────────
     ○ Custom…          (label shows "Custom (1.37×)…" when active)
```

Exclusive check marks, refreshed on `aboutToShow` from the current image.
**Custom…** opens a `QInputDialog.getDouble` (1.00–3.00, 2 decimals, seeded with
the current factor). A selection push one undo snapshot per changed image.
The presets cover 1.33× (4:3 → 16:9 adapters), 1.5/1.55/1.6/1.65/1.8× (modern
still/cine anamorphics) and 2.0× (classic CinemaScope / Kowa / Hawk).

## Where it applies — the axis

The factor stretches the **displayed** horizontal axis, i.e. AFTER rotate →
fine-rotate → mirror (`apply_orientation`). A frame scanned sideways and turned
upright is stretched along its now-horizontal width, which is what the lens
squeezed. Mirrors commute with an axis-aligned scale, so its position relative
to them is free.

## Implementation

| Surface | Change |
| --- | --- |
| `CCRImage` | `desqueeze=1.0` ctor kwarg + attribute; in `capture_undo_state` / `pop_undo_state` (`.get(..., 1.0)` for old snapshots). |
| `ccr_processor.apply_desqueeze(img, factor)` | `cv2.resize` width → `round(w·factor)`, Lanczos; float inputs clipped to the input range (no ringing outside it). Identity when factor ≈ 1. |
| `write_export_image` | Calls `apply_desqueeze` FIRST (before the max-long-side resize, so the limit bounds the final file). It is the single chokepoint all four pipelines go through after orientation. |
| `ImagePreview` | `current_desqueeze`; the stretch lives in the **view** transform (`scale(s·k, s)`), not in the items — every `mapToScene`/item transform/sampling path stays correct by construction. `_fit_view_to_content`, `_compute_fit_scale` fit the stretched width; `_view_scale` reads `m22` (vertical = the unstretched scale), so zoom %, handle sizes and tolerances keep meaning "per source pixel". Zoom uses uniform `scale(f, f)` and so preserves the anisotropy. |
| `ThumbnailList.apply_frontend_transformations` | `QTransform.scale(k, 1)` after the mirrors. |
| `catalog` | `"desqueeze"` written always, read with default `1.0` (additive, no migration); `_is_pristine` treats `!= 1.0` as an edit. |
| `ccr_backend` | duplicate / slice children / reset-to-parent carry it; `reset_image` (back to import) resets it to 1.0. |
| tether template (`main_window`) | carried like orientation. |
| `SYNC_GROUPS` | new `("desqueeze", "Anamorphic De-squeeze", ())` after `orientation`; Sync to All / Copy / Paste handle it like orientation (display-only: no reprocess). |

## Test plan

- `apply_desqueeze`: shape (w·k), identity at 1.0, uint16 dtype kept, float
  output within the input range, flat image stays flat.
- `write_export_image` with a stub image at 1.5× writes a 1.5× wider file, and
  with `max_long_side` the long side is bounded after the stretch.
- Undo: setting the factor then `pop_undo_state` restores it; old snapshots
  without the key restore 1.0.
- Catalog round trip preserves 1.44; a state with only a de-squeeze is not pristine.
- Sync group: copy/paste and sync carry the factor and nothing else.
- Menu: Edit menu exists with the preset list and Custom; triggering a preset
  sets the selected images.

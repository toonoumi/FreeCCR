# Full-resolution zoom

## 1. Goal

When the user zooms a converted (or positive-mode) image to 100%, the pixels on
screen must be the source's **real** pixels — one source pixel per screen pixel,
decoded and converted at full resolution. Today 100% shows a half-resolution
decode interpolated 2x, which reads as soft.

### 1.1 What exists today

Zoom detail is already asynchronous and color-matched. `ImagePreview` keeps a
single full-frame hi-res tile (`self._hires`), requested by
`_maybe_request_hires` whenever `_zoomed_in_enough()`, rendered off the GUI
thread by `HiResDetailWorker`, and displayed under identical scene geometry via
`_item_prescale = current_pixmap.width() / tile.width()`.

The gap is one line in `CCRImage.render_hires_base`:

```python
img = self.read_image(self.file_path, preview=True, max_long_side=max_long_side)
```

with `max_long_side = ImagePreview.HIRES_MAX_LONG_SIDE = 4500`.

* `preview=True` means rawpy `half_size=True` on every RAW branch (colour via
  `_raw_color_postprocess_kwargs`, the monochrome mosaic read, and
  `_read_merged`). An 8256px-wide RAW decodes to 4128px — **half** the linear
  resolution.
* Non-RAW decoders ignore `preview`, so a scan is bounded only by the 4500 cap.

So the tile's resolution is fixed regardless of zoom: at 100% on a 45 MP RAW
every displayed pixel is interpolated from a 4128px decode, and at 200-400%
(`MAX_PERCENT = 4.0`) it is worse.

### 1.2 Non-goals

* **Viewport-region (tiled) rendering.** The tile stays whole-frame. Region
  decoding would need sub-rect scene placement, a re-render on every pan, and
  region-aware mapping for the frame-normalized crop/dust/area geometry.
* No change to the 1080px preview, to conversion math, or to export (exports
  already decode full-size via `_load_export_source`).
* No new zoom ceiling. `MAX_PERCENT = 4.0` stays; at 400% the full-res tile is
  interpolated 4x, which is correct.

## 2. UX

Invisible by default, except that zooming in gets sharper. Progressive by
construction: the existing tile stays on screen while the larger render runs, so
the sequence is preview -> half-res tile -> full-res tile with no blank frame
(see §5.3).

**Settings -> General -> Zoom** gains one checkbox, "Load full resolution at
100% zoom" (default **on**, QSettings key `view/full_res_zoom`), with muted
help text noting the memory/time cost. Turning it off restores exactly today's
behaviour and frees any oversized cached tile.

## 3. Resolution rule

### 3.1 How many pixels the zoom needs

The scene is in preview-pixmap pixel units and the tile is prescaled onto the
preview's scene footprint, so a tile is being magnified exactly when it has
fewer pixels across than the footprint covers on screen. Required *cropped*
tile width is `_view_scale() * current_pixmap.width()`; since the tile and the
preview carry the **same** normalized crop, the cropped/full ratio is equal for
both, and the requirement on the full-frame tile reduces to

```
want_long = _view_scale() * preview_full_long
```

With `_preview_to_source_ratio() = preview_full_long / source_long` and
`_current_percent() = _view_scale() * ratio`, this is exactly

```
want_long = _current_percent() * source_long          # source pixels
```

100% -> `source_long` (true full resolution); 50% -> half. The formula needs no
crop special-case: a confirmed crop raises `_view_scale()` (the kept region is
scaled up to fill the view), so crop magnification asks for more pixels on its
own — which is what `_crop_wants_hires()` already wants.

### 3.2 Choosing the decode

Mirrors `_load_export_source`'s half/full decision, and is **floored at today's
tile size so no zoom level can ever regress**:

```
half_long    = source_long // 2   if the decode honours half_size else source_long
legacy_floor = min(half_long, legacy_cap)          # legacy_cap = 4500
target       = clamp(want_long, legacy_floor, source_long)
preview      = (half_long >= target)               # half-size decode suffices
```

`half_size` is honoured by RAW extensions and by merged images; non-RAW ignores
it, hence `half_long = source_long` there, which makes `legacy_floor` equal
`min(source_long, 4500)` — precisely today's effective tile for a scan.

Worked cases (8256px RAW, 10000px TIFF):

| Situation | want | target | decode |
|---|---|---|---|
| Fit / dust mode, RAW | small | 4128 (floor) | `preview=True`, cap 4128 — same as today |
| 100% zoom, RAW | 8256 | 8256 | `preview=False` — **full res** |
| 60% zoom, RAW | 4954 | 4954 | `preview=False`, downsized in-reader |
| 100% zoom, TIFF | 10000 | 10000 | cap 10000 (no downsize) — full res |
| Toggle off | — | — | `preview=True`, cap 4500 — unchanged |

Only one escalation per image is possible (full res is the ceiling), so wheel
steps cannot thrash between tiers.

`max_long_side` is always passed; at `target == source_long` it is a no-op
(`resize_image_to_max_pixel` returns the input unchanged). Requesting the
downsize *inside* the reader keeps the RAW saving described in `read_image`'s
docstring (downsize before white-level scaling).

### 3.3 Colour match

Nothing about the conversion changes. `render_hires_base`'s replays
(`apply_reference_normalization`, `apply_bwpoint_normalization`) are
resolution-independent point ops replayed from the convert-time
`conversion_inputs` snapshot — that is the existing colour-match guarantee
(`test_zoom_hires.py::test_hires_scales_consistently`), and it holds at full
resolution for the same reason it holds at half.

## 4. Data model

* `CCRImage.hires_decode_request(want_long, legacy_cap)` -> `(preview, max_long_side)`
  implements §3.2. It lives on the model because the half-size semantics belong
  to `read_image`; the widget owns only the cap and the zoom measurement.
* `CCRImage.render_hires_base(..., preview=True)` — new keyword appended at the
  **END** of the signature (positional-compat rule: the worker calls it by
  keyword, tests call it with no args).
* `ccr_backend.full_res_zoom: bool = True` — global display flag, persisted by
  `MainWindow`, like `gamma_luminance`.
* `ImagePreview._hires` gains `"res"`: the `target` the cached tile was rendered
  at. Read with `.get("res", 0)` so externally-constructed cache dicts (tests)
  still work.
* `HiResDetailWorker` gains `req_res` (for in-flight dedup) and a `preview`
  flag, both snapshotted at construction like every other field.

## 5. Integration points

### 5.1 Request

`_maybe_request_hires` computes `target` via a new
`_hires_target_long_side()` helper, then:

* cache matches image+sig, `cache["res"] >= target`, `adj_sig` matches,
  `full_pm` present -> display it, return (unchanged).
* cache matches image+sig, `cache["res"] >= target`, `adj_sig` differs -> reuse
  `base`, re-adjust only (unchanged).
* cache matches image+sig but `cache["res"] < target` -> the cached base is too
  small: request with `base=None` so it is re-decoded at `target`.
* in-flight dedup additionally requires `w.req_res >= target`.

### 5.2 Zoom-out

A tile sharper than needed is **kept** — resolution is never downgraded on
zoom-out, matching the existing "keep detail cached until the image is
switched" behaviour and avoiding a pointless re-decode on re-zoom.

### 5.3 Progressive refinement

Escalation deliberately does not clear `full_pm`, so the half-res tile stays
displayed until the full-res one lands. `_on_hires_ready` then swaps it in and
stores the new `res`.

### 5.4 Settings

* `ccr_backend.full_res_zoom` default in `CCRBackend.__init__`.
* `MainWindow` startup restore (`view/full_res_zoom`, default True).
* `MainWindow.on_full_res_zoom_toggled` — set flag, persist, then re-evaluate
  the current view: on **on**, `_update_hires_state()` requests the sharper
  tile; on **off**, `_release_hires()` frees the oversized one. Display-only, so
  no `_rerender_all_for_global_mode`, no re-conversion, no catalog write (like
  `on_warn_no_anchor_toggled`).
* `SettingsDialog`: checkbox in a new "Zoom" group, entry in the
  `_sync_from_backend` tuple, and an `_apply` branch calling the handler.

The flag is **not** part of `_current_adj_sig()` — it changes resolution, not
the look, and resolution is tracked by `"res"`.

## 6. Cost

A full-res tile for a 45-60 MP frame holds the whole frame: ~270-360 MB for the
uint16 base, ~135-180 MB for the 8-bit display, plus the QPixmap. One image is
cached at a time and it is freed on image switch (`_release_hires`) or when the
toggle goes off. The full RAW decode costs ~2-4 s on a worker thread, with the
previous tile visible throughout. This cost is the accepted trade for true
100% pixels; the toggle is the escape hatch.

## 7. Test plan

`tests/test_full_res_zoom.py`, extending the headless-`ImagePreview` pattern of
`test_zoom_percent.py` and the stub pattern of `test_zoom_hires.py`.

Decode-request rule (`hires_decode_request`, pure — no Qt, no I/O):
1. RAW at 100% -> `preview=False`, `max_long_side == source_long`.
2. RAW at fit/dust (tiny want) -> `preview=True`, `max_long_side == source_long // 2`
   (the floor; today's resolution, no regression).
3. RAW at 60% -> `preview=False` with the exact intermediate target.
4. Non-RAW at 100% -> `max_long_side == source_long`; at fit -> the 4500 floor.
5. `want_long` above `source_long` (200-400% zoom) clamps to `source_long`.
6. Toggle off -> `(True, legacy_cap)` exactly.

Zoom measurement (`_hires_target_long_side`, headless preview):
7. At `zoom_to_percent(1.0)` the target equals `source_long` (ties §3.1's
   derivation to the real geometry).
8. A heavy crop at the fitted view raises the target above the floor.
9. Returns the legacy request when `original_full_size` is unknown.

Cache/escalation (stubbed worker, no threads):
10. Cached `res` below target -> a new request is made with `base=None`.
11. Cached `res` at/above target with matching `adj_sig` -> no new worker.
12. Zooming out after a full-res render does not downgrade or re-request.
13. In-flight worker with `req_res >= target` suppresses a duplicate request.

Plumbing:
14. `render_hires_base(preview=False)` forwards `preview=False` to `read_image`
    (monkeypatched capture), and the default stays `preview=True`.
15. Existing `test_zoom_hires.py`, `test_zoom_percent.py` and
    `test_crop_hires.py` keep passing unchanged — including
    `test_crop_hires.py`'s `_inject_hires`, which builds a cache dict with no
    `"res"` key.

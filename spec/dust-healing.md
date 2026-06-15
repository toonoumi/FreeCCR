# Spec: Dust Healing (Spot & Scratch Removal)

Status: REFINED v2
Owner: FreeCCR
Feature branch: `feature/dust-healing`

> §9 (Refinement) resolves open questions and **overrides** the v1 body where
> they conflict — it is the authoritative source for the coordinate transform
> (§9.1), brush sizing (§9.4), the inpaint solver (§9.2), tuned constants
> (§9.7), and the added Visualize mode (§9.3).

## 1. Summary

Add a **Dust Healing** tool that removes the dust spots and scratches a DSLR
"scan" of film leaves behind — small bright blobs and thin curly bright strings
in the converted positive (e.g. white specks on a blue sky). The tool offers
**automatic detection** of all spots in the current image, lets the user
**reject any individual auto-detection** with a click, and lets the user **add
heal strokes manually** (click a missed spot, or drag-trace a curly scratch).
The healing is stored as a resolution-independent list of vector "strokes" on
the image and replayed through the existing `apply_adjustments` hook so it shows
identically in the preview, the hi-res zoom detail, and the exported file. The
controls live in the top toolbar of `ImagePreview`, next to the other
canvas-mode tools (Crop, Areas).

## 2. Goals / Non-goals

### Goals
- A **Heal** toolbar toggle that enters/exits a dust-healing canvas mode (like
  Crop / Area modes), gated to **converted** images.
- An **Auto-Detect** toolbar button that finds all dust spots/scratches in the
  current converted image and adds them as heal strokes.
- A **sensitivity** control (Low / Medium / High) governing auto-detection.
- A **Visualize** toggle that shows a high-contrast "find the dust" view while
  curating (Lightroom-style), since dust is near-invisible at fit zoom (§9.3).
- In heal mode, an **overlay** marks every heal stroke (auto vs. manual tinted
  differently).
- **Reject a single detection**: left-click a marked spot to remove that stroke.
- **Manual heal**: left-click an unmarked spot to heal it; left-drag to paint a
  stroke along a curly scratch. A **brush size** control (S / M / L) sets the
  radius.
- A **Clear Spots** toolbar button that removes all heal strokes for the image.
- Healing affects the preview, the hi-res zoom detail, and the export, at every
  resolution (resolution independent), applied **after** negative inversion and
  **before** the user's tonal adjustments and crop.
- Heal strokes persist in the catalog and participate in Undo.

### Non-goals
- **Not** a general content-aware fill / object-removal tool. The fill targets
  small spots and thin scratches over locally smooth-ish backgrounds (sky,
  smooth skin/fabric, defocused areas, uniform film grain). It is **not** meant
  for dust sitting on fine texture (grass blades, foliage, bark, fabric weave,
  text) where a synthesized patch reads as a smudge — heuristic: if the
  background's texture wavelength is smaller than the defect, the fill will look
  artificial. Large/textured defects are out of scope.
- **Auto-detection is inherently imperfect** — with no IR channel, bright/dark
  *real* scene detail (specular highlights, snow, white flowers, distant birds,
  text, stars) can be flagged as dust, and faint dust can be missed. The
  intended workflow is auto-detect → **curate** (reject false positives, add
  misses). The tool optimizes for *easy curation*, not perfect recall/precision.
- No infrared (Digital ICE / iSRD) channel — DSLR scans have no IR, so detection
  is purely from the RGB positive (this is the fundamental constraint; see §5).
- No clone-source picking / aligned-healing brush (always synthesized fill).
- No cross-image "detect dust in all frames" batch and no Sync-to-All for heal
  strokes — dust is physical to one captured frame, so strokes are per-image
  only (see §9 once refined).
- No GPU/OpenCL path; detection + tiled inpaint on the ≤1080 preview (and tiled
  at export) are fast enough on CPU.
- Healing is offered only on **converted** images (dust is meaningful on the
  positive; the un-converted negative is the wrong space and the tool is
  disabled there, like the Crop / Area tools).

## 3. UX / Interaction

### 3.1 Toolbar placement
Four controls are added to the existing `ImagePreview` toolbar
(`image_preview.py:588-703`), inserted **after the "Un-convert" action
(`:668-671`) and before "Export…" (`:673`)**, each followed by `add_spacer()`,
matching the existing toolbar idiom (`QAction` + `triggered.connect`):

1. `self.heal_action` — checkable `QAction` "🩹 Heal" (text-only, like
   "◯ Area"). Toggles dust-healing mode.
2. `self.auto_dust_action` — `QAction` "Auto-Detect". Runs detection on the
   current image.
3. `self.clear_dust_action` — `QAction` "Clear Spots". Removes all strokes.
4. A compact `QComboBox` **brush size** (`S / M / L`) and a compact `QComboBox`
   **sensitivity** (`Low / Med / High`), inserted with the same
   `insertWidget`/spacer idiom used by the zoom combo (`:679-701`).

All five are enabled only when the current image is converted (wired into the
existing `_update_unconvert_action_state()` enable/disable pass). `Auto-Detect`,
`Clear Spots`, the brush combo, and the sensitivity combo are further enabled
only while **Heal mode** is active (entering heal mode reveals the working
controls; leaving it greys them).

### 3.2 Entering / leaving heal mode
`toggle_heal_mode(checked)` mirrors `enter_crop_mode` / `enter_area_mode`
(`image_preview.py:2040-2078`, `2530-2551`):
- On enter: cancel conflicting modes (crop, area, slice, bwpoint, wb-pick),
  set `self.heal_mode = True` and `self.view.heal_mode = True`, set the canvas
  cursor to a **brush-ring cursor** (a crosshair plus a live ring overlay sized
  to the current brush radius — see §3.5), draw the stroke overlay
  (`_draw_dust_overlay()`), and show a one-line hint via the existing
  `sliders_panel` hint channel ("Click a marked spot to remove it; click or drag
  to heal; Auto-Detect finds spots automatically").
- On leave: clear `heal_mode`, restore the arrow cursor, remove the overlay and
  the brush ring. Any external `update_preview` (image switch, slider change,
  conversion, undo) leaves heal mode, exactly as crop/slice/area modes do via
  the guards at `update_preview` (`:875-894`).

### 3.3 Coordinate systems
Heal strokes are authored on the displayed canvas but **stored in the same
normalized, un-rotated / un-flipped / un-cropped `resized_raw` image space as
`crop_rect`** (`ccr_processor.py:1303-1308`). **See §9.1 — the chain below is
superseded** to also invert *fine rotation* (the v1 `_base_transform` chain
handles only coarse 90°/flip). The authoritative chain uses the displayed
`pixmap_item`'s own transform (coarse + fine rotation), reconstructed in preview
space without prescale:

1. viewport px → `view.mapToScene(event.pos())` → scene.
2. scene → `_image_transform().inverted()[0].map(...)` → displayed-pixmap-local
   — where `_image_transform()` is the coarse base + fine rotation built exactly
   as `apply_transformations` builds `img_transform` (`image_preview.py:1100-1121`),
   minus the prescale (scene relates to preview-local by `img_transform`, since
   the prescale maps hi-res-local → preview footprint).
3. local → `map_displayed_to_full(x, y)` (`:1970`) → full-image px (undoes the
   display crop).
4. full px → normalized: `nx = x / W`, `ny = y / H`, where `(W,H)` are the full
   image dims (`current_pixmap` size when uncropped == `resized_raw` dims).

The **radius** is stored normalized to the image's long side
(`r_norm = r_px / max(W, H)`) so it scales correctly to any export resolution.

For the **overlay** (drawing stored strokes back onto the canvas) the inverse is
used, mirroring the reference-frame overlay at `update_preview:982-993` but with
`_image_transform()` (not coarse-only) so markers track fine rotation: build the
marker geometry in preview-local via `map_full_to_displayed` (new helper =
inverse of `_crop_display_transform`), create the `QGraphicsItem`, and
`setTransform(_image_transform())`. Markers scale with the displayed image
(showing true heal footprint) with a bi-color cosmetic outline pen for
visibility on any background.

### 3.4 Mouse behaviour (heal mode)
Handled by a new branch in `GraphicsImageView.mousePressEvent` /
`mouseMoveEvent` / `mouseReleaseEvent` (`:127-410`), placed **before** the
generic reference-draw branch (`:181`), guarded by `self.heal_mode`:

- **Left-press**: convert to full-image px (§3.3) and hit-test existing strokes
  (`HEAL_HIT` = the stroke's draw radius + 6 screen px, scaled by `_view_scale`).
  - **On a stroke** → remove that stroke (reject detection / delete manual),
    push one undo step, reprocess. No drag starts.
  - **On empty canvas** → begin a new **manual** stroke: start `points=[p]`,
    `_heal_drag = True`.
- **Mouse-move with `_heal_drag`** → append the current point to the live stroke
  when it is ≥ (brush radius) from the last point (throttle), and update a live
  ghost overlay (a polyline of discs). No reprocessing during the drag.
- **Left-release with `_heal_drag`** → commit the stroke
  (`source="manual"`, `connect = len(points) > 1`, `radius` = current brush),
  push one undo step, reprocess.
- **Right-press on a stroke** → remove it (explicit alternative to left-click
  toggle). **Right-press on empty** → ignored.
- **Middle-drag** still pans (the existing `_mid_pan` guard at `:133-147`
  already lists the interaction flags; add `heal_drag` there).

A single click with no movement commits a one-point stroke (`connect=False`) —
the manual "heal this missed spot" action.

### 3.5 Brush ring + cursors
While in heal mode, a `QGraphicsEllipseItem` ring follows the cursor at the
brush radius (drawn in `mouseMoveEvent`, like the slice ghost line), with a
bi-color (white-over-black) cosmetic pen so it is visible on bright sky or dark
frames. **See §9.4** — the brush is sized in **long-side-normalized** units
(image-relative), not fixed preview pixels, so it is stable across zoom and
across source resolutions: `S/M/L = 0.004 / 0.008 / 0.016` of `max(W,H)`. The
ring is drawn in image space so it scales naturally with the displayed image.
Changing the brush updates the ring immediately and applies to the **next**
stroke; the radius of an in-progress drag is fixed at drag start.

### 3.6 Auto-Detect button
`auto_detect_dust()` calls `ccr_backend.detect_dust_by_index(idx, sensitivity)`,
which runs `dust_removal.detect_dust(img.resized_raw, ...)` on the converted
positive preview, **replaces the existing auto strokes** (keeps manual strokes),
stores the merged list on the image, pushes one undo step, reprocesses, and
saves the catalog. A busy hint is shown for large images. If heal mode is off it
is auto-entered so the user immediately sees and can curate the results.

### 3.7 Clear Spots button
`clear_dust()` removes **all** strokes (auto + manual), pushes one undo step,
reprocesses, saves. (A future enhancement could offer "clear auto only".)

## 4. Data model

### 4.1 Storage
A new dedicated attribute on `CCRImage` (NOT inside `adjustment_settings`, to
avoid the slider-rebuild merge hazard that the Curves spec calls out in its
§4.2):

```python
# CCRImage.__init__
self.dust_heal_strokes: list[dict] = []   # [] = nothing to heal
```

Each stroke is a JSON-clean dict:

```python
{
    "points": [[nx, ny], ...],  # normalized 0..1, un-rotated/un-cropped space
    "radius": 0.0056,           # normalized to max(W, H)
    "connect": False,           # True => also draw thick lines between points
    "kind": "spot",             # "spot" | "scratch" (overlay shape hint)
    "source": "auto",           # "auto" | "manual" (overlay tint + re-detect)
}
```

Representations produced:
- **Manual click** → `points=[p]`, `connect=False`, `kind="spot"`,
  `source="manual"`.
- **Manual drag** → ordered `points`, `connect=True`, `kind="scratch"`,
  `source="manual"`.
- **Auto round spot** → `points=[centroid]`, `radius = equiv_radius + pad`,
  `connect=False`, `kind="spot"`, `source="auto"`.
- **Auto elongated/curly component** → `points` = component pixels grid-sampled
  at spacing ≈ the disc radius, `radius = half_thickness + pad`, `connect=False`
  (overlapping discs cover the squiggle with no need for skeleton ordering),
  `kind="scratch"`, `source="auto"`.

### 4.2 Persistence
`catalog.py` saves/loads `dust_heal_strokes` alongside the existing per-image
state (mirroring `crop_rect` at `catalog.py:154-155` save / `:369-371` load):
- save: `"dust_heal_strokes": img.dust_heal_strokes or []`
- load: `img.dust_heal_strokes = state.get("dust_heal_strokes", []) or []`

### 4.3 Undo
`capture_undo_state` (`ccr_image.py:721`) gains a **deep-copied**
`dust_heal_strokes` entry (each stroke nests a `points` list, so a shallow copy
would alias live structure — same reasoning as `area_layers` at `:735`).
`pop_undo_state` (`:749`) restores it. Every add/remove/auto-detect/clear pushes
exactly one undo step **before** mutating, via the existing
`push_undo_state()`.

## 5. Processing / math

New module **`src/core/dust_removal.py`** — pure `numpy`/`cv2`, no Qt, no new
dependencies (OpenCV + NumPy are already required). The two public functions:

### 5.1 `detect_dust(img_u16, sensitivity="med", brush_radius_norm=None) -> list[dict]`
Input: the converted positive (`resized_raw`, uint16 RGB, ≤1080 px). Steps:

1. **Luminance**: `lum = cv2.transform(img, _LUM_WEIGHTS)` (reuse the weights at
   `ccr_processor.py:1352`), float32.
2. **Morphological residuals** with an elliptical kernel sized to the image
   (`k = odd(round(0.006 * L))`, clamped to `[5, 31]`, `L=max(H,W)`):
   - `tophat = lum - open(lum, k)` → **bright** dust (the dominant case: bright
     specks/strings on the positive).
   - `blackhat = close(lum, k) - lum` → **dark** dust/scratches.
   - `residual = max(tophat, blackhat)`.
3. **Threshold** by a robust noise estimate: `thr = k_sens * MAD(residual)` where
   `MAD` is the median absolute deviation over the whole residual and
   `k_sens ∈ {Low:5.0, Med:3.5, High:2.5}`. `mask = residual > thr`.
4. **Connected components** (`cv2.connectedComponentsWithStats`, 8-conn).
   Filter by area: `min_area = max(2, round((0.0012*L)**2))`,
   `max_area = round(0.0006 * H * W)` (≈ 0.06 % of the frame; bigger blobs are
   not "dust").
5. **Prominence gate** (false-positive control on textured detail): keep a
   component only if its peak residual exceeds the surrounding annulus median by
   `> thr` — naturally rejects busy texture where the annulus is itself bright.
6. **Classify + vectorize** each kept component into a stroke (§4.1):
   - bbox aspect `≤ 2` and high `solidity` → **spot**: centroid + equivalent
     radius (`sqrt(area/π)`), `pad = 1px`.
   - else → **scratch**: grid-sample the component's pixels at spacing
     `s = max(1, round(half_thickness))` (half_thickness from the component's
     distance transform), `radius = half_thickness + pad`.
   Pixel coords → normalized; radius px → `/max(H,W)`.

Returns a list of `source="auto"` strokes. Deterministic (no RNG).

### 5.2 `inpaint_dust(img_u16, strokes, grain=0.5) -> np.ndarray`
Input: any-resolution converted positive (preview, hi-res tile, or full export).

1. If `not strokes` → return `img` unchanged (same object; preserves the
   `apply_adjustments` fast path).
2. **Rasterize** `mask` (uint8, H×W) via `rasterize_strokes(strokes, H, W)`:
   for each stroke draw filled discs (`cv2.circle`) at every point at
   `r_px = round(radius * max(H,W))`; if `connect`, also `cv2.line` thick
   (`2*r_px`) between consecutive points. Dilate the mask by `HEAL_PAD = 2px` so
   the inpaint covers each defect's soft edge.
3. **Tile**: compute connected mask regions' bounding boxes, pad each by
   `r_px + 8`, merge overlapping boxes. Inpaint each tile independently — only
   small neighborhoods are touched, so cost is ∝ defect count, not image size
   (important at full export resolution).
4. **16-bit harmonic fill** per tile (`_inpaint_tile`), numpy/cv2 only, no
   8-bit round-trip. **See §9.2 for the authoritative solver** (coarse-to-fine
   seed + iteration formula + gradient-fidelity rationale). In brief: solve the
   Laplace equation on the hole with Dirichlet BC (known pixels fixed) via a
   `cv2.blur` relaxation seeded from a `pyrDown`/`pyrUp` coarse fill so few
   fine-level iterations are needed. At convergence a harmonic fill **reproduces
   a local linear gradient exactly** (linear ramps are harmonic), so a thin
   defect across a sky gradient fills to the correct interpolation with no halo —
   the v1 risk was under-convergence, fixed by the coarse seed.
   - **Grain** (default on, `grain=0.3`): add deterministic zero-mean Gaussian
     noise (per channel) scaled to the local std of the known boundary ring, only
     on `hole`, so the patch isn't a tell-tale smooth blob (§9.2 pins the RNG).
     `grain=0` disables. Grain is cosmetic and **not** resolution-matched across
     preview vs. export in v1 (documented limitation).
   - Write the tile's `hole` pixels back into a copy of `img` (clip to
     `[0,65535]`, `uint16`).

Telea/Navier-Stokes (`cv2.inpaint`) were considered but are 8-bit-only; the
harmonic fill is 16-bit-native and ideal for thin defects on smooth backgrounds
(the target), avoiding the smudgy look 8-bit inpaint can give film grain.

### 5.3 Where it applies (the single hook)
First step inside `CCRImage.apply_adjustments` (`ccr_image.py:556`), **before**
the early-return fast path at `:570` so a heal-only edit (no sliders) still
heals:

```python
def apply_adjustments(self, image, settings=None, ..., areas_override=None,
                      dust_strokes_override=None):                 # NEW param
    s = ...; cb = ...; tb = ...; bb = ...; profile = ...; areas = ...
    has_areas = ...
    strokes = (self.dust_heal_strokes if dust_strokes_override is None
               else dust_strokes_override)                         # NEW
    if strokes:
        from core.dust_removal import inpaint_dust
        image = inpaint_dust(image, strokes)                       # NEW
    if not s and cb == 0 and tb == 0 and bb == 0 and not has_areas:
        return self._to_grayscale(image) if profile == "bw" else image
    adjusted = adjust_image_opencl(image, ...)
    ...
```

The `dust_strokes_override` parameter mirrors `areas_override`: the zoom hi-res
worker (`HiResDetailWorker`) snapshots a deep copy of `dust_heal_strokes` at
request time and passes it, so a concurrent edit on the GUI thread can't race the
worker (the worker must not read live `self.dust_heal_strokes`).

This one hook covers **all** render paths automatically:
- **Preview**: `update_thumbnail_and_preview` → `apply_adjustments(resized_raw)`
  (`ccr_image.py:473`).
- **Export**: `ccr_normalize_with_*` → `apply_adjustments(rgb_brightness_normalized)`
  at full res (`ccr_processor.py:863`), **before** `apply_crop_to_image` (`:871`)
  and the flips/rotation (`:876-894`) — so strokes in uncropped space line up.
- **Zoom hi-res detail**: `render_hires_base` returns the converted,
  pre-adjustment base (`ccr_image.py:663-716`); the zoom worker then calls
  `apply_adjustments`, so the detail view is healed too.

`_adjust_for_area` (`:621`) operates on the already-globally-adjusted (already
healed) base and does **not** call `apply_adjustments`, so there is no
double-healing.

### 5.4 Performance
Detection on ≤1080 px: a few morphology passes + one CC labeling, ~10–40 ms.
Inpaint is tiled, so preview re-heal on each add/remove is dominated by a few
small Jacobi solves (sub-10 ms typical). Export inpaint is the same tiles at
full res — cost scales with total defect area, not image size.

## 6. Integration points

| Location | Change |
|---|---|
| `core/dust_removal.py` (**add**) | `detect_dust`, `inpaint_dust`, `rasterize_strokes`, `_inpaint_tile`, helpers. |
| `core/ccr_image.py` `__init__` | add `self.dust_heal_strokes = []`. |
| `core/ccr_image.py` `apply_adjustments:556` | call `inpaint_dust` before the early-return (§5.3). |
| `core/ccr_image.py` `capture_undo_state:721` / `pop_undo_state:749` | deep-copy / restore `dust_heal_strokes`. |
| `core/ccr_backend.py` | `detect_dust_by_index(idx, sensitivity)`, `set_dust_heal_strokes_by_index`, `add_dust_stroke_by_index`, `remove_dust_stroke_by_index`, `clear_dust_strokes_by_index`. |
| `core/catalog.py` `serialize_image:141` / `_restore_image:347` | persist `dust_heal_strokes` (+ `_strokes_from_json` defensive loader, like `_areas_from_json`). |
| `core/catalog.py` `_is_pristine:168` | add `and not state.get("dust_heal_strokes")` so a heal edit is saved. |
| `widgets/image_preview.py` toolbar (after Un-convert, before Export) | add Heal toggle, Auto-Detect, Visualize toggle, Clear, brush & sensitivity combos. |
| `widgets/image_preview.py` `_update_unconvert_action_state` | enable/disable the new controls on convert state + heal mode. |
| `widgets/image_preview.py` handlers (**new**) | `toggle_heal_mode`, `auto_detect_dust`, `toggle_visualize`, `clear_dust`, `_set_brush`, `_set_sensitivity`, `_draw_dust_overlay`, `_image_transform`, `map_full_to_displayed`, `_heal_hit_test`, `_commit_manual_stroke`. |
| `widgets/image_preview.py` `apply_transformations:1100-1121` | extract reusable `_image_transform()` (base + fine rotation, no prescale); reuse for authoring + overlay. |
| `widgets/image_preview.py` `GraphicsImageView` `:127-410` | heal-mode branches in press/move/release + brush ring; `heal_mode`/`_heal_drag` state; add `heal_drag` to the mid-pan interaction guard `:137-141`. |
| `widgets/image_preview.py` `update_preview:956-966` | reset `self._dust_overlay_items = []`; redraw overlay in `apply_transformations`. Escape-key exits heal mode. |
| `widgets/image_preview.py` `HiResDetailWorker` | snapshot `dust_heal_strokes` (deep copy) and pass as `dust_strokes_override`. |
| `tests/test_dust_removal.py` (**add**) | detection + inpaint unit tests (§8 + §9.6). |

Backend API shape:

```python
# CCRBackend
def detect_dust_by_index(self, idx, sensitivity="med") -> int   # returns #strokes
def add_dust_stroke_by_index(self, idx, stroke) -> None
def remove_dust_stroke_by_index(self, idx, stroke_index) -> None
def clear_dust_strokes_by_index(self, idx) -> None
def get_dust_strokes_by_index(self, idx) -> list[dict]
```

## 7. Files touched / added

- **add** `src/core/dust_removal.py` — detection + 16-bit tiled harmonic inpaint.
- **edit** `src/core/ccr_image.py` — attribute, `apply_adjustments` hook, undo.
- **edit** `src/core/ccr_backend.py` — detect/add/remove/clear/get API.
- **edit** `src/core/catalog.py` — persist `dust_heal_strokes`.
- **edit** `src/widgets/image_preview.py` — toolbar controls, heal mode, mouse
  handling, overlay, coordinate helper.
- **add** `tests/test_dust_removal.py` — unit tests (synthetic injected dust).

## 8. Test plan

Unit (`tests/test_dust_removal.py`, pure numpy/cv2, no Qt — mirrors
`tests/test_curves.py` / `test_area_editing.py` synthetic-image patterns):
- **Detect single bright spot**: inject a bright disc on a noisy smooth gradient;
  `detect_dust` returns ≥1 stroke whose point is within tolerance of the center.
- **Detect multiple spots**: 3 injected spots → ≥3 strokes.
- **Detect a curly/elongated streak**: inject a thin bright squiggle → a
  `kind="scratch"` stroke whose disc-set, when rasterized, covers ≥80 % of the
  injected streak pixels.
- **No false positives on a clean image**: clean noisy gradient → ≤ small
  tolerance (≤2) strokes; sensitivity Low stricter than High.
- **Inpaint removes the spot**: `inpaint_dust` makes the healed region closer to
  the pre-dust original (lower MAE) than the dusty input.
- **Inpaint preserves untouched pixels**: pixels far from any stroke are exactly
  unchanged (`array_equal` on far corners).
- **Empty strokes is identity**: `inpaint_dust(img, [])` returns the same object.
- **dtype/shape**: output is `uint16`, same shape.
- **Resolution independence**: rasterizing the same normalized strokes at 1× and
  2× yields geometrically matching masks (centroid/coverage scales correctly).

Manual:
- Heal toggle only enabled after Convert; entering shows the hint + overlay.
- Auto-Detect marks the sky spots; Low/Med/High change how many are found.
- Click a marked spot → it disappears (un-healed); the dust reappears.
- Click an unmarked spot → it heals; drag along the curly string → it heals.
- Brush S/M/L changes the ring + healed footprint.
- Clear Spots removes everything; Undo restores; survives image switch, app
  restart (catalog), and is visible in zoom detail + the exported TIFF/JPEG.
- A cropped image heals correctly (strokes in uncropped space, healed before
  crop); a rotated/flipped/B&W image heals in the right place.

## 9. Refinement (v2) — resolved decisions & added detail

### 9.1 Coordinate transform must invert fine rotation (correctness)
The v1 §3.3 chain used the coarse-only `_base_transform` (the same one the Crop
and B/W-point tools use), which does **not** invert the straighten "fine
rotation". Healing, unlike crop, runs with fine rotation *displayed*, so a click
on the rotated view would be stored in the wrong pixels. Verified against code:

- The displayed `pixmap_item` carries `pre * img_transform`, where `img_transform`
  = coarse base **+ fine rotation** (`apply_transformations`,
  `image_preview.py:1100-1131`); `pre` is the hi-res prescale.
- Scene relates to **preview-local** by `img_transform` alone (the prescale maps
  hi-res-local → preview footprint), so the inverse map is
  `_image_transform().inverted()` with `_image_transform()` = base + fine
  rotation, **no prescale**. This works whether or not a hi-res pixmap is
  displayed.

So authoring uses `_image_transform().inverted()[0].map(scene)` →
`map_displayed_to_full` → normalize; the overlay uses the forward
`_image_transform()` on items built in preview-local via `map_full_to_displayed`.

**Why this is correct at export for both conversion modes** (verified):
- *Reference mode*: `resized_raw` is **not** fine-rotated; at export, fine
  rotation is applied only to the 1080 reference copy for percentiles
  (`ccr_processor.py:666-675`), and the working array reaching `apply_adjustments`
  (`:863`) is **not** fine-rotated → same space as `resized_raw` → strokes align.
- *B/W-point mode*: `resized_raw` **is** fine-rotated (baked at `:1029-1034`), and
  the export array reaching `apply_adjustments` (`:1107`) is fine-rotated once
  too → still the same space → strokes align. (The bwpoint export then rotates
  the *already-healed* result again at `:1147-1160` — a **pre-existing**
  double-rotation bug, out of scope here; healing rides along with the content.)

**Constraint (documented):** author heal strokes *after* setting the straighten
slider. In ref mode, changing fine rotation afterward is safe (strokes live in
the non-rotated space; the view/export rotate the healed result as a whole). In
bwpoint mode, changing fine rotation after authoring can shift strokes (its
`resized_raw` is re-baked); the UI hint notes "straighten before healing".

### 9.2 Inpaint solver — convergence, gradient fidelity, grain RNG
`_inpaint_tile(tile_u16, hole_mask, grain)` solves Laplace with Dirichlet BC.
A naïve mean-seeded `cv2.blur` Jacobi loop converges too slowly for wide holes
(the v1 `iters = 3*thickness` was wrong). Authoritative algorithm:

1. **Coarse-to-fine seed** (multigrid V-cycle, `cv2.pyrDown`/`pyrUp`, numpy/cv2
   only): downsample tile+mask by 2 until the hole's max thickness ≤ 4 px (≤ 3
   levels), fill the coarsest level by mean-seed + a few relaxations, then
   `pyrUp` each level as the seed for the next and relax. This removes the
   slow low-frequency error so the fine level needs few passes.
2. **Fine relaxation** (per level): `filled = float32(tile)`;
   `filled[hole] = cv2.blur(filled,(3,3))[hole]` for
   `iters = clip(round(1.5 * level_thickness), 8, 64)` (`level_thickness` =
   `2 * max(distanceTransform(hole))` at that level). Known pixels stay pinned
   (Dirichlet), so the boundary tone/gradient is honored exactly.
3. **Gradient fidelity:** at convergence the harmonic solution is exact for any
   locally-linear ramp (linear functions are harmonic) — a thin scratch across a
   sky gradient fills to the correct linear interpolation, no halo. The seam risk
   is purely under-convergence, eliminated by step 1. Tested in §9.6.
4. **Grain (cosmetic, default `GRAIN=0.3`):**
   `rng = np.random.RandomState(int(tile.mean()) & 0xFFFFFFFF)`; per channel add
   `rng.normal(0, GRAIN * std(known_ring), size=hole.sum())` to `filled[hole]`,
   where `known_ring` = pixels within `dilate(hole, 6px) \ hole`. Deterministic
   (reproducible tests). Not spectrum- or resolution-matched in v1 (limitation).

### 9.3 Visualize Dust mode (added to v1)
A checkable **Visualize** toolbar toggle (enabled only in heal mode). When on,
`update_preview` renders a high-contrast "find dust" view instead of the normal
positive: `vis = normalize(|lum - median_blur(lum, k)|)` stretched and shown
desaturated, so specks/scratches pop. It is **display-only** (never stored,
never exported), implemented as a branch in the preview compose path keyed on
`self.dust_visualize`. Auto-Detect and manual healing operate on the real
`resized_raw` regardless of the visualize toggle. This makes curation feasible
(dust is otherwise sub-pixel at fit zoom).

### 9.4 Brush sizing is image-relative (not preview pixels)
Brush radius is stored **normalized to the long side** (`r_norm`), with
`S/M/L = 0.004 / 0.008 / 0.016` (≈ 0.4 / 0.8 / 1.6 % of `max(W,H)`; ~24/48/96
source px on a 6000 px scan, ~4/8/16 px on a 1080 preview). The ring is drawn in
image space (transformed by `_image_transform()`), so it tracks the true footprint
at every zoom — fixing the v1 "16 px ring dwarfs the dust at 100 %" problem.
`Med` is the default.

### 9.5 Interaction: disambiguation, exit, feedback
- **Left-click toggle is retained** (the user's explicit ask is "click a detected
  spot to remove it" / "click a missed spot to heal it"). Disambiguation: a
  left-press whose image point is inside an existing stroke's core
  (`HEAL_HIT` = stroke radius, in screen px via `_view_scale`) **removes** that
  stroke; otherwise it **starts a heal**. Removal is non-destructive + a single
  undo step; the hint always shows "Ctrl+Z undoes". **Right-click** is an explicit
  remove-under-cursor.
- **Escape** exits heal mode (handled like Crop/Area exit), in addition to the
  implicit exit on any external `update_preview`.
- **Feedback:** the hint shows live counts, e.g. "12 spots (9 auto, 3 manual) ·
  click to remove/heal · Ctrl+Z undo · Esc exit". After Auto-Detect it shows
  "Detected N spots (replaced M previous auto)".
- **Re-run policy:** Auto-Detect replaces prior **auto** strokes, keeps manual.
  This is undoable, and the count message states what was replaced, so a
  sensitivity re-run is recoverable. (A merge dialog is a possible later
  enhancement, not v1.)
- **Brush mid-drag:** size change applies to the *next* stroke; the active drag
  keeps its start radius.
- **Overlay tints:** auto = `(0,180,255)` (cyan-blue), manual = `(0,210,0)`
  (green), live drag ghost = `(255,210,0)` (amber); each with a 1 px black
  cosmetic outline for contrast.

### 9.6 Detection: shape classifier, false-positive control, round-trip
- **Shape filter** (replaces vague "high solidity"): for each thresholded
  component compute `aspect = max(w,h)/min(w,h)` and `solidity = area/hull_area`.
  Keep as a **spot** iff `aspect ≤ 2 AND solidity ≥ 0.6`; keep as a **scratch**
  iff `aspect > 2 AND area ≥ 4*min(w,h) AND solidity < 0.6`; otherwise **reject**
  (rejects mid-size solid blobs that are usually real subject detail).
- **Prominence gate** (concrete): `annulus` = `dilate(component, 3*half_thickness)
  \ dilate(component, 1)`; keep only if `max(residual[component]) -
  median(residual[annulus]) > thr`. Rejects components sitting on already-bright
  texture.
- **Curly round-trip:** scratch components are grid-sampled at spacing
  `s = max(1, round(0.7 * radius_px))` so the rasterized discs (radius `r_px`)
  always overlap (pitch `< 2·r_px`) → guaranteed coverage of curls/forks; radius
  is normalized, so coverage scales at export.
- **False positives are expected** (§2) and handled by curation; defaults err
  toward fewer detections (Low/Med) to keep curation light.
- **Added tests** (beyond §8): (a) thin 2 px scratch across a 50-level gradient →
  healed gradient slope within ±5 % of the original (no halo); (b) synthetic
  curly streak → detect → rasterize at 1×/2×/3× with matching strokes → IoU ≥ 0.8
  at every scale; (c) two runs of `inpaint_dust` on the same input are bit-identical
  (grain RNG determinism); (d) a bright "flower-like" solid blob (aspect≈1.3,
  large area) is **rejected** by the shape filter at Med sensitivity.

### 9.7 Tuned constants (module-level in `dust_removal.py`)
Starting defaults, exposed as named constants for easy tuning (we cannot run a
50-image empirical sweep here; these are conservative and curation covers the
rest). `L = max(H, W)` of the array being processed.

| Constant | Value | Meaning |
|---|---|---|
| `MORPH_KERNEL` | `odd(round(0.006*L))`, clamp `[5, 51]` | top-hat/black-hat ellipse size |
| `K_SENS` | `{low:5.0, med:3.5, high:2.5}` | threshold = `K_SENS * MAD(residual)` |
| `MIN_AREA` | `max(2, round((0.0012*L)**2))` | reject sub-grain specks |
| `MAX_AREA` | `round(0.0006*H*W)` | bigger ⇒ not "dust" |
| `ASPECT_SPOT` / `SOLIDITY_SPOT` | `2.0` / `0.6` | spot vs. scratch split |
| `SCRATCH_SAMPLE` | `0.7 * radius_px` | grid spacing for curly components |
| `HEAL_PAD` | `max(1, round(0.002*L))` (~2 px@1080, ~6 px@3000) | mask dilation before fill |
| `TILE_PAD` | `r_px + 8` (current res) | inpaint tile bbox padding |
| `GRAIN` | `0.3` | grain strength (×local std); `0` disables |
| `BRUSH_NORM` | `{S:0.004, M:0.008, L:0.016}` | brush radius / `L` |
| `LUM_WEIGHTS` | Rec.601 (reuse `ccr_processor._LUM_WEIGHTS`) | luminance for detection |

All sizes derive from `L`, so detection/fill behave consistently from the 1080
preview up to a full-res export.

### 9.8 Persistence / undo / worker / copy-paste integration
- **Catalog:** `serialize_image` (`catalog.py:141`) adds
  `"dust_heal_strokes": img.dust_heal_strokes or []`; `_restore_image`
  (`:347`, after the crop restore at `:371`) sets
  `img.dust_heal_strokes = _strokes_from_json(state.get("dust_heal_strokes"))`
  (a new defensive loader mirroring `_areas_from_json`, dropping malformed
  entries); `_is_pristine` (`:168`) gains `and not state.get("dust_heal_strokes")`.
- **Undo:** `capture_undo_state` (`ccr_image.py:721`) adds
  `"dust_heal_strokes": copy.deepcopy(self.dust_heal_strokes)` (deep, like
  `area_layers` at `:735`); `pop_undo_state` (`:749`) restores a deep copy.
- **Zoom worker:** `HiResDetailWorker` deep-copies `dust_heal_strokes` at
  construction and passes `dust_strokes_override` (§5.3) so it never reads live
  state.
- **Copy/Paste adjustments & Sync-to-All:** these operate on
  `adjustment_settings` only; since strokes are a **separate attribute**, they are
  automatically excluded — which is correct, dust is physical to one frame. No
  code needed; called out so a future refactor doesn't accidentally fold strokes
  into the adjustment dict.

### 9.9 Out-of-scope confirmations & known limitations
- Clone-source picking, batch detect-across-frames, Sync-to-All of strokes,
  per-point smoothing, and a GPU path remain non-goals.
- **Known limitations:** (1) fills on fine *texture* read as smudges (§2); (2)
  healed-region grain is not spectrum/resolution-matched (§9.2); (3) the
  pre-existing bwpoint export double-fine-rotation bug is not fixed here; (4)
  changing fine rotation *after* authoring can shift bwpoint strokes (§9.1).

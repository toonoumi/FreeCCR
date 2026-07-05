# Spec: Dust Removal

Status: REFINED v2
Owner: FreeCCR
Feature branch: `feature/dust-removal`
Reference: ported/adapted from the `openenlarge` repo (manual eraser + AI auto-dust).

## 1. Summary

Add a **Dust Removal** feature for spotting out dust, hair, and small scratches on
scanned film. It has two paths that share one non-destructive data model:

- **Manual** — the user paints over dust on the canvas with a sized brush; the
  painted region is healed by cloning the best-matching clean patch from its
  neighborhood (§5.2), so the speck is filled with real surrounding texture.
- **AI (hybrid)** — an ONNX neural detector (BOPBTL U-Net, downloaded on first
  use) finds dust automatically; the detected blobs are healed with the same
  clone fill. A sensitivity slider tunes how aggressive detection is.

A single **Dust Removal** button on the image toolbar enters "dust mode": the
right-hand sliders panel is covered by a Dust Removal panel exposing the manual
and AI controls. Edits are stored as normalized 0..1 *spots* on the image,
replayed in the render pipeline at preview and full-export resolution, persisted
in the catalog, and undoable.

## 2. Goals / Non-goals

### Goals
- One **Dust Removal** toolbar button that toggles dust mode (enter/exit).
- In dust mode, the sliders panel is replaced (covered) by a **DustRemovalPanel**
  with a **Manual** section and an **AI** section, plus a **Done** button.
- **Manual**: brush-size slider; click or click-drag on the canvas paints a
  spot/stroke; the painted area is inpainted on release; **Undo last spot** and
  **Clear all** controls; a live red mask overlay while painting.
- **AI**: a **Detect & Remove** button and a **Sensitivity** slider. If the ONNX
  detector model is not present, an explicit **Download AI model (~150 MB)** gate
  (off-thread, with progress + SHA‑256 verification) is shown first.
- **Non-destructive**: edits are stored as normalized spots on `CCRImage`; the
  render pipeline rasterizes a mask and inpaints at whatever resolution it is
  processing, so preview, hi-res zoom, and export all match and scale.
- Dust edits **persist** in the catalog, **participate in Undo** (Ctrl+Z), and
  survive image switch / app restart.
- Manual dust removal works with **no new runtime dependencies** and **fully
  offline** (the clone heal is pure numpy/OpenCV; `cv2` ships with the
  already-required `opencv-python`).
- The AI detector is **optional and degrades gracefully**: if `onnxruntime` or
  the model is unavailable, the manual path is unaffected and the AI section
  shows an unavailable/needs-download state instead of erroring.

### Non-goals
- No MI-GAN / neural *inpainting* fill (openenlarge's second ONNX model). Fill is
  the classical clone heal (+ `cv2.inpaint` fallback) only. The detection backend
  is abstracted so a neural inpainter could be added later.
- No IR-channel ("infrared cleaning") dust removal — FreeCCR has no IR plane.
- No per-spot re-editing handles (move/resize an existing spot). Edits are
  add-only with Undo-last and Clear-all. (Future enhancement.)
- No elongated-scratch vectorization. AI-detected components are approximated by
  circular spots; long scratches are best handled with the manual brush (§5.4).
- AI detection is **not** re-run inside the render pipeline (too slow for the hot
  path). Detection is an explicit user action that *adds spots*; rendering only
  rasterizes + inpaints stored spots (§5.3).

## 3. UX / Interaction

### 3.1 Entering / exiting dust mode
- A **Dust Removal** checkable `QAction` is added to `ImagePreview`'s toolbar,
  immediately after the `▤ Gradient` action and preceded by a separator (grouped
  with the area-editing tools). Clicking toggles dust mode.
- Gating: enabled only when the current image's sliders are enabled (a converted
  negative, or any image in Positive mode) — the same `sliders_enabled` condition
  computed in `_update_unconvert_action_state` (image_preview.py:1269-1292).
  Disabled otherwise.
- **Entering** dust mode (`ImagePreview.enter_dust_mode()`):
  1. Exit any other canvas mode (crop / slice / area / bw-point / wb-pick).
  2. Set `self.dust_mode = True`; reset zoom to fit and `_release_hires`.
  3. `update_preview(idx)` now shows the **full, un-cropped** working positive
     with **coarse rotation/flip only** (no fine rotation, no displayed crop) —
     mirroring crop mode — so canvas pixels map 1:1 to `resized_raw` (§5.5).
  4. Set a brush-circle cursor and draw the red mask overlay for the image's
     stored spots.
  5. Call `self.window().toggle_dust_removal(True)` so `MainWindow` swaps panels.
- **Exiting** dust mode (the **Done** button, the toggled-off toolbar action, or
  `Esc`): hide the red mask overlay, clear `dust_mode`, restore the normal
  (cropped, fine-rotated) preview, restore the sliders panel, and re-enable the
  toolbar actions. **Previously applied dust edits remain on the image.**

### 3.2 DustRemovalPanel layout (top → bottom)
1. Header: **Dust Removal**.
2. **Manual** group:
   - **Brush size** slider, log-scaled over 0.05%–20% of image width so tiny
     specks get fine steps (0.05% ≈ 3 px radius on a 6000 px scan) while big
     scratch-covering sizes stay reachable; the label shows the size as a % of
     image width. The canvas Ctrl+wheel resize clamps to the same range.
   - **Feather** slider (0–1.0% of image width, default 0.30%): the edge fade
     width of every heal on the image — manual and AI spots alike. A
     per-image render parameter stored as `CCRImage.dust_feather`; changes
     re-heal live (debounced ~200 ms), persist in the catalog, and are NOT
     part of undo snapshots (like brush size). Defect-like pixels are always
     fully filled regardless of the feather (§5.2 step 6), so a wide fade
     cannot blend the defect back in.
   - Hint: "Click or drag over dust to remove it."
   - **Undo last spot** button and **Clear all** button.
3. Separator.
4. **AI** group:
   - When the model is **absent**: an explanatory line + **Download AI model
     (~150 MB)** button; while downloading, a progress bar; on failure, an error
     line + retry.
   - When the model is **present**: a **Sensitivity** slider (0–100), a
     **Detect & Remove** button (current image), and a **Detect & Remove — All
     Images** button. The latter runs detection **per image** (each scan has its
     own dust) at the current sensitivity, off the GUI thread with progress, and
     replaces each image's `auto` spots (manual spots kept). Detection always
     heals only the **manual** spots first, never the `auto` spots it is about
     to replace, so re-running never loses prior coverage.
   - When `onnxruntime` is **not importable**: a disabled state with
     "AI detection unavailable in this build."
5. Spacer.
6. **Done** button (returns to the sliders panel; exits dust mode).

The panel reuses the existing dark slider/button styling. It holds an explicit
reference to `MainWindow` and `ImagePreview` passed at construction (it does
**not** rely on `parent().parent()` chains).

### 3.3 Manual painting
- On the canvas in dust mode:
  - **Left-press**: begin a stroke; record the first normalized point at the
    current brush radius; show the red dab.
  - **Left-drag**: append normalized points; extend the red stroke overlay.
  - **Left-release**: commit the stroke as one `brush` spot (§4),
    `push_undo_state()` once for the stroke, re-render the image (inpaint
    applied), and refresh thumbnail + preview. The dust is now visibly gone.
  - **Ctrl + mouse wheel**: resize the brush (kept in sync with the slider).
  - **Ctrl+Z**: undoes the last dust spot (identical to the panel's **Undo
    last spot** button) and **preserves the viewport** — the general
    MainWindow undo resets zoom (crop/rotation can move the displayed
    content), which read as "Ctrl+Z unzoomed" while spotting zoomed-in, so
    `undo_last_action` routes to the dust undo whenever `dust_mode` is on.
  - **Mouse wheel** (no modifier): zoom in/out for precise spotting.
    **Middle-button drag**: pan the zoomed view. (Both match the normal viewer;
    only the brush moves to Ctrl + wheel.)
  - **Hi-res detail loads on entry** (and refines on zoom) so dust is visible at
    full sharpness. The hi-res render reproduces `resized_raw`'s orientation and
    goes through `apply_adjustments` (so it shows the dust-removed result); the
    dust-mode display stays full / un-fine-rotated, so spots stay aligned.
  - A **brush-circle cursor** tracks the pointer; its on-screen radius is
    `r_norm * W * view_scale` display pixels (§5.5).
- Coordinate mapping reuses the existing inverse-`base_transform` +
  `map_displayed_to_full` chain (as B/W-point sampling and the reference frame
  do), producing **un-rotated, un-flipped, un-cropped** working-image pixel
  coords, then normalized by the working image (`resized_raw`) size (§5.5).

### 3.4 AI detect & remove
- **Detect & Remove** runs detection **off the GUI thread** on the current
  working positive (`resized_raw` with existing spots already inpainted, so
  already-cleaned dust is not re-flagged), at preview resolution:
  1. If no cached probability map for this image, run the ONNX detector and cache
     it (so re-tuning Sensitivity does not re-run the net).
  2. Threshold + size-gate + dilate the prob map into a binary mask at detection
     resolution (§5.3).
  3. Extract connected components; convert each to one normalized `auto` spot
     (centroid + radius from component extent).
  4. Replace any prior `auto` spots on the image with the new set (manual `brush`
     spots are untouched), `push_undo_state()` once, re-render.
- **Sensitivity** changes re-threshold the cached prob map and refresh the
  `auto` spots live (no net re-run) unless the working buffer changed.
- A status line reports "Removed N spots" (or "No dust found").

### 3.5 Interaction with other edits
- Dust removal is independent of sliders/curves/areas/crop: it runs earliest in
  the adjustment stage (§5.1), so subsequent color adjustments apply to the
  already-cleaned positive.
- **Compare** (hold) and the main **Reset** button do **not** clear dust spots
  (dust is a separate concern from tonal adjustments). Both code paths
  (`on_compare_pressed`, `on_reset_clicked` in sliders_panel.py) already leave
  `dust_spots` untouched; an explanatory comment is added at each to keep that
  intentional. **Clear all** in the dust panel (and Undo) is the only UI that
  removes spots.

## 4. Data model

### 4.1 Storage
A new attribute on `CCRImage`, initialized in `__init__` next to `area_layers`:

```python
self.dust_spots: list[dict] = []
```

Each entry is a **spot** (a stamp = the union of equal-radius circular dabs along
a polyline):

```python
{
  "kind": "brush" | "auto",   # provenance: hand-painted vs AI-detected
  "pts":  [[x, y], ...],      # normalized; x over WIDTH, y over HEIGHT, in [0,1]
  "r":    0.012,              # radius as a fraction of image WIDTH, in (0,1]
}
```

- A single click is a `brush` spot with one point.
- An AI-detected blob is an `auto` spot with one centroid point and a radius
  from the component's extent.
- Empty list = "no dust removal"; the render fast-path leaves the image
  untouched (§5.2).
- Alongside the spots, `CCRImage.dust_feather` (float, fraction of image
  width, default 0.003) holds the image's heal edge-fade width — one value
  for all spots, set by the panel's Feather slider, serialized in the catalog
  (`dust_feather`, missing key → default), excluded from undo snapshots, and
  part of the hi-res `dust_sig` so a feather change invalidates cached
  renders.

Rationale: mirrors openenlarge's normalized `DustStroke` model, serializes
cleanly to JSON, deep-copies cleanly for undo, and is fully resolution
independent (rasterized at the processing resolution).

### 4.2 Persistence (catalog)
- `serialize_image()` (catalog.py:141) adds `"dust_spots": list(img.dust_spots)`.
- `_restore_image()` (catalog.py:347) restores
  `img.dust_spots = list(state.get("dust_spots") or [])` (missing key → `[]`, so
  old catalogs load unchanged).
- `_is_pristine()` (catalog.py:168) adds `and not state.get("dust_spots")` to its
  AND-chain, so a **dust-only** image (no conversion/adjustment/crop) is treated
  as edited and is **not** purged on save (same hazard the curves spec fixed in
  §9.8). No decoded pixels are persisted — spots replay against the freshly
  decoded scan exactly like the existing conversion/adjustment replay.

### 4.3 Undo
- `capture_undo_state()` (ccr_image.py:794) adds
  `"dust_spots": copy.deepcopy(self.dust_spots)`.
- `pop_undo_state()` (ccr_image.py:822) restores
  `self.dust_spots = copy.deepcopy(state.get("dust_spots", []))`.
- Deep-copied because entries hold nested point lists. `dust_spots` is **global**
  image state (not per-area), so the `active_area_id` validation in
  `pop_undo_state` is unaffected.

### 4.4 Render cache invalidation
The hi-res zoom adjustment identity `_current_adj_sig` (image_preview.py ~1554)
must include a **dust signature** so changing spots invalidates the cached hi-res
render. It goes in the *adjustment* signature (display-time), **not** the base
decode signature `_hires_signature`. Concrete form:

```python
dust_sig = tuple(
    (s.get("kind"),
     tuple((round(p[0] * 10000), round(p[1] * 10000)) for p in s.get("pts", [])),
     round(float(s.get("r", 0)) * 10000))
    for s in getattr(img, "dust_spots", []))
```

(4-decimal rounding ≈ 0.01% of the image — fine enough to never falsely reuse a
stale render, coarse enough not to thrash.)

## 5. Processing / math

### 5.1 Where it applies — the single funnel
Dust removal runs at the **very start of `CCRImage.apply_adjustments(image)`**,
*before* the early-return guard and `adjust_image_opencl`:

```python
def apply_adjustments(self, image, ...):
    image = self._apply_dust_removal(image)   # NEW — no-op when no spots
    s = self.adjustment_settings if settings is None else settings
    ...
    if not s and cb == 0 and tb == 0 and bb == 0 and not has_areas:
        return self._to_grayscale(image) if profile == "bw" else image
    ...
```

`apply_adjustments` is the one method every render context routes the
**post-inversion positive** through, at its own resolution (verified):

- **Preview / thumbnail**: `update_thumbnail_and_preview` →
  `apply_adjustments(resized_raw)` (ccr_image.py:537).
- **Hi-res zoom**: the hi-res worker renders the conversion base, then calls
  `apply_adjustments` for display.
- **Export**: each normalization function calls `apply_adjustments` on the
  full-res post-inversion positive when `output_path is not None` —
  `ccr_normalize_with_reference` (ccr_processor.py:863),
  `ccr_normalize_with_bwpoint` (:1113), positive export (:1206/:1208).

Because user **crop / flip / 90° / fine-rotation are applied *after***
`apply_adjustments` in every export path (e.g. :871, :1119, :1212), spots stored
in `resized_raw` (un-cropped, un-rotated, un-flipped) space line up at every
resolution. The area-layer path (`_adjust_for_area`) does **not** funnel through
`apply_adjustments`, so dust is applied exactly once per render.

`_apply_dust_removal` runs before the guard so a **dust-only** image (all sliders
neutral) is still inpainted. For un-converted negatives, dust runs before the
display-only `_auto_brightness_for_preview` (ccr_image.py:548) — acceptable: the
normalized spots replay correctly at export regardless of preview auto-brightness.

### 5.2 Mask rasterization + clone-heal fill (`ccr_processor.py`)
Functions:

```python
def rasterize_dust_mask(spots, h, w) -> np.ndarray:                  # uint8 {0,255}, (h,w)
def apply_dust_removal(img16, spots, inpaint_radius=3) -> np.ndarray: # uint16 RGB
```

- `rasterize_dust_mask`: zero mask; for each spot `r_px = max(1, round(r*w))`;
  draw `cv2.circle(mask, (round(x*w), round(y*h)), r_px, 255, -1)` at each point,
  and a `cv2.line(..., thickness=2*r_px)` between consecutive points of a stroke
  so a fast drag leaves no gaps. (`r_px` is width-based in **both** axes → a true
  pixel circle at any resolution; pixels are square so it reads as a circle.)
- `apply_dust_removal` — **clone heal** (healing-brush style), replacing the
  original `cv2.inpaint` (Telea) fill. Telea diffuses a distance/direction
  weighted **average** of the surrounding ring inward, which produced a smooth,
  grainless round patch with radial fan-like streaks — obvious on grainy film.
  The heal instead copies real neighboring texture. All locality decisions key
  off the hole's **half-thickness** (max distance transform), never its bbox —
  a traced hair's bbox can span half the frame while the stroke is a few px
  wide. Per mask component (`cv2.connectedComponentsWithStats`):
  0. **Segmenting**: components larger than ~6× their thickness (min 32 px)
     are healed in thickness-scaled segments, each from its own local source
     strip — one whole-stroke window would demand a clean area the size of the
     stroke's bbox, which rarely exists (a traced curl would silently fall
     back to diffusion and ghost).
  1. **Window + guarded ring**: patch bbox padded by `guard + ring_w`; `ring`
     = clean known pixels at distance `(guard, guard+ring_w]` from the hole
     (`guard` ≥ 2 px, scaled to half the thickness). The gap matters: the
     defect's soft edge leaks past the brushed mask, and pixels hugging the
     hole are the most likely to be the defect itself.
  2. **Defect-color rejection**: the defect's color is estimated from the
     hole's own content (a tight stroke's hole IS the defect; a generous
     brush's defect is the hole's outlier mode vs the ring). Ring pixels
     closer to that color than half the clean cluster's distance (p95 of the
     ring's defect-distances) are rejected from matching AND tone — a leaked
     hair edge, or the hair's continuation past the stroke's end (which can
     be the ring's *majority*), cannot lift the fill into a bright ghost of
     the stroke. Legit bimodal structure survives (both modes sit far from
     the defect color).
  3. **Search**: candidate source windows on `_HEAL_ANGLES` (16) directions ×
     thickness-scaled distances (`2·half_th + pad + 2`, ×1/2/4, plus the
     window diagonal as a last resort). A candidate must be in-bounds and
     fully clean (checked O(1) against `cv2.integral` of the padded mask), so
     a long stroke heals from the clean strip right beside it. Score = SSD
     between source and destination **kept ring** pixels; best wins.
  4. **Clone + membrane tone correction**: hole pixels are copied from the
     best source, plus a smooth per-channel correction field interpolated
     from the kept-ring differences (normalized Gaussian convolution,
     σ = thickness + pad, so any surviving contamination only tints its own
     neighborhood; plain ring-mean where the ring weight underflows). Grain
     is kept verbatim while low frequencies land on the hole's boundary
     values, so gradients continue through the patch. All in float32 from the
     uint16 source — the fill is **16-bit native** (no 8-bit quantization).
  5. **Fallback**: a patch with no clean in-bounds source window (image
     border, dense dust) or a starved ring (< `_HEAL_MIN_RING_PX`) falls back
     to the old 8-bit `cv2.inpaint(..., inpaint_radius, INPAINT_TELEA)` fill.
  6. **Feathered composite**: alpha rises 0 → 1 from each hole's boundary
     inward over a smoothstep ramp (`_feather_alpha`), `out = img16*(1-a) +
     filled*a` computed inside the mask's bbox only. The ramp width is the
     image's **Feather** setting (`dust_feather`, a fraction of image width —
     default 0.3%, so it covers the same image fraction at preview and
     export), capped per hole by its depth so the core still reaches full
     fill. **Defect-like hole pixels** (colored like the estimated defect)
     are force-filled at alpha 1 regardless of the ramp (`dlike`, with a
     light blurred lip), so a wide feather can never blend the defect back
     in — the soft fade only happens across clean rim pixels. Outside the
     mask alpha is exactly 0 — away from any spot `out == img16`
     bit-for-bit.
  7. **Resolution-stable plan**: the heal's content-adaptive decisions —
     stroke segmentation, each segment's chosen source offset, and the
     diffusion-fallback verdicts — are computed ONCE at the canonical
     preview scale (`_DUST_PLAN_LONG` = 1080 long side) and REPLAYED on
     larger buffers with all pixel geometry scaled (segment size, Telea
     radius, offsets normalized over width/height; segments matched by
     nearest normalized centroid). Re-deriving the plan per resolution let
     the SSD argmin flip between scales, so the export healed from visibly
     different patches than the preview the user approved ("the preview and
     the final don't look the same"). Buffers at or below the canonical
     scale (the preview itself, thumbnails, the detect source) plan on
     themselves — their behavior is unchanged. A replayed offset that no
     longer lands on a clean window (mask-raster rounding — rare) falls back
     to the local search for that segment; the tone membrane and feather
     composite always run natively, so fills stay 16-bit sharp.
  - Identity fast-path: empty `spots` or all-zero mask → return `img16` unchanged.
  - Returns a new `uint16` RGB array; `img16` is never mutated (non-destructive).
  - Residual preview↔export differences are grain/resampling plus a ≤1 px rim
    registration from mask rounding at hard edges — the healed structure is
    identical (regression-tested in `TestResolutionConsistency`).
  - Cost: per-component window work only (no full-frame 8-bit convert unless a
    fallback fires). Planning searches at 1080 px and native execution only
    validates + clones, so a full-res export heals FASTER than the old
    native-resolution search (+ one INTER_AREA downscale of the frame).

### 5.3 AI detection (`src/core/dust_detect.py`)
A self-contained module wrapping the ONNX BOPBTL detector. **`onnxruntime` is
imported only inside functions, never at module level**, so the module (and the
panel that imports it) load even when `onnxruntime` is absent.

- **Availability**: `is_available()` (can `import onnxruntime`) and
  `is_model_present()` (model file on disk).
- **Model management**: `model_path()` →
  `<APPDATA>/FreeCCR/models/detector.onnx` (same app dir as the catalog).
  `download_model(progress_cb)` streams the asset with `requests`, verifies
  SHA‑256 before use, writes atomically (`tmp` + `os.replace`).
  - URL: `https://github.com/MohaElder/openenlarge/releases/download/autodust-assets-v1/detector.onnx`
  - SHA‑256: `61e4a93d4e94b4fc6212e2e9b785fa12b5cbc9654724b02aaf8b212075bb729f`
  - (URL / SHA / filename are module constants so they can be repointed.)
- **`detect(positive_rgb16) -> (prob_h, prob_w, prob: float32[0,1])`**: resize so
  the short side is `DETECT_SHORT=512` and both dims are multiples of 16; Rec.709
  luma; normalize to `[-1,1]`; run the session (`[1,1,dh,dw]` → logits); sigmoid.
  CPU execution provider (matches openenlarge's Windows guidance). Returns the
  prob map at detection resolution (cached per image for cheap Sensitivity).
- **`prob_to_spots(prob, prob_h, prob_w, sensitivity, max_dim) -> list[spot]`**:
  - `thr = 0.85 - 0.60 * (sensitivity/100)` (0 → very selective … 100 →
    aggressive), matching openenlarge.
  - Binarize at `thr`; via `cv2.connectedComponentsWithStats`:
    - drop components whose pixel area exceeds a resolution-normalized
      `max_blob` (film border / real image content, not dust);
    - **drop elongated components** (aspect ratio > `MAX_ASPECT`): the detector
      also fires on legitimate thin LINES (a bike frame, the horizon, a path
      edge); circle-inpainting those smears real detail, so linear defects are
      left to the manual brush (§5.4).
    - **bright-speck gate**: film dust inverts to **white** specks, so a real
      dust blob is brighter than its surroundings. Require the blob's mean luma
      to exceed a surrounding ring by `BRIGHT_MARGIN`; this rejects normal-toned
      content the detector wrongly fires on (a face, a dark feature — which is
      how the AI once removed a person's head). `detect` therefore returns
      `(prob, luma)` so `prob_to_spots` has the detection-resolution grayscale.
      This and the aspect filter are the main guards against AI false positives.
  - Each surviving (compact) component → one `auto` spot: centroid
    `(cx/prob_w, cy/prob_h)`; **area-equivalent** radius
    `r = (sqrt(area/π) + pad) / prob_w` — a tight circle matching the speck, so
    the inpaint stays invisible rather than leaving a smudge (the old
    bounding-extent radius over-covered and was the artifact source).
  - `prob_to_spots` is **pure / model-free** (operates on a numpy prob map) so it
    is unit-testable without ONNX.
- All ONNX use is guarded; any import/inference error surfaces as
  "AI detection unavailable" and never breaks manual dust or the rest of the app.

### 5.4 Known approximation
AI-detected components become circular spots. Round dust/hair specks are covered
well; long diagonal scratches are only partially covered by one circle — the
manual brush (a stroke = many dabs) is the tool for those. Keeps the stored model
uniform and resolution-independent. (Future: component polygons / multi-circle.)

### 5.5 Coordinate mapping (canvas → normalized spot)
Dust mode shows the working positive with **coarse rotation/flip only** — no
displayed crop (the `not self.dust_mode` guard is added to update_preview's
crop-display branch, image_preview.py:951) and no fine rotation (mirroring crop
mode's `apply_transformations`). Therefore canvas pixels map **exactly** to
`resized_raw` pixels via:
1. `scene = view.mapToScene(event.pos())` (auto-accounts for zoom/pan).
2. Invert the display `base_transform` (coarse 90°/180°/270° + H/V flips) → local
   pixmap coords.
3. `map_displayed_to_full(x, y)` — **identity** in dust mode (no crop shown).
4. Normalize: `H, W = resized_raw.shape[:2]`; `x_n = px/W`, `y_n = py/H`,
   `r_n = r_px/W`.

Because spots are applied in `apply_adjustments` **before** crop and fine rotation
(§5.1), and dust mode shows the image **without** those, mapping is exact — no
sub-pixel fine-rotation error. The on-canvas brush radius is `r_n*W*view_scale`
display pixels.

## 6. Integration points

| File | Change |
|---|---|
| `src/core/ccr_image.py` | `self.dust_spots = []` in `__init__`; `_apply_dust_removal()` + call at top of `apply_adjustments`; add `dust_spots` to `capture_undo_state` / `pop_undo_state`. |
| `src/core/ccr_processor.py` | `rasterize_dust_mask`, `apply_dust_removal`, feathered-composite helper. |
| `src/core/dust_detect.py` (new) | ONNX detector: availability, model path/download/verify, `detect`, `prob_to_spots`, constants. `onnxruntime` imported only inside functions. |
| `src/core/catalog.py` | Serialize/restore `dust_spots`; add `not state.get("dust_spots")` to `_is_pristine`. |
| `src/widgets/image_preview.py` | Toolbar **Dust Removal** checkable action (after Gradient, with separator) + gating in `_update_unconvert_action_state`; `dust_mode` flag; `enter_dust_mode`/`exit_dust_mode`; canvas press/move/release + brush cursor + red mask overlay; `_scene_to_norm_spot()`; `and not self.dust_mode` in the crop-display branch + clear_preview/Esc routing; `dust_sig` in `_current_adj_sig`. |
| `src/widgets/dust_panel.py` (new) | `DustRemovalPanel(QWidget)` (manual + AI sections, Done); QThread workers for model download and detection; explicit MainWindow/ImagePreview refs. |
| `src/ui/main_window.py` | Add `dust_panel` as a **direct child** of `central_widget`'s layout (fixed 300 px, hidden); `toggle_dust_removal(on)` toggles `sliders_panel`/`dust_panel` visibility (no `QStackedWidget` — preserves `SlidersPanel`'s `parent().parent()` chains) and drives canvas enter/exit + toolbar action state. |
| `requirements.txt` | Add `onnxruntime` (CPU). Manual path needs nothing new (`cv2`, `requests` already present). |
| `LICENSES/` | Add attribution for the detector model (BOPBTL U-Net, MIT, via openenlarge) — see §9 open item. |
| `tests/test_dust_removal.py` (new) | See §7. |

## 7. Test plan

The harness (`tests/run_tests.py`) sets `QT_QPA_PLATFORM=offscreen`; tests are
pytest-style. Two groups:

**Pure processing (no Qt, no ONNX model):**
- `rasterize_dust_mask`: a normalized spot rasterizes to a filled circle of the
  expected pixel radius at a given `(h,w)`; the **same** spot covers a
  proportionally larger region at 2× resolution (resolution independence).
- `apply_dust_removal`: on a synthetic flat image with a bright speck, a spot over
  the speck removes it (masked pixels move toward the surround) while pixels far
  from any spot are **bit-for-bit unchanged**; dtype stays `uint16`.
- Identity: empty `dust_spots` → `apply_dust_removal` returns the input unchanged.
- `apply_adjustments` integration: a `CCRImage` with only `dust_spots` set (all
  sliders neutral) still inpaints (the early guard does not skip dust).
- `prob_to_spots` (model-free): synthetic prob map → assert sensitivity→threshold
  mapping, size-gating (an over-large blob is dropped), and small blobs become
  spots with sane normalized centroids/radii.

**Persistence / undo (CCRImage, no GUI):**
- `serialize_image`/`_restore_image` round-trips `dust_spots`; an old entry with
  no `dust_spots` key restores to `[]`; `_is_pristine` returns False for a
  dust-only state.
- `capture_undo_state` → mutate → `pop_undo_state` restores the prior `dust_spots`
  as an independent deep copy.

**Graceful degradation:**
- Monkeypatch `sys.modules["onnxruntime"]` to raise `ImportError` *before*
  importing `dust_detect`; `is_available()` is `False` and importing the module
  does not raise.

**Manual (not automated):**
- Toolbar button toggles the panel cover; Done restores sliders; Esc exits.
- Paint a spot → dust disappears; Undo restores it; Clear all empties.
- Download gate appears without the model; after download, Detect & Remove finds
  specks; Sensitivity re-tunes live without a visible re-run.
- Dust survives image switch, app restart (catalog), Ctrl+Z, and is visible in
  zoom detail and the exported file at full resolution.

## 8. Build / packaging

`onnxruntime` is in `requirements.txt`, so from source (`python src/main.py`) AI
works out of the box. Making it work in the **Nuitka standalone exe** took four
fixes (all empirically verified by a startup diagnostic that logs onnxruntime
readiness, and `--windows-console-mode=attach` so it's visible):

1. **Compiler = MSVC, not mingw** (`--msvc=latest`). mingw gcc 14.2 / clang 19
   both crashed (ICE / codegen) compiling FreeCCR's huge generated C files
   (`ccr_processor`, `__helpers`); MSVC compiles them. `--jobs=6` bounds parallel
   compile memory (24 cores otherwise launch 24 compilers and crash on the huge
   units).
2. **Bundle onnxruntime**: `--include-package=onnxruntime
   --include-package-data=onnxruntime` (pulls the modules, the `.pyd`, and the
   native `capi/*.dll`).
3. **Refresh the MSVC runtime** (the key one): Nuitka bundles the VC toolset's
   redist runtime (`msvcp140*.dll` / `vcruntime140*.dll`), which is OLDER
   (14.29/14.32) than onnxruntime needs (14.44+). Loaded against the stale
   runtime the `.pyd` fails its init ("DLL initialization routine failed") and AI
   is silently disabled. `build_exe.bat` overwrites those DLLs in `main.dist`
   with the system's current ones post-build.
4. **`main.py` startup**: before any Qt import (load-order matters on Windows),
   add `<exe>/onnxruntime/capi` to the DLL search path and pre-load
   `onnxruntime.dll` so the bundled `.pyd` resolves it. Guarded / no-op from
   source.

The lazy in-function import still means a build *without* onnxruntime runs fine
(AI reports the reason, manual dust unaffected). The ~150 MB detector model is
**never** bundled; it downloads on demand to the app-data dir.

## 9. Refinement notes — resolved decisions

1. **Panel cover via visibility toggle, not `QStackedWidget`.** Wrapping
   `sliders_panel` in a `QStackedWidget` would insert a layer between it and
   `central_widget`, breaking the ~12 `self.parent().parent()` chains
   `SlidersPanel` uses to reach `MainWindow`. Instead both panels are direct
   children of `central_widget`'s layout and `toggle_dust_removal(on)` flips
   their `setVisible`. `dust_panel` takes explicit refs (no parent-chain
   reliance). Mirrors the proven tether-banner pattern.
2. **Toolbar action** is checkable, placed after `▤ Gradient` with a leading
   separator, gated by the same `sliders_enabled` as the adjustment sliders.
3. **Esc / exit** routes through `_on_escape_key` (`elif self.dust_mode:
   self.exit_dust_mode()`) and `clear_preview` also exits dust mode, alongside
   the existing crop/slice/area handling.
4. **Exact rendering in dust mode**: full image, coarse rotation/flip only, no
   crop, no fine rotation → exact 1:1 canvas↔`resized_raw` mapping (§5.5). This
   removes the fine-rotation sub-pixel error entirely.
5. **Single-funnel verified** against the real export paths (§5.1 line refs); a
   single insertion at the top of `apply_adjustments` covers preview, zoom, and
   all export modes.
6. **Fill** = per-component clone heal (best-patch match + membrane tone
   correction, 16-bit native), with `cv2.INPAINT_TELEA` radius 3 as the
   fallback when no clean source window exists; masked-only feathered composite
   (§5.2). Considered and rejected: `cv2.xphoto.inpaint` SHIFTMAP (exemplar
   based, but requires swapping `opencv-python` → `opencv-contrib-python` and
   re-validating the Nuitka build for no quality win over the custom heal).
7. **`dust_spots` is global** image state and independent of Compare/Reset; both
   leave it untouched (with clarifying comments).
8. **Catalog**: `_is_pristine` must include `dust_spots` so dust-only images
   persist; old catalogs restore `dust_spots → []`.
9. **Cache**: `dust_sig` added to the hi-res *adjustment* signature (§4.4).
10. **Image switch leaves dust mode** (like crop/area): `update_preview` tears
    down dust mode on a different image so a half-drawn stroke can't commit to
    the wrong image and the panel can't drift out of sync.
11. **Fine-rotation slider is disabled in dust mode** (the canvas shows the
    un-fine-rotated image), so it can't show an ignored value or mutate state.
12. **Detection results are discarded** if the user navigates to a different
    image (or out of dust mode) while the off-thread detector runs; the model
    download is cancelled on exit, and a session lock guards the cached ONNX
    session against the detect/download threads.

### Open items (non-blocking)
- Confirm the detector model's exact license text and add
  `LICENSES/license-openenlarge.txt` (BOPBTL U-Net is MIT per openenlarge);
  ensure the Nuitka build bundles it.
- Validate the chosen Nuitka onnxruntime option (§8) in a real build.
- Tune brush-size slider range/default and `max_blob` / `pad` constants against
  real scans during manual testing.

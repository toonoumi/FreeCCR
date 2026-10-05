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
  3. `update_preview(idx)` keeps the **normal display framing** — the confirmed
     crop (when one is set) with **coarse rotation/flip only** (no fine
     rotation) — so the brush works on exactly the frame the user keeps;
     strokes are mapped back to full-frame coords (§5.5).
  4. Set a brush-circle cursor and draw the red mask overlay for the image's
     stored spots.
  5. Call `self.window().toggle_dust_removal(True)` so `MainWindow` swaps panels.
- **Exiting** dust mode (the **Done** button, the toggled-off toolbar action, or
  `Esc`): hide the red mask overlay, clear `dust_mode`, restore the normal
  (fine-rotated) preview, restore the sliders panel, and re-enable the
  toolbar actions. **Previously applied dust edits remain on the image.**
- **An image switch KEEPS dust mode** — unlike crop/area, which are per-frame
  geometry, spotting is a per-ROLL task: selecting another thumbnail renders
  the new image with the brush still loaded, so the user keeps dusting
  straight through the roll without re-entering the mode. On the switch
  `update_preview` drops only the **in-progress stroke** (`_dust_pts`, which
  belongs to the previous image and must never commit to the new one) and the
  previous image's overlay items, then rebinds the panel's per-image state via
  `DustRemovalPanel.bind_image()` (feather value, detector prob cache, AI
  section, status line). The mode **ends** only when the new image is not
  dust-eligible — un-converted and not in Positive mode, the same
  `sliders_enabled` gate as the toolbar action — or when the last image is
  removed (`clear_preview`).

### 3.2 DustRemovalPanel layout (top → bottom)
1. Header: **Dust Removal**.
2. **Manual** group:
   - **Brush size** slider, log-scaled over 0.05%–20% of image width so tiny
     specks get fine steps (0.05% ≈ 3 px radius on a 6000 px scan) while big
     scratch-covering sizes stay reachable; the label shows the size as a % of
     image width. The canvas Ctrl+wheel resize clamps to the same range.
   - **Feather** slider (0–100% of each spot's own radius, default 25%): the
     edge fade width of every heal on the image — manual and AI spots alike.
     Radius-relative, so the fade **grows with the brush size** — a big
     scratch patch fades over a proportionally wide rim while a tiny speck
     stays tight. A per-image render parameter stored as
     `CCRImage.dust_feather`; changes re-heal live (debounced ~200 ms),
     persist in the catalog, and are NOT part of undo snapshots (like brush
     size). The feather governs EVERYTHING (maintainer rule: every pixel
     gets the effect, opacity rolled off from the stroke center — no
     exceptions, §5.2 step 6): at wide settings the heal fades out even
     across the defect itself, by explicit user choice; feather 0 is a
     hard, complete removal.
   - **Show heal sources** checkbox: diagnostic overlay on the canvas —
     stroke borders outlined (yellow) as the contours of the UNION of the
     rasterized strokes (connected/overlapping strokes read as ONE region;
     per-spot outlines drew noise lines through the interior), and for each
     heal segment a
     green arrow drawn FROM the patch it sampled TO the healed segment
     (ring at the source), or an orange ring where the segment fell back to
     diffusion (nothing was sampled). Geometry comes from the cached
     preview heal plan (`dust_debug_geometry` + `_dust_plan_cache`), mapped
     through the same crop-display transform as the brush, so it shows
     exactly what exports will replay. Refreshes on every dust edit /
     feather change; cleared on exit.
   - **Right-button stroke editing** (only while the sources overlay is
     shown): right-press on a stroke and drag to MOVE it (live border while
     dragging; the re-heal lands on release, one undo step). The stroke's
     plan records travel with it with their offsets compensated
     (`translate_dust_plan`), so a moved stroke still samples the very same
     patch of film — the reference location never changes. Right-press on a
     SOURCE ring and drag to RE-PICK where that segment samples from: the
     arrow follows live, the release re-heal binds the segment at its
     unchanged centroid and pins the chosen offset verbatim — the user's
     pick becomes the sticky reference (replayed by hi-res/export,
     persisted in the catalog; a plan-only edit, not an undo step). A
     re-picked source is marked `manual` in its plan record and cloned
     with CLONE-STAMP semantics: copied verbatim — color and texture —
     blended only by the feather at the rim. The membrane re-toning
     (§5.2 step 4) applies to automatic picks only; applied to a manual
     pick it fought the user, keeping the destination's old color and
     toning away the picked look. Double
     right-click on a stroke DELETES it (one undo step). Hit-testing tries
     source markers first (9 px), then the brush footprint
     (`dust_spot_hit`, topmost/newest stroke first); a right-click on empty
     canvas does nothing.
   - Hint: "Click or drag over dust to remove it. Ctrl+wheel resizes the
     brush. With heal sources shown: right-drag a stroke to move it (it
     keeps sampling the same patch); double right-click deletes it."
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
    goes through `apply_adjustments` (so it shows the dust-removed result); its
    display layer is crop-matched exactly like the preview, so spots stay
    aligned.
  - A **brush-circle cursor** tracks the pointer; its on-screen radius is
    `r_norm * W * view_scale` display pixels (§5.5).
- Coordinate mapping reuses the existing inverse-`base_transform` +
  `map_displayed_to_full` chain (as B/W-point sampling and the reference frame
  do), producing **un-rotated, un-flipped, un-cropped** working-image pixel
  coords, then normalized by the working image (`resized_raw`) size (§5.5).

### 3.4 AI detect & remove
- **Detect & Remove** runs detection **off the GUI thread** on the current
  working positive (with existing manual spots already inpainted, so
  already-cleaned dust is not re-flagged). The source is a **hi-res
  conversion replay** (`render_hires_base`, the same snapshot replay the
  zoom view uses) downscaled to `DETECT_MAX_LONG`, falling back to
  `resized_raw` when no replay is possible — soft off-focus dust is ~1 px
  at preview resolution and undetectable there. The source build runs
  inside the detect worker (it decodes + converts: seconds). When the base
  is a **windowed working-space buffer**, the detection source is
  de-windowed + window-clamped first (same as the WB picker / AWB): fed raw
  container codes the net sees a black frame and the bright-speck gate's
  absolute margin is unreachable — no dust would ever be found.
  1. If no cached probability map for this image, run the ONNX detector and cache
     it (so re-tuning Sensitivity does not re-run the net).
  2. Threshold + size-gate + dilate the prob map into a binary mask at detection
     resolution (§5.3).
  3. Extract connected components; convert each to one normalized `auto` spot
     (centroid + radius from component extent).
  4. With a confirmed crop, drop `auto` spots whose center falls **outside the
     crop frame** (same rasterized geometry as the display crop): the rebate /
     holder is not scene — it is cropped away everywhere, and its bright edge
     printing (frame numbers, sprocket edges) reads as dust to the detector.
     Manual `brush` spots are never filtered.
  5. Replace any prior `auto` spots on the image with the new set (manual `brush`
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
- Alongside the spots, the catalog stores `dust_plan` — the cached heal
  plan's records (`[cy, cx, [oy, ox] | null, genuine, dlike_on, manual]`),
  written only when they match the current spots — so a reload reseeds
  `_dust_plan_cache` and keeps the exact sources the user saw (including
  manual clone-stamp picks).
- Alongside the spots, `CCRImage.dust_feather` (float, fraction of each
  hole's own half-thickness, 0..1, default 0.25) holds the image's heal
  edge-fade width — one value for all spots, set by the panel's Feather
  slider, serialized in the catalog under `dust_feather_r` (missing key →
  legacy width-fraction `dust_feather` migrated by slider position ×100,
  else default), excluded from undo snapshots, and part of the hi-res
  `dust_sig` so a feather change invalidates cached renders.

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
     back to diffusion and ghost). Segments are also CAPPED at
     `_HEAL_SEG_MAX` (96 px at plan scale, ×`scale_up`): a big compact dab
     is thickness-scaled to one giant segment whose window rarely has a
     clean home anywhere, dumping the whole blob into the flat, grainless
     diffusion fallback — a visible smooth disc on film grain. Tiles keep
     cloning real texture from beside the blob. **Edit locality**: for
     components spanning multiple tiles the grid anchors to the IMAGE
     (absolute multiples of `seg`), not the component bbox, and each tile's
     guard/ring/search geometry keys off the tile-bounded local scale
     `min(half_th, seg/2)` (inert for uncapped components) — otherwise a
     new dab merging into a blob shifted every tile and resized every
     window, re-sampling segments far from the edit on each stroke.
     Compact single-tile components keep one un-split tile at their bbox.
  1. **Window + guarded ring**: patch bbox padded by `guard + ring_w`; `ring`
     = clean known pixels at distance `(guard, guard+ring_w]` from the hole
     (`guard` ≥ 2 px, scaled to half the thickness). The gap matters: the
     defect's soft edge leaks past the brushed mask, and pixels hugging the
     hole are the most likely to be the defect itself. Near-black pixels
     (`_HEAL_BLACK_FLOOR`, ~3% of white) are dropped from the ring while
     they are its minority — the film holder / rebate inverts to near-black
     and is not scene context; letting it anchor tone painted a dark smudge
     on strokes near the frame edge. (If dark dominates, the stroke really
     is in dark content and the ring stays.) Content OUTSIDE the image's
     confirmed crop (`CCRImage.crop_rect`/`crop_angle`, rasterized by
     `_crop_keep_mask` incl. rotation) is not scene either: it is merged
     into `mask_pad` like dust, so rings never anchor on it and fresh
     source searches never sample it — pinned sources keep their reference
     regardless (sticky), routing through the deferred pass onto the same
     pixels if they poke outside a newer crop.
  2. **Defect-color rejection**: the defect's color is estimated from the
     hole's own content (a tight stroke's hole IS the defect; a generous
     brush's defect is the hole's outlier mode vs the ring). Ring pixels
     closer to that color than half the clean cluster's distance (p95 of the
     ring's defect-distances) are rejected from matching AND tone — a leaked
     hair edge, or the hair's continuation past the stroke's end (which can
     be the ring's *majority*), cannot lift the fill into a bright ghost of
     the stroke. Legit bimodal structure survives (both modes sit far from
     the defect color). **The split validates itself under the white-dust
     prior**: the rejection is trusted only when the REJECTED pixels are
     luma-brighter than the KEPT ones by a noise margin — for a faint speck
     or clean-ish hole the "defect" estimate collapses onto the background,
     and the (inverted) split would strip the background and anchor the
     entire match/tone on some foreign minority in the ring (a wall across
     an edge), sending the source hunt far away while perfect same-tone
     context sits beside the spot. **Sanity cap** (`_HEAL_MIN_KEEP_FRAC`):
     if the rejection would keep < 20% of the ring, whatever survives is
     some small foreign mode — a black border, a dark roofline — that would
     become the entire anchor (clean sky strokes healed to solid black,
     cloned from the border). Either failure distrusts the estimate
     (`genuine=False`): full ring, no defect force-fill.
  3. **Search — THE AUTOMATIC SAMPLING RULE (maintainer, verbatim)**,
     applied to EVERY automatically picked initial sample — AI-detected
     spots and painted strokes alike (only the user's explicit right-drag
     re-pick is exempt). **Strokes are never auto-connected**: each spot is
     its own patch with its own segments, argmin and per-patch offset even
     when spots touch or overlap (drawing order owns shared pixels; only
     the never-clone-dust checks and the feathered composite see the
     union). Pinned and shared offsets validate touching against THE WHOLE
     PATCH, at plan time and replay alike: (1) the sample area MUST TOUCH the patch area —
     border-to-border, union contiguous (candidates: 64 directions at
     `2·half_th` +0/+1/+2 px of raster slack, sample-area cleanliness
     checked against the RAW mask so the dilation cannot push the sample
     off the border it must touch); (2) among the touching candidates the
     one with the lowest combined Δhue + Δsat + ΔV — measured against the
     patch's own background, estimated as the darker half of the HOLE's
     pixels (film dust is white) — plus the lowest texture wins; (3)
     again: the areas must touch. A TOTAL selection — no distance
     escalation, no "good enough" gate, no fallback ranking. On a windowed
     working-space buffer the deltas are computed on de-windowed display
     values. No ring statistic participates in the choice, and the
     defect-leak rejection (§ step 2) is disabled for auto components (it
     is a manual-trace defense; an auto spot cannot leak by construction).
     **One sample offset per patch**: the rule speaks of the patch as a
     unit — the component's first segment runs the argmin and the rest of
     the patch reuses its offset (nudged ≤ 4 px around raster grazes), so
     a long stroke cannot get differently-phased clones butted together
     mid-stroke. Known consequences, accepted with the rule: a brush
     NARROWER than the defect it traces samples the defect's own
     continuation (cover a defect to remove it), and on patterned content
     the fill may land phase-shifted (the rule has no pattern-phase term;
     the re-pick drag is the alignment tool). Invariant intersections:
     dust is never cloned (a touching window overlapping any spot is
     excluded; when NO in-bounds touching candidate exists at all — patch
     at frame edge or fully encircled — the boundary-diffusion Telea fill
     applies, growing from the patch's own touching border), and a
     candidate needs ≥ 1 clean ring px for the tone membrane (raw-mask
     subset, scale-stable).
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
  6. **Feathered composite**: alpha rises ~0 → 1 from each hole's boundary
     inward over a smoothstep ramp (`_feather_alpha`, evaluated at
     pixel-center depths `inside − 0.5` with FLOAT ramp widths, so the
     relative profile is identical at every resolution — an integer ramp
     with a boundary lift made thin-stroke rims visibly harder at preview
     than at export), `out = img16*(1-a) +
     filled*a` computed inside each **component's own padded window** (one
     global mask bbox degenerates to nearly the whole frame when spots are
     scattered, and the full-frame float blend then dominates the heal —
     ~0.9 s at 6000 px; components never touch, so per-component alpha is
     exactly the global answer). The ramp width is the image's **Feather**
     setting (`dust_feather`) as a fraction of **each hole's own depth**
     (its half-thickness) — the fade scales with the brush size, and with
     resolution, since the rasterized hole does — capped by that depth so
     the core still reaches full fill. The dome is the WHOLE story
     (maintainer rule: every pixel gets the effect, opacity rolled off
     from the stroke center — no exceptions): there is no defect
     force-fill floor, so a wide feather fades the heal out even across
     the defect itself, and the slider has visible authority on every
     heal — the old `dlike` floor (alpha pinned to 1 on defect-colored
     pixels, with a ramp-wide lip) made the feather do nothing exactly
     where users reach for it, on real dust. The `dlike_on` verdict (a
     `genuine` estimate whose defect color is > `_DLIKE_SEP_MADS`
     ring-grain MADs from the cleaned ring's median and brighter than the
     surround — film dust inverts WHITE) is still computed and carried in
     the plan as metadata. Outside the
     mask alpha is exactly 0 — away from any spot `out == img16`
     bit-for-bit.
  7. **Resolution-stable plan**: the heal's content-adaptive decisions —
     stroke segmentation, each segment's chosen source offset, the
     diffusion-fallback verdicts, and the per-segment `genuine`/`dlike_on`
     verdicts — are computed ONCE at the canonical
     preview scale (`_DUST_PLAN_LONG` = 1080 long side) and REPLAYED on
     larger buffers with all pixel geometry scaled (segment size, Telea
     radius, offsets normalized over width/height; segments matched by
     nearest normalized centroid). Per-pixel threshold classifications
     (holder-black, defect-like) read a luma smoothed to the plan scale
     (`img16_c`, box = the scale factor; identity on canonical buffers), so
     full-res grain crosses those thresholds no more often than the
     preview's area-averaged pixels did. Re-deriving the plan per resolution let
     the SSD argmin flip between scales, so the export healed from visibly
     different patches than the preview the user approved ("the preview and
     the final don't look the same"). Buffers at or below the canonical
     scale (the preview itself, thumbnails, the detect source) plan on
     themselves — their behavior is unchanged.
     **The preview's plan is the source of truth**: `CCRImage`'s preview
     render caches its plan (`_dust_plan_cache`, keyed by the spots'
     content; feather excluded — it shapes the blend, not the plan), and
     hi-res/export renders replay THAT plan via `apply_dust_removal(plan=)`.
     **Sticky sources**: once a segment's source is set, NOTHING may move
     it — every preview-scale re-heal is seeded with the previous plan
     (`prior_plan`, bound tightly by `_DUST_EDIT_TOL` so a NEW stroke's
     segments never inherit a neighbor's source unscored), and segments at
     unchanged centroids reuse their prior offset/verdicts verbatim instead
     of re-searching. Painting a new stroke therefore cannot re-sample the
     strokes already on screen. A pinned offset is never re-searched, only
     clamped into bounds (scale rounding); even a stroke landing ON a prior
     source keeps the reference — that segment defers to a second pass
     (after the clone/Telea fills) and clones the HEALED content at the
     very same location, so no dust is copied and the arrow never moves.
     The plan
     persists in the catalog (`dust_plan`, dropped when stale) and reseeds
     the cache on restore, so reloads keep the exact sources the user saw —
     a from-scratch replan of an incrementally-built plan could differ
     — a fresh plan from the export's own re-decoded/downscaled pixels can
     still flip near-tie source choices ("it sampled different spots than
     the preview"). A replayed offset that lands on dust or out of bounds
     after scaling (mask-raster rounding — rare) is rescued by the nearest
     clean offset within a ≤4 px Chebyshev spiral (essentially the same
     patch); only if that fails does the local search run. The tone
     membrane and feather composite always run natively, so fills stay
     16-bit sharp.
  - Identity fast-path: empty `spots` or all-zero mask → return `img16` unchanged.
  - Returns a new `uint16` RGB array; `img16` is never mutated (non-destructive).
  - Residual preview↔export differences are grain/resampling plus a ≤1 px rim
    registration from mask rounding at hard edges — the healed structure is
    identical (regression-tested in `TestResolutionConsistency`).
  - Cost: per-component window work only (no full-frame 8-bit convert unless a
    fallback fires). Planning searches at 1080 px and native execution only
    validates + clones, so a full-res export heals FASTER than the old
    native-resolution search. Every heal logs its step timing
    (`Dust heal WxH: setup …, N segments …, telea …, composite …` plus a
    `Dust heal total` line with the plan provenance) — measured at
    6000×4000 / 40 spots: preview 0.08 s; export 0.65 s with the cached
    preview plan (was 1.79 s before the plan replay + per-component
    composite).

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
- **`detect(positive_rgb16) -> (prob_h, prob_w, prob: float32[0,1])`**: run at
  the input's **native resolution** capped at `DETECT_MAX_LONG` (both dims
  rounded to multiples of 16), **tiled** through the U-Net (`DETECT_TILE`,
  `DETECT_OVERLAP` overlap, max-stitched so a seam can't split a speck);
  Rec.709 luma; normalize to `[-1,1]`; per tile `[1,1,th,tw]` → logits →
  sigmoid. CPU execution provider (matches openenlarge's Windows guidance).
  Returns the prob map at detection resolution (cached per image for cheap
  Sensitivity). **Why native/tiled:** most real film dust sits off the focal
  plane — soft, faint blobs a few scan-pixels wide. They are ~1 px blips at
  the 1080 preview and vanished entirely at the old fixed 512-short
  detection downscale; recall keeps rising up to ~3.2k long side (measured
  on the example dusty-sky scan), beyond which the time quadruples and the
  now-huge diffuse blobs start to fragment.
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
    - **smooth-surround rule**: auto-detection keeps spots on SMOOTH surrounds
      only — sky, open shadow (surround-ring luma std ≤ `SMOOTH_MAX_STD`). In
      busy texture (foliage, stucco, gravel) a compact bright glint is
      indistinguishable from dust at this scale and healing it eats real
      detail; textured areas are the manual brush's job. `detect()` uses the
      same idea (windowed-std map + the confirmed crop's footprint as a
      keep-mask) to skip whole tiles with nothing keepable — on a strip
      cropped to one frame most tiles never hit the net (~3× faster).
    - **bright-speck gate**: film dust inverts to **white** specks, so a real
      dust blob is brighter than its surroundings. The gate is
      **noise-relative** — dust sits off the focal plane, so most real specks
      are soft-edged and faint (a fixed absolute margin missed nearly all of
      them on smooth sky while they stood many grain-sigmas above it): the
      blob's mean-luma lift over a surrounding ring must exceed
      `min(BRIGHT_MARGIN, max(BRIGHT_FLOOR, BRIGHT_SNR·σ))` with σ the ring's
      own luma std. Sharp dust (≥ `BRIGHT_MARGIN`) always passes; faint dust
      passes where the surround is smooth; busy texture keeps the full
      absolute bar. Normal-toned content the detector wrongly fires on (a
      face, a dark feature — which is how the AI once removed a person's
      head) still fails: its lift is ~0 or negative. `detect` therefore
      returns `(prob, luma)` so `prob_to_spots` has the detection-resolution
      grayscale. This and the aspect filter are the main guards against AI
      false positives.
  - Each surviving (compact) component → one `auto` spot: centroid
    `(cx/prob_w, cy/prob_h)`; **area-equivalent** radius scaled by
    `SPOT_SCALE`: `r = (sqrt(area/π)·SPOT_SCALE + pad) / prob_w`. Area-based,
    NOT the bounding-box extent (that over-covered and was the artifact
    source) — but scaled up because the thresholded component is only the
    bright CORE of a soft off-focus speck; an exact-fit circle left its faint
    skirt unhealed.
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
Dust mode shows the working positive with the **normal display framing** — the
confirmed crop (when one is set) with **coarse rotation/flip only**, no fine
rotation (mirroring crop mode's `apply_transformations`). Spots are stored
normalized against the **full un-cropped frame** — they are healed in
`apply_adjustments` **before** crop and fine rotation (§5.1) — so canvas points
map back via:
1. `scene = view.mapToScene(event.pos())` (auto-accounts for zoom/pan).
2. Invert the display `base_transform` (coarse 90°/180°/270° + H/V flips) → local
   (possibly cropped) pixmap coords.
3. `map_displayed_to_full(x, y)` — identity with no crop displayed; under a
   confirmed crop it applies `_crop_display_transform` (a translation, plus a
   rotation for an angled crop) into full-frame pixel coords.
4. Normalize: `H, W = resized_raw.shape[:2]`; `x_n = px/W`, `y_n = py/H`,
   `r_n = r_px/W`.

No fine rotation is displayed, and the crop extraction is isometric
(translate/rotate only) so lengths are preserved: the on-canvas brush radius is
`r_n*W*view_scale` display pixels with `W` the **full-frame** width, and the
live stroke overlay maps its stored full-frame points back through the inverse
crop transform for drawing (`_draw_dust_stroke`).

## 6. Integration points

| File | Change |
|---|---|
| `src/core/ccr_image.py` | `self.dust_spots = []` in `__init__`; `_apply_dust_removal()` + call at top of `apply_adjustments`; add `dust_spots` to `capture_undo_state` / `pop_undo_state`. |
| `src/core/ccr_processor.py` | `rasterize_dust_mask`, `apply_dust_removal`, feathered-composite helper. |
| `src/core/dust_detect.py` (new) | ONNX detector: availability, model path/download/verify, `detect`, `prob_to_spots`, constants. `onnxruntime` imported only inside functions. |
| `src/core/catalog.py` | Serialize/restore `dust_spots`; add `not state.get("dust_spots")` to `_is_pristine`. |
| `src/widgets/image_preview.py` | Toolbar **Dust Removal** checkable action (after Gradient, with separator) + gating in `_update_unconvert_action_state`; `dust_mode` flag; `enter_dust_mode`/`exit_dust_mode`; canvas press/move/release + brush cursor + red mask overlay; crop-aware stroke mapping (`_dust_scene_to_norm` via `map_displayed_to_full`, normalized by the full frame) + clear_preview/Esc routing; `dust_sig` in `_current_adj_sig`. |
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
4. **Exact rendering in dust mode**: the normal display framing (incl. the
   confirmed crop), coarse rotation/flip only, no fine rotation → exact
   canvas↔`resized_raw` mapping through the isometric crop transform (§5.5).
   This removes the fine-rotation sub-pixel error entirely.
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
10. **Image switch KEEPS dust mode.** (Supersedes the original decision, which
    tore the mode down like crop/area.) Dusting is a per-roll task, so
    `update_preview` leaves `dust_mode` set on a different image and only
    discards the in-progress stroke plus the previous image's overlay items,
    then rebinds the panel to the new image. A half-drawn stroke still cannot
    commit to the wrong image (`_dust_pts` is cleared on the switch) and the
    panel cannot drift out of sync (`bind_image()` re-reads the current image,
    invalidating the per-image detector cache). The mode still ends on a
    non-dust-eligible image (un-converted, outside Positive mode) and in
    `clear_preview` when the last image goes away.
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

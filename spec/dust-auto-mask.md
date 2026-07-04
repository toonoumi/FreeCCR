# Spec: Dust Auto-Mask (outlier-targeted heal within the brushed selection)

Extends `spec/dust-removal.md` §5.2. Prompted by user feedback: their previous
tool "checked and masked for outliers right in the selection and only replaced
the masked area plus 1px buffer. The sample was taken from the non-masked part
of the selection. That is especially useful when removing dust close to a
border (e.g. bright face to dark background)."

## 1. Summary

Today the entire brushed selection is the heal hole: the clone fill replaces
every pixel under the stroke. When the brush is a generous circle around a
small speck — the natural gesture, and what every AI-detected spot looks like —
that replaces far more of the image than necessary, and when the circle
straddles a high-contrast edge the ring-matched source search must reproduce
BOTH sides of the edge from a single distant window, which often fails
visibly.

This feature adds a render-time **mask shrink** pass: per brushed component,
detect the defect pixels *inside* the selection as statistical outliers
against the local surround, shrink the heal hole to just those pixels plus a
small buffer, and let the existing clone heal fill them. Because the clean
part of the selection is no longer masked, it automatically becomes valid
source area for the (thickness-scaled, therefore now very local) source
search — the fill is sampled from right next to the defect, on the correct
side of any edge. Pixels of the selection that are not defect-like stay
bit-for-bit untouched.

When the selection does not contain a confident outlier subset (tight traces
over hairs, dabs over clean areas, indistinct low-contrast defects), the pass
leaves the component's mask unchanged and the behavior is exactly today's
whole-stroke heal.

## 2. Goals / Non-goals

### Goals
- A generous dab/circle around a defect heals only the defect (+ buffer);
  the rest of the selection is preserved bit-for-bit.
- Heals near high-contrast borders stop smearing: the fill for a defect on
  the bright side is sourced from the bright side.
- Applies identically to manual strokes and AI-detected ("auto") spots — it is
  keyed on mask content, not spot kind.
- Fully automatic by default, with a predictable fallback to whole-stroke
  behavior when the outlier picture is not clear — plus a global **Settings
  choice** of heal engine (Auto-mask / Whole stroke / Inpaint legacy) so
  users can compare and opt out per taste.
- Preserves every existing guarantee of `apply_dust_removal`: non-destructive,
  16-bit native, resolution-independent contract, only-masked-pixels-change
  (the buffer may extend ≤ buffer-radius px past the stroke, see §5.4),
  feather semantics, Telea fallback.

### Non-goals
- No change to spot storage (`dust_spots` stays `{kind, pts, r}` normalized),
  persistence, or undo — the shrink is a pure render-time function of
  (image pixels, rasterized mask, global method setting).
- No dust-panel controls or overlay changes (the method picker lives in the
  Settings dialog). The red stroke overlay continues to show the brushed
  area; under auto-mask its meaning is "heal what's dusty in here", which is
  what users already expect of a healing brush.
- No per-image method: the engine choice is global, like Auto Gain.
- No change to AI detection itself (`dust_detect.py`).
- Not a general blemish/content-aware eraser: detection is local-outlier
  based; large-area retouching stays out of scope.

## 3. UX / Interaction

Unchanged surfaces: dust panel, brush, feather slider, Ctrl+Z, AI detect flow.

**Settings → General → "Dust removal" → "Heal method"** (`QComboBox`, staged
and applied on Done like every other Settings toggle):

| Choice | id | Engine |
| --- | --- | --- |
| Auto-mask (default) | `automask` | outlier shrink + clone heal (this spec) |
| Whole stroke | `clone` | clone-heal the entire stroke (v1.1 behavior) |
| Inpaint (legacy) | `inpaint` | cv2.inpaint diffusion on the whole stroke (pre-v1.1) |

The setting is **global and live** (mirrors Auto Gain / Gamma mode): spots are
stored normalized and healed at render time, so applying a change re-renders
every loaded image with all existing spots replayed through the new engine —
users can flip between the three on the same spots to compare. Persisted in
QSettings (`adjust/dust_method`); unknown stored values fall back to
`automask`.

Behavioral contract under Auto-mask (the default):
- Circle loosely around a speck/hair → only the defect disappears; grain and
  detail inside the circle survive. Feather applies to the (small) healed
  patch's rim.
- Trace a hair tightly (brush ≈ hair width) → identical to Whole stroke: the
  shrink pass declines (nearly everything under the stroke is defect).
- Dab over clean area (nothing to remove) → identical to Whole stroke: the
  dab is replaced with a matched clone (harmless no-op visually). The shrink
  pass declines rather than silently doing nothing, so a user who paints over
  a *subtle* defect below the outlier threshold still gets it replaced.

## 4. Data model

Per-image: none. `dust_spots`, catalog persistence, and undo snapshots are
untouched — the shrink is deterministic per (pixels, mask, method), so no new
per-image state exists.

Global: `ccr_backend.dust_method: str` (default `"automask"`), loaded at
startup from QSettings `adjust/dust_method` (validated against
`DUST_METHODS`), written by `MainWindow.on_dust_method_changed`. The method
string joins `_current_adj_sig` (next to the `auto_gain` / `gamma_luminance`
booleans) so a change invalidates cached hi-res renders.

## 5. Processing / math

### 5.1 Placement — one new pass in the single funnel

`apply_dust_removal(img16, spots, ..., method=DUST_METHOD_DEFAULT)` routes on
the engine right after rasterization (`CCRImage._apply_dust_removal` passes
the live `ccr_backend.dust_method`, the funnel every render path shares):

```
mask = rasterize_dust_mask(spots, h, w)
if method == "automask":
    mask = _automask_shrink(img16, mask)  # NEW — may return mask unchanged
... existing component/segment clone-heal pipeline ...
    # method == "inpaint": each component is routed whole into the existing
    # Telea fallback (same thickness-capped feather the no-clean-source
    # path uses) — the pre-v1.1 engine, kept as a selectable legacy option.
```

Everything downstream (components, `mask_pad`, the integral cleanliness image,
`_heal_patch`, feather, Telea fallback, composite) operates on the shrunken
mask with **no changes**. Two properties fall out for free:

- The clean part of the selection is unmasked, so the integral-image "source
  window touches dust" check passes there: `_heal_patch`'s source search — at
  distances scaled to the (now small) defect thickness — samples from
  immediately beside the defect, i.e. the reporter's "sample from the
  non-masked part of the selection".
- A shrunken hole is mostly defect, which is exactly the "tight stroke" case
  `_heal_patch`'s internal defect-color defenses are designed for.

### 5.2 Outlier detection (per connected component)

For each connected component `C` of the rasterized mask (each with its own
window, as `_heal_patch` does: `half_th` from the distance transform;
`guard = max(_HEAL_GUARD, 0.5*half_th)`; `ring_w = max(_HEAL_RING, guard+3)`;
`pad = guard + ring_w`; window = bbox(C) padded by `pad`, clipped to frame):

1. **Ring** = pixels with `guard < dist(C) <= pad`, excluding the 1-px-padded
   ORIGINAL mask (all spots — neighbors have not shrunk yet, order-free).
   Fewer than `_HEAL_MIN_RING_PX` ring pixels → keep `C` unshrunk (no
   context to judge outliers against).
2. **Two-anchor surround model.** 2-means over ring pixels (distance =
   per-channel mean absolute difference), anchors initialized at the
   per-channel medians of the ring's bottom and top luma DECILES, two
   assignment/update iterations. For a unimodal surround the anchors converge
   near-coincident and the model degrades to a single anchor. For a bimodal
   surround (the bright-face/dark-background case) they land on the two modes
   even when the split is unbalanced (a 90/10 ring defeats a plain
   median-luma split — the minority mode ends up inside one half and its
   clean pixels get flagged). This is what makes edge-straddling selections
   work: a single ring-median would flag the whole minority side of the edge
   as "outlier".
   `d(px) = min(mean|px - m_lo|, mean|px - m_hi|)` (mean over channels).
3. **Noise scale** `sigma = median(d(ring))` — the ring's own scatter about
   its anchors (a MAD analogue that stays honest for bimodal rings).
4. **Bright outlier seeds** (white-dust prior — film dust blocks light on
   the negative, so it inverts to WHITE specks and strings):
   `seed = (d > max(K*sigma, ABS)) AND luma > luma(nearest anchor)`.
   The brightness test is against the NEARER anchor, so dust on the dark
   side of an edge counts even when it is darker than the far (bright) side.
   Dark outliers are image detail (a shadow, a dark feature) and are never
   flagged — a dab that contains only dark outliers falls back to the
   whole-stroke heal, which still removes deliberately painted dark content.
   `_AUTOMASK_K = 5.0` (grain lives well under 5 sigma; dust far over);
   `_AUTOMASK_ABS = 900` (16-bit, ≈1.4%) floors near-noiseless content.
5. **Hysteresis growth**: seeds grow through connected pixels above
   `_AUTOMASK_GROW = 0.35` of the seed threshold (bright side only) — the
   speck's soft halo joins the mask so no bright ring survives the heal;
   weak pixels not connected to a seed (stray grain) are dropped.
   **Brush vs auto**: brush growth is clipped to the stroke (the user's
   boundary is authoritative); an AUTO (AI) spot's circle is a machine
   guess, so its growth may follow the connected halo PAST the circle
   (bounded by the analysis window). Auto components also use a wider ring
   guard (2x thickness vs 0.5x): the circle hugs the speck, so its halo
   leaks into a near ring and would otherwise read as a background mode
   (stalling the growth and poisoning the heal's tone anchor).
6. **Gates** — shrink `C` only if:
   - a seed exists (brush AND auto); else keep `C` whole;
   - brush only: clean fraction `1 - grown_in_stroke/hole >=
     _AUTOMASK_CLEAN_MIN` (`0.4`) — a tight trace is mostly defect and must
     keep whole-stroke behavior (also guarantees in-stroke source area).
     Auto circles are mostly defect BY DESIGN, so this gate must not apply
     to them: gated whole-circle heals tone-anchor on the halo just outside
     the circle and leave a bright ring (the "auto detect doesn't work"
     report).
   (No second confidence threshold beyond `K*sigma` — resolved, §9.)
7. **Buffer**: dilate the grown mask by a disk of radius
   `buf = max(1, round(w/1080))` px — "+1px" at preview scale, proportional
   at export — then CLIP to `C`: the heal never reaches outside what the
   user painted (consistent with the stroke-feather semantics, §5.3).
8. Replace `C`'s pixels in the output mask with the buffered outlier mask.
   Several defects under one dab become several small components; the
   existing pipeline heals each independently.

### 5.3 Interaction with existing machinery

- **Feather = the stroke's soft edge** (user-corrected semantics): under
  automask the feather softens the STROKE's apply area — alpha ramps inward
  from the ORIGINAL brushed boundary, so the border of the stroke gets less
  effect than its center (capped per stroke by its depth so the center
  always reaches full effect). A defect hugging the stroke edge heals
  partially by design; brush fully over the dust for full removal. Two
  exceptions keep their full-strength fill: **auto (AI) spots** (tight
  circles that are mostly defect — stroke falloff there would leave halo
  rings; they keep the per-hole ramp + `dlike` force-fill) and **strokes
  that kept their whole-stroke heal** (tight traces — preserving the
  wide-feather-never-blends-the-defect-back guarantee).
- **Clone sources prefer the stroke**: `_heal_patch` scores candidates as
  before, but any candidate whose CLONED subregion lies fully inside the
  original selection wins over out-of-stroke candidates — the fill texture
  is sampled "from the non-masked part of the selection" whenever such a
  window exists (the matching ring may still look outside). Falls back to
  the unrestricted search, then Telea, as before.
- **Long strokes**: a traced hair that fails the clean-fraction gate flows
  through segmentation exactly as today. One that passes (loose wide trace)
  shrinks to the hair-shaped outlier mask — segmentation then works on the
  hair's own thickness, which is the geometry the segment heal was built for.
- **Telea fallback**: unchanged; shrunken components near borders/dense dust
  can still fall back per-patch.
- **Resolution independence**: the shrink re-runs per render resolution from
  the same normalized spots, like every other stage. Detection is driven by
  ring statistics that are stable across scales; the healed set may differ by
  edge pixels between preview and export — same class of approximation as
  `_heal_patch`'s per-resolution source choice (spec/dust-removal.md §5.4).

### 5.4 Contract (unchanged)

Only pixels under the rasterized selection can change — the shrunken mask's
buffer is clipped to the stroke (an earlier draft let it spill `buf` px past
the stroke; dropped when the feather became the stroke's soft edge, §5.3).

## 6. Integration points

| File | Change |
| --- | --- |
| `src/core/ccr_processor.py` | New `_automask_shrink(img16, mask)` + constants (`_AUTOMASK_K`, `_AUTOMASK_ABS`, `_AUTOMASK_CLEAN_MIN`); `DUST_METHODS` / `DUST_METHOD_DEFAULT`; `method=` parameter on `apply_dust_removal` (automask shrink / whole-stroke clone / whole-stroke Telea). |
| `src/core/ccr_image.py` | `_apply_dust_removal` passes the live `ccr_backend.dust_method`. |
| `src/core/ccr_backend.py` | `dust_method` attribute (default `"automask"`). |
| `src/ui/main_window.py` | Startup restore from QSettings (validated); `on_dust_method_changed` → persist + `_rerender_all_for_global_mode`. |
| `src/widgets/settings_dialog.py` | "Dust removal / Heal method" combo on the General page; staged like the other toggles (`_init_toggles` / `_apply_pending`). |
| `src/widgets/image_preview.py` | `dust_method` joins `_current_adj_sig` (hi-res cache invalidation). |
| `spec/dust-removal.md` | One-line cross-reference from §5.2 to this spec. |
| `tests/test_dust_removal.py` | `TestAutoMaskShrink`, `TestAutoMaskPipeline`, `TestHealMethodChoice` (see §7). |

No catalog or per-image changes.

## 7. Test plan

Helper-level (`_automask_shrink` directly, synthetic 16-bit fields with
seeded grain like the existing tests):

1. Generous dab around a bright speck → shrunken mask covers speck+buffer,
   excludes the rest of the dab (and nothing outside dab+buffer).
2. Dark speck variant (sign-agnostic detection).
3. Tight trace over a hair (stroke ≈ hair width) → mask returned unchanged
   (clean-fraction gate).
4. Dab over clean noise (no outlier) → unchanged (K·sigma gate).
5. Edge-straddling dab (bimodal surround), speck on the bright side → only
   the speck flagged; both clean sides of the edge unmasked (two-anchor
   model regression).

Pipeline-level (`apply_dust_removal`):

7. Generous dab: clean selection pixels bit-exact untouched; speck healed to
   surround level (tone within existing tests' tolerances).
8. Edge case from §1: dark-side pixels inside the dab bit-exact; healed speck
   lands at bright-side statistics, not pulled toward the dark side.
9. Two specks under one dab → both healed, pixels between them untouched.
10. Method choice (`TestHealMethodChoice`): `method="clone"` replaces the
    whole dab (clean dab pixels change) and skips the shrink;
    `method="inpaint"` fills through the 8-bit Telea path (fill values are
    multiples of 257 on a flat field — the signature the 16-bit-native test
    uses in reverse); an unknown method string behaves as `automask`;
    `apply_adjustments` honors `ccr_backend.dust_method` end-to-end.
11. Full existing `test_dust_removal.py` suite passes (grain preservation,
    gradient continuation, hair ghost, feather guarantees, fallback,
    persistence, undo, panel wiring). One exception by construction:
    `test_feather_param_softens_rim` asserts the feather's cross-fade at the
    DAB rim of a generous dab — with auto-mask that rim is no longer part of
    the hole (the very point of the feature), so it passes
    `method="clone"` to keep exercising the whole-stroke feather mechanics
    it was written for.

## 8. Rollout / debugging

- The Settings "Heal method" picker doubles as the support/A-B path: all
  three engines replay the same stored spots live, so a user can flip
  between them on their own scan and report which looks right. (An earlier
  draft used a `FREECCR_DUST_AUTOMASK` env knob; superseded by the picker.)
- No release-notes entry until the next release's "What's New" is written.

## 9. Refinement notes — resolved decisions

- **Outlier test instead of defect-color estimation.** First draft mirrored
  `_heal_patch`'s defect-color recipe (p75-of-deviation median). Rejected for
  the mask: with a small speck under a generous dab the top-quartile is
  dominated by grain and the "defect color" lands on background; and with an
  edge-straddling dab a single ring median flags the minority side of the
  edge as the defect. The two-anchor K·sigma outlier test handles any defect
  fraction and bimodal surrounds without estimating a defect color at all.
  (`_heal_patch`'s internal defect estimation stays as-is — its defensive use
  is tolerant of those biases.)
- **2-means with decile init over a median-luma split** for the anchors: the
  median split mis-models unbalanced bimodal rings (90/10 puts the minority
  mode inside one half; its clean pixels then read as outliers and a clean
  sliver of the dab gets pointlessly healed). Decile-initialized 2-means
  separates any split the ring can meaningfully witness (minority ≥ ~10%).
- **Automatic with fallback + a global Settings picker** (maintainer call,
  superseding the first draft's "no toggle, env knob only"): the default
  stays fully automatic, but Settings → General exposes the heal engine
  (Auto-mask / Whole stroke / Inpaint legacy) following the Auto Gain
  pattern — global, live, persisted, re-renders on change. The env knob was
  dropped: one control surface, and the picker is strictly more useful (it
  also restores the pre-v1.1 diffusion engine for users who preferred it).
- **Buffer scales with width** (`round(w/1080)`), not fixed 1px: a fixed 1px
  at 6000px export is proportionally 5× thinner than at preview, and defect
  soft edges scale with resolution.
- **No second confidence threshold** beyond `K*sigma` (draft had
  `median(d) >= 2K*sigma`): redundant with K and would reject legitimate
  faint-but-clear defects; K alone is the sensitivity dial.
- **Ring exclusion uses the ORIGINAL mask** during the pre-pass so results
  don't depend on component iteration order.

- **User-feedback round (2026-07-04)** — three corrections from field use:
  (1) *"feathering is to feather the apply area of the stroke — the border
  gets less effect than the center"* → feather became the stroke's soft
  edge under automask (§5.3), replacing the shrunken hole's own rim ramp,
  and the buffer is clipped to the stroke (§5.4). (2) *"sample from within
  the stroke to cover the outlier"* → in-stroke clone-source preference in
  `_heal_patch` (§5.3). (3) *"all dust are white spots/strings"* → the
  outlier test is bright-only (§5.2 step 4); dark content under a dab is
  preserved, dark-only dabs fall back to whole-stroke; plus hysteresis halo
  growth (§5.2 step 5) so auto-detected specks heal without ring remnants.
  Investigating (3) with the real ONNX model on synthetic film positives
  showed detection itself works at all brightness levels; the failures were
  heal-side — auto circles healed whole with halo-poisoned tone anchors
  (clean gate mis-applied to them) and halos outside the circle surviving.
  With auto growth + gate exemption the worst residual at detected spots
  dropped from ~2x grain to the pure-grain baseline (7705 vs 7692 in the
  probe). The in-stroke source preference needed a quality tolerance
  (`_HEAL_IN_STROKE_TOL`): an edge-straddling stroke's only in-stroke
  candidates can sit on the wrong side of the edge, and unconditional
  preference picked them over far better out-of-stroke sources.
- **Screenshot round (2026-07-04)** — a user screenshot of a hard-edged
  healed disc on sky: the feather was a fraction of IMAGE WIDTH (default
  0.3% ≈ 3 px at preview), constant regardless of brush size, so a big dab
  read as a sharp-edged patch. The Feather became a fraction of the
  STROKE's half-thickness (slider 0–100%, default 35%; catalog key
  `dust_feather_rel`, legacy values migrate proportionally) — big dabs
  feather proportionally wide, tight traces stay near-hard, and the ramp is
  resolution-independent by construction. Same round: "AI says no dust
  found" — the net fires on faint dust (0.999 prob on 3% strings) but the
  post-gates rejected it; recalibrated in dust_detect.py (adaptive bright
  margin, dust-string branch with polyline spots) — see
  spec/dust-removal.md §5.3.

### Open items (non-blocking)
- If grain speckle ever produces distracting micro-heals, add a minimum
  outlier-component area (≥2 px) before buffering. Not included: thin hairs
  are 1–2 px and must not be filtered out.
- A future "Whole stroke" per-stroke modifier (e.g. paint with Alt) if users
  ever want to force replacement of clean-looking areas; no evidence yet.

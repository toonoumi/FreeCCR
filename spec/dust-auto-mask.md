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
- Fully automatic: no new UI. Predictable fallback to current behavior when
  the outlier picture is not clear.
- Preserves every existing guarantee of `apply_dust_removal`: non-destructive,
  16-bit native, resolution-independent contract, only-masked-pixels-change
  (the buffer may extend ≤ buffer-radius px past the stroke, see §5.4),
  feather semantics, Telea fallback.

### Non-goals
- No change to spot storage (`dust_spots` stays `{kind, pts, r}` normalized),
  persistence, undo, or render-cache signatures — the shrink is a pure
  render-time function of (image pixels, rasterized mask).
- No new panel controls or overlay changes. The red stroke overlay continues
  to show the brushed area; with this feature its meaning is "heal what's
  dusty in here", which is what users already expect of a healing brush.
- No change to AI detection itself (`dust_detect.py`).
- Not a general blemish/content-aware eraser: detection is local-outlier
  based; large-area retouching stays out of scope.

## 3. UX / Interaction

Unchanged surfaces: panel, brush, feather slider, Ctrl+Z, AI detect flow.

Behavioral contract the user sees:
- Circle loosely around a speck/hair → only the defect disappears; grain and
  detail inside the circle survive. Feather applies to the (small) healed
  patch's rim.
- Trace a hair tightly (brush ≈ hair width) → identical to today: the whole
  stroke heals (the shrink pass declines: nearly everything under the stroke
  is defect).
- Dab over clean area (nothing to remove) → identical to today: the dab is
  replaced with a matched clone (harmless no-op visually). The shrink pass
  declines rather than silently doing nothing, so a user who paints over a
  *subtle* defect below the outlier threshold still gets it replaced.
- Escape hatch for A/B comparison and support: env `FREECCR_DUST_AUTOMASK=0`
  disables the shrink pass entirely (repo convention, cf. `FREECCR_DENSITY_*`).

## 4. Data model

None. No stored state changes. `dust_spots`, catalog persistence, undo
snapshots, and the hi-res `dust_sig` cache signature are untouched — the
shrink is deterministic per (pixels, mask), so cached renders remain valid
under the existing signatures.

## 5. Processing / math

### 5.1 Placement — one new pass in the single funnel

`apply_dust_removal(img16, spots, ...)` gains one line after rasterization:

```
mask = rasterize_dust_mask(spots, h, w)
mask = _automask_shrink(img16, mask)      # NEW — may return mask unchanged
... existing component/segment clone-heal pipeline, unchanged ...
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
4. **Outliers**: `like = d(hole_px) > max(_AUTOMASK_K * sigma, _AUTOMASK_ABS)`.
   `_AUTOMASK_K = 5.0` (grain lives well under 5 sigma; dust/hairs far over);
   `_AUTOMASK_ABS = 900` (16-bit units, ≈1.4% full scale) is a floor for
   near-noiseless synthetic/flat content where `5*sigma` degenerates to ~0.
5. **Gates** — shrink `C` only if BOTH hold, else keep `C` whole:
   - `like.any()` — something to target;
   - clean fraction `1 - mean(like) >= _AUTOMASK_CLEAN_MIN` (`0.4`): a tight
     trace is mostly defect and must keep whole-stroke behavior; also
     guarantees the local source search has in-selection clean area.
   (No second confidence threshold beyond `K*sigma` — resolved, §9.)
6. **Buffer**: dilate `like` by a disk of radius `buf = max(1, round(w/1080))`
   px — "+1px" at preview scale, proportionally larger at export so the
   healed fraction of the frame matches (the defect's soft edge also spans
   proportionally more pixels there). The dilation is NOT clipped to `C`: a
   defect touching the stroke edge gets its ≤`buf`-px halo healed too.
7. Replace `C`'s pixels in the output mask with the buffered outlier mask.
   Several defects under one dab become several small components; the
   existing pipeline heals each independently.

### 5.3 Interaction with existing machinery

- **Feather**: unchanged code. On a shrunken hole the ramp is capped by the
  (small) hole depth, so small heals stay near-hard — same rule as today's
  small dabs. The `dlike` force-fill keeps working (a shrunken hole is
  defect-like almost everywhere, so wide feathers still cannot blend the
  defect back).
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

### 5.4 Contract change (documented)

Today: only pixels under the rasterized stroke can change. With auto-mask:
pixels within `buf` px outside the stroke can also change (buffer of a defect
touching the stroke edge). Bounded, intentional, and listed in §2 Goals.

## 6. Integration points

| File | Change |
| --- | --- |
| `src/core/ccr_processor.py` | New `_automask_shrink(img16, mask)` + constants (`_AUTOMASK_K`, `_AUTOMASK_ABS`, `_AUTOMASK_CLEAN_MIN`); one call added in `apply_dust_removal` after rasterization; docstring update. |
| `spec/dust-removal.md` | One-line cross-reference from §5.2 to this spec. |
| `tests/test_dust_removal.py` | New `TestAutoMask` class (see §7); existing tests must pass unchanged. |

No UI, backend, catalog, or cache-signature changes.

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
6. `FREECCR_DUST_AUTOMASK=0` → unchanged even with a clear speck.

Pipeline-level (`apply_dust_removal`):

7. Generous dab: clean selection pixels bit-exact untouched; speck healed to
   surround level (tone within existing tests' tolerances).
8. Edge case from §1: dark-side pixels inside the dab bit-exact; healed speck
   lands at bright-side statistics, not pulled toward the dark side.
9. Two specks under one dab → both healed, pixels between them untouched.
10. Full existing `test_dust_removal.py` suite passes (grain preservation,
    gradient continuation, hair ghost, feather guarantees, fallback,
    persistence, undo, panel wiring). One exception by construction:
    `test_feather_param_softens_rim` asserts the feather's cross-fade at the
    DAB rim of a generous dab — with auto-mask that rim is no longer part of
    the hole (the very point of the feature), so the test pins
    `FREECCR_DUST_AUTOMASK=0` to keep exercising the whole-stroke feather
    mechanics it was written for.

## 8. Rollout / debugging

- `FREECCR_DUST_AUTOMASK=0` env knob (checked per call, so support can A/B a
  user's scan without rebuilding).
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
- **Automatic with fallback, no toggle.** Matches the panel's "it just works"
  design; the fallback keeps every current workflow (tight traces, subtle
  defects) byte-identical. Env knob covers support/A-B needs.
- **Buffer scales with width** (`round(w/1080)`), not fixed 1px: a fixed 1px
  at 6000px export is proportionally 5× thinner than at preview, and defect
  soft edges scale with resolution.
- **No second confidence threshold** beyond `K*sigma` (draft had
  `median(d) >= 2K*sigma`): redundant with K and would reject legitimate
  faint-but-clear defects; K alone is the sensitivity dial.
- **Ring exclusion uses the ORIGINAL mask** during the pre-pass so results
  don't depend on component iteration order.

### Open items (non-blocking)
- If grain speckle ever produces distracting micro-heals, add a minimum
  outlier-component area (≥2 px) before buffering. Not included: thin hairs
  are 1–2 px and must not be filtered out.
- A future "Whole stroke" per-stroke modifier (e.g. paint with Alt) if users
  ever want to force replacement of clean-looking areas; no evidence yet.

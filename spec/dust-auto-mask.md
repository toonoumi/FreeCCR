# Spec: Dust Auto-Mask (outlier-targeted heal within the brushed selection)

Extends `spec/dust-removal.md` §5.2. Prompted by user feedback: their previous
tool "checked and masked for outliers right in the selection and only replaced
the masked area plus 1px buffer. The sample was taken from the non-masked part
of the selection. That is especially useful when removing dust close to a
border (e.g. bright face to dark background)." Revised twice from field
testing (§9); the contracts below are the authoritative semantics.

## 1. The contracts

Maintainer-set rules this feature must never violate:

1. **Film dust is WHITE** — it blocks light on the negative, so every real
   defect inverts to a bright speck or string. Dark content under a
   selection is image detail and is preserved.
2. **A dab heals its outliers or does NOTHING.** No confident bright outlier
   under a dab → the dab is a no-op (matching the reporter's original tool).
   Never replace a whole dab "just in case" — that is how a sky dab near the
   film rebate became a black disc.
3. **Auto-mask samples from WITHIN the stroke.** The replacement texture for
   a dab's outliers comes from the clean, non-masked part of the same
   selection whenever a clone window fits there — an in-stroke candidate
   beats ANY out-of-stroke one, with no quality tolerance. When the dab's
   clean margin is geometrically too thin to host a window, the nearest
   TONE-COMPATIBLE content is used (every candidate must first pass the
   tone gate against the stroke's own clean interior — alien content like
   the black rebate can never win), else diffusion from the hole's in-stroke
   rim.
4. **The feather softens the stroke's apply area** — the ramp is a fraction
   of the stroke's own half-thickness (border gets less effect than the
   center), so big dabs feather proportionally wide and tight traces stay
   near-hard.
5. **Spots heal independently, in order.** Each spot is healed on the
   already-healed result of the spots before it (sequential replay). Adding
   or removing a later spot never changes an earlier heal — clicking
   adjacent dust must not affect its neighbors.
6. **Deliberate whole-area replacement is an explicit choice**: a TRACE
   gesture (path much longer than the brush radius — the user outlined the
   defect itself) heals the whole stroke, and the Settings "Whole stroke" /
   "Inpaint" engines always do. Tonally alien sources (the black rebate next
   to sky) are rejected everywhere.

## 2. Goals / Non-goals

### Goals
- A dab loosely circling white dust heals exactly the dust (+ small
  buffer); the rest of the selection is bit-for-bit untouched.
- Heals near high-contrast borders stay on the correct side: fill texture
  comes from within the selection.
- Auto (AI) spots heal their speck AND its soft halo with no bright ring
  remnant.
- Stable editing: heals never shift as more spots are added.
- Fully automatic; no new panel controls. Settings → General exposes the
  engine choice (Auto-mask / Whole stroke / Inpaint legacy).

### Non-goals
- No change to spot storage (`dust_spots` = `{kind, pts, r}` normalized),
  undo, or AI detection interfaces. (Detection gate recalibration lives in
  spec/dust-removal.md §5.3.)
- No per-image engine choice; global, like Auto Gain.
- Not a content-aware eraser for arbitrary content: removing something that
  is not a bright outlier is the trace gesture's or the Whole-stroke
  engine's job.

## 3. UX / Interaction

Unchanged surfaces: dust panel, brush, Ctrl+Z, AI detect flow.

**Settings → General → "Dust removal" → "Heal method"** (staged combo,
applied on Done; QSettings `adjust/dust_method`; re-renders all images on
change since spots replay live through the chosen engine):

| Choice | id | Engine |
| --- | --- | --- |
| Auto-mask (default) | `automask` | contracts §1 |
| Whole stroke | `clone` | clone-heal the entire stroke (v1.1) |
| Inpaint (legacy) | `inpaint` | cv2.inpaint diffusion (pre-v1.1) |

Behavior under Auto-mask:
- **Dab** (click or short drag) around dust → only the dust disappears.
  Dab over clean area → nothing happens, and the panel SAYS SO (a status
  hint via `dust_spot_effect_px`; a silent no-op reads as a broken brush)
  with pointers to the trace gesture and the Whole-stroke engine.
- **Trace** (drag along a hair, path length > ~2.5x brush radius) → the
  whole stroke heals, as in v1.1 (that gesture means "replace exactly
  this").
- **Feather slider** (0–100% of stroke half-thickness, default 35%):
  softens the stroke's apply area (dab border heals less than its center).
  Auto spots and traces keep full-strength defect fill regardless.

## 4. Data model

Per-image: none beyond `dust_feather` (0..1, fraction of stroke
half-thickness; catalog key `dust_feather_rel`, legacy width-fraction
`dust_feather` migrates proportionally on load).
Global: `ccr_backend.dust_method` (validated against `DUST_METHODS`,
persisted by MainWindow, in `_current_adj_sig` for hi-res invalidation).

## 5. Processing / math

### 5.1 Sequential replay (contract 5)

`apply_dust_removal(img16, spots, ...)` folds spots one at a time, in stored
order, into a working copy: `work = heal_one_spot(work, spot)`. A spot's
result depends only on itself and PRIOR spots — never on later ones — so
render-time replay reproduces exactly what the user saw when they placed
each spot, and new spots cannot reshuffle old heals. Source-cleanliness
exclusion is each spot's OWN mask only: earlier dust is already healed in
`work`; later spots did not exist when this one was placed.

### 5.2 Per-spot analysis (automask engine)

For each brush-spot component (window/ring construction as `_heal_patch`:
thickness-scaled guard + ring, 1px-padded distance transforms):

1. **Two-anchor surround model** — 2-means over ring pixels, anchors
   initialized at the bottom/top luma-decile medians, 2 rounds. Unimodal
   ring → anchors coincide; bimodal (dab straddling an edge) → the two
   modes, even for unbalanced splits a median split mis-models.
2. **Bright outlier seeds** (contract 1):
   `seed = d > max(K*sigma, ABS)` AND brighter than the NEARER anchor
   (dust on the dark side of an edge counts even when darker than the far
   side), AND part of a connected cluster of `_AUTOMASK_SEED_MIN=4` px —
   dust is a shape; lone bright grain pixels are noise. `_AUTOMASK_K=4`
   (calibrated on the maintainer's real Portra scans: smooth-area dust
   measures 7–40x the local scatter, dust buried in texture 1–3x — locally
   indistinguishable from texture/clouds, which is the AI detector's job to
   find), `_AUTOMASK_ABS=900/65535`, sigma = ring median distance to
   nearest anchor.
3. **Hysteresis growth**: seeds grow through connected bright pixels above
   `_AUTOMASK_GROW=0.35` of the seed threshold (covers the soft halo; stray
   grain not connected to a seed is dropped). Brush growth is clipped to
   the stroke; an AUTO spot's circle is a machine guess, so its growth may
   follow the halo past the circle (bounded by the analysis window), and
   auto components use a 2x-thickness ring guard so their own halo doesn't
   read as a background mode.
4. **Buffer**: dilate by `max(1, round(w/1080))` px, clipped to the stroke
   for brush spots.
5. **No seeds** → brush dab: NO-OP (contract 2); auto spot: no-op as well
   (the detector's own gates should prevent this arising).

### 5.3 Gesture: dab vs trace (contract 6)

From the SPOT DATA, not pixels: `path_len = Σ|pts[k+1]-pts[k]|` (normalized
units). `path_len > _TRACE_LEN_R x r` (2.5) → TRACE → the whole stroke
heals through the v1.1 clone path (with `dlike` force-fill so wide feathers
cannot blend the traced defect back). Otherwise DAB → §5.2 outliers-or-no-op.
Intent-faithful and image-independent — a tight trace over a hair whose
leak contaminates the local ring (making the hair read as "background")
still heals, because the user's gesture said so.

### 5.4 Sampling (contract 3)

- All candidates pass the TONE GATE below before anything else; then a dab
  prefers candidates whose CLONED subregion lies fully inside the original
  stroke — an in-stroke candidate beats any out-of-stroke one (no quality
  tolerance; the tone gate already vetted both). A dab too small to host an
  in-stroke window uses the nearest tone-compatible content (preserving
  grain — a Telea-only rule here failed the 16-bit/grain quality bars);
  no candidate at all → Telea diffusion from the hole's in-stroke rim.
- Auto spots: sources come from the defect's neighborhood (no stroke to
  constrain to) with the tone gate below.
- Traces + Whole-stroke/Inpaint engines: neighborhood sources as v1.1.
- **Tone gate (all clone paths)**: reject any candidate whose cloned
  content's median deviates from the tone anchor(s) by more than
  `max(_HEAL_TONE_ABS=8000, _HEAL_TONE_K=12 x scatter)`. Anchoring is
  STRUCTURAL (defect estimates proved too fragile to anchor on):
  ring-anchored paths (shrunken dabs, traces, auto spots — holes that are
  mostly defect) accept only trimmed-ring-compatible sources; the
  whole-stroke clone engine runs TWO TIERS — sources compatible with the
  hole's own majority are preferred (a dab beside the dark frame border
  has a majority-alien ring, and a ring-matched source filled sky with
  the border's dark interior on a real scan), ring-compatible sources are
  the fallback (an underscoped dab's hole is mostly defect; the ring is
  the honest anchor there). This is what makes cloning the rebate (or its
  frame-number digits) into sky impossible.
- **Force-fill needs a distinct, and (clone engine) minority, defect**:
  on real clean-ish content the defect estimate degenerates to the
  background mode and `like` swallows the hole — forcing it defeated the
  feather on real scans ("still not feathered"). Traces stay ring-anchored
  with forcing, preserving the wide-feather no-blend-back guarantee.

### 5.5 Feather & composite (contract 4)

Per spot: alpha for a brush stroke ramps inward from the ORIGINAL stroke
boundary over `feather x stroke_depth` px (smoothstep; uint8 fmap capped at
250). Auto spots use the shrunken hole's own ramp, and defect force-fill
(`dlike`) applies to auto spots and traces only. Blend runs inside the
spot's bounding box; alpha is exactly 0 outside the spot's masks, so
untouched pixels are bit-for-bit identical (float32 round-trip of uint16 is
exact).

## 6. Integration points

| File | Change |
| --- | --- |
| `src/core/ccr_processor.py` | Dust section refactor: sequential `apply_dust_removal` driver + `_heal_one_spot`; `_automask_outliers` (analysis), `_heal_patch` (clone core with in-stroke constraint + tone gate), Telea fallback, per-spot composite. Constants: `_AUTOMASK_*`, `_TRACE_LEN_R`, `_HEAL_TONE_*`, `DUST_METHODS`, `DUST_FEATHER_DEFAULT`. |
| `src/core/ccr_image.py` | `_apply_dust_removal` passes live `ccr_backend.dust_method`. |
| `src/core/ccr_backend.py`, `src/ui/main_window.py`, `src/widgets/settings_dialog.py` | Engine setting (attr, restore/persist/re-render, staged combo). |
| `src/widgets/image_preview.py` | `dust_method` in `_current_adj_sig`. |
| `src/widgets/dust_panel.py` | Feather slider 0–100% of stroke. |
| `src/core/dust_detect.py` | Gate recalibration + string polylines (spec/dust-removal.md §5.3). |
| `src/core/catalog.py` | `dust_feather_rel` + legacy migration. |
| `tests/test_dust_removal.py` | Rebuilt suite mirroring the contracts (§7). |

## 7. Test plan (mirrors the contracts)

1. Contract 1: dab over bright speck heals it; dark blob under the same dab
   preserved bit-exact; dark-only dab is a NO-OP under automask (Whole
   stroke engine still removes it).
2. Contract 2: dab over clean grain is a bit-exact no-op.
3. Contract 3: dab beside a black band whose only out-of-stroke candidates
   are the band → heal stays sky-toned (never black); edge-straddling dab
   heals its speck at the correct side's statistics.
4. Contract 4: feather widens with the stroke; border defect heals less
   than center defect at high feather; feather 0 heals both fully.
5. Contract 5: healing [A] then adding B leaves A's healed pixels
   bit-identical; order determinism.
6. Contract 6: a trace over a hair heals the whole stroke (hair gone, no
   ghost, wide feather cannot blend it back).
7. Auto spots: speck+halo → no bright ring remnant; two-anchor bimodal
   surround; halo growth past the circle.
8. Engines: whole-stroke replaces clean dab pixels; inpaint fills with the
   8-bit Telea signature; unknown method → automask; backend setting drives
   `apply_adjustments`; tone gate applies to whole-stroke too.
9. Detection gates (model-free `prob_to_spots`): adaptive margin passes
   faint blobs on quiet sky and rejects them in heavy grain; thin bright
   strings kept as polyline spots (area-cap exempt, frame-edge guard);
   thick elongated dropped; dark blobs dropped.
10. Persistence: spots + `dust_feather_rel` round-trip; legacy migration;
    undo snapshot independence.
11. Legacy pipeline invariants (kept from PRs #81/#82): 16-bit-native fill,
    grain preservation, gradient continuation, no input mutation, far
    pixels untouched, Telea fallback when no clean source window exists.

## 8. Rollout / debugging

The Settings engine picker is the support/A-B path: all three engines
replay the same stored spots live. No env knobs.

## 9. History — how the contracts were learned

- v1 shipped shrink-or-whole-stroke with out-of-stroke sampling preferred
  by ring-SSD. Field testing produced: hard-edged discs (feather was a
  fixed ~3px image-width fraction), "AI says no dust found" (detection
  post-gates rejected faint dust and strings the net had found at 0.999),
  BLACK discs on sky (whole-stroke fallback + ring-SSD choosing film-rebate
  sources when the dst ring itself was majority-rebate), and neighbor heals
  reshuffling on every new click (simultaneous healing from one merged
  mask). Each failure became a contract in §1.
- Rejected: defect-color estimation for the outlier mask (grain-dominated
  top-quartile for small specks; single ring median mis-models bimodal
  surrounds); unconditional in-stroke source preference (an edge-straddling
  stroke's in-stroke candidates can sit on the wrong side — superseded by
  the strict in-stroke constraint for dabs plus tone gate everywhere);
  whole-stroke fallback for outlier-less dabs (contract 2 forbids it; the
  trace gesture covers deliberate replacement); clean-fraction gates on
  auto spots (their circles are mostly defect BY DESIGN — gating them left
  halo-poisoned whole-circle heals).
- Real-scan round (2026-07-04) — "no dust found / the stroke does nothing /
  still not feathered; use a real image as test target." All synthetic
  tests passed while the maintainer's real Portra scans failed, exposing
  four gaps: (a) K=5 seeds missed real dust (dust in texture measures only
  1–3x the local scatter; smooth-area dust 7–40x) → K=4 + 4-px seed
  clusters, with the honest limit documented: dust buried in texture is
  locally indistinguishable from clouds and belongs to the AI detector
  (which finds 10–29 spots per real frame); (b) silent no-op dabs read as
  a broken brush → panel status hint via `dust_spot_effect_px`; (c) the
  degenerate dlike force-fill defeated the feather on real content; (d)
  the tone reference degenerated on clean/edge dabs, letting a ring-matched
  source fill sky with the dark frame border's interior → structural
  two-tier anchoring. A crop of the real converted scan is committed as
  `tests/data/real_scan_sky.png`, and `TestRealScan` pins
  dab-heals-real-dust + clean-dab-no-op on it — thresholds can no longer
  drift on synthetic passes alone.

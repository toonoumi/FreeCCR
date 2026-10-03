# Positive-Mode Base Render Curve

> Status: experiment (evaluating the shape and strength on real rolls).
> Supersedes the `gamma_base` draft (a single midtone node) and the +2 EV
> exposure-shift experiment before it — both rejected by measurement, see below.
> Related: spec/positive-mode.md (the neutral baseline this revises),
> spec/curves-tone-control.md (the machinery it rides),
> spec/mono-positive-render.md (the monochrome path this curve is fitted on,
> and the magenta cast that invalidated the first fit).

## Summary

A positive decode starts from a **base render curve** — an S-curve with a toe, a
midtone lift and a shoulder — applied through the existing Curves path. The
sliders all still read 0; the user trims from a normal-looking image instead of
from a flat, dark one.

This is the stage that makes Lightroom look "normal with no adjustments" and
FreeCCR look dark: **a transfer function is not a rendering**. BT.709/sRGB
encodes the data; it does not place the tones. Every raw developer applies a
base curve on top — Adobe's `ProfileToneCurve`, darktable's base curve.

## The problem, measured

Positive decode, no adjustments, versus the **camera's own embedded JPEG** (the
manufacturer's intended look, and roughly what Adobe's profiles target), on a
16-frame roll:

| percentile | ours | camera | |
|---|---|---|---|
| p10 | 0.030 | 0.020 | shadows **deepened** |
| p25 | 0.098 | 0.075 | |
| p50 | 0.243 | 0.537 | midtones **2.2× up** |
| p75 | 0.465 | 0.753 | |
| p90 | 0.617 | 0.820 | highlights compressed toward white |

The implied gamma swings from 1.12 in the shadows to 0.37 in the mids — so this
is an **S-curve, not a gamma**. That is what rules out the previous draft: a
single midtone node lifts the shadows, where the real render darkens them.

## Why not the alternatives

- **Exposure gain (+2 EV, `exp_shift`).** These frames already have their white
  point placed — p99 at 1.000 with 0.5–6.2% of pixels clipped, because the clear
  film base and speculars reach the sensor ceiling. A gain drives those clipped
  highlights further into the ceiling: measured **+2 EV clipped 48–60%** of all
  pixels across the roll.
- **libraw auto-brightness.** It normalises the *highlight*, which is already at
  the top. Measured a **no-op on every frame** (1.00×, identical clipping); it
  only engages once a frame is underexposed by hand.
- **A single gamma node.** Wrong shape (see above). Also needed +27…+87 slider
  units depending on the frame, with one frame unreachable even at +100.
- **Adobe's real `ProfileToneCurve`.** FreeCCR already parses it
  (`dcp_profile.py`) but never applies it (`apply_look=False`), and positive mode
  skips DCPs entirely. The only profile in the test library is FreeCCR's own
  IT8-generated one, which carries no tone curve. Viable per-camera route later;
  nothing to use today.

## The curve

Fitted to the measured ours→camera transfer as a **log-domain sigmoid**,
pivot 0.300, k 1.80, rmse 0.036, then expressed as control points for the
Curves editor's monotone cubic:

```
[[0,0], [12.8,2.5], [30.6,22.6], [63.8,102.2],
 [102,180.6], [153,228.4], [204,246.9], [255,255]]
```

**Fitted against the NEUTRAL monochrome decode.** The first fit used the colour
decode, whose heavy magenta cast (see spec/mono-positive-render.md) biased it
about 0.09 too dark at the median; it gave pivot 0.280 / k 1.55 and is
superseded. Comparing a neutral mono decode against the camera's B&W JPEG is an
apples-to-apples match.

**Deliberately a smooth parametric fit, not the raw histogram match.** Matching
the histograms directly demanded a slope of **3.4** through the midtones — an
artifact of the camera's per-frame auto tone acting on bimodal histograms.
Baking that would posterise the mids. The parametric fit's max slope is **2.63**.

Verified per frame, the mono decode through the curve versus the camera's render:

```
9904  0.310 -> 0.544   camera 0.537
9905  0.352 -> 0.628   camera 0.655
9914  0.376 -> 0.670   camera 0.702
9916  0.279 -> 0.474   camera 0.439
```

And end-to-end through the real pipeline (`read_image` + `apply_adjustments`):

```
9904  0.535   camera 0.537      (colour+positive today: 0.383)
9923  0.484   camera 0.463      (today: 0.390)
9915  0.705   camera 0.741      (today: 0.579)
```

## Baked brightness baseline

Positives also carry a baked `brightness_base`, the way negatives carry their
`-8` film-look offset. `POSITIVE_BASE_BRIGHTNESS = 15` is expressed in
Brightness-**slider** units so it reads as the number a user would dial in by
hand; `apply_adjustments` computes `0.5*slider + base` — the slider is half
strength while the base is full weight — so the stored value is **half** the
slider number it reproduces. `FREECCR_POSITIVE_BRIGHTNESS` overrides it, also
in slider units.

`brightness_base` is an **integer**, so an odd slider value cannot be hit
exactly. The conversion **truncates** rather than rounds — 15 → 7.5 → stored as
7, the slider-14 equivalent, half a step *darker* than asked. That direction is
deliberate: rounding up would quietly return a brighter render than the number
requested, and the measured difference either way peaks at 0.02, well below
visible. Even values are exact — 20 → 10 rendered bit-identically to the slider
at 20.

Keeping it an integer is deliberate too: `catalog.serialize_image` stores
`int(img.brightness_base)`, so a float base would be truncated on restore and
the live and restored renders would silently disagree.

This is the third revision of spec/positive-mode.md §4.2a's "neutral baseline":
measured renders kept landing darker than the user's eye wanted, and the lift
needed is a plain brightness offset rather than more curve or more gain.

Known limitation: `catalog._restore_image` restores a stored `brightness_base`,
so positives already in the catalog keep their previous value and only fresh
imports pick up the new baseline.

## Data Model

- `CCRImage.base_curve: int` — strength in percent. Set in `__init__` and
  `reload_image_decode_only` to `_positive_base_curve_strength()` in Positive
  mode, else `0`; re-derived whenever the mode is toggled, like
  `brightness_base`.
- `POSITIVE_BASE_CURVE_STRENGTH = 100`, overridable with
  `FREECCR_POSITIVE_CURVE` (clamped to 0–100); **0 disables it entirely**,
  restoring the previous decode exactly for a clean A/B.
- `ccr_processor.POSITIVE_BASE_CURVE` holds the control points;
  `positive_base_curve_points(strength)` blends them toward the identity
  diagonal, so every strength is still monotone.
- Serialised as `"base_curve"` beside the other bases. Purely additive — an
  absent key restores as whatever the decode set.

## Processing

In `apply_adjustments`, **before** the user's Gamma and Curves:

```python
if bc:
    adjusted = apply_curves(adjusted, {"rgb": positive_base_curve_points(bc)})
```

- It runs first because it is the *rendering transform*: the user's Gamma and
  Curves then operate on display-referred data, which is what they expect.
- `bc` joins the no-adjustments early-return guard — otherwise a fresh positive
  with no sliders would short-circuit straight past its own render curve.
- **Not** applied in `_adjust_for_area`: area layers deliberately zero the base
  offsets, because those are global look already baked into the layer an area
  composites onto.
- Threaded through an `apply_adjustments` override and snapshotted by the hi-res
  zoom worker, like every other base — otherwise a zoomed tile renders without
  the curve and stops matching the preview.

## Non-Goals

- **No per-frame auto.** This is a fixed render, like Lightroom's. A dark frame
  stays dark — `DSC_9912` (median 0.028) renders at 0.006 here and 0.016 in the
  camera's own JPEG. Lr with no adjustments would leave it dark too.
- **No highlight reconstruction.** Frames whose p99 is already clipped have more
  scene range than the output holds; the shoulder compresses, it cannot recover.
- **No change to negatives.** `base_curve` is 0 there; the conversion and Auto
  Gain own that look.
- **Not per-camera.** One fitted curve, from one body. A per-camera route exists
  (the DCP tone curve) and is deliberately left for later.

## Edge Cases

- **Strength 0** is an exact identity — no curve object is built at all.
- **Toggling Positive mode** re-decodes; `base_curve` is re-derived in
  `reload_image_decode_only`, so it appears/disappears with the mode.
- **Catalogs written before this** have no `base_curve` key and restore as the
  decode set it.
- **A B&W-profile or monochrome image** is unaffected in kind — the curve is
  applied per channel on neutral data, so it stays neutral.

## Test Plan

`tests/test_positive_base_curve.py`:

1. `positive_base_curve_points`: 0 → the identity diagonal, 100 → the fitted
   points, 50 → halfway; monotone at every strength; endpoints always pinned.
2. `base_curve` is `POSITIVE_BASE_CURVE_STRENGTH` in Positive mode, `0`
   otherwise, at construction and after a reload; env override clamps and falls
   back on garbage.
3. **Shape** (what distinguishes it from a gamma): black and white pinned,
   midtones lifted, **deep shadows deepened**.
4. A positive with no sliders renders brighter overall (the early-return guard
   does not swallow it); a negative is untouched.
5. The override parameter wins over the attribute; area layers do not re-apply it.
6. Catalog round-trips `base_curve`; a record without the key is safe.
7. The hi-res zoom signature changes with `base_curve`.

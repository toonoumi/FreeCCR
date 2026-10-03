# Positive-Mode Base Tone Curve

> Status: experiment (evaluating the default value on real rolls).
> Related: spec/positive-mode.md (the neutral baseline this revises),
> spec/gamma-slider.md (the curve this rides), spec/auto-gain.md (the
> negative-mode counterpart).

## Summary

A positive decode starts from a **baked Gamma-slider offset** (`gamma_base`,
default **+50**) the same way a negative starts from `brightness_base = -8`.
The slider still reads 0; the user trims from a sensible baseline instead of
hauling the midtones up by hand on every frame.

## The problem, measured

The positive decode maps **sensor saturation** to display white and has no
auto-exposure anywhere — libraw's auto-bright is off, the preview stretch is
skipped for positives, and Auto Gain is gated on `converted`. BT.709 is a
*transfer function*, not a *rendering* curve, and nothing supplies the base
curve that every other raw developer applies on top of it.

Measured on a 16-frame roll (`20261002BWNefTest`), positive decode, no
adjustments:

| | |
|---|---|
| median tone | 0.028 – 0.331 |
| Gamma slider needed to reach 0.45 | **+27 … +87, median +54** |
| frames where +100 is still not enough | 1 of 16 (`DSC_9912`) |

That is the user-visible complaint — "I have to pull Gamma almost all the way
up" — reproduced as a number.

## Why a curve, not a gain

The first attempt was a +2 EV exposure shift (libraw `exp_shift`). It is the
wrong operator for this data, and the measurement says so plainly:

- These frames **already have their white point placed** — p99 sits at 1.000
  with 0.5–6.2% of pixels clipped, because the clear film base / speculars reach
  the sensor ceiling.
- A gain multiplies everything, so it drives those already-clipped highlights
  further into the ceiling in order to lift the midtones: **+2 EV clipped
  48–60% of all pixels** across the roll.
- Auto-brightness cannot help either: it normalises the *highlight*, and the
  highlight is already at the top. Measured, it is a **no-op** on every frame of
  this roll (1.00×, identical clipping); it only engages once a frame is
  artificially underexposed.

With black and white correctly placed and only the middle low, the operator that
fits is the one that **pins both endpoints and bends the middle** — a gamma
curve. Both highlight-anchored mechanisms (libraw auto-bright, FreeCCR's Auto
Gain) answer "where is white?" when the question here is "where is middle grey?"

## Data Model

- `CCRImage.gamma_base: int` — set in `__init__` and `reload_image_decode_only`
  to `_positive_base_gamma()` in Positive mode, else `0`. Exactly the shape of
  `brightness_base`, so it is re-derived whenever the mode is toggled.
- `POSITIVE_BASE_GAMMA = 50`, overridable with `FREECCR_POSITIVE_GAMMA` while
  the value is being evaluated; **0 disables it entirely**, restoring the
  previous decode exactly (a clean A/B).
- Serialised as `"gamma_base"` and restored alongside the other bases. Purely
  additive — an absent key restores as whatever the decode set, so no catalog
  migration.

## Processing

In `apply_adjustments`, the base rides the **existing** Gamma stage:

```python
gamma = max(-100, min(100, s.get('gamma', 0) + gmb))
```

- It is the user's slider **plus** the baseline, clamped to the slider's
  designed `[-100, 100]` domain. The monotone-cubic curve stays monotone outside
  that range, but the node geometry is only calibrated inside it.
- `gmb` joins the no-adjustments early-return guard — otherwise a fresh positive
  with no sliders would short-circuit straight past its own base curve.
- **Not** applied in `_adjust_for_area`: area layers deliberately zero the base
  offsets, because those are global-look values already baked into the base an
  area composites onto.
- Threaded through the `apply_adjustments` override parameter and snapshotted by
  the hi-res zoom worker, like every other base — otherwise a zoomed tile would
  render without the curve and stop matching the preview.

## Non-Goals

- **No auto/measured per-frame placement.** This is a fixed baseline, like
  `brightness_base`. A measured grey placement is the natural follow-up and is
  deliberately kept separate.
- **No highlight shoulder.** `DSC_9912` (median 0.028, p99 already clipped) has
  more scene range than the output can hold; no endpoint-pinned curve fixes that
  frame, and it needs a shoulder/roll-off that is its own piece of work.
- **No change to negatives.** `gamma_base` is 0 there; the conversion and Auto
  Gain own that look.
- **Not Adobe's `ProfileToneCurve`.** FreeCCR already parses it
  (`dcp_profile.py`) but never applies it (`apply_look=False`), and positive
  mode skips the DCP entirely. Wiring that up is a separate, per-camera route to
  the same goal.

## Edge Cases

- **A fixed baseline cannot serve every frame.** The roll needs +27…+87; at +50
  a frame wanting +27 is over-lifted and one wanting +87 is still short. That is
  expected of a base curve — it gets every frame into the ballpark instead of
  starting near-black, and the slider covers the rest.
- **Slider saturation.** A user at +100 with a +50 base clamps to +100, so the
  base can never *reduce* the reachable range.
- **Toggling Positive mode** re-decodes every image, and `gamma_base` is
  re-derived in `reload_image_decode_only`, so it appears/disappears with the mode.
- **Catalogs written before this** have no `gamma_base` key and restore as the
  decode set it.

## Test Plan

`tests/test_positive_base_gamma.py`:

1. `gamma_base` is `POSITIVE_BASE_GAMMA` in Positive mode and `0` otherwise, at
   construction and after a reload.
2. `FREECCR_POSITIVE_GAMMA` overrides it; `0` disables; garbage falls back.
3. A positive image with **no sliders** renders brighter than its input (the
   early-return guard does not swallow the base), while a negative is untouched.
4. The base and the slider **sum**, and the sum clamps at ±100.
5. Area layers do not re-apply the base.
6. The catalog round-trips `gamma_base`, and a record without the key is safe.
7. The hi-res zoom signature changes with `gamma_base`, so a stale tile is
   invalidated.

# Monochrome Rendering for a Mono-Converted Camera

> Status: experiment (validated against the camera's own JPEG on one body).
> Extends spec/monochrome-raw-mode.md (the forced mono read) and
> spec/positive-base-curve.md (the base render curve this feeds).

## Summary

Three changes that together make a mono-converted camera's RAW open looking
*normal* — matching the camera's own rendering — instead of dark and magenta:

1. **Per-phase gain normalisation** in the forced monochrome read, removing the
   residual CFA checkerboard.
2. **Monochrome + Positive mode is display-referred**: the mono read encodes to
   sRGB in Positive mode instead of returning scene-linear data.
3. The **base render curve** (spec/positive-base-curve.md), re-fitted against
   the now-neutral monochrome decode.

## What was actually wrong

The reported symptom was "too dark; I have to pull Gamma almost all the way up".
Measured against the camera's embedded JPEG (the manufacturer's intended look,
and roughly what Adobe's profiles target), two separate faults were present:

**The decode was magenta.** Not a tone problem at all — it dominated the image
and no curve could fix it:

- The sensor is **mono-converted**: the R/B ratio varies **0.8%** across 115
  patches of the frame (a live colour CFA on a known-good body varies **50.5%**).
  There is essentially no scene colour information, only a fixed offset.
- But the file still reports a normal Bayer CFA and normal daylight white
  balance (`camera_whitebalance ≈ [1.906, 1.0, 1.391]`).
- So the near-neutral raw (R/G 1.15) goes through the camera matrix (→ 1.38) and
  the WB multipliers (→ ~2.9) and comes out heavily magenta.

The monochrome read already fixes this by construction — it bypasses both the
CFA interpretation and the colour matrix.

**Monochrome and Positive mode did not compose.** `is_monochrome` forced
`positive_decode = False`, so the mono read returned scene-linear data in
Positive mode. The display-referred base curve then landed on linear values and
**darkened** them: measured median 0.0785 → **0.0589**, against a 0.537 target.

## 1. Per-phase gain normalisation

Stripping a CFA is never perfect; each of the four photosite phases keeps a
slightly different sensitivity. Measured on the test body:

```
phase means   1577.7 / 1357.6 / 1350.0 / 1496.8      spread 15.75%
after scaling each phase to their common mean         spread  0.00%
```

It is a **fixed, purely multiplicative sensor property**, not scene content:
across 16 frames of very different subjects the per-phase gains varied by only
**0.23–0.64%**. So it is computed **per frame** (self-calibrating to any body,
and stable enough that it cannot flicker across a roll) rather than baked.

`ccr_merge.normalize_cfa_phases(plane)` — pure, unit-testable. Applied in
`CCRImage._read_mono_mosaic` **before `bin2x2`**: binning averages the four
phases and therefore hides the checkerboard in previews while it survives at
full resolution, where zoom and export would show it.

**This revises a non-goal.** spec/monochrome-raw-mode.md said "No flat-field /
per-site gain correction of residual CFA response". That was written before
there was evidence the residual exists and is fixed; it does, and correcting it
is what makes the mono read usable on a converted body. The forced mono read is
the only caller — a true monochrome sensor takes the libraw postprocess branch
and is untouched, as is the trichrome merge.

## 2. Monochrome + Positive mode

`mono_positive = positive_mode and is_monochrome`, deliberately a **separate
flag** rather than folding it into `positive_decode`. `positive_decode` also
gates the white-level scaling (which this path still needs), the field
correction's `encoded=` hint, and the camera-profile skip — all of which stay
correct as they are.

The sRGB encode runs **after** the white-level scaling, which is the one point
where the buffer is both full-range and still linear. Negative mode is
unchanged: it keeps the scene-linear values the density math requires.

## 3. The re-fitted base curve

The curve in spec/positive-base-curve.md was originally fitted against the
**colour** decode, whose magenta cast biased it ~0.09 too dark at the median.
Re-fitted against the neutral monochrome decode: log-domain sigmoid, pivot
0.300, k 1.80, rmse 0.036, max slope 2.63.

## Result

End-to-end through the real pipeline, versus the camera's own JPEG:

| frame | colour+positive (before) | mono+positive+curve | camera |
|---|---|---|---|
| DSC_9904 | 0.383 | **0.535** | 0.537 |
| DSC_9923 | 0.390 | **0.484** | 0.463 |
| DSC_9915 | 0.579 | **0.705** | 0.741 |

## Non-Goals

- **No auto-detection of a mono-converted body.** The metadata actively lies
  (it reports RGGB + a daylight WB), which is why the toggle is manual. The
  R/B-variance test used here (0.8% vs 50.5%) would be a sound basis for
  detection later, but it is not wired up.
- **No per-frame auto tone.** The render is fixed, like Lightroom's; a genuinely
  dark frame stays dark.
- **No change to negatives, true mono sensors, or the trichrome merge.**

## Test Plan

`tests/test_mono_positive_render.py`:

1. `normalize_cfa_phases` flattens a synthetic fixed pattern to ~0 spread,
   preserves the mean, preserves real detail (it is a gain, not a blur), is a
   no-op on an already-flat plane, does not mutate its input, survives odd
   dimensions and an all-zero plane, and rejects a 3-D array.
2. The mono read applies it **before** binning, and the binned preview still
   reports the canonical full size.
3. Monochrome + Positive mode comes out sRGB-**encoded** (matching
   `srgb_encode` on the linear value, and markedly brighter), while negative
   mode keeps the scene-linear value.

`tests/test_monochrome_raw.py` is updated: the photosite-exactness assertion now
includes the normalisation (its fixture is a ramp, whose phases genuinely
differ, so the correction is large there — on a real frame it is small).

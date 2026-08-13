# Spec: White balance from a range of neutral patches

Status: REFINED — ready to implement
Owner: FreeCCR
Feature branch: `feat/wb-neutral-range-average`

## 1. Problem

`fit_camera_matrix` derives the camera fit's white balance from **one** patch:

```python
chosen_wb = _pick_wb_id(samples, ref, wb_id)     # lightest patch with Lab chroma < 6
n_raw = np.clip(samples[chosen_wb].rgb / 65535.0, 1e-6, None)
wb = n_raw[1] / n_raw                            # green-normalised gains
```

Everything downstream rides on that one measurement: the fitted matrix consumes
balanced data, `build_camera_dcp` writes `ColorMatrix1 = inv(M·diag(wb))`, and since
`spec/camera-profile-calibration-wb.md` the profile *bakes* `1/wb` as its calibration
neutral — so a single patch's sensor noise, a dust speck, or a stray specular
highlight now propagates into every frame the profile is applied to.

Two concrete failures on the maintainer's chart (LaserSoft **ISO 12641-2 advanced**,
Provia 100F, block `A49`–`L72`, greys = the 96-step wedge in rows G–J):

1. **The wrong patch wins.** `_pick_wb_id` takes the *lightest* near-neutral patch,
   which is `C49` (L\*87.25, chroma 1.40) — the near-white left end of a dye ramp,
   not a designed neutral, and the patch most likely to clip a channel. The 96-step
   grey wedge contributes nothing.
2. **One sample, no averaging.** Nothing rejects a bad patch or averages away noise.

## 2. Goals / non-goals

**Goals**
- Estimate the WB from a **range of neutral patches**, averaged.
- Default to the **midtone band** of the neutral ramp: below the highlight end
  (clipping, and the chart's Dmin where dye density flattens) and above the deep
  shadows (veiling flare and black-level error bend channel *ratios*, which is
  exactly the estimated quantity). On the maintainer's chart that band is rows
  **H and I** (`H49`–`I72`, L\* 24.2–63.7); on a classic IT8 GS strip it is roughly
  `GS16`–`GS22`.
- **Correct each patch against its reference Lab** rather than forcing equal-RGB —
  see §4.2. Mandatory, not an optimisation: the measured cast on H49–I72 is
  systematic (mean a\* −3.51, b\* −2.09), so averaging does not cancel it.
- Robust to a single bad patch (dust, sheen, a clipped channel).
- Identical results to today when only one neutral is available (classic fallback).

**Non-goals**
- No UI control. The band is a documented default plus two API parameters; the
  wizard passes neither. A chart whose neutrals sit outside the default band still
  works via the fallback in §4.4.
- No change to the *exposure* anchor. `_pick_wb_id`'s patch keeps defining the
  white-relative normalisation (`white_Y`, `n_raw[1]`) and the cLUT residual anchor
  (`build_residual_clut` reads `fit.wb_id`). Only the **channel ratios** change.
- No change to `CameraFit.matrix` semantics, the D50 pin, or ΔE scoring.

## 3. Selecting the patches

A patch is a **WB candidate** when all hold:

| gate | rule | why |
|---|---|---|
| sampled & valid | `ps.valid` | drops clipped / out-of-frame patches |
| has reference | `ref.lab(id) is not None` | needed for the §4.2 correction |
| near-neutral | Lab chroma < `_NEUTRAL_CHROMA` (6.0) | reuses today's definition |
| in the band | `WB_L_RANGE[0] <= L* <= WB_L_RANGE[1]` | §2, the midtone band |
| above the noise floor | every channel ≥ 1% of full scale | an underexposed shot puts even a
  midtone patch near the black level, where the ratio is noise |

`WB_L_RANGE = (24.0, 64.0)`. Chosen to bracket the midtone band of a neutral ramp;
on the maintainer's chart it selects exactly `H49`–`I72` (48 patches), on a classic
24-step GS strip roughly `GS16`–`GS22` (7 patches). Both correspond to ~5–45% of the
chart white's luminance.

Two overrides on `fit_camera_matrix`:

- `wb_ids=[...]` — an explicit patch list (still filtered by validity/reference).
- `wb_l_range=(lo, hi)` — a different band.

`wb_id=` keeps its current meaning: it forces the **anchor** patch, not the set.

Candidacy is independent of the matrix fit's own patch list (`ids=`): a patch may
inform the white balance without being one of the patches `M` is fit on, and vice
versa. Both are filtered by `ps.valid`, so neither can consume a clipped patch.

## 4. Processing

### 4.1 Why a naive average of ratios is wrong

The obvious implementation — average each patch's `n[1]/n` — assumes every selected
patch *is* neutral. The maintainer's wedge is not: chroma runs 3.0–4.8 across
H49–I72 with mean a\* = −3.51, b\* = −2.09. That is a systematic green-blue cast
(Provia dye drift with density), not scatter, so averaging **preserves** it. Forcing
those 48 patches to render equal-RGB would bake that cast into the profile's neutral
with the sign flipped.

### 4.2 Reference-corrected estimate

Correct each patch against the colour the chart *says* it is. With `M` the fitted
matrix (balanced device → XYZ D50, pinned so `M·(1,1,1) = D50`):

```
b_i = d_i · wb / n1            measured balanced device  (n1 = anchor patch green)
X_i = xyz_i / 100 / white_Y    reference XYZ, white-normalised as the fit does
p_i = M⁻¹ · X_i                the balanced device value that WOULD produce X_i
c_i = p_i / b_i                per-channel correction, then green-normalised
```

`c_i` is the factor by which patch *i* says the current `wb` is wrong. For an exactly
neutral reference this reduces to the classic condition: `M⁻¹·D50 = (1,1,1)` under
the pin, so `c_i ∝ 1/b_i` — "make this patch equal-RGB". For a chromatic reference it
targets the patch's true colour instead. Non-neutral candidates therefore become
*safe to include*, which is what makes averaging over a whole wedge possible.

### 4.3 Combination and iteration

Per channel, over the candidate set: take `log c_i`, drop entries more than
`2.5 · 1.4826 · MAD` from the median (one dusty or sheened patch cannot pull the
result), then average the survivors and exponentiate — a geometric mean, the natural
mean for ratios.

`M` was fit with the old `wb`, so the correction is applied as a short fixed point:

```
for _ in range(_WB_MAX_ITERS = 4):
    M   = fit(wb)                       # the existing least-squares + D50 pin
    c   = robust_geomean(corrections)   # §4.2
    if max|log c| < 1e-4: break
    wb *= c                             # c[1] == 1, so wb stays green-normalised
```

Guard: if any iteration's correction exceeds 2× in a channel the loop stops and keeps
the previous `wb` (a degenerate fit must not run away). The final `M` is refit with
the final `wb` so matrix and multipliers always agree.

**Why this converges fast.** Balancing is a diagonal transform and `M` is a full 3×3
least squares, so the unpinned device→XYZ solution `A' = diag(wb)·A/n1` is *invariant*
to the choice of `wb`: refitting cannot fight the update. Substituting the pinned
`M = diag(D50/white)·n1·A'ᵀ·diag(1/wb)` with `white = n1·A'ᵀ·(1/wb)` into §4.2, the
`diag(wb)` factors of `p_i` and `b_i` cancel outright:

```
c_i = [ (A'ᵀ)⁻¹ · diag(white/D50) · X_i ] / d_i
```

so `wb` survives only inside `white` — i.e. `wb` enters the render **only** by
deciding which device triple is declared white, which is exactly the quantity being
estimated. The first iteration therefore lands essentially on the answer and later
ones settle only that second-order coupling; 4 is a generous cap, not a working
budget. It also means the estimate does not depend on which anchor patch it started
from (test 8).

### 4.4 Fallback

Fewer than **3** candidates ⇒ keep today's single-patch behaviour exactly
(`wb = n_anchor[1]/n_anchor`, no iteration), with `wb_ids = [wb_id]`. This covers
charts whose neutrals fall outside the band, heavily clipped shots, and every
existing test. A `numpy.linalg.LinAlgError` on `M⁻¹` falls back the same way.

## 5. Data model

`CameraFit` gains one field:

```python
wb_ids: List[str]        # patches averaged for the WB (anchor-only on fallback)
```

Sorted canonically (row letter, then column number; `GS<n>` numerically) so the
summary line and the tests read in chart order rather than dict order.

`wb_id` (anchor), `wb_mult`, `matrix` and the ΔE fields keep their meaning.
`CameraFit` is transient — built by the wizard, consumed by the builders — so there
is no persistence or migration concern.

## 6. Integration points

- `src/core/it8_profile.py` — `WB_L_RANGE`, `_wb_candidates`, `_robust_log_mean`,
  `_refine_wb`; `fit_camera_matrix(..., wb_ids=None, wb_l_range=WB_L_RANGE)`;
  `CameraFit.wb_ids`.
- `src/widgets/it8_profile_dialog.py:825` — the summary line becomes
  "white-balanced on H49–I72 (48 patches)" (single id unchanged on fallback).
- `src/core/color_management.py`, `src/core/dcp_profile.py` — **unchanged**. They
  consume `fit.wb_mult`, so the improved neutral flows into the ICC `CCRn` tag and
  the DCP `AsShotNeutral` automatically (`spec/camera-profile-calibration-wb.md`).
- `build_residual_clut` — unchanged; still anchors on `fit.wb_id`.

## 7. Test plan

`tests/test_it8_profile.py`:

1. **Exact-neutral ramp** — synthetic camera, perfectly neutral greys: the averaged
   estimate recovers the true `wb` to float precision (parity with single-patch).
2. **Systematic cast (the reason for §4.2)** — greys carrying the real chart's
   mean cast (a\* −3.5, b\* −2.1): the reference-corrected estimate recovers the true
   `wb`, while a naive ratio average is measurably wrong. Asserts both directions.
3. **Noise** — per-patch noise on the ramp: the averaged estimate is closer to truth
   than the single lightest patch, over a seed sweep.
4. **Outlier rejection** — one patch corrupted (blown channel / dust): the estimate
   moves by less than a tight bound, and the patch is absent from `wb_ids`.
5. **Band selection** — only patches with L\* in `WB_L_RANGE` appear in `wb_ids`;
   `wb_l_range=` and `wb_ids=` overrides both honoured.
6. **Fallback** — a chart with a single neutral gives bit-identical results to the
   pre-change code path, and `wb_ids == [wb_id]`.
7. **Anchor unchanged** — `wb_id`, `white_Y` normalisation and the cLUT anchor are
   the same patch as before; `M·(1,1,1) == D50` still holds exactly.
8. **Convergence** — the iteration terminates, and the result is independent of which
   valid anchor patch it started from.
9. **Downstream identities** — `build_camera_dcp`: `ColorMatrix1·D50 == 1/wb_mult`;
   `build_camera_icc`: `CCRn == 1/wb_mult` (both containers).
10. **Invalid patches** — clipped/invalid candidates never enter `wb_ids`.

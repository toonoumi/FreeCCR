# Spec: Auto Gain

Status: REFINED (open questions resolved — ready to implement)
Owner: FreeCCR
Feature branch: `feature/auto-gain`
Related: `spec/auto-exposure-default-slope.md` (the legacy baked auto-exposure this
supersedes when on), [`spec/working-space-headroom.md`](working-space-headroom.md),
[`spec/working-space-white-balance.md`](working-space-white-balance.md),
`spec/settings-page.md` (the Settings dialog this adds a tab to).

## 0. Decisions (locked)

- **Retuned 2026-07-07 (maintainer): top-2% → 95%.** Originally the top-0.1%
  in-bound highlight was placed at 99.8% of the window (`AG_PERCENTILE = 99.9`,
  `AG_TARGET = 0.998`), which parked highlights against the window top. Now the
  **98th percentile** is placed at **display 0.95**, leaving a visible safety
  margin. Consequence: a frame whose highlights already sit at the window top is
  no longer a no-op — it gets a small pull *down* to the target, and the g = 0.6
  floor is unreachable for in-bound values (p ≤ 1 → g ≥ 0.95); the clamp stays as
  a safety net only.
- **Reference exclusion (req 4) = sampled clear/dense conversion points.** Only
  pixels whose de-windowed display value lies *between* the sampled clear point
  (→ display 0) and the sampled dense point (→ display 1) count toward the
  highlight reference. Over-range/specular pixels (display > 1, denser than the
  dense sample) and sub-black pixels (display < 0, clearer than the clear sample /
  film holder) are discarded. Because the inversion already maps clear→0 and
  dense→1, "between the sampled points" is **exactly the working-space window
  `[0, 1]`** — so the exclusion needs only the window geometry, not the raw
  sampled BGR values, and it is identical for default-slope (clear sample → 0,
  the SLOPE ceiling → 1) and two-point conversions.
- **Relationship to the legacy `exposure_base` = suppress-overlap.** When Auto
  Gain is ON, the baked default-slope auto-exposure (`exposure_base`) is skipped
  so the two never compound. When Auto Gain is OFF, today's baked auto-exposure
  is applied exactly as now (no behaviour change for the OFF state).
- **Default ON.** It is an "auto" convenience and the user phrasing ("this *can
  be* switched off") implies on-by-default.
- **Computed live, not baked.** The offset is recomputed in `apply_adjustments`
  from the base in hand. It depends only on the base pixels + window geometry
  (NOT on any slider), so it is constant across a slider drag (no wobble) and
  needs no per-image persisted field or cache-invalidation machinery.

## 1. Summary

Auto Gain is a toggleable (Settings → **General**) convenience that, **without
moving the Master Gain slider**, secretly offsets Channel Levels' Master Gain
stage so the top 2 % of
in-bound highlights lands at 95 % of the working-space window (display 0.95). It
rides the exact mechanism the legacy default-slope auto-exposure already uses —
an invisible additive offset on the `exposure` (Gain) parameter — but is general
(all converted images), live, user-toggleable, and respects the sampled
clear/dense range when measuring the highlight.

## 2. Goals / Non-goals

### Goals
- A persisted **Settings → General → "Auto gain"** toggle (default ON), staged-
  and-applied-on-Done like the other global toggles; toggling reprocesses loaded
  images.
- A per-image **invisible Gain offset** added to the Gain-stage value
  (`s['exposure'] + offset`) — the UI slider is never touched (mirrors the
  existing `exposure_base`).
- Reference: the **98th percentile** of in-bound luminance is placed at
  **display 0.95** ("95 % of the workspace window value").
- **In-bound** = de-windowed display value in `[0, 1]` (between the sampled
  clear→dense points); headroom/over-range and sub-black pixels are discarded.
- When ON, **suppress the legacy `exposure_base`** so they don't double-apply;
  when OFF, restore today's behaviour exactly.
- **CPU/GPU parity is automatic** — the offset is a scalar Gain value fed to the
  same shared stage on both the numpy and OpenCL paths.

### Non-goals
- Changing the Master Gain slider's value, range, or UI; the offset stays invisible.
- Per-channel / white-balance behaviour (Auto Gain is a uniform gain).
- Auto-gaining **area layers** (`_adjust_for_area` zeroes base offsets — areas
  composite on the already-gained whole-image base and must not re-gain).
- The reference-frame normalize conversion math, dust, crop, curves.
- Fixing the pre-existing units quirk in `compute_auto_exposure_gain` (it returns
  `50·log2(g)`, a stops mapping consumed by the `/300` curve — left as-is for the
  OFF path; the new function uses the correct inverse of the actual stage). Note
  that the legacy `eb` still rides the slider-less `exposure` stage, which keeps
  its `/300` curve; only Auto Gain moved to Master Gain.

## 3. Current behaviour (as-is)

`ccr_image.apply_adjustments` already adds a hidden offset to the gain stage:

```
adjust_image_opencl(image,
    ...,
    s.get('exposure', 0) + eb,      # eb = self.exposure_base (default-slope only)
    ...,
    ws_windowed=self._ws_windowed)
```

`eb` is computed **once** at conversion by `compute_auto_exposure_gain` (98th
percentile → 0.98, film-holder excluded via a luminance cut) and stored on the
image; it is non-zero **only** for default-slope (black-point-only) conversions.
The Master Gain stage is `g = 1 / (1 − v/CH_SLIDER_DIV)` (`CH_SLIDER_DIV = 150`)
for slider value `v ∈ [−100, 100]`
(`g ∈ [0.6, 3.0]`), applied **un-clamped before the window clamp** in
`_apply_working_space_recovery` when `ws_windowed` (so it can lift dark frames
*and* pull blown highlights down out of headroom), or on the clamped range in the
legacy path.

The early-return guard skips the whole adjustment pass when nothing is set:

```
if not s and cb == 0 and tb == 0 and bb == 0 and eb == 0 and not has_areas:
    ... de-window only ...
```

## 4. Design

### 4.1 The offset

`compute_auto_gain_offset(base_u16, ws_windowed) -> float` (new, in
`ccr_processor.py`):

```
d = base_u16.astype(float32)
if ws_windowed:                      # de-window; keep headroom (d may exceed 1)
    d = (d - WS_B) / (WS_W - WS_B)
else:                                # legacy full-range base
    d = d / 65535.0
lum = 0.299*d[...,0] + 0.587*d[...,1] + 0.114*d[...,2]    # RGB (index 0=R,2=B)
keep = (lum >= 0.0) & (lum <= AG_HI)            # in-bound: between clear(0)/dense(1)
vals = lum[keep]
if vals.size < MIN_CONTENT_FRACTION * lum.size:  # not enough in-bound content
    return 0.0
p = percentile(vals, AG_PERCENTILE)              # 98th
if p <= AG_EPS:
    return 0.0
g = AG_TARGET / p                                # gain that puts p at 0.95
g = clip(g, AG_GMIN, AG_GMAX)                    # [0.6, 3.0] — the stage's range
v = CH_SLIDER_DIV * (1.0 - 1.0/g)                # inverse of g = 1/(1 - v/CH_SLIDER_DIV)
return clip(v, -100.0, 100.0)
```

Constants: `AG_PERCENTILE = 98.0`, `AG_TARGET = 0.95`, `AG_HI = 1.0`,
`AG_GMIN = 0.6`, `AG_GMAX = 3.0`, `AG_EPS = 1e-4`; reuse `MIN_CONTENT_FRACTION`
(0.005). The bottom bound (`lum < 0`) is excluded for completeness but never
affects a top-2 % percentile.

Why this is the correct inverse: the offset is added to the user's Gain value and
the **sum** is fed to `g = 1/(1 − v/CH_SLIDER_DIV)`. With the slider at 0, the sum is `v`
and the realized gain is exactly the clamped `g`, so `p·g = 0.95`. With the slider
moved, the user is deliberately deviating (same as `exposure_base` today).

### 4.2 Wiring in `apply_adjustments`

```
auto_on = ccr_backend.auto_gain and self.converted    # film conversions only
ag = compute_auto_gain_offset(image, self._ws_windowed) if auto_on else 0.0
eb_eff = 0.0 if auto_on else eb          # suppress-overlap (decision 0)
...
if not s and cb == 0 and tb == 0 and bb == 0 and eb_eff == 0 and ag == 0 and not has_areas:
    ... de-window only ...               # unchanged fast path
...
adjust_image_opencl(image, ..., s.get('exposure', 0) + eb_eff + ag, ...)
```

`image` here is the (dust-removed) base actually being rendered — the windowed
preview base for the preview/thumbnail, the full-res base for export/zoom. The
98th percentile is resolution-robust, so preview and export agree within a
fraction of a slider unit (imperceptible). `_adjust_for_area` is **not** touched.

### 4.3 Settings → General

Add a **General** category to `SettingsDialog` (new `_build_general_page`) with a
single "Auto gain" checkbox + muted help text, seeded from `ccr_backend.auto_gain`
in `_init_toggles`, staged, and applied on Done in `_apply_pending` by calling
`main_window.on_auto_gain_toggled` only when it differs from the live flag.

## 5. Data model

- `ccr_backend.auto_gain: bool = True` (new flag). Persisted by MainWindow under
  QSettings key `adjust/auto_gain`; restored at startup before any render.
- **No** per-image persisted field, **no** catalog change — the offset is live.

## 6. Integration points

- `ccr_processor.compute_auto_gain_offset(base_u16, ws_windowed)` — new pure
  function; exports `WS_B`/`WS_W` already exist module-level.
- `ccr_image.apply_adjustments` — compute `ag`, suppress `eb`, fold into the Gain
  value and the early-return guard (§4.2). `_adjust_for_area` untouched. The
  `ccr_backend` reference uses the **deferred import already used in this file**
  (`from core.ccr_backend import ccr_backend` *inside* the method — see
  `ccr_image.py:351/363`) because `ccr_backend` imports `CCRImage` at module load
  (a top-level import here would be circular).
- `ccr_backend` — add `self.auto_gain = True` in the flag block (alongside
  `positive_mode` / `rgb_merge_mode` / `density_bwpoint`).
- `ui/main_window` — restore `auto_gain` from QSettings in `__init__` (key
  `adjust/auto_gain`, default True); add `on_auto_gain_toggled(checked)` that sets
  the flag, persists, and reprocesses loaded images. Reprocess is **render-only**
  (no re-conversion, no `save_catalog` — nothing per-image is persisted): release
  the hi-res cache (`image_preview._release_hires(refresh=False)`), update
  thumbnails, refresh the current preview, and show a temporary hint.
- `widgets/settings_dialog` — add the General page + checkbox + staging
  (`_init_toggles` seeds it, `_apply_pending` calls `on_auto_gain_toggled` on
  change). General is added as the **first** category so it reads as the primary
  page.

## 7. Edge cases

- **Too little in-bound content** (mostly headroom/holder, < 0.5 %): offset 0
  (no auto-gain) — a converted frame that is almost all over-range is left to the
  user. When Auto Gain is ON this means a default-slope frame gets neither `eb`
  nor `ag`; acceptable (degenerate frame).
- **Highlights already at the window top** (p98 ≈ 0.99): a small pull *down* to
  the 0.95 target (g ≈ 0.96) — deliberate since the 2026-07-07 retune; the target
  keeps a visible margin below display white.
- **Very dark frame** (p98 < ~0.32): `g` clamps at 3.0 → offset 100, lifted as
  far as the stage allows (cannot reach 0.95; flagged by the histogram, not here).
- **Blown frame in headroom** (p98 > 1, ws on): excluded by `AG_HI`, so the
  *in-bound* p98 (just under 1) drives a pull down to 0.95. With the working space
  on, the un-clamped gain still pulls the over-range pixels down too.
- **Non-windowed / `FREECCR_WORKING_SPACE=0`**: base is already clamped `[0,1]`;
  over-range pixels are a pile at exactly 1.0 → excluded by `lum <= AG_HI` only
  if strictly above; we keep `<= 1.0`, so the clamp pile counts. Best-effort —
  the feature targets the default working-space mode.
- **Area layers / curves / B&W**: unaffected (auto-gain is part of the
  whole-image base they build on).

## 8. Test plan (`tests/test_auto_gain.py`, pure numpy, headless)

- **Placement**: a windowed base whose in-bound p98 = 0.5 → offset `v` with
  `1/(1−v/CH_SLIDER_DIV) ≈ 1.9`; feeding `ch_master_gain=v` through
  `adjust_image(ws_windowed)` puts the 98th percentile of the output at ≈ 0.95·65535.
- **Routing**: `apply_adjustments` adds the offset to `ch_master_gain` and leaves
  the `exposure` argument at 0 — nothing rides the old, slider-less gain stage.
- **Over-range exclusion (req 4)**: a base with ~5 % pixels at d=1.5 and the rest
  at d=0.5 yields the *same* offset as the all-0.5 base (the 1.5 pixels are
  discarded) — the headroom pixels do not pull the gain down.
- **Neutral**: in-bound p98 already ≈ 0.95 → offset ≈ 0.
- **Insufficient content**: < 0.5 % in-bound pixels → offset 0.0.
- **Clamp**: a very dark base → offset 100 (g capped at 3.0). The g = 0.6 floor
  would need in-bound p98 > 0.95/0.6 ≈ 1.58 — impossible for in-bound values
  (≤ 1), so it stays a safety net only.
- **Suppress-overlap**: with `ccr_backend.auto_gain=True`, a default-slope image's
  render uses `ag` and **not** `eb` (assert eb path is bypassed); with it False,
  `eb` is applied and `ag` is 0.
- **CPU/GPU parity**: `adjust_image` vs `adjust_image_opencl` with the auto-gain
  offset as the Gain value agree within tolerance (offset is just a scalar Gain).
- **Backend/Settings**: `ccr_backend.auto_gain` defaults True; the General toggle
  seeds/round-trips; `on_auto_gain_toggled` flips + persists (GUI-light or mocked).

## 9. Open questions — RESOLVED

- **Cache vs live → LIVE.** The 98th percentile is resolution-robust, so the
  live recompute in `apply_adjustments` keeps preview and export within a fraction
  of a slider unit while avoiding any per-image field, conversion hook, or
  cache-invalidation. Revisit only if a measurable preview/export mismatch
  appears. (Cost is ~1–2 ms on the ≤1080 px preview; zero when the toggle is off
  since `compute_auto_gain_offset` is not called.)
- **Positive mode → NO (conversion-only).** Auto Gain gates on `self.converted`
  alone. Its reference (req 4) is the sampled clear→dense range, which exists only
  for film conversions; positive mode has no such anchors and is intentionally an
  identity baseline ("adjustments own the look", `brightness_base = 0`). Applying
  auto-gain there would break the fresh-positive-preview-is-identity contract
  (`test_positive_mode.py`). Un-converted raw negative scans are excluded too.

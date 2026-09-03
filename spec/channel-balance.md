# Channel Balance

Replace the Temperature and Tint sliders with three per-channel **Balance**
sliders (R, G, B), each a single anchored control point low on that channel's
tone curve, driven through the same monotone-cubic path as the Curves editor.
Hotkeys `U`/`I`/`O` raise R/G/B, `J`/`K`/`L` lower them.

## 0. Naming

**Channel Balance** is the tone-*weighted* per-channel control; **Channel
Levels** is the tone-*uniform* one. They are siblings, and the split is the
whole point of this feature:

| | control | acts on | corrects |
| --- | --- | --- | --- |
| Channel Levels | R/G/B **Shift** | uniform density offset | overall cast |
| Channel Levels | R/G/B **Gain** | per-channel slope | contrast mismatch |
| **Channel Balance** | R/G/B **Balance** | low-anchored curve node | crossover (cast that varies with tone) |

The sliders are deliberately **not** called "Density". In this codebase
"density" already has a strict meaning — `ch_*_shift` *is* the density offset,
alongside `compute_density_slopes`, `density_bwpoint` and `ci["density"]` — and
reusing it for a tone-weighted curve node would name two different operations
the same thing. Slider labels stay `R Balance` / `G Balance` / `B Balance` so
they never read as Channel Levels' `R Shift` / `R Gain` / `R Blackpoint`.

## 1. Motivation

### 1.1 Temperature/Tint is a contrast control on a density base

White balance is a flat per-channel **multiply** applied to the de-windowed
display value `d` (`_white_balance_gains` → `_apply_working_space_recovery`,
`ccr_processor.py:1340-1347`).

On a black-point-only or no-anchor conversion, `d` is not linear light — it is
optical **density**:

| conversion mode | base | `d` is |
| --- | --- | --- |
| `bw` with black point (`_default_slope_invert`) | `slope · log10(base/img)` | density |
| `bw` with no anchors (`_unanchored_density_invert`) | `−log10(16p) + 1.0` | density |
| `ref` / `ref_params` (v0.2.3 normalisation) | `65535 − v` | linear positive |

Multiplying a *density* by a constant is a per-channel **gamma/slope** change,
not a colour shift. So on every black-point and no-anchor conversion,
Temperature and Tint have been acting as per-channel contrast sliders. They
behave as real white balance only on reference-frame conversions. This is the
concrete reason they "don't work too well" on negative scans.

### 1.2 Exact density-space balance already exists

The physically exact colour-balance move in density space is a **constant
per-channel offset**, `d' = d + k` — uniform across every tone, equivalent to a
per-channel multiply in linear light (the colour-enlarger filtration model).
FreeCCR already has it: **Channel Levels R/G/B Shift**, which runs pre-clamp on
exactly this base (`_apply_channel_levels` step 2; see
`spec/channel-levels-pre-clamp.md` and `spec/no-anchor-convert.md`).

So a "density offset" slider would duplicate Shift, and would not fix what is
actually wrong.

### 1.3 What is missing: crossover

A cast whose *hue changes with tone* — cyan shadows under a warm highlight
balance, say — comes from the three dye layers having different toe shapes and
gammas. No per-channel offset (Shift) and no per-channel slope (Gain,
film-stock slopes) can correct it, because both are tone-uniform in log space.
Correcting crossover requires a tone-**dependent** per-channel move, and the
standard tool for that is a single node placed low on the channel curve and
dragged vertically.

That is the control this spec adds. It composes with, and does not replace,
Shift (offset) and Gain (slope).

## 2. Goals / non-goals

**Goals**

- Three sliders — R Balance, G Balance, B Balance — replacing Temperature and
  Tint in the same panel position, with colour-coded grooves.
- Each is one anchored node on that channel's curve at a fixed low x, moving
  vertically; endpoints pinned.
- Identical curve engine to the Curves editor, so the result is what dragging
  that node by hand would produce.
- Applied in **density space**, pre-clamp, in the slot Temperature/Tint vacated.
- Hotkeys `U`/`J` = R ±, `I`/`K` = G ±, `O`/`L` = B ±.
- WB Picker and AWB retargeted to solve for the three balance values.
- Existing saved Temperature/Tint values keep rendering byte-identically.

**Non-goals**

- No migration of saved `temperature`/`tint` values into balance values.
- No change to Channel Levels, film-stock slopes, or the Curves editor.
- No new OpenCL kernel code — the stage is a per-channel LUT applied in numpy
  on both paths, so GPU/CPU parity is free (the Channel Levels pattern).
- No user-configurable node position or strength in this iteration (`BALANCE_NODE_X` and `BALANCE_MAX_STOPS` are constants).

## 3. UX

### 3.1 Panel

Temperature and Tint are removed. In their place, directly under the Color
Profile row:

```
  R Balance   [====|====]    0
  G Balance   [====|====]    0
  B Balance   [====|====]    0
```

Range `-100..+100`, default `0`, `ResettableSlider` behaviour unchanged.

Gradient grooves via the existing `GradientSlider`, using each channel's
complementary pair so the direction reads at a glance:

| slider | low (−100) | high (+100) |
| --- | --- | --- |
| R Balance | cyan `#4fb3b3` | red `#d06666` |
| G Balance | magenta `#c264c2` | green `#66aa66` |
| B Balance | yellow `#c9b452` | blue `#6688d0` |

High ends reuse `theme.CH_R` / `CH_G` / `CH_B` so the sliders match the Channel
Levels group labels. New constants `BALANCE_R_GRADIENT`, `BALANCE_G_GRADIENT`,
`BALANCE_B_GRADIENT` in `theme.py` replace `TEMP_GRADIENT` / `TINT_GRADIENT`
(both of which become unused and are deleted).

**Direction**: positive raises that channel in the deep shadows — more
of that colour, matching the gradient's high end.

### 3.2 Hotkeys

`QShortcut` with `WindowShortcut` context, registered in `main_window.py`
alongside the existing `C`/`D`/`W` mode keys:

| key | action | key | action |
| --- | --- | --- | --- |
| `U` | R Balance +step | `J` | R Balance −step |
| `I` | G Balance +step | `K` | G Balance −step |
| `O` | B Balance +step | `L` | B Balance −step |

`BALANCE_HOTKEY_STEP = 5` — 20 presses of full travel, matching the "nudge"
feel of the rotate keys. None of `U I O J K L` collide with the existing
single-letter shortcuts (`C` crop, `D` dust, `W` WB picker, `[` `]` rotate).
Modal dialogs are separate windows, so `WindowShortcut` does not reach text
entry in them — the same reason the existing letter keys are safe.

Each press:
- is a no-op without a current image, or when the sliders are disabled
  (unconverted image), mirroring the existing shortcut handlers;
- goes through the normal slider `setValue` path, so it is debounced,
  undoable, and repaints like a drag;
- **merges into one undo burst** while presses continue (`_undo_burst_timer`),
  so holding a key is one undo step, not twenty.

Repeats are handled by Qt's auto-repeat on the shortcut.

### 3.3 WB Picker / AWB

Both keep their buttons and behaviour; they now solve for the three balance
slider values instead of temperature/tint (§4.4). The hint text becomes
`Auto WB applied — R: {r}, G: {g}, B: {b}.`

### 3.4 Sync / Copy Settings

The `wb` sync group's keys change from `("temperature", "tint")` to
`("balance_r", "balance_g", "balance_b")`; its label becomes
`"Colour Balance (R/G/B Balance)"`. The group id stays `"wb"` so a remembered
`{gid: bool}` selection from a previous session still applies.

## 4. Processing / math

### 4.1 The curve

Per channel, three control points in the normalised display domain:

```
(0, 0)                          # black, pinned
(X0, X0 + off)                  # the anchored node
(1, 1)                          # white, pinned
```

with

```
X0    = 0.1875                              # BALANCE_NODE_X  (3/16)
stops = (slider / 100) * 1.5                # BALANCE_MAX_STOPS
y     = X0 ** (1 / 2**stops)                # the node moves in GAMMA
```

`X0 = 3/16` sits between 1/8 and 1/4 — low in the toe, where crossover actually
lives, but not so low that it wrenches the darkest tones. It is materially lower
than a midpoint control: the correction concentrates in the shadows and low
midtones and has faded to almost nothing by the highlights, which is exactly why
it fixes casts that Temperature/Tint (and any tone-uniform offset) cannot.

**Why not 1/8.** At `X0 = 1/8` the curve bit too hard in the deep shadows — a
full `+100` moved `x = 0.05` by `+0.167`. At 3/16 that drops to `+0.111`, a third
less, while the PEAK deviation is essentially unchanged (`+0.40` vs `+0.41`): the
control keeps its reach and loses only the deep-shadow harshness. Moving the node
up also buys downward range, since it can fall further before hitting 0.

**The node moves in gamma, not by a linear offset.** A linear offset has to stay
below `X0` or the node crosses the pinned `(0,0)` endpoint and breaks
monotonicity — which capped downward travel at `X0` and, for symmetry, capped
upward travel with it. That cap is why a heavily yellow frame could not be fully
corrected at the slider ends. A gamma move approaches 0 and 1 asymptotically, so
**no endpoint-crossing invariant exists to violate at any slider value**, and
the upward side is freed from the downward side's geometric limit.

At `±100` the peak deviation from identity is about **+0.40 / −0.26**, roughly
**3× / 1.8×** the old linear scheme. The correction that used to need the whole
slider now sits near 30, leaving real headroom above it.

The asymmetry is inherent, not a defect: a node at `x = 3/16` moving vertically
can rise most of the way to 1 but can never fall further than 3/16, so the
downward side saturates near −0.26 peak deviation whatever the scaling.
`BALANCE_MAX_STOPS = 1.5` already sits within a hair of that limit, so raising
the constant further buys upward range only. A cast needing more downward travel
wants Channel Levels or the Curves editor as well.

The Fritsch–Carlson tangent limiter keeps the result monotone at both extremes
regardless, so the curve can never invert local contrast.

Interpolation is `_monotone_cubic` — the exact function `build_channel_lut`
uses — so the stage is what dragging that node in the Curves editor gives.

### 4.2 Where it runs

**In density space, pre-clamp**, inside `_apply_working_space_recovery`,
immediately after Channel Levels and before White Point / Gain — the slot
Temperature/Tint occupied:

```
de-window
  → Channel Levels        (un-clamped)
  → R/G/B Balance         (un-clamped, NEW — was White Balance)
  → White Balance         (legacy temperature/tint, still here for old edits)
  → White Point
  → Gain/Exposure
  → window clamp
```

On a `bw` or no-anchor conversion this puts the node on a genuine log signal,
which is what makes the node act on a genuine log signal rather than a display-value tweak.
On a `ref` conversion the base is a linear positive and the node is an ordinary
per-channel curve — still the right correction, just not literally "density".

Placing it before the window clamp also means a balance move can push content
into the highlight headroom and pull it back, exactly like Channel Levels.

### 4.3 Un-clamped domain

`d` at this point may sit below 0 (shadow margin — sub-base film density) or
above 1 (highlight headroom). The curve is only defined on `[0,1]`, and both
endpoints are pinned to identity, so the stage extends as **identity outside
`[0,1]`**:

```
out = where((0 <= d) & (d <= 1), curve(d), d)
```

This is continuous at both ends (the curve passes through `(0,0)` and `(1,1)`)
and it is what preserves the sub-black data Channel Levels depends on —
`np.interp` alone would flatten the entire shadow margin onto 0 and destroy the
film-base separation that `spec/channel-levels-pre-clamp.md` §3.4 exists to
protect.

Implementation: a 1024-entry float32 LUT over `[0,1]` built once per render
from `_monotone_cubic`, sampled with `np.interp`, masked as above. 1024 entries
over a curve this smooth is well below the 10-bit window's own precision.

The non-windowed path uses the **same float LUT and the same function**, on
`img/65535` — not a separate 16-bit LUT. One implementation for both paths is
what makes CPU/GPU parity exact rather than merely within tolerance.

### 4.3.1 Ordering against legacy White Balance

Channel Balance runs *before* the legacy WB multiply. The order is unobservable in
practice: an edit made before this change has no `balance_*` keys (identity),
and an edit made after it has no `temperature`/`tint` (identity), so the two
stages are never both active on the same image unless a user deliberately hand-
edits a catalog.

### 4.4 Neutralising inverse (WB Picker / AWB)

The neutral solve is a **closed loop on the real render**, not an inverse of the
Balance curve. `CCRImage.solve_neutral_balance(sample_patch)`:

1. renders the sampled **area** through the actual adjustment pipeline,
2. measures the mean RGB that came out,
3. corrects, and repeats.

**Why not an analytic inverse.** Balance is one stage among many between the
base and the pixel the user sees. Inverting its curve requires *modelling* all
the others — Channel Levels, the hidden Auto Gain offset, gamma, the tone
sliders, saturation, per-channel Curves, the Cineon transform — and every such
model was wrong in some configuration. Three successive analytic versions failed
in the field. Measurement has no model to get wrong. Do not reintroduce an
open-loop inverse.

**The loop.** Red is the anchor and is never moved. Each pass solves **blue,
then green**, onto the mean of the other two *measured* channels — R=134, G=120
drives B to 127. Iterating that is what converges: with red pinned, "each
channel toward the mean of the other two" halves the remaining gap per pass and
settles on red.

```
pass 0:  R 74.0  G 62.6  B 34.0      mean(R,G) = 68.3
pass 1:  R 74.0  G 71.0  B 68.3   <- blue lands exactly on the mean
pass 2:  R 74.0  G 73.0  B 72.2
pass 3:  R 74.0  G 74.0  B 73.2
pass 4:  R 74.0  G 74.0  B 74.2      grey
```

Each channel is solved by **integer bisection on the measured output**, which is
monotone in that channel's slider. The loop keeps the best (lowest-spread)
result it has seen, so more passes can never regress. Measured spot spread,
full-frame render, on a frame that starts 54–86% off:

| preceding state | after |
| --- | --- |
| nothing set | 0.3% |
| Channel Levels set | 0.5% |
| Master Gain 35 | 0.6% |
| Input Gain −20 | 1.2% |
| Cineon log | 0.4% |
| gamma + contrast + saturation | 0.5% |
| **per-channel Curves** | 0.4% |
| **all of the above** | 0.2% |

Cost is 55–310 ms: the loop renders only the 7×7 sample area (~0.9 ms), never
the full frame.

**Graceful degradation.** Targeting the mean of the other two rather than red
directly matters when a channel cannot reach: it lands on the achievable middle
and the other channel follows it down, instead of one slider pegging and leaving
the rest stranded.

**Two hooks make the patch render match the real one** (`apply_adjustments`):

* `auto_gain_override` — Auto Gain is measured from whatever buffer it is given,
  and a patch carries no highlights, so it would compute a wildly different
  gain. The caller measures once from the **full base** and passes it in.
* `skip_dust` — dust healing is spatial and meaningless on a detached patch.
  Area layers are suppressed with the existing `areas_override=[]`.

**Flat-response guard.** Under a Black & White profile the output is luminance,
so no Balance value changes the spread. The solver detects the unresponsive
channel and leaves it at 0 instead of driving it to an endpoint.

**AWB is a regression over the WHOLE frame**, not a solve at one estimated
pixel. `compute_awb_balance` downscales the (cropped) frame to a 256px long
side, and the loop renders *that* every iteration and drives the illuminant
estimator's reading of the **rendered** result to grey. A downscale rather than
a scattered pixel sample because `gray_edge` needs spatial neighbours. Cost is
~350–450 ms.

Solving at a single estimated pixel was fragile: the estimate is measured on the
base, but the tone it lands on after Channel Levels and Auto Gain may be up
where a low-anchored node has almost no authority, so the solve produced tiny or
zero values and the button looked dead.

Two scale/direction bugs in the shared loop also made AWB silently do nothing,
and both are worth remembering when writing a new `combine`:

* **`combine` must report in the render's 0–65535 scale.** `estimate_neutral_rgb`
  returns normalised `[0,1]`; feeding that in unscaled made the flat-response
  guard (`|hi−lo| < 8` counts) fire for every channel, so every channel was
  declared unresponsive and left at 0. The guard and the convergence tolerance
  are now both **relative**, so a scale mistake degrades instead of silently
  zeroing.
* **The measured response need not be rising.** The bisection detects direction
  from its two endpoint probes. `gray_edge` reports gradient ENERGY, which
  *falls* as a channel is lifted (the curve compresses above its node);
  assuming a rising response drove it backwards, and the keep-best net then
  discarded the result as "no change".

`gray_edge` needs one further adaptation: gradient energy is not a per-channel
*value* statistic, so driving the three energies to equality moves colour the
wrong way regardless of direction handling. Under the closed loop it is used for
what it actually asserts — *these* pixels carry the illuminant —
`gray_edge_pixel_mask` selects the strongest-edge pixels once on the base (fixed,
so the objective does not move between iterations) and the loop neutralises
their rendered mean.

Measured on a frame with a 27.6% cast: gray_world 0.4%, white_patch 1.6%,
shades_of_gray 0.4%, gray_edge 0.4%. Through later stages (Channel Levels,
Master Gain, Cineon) all stay under 1%.

Note the correction is exact only **at the sampled tone** — inherent to a
tone-dependent control. A picked midtone and a picked shadow legitimately give
different values.

### 4.5 Dispatch

Mirroring Channel Levels:

- **Windowed base** — applied inside `_apply_working_space_recovery`;
  `adjust_image` / `adjust_image_opencl` then **zero** `balance_r/g/b` so
  neither the CPU body nor the kernel can re-apply them.
- **Non-windowed base** (reference mode, positive mode, area layers) — applied
  as a 16-bit per-channel LUT on `img16` at the top of `adjust_image` /
  `adjust_image_opencl`, before any other stage. Same numpy code on both paths,
  so GPU/CPU parity is automatic and **the OpenCL kernel is not touched**.
- All-zero sliders skip the stage entirely, so a neutral render stays
  bit-identical.

### 4.6 Legacy Temperature/Tint

`_white_balance_gains`, the `kelvin_shift`/`tint_shift` parameters,
`compute_neutral_temp_tint`, and `temperature_base` all stay exactly as they
are. `ccr_image.apply_adjustments` keeps passing `s.get('temperature', 0) + tb`
and `s.get('tint', 0)`. Catalogs written before this change therefore render
byte-identically; the values simply have no slider — the same treatment
`exposure` got when the Gain slider was removed.

## 5. Data model

Three new adjustment keys: `balance_r`, `balance_g`, `balance_b`, ints in
`[-100, 100]`, default 0.

In `SlidersPanel.ADJUSTMENT_KEYS` they take the two positions
`"temperature", "tint"` occupied plus one more, at the head of the list:

```python
ADJUSTMENT_KEYS = [
    "balance_r", "balance_g", "balance_b", "brightness", "gamma", ...
]
```

The list is zipped positionally against `create_slider()` call order, so the
three `create_slider` calls replace the two Temperature/Tint calls in place and
everything after shifts by one — no other reordering.

`adjustment_settings` is a plain dict persisted as JSON, so catalog
save/restore needs no change. Reading an old catalog yields no `balance_*`
keys, which `s.get('balance_r', 0)` resolves to the 0 default.

Area layers carry the keys like any other slider key (`_adjust_for_area` passes
them through), so a local balance correction works per area.

### 5.1 Signature compatibility (mandatory)

`adjust_image`, `adjust_image_opencl` and `_apply_working_space_recovery` are
all called **positionally** in existing code and tests — e.g.
`adjust_image(img, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, ...)`
(`test_channel_levels.py:78`), `adjust_image(test_img, kelvin, tint, exposure,
brightness, ...)` (`test_opencl_accuracy.py:98`),
`_apply_working_space_recovery(base, 0.0, 0.0, 0.0, 0.0)`
(`test_white_balance_ws.py:93`).

The three new parameters are therefore **appended at the very end** of all
three signatures, after `ws_windowed`:

```python
def adjust_image(img16, kelvin_shift=0.0, ..., ws_windowed=False,
                 balance_r=0.0, balance_g=0.0, balance_b=0.0)
```

Inserting them next to `kelvin_shift`/`tint_shift` — the position they occupy
conceptually — would silently shift every positional argument in those callers.
The stage's *pipeline* position (§4.2) is independent of its argument position.

### 5.2 Auto-AWB-at-conversion

`ccr_backend.maybe_auto_awb` writes `ci["temperature"], ci["tint"]` and guards
on those keys (`ccr_backend.py:925-930`). Both become the three balance keys:
the guard skips when any of `balance_r/g/b` is already non-zero, so a saved
value is still never clobbered.

## 6. Integration points

| file | change |
| --- | --- |
| `src/ui/theme.py` | drop `TEMP_GRADIENT`/`TINT_GRADIENT`, add the three balance gradients |
| `src/core/ccr_processor.py` | `BALANCE_NODE_X`, `BALANCE_MAX_STOPS`, `balance_curve_points()`, `_balance_lut()`, `_apply_channel_balance()`, `compute_neutral_balance()`; wire into `_apply_working_space_recovery`, `adjust_image`, `adjust_image_opencl` |
| `src/core/ccr_image.py` | pass `balance_r/g/b` in both `apply_adjustments` and `_adjust_for_area` |
| `src/core/awb.py` | `compute_awb_balance()` replaces `compute_awb_temp_tint()`; `estimate_neutral_rgb` unchanged |
| `src/core/ccr_backend.py` | `maybe_auto_awb` writes/guards the balance keys (§5.2) |
| `src/widgets/sliders_panel.py` | slider creation, `ADJUSTMENT_KEYS`, `SYNC_GROUPS["wb"]`, `on_wb_sampled` → three values, `nudge_balance()` for the hotkeys |
| `src/widgets/image_preview.py` | `_sample_wb_point` → `compute_neutral_balance` |
| `src/ui/main_window.py` | six `QShortcut`s + handlers |
| `CLAUDE.md` | a Key Patterns entry for the stage and its pipeline position |

No change to: the OpenCL kernel, `catalog.py`, `dcp_profile.py`,
`color_management.py`, export routing, the Curves editor.

## 7. Test plan

`tests/test_channel_balance.py`:

1. **Neutral is identity** — all three at 0 leaves a windowed and a
   non-windowed base bit-identical.
2. **Endpoints pinned** — pure black and pure white are unchanged at any slider
   value, on every channel.
3. **Monotonicity** — for `s` in `-100..100` step 5, the 1024-entry LUT is
   non-decreasing (the limiter never inverts contrast).
4. **Direction** — `balance_r = +50` raises R at `d = X0` and leaves G and B
   untouched; `-50` lowers it.
5. **Tone weighting** — the deviation from identity is 0 at both endpoints,
   peaks in the lower half (measured: max near `d ≈ 0.30` for a node at
   `X0 = 3/16`), and has decayed to under a fifth of its peak by `d = 0.9`.
   This is the property that distinguishes Balance from Channel Levels Shift,
   which is uniform across every tone. Note the peak sits ABOVE `X0`: a
   3-point monotone cubic spreads a low node's influence through the low
   midtones rather than confining it to the node, which is exactly what
   dragging that node in the Curves editor does.
6. **Un-clamped pass-through** — values below 0 and above 1 survive the stage
   unchanged, so the shadow margin and highlight headroom are intact.
7. **GPU/CPU parity** — `adjust_image` vs `adjust_image_opencl` agree within
   the existing tolerance on both windowed and non-windowed bases (skipped
   without OpenCL, like `test_opencl_accuracy.py`).
8. **Double-apply guard** — a windowed render with non-zero balance equals the
   pre-stage result, proving the params were zeroed before the kernel.
9. **Inverse round-trip** — for a spread of sampled triples,
   `compute_neutral_balance` then applying the curve brings the three channels
   within one slider step of each other at the sampled tone; out-of-reach
   targets clamp to ±100 without raising.
10. **Legacy render unchanged** — a settings dict carrying only
    `temperature`/`tint` renders byte-identically to `main`.
11. **Catalog round-trip** — balance values survive save/restore; a catalog
    without the keys loads as 0.

Panel/UI, in `tests/test_channel_balance_ui.py` (offscreen, following
`test_copy_settings_dialog.py`):

12. **Positional zip** — `ADJUSTMENT_KEYS` and the created sliders line up, and
    `balance_r/g/b` map to the first three sliders.
13. **Hotkeys** — each of `U I O J K L` moves its slider by `±BALANCE_HOTKEY_STEP`,
    clamps at the ends, and is a no-op with no image.
14. **Undo burst** — a run of presses collapses to one undo step.
15. **Sync group** — the `wb` group carries the three balance keys and syncs
    them.

16. **Auto-AWB at conversion** — `maybe_auto_awb` writes the balance keys and
    skips when any is already non-zero.

Updates to existing tests: `test_awb.py` (the `compute_awb_temp_tint` and
`on_wb_sampled(-30, 12)` call sites), `test_copy_settings_dialog.py` (the `wb`
group's keys), `test_area_editing.py`, and any test asserting slider indices or
the `temperature`/`tint` keys.

`test_white_balance_ws.py` and `test_working_space.py` must keep passing
**unchanged** — that is the regression proof for §4.6 (legacy WB untouched) and
for §5.1 (no positional signature drift).

Baseline note: the full pytest run hangs for order-dependent reasons unrelated
to this work, so tests are run in small file groups.

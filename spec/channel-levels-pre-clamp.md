# Spec: Channel Levels as the first pipeline stage (pre-clamp)

Status: REFINED v1 — ready to implement
Owner: FreeCCR
Feature branch: `feat/channel-levels-pre-clamp`

## 1. Summary

Channel Levels currently runs *late* in the look domain, on data the working-space
window clamp has already flattened to `[0, 1]`. That placement has three
consequences the user hit in practice:

1. **A per-channel shift tints the film base.** After `_default_slope_invert`
   floors density at zero (`ccr_processor.py:1224`), the film base and the true
   image shadows both sit at exactly `0` and are numerically indistinguishable.
   A `+blue` shift lands on both, painting the base a saturated `(0, 0, s)`.
2. **The panel cannot express "raise blue without tinting black".** With the
   current chain `out = ((in*ig + s) - b) / (w - b)`, requiring `out = 0` at
   `in = 0` forces `b = s`, which collapses the whole stage to `out = in/(w - s)`
   — a pure multiply pivoted on zero, which by definition cannot lift the shadow
   end. The limitation is structural, not a tuning problem.
3. **A shift only adds to what is already displayed.** Data below display-black
   is either gone (default-slope mode) or unreachable (clamped away before the
   slider runs), so a shift can never *reveal* it.

This feature moves Channel Levels to the **front** of the pipeline, inside the
working-space recovery domain where values are still un-clamped and may sit below
`0` or above `1`. A shift then translates the whole histogram, pulling sub-black
content into view from below and pushing in-window content out into the shadow
margin instead of clipping it. It also splits Input Gain / per-channel / Master
into three ordered sub-stages, doubles the slider strength, stops the
default-slope inversion from destroying sub-base data, and moves the section's UI
to sit directly under the conversion buttons.

## 2. Goals / Non-goals

### Goals
- Channel Levels is the **first** adjustment stage — ahead of White Balance,
  White Point, Gain/Exposure and the entire look domain.
- On a windowed base it runs **un-clamped**: a shift moves the whole histogram,
  bringing content from below display-black into the window and pushing content
  out the bottom into the shadow margin (recoverable by shifting back).
- Sub-black data actually **exists** to be recovered: the default-slope inversion
  stops flooring density at zero.
- Ordering within the stage: **Input Gain -> per-channel Shift/Gain/Blackpoint ->
  Master Shift/Gain**. Master is a separate uniform stage *after* the per-channel
  work, not summed into it as today.
- Sliders have roughly **2x the effect** they have now.
- The section sits **directly below the Convert row and above the WB/AWB row**,
  default-collapsed, all sliders defaulting to 0.
- CPU (numpy) and GPU (OpenCL) render identically; preview, hi-res zoom and
  export agree.

### Non-goals
- No change to the look-domain operators (brightness, contrast, highlights/
  shadows, saturation, curves, bands) or their order.
- No new sliders and no renames. The twelve existing keys keep their names, so
  saved catalogs and sync groups load unchanged (see §5 on the value remap).
- No change to the Curves or Subtractive Saturations sections.
- Not a color-management change — this is stage ordering, range and strength.

## 3. Processing / math

### 3.1 The stage

One helper, `_apply_channel_levels(d, ..., clamp: bool)`, operating on float
**display values** `d` (0 = display black, 1 = display white; may fall outside
`[0, 1]`). Index 0 = R, 1 = G, 2 = B.

```
ig    = 2 ** (clip(ch_input_gain, -100, 100) / 25)           # +-4 stops
s[c]  = clip(ch_<c>_shift,      -100, 100) / 150             # +-0.667
g[c]  = clip(ch_<c>_gain,       -100, 100) / 150
bp[c] = clip(ch_<c>_blackpoint, -100, 100) / 150
ms    = clip(ch_master_shift,   -100, 100) / 150
mg    = clip(ch_master_gain,    -100, 100) / 150

1. Input Gain (uniform):   d *= ig
2. Per channel c:          den  = max((1 - g[c]) - bp[c], CH_MIN_RANGE)
                           d[c] = (d[c] + s[c] - bp[c]) / den
3. Master (uniform):       d = (d + ms) / max(1 - mg, CH_MIN_RANGE)

4. if clamp:               d = clip(d, 0, 1)
```

`CH_MIN_RANGE = 0.1` (<=10x channel gain). **This guard is new and required**: at
the doubled `/150` scale, `gain = +100` with `blackpoint = +100` gives
`den = (1 - 0.667) - 0.667 = -0.333`, and a negative denominator inverts the
channel. At the old `/300` scale `den` could not go below `0.333`, so no guard
was needed.

Neutral (every slider 0) is exactly identity: `ig = 1`, all offsets 0, `den = 1`.
The per-channel early-skip that avoids +-1 LSB drift on untouched channels is
kept.

### 3.2 Where it runs

**Windowed base** (`_apply_working_space_recovery`) — the new first step, with
`clamp=False`:

```
windowed uint16 -> de-window -> d          (un-clamped; d may be < 0 or > 1)
  1. CHANNEL LEVELS      (clamp=False)     <- new position
  2. White Balance gains
  3. White Point (2^stops headroom recovery)
  4. Gain / Exposure  (incl. auto-gain / exposure_base)
  5. clip(d, 0, 1)                         <- window clamp
  -> full-range uint16 into the look domain
```

Because steps 1-4 are all un-clamped, a positive shift in step 1 lifts sub-black
content up through zero and into the visible window — requirement (3) of §1.

**Non-windowed base** (reference-mode conversions, positive mode, area layers):
the stage runs first in `adjust_image` / at the top of the OpenCL kernel, on
`d = img/65535`, with `clamp=True`. There is no sub-black data on these paths
(the conversion clipped at `[0, 1]`), so clamping matches today's behaviour and
protects the later `pow()` stages from negatives.

### 3.3 GPU/CPU parity

The windowed pre-stage is numpy and shared by both paths, exactly as White
Balance and the recovery controls already are. When `ws_windowed`, the twelve
channel params are **zeroed** after the pre-stage (like `exposure`, `whitepoint`,
`kelvin_shift`, `tint_shift` today) so the kernel never re-applies them. Parity is
therefore free on the windowed path; the kernel's own block is restructured only
for the non-windowed case.

### 3.4 Sub-black data must exist: the default-slope floor

`_default_slope_invert` clamps optical density at zero before encoding
(`ccr_processor.py:1224`), so everything at or clearer than the sampled black
point collapses to exactly display-black. Under the working space that floor is
**removed** — negative density is kept and `encode_window` carries it into the
shadow margin.

- Rendered output is unchanged for existing edits: the window clamp at the end of
  the recovery still clips `[0, 1]`, so a neutral image looks identical.
- The legacy full-range branch (`FREECCR_WORKING_SPACE=0`) keeps the floor and
  stays byte-identical.
- `DEFAULT_DENSITY_GAMMA` is 1.0 today so the display-gamma branch is inert, but
  it must not raise a negative to a fractional power (-> NaN). Apply it with
  `where=(out > 0)`, leaving the sub-black region linear. Continuous at 0.

`_twopoint_invert`'s working-space branch already keeps sub-black overshoot; no
change there.

### 3.5 Shadow margin: `WS_LO` 0.5 -> 1.0

The retained sub-black range is `WS_LO` display units. At `DEFAULT_DENSITY_SLOPE
= 0.8`, film base that is `k` stops brighter than the sampled black point lands at
`d = -0.8 * log10(2^k) ~= -0.241*k`, so `WS_LO = 0.5` holds only about **2 stops**
of sub-base data — and the doubled shift range alone is `+-0.667`, wider than the
margin itself. Raise the default to `1.0` (~4 stops).

The cost is negligible because the window *width* is unchanged (1024 codes):

| | `WS_LO = 0.5` | `WS_LO = 1.0` |
|---|---|---|
| `B` (display black) | 512 | 1024 |
| `W` (display white) | 1536 | 2048 |
| window width | 1024 | 1024 |
| highlight headroom | 5.989 stops | 5.977 stops |

Display precision is untouched and highlight headroom loses 0.012 stops. Still
overridable via `FREECCR_WS_LO`.

**Verified safe to change**: no code hardcodes 512/1536 — every consumer derives
from `WS_B`/`WS_W` (`compute_auto_exposure_gain`, `compute_auto_gain_offset`,
`encode_window`, the dust de-window helper at `ccr_processor.py:3740`). Windowed
bases live only in memory for the session and are rebuilt on reconversion, so
there is no persisted buffer to migrate.

Dust removal reads the windowed base *before* `apply_adjustments` and de-windows
with `np.clip(..., 0, 65535)` (`ccr_processor.py:3740-3744`), so the newly
retained sub-black pixels flatten to 0 there exactly as they do today. Dust
behaviour is unaffected by both the floor removal and the `WS_LO` change.

### 3.6 Interaction with Auto Gain

`compute_auto_gain_offset` measures the **base** and rides the Gain stage, which
now sits *after* Channel Levels. Two consequences, both accepted:

- Auto Gain's factor multiplies the channel-levels result, so a shift is scaled by
  it. It is a normalization that usually lands near 1.0 (clamped to `[0.6, 3.0]`),
  and it can be turned off in Settings -> General.
- The measurement stays on the raw base, *not* on the channel-adjusted data. This
  is deliberate: `compute_auto_gain_offset` is documented to depend only on the
  base pixels and window geometry so it stays constant across a slider drag.
  Re-measuring after Channel Levels would make it move while the user drags a
  channel slider — the exact instability that property exists to prevent.

Removing the default-slope floor also means genuinely sub-black pixels are now
excluded by Auto Gain's `keep = (lum >= 0) & (lum <= 1)` in-bound test, where
before they sat at exactly 0 and were included. This nudges the 98th percentile up
slightly and the gain down slightly. Correct behaviour (those pixels are film
base, not content), small in magnitude, covered by a test.

## 4. UX / interaction

- The **Channel Levels** `CollapsibleSection` moves from the bottom group (below
  Curves and Subtractive Saturations) to directly **below the Convert Current /
  Convert All row** and **above the WB Picker / AWB / Crop / Slice row**, with a
  section separator on each side.
- It stays **default-collapsed** (`CollapsibleSection` already starts collapsed)
  and every slider still defaults to 0 (`SLIDER_DEFAULTS` carries no channel key).
  Both are pinned by tests rather than changed.
- Slider ranges, labels, tick marks and double-click-to-reset are unchanged; only
  the *strength* per unit changes.
- The Cineon Log -> Rec.709 checkbox lives inside this section and moves with it.
  It remains a final-stage flag; only its screen position changes.
- Section population order in `create_slider()` is **unchanged**. Placement and
  population are already decoupled (`sliders_panel.py:640-646`), so moving the
  section widget cannot disturb the positional `ADJUSTMENT_KEYS` zip.

Concretely: only the two lines that construct `self.od_section` and add it to
`scroll_layout` move up to just after `convert_row` (`sliders_panel.py:534`). The
population block (`:659-697`) stays exactly where it is — it runs after the main
sliders (`:584-602`) and before the band sliders (`:749`), which is the order
`ADJUSTMENT_KEYS` requires, and `self.od_section` already exists by then.

## 4.1 Master Gain is the one gain control (v2)

The general-adjustments **Gain** slider and Channel Levels' **Master Gain** were
the same math at different scales — `g = 1/(1 - v/300)` over `±200` versus
`g = 1/(1 - v/150)` over `±100`. Both span exactly `g ∈ [0.6, 3.0]`. Keeping two
sliders for one control, on two different stages, is what made "which gain does
Auto Gain move?" ambiguous. So:

- **The Gain slider is removed.** `"exposure"` leaves `ADJUSTMENT_KEYS` and the
  `tone` sync group (whose label drops "gain").
- **Master Gain moves outside the collapsible**, immediately below it, so it is
  always visible. Its `create_slider()` call stays third among the channel
  sliders — only the layout it is inserted into changes — so the positional
  `ADJUSTMENT_KEYS` zip is untouched. The panel reserves the slot with
  `scroll_layout.count()` right after adding the section and fills it later with
  `insertLayout`.
- **Auto Gain rides Master Gain.** `compute_auto_gain_offset` returns
  `v = CH_SLIDER_DIV * (1 - 1/g)` clamped to `±100`, and
  `apply_adjustments` adds it to `ch_master_gain` instead of `exposure`.
  `AG_GMIN`/`AG_GMAX` (0.6/3.0) are exactly the slider's endpoints, so no
  reachable gain is clipped.
- Master Gain stays in the **`channels`** sync group rather than moving to
  `tone`: it is a Channel Levels key on the Channel Levels stage, and splitting
  one stage across two sync groups would be worse than the label change. The
  group's label becomes "Channel Levels (incl. Master Gain)".

**The `exposure` pipeline parameter survives** — it is not dead. It still carries
the legacy baked auto-exposure `eb` (default-slope conversions when Auto Gain is
off) and whatever an area layer sets programmatically. It simply has no slider.
`eb` was deliberately *not* moved: it is computed in `50·log2(g)` stops units and
consumed by the `/300` curve, a pre-existing mismatch (it delivers ~1.2× when it
asks for 2×), and re-pointing it at Master Gain would change the look of
default-slope conversions for anyone with Auto Gain off without fixing the
underlying units bug. That fix is its own change.

### Ordering note

Auto Gain moving from the `exposure` stage to Master Gain does **not** reorder it
relative to the per-channel work: after §3.2, Channel Levels already ran ahead of
`exposure`, and Master Gain is the last sub-stage of Channel Levels. Master Gain,
White Balance, White Point and `exposure` are all pure multiplies on un-clamped
`d`, so they commute — the only non-commuting neighbour is Master *Shift*, which
Master Gain already followed.

## 5. Data model

No key changes. `ch_input_gain`, `ch_master_shift`, `ch_master_gain`,
`ch_{r,g,b}_{shift,gain,blackpoint}` keep their names, `[-100, 100]` range and 0
default, so catalogs, the `channels` sync group, Copy Settings and area layers
need no migration.

Saved values are **reinterpreted, not migrated**: an existing catalog entry with
`ch_b_shift = 20` now applies twice the offset, at a different pipeline position.
Documented in the release notes as a look change for images carrying non-zero
Channel Levels; images with the section untouched (the overwhelming majority,
since it defaults to all zeros) render identically.

## 6. Integration points

| Area | File / function | Change |
|---|---|---|
| Stage helper | `ccr_processor.py` (new `_apply_channel_levels`) | The §3.1 math, `clamp` flag, `CH_MIN_RANGE` guard |
| Windowed recovery | `_apply_working_space_recovery` (`:1160`) | Accept the 12 channel params; run the stage first, `clamp=False` |
| CPU adjust | `adjust_image` (`:2456`) | Zero channel params when `ws_windowed`; else run the stage first (`clamp=True`); delete the old late block (`:2609-2652`) |
| GPU wrapper | `adjust_image_opencl` (`:2909`) | Forward channel params into the pre-stage, then zero them |
| OpenCL kernel | kernel source (`:363-400`) | Move the block to the top of the kernel; restructure to Input Gain -> per-channel -> Master; add the range guard |
| Default-slope invert | `_default_slope_invert` (`:1206`) | Drop the density floor on the WS branch; keep it on the legacy branch; gamma via `where=(out > 0)` |
| Window geometry | `_WS_LO` (`:1112`) | Default 0.5 -> 1.0 |
| Panel layout | `sliders_panel.py` (`:534`, `:655`) | Create/place `od_section` after the Convert row; drop the old placement + separator |
| Docs | `CLAUDE.md`, `RELEASE_NOTES.md` | Pipeline-order note; release entry |

## 7. Test plan

New `tests/test_channel_levels.py`:

1. **Neutral identity** — all twelve at 0 is a bit-exact no-op, windowed and not.
2. **Stage order** — Input Gain applies before the per-channel shift, Master
   after: assert `ch_input_gain` + `ch_r_shift` composes as `(in*ig + s)` and not
   `(in + s)*ig`, and that Master Shift is *not* summed into the channel shift (a
   case where the two orderings differ numerically).
3. **Shift reveals sub-black** (the headline behaviour) — build a windowed base
   holding a known negative `d`, apply a positive shift, assert the value becomes
   visible in `[0, 1]` at the expected level, and that the same shift applied
   post-clamp could not have (guard against regression).
4. **Shift preserves, not destroys, pushed-out data** — a `+shift` then an equal
   `-shift` round-trips within window quantization for in-window content.
5. **Film base stays black** — the motivating case: a base pixel at `d < 0` and a
   shadow pixel at `d > 0`, a `+blue` shift sized to lift the shadow; assert the
   shadow's blue rises while the base clamps to neutral black (all three channels
   equal at 0).
6. **Doubled strength** — a given slider value produces the offset/gain the `/150`
   and `2^(v/25)` mappings predict.
7. **Denominator guard** — `gain = +100, blackpoint = +100` yields a positive,
   finite, non-inverted result (the case that would divide by `-0.333`).
8. **Default-slope floor removed** — a pixel brighter than the black point encodes
   *below* `WS_B` with WS on, and still exactly at 0 with
   `FREECCR_WORKING_SPACE=0`; no NaN when `DEFAULT_DENSITY_GAMMA != 1`.
9. **Window geometry** — `WS_B == 1024`, `WS_W == 2048`, width still 1024;
   headroom still > 5.9 stops.
10. **CPU/GPU parity** — `adjust_image` vs `adjust_image_opencl` agree within 1
    LSB over a randomized channel-levels parameter sweep, windowed and not.
    Skipped when OpenCL is unavailable.
11. **UI placement** — the Channel Levels section's index in `scroll_layout` is
    greater than the Convert row's and less than the WB/AWB row's; the section is
    collapsed on construction; every channel slider reads 0.
12. **ADJUSTMENT_KEYS integrity** — the positional zip still maps every channel
    key to the slider with the matching label after the move.

Existing suites to update: `tests/test_working_space.py`
(`test_window_constants_are_10bit_default` and any `WS_LO = 0.5` assumption),
`tests/test_default_slope.py` (floor removal), `tests/test_auto_gain.py` (in-bound
exclusion of sub-black pixels).

### 6.1 Call sites verified

- `_apply_working_space_recovery` has three callers: `adjust_image` (`:2485`),
  `adjust_image_opencl` (`:2938`), and the identity fast-path in
  `ccr_image.apply_adjustments` (`:1173`, called as `(image, 0.0)`). The twelve
  new parameters default to 0, so the identity call site needs no change and
  stays a pure de-window + clamp.
- No consumer of the channel keys exists outside `ccr_processor.py`,
  `ccr_image.py` and `sliders_panel.py`. Copy Settings and Sync to All address
  them by name through the `channels` sync group, so both are unaffected.
- `exposure_base` rides the **`exposure`** argument, not `ch_input_gain`
  (`ccr_image.py:1186`), despite the stale comment at `ccr_image.py:225` naming
  "ch_input_gain units". Rescaling `ch_input_gain` therefore cannot disturb
  auto-exposure. Fix the stale comment while here; do not change the behaviour.

## 8. Resolved decisions

- **Strength factor: 2x**, uniform across shift, gain, blackpoint and input gain
  (`/300 -> /150`, `2^(v/50) -> 2^(v/25)`). Uniformity keeps the controls
  predictable relative to each other; revisit only if it overshoots in practice.
- **Cineon checkbox stays inside the section** and rides it to the top of the
  panel. Relocating a final-stage flag is a separate UI change and out of scope.
- **Auto Gain keeps measuring the raw base** (§3.6) rather than the
  channel-adjusted data, preserving its stability-across-a-drag property.

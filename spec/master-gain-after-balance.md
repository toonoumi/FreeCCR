# Master Gain (and Auto Gain) after Channel Balance

## 0. Why

Channel Balance is **tone-weighted**: its node sits at a fixed *display* value
(`BALANCE_NODE_X = 3/16`), so how much a Balance slider moves a given pixel
depends on where that pixel sits in the window. Master Gain is an **exposure**
control — a uniform scalar — and it was running *before* Balance, as the last
sub-stage of Channel Levels.

Putting an exposure control in front of a tone-weighted colour control couples
the two: brightening the image slides every pixel up past Balance's node, so the
correction the user dialled in at one exposure is a different correction at
another. **Changing the gain changed the colour.**

The coupling is not hypothetical, and it is not opt-in: the hidden **Auto Gain**
offset rides `ch_master_gain` and is ON by default for converted images
(`v = CH_SLIDER_DIV·(1 − 1/g)`), so every converted frame carried some gain ahead
of Balance. #130 already had to work around this in the neutral solve — the WB
picker measures Auto Gain from the whole base and folds it in before solving,
precisely because "Auto Gain moves the tone, and Balance is tone-dependent, so a
solve computed at the un-gained tone lands on the wrong part of the curve."

Moving Master Gain behind Balance removes the coupling at the source: Balance
sees a gain-independent tone, so exposure and colour are independent controls.

### Goals

- Apply Master Gain (and therefore the Auto Gain offset riding it) **after**
  Channel Balance, on every path: windowed, non-windowed, and OpenCL.
- Change nothing else about either control's math.

### Non-goals

- **Master Shift does not move.** It stays the last sub-stage of Channel Levels.
  It is not an exposure control but part of *placing the histogram in the
  window*, which is the whole reason Channel Levels runs ahead of Balance: on a
  windowed base much of the data sits outside `[0,1]`, where the Balance node is
  identity, so Levels must bring it into the window first. A shift is also a
  density offset — tone-uniform in the log domain — which is exactly the kind of
  correction that belongs before a tone-weighted one.
- No change to how the Auto Gain offset is *computed*. It is still measured from
  the whole base, placing the top 0.1% of in-range highlights near the top of
  the window. What changes is only where that gain is applied.
- No UI change. Master Gain keeps its slider, its position outside the Channel
  Levels collapsible, and its `ch_master_gain` key.

## 1. Order

```
  Channel Levels      Input Gain -> per-channel Shift/Gain/Blackpoint -> Master Shift
  Channel Balance     the tone-weighted node, un-clamped
  Master Gain         <- MOVED HERE (Auto Gain rides it)
  White Balance       flat per-channel multiply
  White Point / Gain  headroom recovery
  window clamp
```

Master Gain is a pure scalar, so it **commutes exactly** with the White Balance
multiply that follows — its placement between Balance and WB is the readable
one, not a mathematical requirement.

## 2. Implementation

`_master_gain_divisor(ch_master_gain)` is split out of `_apply_channel_levels`,
which gains `include_master_gain: bool = True`. Every pipeline caller passes
`include_master_gain=False` and applies the divisor itself after Balance:

| path | where |
|---|---|
| windowed (`_apply_working_space_recovery`) | after `_apply_channel_balance`, before the WB block, un-clamped |
| non-windowed (`adjust_image`) | in the single normalised Levels → Balance → Master Gain pass |
| OpenCL (`adjust_image_opencl`) | in the numpy pre-stage that already consumes Levels + Balance when Balance is active |

The default stays `True` so the parameter's meaning is "the whole stage", and a
direct caller of `_apply_channel_levels` (tests) still gets all of it.

**The kernel's own Channel Levels block is unchanged.** It only runs for a
non-windowed base with Balance *inactive* — when Balance is active, Levels,
Balance and Master Gain are all consumed in numpy and zeroed for the kernel — and
with Balance identity, moving a scalar across it is a no-op. So CPU/GPU parity is
preserved without touching the kernel.

### The non-windowed clamp

Channel Levels used to clamp to `[0,1]` on its own way out on non-windowed bases
(reference mode, positive mode, area layers), because those paths carry no
sub-black data and the later `pow()` stages must not see negatives. With Master
Gain moved out of that call, the three stages now run as **one normalised pass
with a single closing clamp** after Master Gain:

```
img /= 65535
  Channel Levels (clamp=False, include_master_gain=False)
  Channel Balance
  Master Gain
clip [0, 1]
img *= 65535
```

Keeping the clamp at the *end* rather than in the middle is what makes this
bit-identical to the old code whenever Balance is neutral: nothing between Levels
and Master Gain clipped before, so nothing may clip now. (An intermediate clamp
would have changed a negative Master Gain's result — a value at 1.5 with the gain
pulling down to ×0.5 gave 0.75 before and would have given 0.5.) The same closing
clamp is mirrored in the OpenCL pre-stage, because parity depends on the clamp
sitting in the same place on both paths.

## 3. Compatibility

Renders change **only when Channel Balance and a gain are both non-neutral** —
and since Auto Gain is on by default for converted images, that means in
practice: **every converted image with a non-zero Balance slider renders
differently.** Channel Balance shipped in #130, so the exposed surface is one
release of edits.

This is the intended correction, not a regression: those images were being graded
through a Balance node whose position depended on the Auto Gain the frame
happened to need. Nothing else moves — a frame with Balance at zero (the default,
and every frame from before #130) is bit-identical, which is what the regression
tests assert.

## 4. Test plan

- **Order.** On a windowed base, rendering with Balance + Master Gain equals
  rendering with Balance alone and then scaling by `1/_master_gain_divisor` — the
  gain is a scalar applied *after* the node, not before it.
- **Independence (the point of the change).** With Balance set, the R:G:B ratios
  of a mid-tone patch are the same at Master Gain 0, +40 and −40. Under the old
  order they diverge.
- **Auto Gain rides it.** The same holds when the gain arrives as an Auto Gain
  offset on `ch_master_gain` rather than a user slider value.
- **Non-windowed path.** Same order property through `adjust_image` on a
  full-range base.
- **No regression with Balance neutral.** `adjust_image` with any Channel Levels
  + Master Gain combination and Balance at zero is byte-identical to before the
  change (covered by `tests/test_channel_levels.py` and
  `tests/test_working_space.py` passing unchanged).
- **CPU/GPU parity** holds through `tests/test_opencl_accuracy.py`.

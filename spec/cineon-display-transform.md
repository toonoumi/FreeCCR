# Spec: Cineon Film Log → Workspace

Status: REFINED v3 — v1/v2 shipped this as a **final display transform** to
Rec.709 (γ 2.2). v3 moves it to **between Channel Levels and Channel Balance**
and encodes into the **working space's curve (sRGB)**. §6 records what changed
and why; everything above it describes the current behaviour.

Owner: FreeCCR
Original branch: `feature/color-scopes` (the parade's 95/685 reference lines mark
exactly this transform's black/white anchors)

## 1. Summary

A checkbox in the **Channel Levels** section — "Cineon Log → Workspace" — that
interprets the value as Cineon printing-density film log and **decodes it into
the app's working space**, immediately after Channel Levels. Standard Kodak
constants (DaVinci's "Cineon Film Log" flavour): 10-bit code 95 = black (Dmin),
code 685 = 90% white.

It is a **decode out of log**, not a display transform, and it marks the boundary
between the two domains. **Above it** is Channel Levels, the log-domain grading:
a per-channel shift there is a density offset, which is exactly the right tool on
log data. **Below it** is everything built for display-referred data — Channel
Balance's node, which sits at a fixed *display* value; Master Gain; White
Balance's flat per-channel multiply; the contrast S-curve pivoting on 0.5;
saturation's luma weighting; the Curves editor's 0–255 domain. With the decode at
the end of the chain, every one of those was grading log values instead.

## 2. Goals / Non-goals

### Goals
- Per-image on/off flag, stored in `adjustment_settings` under `"cineon_log"`
  (present-and-True when on; absent when off — like the non-slider `curves` key).
- Applied identically on preview, hi-res zoom detail, every export path, and both
  the CPU and OpenCL paths.
- Participates in undo/redo, Reset, Compare, Copy/Paste, Sync-to-All (rides the
  "Channel Levels" group) and the catalog.
- Whole-image only: area layers never carry the flag, and they grade the
  **already-decoded** base. The checkbox is disabled while an area layer is the
  edit target.

### Non-goals
- **No matrix.** The transform is transfer-function-only, as it always was: the
  primaries never change, because the log data is already in the working space's
  primaries. "→ workspace" is about the curve.
- No soft-clip shoulder and no configurable black/white codes.
- No OpenCL kernel port: it runs in the shared numpy pre-stage, like Channel
  Balance and White Balance, so CPU/GPU parity is automatic.

## 3. Math (`ccr_processor.apply_cineon_to_workspace`)

Input is a float **display value** `v` (0 = display black, 1 = display white),
≡ 10-bit code `c = v · 1023`:

```
off = 10^((95 − 685) · 0.002 / 0.6)                     # ≈ 0.0108
lin = (10^((c − 685) · 0.002 / 0.6) − off) / (1 − off)  # scene-linear, 685 → 1.0
out = srgb_encode(max(lin, 0), clip=False)
```

0.002 = log₁₀ density per code value, 0.6 = film negative gamma — the standard
Kodak constants. `10^x` is evaluated as `exp2(x · log₂10)`, which is materially
faster over a full frame and identical in value.

Two properties matter as much as the curve:

- **Floored, not clipped.** Values below Cineon black are floored at 0 (the sRGB
  power segment is undefined below it, and a NaN would poison every stage after).
  Values above white are **not** ceilinged: the stage runs inside the un-clamped
  working-space region, and the highlight headroom has to survive for the White
  Point recovery immediately below it.
- **Float, not a LUT.** The old uint16 LUT cannot represent an un-clamped input,
  so the transform is computed in float32 like its neighbours.

## 4. Position

```
  Channel Levels    Input Gain -> per-channel -> Master Shift    un-clamped
  Cineon Log -> Workspace   <- HERE                             un-clamped
  Channel Balance                                               un-clamped
  Master Gain (+ Auto Gain)                                     un-clamped
  White Balance     temperature / tint
  White Point / legacy exposure
  window clamp
  brightness, highlights/shadows, black/white point, contrast, saturation, bands
  Gamma curve -> Curves -> area layers -> B&W collapse
```

Consequences, all intended:

- **White Balance becomes a real colour shift.** A flat per-channel multiply on a
  log value is a per-channel *gamma* change — the finding that removed
  Temperature/Tint in #130 (see spec/white-balance-restore.md). After the decode
  the multiply lands on display-referred data, which is what it was designed for.
- **Channel Balance grades the decoded image.** Its node is anchored at a fixed
  *display* value (`BALANCE_NODE_X = 3/16`), so on log data it lands on a
  different tone than the one it names.
- **Master Gain scales the decoded image** too, since it follows Balance.
  Note that the hidden Auto Gain offset rides it and is still *measured* on the
  log base, so with the flag on its highlight placement is approximate.
- **A strong cast may exceed what the colour controls can reach.** The decode
  turns a cast from a log *offset* into a linear *ratio*: a 4:1 R/B ratio is
  reachable after decoding a cast that was mild in log, while the WB gains span
  at most 1.4/0.6 = 2.33:1 and the Balance node runs out at slider 100. Neither
  is a defect of the solve — both peg the knob and keep their closest result —
  and both say the same thing: a cast that strong belongs in **Channel Levels,
  above the decode**, where a per-channel shift is a density offset, the exact
  correction for it. Pinned by `test_the_decoded_base_limits_what_wb_can_reach`
  and `test_the_decoded_base_can_outrun_the_node`.
- **Area layers grade the decoded base**, since they composite after
  `adjust_image` returns.

## 5. Integration

1. **Parameter**: `cineon_log: bool` is appended at the END of the
   `adjust_image` / `adjust_image_opencl` / `_apply_working_space_recovery`
   signatures — those are called positionally in tests, so a mid-signature insert
   would shift every later argument (the same rule `balance_*` follows).
2. **Windowed base**: applied inside `_apply_working_space_recovery`, after the
   Channel Levels block and before Channel Balance; the callers then zero the
   flag so nothing can apply it twice.
3. **Non-windowed base** (reference mode, positive mode, area layers): applied in
   `adjust_image`'s single normalised Levels → decode → Balance → Master Gain
   pass, before its closing clamp.
4. **OpenCL**: consumed in the numpy pre-stage, which now runs when Balance is
   active **or** the flag is set — the kernel would otherwise apply Channel
   Levels after the decode and reverse the order.
5. **`ccr_image.apply_adjustments`** passes `cineon_log=bool(s.get('cineon_log'))`
   into the adjustment call. It no longer applies anything itself.
6. **UI** (`sliders_panel`): checkbox at the bottom of the Channel Levels section.
   Toggle = a discrete single-undo edit on the global dict. `_attach_cineon`
   re-attaches the flag on the dict-rebuild paths; `_load_active_layer` populates
   and disables it for area layers; paste strips it onto an area layer;
   `"cineon_log"` rides the "channels" sync group.

## 6. What changed in v3, and why

| | v1/v2 | v3 |
|---|---|---|
| position | final stage, after curves and areas | between Channel Levels and Channel Balance |
| output curve | Rec.709 gamma 2.2 | working space (sRGB) |
| implementation | cached 65536-entry uint16 LUT | float32, un-clamped |
| clipping | linear clipped to [0,1] in-stage | floored at 0, headroom kept |

Both changes follow from what the stage is *for*. Called a display transform, it
belonged at the end; called a decode out of log, it belongs where the log ends —
and then the curve it encodes into should be the working space's, not a video
standard's. The transform never applied a matrix, so the Rec.709 label described
only the gamma.

**Renders change** for every image with the flag set: the stage moved and its
curve changed. `cineon_log` keeps its key and its meaning, so nothing in the
catalog needs migrating, but a frame graded under v2 will need re-grading.

## 7. Test plan (`tests/test_cineon_log.py`)

- **Curve**: code 95 → 0 and code 685 → 1.0; monotonic across 0…1.2; a mid-scale
  value matches the closed form **with the sRGB curve**, and is measurably
  different from the old `^(1/2.2)` (so the test cannot pass on the old encode);
  neutral input stays neutral.
- **Un-clamped**: a value above white comes out above 1 (headroom survives);
  below-black input is floored, finite, never NaN.
- **Position**: with nothing else set the render is exactly the decoded image;
  Channel Levels applied with the flag equals the decode *of* the levelled
  render (it is above the decode); Channel Balance, Master Gain, White Balance
  and contrast each equal that stage applied *to* the decoded render (they are
  below it).
- **Windowed**: the recovery consumes it (anchors land on black/white through the
  window), and a blown highlight is still recoverable by White Point afterwards.
- **Flag off** is bit-identical to not passing it.
- **Model + panel**: `apply_adjustments` forwards the flag; the "channels" sync
  group carries the key; the checkbox exists, is unchecked by default and is
  labelled "Cineon Log → Workspace".

# Restoring White Balance / Tint, and demoting Channel Balance

## 0. Why

PR #130 removed Temperature/Tint and put **Channel Balance** in their panel slot.
The argument for the removal was narrow and correct as far as it went: on a `bw`
or no-anchor conversion the display value **is** optical density, so White
Balance's flat per-channel multiply is a per-channel *gamma* change, not a colour
shift. Temperature/Tint really was a contrast control on those bases.

That argument does not cover every base the app produces:

- **`ref` conversions** — after the affine normalise + linear `65535 - v`
  inversion the value is scene-linear-ish, and a per-channel multiply is real
  white balance. #130 conceded this in passing.
- **Positive mode** — `CCRImage.read_image` decodes positives with
  `output_color=sRGB` + sRGB gamma + `use_camera_wb=True`
  (`src/core/ccr_image.py:527`). No density anywhere: it is an ordinary
  display-referred positive, and an illuminant cast on it is approximately
  *multiplicative*. Channel Balance's node is a gamma move anchored at
  `x = 3/16`, so it bites hardest in the low midtones and asymptotes toward
  white — it over-corrects the shadows and under-corrects the highlights of
  exactly the cast a slide scan carries.

So both controls have a domain, and the app should carry both. This spec
restores White Balance as the primary, front-of-panel colour control and demotes
Channel Balance to the crossover tool it actually is.

### Goals

- Restore the **Temperature** and **Tint** sliders, in their pre-#130 slot
  directly above Brightness, with their pre-#130 groove gradients.
- **WB Picker** and **AWB** drive Temperature/Tint again — the "normal" white
  balance behaviour.
- Move the three **R/G/B Balance** sliders into their own **collapsible**
  section, placed under Channel Levels, **collapsed by default**.
- Keep every pipeline behaviour byte-identical. Nothing in
  `ccr_processor.py` changes: the Temperature/Tint stage was never removed, only
  its UI, and Channel Balance keeps its stage and its position.

### Non-goals

- No change to the **math** of either control. Temperature/Tint keeps
  `_white_balance_gains`; Channel Balance keeps `BALANCE_NODE_X = 3/16` and
  `BALANCE_MAX_STOPS = 1.5`.
- No gating of White Balance by conversion mode. It is available everywhere, as
  before #130; the `bw`-base caveat is documented, not enforced. (A mode-gated
  control was considered and rejected: `ref`/`bw`/positive/area-layer bases all
  reach the same panel, and a slider that disappears is worse than one that is
  the wrong tool for one base.)
- No return of the **analytic** neutral inverse (`compute_neutral_temp_tint`).
  See §4.
- No new hotkeys, and no change to what U/I/O and J/K/L do. They become
  **opt-in** (see below), but the nudge behaviour itself is untouched.

## 1. UX / Interaction

Panel order, top to bottom (only the changed region shown):

```
  ...
  Convert Current / Convert All
  [+ Channel Levels]            collapsible, existing position
  [+ Channel Balance]           NEW collapsible, collapsed by default
      R Balance                     "  cyan -> red     groove
      G Balance                     "  magenta -> green
      B Balance                     "  yellow -> blue
  Master Gain                   always visible, outside both collapsibles
  ------------------------------
  [WB Picker] [AWB] [Crop] [Slice]
  Color Profile
  Temperature                   RESTORED, TEMP_GRADIENT (blue -> amber)
  Tint                          RESTORED, TINT_GRADIENT (green -> magenta)
  Brightness
  Gamma
  ...
```

This is pipeline order: Channel Levels → Channel Balance → Master Gain →
White Balance → tone, which is exactly the order `_apply_working_space_recovery`
runs them in (Master Gain moved behind Balance in
[`spec/master-gain-after-balance.md`](master-gain-after-balance.md); the panel
follows it).

- The Channel Balance section is a plain `CollapsibleSection`, whose default is
  already collapsed. It is not persisted per image — like Channel Levels and
  Curves, it is a panel-level disclosure, not a setting.
- The **Channel Balance nudge keys** (U/I/O raise R/G/B, J/K/L lower) become
  **opt-in**, off by default, under a new **Keyboard** group on Settings →
  General. Two reasons: the sliders they move are now hidden behind a collapsed
  section, so the keys would edit controls the user cannot see; and six single
  letters is a large share of the keyboard for one control. The flag is
  `ccr_backend.balance_hotkeys`, persisted as `adjust/balance_hotkeys`, and it
  gates the six `QShortcut`s with `setEnabled` rather than an early return in
  the handler — a disabled `QShortcut` does not consume its key, so while the
  setting is off those letters are entirely free. `nudge_balance` itself is
  unchanged, so the sliders still work by hand.
- **WB Picker** samples a 7×7 neighbourhood and sets Temperature/Tint so that
  spot renders neutral. **AWB** does the same from a whole-frame estimate.
  Tooltips, the Settings → General blurb and the hint text all say
  Temperature/Tint again.
- The post-conversion **Auto WB** hook (`ccr_backend.maybe_auto_awb`) writes
  `temperature`/`tint`, and skips an image that already has either set.
  Images auto-balanced under #130 carry `balance_*` instead; those values still
  render exactly as they did, and the hook does not touch them.

## 2. Data model

`SlidersPanel.ADJUSTMENT_KEYS` is zipped **positionally** against
`create_slider()` call order, so both change together:

```
"temperature", "tint", "brightness", "gamma", "highlights", "white_point",
"shadows", "black_point", "contrast", "saturation", "sub_saturation",
<12 ch_* Channel Levels keys>,
"balance_r", "balance_g", "balance_b",     <- moved to here
<band keys>, "band_feather",
```

The Balance keys move from the head of the list to just after the Channel Levels
keys, so the creation block that populates the collapsible sections populates
Channel Levels then Channel Balance — the same order they appear on screen, and
the same order the pipeline runs them.

Persistence needs **no migration**: adjustment settings are a dict keyed by name,
so a catalog written under #130 keeps its `balance_*` values and gains
`temperature`/`tint` defaults of 0, and a catalog written before #130 keeps its
`temperature`/`tint` and gains `balance_*` defaults of 0. Both render as they
always did.

`SYNC_GROUPS` (the Sync to All / Copy Settings group list):

- `("wb", "White Balance / Tint", ("temperature", "tint"))` — the id `wb` and its
  pre-#130 keys and label are restored, so a remembered `{gid: bool}` selection
  keeps meaning "the white balance controls".
- `("balance", "Channel Balance (R/G/B)", ("balance_r", "balance_g", "balance_b"))`
  — new, inserted after `channels` to match the panel's adjacency.

The invariant that the adjustment-key groups partition `ADJUSTMENT_KEYS` exactly
still holds, and is still asserted.

## 3. Processing / math

**Unchanged.** `_apply_working_space_recovery` already runs, in order:

1. Channel Levels (un-clamped)
2. Channel Balance (un-clamped, identity outside [0,1])
3. White Balance — flat per-channel gain from `_white_balance_gains`
4. White Point / Gain recovery, then the window clamp

and `adjust_image` / `adjust_image_opencl` mirror that order on the non-windowed
and GPU paths. #130 kept the White Balance stage alive for legacy catalogs, so
restoring the sliders is pure UI plumbing — the parameters were never
disconnected.

For the record, the gains being restored to the UI:

```
temperature:  s = (slider/100) * 0.40          R *= (1+s),  B *= (1-s)
tint:         t = tanh(slider*0.02) * 0.26 * balance_factor
                                               G *= (1-t),  R,B *= (1+0.3t)
```

## 4. The neutral solve

The WB Picker and AWB retarget from `balance_*` to `temperature`/`tint`, but they
keep the **closed loop on the real render**. This is the one part of #130 that
must not be reverted: three analytic inverses shipped before it and each was
wrong in some configuration, because inverting any colour stage means modelling
every stage between it and the pixel the user sees — Channel Levels, the hidden
Auto Gain offset, gamma, the tone sliders, saturation, per-channel Curves, the
Cineon transform. `compute_neutral_temp_tint` is not reinstated.

`CCRImage`'s solver is generalised over *which knobs it turns*:

- `_solve_neutral(knobs, sample_patch, ...)` — the shared loop. Renders the
  sample area through the real pipeline (`apply_adjustments` with
  `areas_override=[]`, `skip_dust=True`, and `auto_gain_override` measured from
  the whole base), measures what comes out, moves one knob by integer bisection
  on the measured output, repeats. Keeps the best-so-far result, so more passes
  can never regress, and leaves a knob alone when a full sweep does not move the
  measurement (a Black & White profile).
- `solve_neutral_balance(...) -> (balance_r, balance_g, balance_b)` — unchanged
  behaviour: red is the anchor, blue then green are driven onto the mean of the
  other two measured channels.
- `solve_neutral_wb(...) -> (temperature, tint)` — **new**, and what the UI uses.

`solve_neutral_wb` differs from the Balance solve in the shape of its objective,
because Temperature and Tint are not per-channel knobs:

| knob | objective driven to zero | monotone because |
|---|---|---|
| Temperature | `R - B` of the rendered mean | `R *= (1+s)`, `B *= (1-s)`, and every later stage is monotone increasing per channel |
| Tint | `G - (R+B)/2` of the rendered mean | `G *= (1-t)` while `R,B *= (1+0.3t)` |

Each is solved by the same integer bisection over `[-100, 100]`, with the
direction detected from the two endpoint probes rather than assumed (a `combine`
reduction may be monotone *decreasing* — `gray_edge` reports gradient energy,
which falls as a channel is lifted). Both knobs are solved from a fixed `(0, 0)`
start and over the full range each pass, so the result does not depend on the
image's current Temperature/Tint — the solve replaces the white balance rather
than accumulating onto it, and is idempotent.

Two knobs and two independent objectives means the solve converges in far fewer
passes than the Balance one, but the outer loop, its relative tolerance
(`NEUTRAL_SOLVE_TOL`), its relative flat-response guard (`_NEUTRAL_FLAT_REL`) and
its keep-best net are shared verbatim.

**AWB** (`core/awb.py`) gains `compute_awb_wb(image, algorithm=None)` alongside
`compute_awb_balance`. Everything before the solve is shared and unchanged: crop
the base, downscale to 256px (spatially intact, because `gray_edge` needs
neighbours), bail out if the estimator cannot read the base, then either
neutralise the rendered frame's own estimate (scaled to the render's 0–65535
domain) or, for `gray_edge`, the rendered mean of the pixels
`gray_edge_pixel_mask` selected once on the base.

`solve_neutral_balance` and `compute_awb_balance` are **retained** as the tested
inverse for the Channel Balance control, and keep their test suites. No UI path
calls them after this change.

## 5. Integration points

| File | Change |
|---|---|
| `src/widgets/sliders_panel.py` | `ADJUSTMENT_KEYS` order; Temperature/Tint sliders restored above Brightness; `balance_section` collapsible created under the Master Gain slot and populated after Channel Levels; `SYNC_GROUPS` `wb` restored + `balance` added; `on_wb_sampled(temp, tint)`; `_on_auto_wb` → `compute_awb_wb`; button tooltips |
| `src/widgets/image_preview.py` | `_sample_wb_point` → `solve_neutral_wb`; docstrings |
| `src/core/ccr_image.py` | `_solve_neutral` extracted; `solve_neutral_wb` added |
| `src/core/awb.py` | `compute_awb_wb` added; shared prep factored out of `compute_awb_balance` |
| `src/core/ccr_backend.py` | `maybe_auto_awb` writes `temperature`/`tint` again |
| `src/ui/theme.py` | `TEMP_GRADIENT` / `TINT_GRADIENT` restored (Balance gradients kept) |
| `src/ui/main_window.py` | Auto WB toggle wording; `balance_hotkeys` restored from QSettings before the shortcuts are built, `_apply_balance_hotkey_state()`, `on_balance_hotkeys_toggled()` |
| `src/core/ccr_backend.py` | `balance_hotkeys` flag (default False) |
| `src/widgets/settings_dialog.py` | Auto WB blurb wording; new **Keyboard** group with the nudge-keys checkbox, staged like every other toggle |
| `src/core/ccr_processor.py` | **none** |

## 6. Test plan

New — `tests/test_white_balance_restore.py`:

- **Panel wiring.** `temperature`/`tint` are back in `ADJUSTMENT_KEYS`;
  `len(sliders) == len(adjustment_keys)`; setting a slider well below the
  insertion point (Brightness, a `ch_*` key, a band key) writes *that* key and
  nothing else — the positional zip is the failure mode with no error message.
- **Placement.** Temperature and Tint are the two rows directly above
  Brightness in the scroll layout; the Channel Balance section widget sits
  after the Master Gain row and before the WB/Crop button row.
- **Default collapsed.** The Channel Balance section's content is hidden on
  construction, and toggling shows it; the three Balance sliders are inside it,
  not in the scroll layout.
- **Gradients.** Temperature/Tint carry `TEMP_GRADIENT`/`TINT_GRADIENT`, Balance
  keeps its channel gradients.
- **Sync groups.** `wb` → `("temperature", "tint")`; `balance` → the Balance
  trio; the groups still partition `ADJUSTMENT_KEYS`; the group-id order is the
  documented one.
- **The solve.** Against the same `PRESETS` matrix the Balance solver is tested
  under (nothing set / Channel Levels / Master Gain / Input Gain / Cineon /
  tone+saturation / per-channel Curves / everything), a picked spot renders
  neutral: spread > 0.2 before, < 0.02 after. Plus: a flat response (Black &
  White profile) returns `(0, 0)` rather than pegging; an empty patch returns
  `(0, 0)`; more passes never regress; the result is idempotent.
- **Apply path.** `on_wb_sampled(temp, tint)` writes both sliders, renders
  before it shows, and leaves no pending debounced reprocess.
- **Nudge keys.** A fresh `CCRBackend` has `balance_hotkeys` False; the six
  shortcuts follow the flag through `_apply_balance_hotkey_state`;
  `on_balance_hotkeys_toggled` sets the flag, persists
  `adjust/balance_hotkeys`, and re-applies the shortcut state; `nudge_balance`
  itself still works regardless. The dialog's checkbox seeds from the backend
  and stages until Done, like every other toggle on the page.
- **AWB.** `compute_awb_wb` corrects a cast end-to-end under every algorithm id,
  returns `None` with no base, and is idempotent. `maybe_auto_awb` writes
  `temperature`/`tint`, never clobbers either when set, and stays inert when the
  toggle is off or the image is unconverted.

Updated:

- `tests/test_channel_balance_ui.py` — drop `test_temperature_and_tint_are_gone`;
  the Balance keys no longer lead the zip; the `wb` sync group no longer carries
  them. The nudge-hotkey and closed-loop solve tests are unchanged.
- `tests/test_awb.py` — the hook tests assert `temperature`/`tint`; the panel
  apply tests use the two-argument `on_wb_sampled`. The estimator and
  `compute_awb_balance` tests are unchanged.
- `tests/test_copy_settings_dialog.py`, `tests/test_orientation_sync_group.py`,
  `tests/test_sub_saturation_crop_undo.py` — the `wb` group's keys and the
  expected group-id list.

Unchanged and expected to pass verbatim as the regression proof:
`tests/test_white_balance_ws.py`, `tests/test_working_space.py`,
`tests/test_channel_levels.py`, `tests/test_channel_balance.py`,
`tests/test_opencl_accuracy.py`.

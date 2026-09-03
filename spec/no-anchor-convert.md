# Spec: No-anchor conversion (Direct Invert)

Status: REFINED v1 — ready to implement
Owner: FreeCCR
Feature branch: `feat/no-anchor-convert`

## 1. Summary

Every conversion path today requires the user to sample something first: a
reference frame, or a black point (with an optional white point). Those samples
exist to *normalize* the negative — to decide where black and white land per
channel — because until now nothing downstream could recover that placement.

Channel Levels changed that. It is now the first pipeline stage, runs un-clamped
before the window clamp, and carries per-channel shift/gain/blackpoint plus a
Master Gain that Auto Gain already drives (`spec/channel-levels-pre-clamp.md`).
Per-channel placement is therefore a *grading* decision the user can make after
the fact, with live feedback, instead of a sampling decision they must get right
before seeing anything.

So: allow **Convert with no anchors at all**. The conversion becomes a plain
per-channel flip of the scan — `d = 1 − v/65535` — and the user grades from there
with Channel Levels. Because an unanchored conversion is a genuine departure from
the normal workflow (and easy to hit by accident when a black point was simply
forgotten), it is gated behind a confirmation warning that can be switched off in
Settings.

## 2. Goals / Non-goals

### Goals
- **Convert Current** and **Convert All** work with no black point and no
  reference frame. The conversion flips each channel across the container.
- A **warning dialog** explains what an unanchored conversion means and asks for
  confirmation. **On by default**, switchable off in Settings → General.
- The result is a normal converted image: it replays identically at preview,
  hi-res zoom, export, slice and catalog reload, and every Channel Levels /
  adjustment control works on it as usual.
- Auto Gain still applies, so an unanchored conversion is not left absurdly dark
  or bright before the user touches anything.

### Non-goals
- No new inversion math beyond the flip — deliberately the simplest possible
  transform, so what the user grades is the scan itself.
- No auto-estimation of anchors (that is what the reference-frame mode is for).
- No change to the reference, black-point-only, or two-point conversions.
- No new conversion **mode** in `conversion_inputs` (see §4).

## 3. Processing / math

`_flip_invert(img_f)` — per channel, on the float scan:

```
d = 1 - v / 65535
```

then the existing tail: `encode_window(d)` when the working space is on, else
`clip(d, 0, 1) * 65535 → uint16`.

The scan is already scaled to the full 16-bit container by `read_image`
(`* 65535/white_level`, `ccr_image.py:776-779`), so the flip is meaningful
without any further normalization. Input is `uint16`, so `d` lands exactly in
`[0, 1]`: this mode uses neither highlight headroom nor shadow margin, and it
does not need to — Channel Levels supplies both ends afterwards, and anything it
pushes out of the window lands in the headroom that already exists.

No black point means no film-base reference, so:
- **No sprocket/clear-film mask.** `compute_sprocket_alpha` already returns
  `None` for a `None` black point (`ccr_processor.py:1076`), so the reversal-look
  overlay is simply absent. No change needed.
- **No film-stock slopes and no density toggle.** Both are anchor-relative and
  are ignored (they are already gated on the black-point-only / two-point paths).

## 4. Data model: no new mode

The flip is modelled as the **existing `mode: "bw"` recipe with both anchors
`None`** — `conversion_inputs = {"mode": "bw", "bw": (None, None), "fine_rot": …,
"density": False, "slopes": None}` — rather than a new `"flip"` mode.

This is not a shortcut; it is what the recipe actually is: the B/W-point pipeline
with zero points sampled. The payoff is that **every existing dispatch site keeps
working untouched**, because they all destructure `ci["bw"]` and hand it straight
to the converter:

| Site | File | Works via |
|---|---|---|
| Hi-res zoom replay | `ccr_image.py:1357` | `apply_bwpoint_normalization(img, None, None)` |
| Profile re-grade replay | `ccr_backend.py:862` | `ccr_normalize_with_bwpoint(obj, None, None)` |
| Export | `ccr_backend.py:1585` | same |
| Slice child | `ccr_backend.py:1995` | passes `parent_ci["bw"]` through |
| Slice reset (parent) | `ccr_backend.py:2158` | passes `bw_points` through |
| Duplicate | `ccr_backend.py:1836` | copies the dict |
| Density re-apply | `ccr_backend.py:1306` | excluded — requires a non-None white point |

Adding a tenth `"flip"` branch to each of those is strictly more code and more
places to forget. The two functions that must learn about `None` are the
converters themselves (§5), plus one catalog fix.

**Catalog serialization is the one real change.** `_ci_to_json` /
`_ci_from_json` (`catalog.py:118-120`, `133-135`) do `list(black)` / `tuple(black)`
unconditionally and would raise `TypeError` on `None`. Both need the same
None-guard the white point already has.

## 5. Integration points

| Area | File / function | Change |
|---|---|---|
| Flip helper | `ccr_processor.py` (new `_flip_invert`) | `d = 1 − v/65535`, windowed or legacy tail |
| Preview/export convert | `ccr_normalize_with_bwpoint` (`:1580`) | `black_point_bgr=None` default; dispatch to `_flip_invert` when it is None; docstring |
| Replay | `apply_bwpoint_normalization` (`:2133`) | same three-way dispatch (flip → default-slope → two-point) |
| Catalog | `_ci_to_json` / `_ci_from_json` | None-safe black point |
| Convert (single) | `sliders_panel._on_convert_current_bwpoint` (`:1873`) | replace the hard block with the §6 warning; snapshot `bw: (None, None)` |
| Convert (all) | `sliders_panel._on_convert_all_bwpoint` (`:1919`) | same |
| Convert-all worker | `ccr_backend.apply_bwpoint_to_all_images` (`:2270`) | drop the `raise ValueError` on a missing black point |
| Mode label | `sliders_panel._update_bwp_mode_label` (`:1697`) | name the no-anchor state instead of showing nothing |
| Cleared-points hint | `sliders_panel` (`:1693`) | stop saying a black point is required |
| Setting (state) | `ccr_backend` | `warn_no_anchor_convert: bool = True` |
| Setting (persist) | `main_window` | load/save `convert/warn_no_anchor`; toggle handler |
| Setting (UI) | `widgets/settings_dialog.py` General page | "Conversion" group + checkbox |

### 5.1 Drive-by fix

The Auto gain help text in Settings still says "without moving the Gain slider"
and "control exposure with the Gain slider alone". That slider was removed when
Auto Gain moved onto Master Gain. Update both sentences to name Master Gain.

## 6. UX / interaction

### The warning

Shown when **Convert Current** or **Convert All** is pressed with
`black_point_bgr is None`, and `ccr_backend.warn_no_anchor_convert` is true:

> **Convert without a black point?**
>
> No film base has been sampled, so the conversion will simply invert each
> channel with no colour or density normalisation. Expect a strong cast — use
> **Channel Levels** (Master Gain, per-channel Shift / Gain / Blackpoint) to
> grade it.
>
> Set a Black Point first for a normalised conversion.
>
> [Convert Anyway] [Cancel]   ← Cancel is the default button

Cancel is default so a stray Return does not convert a roll unanchored. For
Convert All the existing "convert all N images?" confirmation still follows, so
the batch case asks twice — deliberate, since it is the destructive one.

When the setting is off, both buttons convert immediately with no dialog.

### The mode label

`bwp_mode_label` currently renders empty (and hides) with no black point. It now
reads:

> Anchor source: none — direct invert, grade with Channel Levels

so the panel always states what the next conversion will do. It stays hidden only
in Positive mode, where the section is disabled anyway.

### Settings

Settings → General gains a **Conversion** group:

- [x] **Warn when converting without a black point** — "Converting with no
  sampled film base inverts each channel with no normalisation. Turn this off to
  skip the confirmation and convert straight away."

Persisted under `convert/warn_no_anchor`, default **true**.

## 7. Test plan

New `tests/test_no_anchor_convert.py`:

1. **Flip math** — `_flip_invert` maps 0 → display white, 65535 → display black,
   mid → mid, per channel independently; windowed and legacy tails both correct.
2. **Endpoints in the window** — with the working space on, 65535 encodes to
   exactly `WS_B` and 0 to exactly `WS_W`.
3. **Dispatch** — `apply_bwpoint_normalization(img, None, None)` equals
   `_flip_invert(img)`; a black point still routes to the default-slope path and
   a black+white pair to the two-point path (no regression in the three-way
   dispatch).
4. **Channel independence** — a channel-varying scan flips each channel on its
   own; no cross-channel normalisation happens.
5. **Replay agreement** — the preview conversion and
   `apply_bwpoint_normalization` produce identical output for the same scan (the
   zoom/export replay contract).
6. **Catalog round-trip** — `_ci_to_json` / `_ci_from_json` survive
   `bw: (None, None)` and return it unchanged; a normal `(black, None)` and a
   `(black, white)` recipe still round-trip.
7. **Sprocket mask absent** — `compute_sprocket_alpha(scan, None)` is `None`, so
   an unanchored conversion carries no clear-film overlay.
8. **Backend allows it** — `apply_bwpoint_to_all_images` no longer raises with
   `black_point_bgr is None`.
9. **Setting default + persistence** — `warn_no_anchor_convert` defaults true;
   the toggle writes `convert/warn_no_anchor`.
10. **Warning gating (UI)** — with the flag on, converting with no black point
    raises the dialog and a Cancel leaves the image unconverted; with the flag
    off, no dialog appears. Driven by monkeypatching the message box.
11. **Mode label** — reads the no-anchor text when no black point is set, and
    the existing texts otherwise.

## 8. Resolved decisions

- **No new `conversion_inputs` mode** (§4): `bw: (None, None)` keeps ten dispatch
  sites unchanged and is an honest description of the recipe.
- **A plain flip, not a density inversion.** The point of the mode is that no
  anchor is assumed; a density inversion needs a base to measure density
  *against*, which is exactly what is missing.
- **The warning is Settings-only.** No "don't show this again" checkbox in the
  dialog itself — one off-switch, in the place the user specified.
- **Auto-exposure (`eb`) still applies**, since its trigger is
  `white_point_bgr is None` and Auto Gain supersedes it whenever it is on. No
  reason to special-case the unanchored path.

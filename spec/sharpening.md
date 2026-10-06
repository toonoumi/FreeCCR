# Sharpening — the Details section

## Summary

A collapsible **Details** section in the sliders panel with three controls —
**Amount** (default 25), **Radius** (default 25 ≈ 1.0 px) and **Masking**
(default 0) — applying a luma-only unsharp mask as the final stage of the
adjustment chain.

The default is **visible, not hidden**: the slider reads 25 on a new image, so a
user can see what is being applied and can zero it. The purpose is to let the
user *see the grain* and confirm the scan is in focus, which a soft render hides.

## Goals

* A capture-sharpening control with a sane non-zero default, visible on the
  slider rather than baked silently into the render.
* **Sharpening is applied in the preview**, not deferred to export — the user
  must be able to judge focus and grain without exporting.
* The 1080 preview **simulates what the full-resolution file looks like when
  inspected at 100 %**, so the preview communicates intent (see Processing).
* At 100 % zoom and on export the radius is exact, so what is inspected closely
  and what is written to disk agree.
* Luma-only, so a negative's inverted shadows do not get their chroma noise
  amplified.
* CPU and OpenCL paths produce identical output.

## Non-Goals

* **Area layers.** Sharpening is global; area layers grade colour, not detail.
* **A separate thumbnail path.** None is needed:
  `update_thumbnail_and_preview` calls `apply_adjustments(self.resized_raw)`
  **once** and derives both the 1080 preview and the 156 px thumbnail by
  resizing that single adjusted buffer. The thumbnail is therefore a downscale
  of the sharpened preview, which attenuates the effect naturally.
* **Deconvolution.** A blur-matched capture sharpen (e.g. modelling the
  demosaic's bilinear kernel) is a separate, later feature. This is a plain USM.
* **Recovering lost MTF.** Sharpening amplifies what survived, noise included.
  It is not a fix for microlens scatter or a surviving OLPF.

## UX

A `CollapsibleSection("Details")`, **collapsed by default**, matching the
existing Channel Levels / Channel Balance / Curves sections. It sits last, below
Subtractive Saturations — matching the render order, where sharpening is the
final stage.

| Slider | Range | Default | Meaning |
|---|---|---|---|
| Amount | 0–100 | **25** | strength of the high-pass add |
| Radius | 0–100 | **25** | native-pixel radius; 25 ⇒ 1.0 px |
| Masking | 0–100 | **0** | 0 = sharpen everywhere; higher = edges only |

All three use the panel's existing 0–100 integer convention, like every other
slider — no bespoke units. `Radius = value / 25` px, so the default reads 25 and
means exactly 1.0 px at full resolution, and the range tops out at 4.0 px.

## Data Model

Three new adjustment keys, appended to `SlidersPanel.ADJUSTMENT_KEYS` **after**
`band_feather` so the positional zip is unaffected:

```
"sharpen_amount", "sharpen_radius", "sharpen_masking"
```

`SLIDER_DEFAULTS` gains `sharpen_amount: 25` and `sharpen_radius: 25`
(`sharpen_masking` defaults to 0 like everything else).

**Backward compatibility — this is the part that must not be got wrong.** A
catalog written before this feature has an `adjustment_settings` dict with no
`sharpen_*` keys. Falling through to `SLIDER_DEFAULTS` would silently apply 25
sharpening to every scan ever made. So `catalog._restore_image` injects
`sharpen_amount = 0` (and radius/masking 0) when the restored dict is non-empty
and carries no `sharpen_amount`. Old scans keep rendering exactly as they did;
only newly created images get the 25 default.

## Processing

### Where it runs

The **very last stage**, after Channel Levels, Balance, White Balance, tone,
saturation, bands and curves. It is output-referred: it sharpens what the user
sees, not the working-space base.

It lives in `apply_adjustments`, **outside** `adjust_image` /
`adjust_image_opencl`, for two reasons that only became clear against the real
code:

* **Gamma, the curves, the area composite and the B&W collapse all run AFTER
  `adjust_image`** in `apply_adjustments`. Putting sharpening inside it would
  not have been last at all.
* **`_adjust_for_area` renders through `adjust_image` too**, so a parameter
  there would have given every area layer its own sharpening pass — against the
  Non-Goal above.

Sitting after both render paths have returned also makes CPU/GPU parity trivial:
there is exactly one numpy call and neither path can diverge from it. The three
`sharpen_*` keys therefore never enter the `adjust_image` signature, so the
positional-call hazard that constrains `balance_*` and `cineon_log` does not
arise here.

### Radius scaling — "preview simulates the full-res look"

A 1 px halo at 6000 px is 0.18 px at 1080 px, i.e. invisible. A faithful
miniature of the export would therefore show the user nothing, which defeats the
feature's purpose. So the preview deliberately **exaggerates**:

```
r_eff = clamp(radius_px * sharpen_scale, radius_px, R_MAX)
R_MAX = 8.0 px
```

**`sharpen_scale` defaults to 1.0 and is passed explicitly by the one caller
that wants exaggeration.** Deriving it inside `apply_adjustments` from
`original_full_size / buffer_long_side` looks tempting and is **wrong**: a
cropped or sliced export hands over a buffer whose long side is the *crop* at
native resolution, so that ratio reads > 1 and would over-sharpen the export.
Making 1.0 the default means every path is exact unless it opts in.

* **Export** and **zoom tiles** — decoded at the resolution they need, so they
  are already native for their display scale. `scale = 1` ⇒ `r_eff =
  radius_px`. Exact, and they agree with each other.
* **`update_thumbnail_and_preview`** — the only caller that passes a scale. It
  renders `resized_raw` (1080 max side) and knows `original_full_size`, so it
  passes `full_long / 1080`. For a 6000 px frame that is 5.55 ⇒ `r_eff =
  5.55 px`, putting the halo at roughly the same *screen* size it would occupy
  when inspecting the full-res file at 100 %.

This is a preview **of intent, not of pixels** — the chosen trade-off. `R_MAX`
exists so a high Radius setting cannot produce a grotesque halo (3.0 px × 5.55
would be 16.6 px without it).

`skip_dust=True` **also skips sharpening.** That flag already means "this is a
detached sample patch, spatial stages are meaningless here" — it is what
`solve_neutral_*` passes while rendering a small patch in a closed loop, and a
sharpening halo would perturb the measured mean it is solving against. Sharpening
is spatial for exactly the same reason dust healing is.

### The filter

Luma-only additive high-pass, so chroma noise is never amplified:

```
luma  = 0.2126 R + 0.7152 G + 0.0722 B          (Rec.709, on the display result)
blur  = gaussian_blur(luma, sigma = r_eff / 2)
hp    = luma - blur
out   = rgb + A * hp * mask                      (same scalar added to R, G and B)
A     = amount / 100 * 1.5                       (0 .. 1.5)
```

Adding one scalar to all three channels is what makes it luma-only: hue and
saturation are untouched by construction, no RGB→luma→RGB round trip needed.

### Masking

An edge mask so smooth areas (sky, dense shadow) do not get their noise lifted:

```
g      = |gradient(blur)|                        (Scharr magnitude, on the blurred luma)
t      = (masking / 100) * g_p95                 (threshold, relative to this frame)
mask   = 1.0                       if masking == 0
       = smoothstep(0, t, g)       otherwise
```

Thresholding against the frame's own 95th-percentile gradient keeps the control
meaningful across wildly different subjects, rather than tying it to an absolute
level that means something different on every scan. `masking = 0` short-circuits
to `mask = 1` and skips the gradient entirely (the common case — no cost).

The gradient is taken on the **blurred** luma, not the raw luma, so sensor and
grain noise do not themselves register as edges.

Output is clipped to the 16-bit range at the end; sharpening overshoot is what
clipping is for.

## Integration Points

| Where | Change |
|---|---|
| `ccr_processor.py` | `apply_sharpening(img, amount, radius, masking, scale)` — pure, unit-testable, standalone. The `adjust_image` signatures are **untouched**. |
| `ccr_image.py` | `apply_adjustments` gains `sharpen_scale: float = 1.0` and calls `apply_sharpening` as its last act, after the area composite and B&W collapse; skipped when `skip_dust`. `update_thumbnail_and_preview` passes `full_long / preview_long`. |
| `sliders_panel.py` | `self.details_section = CollapsibleSection("Details")`; three `create_slider` calls; `ADJUSTMENT_KEYS` += the three keys (after `band_feather`); `SLIDER_DEFAULTS` += amount/radius 25. |
| `catalog.py` | `_restore_image`: inject `sharpen_* = 0` into a non-empty restored dict that has no `sharpen_amount`. |
| `sliders_panel.SYNC_GROUPS` | a new `"details"` group carrying the three keys. Every adjustment key must belong to exactly one group (`test_groups_still_partition_adjustment_keys` enforces it), and sharpening gets its own rather than riding `tone`: it is a per-capture judgement — focus, grain, how far the frame was downscaled — that a user syncs independently of colour. |

## Edge Cases

* **Amount 0** — the whole stage short-circuits and returns the input untouched,
  so an unsharpened render costs nothing.
* **Radius 0** — likewise a no-op (`r_eff` of 0 has no blur to subtract).
* **Monochrome renders** — the luma of three identical channels is that value, so
  the same scalar is added back to all three and the frame stays neutral.
* **Solve-neutral sample patches** — skipped via `skip_dust` (see Processing).
* **Buffers smaller than the radius** — `gaussian_blur` needs a kernel that fits;
  guard with `r_eff <= 0 or min(h, w) < 4` ⇒ return unchanged.
* **Windowed working-space bases** — irrelevant: sharpening runs after the window
  clamp, on display-referred data.

## Test Plan

`tests/test_sharpening.py`:

1. **Pure filter** — a step edge gains overshoot on both sides; a flat field is
   returned **unchanged** (no high-pass energy to add); amount 0 and radius 0 are
   each byte-identical to the input.
2. **Luma-only** — a saturated patch keeps its hue and saturation: the R−G and
   G−B differences are preserved where the high-pass is non-zero.
3. **Radius scaling** — `scale=1` gives the native radius; a 5.55× scale widens
   the halo; `R_MAX` clamps a large radius × large scale.
4. **Masking** — at `masking=100`, a flat region's change is ~0 while a strong
   edge still sharpens; at `masking=0` both change.
5. **CPU/GPU parity** — `adjust_image` and `adjust_image_opencl` agree within the
   existing tolerance with sharpening active (skipped when no OpenCL).
6. **Defaults and backward compatibility** — `SLIDER_DEFAULTS` gives 25/25/0; a
   catalog dict **without** `sharpen_amount` restores to 0 (old scans unchanged);
   a dict **with** it round-trips its value.
7. **Positional safety** — `adjust_image` called positionally with the
   pre-sharpening argument list still works (the new params are last).

## Decisions

* **Visible default, not hidden.** 25 matches the Lightroom raw convention and,
  more importantly, shows on the slider so the user knows it is there.
* **Preview exaggerates on purpose.** The alternative (strict WYSIWYG) makes
  sharpening invisible at fit-to-window, which defeats "let me see the grain".
  100 % zoom and export remain exact, so nothing is misrepresented where it
  counts.
* **Luma-only, no option.** On an inverted negative the dense regions are
  stretched hardest and carry the most chroma noise; a chroma-sharpening option
  would exist only to make scans worse.
* **Old catalogs render unchanged.** A default that silently re-renders historical
  work is a breaking change, however good the default.

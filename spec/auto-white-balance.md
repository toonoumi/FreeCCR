# Spec: Auto White Balance (AWB)

Status: REFINED v2 (open questions resolved — ready to implement)
Owner: FreeCCR
Feature branch: `feature/awb`
Related: [`spec/working-space-white-balance.md`](working-space-white-balance.md) (the
flat WB gain model this inverts), [`spec/auto-gain.md`](auto-gain.md) (the Settings
toggle + post-conversion hook pattern), `spec/settings-page.md`.

## 0. Decisions (locked)

- **Q1 — Button label**: "AWB"; the eyedropper's label is shortened
  "Auto WB Picker" → **"WB Picker"** ("Auto" now belongs to the new button; four
  buttons `WB Picker | AWB | Crop | Slice` share the row at equal stretch).
- **Q2 — Auto-AWB default OFF.** A newly automatic behavior must not change
  existing users' conversions; the algorithm dropdown defaults to Gray World.
- **Q3 — Crop-aware (revised per user).** With a crop set, only the kept region
  drives the estimate: `compute_awb_temp_tint` passes the base through
  `apply_crop_to_image(base, crop_rect, crop_angle)` (normalized rect in
  resized_raw space; None → unchanged, so uncropped images are the full frame).
  A rotated crop's black corner fill de-windows below `AWB_LO` and is discarded
  by the in-bound mask — no special-casing. The content-fraction gate is
  relative to the cropped pixel count, so small crops still estimate.
- **Q4 — Active-image access**: `sliders_panel` already imports `ccr_backend`
  and reads the active image via `ccr_backend.get_image_by_index(self.current_idx)`
  throughout — the button handler does the same. No routing through
  `image_preview`.
- **Q5 — One backend choke point.** A new `ccr_backend.maybe_auto_awb(image_obj)`
  carries the toggle + already-saved guard; it is called from the **three** fresh-
  conversion sites (§4.5). `convert_negative_by_index` transitively covers the
  single ref-mode convert (`image_preview.convert_ccr`), `convert_all_images`,
  and `auto_frame_all_images`.
- **Slider re-sync is automatic**: every conversion path ends in
  `image_preview.update_preview(idx)`, which calls
  `sliders_panel.set_current_idx(idx)` (`image_preview.py:1114`) →
  `_load_active_layer` re-reads the settings dict — the AWB-written temp/tint
  appear on the sliders with no new wiring.
- **Windowing is self-consistent**: the converters set `_ws_windowed` before
  returning (`ccr_processor.py:1404` bwpoint = windowed when enabled, `:1573`
  reference = full-range), so the hook and the button always read the correct
  flag for the base in `resized_raw`.
- **Thread-safety**: `auto_frame_all_images` calls `convert_negative_by_index`
  from `ThreadPoolExecutor` workers; the hook is pure numpy + a dict write on
  that image's own settings (no Qt) — safe.

## 1. Summary

Add a fully automatic white balance:

1. An **"AWB" button** next to the existing WB eyedropper ("Auto WB Picker") in the
   sliders panel. One click — no grey-spot picking — estimates the neutral color of
   the whole converted frame with a classical, learning-free illuminant-estimation
   algorithm and sets the **Temperature / Tint sliders** to neutralize it. The
   result is visible on the sliders, undoable, and persisted exactly like an
   eyedropper pick.
2. A **Settings → General** checkbox — **"Auto white balance after conversion"** —
   that applies the same estimate automatically when a negative is converted,
   **only if the image has no temperature/tint already saved** (both at 0).
3. A **Settings → General dropdown** to choose the AWB **algorithm**: Gray World,
   White Patch, Shades of Gray, Gray Edge. The button and the auto-hook both use
   the selected algorithm.

This is deliberately NOT the abandoned deep-learning AWB (PR #52, `net_awb.onnx`,
judged not useful on film positives). These are transparent statistics on the
converted base, and the result lands on the sliders where the user can see and
correct it.

## 2. Goals / Non-goals

### Goals
- "AWB" button in the same button row as the WB picker, enabled under the same
  conditions (converted image active).
- The estimate rides the **existing picker math**: algorithm → estimated neutral
  RGB → `compute_neutral_temp_tint()` → `on_wb_sampled()`. AWB behaves exactly as
  if the user had clicked a perfectly chosen grey spot: sliders move, hint shows
  the values, undo burst, catalog persistence — all for free.
- **Idempotent**: the estimate is computed from the converted *base*
  (`resized_raw`), not the adjusted preview, so clicking AWB twice yields the same
  slider values.
- Settings → General: "Auto white balance after conversion" checkbox
  (default **OFF**) + "AWB algorithm" dropdown (default **Gray World**), both
  QSettings-persisted, staged-and-applied-on-Done like the other General toggles.
- The post-conversion hook fires only on **fresh conversions**
  (`convert_negative_by_index`, the B/W-point batch), never on replay/reconvert
  paths, and only when `temperature == 0 and tint == 0` in the image's saved
  adjustments.
- Pure-numpy core (`src/core/awb.py`), unit-testable headless.

### Non-goals
- No neural / learned estimator (explicitly rejected previously).
- No new WB math: the flat per-channel gain model and its inverse
  (`_white_balance_gains` / `compute_neutral_temp_tint`) are unchanged.
- No per-channel gain channel separate from the sliders (unlike Auto Gain's
  invisible offset — AWB *must* be visible on the sliders per the requirement).
- No AWB for un-converted negatives or positive-mode identity frames beyond the
  existing picker gate (button shares the picker's enable state).
- No live "AWB mode" that re-estimates on every render (it writes sliders once).

## 3. Current behaviour (as-is)

- The WB eyedropper (`sliders_panel.py:493` "Auto WB Picker") samples a 7×7 patch
  of `resized_raw` (`image_preview._sample_wb_point`, `image_preview.py:304`),
  de-windows it when the base is working-space windowed, and calls
  `compute_neutral_temp_tint(r, g, b, tint_balance_factor)`
  (`ccr_processor.py:2261`) — the exact inverse of the flat WB gains
  (`_white_balance_gains`, `ccr_processor.py:1012`):
  - temperature: `s = (slider/100)·0.40` → `R·(1+s)`, `B·(1−s)`
  - tint: `t = tanh(slider·0.02)·0.26·balance` → `G·(1−t)`, `R·(1+0.3t)`, `B·(1+0.3t)`
- The result is applied by `sliders_panel.on_wb_sampled(temp, tint)`
  (`sliders_panel.py:1527`): blockSignals slider set + label update +
  `on_slider_changed()` (stores into `adjustment_settings`, reprocesses, hint) +
  `_settle_preview()` (renders and shows the result now — see §4.4).
- `compute_neutral_temp_tint` is **scale-invariant** (it only uses channel
  ratios), so any positive-scaled RGB triple works as input.
- Auto Gain (`spec/auto-gain.md`) established: backend flag + QSettings key +
  Settings→General checkbox staged via `_init_toggles`/`_apply_pending` +
  `main_window.on_*_toggled` handler.
- Temperature/tint are stored in `CCRImage.adjustment_settings` under
  `"temperature"` / `"tint"`; missing keys mean 0. `temperature_base` is zeroed at
  conversion (`ccr_processor.py:1403`) and tint has no base channel, so
  `ci.get("temperature", 0) == 0 and ci.get("tint", 0) == 0` is exactly
  "no temp/tint saved".

## 4. Design

### 4.1 Estimation domain

All algorithms run on the converted base `CCRImage.resized_raw` (16-bit RGB,
~1080 px long side), converted to float and **de-windowed** when
`_ws_windowed` (mirroring `compute_auto_gain_offset`, `ccr_processor.py:1181`):

```
d = base.astype(float32)
d = (d - WS_B) * _WS_INV_WIDTH        if ws_windowed else d / 65535.0
```

**Valid-pixel mask — midtones only.** An uncropped film scan *always* contains
pure black and pure white that is not scene content: the holder masks the film
to pure black in the scan (→ clipped white once inverted) and the clear film
base / sprocket holes are the scan's maximum (→ crushed black once inverted).
Neither carries a usable cast, both are large, and they are noisy rather than
exactly 1.0/0.0 — so a wide gate lets them through and they hijack the
estimator (measured on a synthetic uncropped frame: `white_patch` returned a
perfectly neutral 1.000/1.000, i.e. it balanced on the holder; `shades_of_gray`
lost a third of the cast). AWB therefore reads the **midtones plus a slice of
the shadows and highlights**, via two independent rejections:

```
lum   = 0.299·R + 0.587·G + 0.114·B          # Rec.601, as compute_auto_gain_offset
valid = all_channels(d >= AWB_LO) & all_channels(d <= AWB_HI)   # 1. ratios are real
        & (lum >= AWB_TONE_LO) & (lum <= AWB_TONE_HI)           # 2. tonal region

AWB_LO      = 0.06   # per-channel floor: clear-film black / crushed channel
AWB_HI      = 0.94   # per-channel ceiling: holder white / blown channel
AWB_TONE_LO = 0.15   # luminance band: midtones + a slice of the shadows...
AWB_TONE_HI = 0.85   # ...and a slice of the highlights
```

The per-channel gate guarantees no channel is crushed or blown, so the pixel's
ratios mean something; the luminance band picks the tonal region the WB decision
is made from. `AWB_LO < AWB_TONE_LO < AWB_TONE_HI < AWB_HI`, so the band is the
binding limit on neutral content and the gate additionally catches single blown
channels in saturated colors.

This applies to **every** algorithm, `white_patch` included: it takes the
brightest *retained* pixel, i.e. the brightest real scene highlight rather than
the film surround.

If `valid.sum() < MIN_CONTENT_FRACTION · N` (reuse 0.005), the estimate is
`None` → the button shows a hint ("AWB: not enough usable image content") and the
auto-hook is a silent no-op.

### 4.2 Algorithms (all return an estimated neutral RGB triple)

Each algorithm estimates the color that *should* be neutral — the illuminant/cast
color. That triple is exactly what a perfect grey-spot pick would sample, so it
feeds `compute_neutral_temp_tint` unchanged. Working set `V` = valid pixels of
`d` (float, linear working domain).

| id | label | estimate per channel c |
|---|---|---|
| `gray_world` | Gray World | `mean(V_c)` |
| `white_patch` | White Patch | `percentile(V_c, 99)` |
| `shades_of_gray` | Shades of Gray | `mean(V_c ** 6) ** (1/6)` (Minkowski p=6) |
| `gray_edge` | Gray Edge | `mean(|∇(G_σ * d_c)| ** 6) ** (1/6)` over valid, p=6, σ=1 |

Notes:
- `gray_world` assumes the scene averages to grey — the classical default, robust
  on varied film scenes.
- `white_patch` is the robust max-RGB variant (99th percentile instead of max, on
  already near-clip-masked data).
- `shades_of_gray` (Finlayson & Trezzi) generalizes both; p=6 is the literature
  default.
- `gray_edge` (van de Weijer et al.): Gaussian-smooth each channel (σ=1), take the
  gradient magnitude `sqrt(gx²+gy²)` (numpy `np.gradient`), then the Minkowski
  p=6 mean over pixels whose source pixel is valid. Uses `cv2.GaussianBlur` if
  available (cv2 is already a dependency), else a small numpy separable blur.
  The mask is **eroded** by a 5×5 kernel first: the blur+gradient stencil reaches
  ~2 px, so a sample merely *next to* a rejected pixel still carries that pixel's
  edge — and the holder border is the strongest edge in an uncropped scan.
- Degenerate guard: if any channel estimate ≤ `AWB_EPS` (1e-6), return `None`.

### 4.3 New module `src/core/awb.py`

```python
AWB_ALGORITHMS = [("gray_world", "Gray World"),
                  ("white_patch", "White Patch"),
                  ("shades_of_gray", "Shades of Gray"),
                  ("gray_edge", "Gray Edge")]        # (id, UI label), ordered
AWB_DEFAULT = "gray_world"

def estimate_neutral_rgb(base_u16, ws_windowed: bool, algorithm: str
                         ) -> tuple | None:
    """De-window, mask, run the selected estimator; None if not enough content.
    Unknown algorithm ids fall back to gray_world (forward-compat settings)."""

def compute_awb_temp_tint(ccr_image) -> tuple | None:
    """apply_crop_to_image(resized_raw, crop_rect, crop_angle) →
    estimate_neutral_rgb(cropped, ccr_image._ws_windowed,
    ccr_backend.awb_algorithm) → compute_neutral_temp_tint(r, g, b,
    ccr_image.tint_balance_factor). Returns (temp:int, tint:int) or None."""
```

`compute_awb_temp_tint` imports `ccr_backend` lazily (deferred import, same
pattern as `ccr_image.py`) to read the selected algorithm.

### 4.4 The AWB button

- `self.auto_wb_btn = QPushButton("AWB")` in `sliders_panel`, inserted in
  `wb_crop_row` immediately after `wb_picker_btn` (`sliders_panel.py:516-520`),
  `setFixedHeight(theme.CONTROL_H)`, stretch 1. The picker's label is shortened
  from "Auto WB Picker" to **"WB Picker"** so the four buttons
  (`WB Picker | AWB | Crop | Slice`) fit; "Auto" now unambiguously belongs to the
  new button. Tooltip: "Automatic white balance — estimates the neutral color
  from the whole image and sets Temperature and Tint."
- Enable gate: same line as the picker (`sliders_panel.py:867`).
- Handler `_on_auto_wb()`:
  ```python
  img = ccr_backend.get_image_by_index(self.current_idx)
  if img is None or not img.converted:
      return
  res = compute_awb_temp_tint(img)
  if res is None:
      self.set_temporary_hint("AWB: not enough usable image content.", 5000)
      return
  self.on_wb_sampled(*res)
  ```
  Everything downstream (undo burst, slider set, store, reprocess, hint) is the
  existing `on_wb_sampled` path.
- **The result must appear immediately.** `on_slider_changed()` only *queues* a
  debounced (150 ms) reprocess and paints the currently cached
  `resized_preview` — fine for a slider drag, where the next tick redraws, but a
  one-shot click would leave the canvas on the pre-AWB render until the user
  touched something else. `on_wb_sampled` therefore ends with
  `_settle_preview()`: cancel the pending reprocess, `update_thumbnail_and_preview()`,
  *then* `update_preview()` + thumbnail refresh. `_settle_preview` is the shared
  discrete-edit settle, factored out of `_on_curve_edit_finished`
  (`sliders_panel.py`), and covers the WB eyedropper too — same one-shot shape.

### 4.5 The post-conversion hook

One backend method carries the whole policy:

```python
def maybe_auto_awb(self, image_obj):            # ccr_backend
    """One-shot AWB at conversion: writes temperature/tint into the image's
    whole-image settings when the toggle is on and neither is already set."""
    if not self.auto_awb or not image_obj.converted:
        return
    ci = image_obj.adjustment_settings
    if ci.get("temperature", 0) or ci.get("tint", 0):
        return                                   # already saved → never clobber
    res = compute_awb_temp_tint(image_obj)       # from core.awb (lazy import)
    if res is not None and any(res):
        ci["temperature"], ci["tint"] = res
```

Called from the **three fresh-conversion sites**, always after the
`conversion_inputs` snapshot and **before** `update_thumbnail_and_preview()` so
the first render already includes the WB:

1. `ccr_backend.convert_negative_by_index` (`ccr_backend.py:798`) — covers the
   single ref-mode convert (`image_preview.convert_ccr`), `convert_all_images`,
   and `auto_frame_all_images` (worker threads — hook is thread-safe, §0).
2. `ccr_backend.apply_bwpoint_to_all_images` (`ccr_backend.py:1929`) — the
   B/W-point batch loop.
3. `sliders_panel._on_convert_current_bwpoint` (`sliders_panel.py:1740`) — the
   single B/W-point convert (GUI-side; calls
   `ccr_backend.maybe_auto_awb(img)`).

- Backend-side (no GUI dependency) so batch conversion just works; the render
  that already follows conversion picks the values up, and the panel re-sync is
  automatic (§0).
- **Not** fired from `_reconvert_in_place` / replay / reload paths: reconversion
  preserves the user's adjustments, and a global-mode reconvert must not
  surprise-move sliders. (For images whose temp/tint were AWB-written earlier,
  the guard also skips them — the values are now "saved".)
- The guard condition is the user requirement verbatim: apply only when the image
  does **not** have a temp and tint already saved. `0/absent` = not saved; any
  non-zero value (user- or AWB-written, catalog-restored) = saved → skip. An
  all-zero estimate is not written (no pointless dict pollution).

### 4.6 Settings → General

New `QGroupBox("White balance")` in `_build_general_page`
(`settings_dialog.py:93`), after the "Exposure" group:

- `QCheckBox("Auto white balance after conversion")` — muted help: "When a
  negative is converted, automatically estimate and set Temperature/Tint —
  only for images with no white balance already set."
- Row: `QLabel("Algorithm")` + `QComboBox` with the four labels (ids as
  `userData`), muted help naming the default.

Wiring (the Auto Gain 4-point pattern):
- Backend flags (`ccr_backend.py` flag block): `self.auto_awb: bool = False`,
  `self.awb_algorithm: str = "gray_world"`.
- QSettings: `adjust/auto_awb` (bool, default False), `adjust/awb_algorithm`
  (string id, default `"gray_world"`), restored in `main_window.__init__`
  alongside `adjust/auto_gain`. Unknown stored ids fall back to the default.
- `main_window.on_auto_awb_toggled(checked)` / `on_awb_algorithm_changed(algo)`:
  set flag + persist. **No rerender/reprocess** — these affect only future
  conversions and button presses, not existing renders.
- Dialog staging: seed both in `_init_toggles`; commit in `_apply_pending` only
  on change (checkbox → toggled handler; combo → algorithm handler).

## 5. Data model

- No new per-image fields, no catalog change: AWB writes the existing
  `"temperature"` / `"tint"` keys in `adjustment_settings`.
- Two new global backend flags (`auto_awb`, `awb_algorithm`) + two QSettings keys
  (`adjust/auto_awb`, `adjust/awb_algorithm`).

## 6. Integration points

- **new** `src/core/awb.py` — algorithms + `compute_awb_temp_tint` (pure numpy;
  imports `compute_neutral_temp_tint`, `WS_B`, `_WS_INV_WIDTH`,
  `MIN_CONTENT_FRACTION` from `ccr_processor`; `ccr_backend` lazily).
- `src/widgets/sliders_panel.py` — button creation (~:493), row insert
  (~:516-520), picker label "WB Picker", click connect (~:719), enable gate
  (~:867), `_on_auto_wb` handler near `_on_pick_neutral_point` (~:1492),
  `maybe_auto_awb` call in `_on_convert_current_bwpoint` (~:1774).
- `src/core/ccr_backend.py` — flags in the flag block (~:67);
  `maybe_auto_awb()`; calls in `convert_negative_by_index` (after ~:818) and
  `apply_bwpoint_to_all_images` (after ~:1968).
- `src/ui/main_window.py` — QSettings restore (~:204), two handlers near
  `on_auto_gain_toggled` (~:687).
- `src/widgets/settings_dialog.py` — General page group (~:93), `_init_toggles`
  (~:264), `_apply_pending` (~:282).

## 7. Edge cases

- **Not enough valid pixels** (< 0.5%: nearly all holder/headroom/blown):
  estimate `None` → button hints, auto-hook no-ops. The image is left untouched.
- **Monochrome/neutral frame**: all channel stats equal → temp=0, tint=0 —
  harmless no-op write.
- **Extreme cast beyond slider range**: `compute_neutral_temp_tint` already
  clamps to ±100; the hint shows the (clamped) values, user sees the sliders
  pegged — honest behavior.
- **Auto-AWB + Auto Gain together**: independent — Auto Gain is a uniform gain
  offset (never touches sliders), AWB is per-channel via the sliders. Order
  irrelevant (both multiplicative, and AWB estimates from the base, not the
  gained render).
- **User undoes an auto-AWB'd conversion**: conversion undo restores the
  pre-conversion state snapshot; the AWB-written keys are inside
  `adjustment_settings`, captured by the same `to_state` deep-copy — verify the
  conversion undo snapshot ordering at implementation.
- **Batch conversion cost**: gray_world/white_patch/shades ≈ a few ms per ~1080px
  frame; gray_edge ~10-20 ms (blur + gradient). Negligible next to the
  conversion itself.
- **Stored algorithm id unknown** (future rename/downgrade): fall back to
  `gray_world` silently.
- **Cropped image**: only the kept region is sampled (§0-Q3). A rotated crop's
  out-of-source black fill is masked out by `AWB_LO`; a degenerate/absent rect
  falls back to the full frame (`apply_crop_to_image` returns the input).
- **Area layers**: AWB writes the whole-image layer (`adjustment_settings`),
  never an area layer — the button uses `on_wb_sampled`, which already targets
  the whole-image sliders; the hook writes `adjustment_settings` directly.
  (If an area layer is active when the button is clicked, `on_wb_sampled`'s
  behavior is the eyedropper's — identical semantics, no new rules.)

## 8. Test plan (`tests/test_awb.py`, pure numpy + GUI-light)

- **Gray-world recovery**: synthetic neutral-scene base × illuminant
  (1.25, 1.0, 0.8); estimate ∝ illuminant; applying the returned temp/tint
  through `_white_balance_gains` equalizes the valid-pixel channel means within
  slider-int quantization (~1%).
- **White patch**: base with a near-white patch tinted by the cast → estimate
  matches the patch color, not the scene mean.
- **Shades of Gray**: on a uniform image equals gray_world; on a skewed image
  lies between gray_world and white_patch.
- **Gray edge**: cast applied to an edge-rich synthetic → recovers the cast.
- **Mask**: headroom pixels (d > 0.98) and sub-black (d < 0.02) excluded —
  adding a blown region does not change the gray_world estimate.
- **Insufficient content**: 99.9% out-of-bounds → `None`.
- **De-window correctness**: same scene encoded windowed vs plain, ws flag set
  respectively → same slider values.
- **Idempotence**: estimate from `resized_raw` is independent of current
  temp/tint settings.
- **Hook guard**: fake image with `{"temperature": 5}` (or `{"tint": -3}`) →
  hook leaves settings unchanged; with both 0/absent → keys written; with
  `auto_awb=False` → untouched.
- **Crop-aware**: cast-A content inside the crop rect, cast-B junk outside →
  the estimate matches cast A (and differs from the uncropped estimate); a
  rotated crop's black fill does not skew the estimate.
- **Unknown algorithm id** → gray_world result.
- **Settings round-trip** (GUI-light or mocked): backend defaults
  (False, "gray_world"); toggling handlers persist and update flags.

## 9. Open questions — RESOLVED (see §0 for the locked answers)

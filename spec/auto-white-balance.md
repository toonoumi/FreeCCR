# Spec: Auto White Balance (AWB)

Status: REFINED v1
Owner: FreeCCR
Feature branch: `feature/auto-white-balance`

## 1. Summary

Add a one-click **Auto White Balance** to the right adjustment panel: a checkbox
directly above the Temperature slider. When ticked, FreeCCR estimates the
image's colour cast with a learning-based model (`net_awb.onnx`) and removes it
by applying a per-channel gain to the converted positive **right before all the
custom (slider) adjustments**.

AWB is a separate, non-destructive correction layer: it does **not** move the
Temperature/Tint (or any other) slider positions. The user keeps full manual
control on top of the auto correction.

## 2. Goals / Non-goals

### Goals
- A per-image **"Auto WB"** checkbox in `SlidersPanel`, laid out as a row
  **immediately above the Temperature slider** (same column geometry as the
  Color Profile row).
- When enabled, an estimated per-channel RGB gain is applied to the converted
  positive **before** temperature/tint and every other custom adjustment.
- AWB **never changes any slider value**. Temperature, Tint, etc. stay exactly
  where the user left them.
- The correction is **resolution-independent**: identical look across the 1080px
  preview, the hi-res zoom detail, and the full-res export. Gains are computed
  once (on the whole-image preview) and reused everywhere.
- Per-image state; persisted in the catalog; participates in Undo/Redo and
  "Sync to All".
- Graceful degradation: if `onnxruntime` or the model is unavailable, the
  checkbox is disabled with a tooltip explaining why; nothing else breaks.

### Non-goals
- No eyedropper / neutral-point picking (that already exists as "Auto WB
  Picker", which *does* move the sliders — AWB is the complementary, hands-off
  path).
- No iterative refinement loop (a single pass is enough here).
- No GPU/OpenCL kernel changes. The gain is a trivial per-channel multiply done
  in numpy before the existing adjust pass.
- No new "AWB strength" slider in v1 (full correction only). May come later.
- AWB is a no-op for the Black & White colour profile (a grey image has no cast
  to correct).

## 3. Background — the model

`net_awb.onnx` is an ONNX white-balance network: it takes a display-referred RGB
image in `[0,1]` (NCHW) and returns a **white-balanced** version of the same
image. A per-channel gain is derived from the ratio of input vs. corrected
channel means.

FreeCCR works in a near-linear 16-bit RGB space, so we adapt the gain
derivation (see §6) but reuse the model and the "ratio of means" idea verbatim.

## 4. UX / Interaction

- **Control**: a `QCheckBox` labelled "Auto WB" in a row that matches the
  Color Profile / slider row geometry (right-aligned label column + control),
  inserted between the Color Profile row and the Temperature slider.
- **Tooltip**: "Automatically remove the colour cast. Runs before the manual
  sliders and does not change their positions."
- **Toggling on**:
  - Pushes a single undo snapshot (discrete action, not merged into a slider
    drag burst).
  - Sets `img.awb_enabled = True`, invalidates any cached gains, reprocesses the
    preview + thumbnail, refreshes the main preview.
  - First enable triggers the (cached) ONNX inference once; subsequent
    preview/zoom/export reuse the cached gains with no extra inference.
- **Toggling off**: reverts to the un-balanced positive (gains no longer
  applied); sliders unaffected.
- **Gating**: enabled only when the per-image sliders are enabled (i.e. a
  converted positive exists), via `set_sliders_enabled`. Additionally disabled
  (greyed, with tooltip) when AWB is unavailable (`onnxruntime`/model missing).
- **Reset**: the panel "Reset" button clears AWB (`awb_enabled = False`) when
  resetting the Whole Image layer, alongside the existing slider/crop reset, and
  unticks the checkbox.
- **Compare**: the existing press-and-hold Compare shows the original; AWB is
  part of the "adjusted" state, so it is included in the comparison naturally.

## 5. Data model

New per-image state on `CCRImage` (mirrors how `color_profile` is declared,
persisted, and undone):

| Field | Type | Persisted? | Synced? | Meaning |
|-------|------|-----------|---------|---------|
| `awb_enabled` | `bool` (default `False`) | yes (catalog) | yes (enable flag only) | user toggle |
| `awb_gains` | `tuple[float,float,float] \| None` | **no** (cache) | no | last computed R/G/B multipliers |

`awb_gains` is a derived cache (depends on the image's pixels + conversion), so
it is never serialized or synced — only `awb_enabled` is. It is recomputed on
demand and invalidated whenever the conversion changes (convert / reload /
unconvert) or AWB is toggled.

### Persistence / lifecycle touch-points (must all be updated)
- `CCRImage.__init__` — declare `self.awb_enabled`, `self.awb_gains = None`.
- `CCRImage.serialize_state` / `deserialize_state` (and `catalog.serialize_image`
  / `_restore_image`) — save/restore `awb_enabled` only; `awb_gains` resets to
  `None` on load.
- `CCRImage.push_undo_state` snapshot + its restore — include `awb_enabled`;
  reset `awb_gains` to `None` on restore.
- Conversion / reload paths that reset `conversion_inputs` — also set
  `awb_gains = None`.
- `ccr_backend` per-image copy/duplicate/template sites that copy
  `adjustment_settings`/`color_profile` — also carry `awb_enabled`.

## 6. Processing / math

### 6.1 Where AWB is applied
A single insertion point in `CCRImage.apply_adjustments` (`ccr_image.py`),
**after** dust removal and **before** the slider/`adjust_image_opencl` call and
before the no-op early-return guard:

```
image = self._apply_dust_removal(image)
image = self._apply_awb(image, awb_enabled, awb_gains)   # NEW
s = self.adjustment_settings if settings is None else settings
...
```

Because preview, hi-res zoom, and all export paths route the converted positive
through `apply_adjustments` (verified: `ccr_image.py:678`,
`image_preview.py:3763`, `ccr_processor.py:843/1189/1284`), this one site covers
every resolution. AWB is **not** applied in `_adjust_for_area` — area layers
composite on top of the already-balanced whole-image base.

`_apply_awb` is a per-channel multiply on the uint16 RGB array:
```
out[...,c] = clip(image[...,c] * gain[c], 0, 65535)
```
applied only when `awb_enabled and gains is not None and profile == "color"`.

### 6.2 Computing the gains
Computed once, on the converted positive at preview resolution **cropped to the
user's crop region** (in `update_thumbnail_and_preview`, where the 1080px
positive is in hand), then cached on `self.awb_gains`. The cache is keyed
(`_awb_cache_key`) on the converted positive's identity **and** the crop, so a
re-conversion or a crop change invalidates it. Zoom/export never recompute — they
reuse the cache
(passed through `apply_adjustments` as override params for thread-safety, like
`areas_override`). The model internally downsizes, so input resolution only
affects speed, not the result.

Algorithm (validated against the real model — recovers `1/illuminant` within
~2% across warm/cool/green casts, and beats feeding linear data directly):

1. Take the converted positive (linear-ish, uint16 RGB) **restricted to the
   user's crop region** (`apply_crop_to_image`; whole image when uncropped) so
   the cast is estimated from the kept area only, then normalise to `[0,1]`.
2. **sRGB-encode** it (the net was trained on display-referred sRGB).
3. Resize so the long side ≤ `AWB_INFER_SIZE` (default 256), round both dims to a
   multiple of 16, to NCHW float32.
4. Run `net_awb.onnx` → corrected image in `[0,1]` (NCHW), clip to `[0,1]`.
5. **sRGB-decode** both the (resized) input and the corrected output back to
   linear.
6. Per-channel `gain[c] = mean(out_lin[...,c]) / max(mean(in_lin[...,c]), 1e-6)`.
7. **Luminance-normalise**: divide all gains by
   `0.299*g_R + 0.587*g_G + 0.114*g_B` so AWB shifts only colour, not overall
   exposure (keeps the user's Gain/Brightness meaningful).
8. Clamp each gain to `[AWB_GAIN_MIN, AWB_GAIN_MAX]` (default `[0.3, 3.0]`) to
   guard against pathological corrections.

If inference fails or the model/runtime is missing, return `None` (AWB silently
no-ops; the checkbox is disabled in that case anyway).

### 6.3 The model module — `src/core/awb.py`
A self-contained module that mirrors `src/core/dust_detect.py`:
- `MODEL_FILENAME = "net_awb.onnx"`; resolved via `resource_path` (bundled under
  `src/models/`), with `%APPDATA%/FreeCCR/models/net_awb.onnx` as a secondary
  location for parity with the dust pattern.
- `is_available()` / `availability_reason()` — late `import onnxruntime`,
  reported to the panel.
- `is_model_present()` — model file resolvable.
- `_get_session()` — thread-locked, cached `InferenceSession`
  (`CPUExecutionProvider`); `onnxruntime` imported **only inside functions**.
- `compute_gains(positive_rgb16) -> tuple|None` — the §6.2 pipeline (uses cv2 +
  numpy for resize/encode; no Pillow dependency).

`onnxruntime` is already a project dependency (used by dust detection) and is
already pre-loaded in `src/main.py` and bundled by `build_exe.bat`, so no
changes are needed there. The DLL/runtime plumbing is shared.

## 7. Integration points (file-by-file)

| File | Change |
|------|--------|
| `src/models/net_awb.onnx` | **New** — bundled model (~17.4 MB). |
| `src/core/awb.py` | **New** — model wrapper + `compute_gains` (mirrors `dust_detect.py`). |
| `src/core/ccr_image.py` | `awb_enabled`/`awb_gains` fields; `_apply_awb`; call it in `apply_adjustments` (after dust, before sliders) with override params; compute+cache gains in `update_thumbnail_and_preview`; serialize/deserialize/undo include `awb_enabled`; invalidate `awb_gains` on conversion reset. |
| `src/widgets/sliders_panel.py` | `_create_awb_row()`; insert above Temperature; `on_awb_toggled()` handler (mirror `on_color_profile_changed`); gate in `set_sliders_enabled`; reflect state in `set_current_idx`/`_load_active_layer`; clear on `on_reset_clicked`; add to `SYNC_GROUPS` + `_perform_sync_to_all`. |
| `src/core/ccr_backend.py` | Carry `awb_enabled` in per-image copy/duplicate/template sites that already copy `color_profile`. |
| `src/core/catalog.py` | Save/restore `awb_enabled`. |
| `build_exe.bat` / `Makefile` | `--include-data-dir=src/models=models` so the bundled `.onnx` ships in the exe. |
| `requirements.txt` | No change (`onnxruntime` already present). |

### Zoom worker / override params
`apply_adjustments` gains two optional override params, `awb_enabled_override`
and `awb_gains_override`, defaulting to the instance fields — exactly like the
existing `settings`/`areas_override` snapshot mechanism. The hi-res zoom worker
(`image_preview.py`) snapshots `img.awb_enabled` and `img.awb_gains` at request
time and passes them through, so a concurrent toggle can't tear a zoom render.

## 8. Test plan

Unit (pure / no Qt, follow `tests/test_dust_removal.py` style):
- `test_awb.py`:
  - `compute_gains` on a synthetic neutral image → gains ≈ `(1,1,1)`.
  - `compute_gains` on a synthetic warm-cast image → `g_R < 1 < g_B`; on a cool
    cast → `g_R > 1 > g_B` (cast is reduced). Luminance-normalised mean ≈ 1.
  - Gains are clamped within `[AWB_GAIN_MIN, AWB_GAIN_MAX]`.
  - `_apply_awb` with gains `(1,1,1)` is a no-op; with non-trivial gains it
    multiplies per channel and clips to `[0,65535]`, dtype preserved.
  - `apply_adjustments` with `awb_enabled=True` + identity sliders changes the
    image; with `awb_enabled=False` it does not — and **slider values are
    untouched** either way.
  - Availability: with `onnxruntime` forced absent (monkeypatch `sys.modules`),
    `is_available()` is `False`, `availability_reason()` is non-empty, and
    `compute_gains` returns `None` (no raise).
- Resolution independence: gains computed once are applied identically at two
  resolutions (apply the same cached gains to a downscaled vs full array → same
  per-channel ratio).
- Persistence/undo: `awb_enabled` round-trips through serialize/deserialize and
  through push/restore undo; `awb_gains` resets to `None` on restore.

Manual (in-app, via the `qt-testing`/`run` skills):
- Load a real negative (`example_raw/DSC07096.ARW`), convert, tick **Auto WB**:
  the cast visibly neutralises; Temperature/Tint sliders stay at their values.
- Zoom in → hi-res detail matches the preview's balance. Export → file matches.
- Toggle off → reverts. Undo/redo restores the toggle. Sync to All propagates
  the enable flag.
- B&W profile: ticking Auto WB has no visible effect (no-op), no error.

## 9. Open questions (resolved)
- *Distribution*: bundle the model in-repo (decided).
- *Gain normalisation*: luminance-preserving (colour-only) — decided & validated.
- *Apply point*: in `apply_adjustments`, before sliders — decided & validated as
  the single chokepoint covering preview/zoom/export.

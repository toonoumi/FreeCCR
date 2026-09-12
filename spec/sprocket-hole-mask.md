# Spec: Sprocket-Hole (Clear-Film) White Mask

Status: REFINED (open questions resolved — ready to implement)
Owner: FreeCCR
Feature branch: `feature/sprocket-hole-mask`
Related: `spec/auto-gain.md` (the reference "global display toggle read live in the
render pipeline, re-render on toggle" shape), `spec/settings-page.md` (the Settings
dialog this adds a toggle to), `spec/density-bwpoint-toggle.md` /
`spec/film-stock-slopes.md` (the B/W-point conversion this rides on).

## 0. Decisions (locked)

- **Reversal look.** On a 135 negative the punched sprocket holes and the clear
  rebate are *clearer than the film base*, so after inversion they clamp to pure
  **black**. Photographers replicating a slide/reversal presentation want those
  areas **white** instead. This feature paints every genuinely-clear-film region
  white as the last step before display/export, so no adjustment can tint them.
- **Detect from the RAW, not the positive.** In the inverted positive the holes
  are indistinguishable from crushed shadows and the film border (all sit at 0),
  so the mask MUST be derived from the pre-inversion scan, where "clearer than the
  sampled film base" is a clean, physically-meaningful test. See §4.1.
- **Only in B/W-point (black-point-set) mode.** The threshold is defined relative
  to the user's sampled black point (film base). Reference/auto conversions have
  no explicit base and positive mode has no negative — both skip the mask.
- **Global display overlay, computed at convert time, applied live.** Like Auto
  Gain / Gamma mode, the toggle is read live and toggling only **re-renders**
  (no re-conversion, no catalog write). The mask *alpha* is a pure function of
  (raw scan, black point) and is computed wherever the raw is in hand — cached on
  the image for the preview, recomputed at native res for zoom/export — so preview
  and export agree. See §4.2.
- **Default OFF.** It is a stylistic, non-standard presentation choice; the
  default conversion stays as-is.
- **Format-agnostic mechanism, "135" framing.** Nothing detects 135 vs 120. The
  mask fires only where the scan is clearer than the sampled base, which is
  physically only the holes / rebate / inter-frame gaps. 120 (no sprockets) simply
  gets its clear rebate border whitened, which is harmless and consistent.

## 1. Summary

A persisted **Settings → General → "White sprocket holes / clear film (reversal
look)"** toggle (default OFF). When on, every B/W-point-converted image gets a
mask built from the raw scan — pixels **brighter than the sampled black point plus
a padding**, morphologically cleaned (open to drop specks, close to fill small
gaps) and **feathered ~5–10 px** — and that mask is composited to **white**
(65535) as the final step before the pixels are shown and before they are written
on export. The result: sprocket holes / clear rebate read white like reversal
film, untouched by any slider, curve, area, or crop adjustment.

## 2. Goals / Non-goals

### Goals
- Settings → General checkbox (default OFF), staged-and-applied-on-Done like the
  other global toggles; toggling **re-renders** loaded images (no re-convert).
- A raw-derived, per-channel **"clearer than film base + padding"** binary mask,
  morphology-cleaned and **feathered 5–10 px** (at the 1080 preview; scaled with
  resolution so export/zoom match).
- Composite the mask to **pure white** as the **last** step of the preview,
  hi-res zoom, and export render paths — after all adjustments, dust, areas,
  curves, gain, gamma — so nothing downstream can affect the whitened regions.
- Preview / zoom / export produce the **same** mask (same threshold; geometric
  params scale with resolution), matching FreeCCR's "plan at preview, replay at
  native" contract (`spec/dust-removal.md`).
- Tunable thresholds via `FREECCR_SPROCKET_*` env knobs for field calibration on
  real rolls (mirrors the `FREECCR_*` convention), with sane baked defaults.

### Non-goals
- No format detection (135/120), no sprocket-hole *geometry* fitting; purely a
  clearer-than-base intensity mask.
- No user-facing padding/feather sliders in v1 (fixed constants + env override).
- No change to the conversion math, reference/auto conversion, positive mode, or
  the sampled-point picker.
- Not applied to un-converted scans, reference-frame conversions, positive-mode
  images, or trichrome-merge previews (they lack a sampled base — §7).
- Does not alter the histogram/scopes (they read the pre-mask adjusted preview;
  the mask is a cosmetic final composite — acceptable for v1, noted in §7).

## 3. Current behaviour (as-is)

B/W-point conversion (`ccr_normalize_with_bwpoint`, black point ± white point)
maps the sampled clear/film-base sample to output **0** and everything *clearer*
(the holes/rebate, which scan **brighter** than the base) clamps to 0 as well —
so in the positive they are pure black and indistinguishable from crushed shadow.

Three render paths reproduce a B/W-point conversion and then apply adjustments —
each is where the mask must be composited **last**:

1. **Preview / thumbnail** — `ccr_image.update_thumbnail_and_preview` runs
   `adjusted = apply_adjustments(resized_raw)` then downsizes to preview/thumb.
   `resized_raw` is the *converted* buffer (the raw is gone), produced earlier by
   `ccr_normalize_with_bwpoint(output_path=None)`.
2. **Hi-res zoom** — `HiResDetailWorker` calls `render_hires_base` (re-decodes the
   raw, replays via `apply_bwpoint_normalization`) → `apply_adjustments` → 8-bit.
3. **Export** — `ccr_normalize_with_bwpoint(output_path=…)`:
   `invert → apply_adjustments → crop → flips → rotation → write`.

The invert helpers consume a **float32 BGR** working image (`img_f`) and the
sampled `black_point_bgr` (BGR, HIGH scan values for the clear base). The mask is
computed in the *same* raw BGR space, so channel alignment is automatic; the
composite paints neutral white, so it is channel-order-agnostic.

Global display toggles (`auto_gain`, `gamma_luminance`) already demonstrate the
full pattern: a `ccr_backend` flag, QSettings restore in `MainWindow.__init__`, an
`on_*_toggled` handler that persists + `_rerender_all_for_global_mode(...)`, a
staged checkbox in `SettingsDialog`, and membership in
`image_preview._current_adj_sig()` so the hi-res cache invalidates on toggle.

## 4. Design

### 4.1 The mask (pure functions, in `ccr_processor.py`)

```python
# Baked defaults (env-overridable for field calibration)
GAP_FRAC     = 0.50   # a hole must be >= 50% of the way from base -> the
                      #   MEASURED clear-film level, per channel
MIN_CLEAR_D  = 0.10   # no channel clears the base by this density -> no clear
                      #   film in the frame -> mask nothing
CLEAR_ERODE_PX = 1.0  # erosion radius before taking the clear level @ 1080
MIN_AREA_PX  = 24.0   # speckle cutoff (connected-component area) @ 1080 long side
FEATHER_PX   = 1.0    # edge feather — anti-aliasing only          @ 1080 long side
REF_LONG     = 1080   # reference long side the px params are quoted at

def compute_sprocket_alpha(raw_bgr, black_point_bgr) -> Optional[np.ndarray]:
    """A white-mask alpha (uint8 H×W, 0..255) marking clear-film (sprocket /
    rebate) regions — pixels clearer than the sampled film base by a padding — or
    None if the black point is unset or no region qualifies. Holes are kept SHARP.

    `raw_bgr` is the float/uint working scan (BGR, same array the invert helpers
    consume); `black_point_bgr` is the sampled clear/film-base anchor (BGR, HIGH
    values). Resolution-independent threshold; area cutoff + feather scale with
    this buffer's long side so preview/zoom/export agree geometrically."""
    if black_point_bgr is None:
        return None
    d = raw_bgr astype float32                       # (H,W,3) BGR
    bp   = float32 array(black_point_bgr)            # (3,)
    scale = max(H,W) / REF_LONG
    clear = [max(erode(d[...,c], odd(CLEAR_ERODE_PX*scale))) for c in BGR]
    if max(log10(clear / bp)) < MIN_CLEAR_D: return None   # no clear film here
    thr  = bp + GAP_FRAC * maximum(clear - bp, 0)
    mask = all(d > thr[None,None,:], axis=2)         # AND across BGR -> clearer than base
    if not mask.any(): return None
    m = uint8(mask)*255
    scale = max(H,W) / REF_LONG
    # Speckle removal by connected-component AREA — NOT morphological open, which
    # would erode/round the hole corners. Keeps every component >= min_area sharp.
    min_area = max(1, round(MIN_AREA_PX * scale*scale))
    m = keep_components_with_area(m, >= min_area)     # via connectedComponentsWithStats
    if m is empty: return None
    # Fill INTERIOR holes only (dust/markings inside a hole) via a border
    # flood-fill imfill — NOT morphological close, so the outer edge is untouched.
    m = fill_interior_holes(m)                        # pad 1px bg, floodFill(0,0), OR back
    f = FEATHER_PX * scale
    if f >= 0.5: m = GaussianBlur(m, ksize=odd(2*f), sigma=f/2)   # 1px anti-alias
    return m                                          # uint8 0..255
```

Why per-channel **AND** at half the **measured** base → clear gap: a punched hole
is clear film → the brightest level in **every** channel. Exposed scene content
(even a deep shadow) is *denser* than the zero-exposure base → at or below base,
so it sits near 0% of the gap while the holes sit at 100%; the midpoint gives
both sides the same margin. The AND biases toward **few false positives** (never
whiten real content).

**Why measured, not a fraction of the headroom to 65535** (history: shipped at
20% of `65535 − base` with a 2%-of-full-scale floor, briefly raised to 50%/5%,
then replaced). The decode does not reach 65535: RAW values are black-subtracted
and scaled by `65535/white_level`, so a 14-bit Sony saturates at **63486**. A
colour negative's base can sit a sliver below that in one channel — on the
maintainer's 5207 trichrome scans the base is B=62092 G=35455 R=43587 and the
holes clip at 63486 in all three, so blue's real gap is 1394, not 3444. At 50%
(or with the 5% floor) blue's threshold landed above the holes and **no hole was
masked**. Measured on four frames of that roll, the holes form one cluster at
100% of the measured gap (~34–36k px at 1080) and near-base content stays under
20%, with essentially nothing in between. The clear level is each channel's
maximum after a small erosion (clear film is the brightest thing in a negative
scan; the erosion drops hot pixels and noise specks). A channel with no gap
(base and holes both clipped) degenerates to "above base" and the separating
channels decide. `MIN_CLEAR_D` stops a frame with no clear film from treating
its brightest near-base patch (uneven illumination) as the clear level.

**Keeping the holes sharp.** Real 135 sprocket holes are crisp rounded rectangles;
an early build over-softened them with a morphological **open** (rounds corners) +
**close** (dilates/rounds the outer edge) + an 8 px Gaussian **feather** (a glow).
The cleanup is instead: (1) a **connected-component area filter** to drop tiny
noise specks — no erosion, so the true edge is preserved; (2) a **flood-fill
imfill** to fill only *interior* gaps (a speck of dust or a printed frame number
inside a hole) while leaving the outer boundary exactly where the threshold put
it — no dilation/rounding; (3) a **~1 px feather** for anti-aliasing only. All
three are cheap (linear in pixels) and thread-safe.

```python
def apply_sprocket_mask(rgb_u16, alpha_u8) -> np.ndarray:
    """Composite the clear-film regions to white (65535). alpha 0 = keep image,
    255 = full white, in-between = feathered blend. No-op when alpha is None."""
    if alpha_u8 is None: return rgb_u16
    a = float32(alpha_u8)[...,None] / 255.0
    return clip(rgb_u16*(1-a) + 65535.0*a, 0, 65535).astype(uint16)
```

Both are pure/thread-safe (run on pool + QThread workers). Env knobs read once via
a small `_sprocket_cfg()` helper (like `_ws_enabled`).

### 4.2 Wiring the three render paths

**Convert (`ccr_normalize_with_bwpoint`).** After `img_f`/`img` is loaded and
before it's freed, compute `alpha = compute_sprocket_alpha(img, black_point_bgr)`.
- `output_path is None` (preview convert): store `ccr_image.sprocket_alpha =
  alpha`. Do **not** composite here — the overlay is applied in
  `update_thumbnail_and_preview`, gated by the live toggle (so on/off is a pure
  re-render). Compute it **always** (regardless of the toggle) so turning the
  toggle on later needs no re-conversion; it is a few ms on a ≤1080 buffer.
- `output_path is not None` (export): after `apply_adjustments` and **before**
  crop/flips/rotation, `if ccr_backend.sprocket_mask_white: rgb_result =
  apply_sprocket_mask(rgb_result, alpha)`. Applying before the geometric block
  keeps it in the same un-rotated/un-cropped space as the preview overlay (WYSIWYG
  — the canvas item applies crop/rotation to the whitened preview); white survives
  the warp, and a crop that excludes the holes simply drops them (expected).

**Preview (`ccr_image.update_thumbnail_and_preview`).** After `display_img` is
chosen (the converted+adjusted uint16, same H×W as `resized_raw` and thus as
`sprocket_alpha`) and **before** the thumb/preview downsizes:
```python
from core.ccr_backend import ccr_backend   # deferred import (already used here)
if (ccr_backend.sprocket_mask_white and self.converted
        and not self._positive_mode_active()
        and getattr(self, "sprocket_alpha", None) is not None):
    display_img = apply_sprocket_mask(display_img, self.sprocket_alpha)
```

**Hi-res zoom (`render_hires_base` + `HiResDetailWorker`).** In `render_hires_base`
the `mode == "bw"` branch has the re-decoded raw `img` in hand: compute the alpha
there and return it alongside the base (change the return to `(base, alpha)`;
`alpha=None` for ref / ref_params / positive / unconverted). Thread it through:
- `HiResDetailWorker` receives/caches it (`self._sprocket_alpha`), and after
  `apply_adjustments` (and before 8-bit), composites it when
  `ccr_backend.sprocket_mask_white` (snapshot the flag at construction like
  `_positive_mode`).
- The hi-res cache dict + `finished_hires` signal carry the alpha next to `base`
  so the re-adjust-only fast path (base cached, sliders moved) reuses it.
- Add `bool(ccr_backend.sprocket_mask_white)` to `_current_adj_sig()` so toggling
  invalidates the baked hi-res tile (exactly like `auto_gain`/`gamma_luminance`).

**Slice reset (`ccr_backend` ~L1913, `apply_bwpoint_normalization` on a child's
raw `resized_raw`).** Immediately after producing the child's converted
`resized_raw`, set `parent.sprocket_alpha = compute_sprocket_alpha(<the raw before
inversion>, bw_points[0])`. (Capture the raw before it's overwritten by the
in-place invert.) Keeps sliced strips consistent; cheap.

### 4.3 Settings → General

Add a group to `SettingsDialog._build_general_page`:
> **Film border** · ☐ *White sprocket holes / clear film (reversal look)*
> *muted:* "After you set a Black Point (film base), paint the sprocket holes and
> clear film border white instead of black — the reversal-film look. Applied last,
> so adjustments never tint them. B/W-point conversions only."

Staged like the rest: seed in `_init_toggles` from `ccr_backend.sprocket_mask_white`,
apply on Done in `_apply_pending` by calling `main_window.on_sprocket_mask_toggled`
only when it differs from the live flag.

## 5. Data model

- `ccr_backend.sprocket_mask_white: bool = False` (new flag, in the toggle block).
  Persisted by MainWindow under QSettings key `adjust/sprocket_mask_white`,
  restored at startup before any render.
- `ccr_image.sprocket_alpha: Optional[np.ndarray]` (uint8 H×W, 0..255) — the
  cached feathered alpha for the **preview-resolution** buffer; set on every
  B/W-point convert (and slice reset), `None` otherwise. Not persisted to the
  catalog (recomputed on the reconvert that catalog load already performs).
- **No** conversion_inputs / catalog schema change; the alpha is derived, and the
  overlay is live from the global flag. A pre-feature catalog reconverts through
  `ccr_normalize_with_bwpoint`, which repopulates `sprocket_alpha`.

## 6. Integration points

- `core/ccr_processor`: add `compute_sprocket_alpha`, `apply_sprocket_mask`, the
  `SPROCKET_*` constants + `_sprocket_cfg()` env reader. In
  `ccr_normalize_with_bwpoint`: compute alpha from the raw; store on the image
  (preview) or composite post-adjust/pre-geometry (export), gated by
  `ccr_backend.sprocket_mask_white` on the export path (deferred import).
- `core/ccr_image`: new `sprocket_alpha` attr (init `None`); composite in
  `update_thumbnail_and_preview` (§4.2, gated); `render_hires_base` returns
  `(base, alpha)` and its single caller unpacks.
- `core/ccr_backend`: `self.sprocket_mask_white = False` in the flag block; set
  `parent.sprocket_alpha` in the slice-reset bw branch.
- `widgets/image_preview`: `HiResDetailWorker` snapshots the flag + carries/
  composites the alpha; hi-res cache dict + `finished_hires` payload gain the
  alpha; `_current_adj_sig()` includes the flag.
- `ui/main_window`: restore `sprocket_mask_white` from QSettings in `__init__`
  (key `adjust/sprocket_mask_white`, default False); add
  `on_sprocket_mask_toggled(checked)` → set flag, persist,
  `_rerender_all_for_global_mode(hint)` (render-only, releases hi-res cache).
- `widgets/settings_dialog`: General-page group + checkbox + staging in
  `_init_toggles` / `_apply_pending`.

## 7. Edge cases

- **No black point set / reference or auto conversion** → `sprocket_alpha` is
  None → overlay is a no-op even with the toggle on (the help text says B/W-point
  only). No error.
- **Positive mode** → skipped (`self.converted` is set by inversion; the preview
  gate also excludes positive; export routes through `ccr_export_positive`, which
  this spec does not touch).
- **No qualifying pixels** (frame cropped to just the image; 120 with a dense
  rebate) → `compute_sprocket_alpha` returns None → no-op.
- **Scene highlights / blown skies** → in a negative these are the *densest*
  (darkest scan) regions, structurally below base → never masked. Deep scene
  shadows approach but do not exceed the zero-exposure base; `pad` covers the
  approach. (Field-tunable via `FREECCR_SPROCKET_GAP_FRAC` if a scan's base is
  unusually noisy.)
- **Uneven scan illumination** (vignetted base dimmer in a corner) → the
  midpoint of the measured gap gives margin, and a frame with no clear film at
  all is caught by `MIN_CLEAR_D`; the area filter drops stray corner specks. Documented limitation, not a correctness bug.
- **Crop excludes the holes** → export/preview simply have no clear-film region in
  frame; nothing to whiten (expected — user chose to crop them out).
- **Histogram** reads the PRE-mask adjusted preview (the tones the user edits),
  so a whitened border — which can be ~15 % of a full-width 135 scan — does not
  add a misleading spike at white. `update_thumbnail_and_preview` composites the
  mask into the display pixels (thumbnail + preview pixmap) only and derives the
  histogram from the un-masked `display_img` (§4.2). **Scopes** render the
  displayed viewport, so they DO include the whitened border — acceptable v1
  (they mirror what's on screen); revisit only if requested.
- **Missing `sprocket_alpha` on an image converted by pre-feature code in the same
  session** (shouldn't happen — every convert path repopulates it): overlay is
  skipped for that image until its next conversion. Harmless.

## 8. Test plan (`tests/test_sprocket_mask.py`, pure numpy/cv2, headless)

- **Threshold isolates holes**: synthetic BGR raw — a film-base plate at `bp`
  (BGR), a bright square well above `bp+pad` (the "hole"), and a darker square
  below `bp` (exposed "scene"). `compute_sprocket_alpha` → alpha 255 inside the
  hole, 0 over base and over the scene square.
- **Padding rejects base noise**: base plate + gaussian noise with amplitude <
  `pad` → alpha all zero (returns None).
- **Per-channel AND**: a square brighter than base in G/R only (not B) → not
  masked (proves clear-film requires all channels, so an orange-cast miss can't
  false-positive).
- **Feather is a soft ramp**: the alpha has intermediate values (0<a<255) in a
  band ~`feather` px wide around the hole edge; interior is a solid 255.
- **Resolution scaling**: same synthetic content rendered at 1080 and at 4×
  produces geometrically-matching alpha (feather band width scales ≈4×; masked
  area fraction within tolerance) — proves preview/export agreement.
- **Composite paints white**: `apply_sprocket_mask` sets full-alpha pixels to
  65535 in all channels and leaves zero-alpha pixels byte-identical; a mid-alpha
  pixel is the exact `(1-a)·img + a·65535` blend.
- **No black point → None**: `compute_sprocket_alpha(raw, None) is None`.
- **Backend/Settings**: `ccr_backend.sprocket_mask_white` defaults False; the
  General checkbox seeds/round-trips; `on_sprocket_mask_toggled` flips + persists
  the QSettings key (GUI-light or mocked).
- **Convert stores alpha**: a `ccr_normalize_with_bwpoint(output_path=None)` on a
  synthetic scan with a clear-hole region leaves `ccr_image.sprocket_alpha` set;
  a reference conversion leaves it None.
- **Export composites only when on**: export a bw image with the flag ON → the
  hole region is white in the written file; with the flag OFF → it is black
  (byte-identical to today).

## 9. Open questions — RESOLVED

- **Cache the preview alpha vs recompute live → CACHE at convert.** The alpha
  needs the raw, which `update_thumbnail_and_preview` no longer holds (resized_raw
  is the positive). Storing the small uint8 alpha lets on/off be a pure re-render;
  zoom/export recompute from their own decode. (~3 MB → stored as uint8, <1 MB;
  computed once per convert.)
- **Where in the export order → after adjustments, before crop/flips/rotation.**
  Matches the preview (mask in un-transformed space, canvas applies geometry);
  white survives the warp. (Decision 0 / §4.2.)
- **Default OFF.** Stylistic, non-standard — the conversion default is unchanged
  (contrast with Auto Gain, which defaults ON as a convenience).
- **Per-channel AND vs luminance/any → AND.** Fewest false positives; clear film
  is bright in every channel, exposed content is dark in every channel, so the
  AND is both correct and the safest against whitening real content.
- **User-tunable padding/feather in v1 → NO, env knobs only.** Baked defaults +
  `FREECCR_SPROCKET_*` for calibration on real rolls; promote to UI only if
  field use shows one value doesn't fit all scanners.

# Spec: Field Correction (flat-field profile for lens/light falloff)

Status: REFINED v1
Owner: FreeCCR
Feature branch: `feature/flat-field-correction`

## 1. Summary

Add a **Field Correction** feature that removes the dark corners and colour
shading a camera-scanning rig bakes into every frame — lens vignetting (natural
cos⁴ + mechanical), sensor cover-glass/microlens colour shading, and an uneven
light source (light table / LED panel hot-spot).

The user shoots **one calibration frame**: an evenly lit, featureless neutral
surface — ideally the bare light source they scan on, through the same lens at
the same aperture and distance. A wizard (entry point: **Settings ▸ Color
Management**, alongside the IT8 camera-profile wizard) decodes that frame,
measures the falloff, and writes a **field-correction profile** (`.ffc`) into a
per-user library. Selecting a profile makes every subsequent decode — preview,
zoom, and export — multiply by the profile's per-channel gain map, so the frame
is flat before negative conversion ever sees it.

This is the classic **flat-field correction**: `corrected = raw × (ref / flat)`,
done in linear light, per channel, at decode time.

## 2. Goals / Non-goals

### Goals
- A **"Create Field Correction Profile…"** wizard reachable from Settings ▸
  Color Management, mirroring the IT8 wizard's shape (guidance → analyse →
  review → save/apply).
- Decode the calibration shot in the **same camera-native raw-linear space** the
  pipeline decodes scans in (reusing `read_image`'s profiling decode contract),
  so the measured field matches the data it will correct.
- Build a **smooth, per-channel gain map** robust to dust, noise, and hot pixels
  on the light table.
- **Validate the calibration shot** and tell the user when it is unusable:
  clipped, too dark, or not actually a flat field (they photographed a scene).
- **Report what the profile does**: max gain, per-channel corner falloff in
  stops, and the colour-shading spread — plus a visual gain map and a corrected
  preview of the calibration shot itself.
- A **profile library** (`<appdata>/field_profiles/*.ffc`) with an active
  selection, persisted across sessions, managed in the same Settings page.
- Apply at **decode time inside `read_image`**, so preview, hi-res zoom, slices,
  trichrome merges, and export all inherit it identically with no per-call-site
  plumbing.
- Reuse the existing **profile-mismatch flow**: changing the active field
  profile does not silently re-decode; it flags affected thumbnails with ⚠ and
  the existing *Replace with current camera profile* re-grade.
- Unit tests for map building, normalisation, validation, save/load round-trip,
  apply correctness, and the decode-hook integration.

### Non-goals
- **No automatic per-image profile matching** by lens/aperture/focal length. The
  target user has one fixed copy-stand setup. EXIF is *recorded* in the profile
  (and shown in the library) so a future auto-match is possible, but v1 selects
  one active profile globally, like the camera profile.
- **No distortion correction.** This is a per-pixel gain field only — geometry is
  untouched. (The dormant `correct_lens_distortion_and_vignette` lensfun path
  stays dead; this feature does not revive it.)
- **No dust/scratch removal from the calibration frame's subject.** Dust on the
  light table is *smoothed out* of the map (§5.3) rather than turned into a
  dust-removal feature; frame dust is handled by the existing dust tools.
- **No dark-frame / bias subtraction.** Flat-field division alone; a dark frame
  would matter only for long-exposure sensor glow, which a copy-stand scan at
  1/60 s does not have.
- **No lensfun/external vignetting database.** The measurement is the user's own
  rig, which is the point — it captures the light source, not just the lens.
- **No new dependency.** Pure numpy + cv2 + stdlib (`json`, `base64`), both
  already required.

## 3. Background

### 3.1 What the calibration frame captures
A photograph of a uniform, evenly lit surface records the *product* of every
multiplicative non-uniformity in the imaging chain:

| Source | Behaviour |
|---|---|
| Natural illumination falloff | ≈cos⁴θ, radial, worse at wide angle |
| Mechanical vignetting | radial, strongly aperture-dependent |
| Sensor microlens / cover-glass shading | radial + **per-channel** (colour shading), sensor-specific |
| Light source unevenness | arbitrary shape, often an off-centre hot spot |
| Filter/adapter clipping | asymmetric corner darkening |

Dividing by that measured field removes all of them at once. This is exactly why
a *measured* flat field beats a lens-model vignetting correction for camera
scanning: the light table is part of the optical chain, and no lens database
knows about it.

### 3.2 Why it must happen in linear light, before conversion
Vignetting is a **multiplicative attenuation of scene radiance**. It is only a
simple division in a linear-light space. FreeCCR's negative pipeline decodes
RAW to linear (`gamma=(1,1)`) and inverts on that data, so correcting inside
`read_image` — immediately after decode, before the camera profile, before the
negative inversion — is both physically correct and the single place that covers
every consumer.

Correcting *after* inversion would be wrong twice over: the inversion is
non-linear in the falloff (a density-space subtraction of a spatially varying
term), and the reference-crop normalisation would already have been contaminated
by the falloff it is supposed to be measured against.

### 3.3 Why per-channel
Colour shading (channel-dependent falloff, from the sensor stack and from the
light source's own spectrum varying across the panel) shows as a corner colour
cast, not just darkening. It is strongly amplified by negative inversion: a
neutral film base that reads slightly warmer in the corners inverts to a visibly
cyan corner. A single luminance gain map cannot fix that; three do.

### 3.4 Capture guidance (surfaced in the wizard)
- Shoot **the light source you scan on** — bare panel, no film — through the
  **same lens at the same aperture, focal length and distance** as the scan.
  Aperture matters most: vignetting changes by a stop or more between f/2.8 and
  f/8.
- **Defocus** slightly (or put a diffuser / a sheet of white acrylic on the
  panel) so panel texture and dust don't print into the map.
- **Fill the whole frame**; nothing dark at the edges.
- Expose so the frame's brightest area is around **50–75 % of full scale** and
  **nothing clips** — a clipped flat field measures the wrong falloff.
- **Shoot RAW.** A JPEG/TIFF is accepted, but its gamma encoding and any
  in-camera lens correction make the measurement approximate (warned in the UI).
- Re-shoot the profile whenever the rig changes (lens, aperture, panel, height).

## 4. Data model & files

### 4.1 New module `src/core/flat_field.py` (pure logic, no Qt)

```python
GRID_LONG_SIDE = 128       # stored gain-grid long side (aspect-preserving)
DEFAULT_MAX_GAIN = 4.0     # +2 stops; the build clamps to this
FORMAT = "freeccr-field-correction"
FORMAT_VERSION = 1

class FieldProfileError(Exception): ...

@dataclass
class FieldProfile:
    name: str
    gain: np.ndarray            # (gh, gw, 3) float32, >= ~1.0, C-contiguous
    aspect: float               # source frame w/h, for a mismatch hint
    meta: dict                  # camera/lens/EXIF, source file, created, stats
    path: Optional[str] = None  # library path once saved
    content_id: Optional[str] = None   # short hash -> grading signature

# --- capture -> profile ---
def decode_calibration(path, sample_max=1024) -> np.ndarray      # uint16 HxWx3
def analyze_field(arr_u16) -> FieldAnalysis                      # validation + stats
def build_profile(arr_u16, name, *, meta=None,
                  max_gain=DEFAULT_MAX_GAIN) -> FieldProfile

# --- storage ---
def save_profile(profile, path) -> None                          # .ffc (JSON)
def load_profile(path) -> FieldProfile                           # raises FieldProfileError

# --- apply ---
def apply_field(arr_u16, profile, *, encoded=False) -> np.ndarray
def gain_for_size(profile, w, h) -> np.ndarray                    # resized map cache

# --- global active profile (mirrors color_management's globals) ---
def set_active_profile(profile | None) -> None
def get_active_profile() -> Optional[FieldProfile]
def field_profile_active() -> bool
def active_field_signature() -> str        # 'ff:<content_id>' | 'none'
```

`FieldAnalysis` carries what the review page shows and what blocks/warns:
`level` (reference level, 0–1), `clipped_fraction` (per channel, max),
`flatness_residual` (relative RMS after smoothing), `corner_stops` (3,),
`cast_stops`, `max_gain`, and `warnings: list[str]`.

### 4.2 Profile file format (`.ffc`)
A UTF-8 JSON document — inspectable metadata, opaque payload:

```json
{
  "format": "freeccr-field-correction",
  "version": 1,
  "name": "Copy stand — 55mm f/8",
  "created": "2026-08-03T14:02:11",
  "source": "IMG_4471.CR3",
  "aspect": 1.5,
  "camera": {"make": "Nikon", "model": "Z 8", "lens": "NIKKOR Z MC 105mm",
             "focal_length": 105.0, "aperture": 8.0},
  "stats": {"max_gain": 1.84, "corner_stops": [0.79, 0.71, 0.88],
            "cast_stops": 0.17, "level": 0.62, "clipped_fraction": 0.0,
            "flatness_residual": 0.008},
  "max_gain_cap": 4.0,
  "grid": {"w": 128, "h": 85, "channels": 3, "dtype": "float32",
           "data": "<base64 of C-contiguous (h,w,3) little-endian float32>"}
}
```

`GRID_LONG_SIDE=128` keeps the file ≈180 KB while over-resolving a field whose
real bandwidth is a handful of cycles across the frame; the grid is upsampled
bicubically at apply time (§5.5). Unknown top-level keys are ignored on load
(forward compatibility); a `version` above `FORMAT_VERSION` is rejected with a
clear message.

### 4.3 Library storage
`<appdata>/field_profiles/*.ffc`, created on demand, next to the existing
`camera_profiles/` library (`ccr_backend.camera_profiles_dir()`'s sibling) so it
survives updates. Backend surface mirrors the camera-profile library:
`field_profiles_dir()`, `list_field_profiles()`, `set_active_field_profile(path)`,
`delete_field_profile(path)`.

### 4.4 Edits to existing files
| File | Change |
|---|---|
| `src/core/ccr_image.py` | apply hook at the three decode exits (§6.1) |
| `src/core/ccr_backend.py` | library + active selection; fold into `active_profile_signature()` |
| `src/widgets/settings_dialog.py` | "Field correction" group on the Color Management page |
| `src/ui/main_window.py` | wizard/select/delete handlers + startup restore |
| `src/widgets/field_correction_dialog.py` | **new** — the wizard |
| `src/core/flat_field.py` | **new** — all the logic |

## 5. Processing / math

### 5.1 Decode the calibration frame
Reuse the profiling-decode contract (`spec/it8-camera-profile.md` §5.1/§12.5):

```python
img = CCRImage.__new__(CCRImage); img.source_ops = []
arr = img.read_image(path, preview=True, max_long_side=1024,
                     positive_override=False, apply_input_icc=False)
```

This yields **camera-native, raw-linear, unbalanced, absolute-sensor-value**
RGB — the same space `read_image` hands to the correction hook, so the field is
measured in the space it is applied in. `apply_input_icc=False` additionally
suppresses the field correction itself (§6.1) — a calibration decode must never
be corrected by an existing profile, or profiles would compound.

A 1024-px decode is ~100× more samples than the 128-px grid needs, so noise is
already averaged away by the downsample alone.

### 5.2 Validation (`analyze_field`)
Computed on the decoded frame, normalised to `f = arr/65535`:

1. **Reference level** `level` — the robust central level (§5.3 step 4), max over
   channels. `level < 0.05` → *"too dark to measure; the corrected corners will
   be noisy"*. `level > 0.95` → *"over-exposed"*.
2. **Clipping** — per channel, fraction of pixels ≥ 0.99. `> 0.001` in any
   channel → *"highlights are clipped; the falloff can't be measured there"*.
3. **Flatness** — downsample to a 256-px working field, blur heavily (σ ≈ 1/12 of
   the long side), and take the relative RMS of `(field − smooth)/smooth`.
   `> 0.05` → *"this doesn't look like an evenly lit blank surface"* (the guard
   against profiling a photograph of an actual scene). Dust specks and panel
   texture sit far below this threshold after the 256-px downsample; a scene's
   edges sit far above it.

Warnings are **advisory** — the wizard shows them prominently and still lets the
user continue (their rig, their call), except that a build is refused outright
when the reference level is below 1 % of full scale (the map would be pure
noise amplification).

### 5.3 Build the gain map (`build_profile`)
1. **Downsample** the decoded frame to `GRID_LONG_SIDE` (aspect-preserving) with
   `cv2.INTER_AREA` — a box average over ~8×8 source blocks, which is already a
   strong low-pass and kills sensor noise (measured error: 2e-5).
2. **Reject blemishes** — a 5×5 median for isolated hot/dead pixels, then a
   **large median** whose window is `BLEMISH_WINDOW_FRAC` (0.13) of the grid's
   long side. A median is the right robust estimator because it is *exact on
   monotone data*: a genuine sharp shading edge (the fast corner cut of a
   mechanical vignette) passes through untouched, while an isolated blob smaller
   than the window is replaced by the field around it. §10.8 records the
   alternatives that were measured and rejected.
3. **Gaussian blur**, σ = `GRID_LONG_SIDE/128` (1 px) — just enough to
   de-staircase the median. Deliberately minimal: blurring a curved field biases
   it by ~σ²/2·∇²f, worst exactly at the corners, and at σ=2 a sharp
   mechanical-vignette knee was reproduced only to 10% (vs 6% at σ=1). All
   filtering pads by **odd (antisymmetric) reflection**, `f(−k) = 2f(0) − f(k)`,
   which continues the field's slope past the frame edge; every OpenCV border
   mode instead holds it flat (REPLICATE) or folds it back (REFLECT), which
   understates a still-falling falloff and under-corrects the corners by ~1.2%.
4. **Reference level per channel** `ref_c`: the median of the smoothed grid over
   a **central window** covering the middle 10 % of frame area. Normalising on
   the centre (not on each channel's max) is what keeps the profile
   *photometrically neutral*: gain is exactly 1.0 at the centre in all three
   channels, so the profile changes neither the frame's exposure nor its white
   balance — it only removes **spatial variation**. A region brighter than the
   centre (an off-centre light-table hot spot) correctly gets gain < 1.
5. **Gain** `g = ref_c / max(smooth_c, floor)`, `floor = 1e-4` (guards a black
   corner from producing an infinity), then **clamped to `[0.25, max_gain]`**
   (default `max_gain = 4.0`, i.e. +2 stops). The clamp is the noise-amplification
   guard: past +2 stops the corner is mostly read noise, and a botched
   calibration shot (a lens cap, a partially covered panel) can't produce a map
   that destroys images. `stats.max_gain` reports the value actually reached, and
   the wizard says so when the clamp bites.
6. **Stats**: `corner_stops[c] = log2(mean gain over the four corner regions)`
   (each region = the outer 5 % × 5 % of the frame), `cast_stops =
   max(corner_stops) − min(corner_stops)` (how much of the falloff is *colour*
   shading rather than plain darkening).

The stored grid is `float32`, C-contiguous `(gh, gw, 3)`.

### 5.4 Why a grid and not a radial polynomial
A radial model (the usual lens-profile parameterisation) assumes the falloff is
centred and rotationally symmetric. Light-table hot spots are neither, and a
mis-centred adapter breaks symmetry too. A 128-px grid represents any smooth
field, costs 180 KB, and needs no fitting step that could fail to converge.

### 5.5 Apply (`apply_field`)
```
gain = resize(profile.gain, (w, h), INTER_CUBIC)      # cached per (w,h)
out  = clip(arr.astype(float32) * gain, 0, 65535).astype(uint16)
```
- **Bicubic** upsampling: the map is smooth, and bicubic's C¹-ish continuity
  avoids the faint slope-discontinuity facets bilinear can leave when a 128-px
  grid is stretched to 6000 px.
- **Banded** — the multiply runs in horizontal bands of 512 rows so a
  6000×4000 export never materialises a 288 MB float32 temporary.
- **Cached** — the resized map is memoised on the profile keyed by `(w, h)`
  (bounded to the last few sizes), because a batch export hits the same size for
  every frame.
- **Clipping**: gain ≥ 1 away from the centre means a value already at full scale
  clips. For a negative scan the brightest area is the film base, and the
  windowed working space (`spec/working-space-headroom.md`) plus the fact that
  the corners being lifted are *dark* to begin with make this a non-issue in
  practice; the wizard's "don't clip the calibration shot" guidance is the real
  protection.
- **`encoded=True`** (the Positive-mode RAW decode, the one decode we *know*
  carries an sRGB TRC because we asked rawpy for it): linearise with the sRGB
  EOTF, multiply, re-encode. A linear-light gain applied directly to
  gamma-encoded data under-corrects by roughly the gamma exponent.
  Non-RAW sources are left in their own space (§10.4).

### 5.6 Monochrome / grayscale sources
A monochrome RAW (and a grayscale TIFF/JPEG) is decoded to three *identical*
channels. Applying a three-channel gain map to it would inject colour into an
image that has none. The hook therefore passes `mono=True` for those branches
and `apply_field` uses the **mean of the three channel gains**, keeping the
frame neutral while still correcting the falloff. A monochrome *calibration*
shot needs no special case — its map simply carries three equal channels.

### 5.7 Aspect mismatch
`profile.aspect` records the calibration frame's `w/h`. At apply time the map is
resized to whatever the frame is, which is correct for any resolution of the same
frame shape. A frame whose aspect differs by more than 2 % (a different sensor, a
different in-camera crop) is **still corrected** but logged once, and the
Settings library row shows the profile's camera/lens/aspect so the user can see
the mismatch. Blocking is worse than a slightly-off correction the user asked
for.

## 6. Integration points

### 6.1 Decode hook (`ccr_image.read_image`)
The correction is applied at the **full-frame, pre-`_apply_source_ops`** point of
every decode branch — before slicing crops the array, so the map always
corresponds to the whole sensor frame regardless of slice/crop state:

| Branch | Insertion point |
|---|---|
| RAW (`raw.postprocess`) | immediately after `postprocess`, before `_apply_source_ops` |
| non-RAW (TIFF/JPEG/PNG) | after channel normalisation, before `_apply_source_ops` |
| 3-way merge (`_read_merged`) | after `merge_raw_channels`, before `_apply_source_ops` |

Gating, in one helper `_apply_field_correction(arr, *, encoded=False, mono=False)`:
- skipped when `apply_input_icc=False` (the calibration/IT8 profiling decodes
  must see bare, uncorrected device data — §5.1, §10.1);
- skipped when no profile is active;
- `encoded=True` only on the Positive-mode sRGB RAW decode (§5.5);
- `mono=True` on the monochrome-RAW and grayscale-file branches (§5.6).

Ordering relative to the RAW branch's manual white-level scaling is immaterial
(both are linear multiplies; the uint16 clamp is reached at the same place), so
the hook sits at the geometric point that keeps the map aligned.

**Trichrome/linear-TIFF interaction**: `_export_merged_linear` writes its TIFF
from `merge_raw_channels` directly, *not* through `read_image`, so a baked
replacement TIFF stays **uncorrected** on disk and is corrected when reloaded
like any other file. No double correction, no marker needed.

### 6.2 Grading signature (mismatch ⚠ + Replace)
`ccr_backend.active_profile_signature()` appends the field-correction signature:

```python
def active_profile_signature(self) -> str:
    base = ("none" if self.positive_mode else
            "camera_matrix" if color_management.camera_matrix_mode() else
            color_management.active_profile_signature())
    ff = flat_field.active_field_signature()   # 'ff:<content_id>' | 'none'
    return base if ff == "none" else f"{base}+{ff}"
```

The camera-profile part keeps its existing meaning exactly (Positive mode still
short-circuits it to `"none"`); the field part is composed on independently,
because field correction **does** apply in Positive mode. The signature is
compared as an opaque string (`thumbnail_list._profile_mismatch`), so changing or
clearing the field profile flags exactly the images decoded under the other
setting, and the existing **Right-click ▸ Replace with current camera profile**
re-decodes and replays them.

The mismatch tooltip/menu wording becomes "camera profile or field correction".

### 6.3 Settings ▸ Color Management
A new group box **Field correction**, below Camera profiles:

```
┌ Field correction ──────────────────────────────────────────┐
│ Corrects dark corners and colour shading from your lens,   │
│ sensor and light source, measured from one shot of an      │
│ evenly lit blank surface.                                  │
│ Active: [ None                              ▾ ]            │
│ [ Create Field Correction Profile… ] [ Delete ]            │
│ Copy stand — 55mm f/8 · Nikon Z 8 · 105 mm f/8 · 3:2       │
│ Max +0.9 EV · colour spread 0.17 EV                        │
└────────────────────────────────────────────────────────────┘
```
The combo lists `None` + every library profile. Selection is **immediate** (like
camera-profile selection, not staged like the global toggles), persisted under
`QSettings("import/field_profile_path")`, and triggers the ⚠ re-flag — never a
silent re-decode.

### 6.4 Startup restore
`MainWindow.__init__` restores the persisted selection next to the camera-profile
restore, before any image loads. A missing/unparseable file falls back to None
and clears the setting.

## 7. UX — the wizard (`FieldCorrectionDialog`)

Three pages, Back/Next/Cancel, same chrome as `IT8ProfileDialog`.

**Step 1 — Calibration shot.** Capture guidance (§3.4) + *Use current image* /
*Browse…*. A non-RAW pick warns (gamma-encoded / possibly already corrected
in-camera) but is allowed. Next decodes (wait cursor).

**Step 2 — Review.** Three panes: the **calibration shot** (gamma-stretched),
the **gain map** (false colour, with a `1.0× … max×` legend), and the
**corrected shot** (the map applied to the calibration frame — visibly flat when
the profile is good). Below them: level, max gain, per-channel corner falloff in
EV, colour spread in EV, and any validation warnings in amber. A **Profile name**
field, defaulted to `Field <camera/lens> <date>`.

**Step 3 — Save.** The profile is always written **into the library folder**,
named from the profile name (no path picker — a field profile is only useful
through the library, and IT8's "save anywhere, then import it back" round-trip is
avoidable complexity). The resolved path is shown read-only; an existing file of
the same name asks before overwriting. Checkbox **"Use this profile now"**
(default on). Finish writes the `.ffc`, and on *use now* activates it — which
flags the loaded thumbnails ⚠ so the user can re-grade the frames they care
about.

## 8. Test plan

Unit (`tests/test_flat_field.py`, pure numpy — no Qt):
- **Synthetic round-trip (key test)**: build a known smooth falloff field
  `F(x,y)` (radial + an off-centre hot spot, per-channel), synthesise a
  calibration frame from it, build a profile, and assert the map recovers
  `ref/F` within tolerance; then apply the profile to a *synthetic scene*
  multiplied by the same `F` and assert the corrected result matches the
  original scene within ~1 %.
- **Neutrality**: the centre gain is 1.0 (all channels) and applying the profile
  to a flat neutral frame leaves both its level and its channel ratios unchanged
  — the profile must not become a white-balance or exposure change.
- **Colour shading**: a field with different per-channel falloff is corrected to
  a neutral corner; `cast_stops` reports the input spread.
- **Robustness**: dust specks (small dark blobs) and hot pixels injected into the
  calibration frame do not print into the map (max deviation from the clean-frame
  map below tolerance).
- **Clamping**: a near-black corner produces `max_gain` exactly, not an infinity
  or a NaN; `stats.max_gain` reflects it.
- **Validation**: a clipped frame, a black frame, and a *photograph of a scene*
  (high-frequency content) each raise their expected warning; a good frame raises
  none; a sub-1 % frame refuses to build.
- **Save/load round-trip**: `save_profile` → `load_profile` reproduces the grid
  bit-exactly (base64 float32) and all metadata; a bad JSON, a truncated base64
  payload, a wrong `format`, and a future `version` each raise
  `FieldProfileError` with a useful message.
- **Apply**: resolution independence (the same normalised gain at the same
  relative position for a 1080-px preview and a 6000-px export, ≤1 % deviation);
  banding path equals the single-shot path exactly; `encoded=True` round-trips a
  gamma-encoded flat frame correctly; dtype/shape/clipping preserved.
- **Signature**: `active_profile_signature()` changes when the field profile is
  set/cleared/swapped, and is stable when unchanged; a mismatch is detected by
  the thumbnail predicate's comparison.
- **Decode hook**: with a profile active, a stubbed decode returns corrected
  data; `apply_input_icc=False` returns it **un**corrected (calibration decodes
  must not compound); no profile ⇒ byte-identical passthrough.

Manual:
- Full wizard on a real light-table shot; corner falloff readout matches the
  visible vignette; corrected preview is flat.
- A converted negative before/after activation: corners lift, no colour shift in
  the centre, histogram unchanged in the middle.
- Export at full resolution matches the on-screen preview's corners.
- Switching profiles flags ⚠ and *Replace* re-grades correctly.
- A profile built from a *clipped* shot warns and (if used anyway) does not crash.

## 9. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Calibration shot clipped → wrong falloff | Detected and warned (§5.2); guidance targets 50–75 % exposure |
| Aperture/lens differs from the scan | Guidance + EXIF recorded and shown; aspect mismatch logged |
| Noise amplification in deep corners | INTER_AREA + median + Gaussian smoothing; gain clamped to +2 EV |
| Highlight clipping after correction | Corners lifted are dark; windowed working space carries headroom; documented |
| Double correction (profile applied to its own calibration decode) | `apply_input_icc=False` suppresses the hook (§6.1) — unit-tested |
| Baked trichrome TIFF corrected twice | Bake path bypasses `read_image`, so the TIFF is uncorrected on disk (§6.1) |
| User profiles a photo by mistake | Flatness check (§5.2 step 3) catches scene content |

## 10. Refinement (v1) — resolved decisions

### 10.1 `apply_input_icc=False` is the "bare device decode" flag
The field correction is suppressed by the **existing** `apply_input_icc=False`
parameter rather than a new one. That parameter already means more than its name
(it also forces the camera-native decode space, `ccr_image.py` §"icc_device_space"),
and both of its callers — `it8_profile.decode_target` and the new
`flat_field.decode_calibration` — are *profiling* decodes that must see bare
device data. A separate `apply_field` parameter would have to be threaded through
both call sites to reach the same behaviour, with a live footgun if either forgot
it. The docstring is updated to state the parameter's real contract: **bare
device decode — no camera profile, no field correction**.

Consequence, accepted deliberately: an **IT8 target shot is not field-corrected**
before the matrix fit. The chart is shot filling the *centre* of the frame (the
wizard's own guidance), where falloff is smallest, and a field profile measured on
the scanning rig would be simply wrong for a chart shot in a different setup.

### 10.2 Normalise on the centre, not on the maximum
Both keep gain ≥ 1 almost everywhere, but centre-normalisation is the one that
makes the profile **photometrically and chromatically neutral**: gain is exactly
1.0 at the centre in all three channels, so activating a profile changes neither
exposure nor white balance — only spatial variation. Max-normalisation would tie
the profile's overall gain (and, per-channel, its white balance) to wherever the
light table happened to be brightest. It also handles an off-centre hot spot
correctly, by darkening it (gain < 1) instead of lifting the whole frame to meet
it. Unit-tested as the "neutrality" case in §8.

### 10.3 Gain clamp at +2 EV
`max_gain = 4.0`. Real camera-scan rigs land at +0.5…+1.5 EV in the corners;
anything beyond +2 EV is either a badly wrong calibration shot (lens cap,
partially covered panel, a frame that clipped to black) or a corner so dark that
the corrected result is read noise. Clamping is the difference between "the
profile helps a bit less than it could" and "the profile destroys the image", and
the wizard reports when the clamp binds so a genuine super-wide-angle user knows
why.

### 10.4 Encoding: the gain is applied in the array's own space
A multiplicative light attenuation stays multiplicative through a pure power-law
encoding (only the exponent changes), so **measuring and applying the ratio in
the same space is self-consistent** — a JPEG calibration shot correcting JPEG
scans is as correct as a RAW one correcting RAW scans. The only decode where the
space provably differs from the calibration decode is the **Positive-mode RAW**
path (rawpy applies an sRGB TRC while `decode_calibration` pins `gamma=(1,1)`),
and that one gets the explicit linearise/re-encode round trip (§5.5).

Not attempted: guessing whether an arbitrary TIFF/JPEG is linear or gamma-encoded
(e.g. by bit depth). FreeCCR's pipeline treats every non-RAW decode as linear
throughout, and introducing a contradicting assumption in this one place would be
worse than the documented limitation — **a RAW-built profile under-corrects
gamma-encoded non-RAW scans**; the fix is to calibrate in the same format you
scan in.

### 10.5 No import, no rename, delete only
The wizard writes straight into the library (§7 step 3). A field profile
describes one physical rig, so cross-machine sharing — the thing Import exists
for on camera profiles — has no real use case here; a user who copies a `.ffc`
into the folder by hand still gets it listed. Renaming is "create it again with
the right name", which takes seconds because the calibration shot is already on
disk.

### 10.6 Resized-map cache
`gain_for_size` memoises the upsampled map on the profile object, keyed by
`(w, h)`, bounded to the 4 most recent sizes (preview 1080, thumbnail, zoom tile,
export). Decodes run on up to 8 worker threads, so two threads can race to
compute the same entry — harmless (identical results, dict writes are atomic
under the GIL), and cheaper than holding a lock across a resize. The cache is
never a correctness input: it is a pure function of `(profile.gain, w, h)`, and
selecting a different profile replaces the object entirely.

### 10.7 Cost
Preview (1080 px): resize + banded multiply ≈ 3 ms, invisible next to a ~300 ms
RAW decode. Full export (6000×4000): ≈ 200 ms, against several seconds of decode
+ render. No caching of corrected pixels is needed, and none is added.

### 10.8 Blemish rejection: what was measured
All figures are the worst-case relative error of `gain × field` (a perfect
profile is uniform), on synthetic fields at 768×1152.

| Estimator | Clean smooth field | Off-centre hot spot | Sharp 1 EV vignette knee | Dust |
|---|---|---|---|---|
| 3×3 median + Gaussian σ=GRID/24 | 2.9% | — | — | ok |
| 5×5 median + morphological CLOSE | 0.7% | 1.7% | — | fails past the kernel size |
| Iterated clipped smoothing (reject by depth) | 0.7% | 1.7% | **29% — rewrote 54% of the grid** | best |
| **5×5 + large median (shipped)** | **0.9%** | **1.6%** | **6%** | ≤2.5% up to a 2.6%-wide smudge |

The decisive case is the sharp knee. Rejecting by *depth* rather than *size*
looks more principled — "dust is a locally deep excursion, shading is smooth" —
but a mechanical vignette's corner cut is also deep and also local, so that
estimator ate it. Rejecting by *size* with a median keeps the monotone edge by
construction. The window is then a straight trade: 0.13 of the frame rejects
smudges up to ~2.5% of the frame width at 0.9% fidelity cost; 0.20 rejects
5%-wide smudges but degrades the knee to 11%. Blemish rejection runs once per
profile build (~60 ms), so its cost is irrelevant.

The residual border error was likewise measured, not assumed: filtering with
OpenCV's border modes left a **−0.94% corner** and **−1.6% edge** bias, which odd
reflection (§5.3) reduced to −0.05% / −0.44%.

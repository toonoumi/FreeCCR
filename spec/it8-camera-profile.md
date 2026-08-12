# Spec: Create Camera ICC Profile from an IT8 Target

Status: REFINED v1
Owner: FreeCCR
Feature branch: `feature/it8-camera-profile`

## 1. Summary

Add a File-menu feature that builds a **camera input ICC profile** from a
photograph of an **IT8 calibration target** shot under the user's capture light.
The user photographs a printed IT8 chart (a grid of patches with factory-measured
colours), supplies the matching batch reference file, locates the patch grid in
the shot, and the app fits a camera-RGB→XYZ colour transform and writes a valid
ICC profile. The profile plugs directly into FreeCCR's **existing global Input
ICC Profile** system (`File ▸ Set Input ICC Profile…`), so once built it is
applied to every decoded scan before negative conversion/adjustments.

The whole feature is **self-contained in pure numpy + OpenCV** (both already
dependencies). It reuses the in-process ICC writer
(`color_management.build_matrix_shaper_icc`) and the `InputProfile` apply path
that already exist — **no ArgyllCMS, lcms, Pillow, scipy, or colour-science**
dependency is added.

### Why this fits FreeCCR exactly
- The IT8 profiling decode explicitly requests **raw-linear sensor RGB** by
  passing `apply_input_icc=False` to `read_image` (forces `rawpy`
  `output_color=raw`, `gamma=(1,1)`, no white balance, `no_auto_scale=True` with
  white-level scaled to 16-bit) — see `_raw_color_postprocess_kwargs`/
  `read_image`. That is precisely the *device RGB* an input ICC profile maps
  from. (Note: under a separate change the *default* unprofiled negative decode
  is Adobe RGB + rawpy auto-scale — `no_auto_scale=False`, which also applies a
  daylight WB — whereas the ICC-corrected and the IT8 bare-device decode are raw
  primaries with absolute sensor values, so IT8 sampling still gets the unscaled
  device RGB it fits on.)
- A camera profile fit as a **3×3 matrix (linear device RGB → XYZ D50) + identity
  tone curve** is exactly a **matrix-shaper** ICC profile — the only kind the app
  builds (`build_matrix_shaper_icc`) and the only kind `InputProfile` accepts.
- The created profile is therefore round-trippable: we synthesise it, and the
  app's own `InputProfile.from_bytes` parses and applies it with no new plumbing.

## 2. Goals / Non-goals

### Goals
- A File-menu item **"Create Camera Profile from IT8…"** that opens a guided
  multi-step dialog (wizard).
- Parse an IT8 **CGATS.17** reference file (Wolf Faust `.it8`/`.txt` and
  equivalents), reading columns dynamically from the `DATA_FORMAT` block.
- Decode the IT8 target shot to **raw-linear device RGB**, identically to the
  negative pipeline and **independent of** the global Positive-mode toggle and of
  any currently-set input profile.
- A **patch locator**: the user drags 4 corner handles onto the colour grid; the
  app overlays all 288 sample points (264 colour + 24 grayscale) via a perspective
  homography and samples each patch robustly (central window, trimmed mean, clip
  rejection).
- Fit a **3×3 camera matrix** (least squares, neutral-axis pinned to D50) and
  write a valid matrix-shaper ICC profile via the existing builder.
- Report **fit quality** (mean / median / max ΔE2000) with a pass/warn indicator
  and a per-patch list.
- **Save** the `.icc` to a per-user profiles folder, and optionally **apply it
  now** as the global input profile (reusing `ccr_backend.set_input_icc`).
- Unit tests for the parser, grid geometry, matrix fit (synthetic round-trip),
  ΔE math, and ICC build→reparse.

### Non-goals
- **No ArgyllCMS / external binaries.** No `scanin`/`colprof`, no per-platform
  bundles, no AGPL entanglement.
- **No cLUT / polynomial / root-polynomial profiles.** A matrix-shaper ICC can
  only store a 3×3 + per-channel curves, so higher-order fits are out of scope
  (noted in §5.6). 3×3 is the correct, well-behaved camera model here.
- **No automatic chart detection / fiducial recognition.** Manual 4-corner
  placement is the robust, industry-standard choice for hand-held camera shots.
- **No dual-illuminant profiles** (that is a DCP feature; a single ICC is valid
  only under the light it was shot — surfaced as guidance, not engineered around).
- **No application of the camera profile in Positive mode.** The input ICC is, by
  existing design (`read_image` skips it when `positive_decode`), a tool for the
  raw-linear negative path. Positive mode already colour-manages via `rawpy`
  `output_color=sRGB`. Out of scope to change.
- **No spectral / measurement-instrument path** (we consume a reference file; we
  do not measure the chart).
- No bundled reference files — the user must supply the file matching their
  chart's batch (we link them to the source in the UI text).

## 3. Background (researched)

### 3.1 What IT8 is
An IT8 target (ISO 12641) is a printed chart of colour patches whose CIE values
were spectrophotometrically measured at the factory. **IT8.7/1** is transmissive
(film/slide); **IT8.7/2** is reflective (paper). The patch *layout is identical*
between them; only the substrate differs. The feature accepts either (the chart
type is read from the reference file but does not change the math). Common source:
Wolf Faust (coloraid.de / colorreference.de), which ships a **free reference file
per production batch**.

### 3.2 Chart geometry
- **Colour grid:** 12 rows `A`–`L` (top→bottom) × 22 columns `1`–`22`
  (left→right) = **264 patches**. Patch IDs concatenate row+col: `A1`…`A22`,
  `B1`… `L22`. Iteration order: column fastest, row slowest.
- **Grayscale strip:** **24 steps `GS0`…`GS23`** in one horizontal row **below
  row L**, `GS0` (lightest / Dmin) at left → `GS23` (darkest / Dmax) at right. The
  strip uses the same horizontal pitch as the colour columns but has 24 cells and
  begins flush at the chart's left margin, so it is **slightly wider and offset**
  from the 22-column block — it must be modelled as its own 1×24 row, not as
  "22 + 2".
- ArgyllCMS's `it8.cht` gives proportionally exact extents (active frame
  615.5 × 409 units; colour block x∈[26.625, 590.375], y∈[26.625, 334.125]; gray
  strip x∈[0, 615], y∈[358.75, 410]; `BOX_SHRINK 3.5` ⇒ sample ~73% central). We
  use these ratios to place the grayscale row relative to the colour block.

### 3.3 Reference file (CGATS.17) format
Plain ASCII. Structure:
- **Line 1**: chart-type token, e.g. `IT8.7/1` or `IT8.7/2`.
- **Preamble**: `KEYWORD "value"` lines — `ORIGINATOR`, `DESCRIPTOR`,
  `MANUFACTURER`, `SERIAL "E111007 Batch Average Data"` (batch id),
  `NUMBER_OF_FIELDS n`, `NUMBER_OF_SETS 288`. `#` starts a comment. A
  `KEYWORD "NAME"` line registers a custom column name.
- **`BEGIN_DATA_FORMAT` … `END_DATA_FORMAT`**: one whitespace-separated line of
  column names, e.g.
  `SAMPLE_ID XYZ_X XYZ_Y XYZ_Z LAB_L LAB_A LAB_B …`. **Field set varies by
  batch** (17 vs 18 fields seen). The only reliably-present columns are
  `SAMPLE_ID`, `XYZ_X/Y/Z`, and `LAB_L/A/B`.
- **`BEGIN_DATA` … `END_DATA`**: one row per patch, tokens positional against the
  format line, keyed by `SAMPLE_ID`. XYZ are D50 / 2° tristimulus scaled to white
  ≈ Y=100. Lab is CIE L\*a\*b\* against D50.
A parser **must** read columns from the `DATA_FORMAT` block (never hard-code
positions), tolerate CRLF/BOM/comments/blank lines/quoted-or-unquoted values, and
validate row/field counts.

### 3.4 The fitting model (researched, reduced to our container)
For raw-linear device RGB the well-behaved camera model is:
**identity TRC + a single 3×3 matrix `M` (device-linear RGB → XYZ D50), least
squares, neutral-axis pinned so equal RGB → D50 white.** Because the reference is
already D50, `M`'s columns are directly the ICC `rXYZ`/`gXYZ`/`bXYZ` colorants
(no Bradford adaptation needed). Good 3×3 camera profiles land **avg ΔE2000 ≈
1–3, max ≈ 5–8**; avg < 2 is good, avg < 1 is rarely reachable without
cLUT/polynomial fits (out of scope).

### 3.5 Capture conditions (surfaced as in-app guidance)
A camera profile bakes in the light's spectrum and is valid only under that light.
Guidance to show the user (Step 1 of the wizard and a "?" help): broad-spectrum
daylight/strobe (not tungsten/fluorescent); even 45° lighting, no glare/specular
hotspots, no vignetting; chart flat and fronto-parallel; fill the centre of the
frame; expose so the brightest neutral (GS0) is bright but **not clipped** in any
channel; **shoot RAW** (the profiling decode uses raw-linear: no WB, linear gamma,
no camera matrix — the same device space the negative pipeline relies on).

## 4. Data model & files

### 4.1 New module `src/core/it8_profile.py` (pure logic, no Qt)
Public surface:
```python
# --- reference file ---
class IT8Reference:
    chart_type: str            # 'IT8.7/1' | 'IT8.7/2' | '' (unknown)
    batch: str                 # SERIAL value, for display/confirmation
    descriptor: str
    fields: list[str]          # column names from DATA_FORMAT
    patches: dict[str, dict]   # 'A1' -> {'XYZ_X':..,'XYZ_Y':..,'XYZ_Z':..,'LAB_L':..}
    def xyz(self, sample_id) -> np.ndarray | None      # (3,) D50, Y≈100 scale
    def lab(self, sample_id) -> np.ndarray | None      # (3,) from LAB_* or derived
def parse_it8_reference(path: str) -> IT8Reference     # raises IT8ReferenceError

# --- patch grid geometry ---
COLOR_IDS:  list[str]    # ['A1',...,'A22','B1',...,'L22'] (264, col-fastest)
GRAY_IDS:   list[str]    # ['GS0',...,'GS23'] (24)
ALL_IDS = COLOR_IDS + GRAY_IDS
def grid_sample_points(quad, gray_offset=DEFAULT_GRAY) -> dict[str, (x, y)]
    # quad = 4 corners (TL,TR,BR,BL) of the COLOR block in array coords.
    # Returns array-space sample centres for all 288 ids via a homography;
    # gray row placed from §3.2 ratios (gray_offset tunes its vertical position).

# --- sampling ---
def sample_patches(img_u16, points, frac=0.5) -> dict[str, PatchSample]
    # PatchSample: rgb (3,) float in [0,65535], valid: bool (clip test), n_pix:int
    # central-window trimmed mean per channel; valid=False if >2% pixels clipped.

# --- fit ---
class CameraFit:
    matrix: np.ndarray         # (3,3) device-norm RGB[0,1] -> XYZ D50 (white≈1)
    avg_de: float; med_de: float; max_de: float
    per_patch: list[(id, dE2000)]
    used_ids: list[str]; dropped_ids: list[str]
def fit_camera_matrix(samples, ref, *, weight='1/Y', wb_id='GS0') -> CameraFit

# --- icc ---
def build_camera_icc(fit, desc, illuminant_note) -> bytes
    # identity TRC + matrix colorants via color_management.build_matrix_shaper_icc
```
`IT8ReferenceError` is a new exception (mirrors `UnsupportedICCError`'s role) so
the dialog can show a friendly message on a malformed/empty file.

### 4.2 New widget `src/widgets/it8_profile_dialog.py` (Qt UI)
- `IT8ProfileDialog(QDialog)` — a `QStackedWidget` of pages with Back/Next/Cancel,
  matching the app's existing dialog styling (cf. `export_dialog.py`,
  `UpdateAvailableDialog`).
- `IT8PatchLocator(QWidget)` — displays the (gamma-stretched for visibility)
  decoded target, 4 draggable colour-block corner handles, a live overlay of all
  288 sample dots, a **Flip 180°** button, and a **gray-strip vertical** nudge
  slider for batch variation. Works in the *sampling array's* coordinate space
  (a display↔array scale is applied for painting).

### 4.3 Edits
- `src/ui/main_window.py` — add the File-menu action + handler
  `create_camera_profile_from_it8()`.
- `src/core/ccr_image.py` — extend `read_image` with two optional, default-noop
  params so the profiling decode is exact and isolated (see §6.1):
  `read_image(..., positive_override: Optional[bool] = None,
   apply_input_icc: bool = True)`.

### 4.4 Profile storage
Saved `.icc` files live in a per-user folder next to the catalog:
`<appdata>/camera_profiles/<name>.icc` (folder created on demand). "Apply now"
copies into the existing input-profile slot via `ccr_backend.set_input_icc(path)`
(which already copies to its own persistent working copy and activates it). The
two storages are independent by design (a saved camera profile survives changing
the active input profile).

## 5. Processing / math (in `it8_profile.py`)

### 5.1 Decode the target (device RGB)
Use `read_image(preview=True, max_long_side=SAMPLE_MAX, positive_override=False,
apply_input_icc=False)` (see §6.1) so the target is decoded **raw-linear, no input
ICC, regardless of Positive mode**. `SAMPLE_MAX = 2048` bounds memory while
leaving ample pixels per patch (~80px/patch ⇒ ~1.5k px in each central window);
the half-size (preview) RAW decode is plenty at that cap and matches the runtime
preview's device space. `decode_target` constructs the `CCRImage` via `__new__`
(setting only `source_ops`) to skip the heavyweight `__init__` (which would do a
redundant 1080px decode + lens correction + thumbnail).

### 5.2 Patch grid geometry (`grid_sample_points`)
- Build a perspective homography `H` mapping the unit square corners
  `(0,0),(1,0),(1,1),(0,1)` → the user's 4 colour-block corners `(TL,TR,BR,BL)`
  (via `cv2.getPerspectiveTransform`).
- **Colour patch centres** in unit space: column `c∈0..21` → `u=(c+0.5)/22`;
  row `r∈0..11` → `v=(r+0.5)/12`. Map `(u,v)` through `H`.
- **Gray strip** in the *same* unit space, from §3.2 ratios relative to the colour
  block (block width 563.75, height 307.5, origin (26.625,26.625) in cht units):
  `u_g(k) = ((k+0.5)*25.625 - 26.625)/563.75` for `k∈0..23`;
  `v_g = (384.375 - 26.625)/307.5 + gray_offset ≈ 1.163 + gray_offset`.
  These land outside `[0,1]`; the homography extrapolates correctly because the
  chart is planar. `gray_offset` (slider, default 0) nudges the strip vertically
  to absorb manufacturer variation.
- Orientation: handles are placed TL→TR→BR→BL; **Flip 180°** reverses the corner
  list (and the dot overlay updates), covering an upside-down placement. A
  validation hint checks that sampled `GS0` is lighter than `GS23` and warns if
  not.

### 5.3 Sampling (`sample_patches`)
For each id's centre `(x,y)`, take a window of `frac × local-cell-size` (default
`frac=0.5`; ~central 50%), `clip` to image bounds, compute a **per-channel
trimmed mean** (drop the 10th/90th percentile tails) over the window. Mark
`valid=False` when >2% of window pixels are clipped (≤0.5% or ≥99.5% of full
scale) in any channel — those patches are excluded from the fit. Local cell size
is derived from neighbouring mapped centres so the window scales with perspective.

### 5.4 Matrix fit (`fit_camera_matrix`)
Inputs: `samples` (device RGB per id) and `ref` (XYZ per id). Steps:
1. Keep ids present in **both** sample set and reference and with `valid=True`.
   Normalise device RGB `d = rgb/65535 ∈ [0,1]`; reference `X = XYZ/100`
   (so GS0 white ≈ Y 0.83, D50 white = 1.0).
2. **White-balance** on the lightest valid neutral (`wb_id='GS0'`, fallback to the
   highest-L valid `GS*`): `gains = mean(d_wb_patch) / d_wb_patch` (per channel);
   `d_wb = d * gains` (now the neutral is equal-RGB).
3. **Weighted least squares** (weight `w_i = 1/max(Y_i, ε)` to stop bright patches
   dominating): with `sw=√w`, solve
   `A, *_ = np.linalg.lstsq(d_wb*sw[:,None], X*sw[:,None], rcond=None)`;
   `M_wb = A.T` (so `XYZ = M_wb @ d_wb`).
4. **Neutral pin to D50**: `M_pin = M_wb @ diag(D50_XYZ / (M_wb @ [1,1,1]))` so
   equal-RGB (post-WB) maps exactly to D50 chromaticity (`D50_XYZ` reuses
   `color_management.D50_XYZ`). This is the "well-behaved" constraint.
5. **Fold WB back in** so the stored matrix consumes *un*-white-balanced device
   RGB: `M = M_pin @ diag(gains)`. (A neutral scene, after the camera's own
   imbalance, still maps neutral because the profile applies `gains` then `M_pin`
   internally — algebraically `M·d = M_pin·(gains·d)`.)
6. `fit.matrix = M`. Columns of `M` are the D50 colorants for the ICC.

### 5.5 Quality (ΔE2000)
Predicted Lab per patch: `xyz_to_lab((M @ d)·100)`; reference Lab from the
file's `LAB_*` columns (or `xyz_to_lab(XYZ_ref)` if absent). Compute **CIEDE2000**
(`kL=kC=kH=1`, Sharma formulation, pure numpy) and **ΔE76** for cross-check;
report mean/median/95th/max over the *used* patches, plus the per-patch list
(sorted worst-first) for the results page. `xyz_to_lab` uses D50 white
`[96.422,100,82.521]`, `ε=216/24389`, `κ=24389/27`.

Indicator thresholds: **good** avg ΔE2000 < 2.0, **ok** < 4.0, **warn** ≥ 4.0 or
max > 12 (suggest re-checking corner placement / clipping / lighting).

### 5.6 Why 3×3 + identity TRC (and not more)
An ICC matrix-shaper stores only a 3×3 colorant matrix + per-channel TRCs. 3×4
(offset), polynomial, and root-polynomial fits are not representable in that
container (they would require a cLUT `mAB`/`A2B` profile, which neither our writer
nor `InputProfile` supports). Raw-linear input ⇒ TRC is identity (gamma 1.0). So
the model is fixed: **identity TRC + 3×3**. Residual black level is handled by
`rawpy`'s black subtraction at decode (optionally refine by subtracting the GS23
sample before the fit to keep the model a pure 3×3 with no offset term).

### 5.7 ICC synthesis (`build_camera_icc`)
Call `color_management.build_matrix_shaper_icc(desc, r_xyz, g_xyz, b_xyz,
trc_para=(1.0,1.0,0.0,1.0,0.0), wtpt=D50_XYZ, copyright_text=…)` where
`r_xyz,g_xyz,b_xyz = M[:,0], M[:,1], M[:,2]`. The TRC params `(g=1,a=1,b=0,c=1,
d=0)` evaluate to the identity (`_eval_para` func 3 ⇒ `y=x` for `x≥0`). `desc`
encodes the user's profile name + illuminant note (e.g.
`"FreeCCR Camera — Daylight (2026-06-17)"`). The bytes are a valid ICC v2.4
matrix-shaper profile that `InputProfile.from_bytes` parses without
`UnsupportedICCError` (verified by a unit test).

## 6. Integration points

### 6.1 `read_image` isolation params (`ccr_image.py`)
Add two optional params, both defaulting to current behaviour:
```python
def read_image(self, file_path, preview=True, max_long_side=None,
               positive_override=None, apply_input_icc=True):
    ...
    positive_mode = (self._positive_mode_active() if positive_override is None
                     else positive_override)
    ...
    # every place that currently calls self._apply_input_icc(x):
    return x if not apply_input_icc else self._apply_input_icc(x)
    # (and the positive-decode short-circuits use the resolved positive_mode)
```
Rationale: the profiling decode needs the **negative raw-linear device space with
no ICC**, irrespective of the live Positive toggle or active profile — and must
not mutate global state (avoids races with any background decode). This is a
minimal, backward-compatible change; all existing callers pass nothing and behave
identically. Unit-test that `positive_override=False, apply_input_icc=False`
yields the raw-linear array even when an input profile is active and Positive mode
is on.

### 6.2 File menu (`main_window.py:create_menu`)
After the Input-ICC actions (≈line 304), add:
```python
file_menu.addSeparator()
it8_action = file_menu.addAction("Create Camera Profile from IT8…")
it8_action.triggered.connect(self.create_camera_profile_from_it8)
```
Handler `create_camera_profile_from_it8()`:
- Construct `IT8ProfileDialog(self)`. If the currently-selected image looks like a
  plausible target the dialog offers it as the default Step-1 source (the user can
  also browse to any file).
- On accept with "apply now": call the existing
  `set_input_icc`/persist/`_reprocess_after_input_icc_change` flow
  (`main_window.set_input_icc_profile` factored to accept a path), then
  `_refresh_input_icc_menu()`. Otherwise just confirm the save path via a hint.

### 6.3 Reuse, not reinvent
- ICC bytes ⇒ `color_management.build_matrix_shaper_icc` (existing).
- Apply/persist ⇒ `ccr_backend.set_input_icc` + the existing reprocess machinery
  (existing; our file is a valid matrix-shaper, so it just works).
- Unicode paths ⇒ `normalize_unicode_path`/`validate_unicode_path` on both the
  target shot and the reference/output paths (existing utils).
- D50 constants / `xyz`↔`lab` adjacency ⇒ reuse `color_management.D50_XYZ` (add a
  small `xyz_to_lab`/ΔE helper in `it8_profile.py`; keep `color_management`
  focused on ICC I/O).

## 7. UX / wizard flow (`IT8ProfileDialog`)

**Step 1 — Target shot.** Pick the IT8 photo (default: current image; "Browse…"
for any file). Show capture-conditions guidance and a **RAW recommended** note
(warn, don't block, if a non-RAW file is chosen — the device space must match the
user's scan workflow). Decode per §5.1 on Next; show a gamma-stretched preview.

**Step 2 — Reference file.** Browse to the batch CGATS `.it8`/`.txt`. Parse and
show chart type, **batch/SERIAL**, patch count, and a "matches your chart's
printed batch number?" reminder + a link to the source. Block Next on a parse
error (friendly `IT8ReferenceError` message).

**Step 3 — Locate patches.** `IT8PatchLocator`: drag the 4 colour-grid corners;
live 288-dot overlay; **Flip 180°**; gray-strip vertical nudge. A small live
read-out shows "valid patches: N/288" and warns if GS0 darker than GS23. Sample
on Next.

**Step 4 — Build & review.** Fit (§5.4–5.5). Show avg/median/max ΔE2000 with the
pass/ok/warn chip, a worst-first per-patch list, and the count of dropped
(clipped/missing) patches. Fields: **Profile name** and **Illuminant** (free text
or a small combo: Daylight/Strobe/Tungsten/Other) folded into the ICC description.
A **Back** to re-place corners if quality is poor.

**Step 5 — Save / apply.** Save the `.icc` (default folder §4.4, default name
`Camera <illuminant> <date>`). Checkbox **"Set as input profile now"** (default
on). Finish writes the file and (if checked) activates it via the existing input-
ICC flow, then a confirmation hint.

## 8. Test plan

Unit (`tests/test_it8_profile.py`):
- **Parser**: small CGATS fixtures for both the 18-field (XYZ+Lab+density) and
  17-field (XYZ+Lab+STDEV_LAB) layouts; assert chart type, batch, dynamic column
  mapping, 288 patches keyed by id, XYZ/Lab values; tolerate CRLF, BOM, comments,
  quoted values; `IT8ReferenceError` on a truncated/empty file and on
  field/row-count mismatch.
- **Geometry**: a unit-square quad maps colour centres to expected positions; a
  known perspective quad round-trips; gray-strip ids land below row L with the 24-
  cell pitch; `Flip 180°` reverses ids correctly.
- **Sampling**: a synthesised chart image (flat patches at known RGB) sampled
  through an identity quad recovers each patch RGB; clipped patches flagged
  `valid=False`; trimmed mean rejects injected outliers.
- **Fit (synthetic round-trip — key test)**: choose a known `M_true`, generate
  device RGB for all 288 patches from the reference XYZ via `M_true⁻¹`, run
  `fit_camera_matrix`; assert recovered matrix ≈ `M_true` and **avg ΔE2000 ≈ 0**;
  neutral pin holds (equal RGB → a\*,b\* ≈ 0); clipped/missing patches excluded.
- **ΔE**: `xyz_to_lab` and `delta_e_2000` validated against published reference
  pairs (Sharma test data) within tolerance; ΔE76 cross-check.
- **ICC build→reparse**: `build_camera_icc` bytes parse via
  `InputProfile.from_bytes` with no `UnsupportedICCError`; `InputProfile.apply`
  on a synthetic neutral patch yields a near-neutral sRGB result; profile header is
  RGB/`acsp`/v2.4 matrix-shaper.
- **`read_image` isolation**: `positive_override=False, apply_input_icc=False`
  returns raw-linear even with Positive mode on and an active input profile;
  defaults unchanged (regression).

Manual:
- Full wizard on a real IT8 shot + matching batch file end-to-end; corner drag +
  flip + gray nudge; quality chip reacts to mis-placement; save; "apply now"
  reprocesses loaded images; profile re-loads on app restart (existing input-ICC
  persistence); reject a LUT/CMYK file chosen by mistake as a *reference* (clear
  error, not a crash).

## 9. Risks & mitigations
- **Wrong batch reference** (most common real failure) → show SERIAL/batch and a
  prominent "must match your chart" reminder; we cannot detect a wrong-but-valid
  file, so we surface ΔE (a wrong batch inflates it).
- **Clipped / under-exposed target** → clip rejection drops bad patches; warn when
  the lightest neutral is clipped or many patches dropped.
- **Manufacturer geometry variation** → gray-strip nudge + Flip; corners are
  user-placed so the colour block is always exact; the gray strip is the only
  derived geometry and is adjustable.
- **Non-RAW target / mismatched device space** → warn that RAW raw-linear is the
  matching device space for FreeCCR's pipeline; ΔE will reveal gross mismatch.
- **Positive-mode confusion** → the profile applies in the negative/raw-linear
  path only (existing design); documented as a non-goal, not silently inert.

## 10. Refinement (v1) — resolved decisions
1. **No ArgyllCMS.** Pure-numpy fit + existing in-process ICC writer. Decisive
   factors: AGPL licensing, per-platform binaries, install bloat, subprocess +
   Unicode-path fragility — all avoided, and the app already produces/consumes the
   exact profile class needed.
2. **Device space = raw-linear via `read_image`** (single source of truth), made
   exact and isolated by the two new optional params (no global-state mutation).
3. **Gray strip = derived geometry + nudge**, not a second mandatory quad — fewer
   handles, robust because only one block's corners are user-critical.
4. **Neutral-pin to D50** included (well-behaved profile) over a marginally lower
   raw ΔE from an unconstrained matrix.
5. **Single global input profile** reused as the apply target (matches the
   existing one-profile design); saved camera profiles are kept separately so they
   are not lost when the active profile changes.
6. **Both IT8.7/1 and IT8.7/2 accepted** (identical layout); chart type shown for
   confirmation only.

## 11. Refinement (v2) — CxF references, mirroring, non-classic grids

Driven by a real user target (LaserSoft **ISO 12641-2 "advanced"** transmissive
film target, 864 patches, batch reference in **CxF**).

### 11.1 CxF reference files (`parse_cxf_reference`)
- `parse_reference(path)` dispatches by content: XML / `.cxf` → CxF, else CGATS.
- CxF3 (ISO 17972) is XML. The parser is **namespace-tolerant** (default *or*
  `cc:` namespace — strip to local names) and **spelling-tolerant**: LaserSoft
  uses British `Colour*` (e.g. `ColourValues`, `ColourCIELab`, `ColourIEXYZ`,
  `ColourSpecification`); X-Rite uses American `Color*`. Values are matched by
  **child structure** (`L/A/B` ⇒ Lab, `X/Y/Z` ⇒ XYZ), not by tag name, so the
  spelling variance is irrelevant.
- Each value's `Colo[u]rSpecification` IDREF resolves against
  `ColorSpecificationCollection`; the D50/2° value is preferred. XYZ are Y≈100.
- Patch id = `Object/@Name`. `ObjectType` substrate/colorant/trick/measurements
  are skipped.

### 11.2 Mirrored capture
A back-lit film target shot through its base is left-right **mirrored**. The
locator gains a **"Mirrored capture"** checkbox that `np.fliplr`s the working
image (display + sampling); patch geometry then maps normally. Distinct from
**Flip 180°** (rotation), which a mirror cannot be corrected by.

### 11.3 Non-classic (plain-grid) targets — block mode
Targets that aren't the classic 12×22-colour-block-plus-GS-strip layout (e.g.
the ISO 12641-2 advanced grid, a regular 12×72 with grays as in-grid rows) use
**block mode**: a regular `rows × cols` grid the user delimits by the **top-left
and bottom-right patch ids** of the block in frame (e.g. `A49`/`L72` for one
photographed panel). `parse_block` derives the row letters + column numbers;
`block_sample_points` lays the grid on the 4-corner quad. The dialog selects
block vs classic by whether the reference has a `GS0` patch.

### 11.4 ID-agnostic, auto-neutral fit
`fit_camera_matrix` no longer assumes the classic id set or a `GS0` white patch.
It fits over whatever ids are sampled-and-in-reference, and `_pick_wb_id`
auto-selects the **lightest low-chroma (near-neutral) patch** from the reference
— working for the classic GS strip *and* in-grid grays. Sampling-window size
scales to the grid density (`ncols`/`nrows`).

### 11.5 Validated on real data
The full path (CxF parse → un-mirror → 12×24 block A49–L72 → auto-neutral fit)
on the user's LaserSoft shot yields **avg ΔE2000 ≈ 1.8** (232 of 288 patches;
dark patches clipped at the low end were rejected).

## 12. Refinement (v3) — usability: bigger window, invalid-patch feedback, multi-card targets

Driven by three field-usability gaps: the wizard was too cramped to place corners
accurately, the "valid patches: N/M" read-out never said *which* patches were
bad, and large IT8 targets that ship as **several physical cards** (the patch grid
split across panels, e.g. columns 1–24 / 25–48 / 49–72) could only contribute one
card's worth of patches to a fit.

### 12.1 Larger window
`IT8ProfileDialog` minimum size → **1024×700** (initial 1120×780) and
`IT8PatchLocator` minimum → **820×480**, so the locate page — the only page whose
accuracy depends on screen real estate — gives the patch overlay enough room to
place corners precisely. No layout restructuring; the locator already expands with
a stretch factor.

### 12.2 Invalid-patch feedback
**Validity criterion fix.** The old test flagged a patch invalid when >2 % of its
window pixels were ≤0.5 % of full scale **or** ≥99.5 %. On raw-linear device data
the low-end half was wrong: a **saturated hue legitimately reads near-zero in its
complementary channels** (a vivid red → green/blue ≈ 0, a vivid blue → red ≈ 0)
and **dense film patches read low**, so the most informative patches for the
matrix fit were all being dropped (a real ISO 12641-2 Provia shot showed only
217/288 valid, with every saturated red/blue/dark patch reddened). A low linear
value is *real signal*, not lost information. `sample_patches` now flags a patch
only when its colour truly can't be trusted:
- **highlight-clipped** — any channel has >2 % of pixels at the ceiling
  (`clip_hi`), judged **per channel** so a single blown channel is caught (and
  doesn't corrupt that primary's fit) rather than diluted across three;
- **black-crushed** — even the brightest channel's trimmed mean is ≤ `black_floor`
  (~0.08 % of full), i.e. no signal at all;
- **out of frame**.

The locate page now **surfaces which ones**:
- `IT8PatchLocator.set_invalid_ids(ids)` stores the dialog-computed invalid id set;
  `paintEvent` draws those dots (and any off-frame dot) **red and enlarged** (the
  prior code only reddened off-frame dots, never clipped-but-in-frame ones).
- `_update_locate_status` lists the invalid ids inline (first 10 + "(+N more)")
  with the **full list in the label's tooltip**, instead of only a count. The set is
  recomputed on every `changed` (i.e. on drag-release / layout change), keeping the
  red overlay in sync with the live "N/M valid" read-out.

### 12.3 Multi-card targets (accumulate samples across up to 3 images)
A target whose grid is split across cards is mapped one card per image, the
**samples accumulating** into a single fit:
- After locating a **block-mode** card, the wizard **asks (at most twice → up to 3
  cards total)** "Map another image?". On *yes* it browses + decodes the next card,
  loads it into the locator, and **auto-advances the block id range by the block's
  own width** (`it8.next_block_ids`, e.g. `A1–L24` → `A25–L48` → `A49–L72`, clamped
  to the reference's max column). On *no* it fits.
- `it8.merge_samples([...])` (new, pure) unions the per-card sample dicts; on the
  rare overlapping-id collision a **valid** sample wins over an invalid one.
- The fit consumes the merged set (`_all_samples`); Step 4 notes "combined from N
  images" when N>1. A "Card N of up to 3" banner shows on the locate page.
- **Classic** (single 12×22 + GS strip) targets are one physical card, so the
  prompt is **gated to block mode** — classic users never see it.
- New pure-logic helpers live in `core.it8_profile` (`merge_samples`,
  `next_block_ids`) and are unit-tested; the dialog stays thin UI.

### 12.4 Multi-card session robustness (from adversarial review)
The accumulation state (`_mapped_cards` / `_all_samples` / `_n_images` / `_fit`)
belongs to the **(target shot, reference)** pair the cards were sampled against,
so:
- **`_reset_card_accum()` is called from `_set_target` and `_browse_ref`** —
  picking a different target or a different batch reference (incl. via *Back*)
  discards any cards mapped against the abandoned inputs and re-seeds the block id
  range from the new reference's extent. Without this, stale cards from an
  abandoned session would silently contaminate the fit (and falsely report
  "combined from N images").
- **`_should_ask_more()` is additionally gated on `self._fit is None`** so that
  *Back-from-build → Next* simply re-runs the fit rather than re-popping the "Map
  another?" prompt. A genuine new card is added during the forward mapping flow,
  before any fit exists.
- **`_advance_block_ids` refuses a degenerate shift** — if the previous card
  already reached the chart's edge, the auto-advance would collapse to a single
  column or overlap, so it leaves the ids for the user to set by hand (with an
  explanatory note) instead of silently mis-mapping the next card.
- **The window floor is clamped to the screen** (`availableGeometry`) so the
  1024×700 minimum can never push the Back/Next row off a small display.

### 12.5 Canonical camera-profiling decode (researched)

The standard device space for **all** camera colour-characterization profiles —
matrix ICC, cLUT ICC, and Adobe DCP — is **camera-native, demosaiced, linear,
16-bit, no auto-brightness, no clipping in the patches**. The reference is Anders
Torger's DCamProf recipe `dcraw -v -r 1 1 1 1 -o 0 -H 0 -T -6 -W -g 1 1`
("16-bit linear TIFF without white balancing"). Sources: torger.se/DCamProf, the
dcraw(1) manpage, the Adobe DNG spec, rawpy/LibRaw docs, ninedegreesbelow (Elle
Stone), RawPedia.

rawpy mapping (FreeCCR's `_raw_color_postprocess_kwargs(no_icc_default=False)`):

| dcraw | meaning | rawpy |
|-------|---------|-------|
| `-o 0` | camera-native (no camera matrix / working space) | `output_color=ColorSpace.raw` |
| `-g 1 1` | linear gamma 1.0 | `gamma=(1, 1)` |
| `-6` | 16-bit | `output_bps=16` |
| `-W` | fixed white level (no auto-bright) | `no_auto_bright=True` + `no_auto_scale=True` + manual `×65535/white_level` |
| `-r 1 1 1 1` | unbalanced (unity WB) | `use_camera_wb=False`, `use_auto_wb=False` |
| `-H 0` | no highlight recovery | (clip rejection drops blown patches in the fit) |

**Why camera-native:** `-o 0` outputs sensor RGB *before* `convert_to_rgb()` (the
camera→XYZ/working-space matrix stage). Fitting on Adobe RGB would make the matrix
characterize Adobe, not the sensor, and — critically — a **DCP's
ColorMatrix/ForwardMatrix operate on camera-native raw**, so a working-space decode
makes DCP application impossible. This is why the decode is camera-native, not the
v0.5/early-v0.6 Adobe-RGB-for-ICC decode.

**White balance — the one divergence (researched):**
- **DCP must be UNBALANCED**: it carries WB derivation (the `ColorMatrix` relates
  raw RGB balance ↔ illuminant CCT) and applies the as-shot WB at render time via
  the `ForwardMatrix`(+LUT) on the white-balanced image.
- **Matrix/cLUT ICC is normally white-balanced at decode** (no render-time WB
  stage) — *but may be fit unbalanced with WB baked into the matrix coefficients.*
  **FreeCCR does the latter**: the decode is unbalanced (no WB) and
  `fit_camera_matrix` folds the per-channel `gains` into the stored matrix
  (§5.4 step 5). So FreeCCR's single unbalanced camera-native decode is the correct
  shared base for both the matrix ICC **and** a future DCP.

> **Superseded twice — see the later specs.** `spec/camera-profile-color-fix.md`
> reverted the "fold WB into the matrix" rule: both containers now fit and apply on
> **white-balanced** data (`CameraFit.wb_mult` holds the balance separately), which
> is what ICC/DNG conventions and RawTherapee expect. Then
> `spec/camera-profile-calibration-wb.md` fixed *which* balance apply uses: a
> generated profile records the chart's own camera-native neutral (`AsShotNeutral`
> for DCP, the private `CCRn` tag for ICC) and that neutral owns the WB, so a
> profile built for one fixed copy-stand setup renders every frame identically
> instead of tracking each frame's (possibly auto-WB) metadata. The decode stays
> unbalanced and camera-native throughout — only the source of the balance changed.

**Implementation:** `read_image` selects the camera-native path
(`no_icc_default=False`) whenever an input ICC is active *or* the bare-device
profiling decode is requested (`apply_input_icc=False`), and the Adobe RGB +
auto-scale default only for a plain unprofiled negative. `decode_target` (the IT8
fit) and the runtime ICC-apply decode therefore hit the **same** kwargs —
`test_fit_and_apply_decode_spaces_are_identical` pins this so the two paths can't
silently diverge.

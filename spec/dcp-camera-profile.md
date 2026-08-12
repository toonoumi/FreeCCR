# Spec: Adobe DCP (DNG Camera Profile) Support — Consume & Generate

Status: IMPLEMENTED (on branch `feature/clut-icc-support`)
Owner: FreeCCR
Feature branch: `feature/dcp-camera-profile`

> Supersedes the earlier never-committed DCP draft (memory:
> `camera-profile-feature`). This is the authoritative on-disk spec.

## 1. Summary

Add **Adobe DCP** (`.dcp` — a DNG camera-profile binary) support to FreeCCR, in
two directions that mirror the existing input-ICC feature:

- **CONSUME** — `File ▸ Load DCP Profile…` loads an external/Adobe `.dcp`,
  parses it (in pure-numpy, no `rawpy`-side DNG plumbing), and applies it to
  every decoded scan in the same decode-time slot the input ICC uses, producing
  **linear Adobe RGB** so the negative inversion downstream reads consistent
  `-log10(value/full)` linear data (identical contract to
  `InputProfile.apply`, see `color_management.py` §"Input ICC profile").
- **GENERATE** — the existing **IT8 wizard** (`it8_profile_dialog`) gains a
  **save format choice (`.icc` | `.dcp`)**. The fitted camera matrix
  (`CameraFit.matrix`, camera-native device RGB[0,1] → XYZ D50) is written as a
  minimal valid matrix DCP (`ProfileName`, `CalibrationIlluminant1`,
  `ColorMatrix1`, `ForwardMatrix1`).

A DCP is the **right** container for FreeCCR's camera-native decode: its
`ColorMatrix`/`ForwardMatrix` are *defined on camera-native raw* — exactly the
space `read_image` already produces for the ICC path
(`output_color=raw`, `gamma=(1,1)`, `no_auto_bright`, `use_camera_wb=False`,
`no_auto_scale` + manual white-level scale — `_raw_color_postprocess_kwargs(
no_icc_default=False)`, see `spec/it8-camera-profile.md` §12.5). The one
divergence from the ICC path is **white balance**: a matrix/cLUT ICC bakes WB
into the fitted matrix, whereas a DCP is fed **unbalanced** raw and applies the
**as-shot WB at render time** (`raw.camera_whitebalance`). FreeCCR's decode is
already unbalanced, so it is the correct shared base; the DCP apply path just
needs the as-shot multipliers threaded through (new plumbing, since
`InputProfile.apply` is WB-agnostic).

The whole feature is **pure numpy + struct** (both present), reusing the
binary-IO idioms already in `color_management.py` (`struct`, big/little-endian
unpacking, `_s15f16`-style fixed-point) and the colour-science helpers in
`it8_profile.py` (`xyz_to_lab`, `delta_e_2000`, `D50_XYZ`). **No** `rawpy` DNG
metadata API, `exiftool`, `libdng`, or DNG SDK dependency is added.

## 2. Goals / Non-goals

### Goals
- **Parse** a `.dcp` (TIFF/DNG-IFD binary) into a `DcpProfile` dataclass:
  matrices (`ColorMatrix1/2`, `ForwardMatrix1/2`, `CameraCalibration1/2`,
  `ReductionMatrix1/2`), `CalibrationIlluminant1/2`, optional `HueSatMapData1/2`
  (+ dims/encoding), optional `LookTableData` (+ dims/encoding), optional
  `ProfileToneCurve`, `ProfileName`, `ProfileEmbedPolicy`. Handle **II and MM**
  endianness.
- **Apply** a `DcpProfile` to camera-native unbalanced raw → **linear Adobe
  RGB**, with: as-shot WB; dual-illuminant CCT interpolation of
  Color/Forward(/HueSat) matrices in **mired (1/CCT)** space; ForwardMatrix →
  XYZ D50 (with the inverse-ColorMatrix fallback when no ForwardMatrix);
  optional HueSatMap/LookTable/ToneCurve (the subjective "look").
- A **colorimetric mode (matrices only) by default**, with the look tables
  (HueSatMap/LookTable/ToneCurve) **off by default** and behind a checkbox —
  they are subjective Adobe "look", not colorimetry, and the negative pipeline
  wants a faithful colour map (see §5.7).
- **Generate** a minimal matrix `.dcp` from the IT8 `CameraFit`
  (`build_camera_dcp(...)`), single-illuminant.
- A **global active-DCP slot** mirroring
  `color_management.get/set_active_input_profile`, **mutually exclusive** with
  the active input ICC.
- `read_image` applies **either** an ICC `InputProfile` **or** a `DcpProfile`,
  threading the as-shot WB for the DCP path; keeps the positive-mode skip and
  the camera-native decode.
- Unit tests: parse a self-built synthetic `.dcp`, endianness, SRATIONAL
  matrices, dual-illuminant interpolation weights, ForwardMatrix vs
  ColorMatrix-fallback, generate→reparse matrix round-trip, neutral→neutral, and
  synthetic-camera round-trip.

### Non-goals
- **No `rawpy`/`libraw` DCP application.** rawpy does not apply DCPs; we apply in
  numpy on the already-decoded camera-native array.
- **No DCP authoring beyond a minimal single-illuminant matrix profile.**
  Generating HueSatMap/LookTable/ToneCurve, dual-illuminant DCPs (needs two
  charts under two lights), or `CameraCalibration` is out of scope for GENERATE
  (a fitted IT8 matrix has none of those). Dual-illuminant is noted as future
  work (§5.8).
- **No tone/look "rendering intent" emulation of Camera Raw.** We apply the
  table operators faithfully when enabled, but do not reproduce ACR's default
  tone/exposure baseline (`BaselineExposureOffset`, `DefaultBlackRender`) — the
  negative pipeline owns tone. We **read** those tags for completeness but do not
  apply them by default (§5.7).
- **No `.dng`-embedded profile extraction** (reading the profile out of a raw
  file's IFD). Only standalone `.dcp` files. (The parser is IFD-generic, so this
  is a small later add.)
- **No simultaneous ICC + DCP.** They occupy the same decode slot; setting one
  clears the other (§6.3).

## 3. Background (researched & verified)

### 3.1 DCP container = TIFF/DNG IFD
A `.dcp` is a single-IFD TIFF stream:
- **8-byte header**: byte-order (`II`=0x4949 little / `MM`=0x4D4D big),
  magic `42`, then `u32` offset to IFD0.
- **IFD**: `u16` entry count, then N × 12-byte entries
  `(tag:u16, type:u16, count:u32, value_or_offset:u32)`. If
  `type_size × count ≤ 4` the value is **inline** in the last field (left-
  justified); else the field is a **file offset** to the data. After the entries,
  a `u32` next-IFD offset (0 for a DCP).
- **TIFF types** used here: `SHORT`=3 (2 B), `LONG`=4 (4 B), `RATIONAL`=5
  (2×u32), `ASCII`=2 (1 B), `SRATIONAL`=10 (2×i32), `FLOAT`=11 (4 B IEEE).
  All multi-byte fields honour the header endianness.

### 3.2 Relevant DNG profile tags (verified)
Verified against the Adobe DNG Specification 1.4–1.6 and the colour-hdri /
rawler tag enumerations (see Sources). Decimal tag id → (TIFF type, count):

| Tag (dec) | Name | Type | Count | Meaning |
|---|---|---|---|---|
| 50721 | `ColorMatrix1` | SRATIONAL | 3×planes (9) | XYZ(D50-ref-adapted) → camera-native reference RGB, illuminant 1 |
| 50722 | `ColorMatrix2` | SRATIONAL | 9 | …illuminant 2 |
| 50723 | `CameraCalibration1` | SRATIONAL | planes² (9) | per-unit camera calibration, illuminant 1 (identity if absent) |
| 50724 | `CameraCalibration2` | SRATIONAL | 9 | …illuminant 2 |
| 50725 | `ReductionMatrix1` | SRATIONAL | 9 | (≥3-plane reduction; unused for RGB) |
| 50726 | `ReductionMatrix2` | SRATIONAL | 9 | |
| 50727 | `AnalogBalance` | RATIONAL | planes (3) | diag analog gains (identity if absent) |
| 50728 | `AsShotNeutral` | RATIONAL | planes (3) | neutral in camera RGB. **Written by `build_camera_dcp` as the profile's calibration neutral and, when present, it OWNS the WB at apply time** (see `spec/camera-profile-calibration-wb.md`); absent ⇒ fall back to the frame's `raw.camera_whitebalance` |
| 50778 | `CalibrationIlluminant1` | SHORT | 1 | EXIF LightSource enum for matrix set 1 |
| 50779 | `CalibrationIlluminant2` | SHORT | 1 | …set 2 |
| 50936 | `ProfileName` | ASCII | n | display name |
| 50937 | `ProfileHueSatMapDims` | LONG | 3 | (hueDiv, satDiv, valDiv) |
| 50938 | `ProfileHueSatMapData1` | FLOAT | 3×∏dims | (Δhue°, satScale, valScale) per cell, illuminant 1 |
| 50939 | `ProfileHueSatMapData2` | FLOAT | 3×∏dims | …illuminant 2 |
| 50940 | `ProfileToneCurve` | FLOAT | 2×N | interleaved (x,y) ∈[0,1] control points |
| 50941 | `ProfileEmbedPolicy` | LONG | 1 | 0=allow copying … 3=no embed |
| 50981 | `ProfileLookTableDims` | LONG | 3 | (hueDiv, satDiv, valDiv) |
| 50982 | `ProfileLookTableData` | FLOAT | 3×∏dims | (Δhue°, satScale, valScale) per cell |
| 51107 | `ProfileHueSatMapEncoding` | LONG | 1 | 0=linear, 1=sRGB (value-axis encoding) |
| 51108 | `ProfileLookTableEncoding` | LONG | 1 | 0=linear, 1=sRGB |
| 50964 | `ForwardMatrix1` | SRATIONAL | 9 | **white-balanced** camera-native RGB → XYZ **D50**, illuminant 1 |
| 50965 | `ForwardMatrix2` | SRATIONAL | 9 | …illuminant 2 |
| 51109 | `BaselineExposureOffset` | SRATIONAL | 1 | read-only (not applied by default) |
| 51110 | `DefaultBlackRender` | LONG | 1 | read-only (not applied by default) |

`CalibrationIlluminant` enum (EXIF LightSource): **17 = Standard light A**
(≈2856 K), **20 = D55** (5503 K), **21 = D65** (6504 K), **22 = D75** (7504 K),
**23 = D50** (5003 K), 1 = Daylight, 2 = Fluorescent, 3 = Tungsten. A typical
Adobe dual profile is illuminant1=A(17), illuminant2=D65(21).

### 3.3 HueSatMap / LookTable layout (verified)
A 3-D table over **HSV**: `dims = (hueDivisions, satDivisions, valDivisions)`.
Data is `FLOAT`, **`3 × hueDiv × satDiv × valDiv`** entries, **value-axis
fastest, then sat, then hue** (DNG ordering: index =
`((h·satDiv) + s)·valDiv + v`), each cell a triple
`(hueShift_degrees, satScale, valScale)`. When `valDiv == 1` the table is
2-D (hue×sat). `…Encoding` says whether the **value** coordinate is sampled in
linear (0) or sRGB (1) space before lookup. Applied by converting working RGB →
HSV, trilinearly interpolating the table at (h,s,v), then `h += Δhue`,
`s *= satScale`, `v *= valScale`, → RGB. HueSatMap is applied in the
**ProPhoto-RGB-ish** linear/HSV reference Adobe uses; we apply it in linear
ProPhoto (D50) for fidelity (§5.5).

### 3.4 Dual-illuminant interpolation (verified — mired space)
For an as-shot neutral with correlated colour temperature `T` (kelvin), Adobe
interpolates the two matrix sets in **reciprocal (mired) space**:
```
inv(T) = 1e6 / T
w = clamp( (inv(T) - inv(T2)) / (inv(T1) - inv(T2)), 0, 1 )     # weight on set 1
M = w · M1 + (1 - w) · M2          # applied to ColorMatrix, ForwardMatrix, CameraCalibration, HueSatMap
```
with `T1 = CCT(illuminant1)`, `T2 = CCT(illuminant2)` (e.g. A≈2856, D65≈6504).
`T` itself is derived from the as-shot camera neutral (§5.3). With one
illuminant present, `w = 1` and only set 1 is used.

### 3.5 White-balance divergence (verified, load-bearing)
- A DCP's `ColorMatrix` relates camera-raw balance ↔ illuminant; the profile
  applies the **as-shot WB at render time** via `ForwardMatrix` on the
  white-balanced image. ⇒ a DCP **must be fed UNBALANCED raw**.
- FreeCCR's decode is already unbalanced (`use_camera_wb=False`,
  `use_auto_wb=False`). The as-shot multipliers come from
  `raw.camera_whitebalance` (length-4 `[R, G1, B, G2]`; `G2` is usually `0` or
  `== G1`). These must be threaded from `read_image` into the DCP apply (new
  plumbing) since `_apply_input_icc` / `InputProfile.apply` are WB-agnostic.

## 4. Data model & files

### 4.1 New module `src/core/dcp_profile.py` (pure logic, no Qt)
```python
class DcpError(Exception): ...        # parse/validate failures, surfaced to the user

# Illuminant enum -> CCT kelvin (StdA, D50/55/65/75, daylight/tungsten).
ILLUMINANT_CCT: dict[int, float]      # {17:2856, 20:5503, 21:6504, 22:7504, 23:5003, ...}

@dataclass
class DcpProfile:
    name: str
    color_matrix_1: np.ndarray                 # (3,3) XYZ->camera, illum 1
    color_matrix_2: Optional[np.ndarray]
    forward_matrix_1: Optional[np.ndarray]     # (3,3) wb-cam->XYZ D50, illum 1
    forward_matrix_2: Optional[np.ndarray]
    camera_calibration_1: np.ndarray           # (3,3), identity if absent
    camera_calibration_2: np.ndarray
    analog_balance: np.ndarray                 # (3,) diag, ones if absent
    illuminant_1: int                          # EXIF LightSource enum (default 21=D65)
    illuminant_2: Optional[int]
    hsm_dims_1: Optional[tuple]; hsm_data_1: Optional[np.ndarray]   # (h,s,v,3)
    hsm_dims_2: Optional[tuple]; hsm_data_2: Optional[np.ndarray]
    hsm_encoding: int                          # 0 linear / 1 sRGB
    look_dims: Optional[tuple]; look_data: Optional[np.ndarray]
    look_encoding: int
    tone_curve: Optional[np.ndarray]           # (N,2) (x,y) in [0,1]
    embed_policy: int

    @property
    def has_forward(self) -> bool: ...
    @property
    def is_dual(self) -> bool: ...

# --- parse ---
def parse_dcp(path: str) -> DcpProfile         # raises DcpError
def parse_dcp_bytes(data: bytes) -> DcpProfile

# --- apply (camera-native unbalanced raw in, linear Adobe RGB out) ---
def apply_dcp(profile: DcpProfile, rgb_u16: np.ndarray, *,
              as_shot_wb: Optional[np.ndarray],   # length-3 camera-native multipliers
              apply_look: bool = False) -> np.ndarray

# --- generate (matrix DCP from an IT8 CameraFit) ---
def build_camera_dcp(fit, name: str, *,
                     illuminant: int = 23,        # 23 = D50
                     wb: Optional[np.ndarray] = None) -> bytes
```

### 4.2 Global active-DCP slot — in `color_management.py`
Mirror the existing input-profile module globals (kept here so `ccr_image` reads
the active colour transform from one module without importing the backend, as it
already does for `get_active_input_profile`):
```python
_active_dcp_profile: "Optional[DcpProfile]" = None
def set_active_dcp_profile(p) -> None
def get_active_dcp_profile()
```
`DcpProfile` lives in `dcp_profile.py`; `color_management` stores it opaquely
(no import cycle — it only holds/returns the object).

### 4.3 Edits
- `src/core/ccr_image.py` — `read_image`: select the DCP apply path when a DCP is
  active (mutually exclusive with the ICC path), threading `raw.camera_whitebalance`
  (§6.1). `_input_icc_will_apply` generalised to
  `_camera_profile_will_apply` (ICC **or** DCP active) so the camera-native decode
  is chosen for both.
- `src/core/ccr_backend.py` — `set_input_dcp(path)`, `load_input_dcp_from_storage`,
  `clear_input_dcp` mirroring the ICC trio; setting a DCP clears the active ICC and
  vice-versa (§6.3). New `input_dcp_path`/`input_dcp_name` persisted state.
- `src/ui/main_window.py` — File-menu `Load DCP Profile…` /
  `Clear DCP Profile`, handler `set_input_dcp_profile()` reusing the
  `_apply_input_icc_path` reprocess flow (factored to a shared
  `_apply_camera_profile`). The IT8 wizard's save step (`it8_profile_dialog`)
  gains the `.icc`/`.dcp` format choice.
- `src/widgets/it8_profile_dialog.py` — Step 5 file-type filter + branch to
  `dcp_profile.build_camera_dcp` vs `it8_profile.build_camera_icc`.

### 4.4 Storage
A DCP working copy lives next to the input-ICC copy:
`<appdata>/input_profile.dcp` (mirror `_input_icc_storage_path`). The persisted
QSettings key is `import/input_dcp_path`; on startup the saved DCP is restored if
present (and clears any saved ICC, since they're exclusive). Generated camera
DCPs are saved to the same per-user `camera_profiles/` folder the ICC wizard
uses.

## 5. Processing / math (in `dcp_profile.py`)

### 5.1 IFD parsing (`parse_dcp_bytes`)
- Read `byte_order` from bytes 0:2 → struct prefix `<`/`>`; validate magic `42`;
  read IFD0 offset.
- Read entry count, iterate 12-byte entries. A small `_read_values(tag_entry)`
  resolves inline-vs-offset by `type_size × count`, slices the bytes, and unpacks
  per type (`SRATIONAL` → `i32/i32` pairs → float; `RATIONAL` → `u32/u32`;
  `FLOAT` → `f`; `SHORT`/`LONG` → int; `ASCII` → NUL-trimmed str). All with the
  header endianness.
- Assemble matrices as `(3,3)` row-major (DNG stores row-major,
  count `3×planes`). Defaults: `CameraCalibration*` → I₃, `AnalogBalance` → ones,
  `illuminant_1` → 21 (D65) if absent.
- `DcpError` on: bad byte-order/magic, truncated IFD, an offset past EOF, a
  matrix tag whose count ≠ 9, or no `ColorMatrix1` (a profile with no CM1 and no
  ForwardMatrix is unusable).
- HueSatMap/LookTable: reshape `data` to `(hueDiv, satDiv, valDiv, 3)` per the
  DNG ordering (value fastest). Validate `len == 3·∏dims`.

### 5.2 As-shot white balance
`as_shot_wb` is the length-3 camera-native neutral *multiplier*
(`m = [mR, mG, mB]`, normalised so `min(m)=1` or `mG=1`, matching DNG's
"multiply raw by these to neutralise"). Threaded from
`raw.camera_whitebalance[:3]` (replace a `0` `G` with `1.0`). Apply
`d_wb = clip(d · m, 0, 1)` on normalised camera RGB `d = rgb_u16/65535`.
(Adobe applies WB *before* the ForwardMatrix; ForwardMatrix is defined on the
white-balanced image — §3.5.)

### 5.3 Deriving the working CCT (dual-illuminant only)
The as-shot neutral in *camera* space is `n = 1/m` (the camera RGB that WB maps
to grey). To pick the interpolation weight we need its CCT. Use the
**single-step approximation** Adobe's reference uses for our purposes:
- Map `n` to XYZ with each illuminant's `ColorMatrix` inverse:
  `XYZ_i = inv(ColorMatrix_i) · n`; convert to xy; compute CCT via
  McCamy's approximation `CCT = 449·t³ + 3525·t² + 6823.3·t + 5520.33`,
  `t = (x − 0.3320)/(0.1858 − y)`.
- Average the two estimates (or iterate once) → `T`. This only chooses the
  blend weight; small CCT error ⇒ negligible matrix error. With one illuminant,
  skip entirely (`w=1`).

### 5.4 Matrix interpolation & forward map
```
inv = lambda T: 1e6 / T
w = clip((inv(T) - inv(T2)) / (inv(T1) - inv(T2)), 0, 1)   # 1.0 if single-illuminant
FM = w·FM1 + (1-w)·FM2        # if forward matrices present
CC = w·CC1 + (1-w)·CC2
AB = diag(analog_balance)
```
**Forward path (preferred — ForwardMatrix present):**
```
XYZ_D50 = FM · inv(CC · AB) · d_wb
```
(`ForwardMatrix` already targets D50, so no further adaptation.) In the common
case `CC=I`, `AB=I` this is just `XYZ_D50 = FM · d_wb`.

**Fallback (no ForwardMatrix):** invert the (interpolated) `ColorMatrix`, which
maps XYZ(at the calibration illuminant's adaptation) → camera RGB, and apply it
to the **un-white-balanced** camera RGB, then Bradford-adapt that illuminant's
white to D50:
```
CM = w·CM1 + (1-w)·CM2
XYZ_ill = inv(CM · AB) · d            # d = unbalanced normalised camera RGB
XYZ_D50 = Bradford(white_ill -> D50) · XYZ_ill
```
where `white_ill` is the chosen illuminant's XYZ (from `ILLUMINANT_CCT` →
daylight/blackbody xy → XYZ). Bradford reuses
`color_management.M_BRADFORD_*`. This fallback is colorimetrically weaker (no
WB-aware forward map) but valid for matrix-only DCPs without a ForwardMatrix.

### 5.5 Optional "look" stage (only when `apply_look=True`)
Operate on `XYZ_D50`:
1. **HueSatMap** (interpolated by `w`): `XYZ_D50 → linear ProPhoto (D50)`
   (`color_management.M_XYZ2PROPHOTO`), → HSV, trilinear table lookup at
   `(h, s, v_enc)` (`v_enc` = `srgb_encode(v)` if `hsm_encoding==1` else `v`),
   apply `h+=Δh, s*=sScale, v*=vScale`, → RGB → XYZ_D50.
2. **LookTable**: same operator with `look_data`/`look_encoding`.
3. **ProfileToneCurve**: build a monotone LUT from the `(x,y)` control points
   (reuse the Fritsch–Carlson interpolation already specced for curves), apply on
   the ProPhoto-encoded value channel (Adobe applies the tone curve on the RGB
   channels in the profile's working space). For the negative pipeline this is
   tone, so it is **off by default**.

### 5.6 Output: linear Adobe RGB (load-bearing)
Final step for **all** paths (matrices-only and look):
```
adobe_lin = M_XYZ_D50_2_ADOBE · XYZ_D50          # reuse color_management constant
out_u16 = rint(clip(adobe_lin, 0, 1) · 65535)
```
This is the **exact** space the no-ICC negative decode and `InputProfile.apply`
produce (linear Adobe RGB, **no sRGB OETF**), so the density inversion
(`-log10(value/full)`) reads consistent linear data. Emitting gamma-encoded data
here would cast the converted negative (the same hazard documented on
`InputProfile.apply`).

### 5.7 Look vs colorimetric — recommendation
**Default = colorimetric (matrices only).** HueSatMap/LookTable/ToneCurve are
Adobe's *subjective camera look* (skin-tone tweaks, film-like tone), authored for
positive rendering. For a negative scan FreeCCR wants a *faithful* device→XYZ
map; the look tables would bend hues and tone *before* inversion, fighting the
v0.2.3 cast-balance and reintroducing exactly the kind of tone-warping the parked
density experiment showed is fragile (see `CLAUDE.md`). So:
- `apply_look=False` by default; a **"Apply Adobe look tables"** checkbox in the
  Load-DCP confirmation (and persisted per-DCP) enables them for users who
  deliberately want the camera-maker look.
- `BaselineExposureOffset`/`DefaultBlackRender` are parsed for display but never
  applied — tone is owned by the negative pipeline + sliders.

### 5.8 GENERATE — `build_camera_dcp(fit, name, illuminant=23, wb=None)`
The IT8 fit gives `M`: camera-native device RGB[0,1] → XYZ D50 (Y≈1), already
D50-pinned and WB-folded-in (`it8_profile.fit_camera_matrix` §5.4). A DCP's
**ForwardMatrix is defined on white-balanced camera RGB**, so:
1. **Derive the WB the DCP assumes.** The camera neutral is the camera RGB that
   maps to D50 grey: `n = inv(M) · D50_XYZ`; the DCP WB multiplier is
   `m = n.max()/n` (so `m·n` is equal-RGB). `AsShotNeutral = n/n.max()`.
2. **ForwardMatrix** = `M · diag(1/m)` — i.e. `M` re-expressed to consume the
   **white-balanced** camera RGB `d_wb = m·d` (`FM·d_wb = M·diag(1/m)·m·d = M·d`).
   Stored as `ForwardMatrix1` (SRATIONAL, count 9, row-major). `CameraCalibration1`
   and `AnalogBalance` are omitted (⇒ identity), so apply reduces to `XYZ=FM·d_wb`,
   exactly reproducing the ICC.
3. **ColorMatrix1** (required) = chromatic-adaptation-aware inverse mapping
   XYZ(at the calibration illuminant) → camera-native reference RGB:
   `CM = inv( M · diag(1/m) ) · Bradford(D50 -> white_ill)`. (`CM·XYZ_ill` gives
   the WB camera RGB; for a single-illuminant D50 profile `white_ill = D50` so the
   Bradford term is I and `CM = inv(FM)`.) This satisfies the DNG requirement that
   a profile carry a `ColorMatrix1` even when a ForwardMatrix is present.
4. **Required tag set** for a minimal valid matrix DCP (Adobe):
   `ProfileName` (50936, ASCII), `CalibrationIlluminant1` (50778, SHORT — default
   23=D50, or the wizard's chosen light), `ColorMatrix1` (50721, SRATIONAL×9),
   `ForwardMatrix1` (50964, SRATIONAL×9). Also write `ProfileEmbedPolicy`
   (50941 = 1, "allow copying without restriction") and — **superseding this
   spec's original `AsShotNeutral`-free rule** — `AsShotNeutral` (50728,
   RATIONAL×3) = the chart's camera-native neutral `1/wb`, so the profile owns the
   white balance of the fixed setup it was calibrated on instead of deferring to
   each frame's metadata. The tag is inert in Adobe/RawTherapee (both read only
   profile tags); `build_camera_dcp(bake_neutral=False)` omits it. See
   `spec/camera-profile-calibration-wb.md`. **Single-illuminant only**
   — a dual-illuminant DCP needs two charts shot under two lights (out of scope,
   §2). Note this in the wizard.
5. **Write the IFD** (`_build_dcp_bytes`): little-endian (`II`), magic 42, IFD0 at
   offset 8; tags **sorted ascending** (TIFF requirement); inline values for
   `SHORT`/`LONG`/short `ASCII` ≤ 4 B, else appended to the data region (offsets
   4-byte aligned); SRATIONAL matrices encoded as `i32` num/den via a
   `_to_srational(x)` (denominator `10000` or a reduced fraction). Mirrors the
   `build_matrix_shaper_icc` byte-layout idiom.

## 6. Integration points

### 6.1 `read_image` (`ccr_image.py`)
Camera-native decode is already chosen whenever a profile is in play. Generalise:
```python
def _camera_profile_will_apply(self) -> bool:
    return (color_management.get_active_input_profile() is not None
            or color_management.get_active_dcp_profile() is not None)
# icc_device_space = (not apply_input_icc or self._camera_profile_will_apply())
```
The DCP needs the **as-shot WB**, which only exists in the RAW branch
(`raw.camera_whitebalance`). Capture it inside the `with rawpy.imread(...)` block
(`wb = np.asarray(raw.camera_whitebalance[:3], float)`, replace 0→1) and apply:
```python
# replaces the single self._apply_input_icc(rgb) tail in the RAW branch
if positive_decode or not apply_input_icc:
    return rgb
dcp = color_management.get_active_dcp_profile()
if dcp is not None:
    return self._apply_input_dcp(rgb, wb)        # new: dcp_profile.apply_dcp(...)
return self._apply_input_icc(rgb)
```
`_apply_input_dcp(arr, wb)` mirrors `_apply_input_icc` (try/except → log + return
input on failure, no-op when no DCP). **Non-RAW files have no as-shot WB**; a DCP
is a camera-raw profile, so on the non-RAW path `_apply_input_dcp` is called with
`as_shot_wb=None` and `apply_dcp` falls back to `m=ones` (the profile still maps
device→Adobe, just unbalanced — acceptable, and rare; surfaced as a warning when
a DCP is active and a non-RAW image is loaded).

`reprocess_all_for_input_icc_change` already full-resets every image; the DCP
set/clear reuses it (decode changes ⇒ full reset), so the parked-conversion
hazard is handled identically.

### 6.2 Backend (`ccr_backend.py`)
`set_input_dcp(src_path)` parses (`dcp_profile.parse_dcp`, raising `DcpError`
before mutating), copies to `<appdata>/input_profile.dcp`, calls
`color_management.set_active_dcp_profile(profile)` **and**
`set_active_input_profile(None)` + clears the ICC storage (exclusivity),
sets `input_dcp_path`/`input_dcp_name = profile.name`. `clear_input_dcp` and
`load_input_dcp_from_storage` mirror the ICC versions. Symmetrically,
`set_input_icc` clears the active DCP.

### 6.3 Mutual exclusivity & menu
The File menu shows both "Set Input ICC Profile…" and "Load DCP Profile…"; the
active one is reflected in its label (`_refresh_input_icc_menu` generalised to
show whichever of ICC/DCP is active, and both Clear actions). Setting either
clears the other (a single global "input colour transform" slot with two
possible kinds). Startup restore prefers whichever path is persisted (ICC and
DCP keys are never both set because set-one-clears-other also clears the other's
QSettings key).

### 6.4 Reuse, not reinvent
- Binary IO idioms (`struct`, fixed-point, IFD/offset table) ⇐ pattern of
  `build_matrix_shaper_icc` / `_read_tag_table` in `color_management.py`.
- `M_XYZ_D50_2_ADOBE`, `M_XYZ2PROPHOTO`, `M_BRADFORD_*`, `D50_XYZ`,
  `srgb_encode` ⇐ `color_management.py` (the apply output + look encodings).
- `xyz_to_lab`, `delta_e_2000`, monotone-curve LUT ⇐ `it8_profile.py` /
  `ccr_processor.build_channel_lut` (tone curve + tests).
- The reprocess/persist/menu plumbing ⇐ the existing input-ICC flow
  (`_apply_input_icc_path`, `reprocess_all_for_input_icc_change`).
- Unicode paths ⇐ `normalize_unicode_path`/`validate_unicode_path` on the `.dcp`
  load + save paths.

## 7. UX

**Consume.** `File ▸ Load DCP Profile…` → file picker (`*.dcp`). On load: parse,
show a small confirmation (profile name, single/dual-illuminant, whether it has a
ForwardMatrix, and whether HueSatMap/LookTable/ToneCurve are present), with an
**"Apply Adobe look tables"** checkbox (default **off**, §5.7). Accept →
`set_input_dcp` + reprocess (reuses the input-ICC progress/reset machinery).
A `DcpError` shows a friendly message (mirrors `UnsupportedICCError` handling).
If a DCP is set while an ICC is active (or vice-versa), a hint notes the other
was cleared. A warning hint appears if a DCP is active and a **non-RAW** image is
loaded (no as-shot WB).

**Generate.** The IT8 wizard's Step 5 (Save/apply) "Save profile" dialog gains a
file-type selector **"ICC profile (*.icc)"** / **"DNG camera profile (*.dcp)"**.
Choosing `.dcp` writes `build_camera_dcp(fit, name, illuminant=<the wizard's
illuminant choice mapped to a LightSource enum>)`. A note states the DCP is
**single-illuminant** (valid under the shot light) and that **"Set as profile
now"** activates it via the DCP slot (clearing any active ICC). Default save name
`Camera <illuminant> <date>.dcp`.

## 8. Test plan

Unit (`tests/test_dcp_profile.py`):
- **Build→parse round-trip (foundational)**: construct a `.dcp` in-test via
  `build_camera_dcp` from a known `M_true`; `parse_dcp_bytes` recovers
  `ForwardMatrix1` ≈ `M·diag(1/m)`, `ColorMatrix1` ≈ `inv(M·diag(1/m))`,
  `illuminant_1`, and `name`; matrices within tolerance of the SRATIONAL
  quantisation.
- **Endianness**: write the same synthetic IFD as `II` and `MM` (a tiny manual
  byte-builder in the test, or byte-swap the generated one); both parse to equal
  matrices/illuminants.
- **SRATIONAL decode**: a tag with known num/den pairs decodes to the expected
  floats (incl. negative values and the inline-vs-offset boundary at count×8>4).
- **Inline vs offset**: a `SHORT` `CalibrationIlluminant` (inline) and a
  `LONG[3]` dims (offset) both resolve correctly.
- **Dual-illuminant interpolation**: a `DcpProfile` with FM1/FM2 and
  illum1=A(17)/illum2=D65(21); at `T=2856`→`w≈1` (FM≈FM1), at `T=6504`→`w≈0`
  (FM≈FM2), at a mid mired → the exact `w` from the §3.4 formula; assert the
  blended matrix equals `w·FM1+(1-w)·FM2`.
- **ForwardMatrix path vs ColorMatrix fallback**: a profile **with** FM maps a
  neutral camera value (post-WB equal-RGB) to D50 chromaticity; the **same**
  profile with FM removed uses `inv(CM)` + Bradford and still lands near-neutral
  (looser tolerance). Assert both produce linear-Adobe output (no OETF: a 50%
  linear grey stays ~mid, not sRGB-bumped).
- **Neutral → neutral**: feeding the camera neutral `n` (so `m·n` is grey) with
  `as_shot_wb=m` yields equal-RGB linear Adobe output (a*,b* ≈ 0 via
  `xyz_to_lab`).
- **Synthetic-camera round-trip (key)**: pick `M_true`; synthesise camera-native
  RGB for the IT8 reference patches via `inv(M_true)`; `build_camera_dcp` →
  `parse_dcp_bytes` → `apply_dcp(as_shot_wb=derived m)`; converting the result
  back through `inv(M_XYZ_D50_2_ADOBE)` to XYZ and to Lab gives **avg ΔE2000 ≈
  0** vs the reference (the DCP reproduces the chart, parallel to the ICC
  `test_fit_synthetic_roundtrip`).
- **Output space parity**: `apply_dcp` (matrices-only) and
  `InputProfile.apply` of `build_camera_icc(same fit)` produce **near-identical**
  linear-Adobe arrays for an unbalanced neutral input (pins the two GENERATE
  paths together, like `test_fit_and_apply_decode_spaces_are_identical`).
- **Look tables**: a HueSatMap with a single non-identity cell shifts hue/sat as
  expected when `apply_look=True`, and is a no-op when `apply_look=False`; an
  identity table is a no-op either way.
- **Errors**: truncated header, bad byte-order, missing `ColorMatrix1` &
  ForwardMatrix, matrix count≠9, offset past EOF → `DcpError`.

Manual:
- Load a real Adobe `.dcp` for a camera FreeCCR can decode; convert a negative;
  compare colorimetric-only vs look-on; verify no cast (linear-Adobe contract).
- Generate `.dcp` from an IT8 shot, "apply now", confirm it reprocesses and
  re-loads on restart; confirm setting a DCP clears an active ICC (and vice
  versa); non-RAW image while DCP active shows the WB warning.

## 9. Risks & mitigations
- **Wrong tag id/type** → all load-bearing ids/types verified against the Adobe
  DNG Spec 1.4–1.6 (Sources); the build→parse round-trip test is the regression
  guard.
- **As-shot WB plumbing** (only in the RAW branch) → captured inside
  `rawpy.imread`; non-RAW path uses `m=ones` with a UI warning. A `0` green
  multiplier from `camera_whitebalance` is sanitised to `1.0`.
- **Dual-illuminant CCT estimate** is approximate → it only selects the blend
  weight; matrix error from a small CCT miss is negligible, and most user DCPs are
  single-illuminant. Single-illuminant short-circuits the estimate entirely.
- **Look tables warping the negative** → off by default; documented as subjective
  (§5.7), behind an explicit checkbox.
- **ForwardMatrix absent** → documented fallback via `inv(ColorMatrix)` +
  Bradford; weaker but valid, and tested.
- **Non-RAW + DCP** → a DCP is a camera-raw profile; surfaced as a warning, applied
  unbalanced rather than crashing.
- **ICC/DCP both persisted** → set-one-clears-other (state + QSettings) makes the
  exclusive-slot invariant impossible to violate across restarts.

---

### Sources (DNG tag verification)
- Adobe **Digital Negative (DNG) Specification 1.6.0.0** (Dec 2021) and 1.4.0.0
  — camera-profile tags, illuminant enum, HueSatMap/LookTable/ToneCurve layout,
  mired-space dual-illuminant interpolation.
- `colour-hdri` `colour_hdri.models.dng` — interpolation in mired space; A≈2856 K
  / D65≈6504 K calibration endpoints; XYZ↔camera-neutral CCT derivation.
- `rawler` `DngTag` enum and `tiny_dng_loader` — confirmed decimal tag ids/types
  (ColorMatrix1=50721, ColorMatrix2=50722, CameraCalibration1/2=50723/50724,
  ReductionMatrix1/2=50725/50726, AnalogBalance=50727, AsShotNeutral=50728,
  CalibrationIlluminant1/2=50778/50779, ProfileName=50936,
  ProfileHueSatMapDims=50937, ProfileHueSatMapData1/2=50938/50939,
  ProfileToneCurve=50940, ProfileEmbedPolicy=50941, ProfileLookTableDims=50981,
  ProfileLookTableData=50982, ProfileHueSatMapEncoding=51107,
  ProfileLookTableEncoding=51108, BaselineExposureOffset=51109,
  DefaultBlackRender=51110, ForwardMatrix1/2=50964/50965).

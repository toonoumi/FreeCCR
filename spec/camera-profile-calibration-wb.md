# Spec: Camera profiles own their white balance (calibration neutral)

Status: IMPLEMENTED
Owner: FreeCCR
Feature branch: `claude/dcp-asshotneutral-computation-jlecqi`

## 1. Problem

A FreeCCR camera profile is built from an IT8 chart shot **on the copy stand the
user scans film with** — fixed light source, fixed camera. The profile *is* that
physical setup. But at apply time both camera-profile paths white-balanced the
frame with **per-frame metadata** instead:

```
ccr_image.read_image:  as_shot_wb = raw.camera_whitebalance
  -> dcp_profile.apply_dcp(profile, rgb, as_shot_wb=...)
  -> InputProfile.apply(rgb, as_shot_wb=...)
```

The profiled RAW decode is deliberately **unbalanced** — `use_camera_wb=False`,
`use_auto_wb=False`, `output_color=raw` (`ccr_image._raw_color_postprocess_kwargs`).
So the camera's WB *setting never touches the pixels*; it only writes the
`camera_whitebalance` metadata. With the camera on **AWB** that metadata is
re-estimated per frame from scene content — and for film scans the "scene" is
orange-mask negative whose density and framing change shot to shot. Identical
sensor data therefore got a different WB diagonal per frame, and `d_wb = d · m`
feeds straight into the ForwardMatrix, so **colour drifted frame to frame under a
light source that never changed**.

Symptom: "funky", inconsistent colour across a roll; worse the more the frames
differ in content.

## 2. Goals / non-goals

**Goals**
- A profile generated from an IT8 chart records the **camera-native neutral it was
  calibrated on**, in the profile file itself (survives library copy, import,
  export, re-selection).
- At apply time that calibration neutral **always** determines the white balance —
  the frame's as-shot metadata is ignored. Same profile ⇒ same WB ⇒ same colour,
  on every frame, regardless of the camera's WB mode.
- Both container formats (`.dcp` and `.icc`, matrix **and** cLUT).
- Zero behaviour change for profiles that carry no calibration neutral.

**Non-goals**
- Dual-illuminant profiles / illuminant interpolation (out of scope, unchanged).
- Any UI toggle. A single-illuminant chart profile calibrated on a fixed stand has
  exactly one correct neutral; a choice here would only be a way to re-introduce
  the bug. Changing the light means building a new profile.
- Retrofitting profiles generated **before** this change — they carry no neutral
  and keep the old per-frame behaviour. Re-run the wizard on the same chart shot
  to get a locked profile.

## 3. Data model

Notation (unchanged from `spec/camera-profile-color-fix.md`): `wb` =
green-normalised multipliers mapping raw → balanced (`wb[1] == 1`), stored as
`CameraFit.wb_mult`; `M` = `CameraFit.matrix`, balanced device → XYZ D50 with
`M·(1,1,1) = D50`.

The **calibration neutral** is the camera-native RGB of the chart's neutral patch,
green-normalised:

```
n = 1 / wb          (so n[1] == 1, and the WB gains are recovered as m = 1/n)
```

`n` and `wb` are exact reciprocals, so only one of them is stored.

### 3.1 DCP — `AsShotNeutral`, tag 50728

Written as `RATIONAL × 3` (the DNG type for this tag). This is the DNG-native name
for exactly this quantity, and matches the identity the generator already relies
on (`dcp_profile.build_camera_dcp`): `ColorMatrix1 · D50 = 1/wb = n`.

Third-party consumers are unaffected: `AsShotNeutral` is a *raw-file* tag, and both
Adobe's `dng_camera_profile` parser and RawTherapee's `DCPProfile` read a fixed
profile-tag list that does not include it, so it is inert in Lightroom/ACR/RT and
authoritative in FreeCCR.

`build_camera_dcp(..., bake_neutral=False)` omits it, producing a *portable* DCP
that defers to the host's per-frame WB.

### 3.2 ICC — private tag `CCRn`

ICC has no standard slot for a device neutral, so the value goes in a private
vendor tag `'CCRn'` carrying an `XYZType` payload (three `s15Fixed16`, ~1.5e-5
resolution — far finer than needed for values around 1.0). The container is a
well-formed ICC tag type, so third-party CMMs parse the tag table cleanly and
ignore the unknown signature. Written by both the matrix-shaper and the cLUT
builder.

## 4. Processing

One resolution helper, `color_management.resolve_wb_gains(calibration_neutral,
as_shot_wb)`, is the single decision point for both paths:

| profile carries `n` | as-shot metadata | gains `m` used |
|---|---|---|
| yes | anything | `1/n` — **profile wins, metadata ignored** |
| no | present | green-normalised `as_shot_wb` (previous behaviour) |
| no | `None` | `None` → unbalanced (previous degraded path) |

Callers keep passing the frame's `as_shot_wb`; it is simply outranked. This puts
the policy at the two apply choke points (`dcp_profile.apply_dcp`,
`InputProfile.apply`) rather than in `ccr_image`, so every consumer — preview,
hi-res zoom, slice, export, and `tools/scan_color_eval` — inherits it identically.

Note a second-order effect on the DCP path: `_interp_weight`/`_neutral_cct` derive
the illuminant blend weight from `m`. Feeding them the calibration neutral is
also more correct (the profile's own light, not a per-frame guess); generated
profiles are single-illuminant so the weight is 1.0 either way.

A non-RAW input (TIFF/JPEG, `as_shot_wb=None`) previously skipped balancing
entirely. With a calibration neutral it is now balanced like any other frame,
which is the correct treatment for camera-native data and a strict improvement.

## 5. Integration points

- `src/core/it8_profile.py` — `build_camera_icc` passes `neutral=1/fit.wb_mult`
  to both ICC builders. `CameraFit` is unchanged.
- `src/core/color_management.py` — `resolve_wb_gains`; `_CCR_NEUTRAL_SIG`;
  `build_matrix_shaper_icc(..., neutral=None)`, `build_clut_icc(..., neutral=None)`;
  `InputProfile.calibration_neutral` parsed in `from_bytes`; `apply` resolves gains.
- `src/core/dcp_profile.py` — `DcpProfile.as_shot_neutral`; `'rational'` writer kind
  in `_write_ifd`; `build_camera_dcp(..., bake_neutral=True)`; `apply_dcp` resolves
  gains.
- `src/core/ccr_image.py` — unchanged (still threads `raw.camera_whitebalance`).
- `src/widgets/it8_profile_dialog.py` — unchanged (the builders bake from the fit).

## 6. Test plan

`tests/test_dcp_profile.py`, `tests/test_it8_profile.py`, `tests/test_clut_icc.py`:

1. **Round-trip** — a built DCP parses back with `as_shot_neutral ≈ 1/wb_mult`
   (green-normalised); a built ICC (matrix and cLUT) parses back with
   `calibration_neutral ≈ 1/wb_mult`.
2. **Metadata is ignored** — applying a baked profile with three wildly different
   `as_shot_wb` vectors (including `None`) yields **identical** output. This is the
   regression test for the reported bug.
3. **Correct neutral chosen** — a baked profile applied to the chart's own neutral
   patch renders equal-RGB (D50 grey), and equals the result of the old path fed
   the matching `as_shot_wb` by hand.
4. **No baked neutral ⇒ old behaviour** — `bake_neutral=False` (and a hand-built
   IFD without 50728) still tracks the passed `as_shot_wb`, including the
   `None` ⇒ unbalanced degraded path.
5. **ICC ≡ DCP** — the existing cross-container equivalence tests still hold with
   both sides baked.
6. **Portability** — a baked DCP still parses as a valid TIFF/DCP with tags in
   ascending order; `AsShotNeutral` is `RATIONAL` type 5, count 3.

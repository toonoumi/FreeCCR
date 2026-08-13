# Spec: Camera profiles for trichrome (3-way RGB merge) captures

Status: REFINED — ready to implement
Owner: FreeCCR
Feature branch: `feat/trichrome-camera-profile`

## 1. Problem

A trichrome capture is **not the same device space** as a normal Bayer capture. The
merge takes the red plane shot under red light, the green plane under green light
and the blue plane under blue light (`ccr_merge.merge_raw_channels`), so each
channel's effective spectral sensitivity is the product of the sensor's response and
that light's spectrum — nothing like a single white-light exposure through a CFA.
The channel balance is set by three independent light intensities and exposures.

Two consequences, both currently unaddressed:

1. **A trichrome camera profile cannot be built.** The IT8 wizard decodes exactly one
   file (`it8_profile.decode_target` → `read_image`), so there is no way to profile
   the space the user actually scans in. Profiling a normal white-light shot of the
   chart and applying it to trichrome frames is simply the wrong transform.
2. **A camera profile is never applied to a merged image.** `CCRImage._read_merged`
   applies field correction and the slice chain, then returns — the DCP/ICC apply that
   the RAW branch performs (`ccr_image.py:748-750`) has no counterpart there. So even
   a correct profile would do nothing on the frames it was built for.

## 2. Goals / non-goals

**Goals**
- The IT8 wizard accepts a **trichrome triplet** (3 RAWs) as its target shot, merges
  them exactly as an import would, and from then on treats the merged frame as if it
  were one RAW — sampling, fit, quality report, ICC/DCP synthesis all unchanged.
- Merged images **apply the active camera profile** on every read path (preview,
  zoom, slice, export), in the same pipeline position as the RAW branch.
- A profile records which device space it was built for, so a trichrome profile and a
  normal profile cannot be silently swapped.

**Non-goals**
- No change to the merge maths, the triplet ordering convention, or the demosaic
  modes. The wizard consumes `ccr_merge` as-is.
- No new profiling maths. §4 is a decode-source change; everything downstream of
  `sample_patches` is untouched.
- No registration/alignment of the three chart frames (the existing merge non-goal).
- No per-image profile override, and no new mismatch UI beyond §6.

## 3. Selecting a triplet in the wizard

Page 1 gains a checkbox **"Trichrome: merge 3 RAWs (R, G, B)"**.

- **Off** (default): unchanged single-file behaviour.
- **On**: *Browse* opens a multi-select dialog restricted to RAW extensions and
  requires **exactly 3** files. They are sorted by basename, case-insensitively
  (`ccr_merge.sort_for_merge`), and taken as (red, green, blue) — the same rule the
  importer uses, so the wizard and an import of the same three files agree.
- **Use current image**: when the app's current image is itself a merged image, the
  wizard adopts its `merge_sources` and `merge_demosaic` directly and ticks the
  checkbox. This is the path a user who already has the chart loaded in 3-way mode
  will take, so it must not require re-picking files. `main_window` therefore passes
  the merge descriptor alongside `current_path`.

The **demosaic mode** is not asked for: it is read from `ccr_backend.rgb_merge_demosaic`,
the same global that governs imports, so the profile is fitted in the space the user's
images are actually merged into. The mode in force is shown in the target label
("single photosite" / "linear demosaic") — silently baking a mode the user cannot see
would make a mismatched profile impossible to diagnose.

Validation errors (not 3 files, a non-RAW among them, mixed sensor types, a decode
failure) surface through the existing target-decode error path.

## 4. Decoding the triplet

`it8_profile.decode_target_merged(sources, demosaic, sample_max=SAMPLE_MAX)` mirrors
`decode_target`: a bare `CCRImage.__new__` carrying only what the merged read needs —

```python
img.source_ops = []
img.is_merged = True
img.merge_sources = list(sources)
img.merge_demosaic = bool(demosaic)
return img.read_image(sources[0], preview=True, max_long_side=sample_max,
                      positive_override=False, apply_input_icc=False)
```

`read_image` dispatches on `is_merged` **before** it looks at the path or the
Positive-mode flag (`ccr_image.py:580`), so the path argument is a formality and the
existing bare-device contract carries over unchanged: `apply_input_icc=False` skips
field correction (§4 of `spec/flat-field-correction.md`) and, after §5 below, the
camera profile too — a profiling decode must never measure through another profile.

Everything after this point in the wizard is untouched: the merged array is a normal
`(H, W, 3)` uint16 camera-native frame.

## 5. Applying a profile to a merged image

`_read_merged` gains the apply step at the end of the chain, in the same position the
RAW branch uses (field correction → slice ops → downsize → **profile**):

```python
if not apply_input_icc:
    return rgb
if color_management.get_active_dcp_profile() is not None:
    return self._apply_input_dcp(rgb, None)
return self._apply_input_icc(rgb, None)
```

**`as_shot_wb` is `None`, deliberately.** A merged frame has three source RAWs with
three different `camera_whitebalance` values, and none of them describes the merged
channel balance — that balance is a property of the three light sources and their
exposures. Any single frame's metadata would be an arbitrary choice. Since
`spec/camera-profile-calibration-wb.md`, a FreeCCR-generated profile carries its own
calibration neutral and **ignores** the as-shot value anyway, so a trichrome profile
built by this feature balances merged frames on exactly the lighting rig it was
calibrated on. A profile with no baked neutral takes the documented unbalanced path;
for a trichrome merge that is the honest answer rather than a guess.

Note the ordering consequence: the profile sees the frame *after* field correction,
matching the RAW branch, so a flat-field profile and a camera profile compose the
same way in both.

## 6. Device-space guard

Applying a normal profile to a merged frame (or the reverse) produces badly wrong
colour with no visible symptom other than "the colour is off" — the exact failure this
feature exists to remove. Profiles therefore record their device space:

- `.dcp`: private tag `_T_CCR_TRICHROME = 52525`, LONG, value 1.
- `.icc`: private tag `CCRk` ('kind'), an XYZType payload with x=1 (mirrors the
  `CCRn` container convention of `spec/camera-profile-calibration-wb.md` §3.2).

Both are absent on a normal profile, so "no tag" means "normal" and every existing
profile stays valid. `DcpProfile.is_trichrome` / `InputProfile.is_trichrome` expose it.

On apply, a mismatch between the profile's kind and the image's kind
(`CCRImage.is_merged`) logs **once per profile+kind pair** — a warning, not a refusal:
the user may have a deliberate reason, and a hard block on a colour-management choice
would be worse than a wrong-looking preview they can see. The wizard also defaults the
profile name to `<chart> trichrome` so the two are distinguishable in the picker.

## 7. Integration points

- `src/core/it8_profile.py` — `decode_target_merged`.
- `src/core/ccr_image.py` — `_read_merged` applies the profile (§5).
- `src/core/color_management.py` — `_CCR_KIND_SIG`; `build_matrix_shaper_icc`/
  `build_clut_icc` gain `trichrome=False`; `InputProfile.is_trichrome`.
- `src/core/dcp_profile.py` — `_T_CCR_TRICHROME`; `DcpProfile.is_trichrome`;
  `build_camera_dcp(..., trichrome=False)`.
- `src/core/it8_profile.py` — `build_camera_icc(..., trichrome=False)` threads it.
- `src/widgets/it8_profile_dialog.py` — the checkbox, multi-select browse, the
  merge-aware "Use current", the demosaic label, and passing `trichrome=` to the
  builders.
- `src/ui/main_window.py` — passes the current image's merge descriptor to the dialog.
- `ccr_merge`, the fit, `build_residual_clut`, export — **unchanged**.

## 8. Test plan

`tests/test_trichrome_profile.py` (new), following the existing merge tests' pattern of
monkeypatching the one rawpy-touching function:

1. **Decode source** — `decode_target_merged` calls `merge_raw_channels` with the given
   sources and demosaic flag and returns its array unchanged; no field correction and
   no camera profile are applied even when both are active (the bare-device contract).
2. **Triplet ordering** — the wizard's selection sorts by basename before merging, so
   any pick order yields the same (R,G,B) assignment as an import.
3. **Apply on merged reads** — with an active DCP (and separately an ICC),
   `_read_merged` returns `apply_dcp(profile, merged, as_shot_wb=None)`; with
   `apply_input_icc=False` it returns the bare merge.
4. **Pipeline position** — the profile is applied after field correction and after the
   slice chain, matching the RAW branch.
5. **Baked neutral wins** — a trichrome profile with a calibration neutral renders a
   merged frame identically regardless of the source frames' metadata (the §5 claim).
6. **Kind round-trip** — `trichrome=True` profiles parse back with `is_trichrome`;
   normal ones report `False`; both containers; an existing profile without the tag
   reads `False`.
7. **Kind does not disturb the rest** — a trichrome DCP still parses as a valid TIFF
   with ascending tags, and the ICC still parses in lcms terms (tag table intact);
   `AsShotNeutral`/`CCRn` round-trip unaffected.
8. **Mismatch warning** — logged once per profile+kind pair, and never raises.

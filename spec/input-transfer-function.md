# Input Transfer Function (non-RAW decode space)

## 1. Problem

FreeCCR's non-RAW reader interprets **no transfer function at all**. In
`read_image`'s non-RAW branch (`src/core/ccr_image.py:811-928`) the only
transform is `_to_uint16_full_range` (`:59`), which rescales the *sample format*
(uint8 → ×257, float 0..1 → ×65535, wide/signed ints → scaled) and nothing else.
`spec/color-management.md:29` states the consequence outright: *"non-RAW files
are read as-is with no profile interpretation."*

So a gamma-encoded TIFF enters conversion with its encoding intact, and every
density stage downstream treats those values as **linear transmission**. The
damage is uneven, and worth stating per mode because it decides how much this
feature is worth:

- **Black-point-only** (`_default_slope_invert`, `ccr_processor.py:1600`) —
  `d = slope·log10(base/v)`. For a *pure* power law `v = p^(1/γ)` this is exactly
  `d_true/γ`: a uniform, per-channel-identical contrast loss, equivalent to
  dividing every density slope by γ. No hue shift, and Channel Levels Gain or a
  film-stock slope absorbs it. **But sRGB is not a pure power law** — below
  0.04045 encoded it is linear with slope 12.92, and the *densest* parts of a
  negative sit exactly there. That is the positive's highlights, and the toe
  breaks the scaling equivalence only there. A real defect, not a contrast
  offset.
- **No-anchor** (`_unanchored_density_invert`, `:1565`) — worse, because `16`
  and `+1.0` (`UNANCHORED_INPUT_SCALE` / `UNANCHORED_BASE_OFFSET`) are fixed
  constants calibrated against linear data. `−log10(16·p^(1/γ))` gets both the
  slope **and** the offset wrong, so the film base no longer lands near Cineon
  black. The docstring even names the premise it relies on: *"the negative decode
  is linear Adobe RGB … so no linearisation is needed here"* — true for RAW,
  false for a gamma-tagged TIFF.
- **Reference-frame / v0.2.3** (`compute_reference_norm_params`, `:2234`) —
  largely self-correcting: the 1st/99th percentile affine stretch is measured and
  applied on the same values. The leak is the OD mean-equalisation
  (`od_factors`), which measures `−log10` of encoded data, so the cast
  correction comes out γ-compressed and undercorrects. Changes the look; does
  not break.
- **Cineon Log → Workspace** assumes the base is log of linear, so it inherits
  whatever the above got wrong.

## 2. Goals / Non-goals

### Goals

- On an import containing TIFFs, **ask** how to interpret their encoding: use
  each file's own metadata, treat them as linear (today's behaviour), or specify
  one space/gamma manually for the batch.
- Linearise into the pipeline's expected input space **inside the reader**, so
  preview, hi-res zoom, export, slice, B/W-point sampling, dust-at-native-res
  and the `.ffc` build all inherit it identically (a resolution-independent
  point op, like the camera profile).
- Capture the decision **per image at import** and persist it, so every re-read
  reproduces the same decode — the `merge_demosaic` contract
  (`spec/trichrome-demosaic-mode.md`).
- **Zero change to every existing catalog and every untagged file**: absent the
  new field, the decode is byte-identical to today.
- Remember the answer optionally, and expose that in Settings → General.

### Non-goals

- **No primaries conversion.** This is transfer-function-only, exactly like the
  Cineon → workspace stage (*"The transform never applied a matrix — it is
  transfer-function-only"*, CLAUDE.md). An Adobe RGB-tagged TIFF gets its gamma
  linearised and **keeps its primaries**. A full colorimetric conversion is what
  the input-ICC / camera-profile path is for (`spec/color-management.md` §1.1).
  Saying otherwise would overclaim in exactly the way that spec warns against.
- **No post-import editing.** The transfer is fixed at import. Changing it on
  loaded images would invalidate existing conversions (a converted base was
  produced from the old decode) and needs a re-decode + re-convert pass with a
  staleness marker, like a camera-profile change. Deferred; see §10.
- **No new dependency.** `tifffile` already reads the tags; `color_management`
  already parses ICC TRCs.
- **RAW is untouched.** A RAW decode's space is chosen at decode time and is
  already linear by construction.
- Not a JPEG/PNG feature in v1 — the machinery is format-agnostic, but only
  TIFF triggers the dialog (§4.1) because that is where scanner output lives.

## 3. Data model

### 3.1 The per-image field

`CCRImage.input_transfer: Optional[str]`, appended to the **end** of
`__init__`'s keyword list (`ccr_image.py:77-103`). Values:

| Value | Meaning |
|---|---|
| `None` / `"linear"` | No transform. **The default**, and today's behaviour. |
| `"embedded"` | Resolve from the file's own metadata on every read (§5.2). |
| `"srgb"` | sRGB EOTF. |
| `"gamma:<g>"` | Pure power law, e.g. `"gamma:2.2"`, `"gamma:1.8"`. |
| `"rec709"` | Rec.709 OETF inverse. |

`"embedded"` is stored as a *decision*, not as a resolved curve, and that is
reproducible **by construction**: the catalog record is already invalidated when
the file's size or mtime changes (`_file_signature`, `catalog.py:69`), so the
bytes the tag is read from cannot drift under a stored `"embedded"`. Storing a
resolved name instead would be strictly worse — an embedded ICC's TRC may be a
table that no name describes.

### 3.2 Persistence

`serialize_image` (`catalog.py:192`) gains `"input_transfer"`;
`_restore_image` (`:596`) reads it back and passes it to the constructor. Absent
key → `None` → today's decode, so **no catalog migration and no
`CATALOG_VERSION` bump**: the field is purely additive and its absence is
already its default.

### 3.3 Globals (Settings)

On `ccr_backend`, following the `warn_no_anchor_convert` pattern
(`ccr_backend.py:85-89`), persisted by MainWindow:

- `tiff_transfer_ask: bool = True` — show the import dialog.
  QSettings `input/tiff_transfer_ask`.
- `tiff_transfer_mode: str = "linear"` — the remembered choice
  (`"embedded"` | `"linear"` | `"manual"`). QSettings `input/tiff_transfer_mode`.
- `tiff_transfer_manual: str = "srgb"` — the space used when mode is
  `"manual"`. QSettings `input/tiff_transfer_manual`.

### 3.4 The import decision

`ccr_backend.import_transfer: Optional[str]` — a **transient** per-import value
(not persisted), set by the UI immediately before `_launch_loader` and consumed
by the loader, exactly as `rgb_merge_demosaic` is a global read at import and
captured per image. `None` = no explicit decision for this import.

**Precedence** (`create_images_for_path` / `_restore_image`):

1. An explicit decision for **this** import (dialog shown, or the remembered
   default applied because the dialog is suppressed) wins.
2. Otherwise the **stored** catalog value.
3. Otherwise `None` (linear).

Rule 1 beating the stored value is deliberate and differs from nothing else in
the app: answering the dialog is an explicit act *about these files*. Rule 2 is
why the field is persisted at all — with the dialog suppressed, a file's
previously-chosen override must survive.

## 4. UX

### 4.1 The import dialog

**Trigger** — in `_import_file_list` (`main_window.py:1324`) after Unicode and
merge validation, and in `_import_folder` (`:1387`), both *before*
`_launch_loader`. Both run on the GUI thread, so a modal is safe; the loader has
not started.

Shown when **all** hold:
- `ccr_backend.tiff_transfer_ask` is on;
- the batch contains ≥1 `.tif`/`.tiff`, **excluding** files
  `ccr_merge.is_freeccr_merge_tiff()` identifies (`ccr_merge.py:63`) — FreeCCR
  writes those itself as genuinely linear and deliberately untagged
  (`spec/trichrome-linear-tiff-export.md:53`: *"tagging it sRGB/ProPhoto would be
  a lie"*). Offering to re-interpret our own linear output is the one way this
  feature could corrupt data it owns.

`_import_folder` never sees a file list, so the trigger enumerates the folder's
TIFFs with a cheap non-recursive `os.scandir` filtered to `.tif`/`.tiff` —
matching Open Folder's non-recursive scope without duplicating
`load_images_from_folder`'s extension list.

**Content** — a summary of what was actually found, then the three choices:

```
TIFF colour space                                           [ ? ]

This import contains 6 TIFF files.
  4 tagged sRGB · 1 tagged gamma 1.8 · 1 untagged

FreeCCR's film conversion measures optical density, which assumes
linear data. How should these files be read?

( ) Use each file's own metadata              [default if any tagged]
      Untagged files are read as linear.
(•) Read as linear (no conversion)            [default if none tagged]
      What FreeCCR has always done.
( ) Specify manually:  [ sRGB            v ]
      Applied to every TIFF in this import, overriding its metadata.

[ ] Don't ask again — remember this choice

                                        [ Cancel ]  [ Import ]
```

- The manual combo: **sRGB**, **Gamma 2.2**, **Gamma 1.8**, **Gamma 2.4**,
  **Rec.709**, **Linear**.
- Cancel aborts the import (nothing is loaded) — the choice materially changes
  every pixel, so "no answer" must not silently mean "linear".
- "Don't ask again" writes `tiff_transfer_ask=False` plus the mode/manual values;
  Settings → General re-enables it.
- The counts come from the same resolver the decode uses (§5.2), so the dialog
  can never describe the file differently from how it will be read.

### 4.2 Settings → General

A new **Input colour space** group next to *Conversion*
(`settings_dialog.py:112-123`), staged-and-applied-on-Done like every other
toggle there (`_init_toggles` `:423`, apply `:471-498`):

- `[x] Ask how to read TIFF colour space on import`
- `Default when not asking: [ Read as linear v ]` + the manual-space combo,
  enabled only for *Specify manually*.
- Muted explainer: changing these affects the **next import** only.

## 5. Processing

### 5.1 Where it runs

In `read_image`'s non-RAW branch, immediately after the sample-format and
channel-order normalisation (`ccr_image.py:899-907`) and **before**
`_apply_field_correction` (`:913`):

```
decode → _to_uint16_full_range → BGR→RGB → [TRANSFER] → field correction
       → source_ops → resize → camera profile → return
```

Before field correction, not after, and the reason matters: a vignetting falloff
is physically multiplicative in **linear light**, and an `.ffc` profile is built
through this same reader — so the flat and the frame are linearised
consistently, and `spec/flat-field-correction.md` §10.4's "measured and applied
in the same space" invariant still holds. Being a point op, its position
relative to `source_ops`/`resize` is irrelevant; being *inside* the reader is
what buys resolution independence for free.

### 5.2 Resolving a file's own encoding (`"embedded"`)

New `core/input_transfer.py`. `resolve_file_transfer(path)` returns a transfer
descriptor, in priority order:

1. **Embedded ICC** — TIFF tag **34675** (`InterColorProfile`), via
   `tifffile.TiffFile(...).pages[0].tags`. The authoritative source: it carries
   real TRC curves. Reuse `color_management._read_tag_table` (`:471`) and
   `_parse_trc_to_lut` (`:492`), which already build a 65536-entry
   device→linear LUT from `rTRC`/`gTRC`/`bTRC` and already handle `curv` and
   `para` forms. Exposed as a new public
   `color_management.transfer_luts_from_icc(icc_bytes) -> list[np.ndarray] | None`
   so this module does not reach into privates.
2. **TIFF `TransferFunction`** tag 301 — an explicit per-channel LUT.
3. **EXIF `ColorSpace`** 0xA001 — `1` ⇒ sRGB; `65535` (Uncalibrated) with
   `InteropIndex` `R03` ⇒ Adobe RGB, i.e. `gamma:2.19921875` (563/256).
4. **EXIF `Gamma`** 0xA500 ⇒ `gamma:<value>`.
5. Nothing found ⇒ **linear** (identity).

Reads only the header, never pixel data — the same cheap sniff pattern as the
existing planar-config check (`:821-833`) and `is_freeccr_merge_tiff`. Any
exception ⇒ linear + a warning, never a failed import.

### 5.3 The math

Input is uint16 in `[0, 65535]`, output is uint16 full-range — the reader's
existing contract, so no headroom question arises and nothing downstream changes
shape or dtype. In float32 on the unit interval, with `v = value/65535`:

- `srgb` → `color_management.srgb_decode(v)` (`:224`).
- `gamma:g` → `v ** g`. (Encoded is `linear^(1/g)`, so decoding is the power
  `g`.)
- `rec709` → `v/4.5` for `v < 0.081`, else `((v + 0.099)/1.099) ** (1/0.45)`.
- `embedded` → per-channel LUT gather, indexed directly by the uint16 value
  (the LUTs are length 65536, exactly as `InputProfile.apply` does at
  `color_management.py:664-666`).
- `linear` / `None` → **return the input array unchanged, same object**. The
  no-op must be an identity short-circuit, not a round-trip through float, so
  the regression guarantee in §2 is structural rather than numerical.

### 5.4 Positive mode

The two modes want different destinations, because the pipeline's expected input
space differs:

- **Negative mode** — linearise (above). The density math needs linear.
- **Positive mode** — the non-RAW path returns display-referred pixels
  (`:924`) and the adjustments own the look, so the target is **sRGB display
  encoding**: linearise, then `srgb_encode`. A file already tagged sRGB is
  therefore an exact identity, and a gamma-1.8 file is corrected into the space
  the rest of the app assumes rather than being left mis-encoded.

Both read the single `positive_mode` value already captured once per call at
`:650` ("read once", `spec/positive-mode.md` §3).

### 5.5 Interaction with a camera profile

If an input ICC or DCP is active, it applies its own TRCs
(`InputProfile.apply` → `self._luts`, `:664`) **after** this stage, so a user
who both assigns a display-profile ICC and marks the TIFF sRGB gets the curve
twice. Both are explicit user choices and both are honoured; the non-RAW camera
profile path is already documented as *"an advanced/at-your-own-risk
reinterpretation"* (`spec/color-management.md:60-61`), and it has no as-shot WB
either. A **one-time** log records the combination. No silent suppression: this
feature must not override a profile the user deliberately assigned.

## 6. Integration points

| File | Change |
|---|---|
| `core/input_transfer.py` | **New.** `resolve_file_transfer`, `apply_transfer`, `describe`, `summarize_batch`, `TRANSFER_CHOICES`. |
| `core/color_management.py` | New public `transfer_luts_from_icc()`; `rec709_decode()`. |
| `core/ccr_image.py` | `input_transfer` ctor kwarg (appended) + attribute; the transfer stage in the non-RAW branch between `:907` and `:913`. |
| `core/catalog.py` | `serialize_image` writes the key; `_restore_image` reads it; `create_images_for_path`/`_plain` apply the import-decision precedence (§3.4). |
| `core/ccr_backend.py` | `tiff_transfer_ask` / `tiff_transfer_mode` / `tiff_transfer_manual` globals + transient `import_transfer`; pass it through `load_images_from_files` / `load_images_from_folder`. |
| `ui/main_window.py` | QSettings restore (next to `:241-251`); `on_tiff_transfer_*` handlers; the dialog trigger in `_import_file_list` / `_import_folder`; folder TIFF scan. |
| `widgets/input_transfer_dialog.py` | **New.** The import dialog (§4.1). |
| `widgets/settings_dialog.py` | The *Input colour space* group (§4.2). |

Nothing is added to `conversion_inputs`: the conversion **math** is unchanged —
only the base pixels it reads differ — so no replay site, dispatch site or
`_ci_to_json` needs touching.

## 7. Test plan — `tests/test_input_transfer.py`

Resolver:
1. TIFF written with `iccprofile=` sRGB bytes → resolves to a LUT transfer whose
   midpoint matches `srgb_decode`.
2. Untagged TIFF → linear. Unreadable/garbage ICC → linear, no raise.
3. EXIF `ColorSpace=1` → sRGB; `Uncalibrated` + `R03` → `gamma:2.19921875`.
4. `TransferFunction` tag → LUT.

Math:
5. `gamma:2.2` and `rec709` round-trip against their encoders.
6. `linear`/`None` returns **the same object** (identity short-circuit).

Reader, through the real load path (the `CCRImage(path)` style already used by
`tests/test_scanner_tiff.py`):
7. A TIFF holding sRGB-encoded mid-grey (48197) loads as ≈32768 with
   `input_transfer="srgb"`, and as 48197 unchanged with `None` — the
   today's-behaviour regression guard.
8. Resolution independence: the same file read with `max_long_side=None`
   (the export/zoom path) applies the same transform.
9. Positive mode: a `gamma:1.8` file comes back sRGB-encoded; an `srgb` file is
   an exact identity.
10. Field correction still lands correctly with a transfer active (ordering).

State:
11. Catalog round-trip: `serialize_image` → `_restore_image` preserves it; a
    record with no key restores as `None`.
12. Precedence: explicit import decision beats the stored value; with no
    decision the stored value is used.

UI (headless, `QT_QPA_PLATFORM=offscreen`):
13. The dialog standalone: it defaults to *metadata* only when some file in the
    batch is actually tagged, and to *linear* otherwise; the manual combo is
    enabled only for *Specify manually*; each mode yields the expected token.
14. `decision_for` — the shared (mode, manual) → token mapping. The suppressed
    path and the dialog both call it, so a remembered answer and a
    freshly-given one cannot diverge.
15. Batch gating via `tiffs_in`: non-TIFFs ignored, and a FreeCCR-baked merge
    TIFF excluded.

Deliberately **not** unit-tested: `_maybe_ask_input_transfer` and the Settings
group, both of which need a live `MainWindow`. A test of either would be
asserting Qt wiring through a stub rather than behaviour, so they are covered by
construction — the decision logic lives in `decision_for`/`tiffs_in` (tested
above), and the existing UI suites that build these widgets must keep passing.

## 8. Risks

- **FreeCCR's own linear TIFFs.** Mitigated by the marker exclusion (§4.1); test
  13 locks it.
- **A wrong answer is invisible until conversion.** The dialog states the
  consequence and shows what the metadata actually says, and Cancel is available;
  the default is always today's behaviour.
- **Double TRC with a camera profile** (§5.5) — honoured + logged, by design.

## 9. Resolved questions

- *Store resolved curve or the `"embedded"` decision?* The decision — the file
  signature already guards the bytes, and a table TRC has no name (§3.1).
- *Primaries?* Out of scope, transfer-only (§2), per the Cineon precedent.
- *Default for untagged?* Linear. It is both today's behaviour and the only
  choice that cannot corrupt FreeCCR's own untagged linear output — even though
  an untagged consumer TIFF is in truth more likely encoded. The dialog is how
  the user says so.

## 10. Deferred

- Changing the transfer on already-loaded images (needs re-decode + re-convert
  and a staleness marker, like a camera-profile change).
- JPEG/PNG triggering the dialog (the resolver and decode are already
  format-agnostic).
- Full colorimetric conversion from the embedded profile's primaries — that is
  the input-ICC path's job.

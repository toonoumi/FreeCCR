# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FreeCCR is a cross-platform desktop application for batch image preview, selection, negative film conversion, and color correction. It supports RAW and standard image formats with a PySide6 (Qt) GUI and compiles to a standalone executable via Nuitka.

## Dev Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Generate version file from git tags (required before first run)
python write_version.py

# Run the app in development
python src/main.py

# Run all tests
python tests/run_tests.py

# Run with pytest
pytest tests/ -v
pytest tests/test_pytest_activation.py -v   # specific file

# Build standalone exe (Windows)
./build_exe.bat
# or via Makefile:
make build-windows

# macOS build
make build
make build-compatible    # older macOS targets (MACOSX_DEPLOYMENT_TARGET=10.15)

# Clean build artifacts
make clean
```

**Critical version requirement**: Python 3.11.0 exactly — newer versions fail with Nuitka compilation.

## Architecture

The app follows a layered MVC-like pattern:

```
src/main.py                 → QApplication setup, launches MainWindow
src/ui/main_window.py       → Main window, menus, file dialogs (~380 lines)
src/core/ccr_backend.py     → Singleton managing all loaded CCRImage instances
src/core/ccr_image.py       → Image model — RAW/standard format abstraction
src/core/ccr_processor.py   → Color adjustment math, lens correction, OpenCL GPU kernels
src/widgets/thumbnail_list.py  → Sidebar thumbnails with async loading dialog
src/widgets/image_preview.py   → Central canvas: histogram, zoom, reference frame
src/widgets/sliders_panel.py   → Adjustment controls (White Balance/Tint, tone, saturation, Channel Levels, Channel Balance)
src/activation/activation.py   → License validation (disabled — always returns True)
src/utils/unicode_path_utils.py → Cross-platform Unicode filename handling
```

**CCRBackend** is a global singleton — always access loaded images through it rather than holding direct CCRImage references.

## Key Patterns

**Async image loading**: `ImageLoaderWorker` (QObject subclass) runs in QThread with a `ThreadPoolExecutor` of up to 8 workers. Images are sorted by filename after the parallel load completes. A cancellable progress dialog is shown during batch loads.

**Image processing pipeline**: RAW files decoded via rawpy → 16-bit numpy arrays → resized to 1080px max side → `ccr_processor.adjust_image()` applies color corrections. OpenCL GPU acceleration is optional and conditionally compiled. Thumbnails (8-bit) are generated separately from full previews.

**Negative inversion**: Uses the v0.2.3 method — per-channel linear black/white-point normalisation against the reference crop (1st/99th percentiles), an optical-density mean-equalisation for cast balance, a linear `65535 - v` inversion, then `apply_postinvert_look` (saturation boost + shadow warmth). `ccr_normalize_with_reference` (auto reference frame) and `ccr_normalize_with_bwpoint` (user-sampled clear/dense points) share this path; the resolution-independent replay (`compute_reference_norm_params` → `apply_reference_normalization`, params `p_lo`/`p_hi`/`od_factors`) reproduces it for zoom/slice/export.

> **Parked experiment — density-space inversion** (branch `feature/density-tone-rendering`, do not delete): a physically-faithful Cineon/negadoctor-style inversion in optical-density (log) space (subtract `Dmin` offset, divide by film gamma, recover scene-linear `H = 10^(d/γ)`, then a display render). It was made permanent on `main` for a few releases (PRs #23/#24) but **reverted** here because converted frames came out far too dark (subject near-black, needed ~+50 brightness) with amplified chroma noise in the shadows — the faithful log stretch reveals scan/grain noise that the v0.2.3 affine pre-balance hid. The branch carries the full density pipeline + a brightness-norm/simple-gamma tone stage and env knobs (`FREECCR_DENSITY_*`); revisit if attempting a faithful inversion again. Note: a per-channel *multiply* before the log is a clean density offset (no tone-dependent color shift); a per-channel *black-subtract*-before-log (v0.2.3's stretch) warps each channel differently and is what trades faithfulness for the brighter, lower-noise look.

**Channel Levels is the FIRST adjustment stage**: on a windowed working-space base it runs inside `_apply_working_space_recovery` — ahead of White Balance, White Point and Gain, and **before the window clamp** — so it sees un-clamped display values that may sit below 0 or above 1. That is what makes a per-channel *shift* translate the whole histogram (sub-black film base rises into view; content pushed out the bottom lands in the shadow margin) instead of merely adding to what is already displayed. Consequences for anyone touching this: the stage order inside it is **Input Gain → per-channel Shift/Gain/Blackpoint → Master Shift** (Master is its own stage, *not* summed into the per-channel values as it once was), and **Master Gain is no longer part of it** — it is applied AFTER Channel Balance (`_master_gain_divisor`, `include_master_gain=False`), because an exposure control in front of a tone-weighted node made a gain change move the colour; Auto Gain rides that same parameter and moves with it (see `spec/master-gain-after-balance.md`); `adjust_image`/`adjust_image_opencl` **zero** the twelve `ch_*` params after the numpy pre-stage so the kernel can't re-apply them (this is why GPU/CPU parity is free on the windowed path); the kernel's own copy of the block runs only for non-windowed bases and clamps. `_default_slope_invert` deliberately no longer floors density at 0 — without that sub-base data the film base and a true image shadow are numerically identical and no per-channel control can separate them. `WS_LO` is 1.0 (not 0.5) to give that data room; the window *width* is unchanged, so display precision and highlight headroom are effectively untouched. See `spec/channel-levels-pre-clamp.md`.

**Channel Balance is the tone-weighted colour control, White Balance the flat one**: the three `balance_r/g/b` sliders are one anchored **curve node per channel** — `(0,0), (0.1875, y), (1,1)` through the Curves editor's own `_monotone_cubic`, so a move is exactly what dragging that node by hand gives. The node moves in **gamma**, not by a linear offset: `y = BALANCE_NODE_X ** (1/2**stops)` with `stops = (slider/100)·BALANCE_MAX_STOPS`. That is deliberate — a linear offset must stay below `BALANCE_NODE_X` or it crosses the pinned `(0,0)` and breaks monotonicity, and that cap is what stopped a heavily cast frame being correctable at the slider ends; a gamma move approaches 0 and 1 asymptotically, so **no endpoint invariant exists to violate**. Peak deviation at ±100 is about +0.40/−0.26. `BALANCE_NODE_X` is **3/16**, between 1/8 and 1/4: at 1/8 the curve bit too hard in the deep shadows (a full +100 moved x=0.05 by +0.167 vs +0.111 at 3/16) while the peak deviation is essentially the same, so 3/16 keeps the reach and drops the harshness. The asymmetry is inherent: a node at x=3/16 can rise most of the way to 1 but can never fall further than 3/16, so the downward side saturates near −0.26 and raising `BALANCE_MAX_STOPS` buys upward range only. Why a curve and not a gain: **Temperature/Tint is a contrast control on a density base.** WB is a flat per-channel *multiply* on the de-windowed display value, and on a `bw`/no-anchor conversion that value IS optical density — multiplying a log value is a per-channel gamma change, not a colour shift. It only behaves as real WB where the base is not density — `ref` conversions and Positive mode — which is exactly why both controls exist: Balance shipped as its REPLACEMENT in #130 and was then demoted to a collapsible under Channel Levels (default collapsed, with Master Gain below it, matching the render order), with Temperature/Tint restored above Brightness, the WB Picker/AWB driving them again, and the U/I/O + J/K/L nudge keys made opt-in (`ccr_backend.balance_hotkeys`, Settings → General → Keyboard, default off — gated by `QShortcut.setEnabled` so the six letters are not consumed at all while off). See `spec/white-balance-restore.md`. The tone-*uniform* density controls already exist (Channel Levels **Shift** = density offset, **Gain** = slope); what was missing is the tone-*weighted* one, which is what corrects **crossover** (dye layers whose casts diverge between shadows and highlights). The stage runs inside `_apply_working_space_recovery`, un-clamped, **after** Channel Levels and **before** White Balance, because on a windowed base much of the data still sits outside `[0,1]` where the node is identity, so Levels must place the histogram in the window first. Values outside `[0,1]` **pass through unchanged** (both endpoints are pinned, so identity-outside is continuous), which is what preserves the shadow margin and highlight headroom. `balance_*` are appended at the very END of the `adjust_image`/`adjust_image_opencl`/`_apply_working_space_recovery` signatures — those are called positionally in tests, so a mid-signature insert would shift every later argument. The non-windowed path runs the *same* float function on `img/65535`; on the OpenCL path Balance is consumed in numpy, and when Channel Levels is also active Levels is consumed in numpy too (the kernel would otherwise run it *after* Balance and reverse the order). `temperature`/`tint` never lost their pipeline stage, so restoring their sliders was pure UI plumbing and every catalog ever written renders unchanged. The neutral solve (WB Picker / AWB) is a **closed loop on the real render**, `CCRImage._solve_neutral_knob` + `solve_neutral_wb` (what the UI drives, returning temperature/tint) or `solve_neutral_balance` (the Balance trio, retained + tested but no longer wired to a button): it renders the sampled AREA through the actual pipeline, measures the mean that came out, corrects, repeats. There is deliberately **no analytic inverse** of any colour stage any more — inverting it means modelling every other stage (Channel Levels, the hidden Auto Gain offset, gamma, curves, Cineon), three such models shipped and each was wrong in some configuration; do not reintroduce one. WB is two knobs with independent objectives (`R−B` for temperature, `G−(R+B)/2` for tint, temperature first) and converges in one pass. For Balance, red is the anchor and never moves; each pass solves blue then green onto the mean of the other two MEASURED channels (R=134,G=120 → B=127), which with red pinned halves the gap per pass and converges to grey. Per-channel bisection on the measured output; the best-so-far result is kept so more passes never regress. `apply_adjustments` gained `auto_gain_override` (Auto Gain must be measured from the full base — a patch has no highlights) and `skip_dust` (spatial, meaningless on a detached patch); area layers are suppressed via `areas_override=[]`. A channel with no response (B&W profile) is left at 0 rather than pegged. **AWB** uses the same loop as a whole-frame regression: it downscales the frame to 256px (a downscale, not scattered pixels, because `gray_edge` needs neighbours), renders it each iteration and drives the estimator's reading of the RENDER to grey. Anything passed as `combine` **must report in the render's 0–65535 scale** and may be monotone DECREASING — `estimate_neutral_rgb` returns normalised [0,1], and feeding that in unscaled made the flat-response guard fire on every channel; `gray_edge` reports gradient energy, which falls as a channel is lifted. Both silently made the AWB button do nothing, so the guard and tolerance are now relative and the bisection detects direction from its endpoint probes. `gray_edge` is additionally used as a pixel SELECTOR (`gray_edge_pixel_mask`, fixed on the base) rather than a value statistic, since gradient energy cannot be driven to grey. See `spec/channel-balance.md` and `spec/white-balance-restore.md`.

**Master Gain is the app's only gain slider, and it runs after Channel Balance**: the old general-adjustments "Gain" was the same math at a different scale, so it was removed — `"exposure"` is no longer in `ADJUSTMENT_KEYS` or the `tone` sync group. Master Gain lives *outside* both the Channel Levels and Channel Balance collapsibles, below them (always visible, and in render order) while its `create_slider()` call stays third among the channel sliders, so the positional `ADJUSTMENT_KEYS` zip is unaffected; the panel reserves its layout slot and fills it with `insertLayout`. **Auto Gain rides `ch_master_gain`** (`v = CH_SLIDER_DIV·(1 − 1/g)`, `AG_GMIN/GMAX` = the slider's exact endpoints), so it lands wherever Master Gain lands — now after Channel Balance, which is what makes a Balance correction exposure-independent. On non-windowed bases Levels → Balance → Master Gain run as ONE normalised pass with a single closing clamp: moving that clamp into the middle would change a negative gain's result. The `exposure` *parameter* still exists and is not dead — it carries the legacy baked auto-exposure `eb` (default-slope mode when Auto Gain is off) and area-layer settings; it just has no slider. `eb` was deliberately left on it: `eb` is computed in `50·log2(g)` units but consumed by the `/300` curve, a pre-existing mismatch, and moving it would change conversions without fixing that.

**Converting with no anchors is allowed**: with neither a black point nor a reference frame, Convert runs a **fixed-constant density inversion** — `d = −log10(16·p) + 1.0` in log space, `_unanchored_density_invert` — and the user grades it with Channel Levels, which on that base is working in the same density space the black-point-only conversion produces. The `16` and `+1.0` are fixed (a base near half scale lands at ~Cineon black), so nothing is measured off the frame. Log, not linear, is the point: a Channel Levels *shift* on this base is a density offset, i.e. a clean per-channel multiply in linear light, where a linear inversion would make the same slider an additive lift. The negative decode is already linear Adobe RGB (`gamma=(1,1)`, `output_color=Adobe`), so the log is taken on genuinely linear data. Pair it with the **Cineon Log → Rec.709** checkbox for the intended display transform — without it the result looks flat by design. Crucially this is **not** a new `conversion_inputs` mode — it is the existing `mode: "bw"` recipe with `bw: (None, None)`, which is what it actually is (the B/W-point pipeline with zero points sampled). That keeps ~10 replay/dispatch sites working untouched, since they all hand `ci["bw"]` straight back to the converter; only `apply_bwpoint_normalization` / `ccr_normalize_with_bwpoint` and the catalog's `_ci_to_json`/`_ci_from_json` (which would `list(None)`) needed None-awareness. No black point also means no sprocket mask and no film-stock slopes — both are anchor-relative and are skipped. The UI confirms first via `_confirm_no_anchor_convert`, suppressible with Settings → General → Conversion (`convert/warn_no_anchor`, default on). See `spec/no-anchor-convert.md`.

**Camera profiles own their white balance**: the profiled RAW decode is deliberately unbalanced (`use_camera_wb=False`, `output_color=raw`), so `raw.camera_whitebalance` reflects only the camera's WB *setting* — on auto-WB it drifts per frame and used to drift colour with it. A profile generated by the IT8 wizard now records the camera-native neutral of the chart shot it was fit from (`AsShotNeutral` in a `.dcp`, the private `CCRn` tag in an `.icc`), and `color_management.resolve_wb_gains()` lets that neutral **override** the frame's metadata at apply time — one fixed copy-stand setup ⇒ one fixed WB across a roll. Profiles without a baked neutral (imported third-party DCPs, and anything generated before this) fall back to the per-frame metadata. See `spec/camera-profile-calibration-wb.md`.

**Trichrome captures are their own device space**: in a 3-way merge each channel's sensitivity is the sensor's response times *its own* light, and the channel balance comes from three independent exposures — so a white-light camera profile does not describe it. The IT8 wizard can therefore take a **triplet** as its target (`it8_profile.decode_target_merged` sets `is_merged`/`merge_sources` on a bare `CCRImage`, so `read_image` merges it and everything downstream treats it as one raw), and `_read_merged` applies the active DCP/ICC in the same pipeline position as the RAW branch, with `as_shot_wb=None` (no source frame's metadata describes a merge — the profile's baked neutral does). Profiles record which space they describe (`AsShotNeutral`-style private tags: DCP 52525, ICC `CCRk`) and a mismatch logs once. See `spec/trichrome-camera-profile.md`.

**Unicode path handling**: Windows non-ASCII filenames require `normalize_unicode_path()` and `validate_unicode_path()` from `src/utils/unicode_path_utils.py`. Always validate paths before passing to image loaders.

**Resource resolution**: `resource_path()` in `image_preview.py` handles both dev and Nuitka-bundled contexts. Nuitka embeds `src/icons/` and `LICENSES/` directories at build time.

**macOS display color management**: Qt raster windows are not color-matched — the Cocoa backing store inherits the NSWindow colorspace (default: the display's own profile), so sRGB-encoded previews render oversaturated on wide-gamut (Display P3) Macs while ICC-tagged exports display correctly. `install_macos_srgb_filter()` in `src/ui/theme.py` tags every top-level NSWindow as sRGB via a ctypes/objc shim so ColorSync matches the whole app to the display profile. Exports embed their ICC profile in `apply_export_colorspace()` (`src/core/color_management.py`) and are unaffected.

**Activation system**: The license verification code in `src/activation/` is present but bypassed — `is_activated()` always returns `True` in this build. Tests in `tests/test_activation_security.py` still validate the HMAC-SHA256 signing logic.

## Build Output

- Windows: `freeccr.exe` (standalone, no Python runtime needed)
- macOS: `.app` bundle (via build scripts in `macos_build_scripts/`)
- Version string is read from `src/version.py`, generated by `write_version.py` from git tags
- Release notes: `RELEASE_NOTES.md` (repo root) is the GitHub release body, read at the tagged commit. Replace its "What's New" with the new release's changes **before tagging** — the workflow's `check-notes` job fails any tag whose version the file doesn't mention.

## Windows Installer Build

```bash
# Step 1: build the standalone exe
./build_exe.bat

# Step 2: build the installer package
# Run the installer build script inside windows_build_scripts/
```

## Workflow

For every major change: create a new branch, make the changes, then open a PR back to `main`.

**New features**: before writing code, author a spec in the `spec/` folder (e.g. `spec/<feature>.md`) covering goals/non-goals, UX/interaction, data model, processing/math, integration points, and a test plan. Refine that spec once more (resolve open questions, tighten integration details) before implementing. Then implement against the refined spec, with tests. See `spec/curves-tone-control.md` for the reference shape.

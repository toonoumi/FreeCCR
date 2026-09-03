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
src/widgets/sliders_panel.py   → Adjustment controls (Kelvin, tint, exposure, etc.)
src/activation/activation.py   → License validation (disabled — always returns True)
src/utils/unicode_path_utils.py → Cross-platform Unicode filename handling
```

**CCRBackend** is a global singleton — always access loaded images through it rather than holding direct CCRImage references.

## Key Patterns

**Async image loading**: `ImageLoaderWorker` (QObject subclass) runs in QThread with a `ThreadPoolExecutor` of up to 8 workers. Images are sorted by filename after the parallel load completes. A cancellable progress dialog is shown during batch loads.

**Image processing pipeline**: RAW files decoded via rawpy → 16-bit numpy arrays → resized to 1080px max side → `ccr_processor.adjust_image()` applies color corrections. OpenCL GPU acceleration is optional and conditionally compiled. Thumbnails (8-bit) are generated separately from full previews.

**Negative inversion**: Uses the v0.2.3 method — per-channel linear black/white-point normalisation against the reference crop (1st/99th percentiles), an optical-density mean-equalisation for cast balance, a linear `65535 - v` inversion, then `apply_postinvert_look` (saturation boost + shadow warmth). `ccr_normalize_with_reference` (auto reference frame) and `ccr_normalize_with_bwpoint` (user-sampled clear/dense points) share this path; the resolution-independent replay (`compute_reference_norm_params` → `apply_reference_normalization`, params `p_lo`/`p_hi`/`od_factors`) reproduces it for zoom/slice/export.

> **Parked experiment — density-space inversion** (branch `feature/density-tone-rendering`, do not delete): a physically-faithful Cineon/negadoctor-style inversion in optical-density (log) space (subtract `Dmin` offset, divide by film gamma, recover scene-linear `H = 10^(d/γ)`, then a display render). It was made permanent on `main` for a few releases (PRs #23/#24) but **reverted** here because converted frames came out far too dark (subject near-black, needed ~+50 brightness) with amplified chroma noise in the shadows — the faithful log stretch reveals scan/grain noise that the v0.2.3 affine pre-balance hid. The branch carries the full density pipeline + a brightness-norm/simple-gamma tone stage and env knobs (`FREECCR_DENSITY_*`); revisit if attempting a faithful inversion again. Note: a per-channel *multiply* before the log is a clean density offset (no tone-dependent color shift); a per-channel *black-subtract*-before-log (v0.2.3's stretch) warps each channel differently and is what trades faithfulness for the brighter, lower-noise look.

**Channel Levels is the FIRST adjustment stage**: on a windowed working-space base it runs inside `_apply_working_space_recovery` — ahead of White Balance, White Point and Gain, and **before the window clamp** — so it sees un-clamped display values that may sit below 0 or above 1. That is what makes a per-channel *shift* translate the whole histogram (sub-black film base rises into view; content pushed out the bottom lands in the shadow margin) instead of merely adding to what is already displayed. Consequences for anyone touching this: the stage order inside it is **Input Gain → per-channel Shift/Gain/Blackpoint → Master Shift/Gain** (Master is its own stage, *not* summed into the per-channel values as it once was); `adjust_image`/`adjust_image_opencl` **zero** the twelve `ch_*` params after the numpy pre-stage so the kernel can't re-apply them (this is why GPU/CPU parity is free on the windowed path); the kernel's own copy of the block runs only for non-windowed bases and clamps. `_default_slope_invert` deliberately no longer floors density at 0 — without that sub-base data the film base and a true image shadow are numerically identical and no per-channel control can separate them. `WS_LO` is 1.0 (not 0.5) to give that data room; the window *width* is unchanged, so display precision and highlight headroom are effectively untouched. See `spec/channel-levels-pre-clamp.md`.

**Master Gain is the app's only gain slider**: the old general-adjustments "Gain" was the same math at a different scale, so it was removed — `"exposure"` is no longer in `ADJUSTMENT_KEYS` or the `tone` sync group. Master Gain lives *outside* the Channel Levels collapsible (always visible) while its `create_slider()` call stays third among the channel sliders, so the positional `ADJUSTMENT_KEYS` zip is unaffected; the panel reserves its layout slot and fills it with `insertLayout`. **Auto Gain rides `ch_master_gain`** (`v = CH_SLIDER_DIV·(1 − 1/g)`, `AG_GMIN/GMAX` = the slider's exact endpoints). The `exposure` *parameter* still exists and is not dead — it carries the legacy baked auto-exposure `eb` (default-slope mode when Auto Gain is off) and area-layer settings; it just has no slider. `eb` was deliberately left on it: `eb` is computed in `50·log2(g)` units but consumed by the `/300` curve, a pre-existing mismatch, and moving it would change conversions without fixing that.

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

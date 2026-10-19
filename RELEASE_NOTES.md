<!--
  This file is the GitHub release body, read at the TAGGED commit by all three
  jobs in .github/workflows/release.yml. Before tagging a release, REPLACE the
  "What's New" section with ONLY that release's changes (git history keeps the
  old ones) — the check-notes job fails the tag build unless this file mentions
  "FreeCCR <version>" for the tagged version.
-->
## What's New

**FreeCCR 2.2.1** — sharpening you can see, slicing that respects the crop, and a cleaner Monochrome merge.

- **Sharpening, with a visible default.** A new collapsible **Details** section adds Amount, Radius and Masking — a luma-only unsharp mask applied as the last stage, so hue and saturation are untouched and an inverted negative's chroma noise is never amplified. New images start at Amount 25 so grain shows and you can confirm a scan's focus; the slider reads 25, so you can see what is applied and zero it. The preview scales the radius to what you would see at 100% on the full-resolution file, while export and zoom tiles apply it exactly. Images already in your catalog are pinned to 0 and render unchanged.
- **Slicing cuts the crop, not the whole scan.** Slicing a cropped image used to tile the entire original, film rebate and all. Slice mode now shows the cropped frame and the pieces tile the kept region, still reading their pixels from the original file at full quality; Reset Slice hands the parent its crop back. Press and hold (or right-click) the Slice button to cut **every loaded image** at the same fractions — a roll scanned at a fixed pitch in one gesture.
- **Monochrome merge no longer shows a mesh at full resolution.** The trichrome Monochrome merge detail skipped the per-phase gain normalisation the single-image monochrome read already applies, so a stripped sensor's residual 15.75% phase spread printed a fine lattice of light lines at 100% zoom and in exports. It is now flattened before binning.
- **Up/Down always switch image, and the Settings wheel scrolls the page.** A focused slider, spinbox, combo or text field can no longer swallow Up/Down in the main window (dialogs and open popups keep their own). In Settings, the mouse wheel over a control scrolls the page instead of silently changing the hovered value, and the dialog opens larger.

## Install

### Windows
Download the installer (`FreeCCR_Install_*.exe`) from the **Assets** below and run it.

### macOS (Apple Silicon)
Download `FreeCCR_macOS_*.zip` from the **Assets**, unzip it, and move `FreeCCR.app` into your **Applications** folder.

⚠️ **macOS may say the app is "damaged and can't be opened" — it isn't.** FreeCCR isn't notarized by Apple, so Gatekeeper blocks unsigned downloads on first launch. Clear the quarantine flag once by running this in **Terminal**:

```
xattr -d com.apple.quarantine /Applications/FreeCCR.app
```

Then open the app normally.

*Alternative:* right-click the app → **Open** → **Open**. On macOS Sequoia (15), if no "Open" button appears, use the Terminal command above, or go to **System Settings → Privacy & Security → Open Anyway**.

### Linux (x86_64)
**AppImage (recommended):** download `FreeCCR_Linux_*-x86_64.AppImage` from the **Assets**, make it executable, and run it:

```
chmod +x FreeCCR_Linux_*-x86_64.AppImage
./FreeCCR_Linux_*-x86_64.AppImage
```

**Portable folder:** download `FreeCCR_Linux_*-x86_64.tar.gz`, extract it anywhere, and run `FreeCCR/FreeCCR`. Fully self-contained — no Python or packages to install.

Built on Ubuntu 22.04, so it runs on any x86_64 distro with glibc 2.35 or newer: Ubuntu 22.04+, Linux Mint 21+, Debian 12+, Fedora 36+, openSUSE, Arch, etc.

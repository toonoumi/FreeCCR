<!--
  This file is the GitHub release body, read at the TAGGED commit by all three
  jobs in .github/workflows/release.yml. Before tagging a release, REPLACE the
  "What's New" section with ONLY that release's changes (git history keeps the
  old ones) — the check-notes job fails the tag build unless this file mentions
  "FreeCCR <version>" for the tagged version.
-->
## What's New

**FreeCCR 2.1.0** — a trichrome mode for monochrome sensors, plus two geometry fixes for anyone scanning with sprocket holes in frame.

- **Trichrome: a "Monochrome" merge detail.** FreeCCR only treated a RAW as monochrome when its metadata said so — but a mono-converted (debayered) camera still reports a colour filter it no longer has, so its frames went down the Bayer path and were either interpolated or had three quarters of their pixels thrown away. The new **Monochrome (full resolution, no demosaic)** option in **Settings → Color Management → Trichrome capture → Merge detail** reads every photosite of each exposure as one luminance sample at full sensor resolution, ignoring the pattern the file claims. Like the other merge modes it is captured per image at import, so every preview, zoom, export and re-open reproduces the same decode. It is for monochrome sensors only: on a colour sensor it shows a checkerboard, because the filter's response under each light *is* the picture it reads.
- **The white sprocket-hole mask finds the holes again.** The mask decided what counted as clear film by measuring up to full scale (65535), but a decoded scan saturates lower than that — white level minus the black pedestal, 63486 on a 14-bit Sony — and a colour negative's base can sit a sliver below the ceiling in one channel. On a roll whose base measured 62092 in blue, just 1394 below saturation, that put blue's threshold above the holes themselves and nothing was masked at all. The threshold is now measured from each frame: the clear-film level per channel, with the cutoff half way between the film base and it. Holes land at 100% of that gap and deep shadows near 0%, so both sides get the same margin — and a frame with no clear film in it simply masks nothing instead of whitening its brightest corner.
- **Mirrored frames export the way the preview shows them.** A mirror and a quarter turn do not commute, and the export applied them in the opposite order from the canvas — mirror first, then rotate, rather than rotate then mirror. Any frame that was both mirrored and rotated by 90° or 270° came out 180° from what you saw, and a mirrored straighten leaned the wrong way at every rotation. Exports and thumbnails now follow the canvas. Frames that are only rotated, or mirrored both ways, are byte-for-byte what they were before; only the mirrored-and-rotated combinations change — to what the preview was showing you all along.

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

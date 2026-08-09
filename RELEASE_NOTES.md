<!--
  This file is the GitHub release body, read at the TAGGED commit by all three
  jobs in .github/workflows/release.yml. Before tagging a release, REPLACE the
  "What's New" section with ONLY that release's changes (git history keeps the
  old ones) — the check-notes job fails the tag build unless this file mentions
  "FreeCCR <version>" for the tagged version.
-->
## What's New

**FreeCCR 1.3.2** — flatter scans, more precise copy/paste, and a few things that were quietly going wrong.

- **Field correction for your scanning rig.** Shoot one frame of your bare light table, and a 3-step wizard in **Settings → Color Management** measures the lens vignetting, sensor colour shading, and uneven light, then saves it as a reusable profile. The active profile flattens every frame before the negative conversion sees it, so corners stop going dark and warm. It changes only the spatial falloff — never your exposure or white balance.
- **Ctrl/Cmd+C now asks what to copy.** Copy just the white balance, just the crop, or just the curves instead of the whole slider set — the same group picker Sync to All uses. Paste keeps every setting you *didn't* copy at the target's own value, and Color Profile, Crop, and Curves are copyable at last (the old copy/paste silently dropped them).
- **Rotation and mirroring can be copied and synced.** A new **Orientation** group in both apply-to-many pickers, so a roll scanned upside-down or emulsion-side-up gets righted once and pushed onto the rest instead of pressing `[` / `]` on every frame.
- **Auto white balance no longer balances on the film holder.** On an uncropped scan, the black holder and the clear film base / sprocket holes are not scene content, but they were dominating the estimate — one algorithm returned a perfectly neutral result on a strongly cast frame. AWB now reads midtones only, and all four algorithms land on the real cast. Clicking **AWB** (and the WB eyedropper) also updates the canvas immediately instead of waiting for your next click.
- **Side panels no longer cut off their own controls.** On systems with a wider UI font, the right-hand panels clipped everything past their edge — Convert All, Slice, the Color Profile dropdown, the slider values. The panels are wider now, and long film-stock and camera-profile names shorten with an ellipsis instead of pushing the layout off-screen or running under the dropdown arrow.
- **A file that fails to load now says why.** Previously it just disappeared from the import. You get one grouped message naming each file and the reason — unsupported compression, damaged, or moved — including the workaround for Nikon Z 8 High Efficiency NEFs, which no open-source decoder can read.

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

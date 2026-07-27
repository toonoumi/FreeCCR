<!--
  This file is the GitHub release body, read at the TAGGED commit by all three
  jobs in .github/workflows/release.yml. Before tagging a release, REPLACE the
  "What's New" section with ONLY that release's changes (git history keeps the
  old ones) — the check-notes job fails the tag build unless this file mentions
  "FreeCCR <version>" for the tagged version.
-->
## What's New

**FreeCCR 1.3.1** — a smoother trichrome (3-way merge) workflow.

- **Merged frames now remember your edits between sessions.** After a 3-way RGB merge, your conversion, sliders, crop, and dust work is saved and restored the next time you merge the same three source frames — just like a normal image. (If any of the three source files changed on disk, it re-merges fresh.)
- **Optionally replace originals with a single linear TIFF (off by default).** A new Trichrome-capture setting bakes each merged frame to a full-resolution 16-bit *linear* TIFF, then permanently deletes the three source RAWs and reloads from the TIFF — turning a trichrome shoot into one archival file per frame. You confirm before anything is deleted, and a frame's RAWs are only removed after its TIFF is written and verified. The generated TIFFs reopen normally even with 3-way merge left on.
- **Smaller lossless linear TIFFs.** The linear-TIFF writer now uses a horizontal predictor, so 16-bit files compress meaningfully (still bit-exact and readable everywhere).

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

<!--
  This file is the GitHub release body, read at the TAGGED commit by all three
  jobs in .github/workflows/release.yml. Before tagging a release, REPLACE the
  "What's New" section with ONLY that release's changes (git history keeps the
  old ones) — the check-notes job fails the tag build unless this file mentions
  "FreeCCR <version>" for the tagged version.
-->
## What's New

**FreeCCR 2.2.0** — zoom that shows real sensor detail, a Positive mode that opens at a natural brightness, and correct decoding for monochrome bodies and gamma-encoded scans.

- **100% zoom now decodes at the resolution the zoom needs.** The detail tile was always decoded as a half-size RAW capped at 4500px, so at 100% every pixel on screen had been interpolated 2× from half the sensor's resolution — you were never actually looking at your scan's detail. The tile now follows the zoom: the frame occupies a known number of screen pixels, so that is how many source pixels it carries, choosing the half-size decode only when that still satisfies the target and never inventing pixels past 100%. The tile cache tracks the resolution it was rendered for, so zooming in re-decodes sharper while zooming out keeps what it has, and the current tile stays on screen while the larger render runs. Nothing renders softer than before, and conversion math is untouched — the replays are resolution-independent. **Settings → General → Zoom → Load full resolution at 100% zoom** (default on) restores the old behaviour for low-memory machines.
- **Positive mode opens at a natural brightness.** The positive decode maps sensor saturation to display white with no auto-exposure anywhere, so every stop of highlight headroom a frame left cost a stop of brightness and each image had to be hauled up by hand — across a 16-frame roll the Gamma slider needed +27 to +87 to reach a normal midtone. Positives now start from a baked base render curve and a brightness baseline, the same way negatives already carry their film-look offset; both sliders still read 0, leaving the full range for per-frame trim. A gain was measured and rejected first: these frames already have their white point placed, so multiplying everything drove the clipped highlights further into the ceiling. Negative mode is untouched, and positives already in your catalog keep the values they were saved with — only fresh imports pick up the new baseline.
- **Mono-converted cameras decode the way the camera does.** A debayered body still reports a normal Bayer pattern and a daylight white balance, so near-neutral raw went through the colour matrix and the WB multipliers and landed heavily magenta — no curve could have fixed it. The new **Monochrome** interpretation bypasses both by construction, and three further faults are fixed with it: residual per-phase gain (measured at a 15.75% spread, printing a checkerboard at full resolution where binning hid it in previews) is normalised per frame to 0.00%; Monochrome plus Positive mode is now display-referred rather than scene-linear, which had been darkening the render instead of brightening it; and the base curve was re-fitted against the neutral decode. End-to-end against the cameras' own embedded renders, three test frames moved from 0.383/0.390/0.579 to 0.535/0.484/0.705 against targets of 0.537/0.463/0.741.
- **Read RAW as monochrome, and write greyscale as one channel.** A global **Settings → Color Management** toggle reads every photosite as one luminance sample at full sensor resolution — no demosaic, no phase slice, the declared filter pattern ignored, which is the point for a mono-converted body. A monochrome image now also renders grey end-to-end and exports as a genuinely single-channel file carrying the metadata that says so: TIFF photometric minisblack, a 1-component JPEG, and a grey ICC profile embedded in both. The forced-grey render means no colour slider can tint a preview that a single-channel file cannot reproduce — preview, thumbnail, zoom tile and file always agree. With the toggle off and Color Profile = Color, every decode, render and exported file is unchanged.
- **Non-RAW scans are read in their own colour space instead of as linear.** The non-RAW reader interpreted no transfer function at all, so a gamma-encoded scanner TIFF entered conversion with its encoding intact and every density stage treated those values as linear transmission. That matters most where the math is absolute: the no-anchor inversion's constants are calibrated against linear data, and sRGB's linear toe sits exactly where a negative's densest areas are. An import containing TIFFs now asks how to read them — from each file's own metadata, as linear (what it always did), or with one space forced for the batch — and the answer is captured per image, so every re-read reproduces the same decode. Untagged files stay linear, which keeps FreeCCR's own baked merge TIFFs correct and leaves every pre-existing catalog bit-for-bit unchanged.
- **Curve endpoints move horizontally as well as vertically.** Dragging the lower-left endpoint right sets the **black input point** and dragging the upper-right endpoint left sets the **white input point** — the levels-style clipping Curves gives you elsewhere. Outside the endpoints the curve holds flat rather than extrapolating, which is both what the clipping means and what keeps it safe: evaluated past the last point the interpolation can turn over and invert contrast in exactly the tones being clipped. New points stay inside the endpoint range, so an endpoint can't be silently demoted, and identity is still the same two points — no stored catalog changes meaning.
- **De-dusting stays on as you move through a roll.** Selecting another frame used to drop you out of dust mode, so spotting a roll meant re-entering the mode — and re-finding your brush size — for every single image. Dust mode now follows the selection: the next frame arrives with the brush still loaded, showing that image's own spots and feather setting. A stroke you were part-way through drawing when you switched is discarded rather than landing on the newly selected frame, each image's spots stay with it, and the mode still steps aside where it has nothing to work on — an un-converted negative outside Positive mode — or when the last image is closed.

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

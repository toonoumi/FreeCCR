<!--
  This file is the GitHub release body, read at the TAGGED commit by all three
  jobs in .github/workflows/release.yml. Before tagging a release, REPLACE the
  "What's New" section with ONLY that release's changes (git history keeps the
  old ones) — the check-notes job fails the tag build unless this file mentions
  "FreeCCR <version>" for the tagged version.
-->
## What's New

**FreeCCR 2.0.0** — a colour pipeline rebuilt around the two things that actually go wrong with a roll of film: the light you shot it under, and what happened in the tank afterwards.

- **Channel Balance — for the cast a white balance slider cannot reach.** Two very different things put a cast on a negative. Shooting daylight film under tungsten, at dusk, or in mixed light exposes the three dye layers by different amounts — an even cast, the same through the whole frame. Development that went wrong — off-temperature, exhausted or contaminated chemistry, a stand development that ran long — changes each layer's *contrast* by a different amount instead, and that cast is **not** even: the shadows go one colour and the highlights go another. No single white balance move can fix both ends at once, which is why a frame can look neutral in the midtones and still be green in the shadows and magenta in the highlights. **Channel Balance** gives you one control point per channel, low on that channel's tone curve: pull red down and the shadows lose their warmth while the highlights stay where they are. It pairs with **Channel Levels** — the *Shift* sliders there move a whole channel evenly, which is the right tool for the shooting-light cast; Channel Balance handles the part that varies with tone. Optional hotkeys `U`/`I`/`O` raise R/G/B and `J`/`K`/`L` lower them (off by default — turn them on in **Settings → General → Keyboard**).
- **White Balance and Tint are back, above Brightness.** They were removed for a while because on a negative they are not really white balance — the value being multiplied *is* optical density there, so the sliders behaved like a contrast control. They are useful wherever the image is not density: reference-frame conversions, positive mode, and anything you have decoded out of log. So both controls now ship: Temperature/Tint in the panel, and Channel Balance folded into its own section under Channel Levels. **WB Picker** and **AWB** drive Temperature/Tint again, and still solve by rendering what you clicked and measuring the result, so a pick lands on neutral no matter what else is set.
- **Channel Levels now runs first, before anything is clipped.** The film base used to sit below display black where nothing could reach it. Now the per-channel Shift, Gain and Blackpoint see the un-clipped values, so a shift *translates* the whole histogram: sub-black film base rises into view, and anything pushed off the bottom lands in the shadow margin instead of being destroyed.
- **"Cineon Log → Workspace" replaces "Cineon Log → Rec.709".** It is a decode *out of* log, not a display transform, so it now runs immediately after Channel Levels instead of at the very end, and it decodes into the app's working space rather than a video standard. Everything below it — Channel Balance, Master Gain, White Balance, contrast, curves — was grading log values before; now it grades a real image. Channel Levels stays above the decode, which is where a per-channel shift genuinely is a density offset.
- **Master Gain no longer moves your colour.** Exposure ran ahead of the tone-weighted colour controls, so brightening a frame slid every pixel past them and quietly changed the correction you had dialled in. Master Gain and the hidden Auto Gain offset now run after Channel Balance: exposure and colour are independent controls.
- **Convert with no black point at all.** Convert now works with neither a sampled film base nor a reference frame, using fixed density constants instead of measuring your film — the frame keeps its own cast and placement, and you grade it with Channel Levels. Pair it with **Cineon Log → Workspace**; without the decode it looks flat by design. FreeCCR confirms first, and the warning can be turned off in **Settings → General**.
- **Camera profiles for trichrome captures, and profiles that own their white balance.** A 3-way merged frame is its own device space — each channel is the sensor's response to *its own* light — so a white-light camera profile does not describe it. The IT8 wizard can now take a triplet as its target and build a profile for it. Separately, a profile built by the wizard records the camera-native neutral of the chart shot it was fitted from and uses that instead of each frame's own metadata, so one fixed copy-stand setup gives one fixed white balance across a whole roll instead of drifting frame to frame on auto-WB.
- **Open files straight from the command line.** `freeccr shot1.nef shot2.nef`, a folder, or a wildcard — handy for shell scripts and for "Open with" in your file manager.
- **Trichrome "replace originals with a linear TIFF" keeps your work.** The bake no longer burns your crop into the file it is about to make the only surviving copy — a crop stays a setting you can widen or clear later. And your edits now follow the frame onto its replacement instead of being stranded on the deleted originals.

> **Heads up before you update:** the colour stages run in a different order now, and the Cineon decode uses a different curve. Images you have already graded — especially any with Channel Balance or the Cineon checkbox set — will render differently and may want a revisit. Nothing in your catalog is lost or migrated; the same settings simply reach the pixels in a better order.

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

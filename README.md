<div align="center">

<img src="src/icons/freeccr_logo.png" width="130" alt="FreeCCR logo">

# FreeCCR

**Turn your color-negative film scans into beautiful, accurate positives — free, local, and physics-based.**

No subscriptions. No license keys. No cloud. Your images never leave your computer.

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue?style=flat-square)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS-lightgrey?style=flat-square)
![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![Built with PySide6](https://img.shields.io/badge/built%20with-PySide6-41CD52?style=flat-square&logo=qt&logoColor=white)
![Price: Free](https://img.shields.io/badge/price-free-success?style=flat-square)

[**Download**](#-download--install) · [Quick Start](#-quick-start) · [How conversion works](#-how-film-conversion-works) · [Build from source](#-build-from-source)

</div>

<div align="center">
  <img src="docs/screenshot.png" width="900" alt="FreeCCR converting a color negative — thumbnail strip of RAW scans, the large frame preview with its film-sprocket border, the RGB histogram, and the Film B/W Point + adjustment panel">
</div>

---

## What is FreeCCR?

FreeCCR is a free, open-source desktop app for photographers who shoot color negative film. Load a whole roll, set two reference points, and it converts every frame consistently — using a **physics-based pipeline** that models how film actually responds to light, instead of a naive `255 − value` invert.

It runs entirely on your machine, handles RAW straight out of your scanner, and includes a full set of color-correction tools for fine-tuning. The output is scientific and color-accurate — a clean starting point you can take into your editor of choice.

## ✨ Features

- 🎞️ **Physics-based negative conversion** — accounts for film's non-linear response, dye characteristics, and density range, not just a pixel flip.
- 🎯 **Two-point anchoring** — sample the film's white and black references once, then convert the entire roll consistently regardless of scene content.
- ⚡ **Auto mode** — automatic frame detection and conversion for a fast first pass.
- 📂 **Batch everything** — load a whole folder or roll at once. Supports RAW (CR3, CR2, NEF, ARW, DNG, …) and standard formats (TIFF, JPEG, PNG).
- 🎚️ **Full color correction** — temperature, tint, exposure, brightness, contrast, saturation, and white/black points, with live histogram and zoom.
- 🔁 **Sync & copy/paste adjustments** across frames so a whole roll matches in seconds.
- 🚀 **Optional OpenCL GPU acceleration** for faster processing on supported hardware.
- 🔒 **Completely offline** — all processing is local; nothing is ever uploaded.
- 🆓 **Free & open source** (AGPL-3.0) — no activation, no watermark gate, no strings.

---

## 📥 Download & Install

Grab the latest build from the [**Releases page**](https://github.com/toonoumi/FreeCCR/releases).

### Windows

Run `FreeCCR_Install_<version>.exe` and follow the installer. That's it.

### macOS

Unzip `FreeCCR_macOS_<version>.zip` and move `FreeCCR.app` to your **Applications** folder.

Because the app isn't notarized, Gatekeeper may block the first launch with a *"damaged and can't be opened"* message. **The app isn't damaged** — macOS just flags unsigned downloads. Clear the quarantine flag once in Terminal:

```bash
xattr -d com.apple.quarantine /Applications/FreeCCR.app
```

Then open it normally. (Alternatively: right-click the app → **Open** → **Open**.)

---

## 🚀 Quick Start

The recommended path is the **B/W Point** workflow — you anchor the whole roll to two physical reference points on the film, so every frame converts consistently:

1. **Load a roll** — *File → Load Images from Folder* (work one roll at a time).
3. **Set White Point** (optional) — find the fully exposed leader (the bright, washed-out strip) and sample it.
4. **Set Black Point** — sample a patch of clear film base.
5. **Convert All (B/W Point)** — every frame inverts from those two anchors.
6. **Fine-tune** with the right-hand sliders, then **Sync to All** (or `Ctrl/Cmd+C` / `Ctrl/Cmd+V`) to carry adjustments across the roll.
7. **Export** — *Export* or *Export All* to your chosen folder (tick **Export JPG** for web-ready files).

> 📖 Full walkthrough and scanning requirements: **[How Film Conversion Works](#-how-film-conversion-works)**.
>
> 🔁 **No exposed leader in your scans?** Fall back to the **Auto** workflow below — it sets the points per frame automatically. It's less consistent across a roll, but it doesn't need a physical white point.

---

## 🎞️ How Film Conversion Works

FreeCCR doesn't guess — it maps what your scanner actually captured. That means a good scan in gives a good positive out, and **the software can't compensate for a bad scan**.

> ⚠️ **Process one roll at a time.** The black/white anchors are physical properties of a specific film stock, development, and scan session. Mixing frames from different rolls — or the same roll scanned in separate sessions — produces inconsistent results.

### Workflow 1 — B/W Point (recommended)

The most accurate method. You sample two anchor values directly from the scan: the **fully exposed head or tail** (white point) and the **clear film base** (black point). FreeCCR maps the whole roll from those absolute anchors, so every frame inverts consistently regardless of scene content.

> 📷 **Scanning requirement:** You must scan a frame (or partial frame) that includes the **fully exposed area** of the roll — usually the head or tail leader exposed to light before or after shooting. Without it, there's no physical reference for the white point.

<div align="center">
  <img src="docs/bw-point-workflow.png" width="900" alt="FreeCCR B/W Point workflow: setting the white point on the fully exposed leader and the black point on the clear film base, then Convert All">
  <br>
  <sub><b>①</b> Set White Point → <b>②</b> sample the fully exposed leader &nbsp;·&nbsp; <b>③</b> Set Black Point → <b>④</b> sample the clear film base &nbsp;·&nbsp; <b>⑤</b> Convert All</sub>
</div>

**Step by step:**

1. Load an entire roll — one roll at a time.
2. Find the scan containing the **fully exposed head or tail** of the roll — the leader that was blown out to light. FreeCCR shows it as the bright, washed-out strip.
3. **①** Click **Set White Point**, then **②** sample that fully exposed area. It's the most light the film ever saw, so it anchors the top (white) of the tonal range.
4. Find a patch of **clear film base** — the unexposed rebate beside the frame, which FreeCCR shows as the dark area.
5. **③** Click **Set Black Point**, then **④** sample the film base to anchor the bottom (black) of the tonal range.
6. **⑤** Click **Convert All (B/W Point)**. Every frame on the roll inverts from those same two anchors.
7. Use the sliders for per-image fine-tuning afterward.

**Why it works:** the fully exposed leader is the most light the film could possibly record, so it defines the absolute maximum for that stock and development; the clear film base defines the absolute minimum. Anchoring to these two physical references makes every frame invert consistently — no matter how bright or dark each scene was.

**If something looks off:**

- *No fully exposed area was scanned* → there's no valid white-point reference. Rescan to include the leader or tail.
- *Film base sample landed on a scratch or fog* → resample the black point from a clean edge.
- *The scanner used auto-exposure* → raw values differ frame to frame and batch conversion will be inconsistent. Rescan with a fixed exposure.

### Workflow 2 — Auto (fallback)

**Use this when your scans don't include a fully exposed leader**, so there's no physical white point to sample. Auto analyzes each frame's histogram independently and sets black/white points for you — no manual sampling required. Because it's per-frame, it has no knowledge of the film base or the stock's physical density range, and it won't match the roll-wide consistency of the B/W Point workflow.

**Reach for Auto when:**

- Your scans don't include a fully exposed leader to anchor a white point.
- Frames are simple and well-exposed, with no extreme shadows or highlights.
- You want a quick first-pass preview before committing to B/W-point work.

**Avoid Auto when:**

- Brightness varies widely across the roll (e.g. interiors next to bright exteriors).
- You need consistent tones across frames for stitching or comparison.
- The roll includes underexposed or push-processed film.

### Scanning Requirements (both workflows)

| Setting | Requirement |
|---|---|
| **Exposure** | **Fixed / manual.** Every frame must be scanned at exactly the same exposure. Auto-exposure changes raw values frame-by-frame and destroys the absolute density information conversion depends on. |
| **Auto-brightness / auto-correction** | **Off** |
| **Per-frame color correction** | **Off** |
| **Bit depth** | 16-bit preferred, 14-bit minimum |
| **Output color space** | Linear, or no ICC profile applied |
| **Roll handling** | **One complete roll per session.** Don't mix frames from different rolls or separate scan sessions. |

> A scan that violates any of these can't be reliably converted by FreeCCR — or any other software. Once the physical density information is absent or corrupted at scan time, no tool can recover it afterward.

---

## 🧪 Need a Lab First?

If you need somewhere to develop and scan your film before using FreeCCR, browse the global lab directory at [FilmPhotoDeveloping.com](http://filmphotodeveloping.com/).

---

## 🛠️ Build From Source

### Run in development

> **Requires Python 3.11.0 exactly** — newer versions fail to compile with Nuitka.

```bash
git clone https://github.com/toonoumi/FreeCCR.git
cd FreeCCR
pip install -r requirements.txt
python write_version.py   # required on first run or after tagging
python src/main.py
```

FreeCCR can also be pointed at images from the command line — files, folders, or
a mix. A single folder opens like **Open Folder**; anything else opens as a file
list, with each folder expanded to the images inside it:

```bash
python src/main.py /rolls/roll-07          # open a whole folder
python src/main.py a.nef b.nef c.nef       # open specific frames
python src/main.py --help                  # usage, then exit
```

<details>
<summary><strong>Build a standalone Windows executable + installer</strong></summary>

#### Step 1 — Standalone executable

```bat
build_exe.bat
```

Generates the version file from the current git tag, then compiles `src/main.py` into a self-contained executable with Nuitka + MinGW64. All dependencies, PyOpenCL kernels, and icon assets are bundled automatically.

**Output:** `main.dist/` containing `freeccr.exe` and everything it needs.

#### Step 2 — Installer

Requires **[Inno Setup 6](https://jrsoftware.org/isinfo.php)** at `C:\Program Files (x86)\Inno Setup 6\ISCC.exe` (update `ISCC_PATH` in `windows_build_scripts/build_inno_installer.bat` if it lives elsewhere).

Before running, open `windows_build_scripts/inno_setup.iss` and point the `Source:` paths under `[Files]` to your local `main.dist\` directory.

```bat
windows_build_scripts\build_inno_installer.bat
```

The script reads the version from `git describe --tags`, injects it into the Inno Setup script, and compiles the installer to `win_installer/FreeCCR_Install_<version>.exe`.

</details>

<details>
<summary><strong>Build a macOS app bundle (signed / notarized)</strong></summary>

**Prerequisites:**

- Xcode command line tools
- An Apple Developer ID certificate (distribution) or Apple Development certificate (local testing)
- `dmgbuild`: `pip install dmgbuild`

Set the `SIGN_CERTIFICATE` variable at the top of the build script to your certificate name as it appears in Keychain Access.

#### Distribution build — signed & notarized DMG

```bash
bash macos_build_scripts/build_compatible.sh
```

This script installs/upgrades dependencies and Nuitka, runs `write_version.py`, compiles targeting macOS 10.15+, assembles `FreeCCR.app` with `Info.plist` and icon, strips extended attributes, code-signs with hardened runtime, packages into `FreeCCR.dmg` via dmgbuild, submits it to Apple for notarization, and on success moves the notarized DMG to `release/<version>/FreeCCR.dmg`.

Store notarization credentials in the keychain under the profile name `notaryccr` once:

```bash
xcrun notarytool store-credentials "notaryccr" \
  --apple-id "your@apple.id" \
  --team-id "YOURTEAMID" \
  --password "app-specific-password"
```

#### Development build — local signing only

```bash
bash macos_build_scripts/create_bundle.sh
```

Assembles and locally signs the `.app` and DMG without notarization — fine for internal testing.

</details>

---

## 🔒 Privacy

Everything happens on your computer. Images are never uploaded, there's no tracking or analytics of your workflow, and core features work fully offline.

## 🤖 Use of AI

FreeCCR started years ago as a personal project, then sat idle for a long while. It was revived and refined with heavy use of AI coding tools — the recent code, fixes, and docs are a collaboration between a human author and AI working together.

I think that's a good thing, but I'd rather be upfront about it: if you'd prefer not to use software co-authored by a human and AI, this project may not be for you — and that's completely fine.

## 💬 Support

Questions, bug reports, and feature requests are all welcome — open an [issue](https://github.com/toonoumi/FreeCCR/issues) right here on GitHub.

## 📄 License

FreeCCR is licensed under the [GNU Affero General Public License v3.0](LICENSE) (**AGPL-3.0**). Like GPLv3, it's **copyleft**: if you distribute modified versions, you must license your changes under the same terms and provide the corresponding source.

AGPL adds a **network-use** clause: if you **run** a modified version as a **service** (including SaaS) where users interact with it remotely over a network, you must offer those users the corresponding source — including code you only deploy on servers. (Plain GPLv3 doesn't require this for typical SaaS.)

Bundled third-party libraries remain under their own licenses in [`LICENSES/`](LICENSES/).

<div align="center">
<sub>Made for film photographers · by Halo Imagery</sub>
</div>

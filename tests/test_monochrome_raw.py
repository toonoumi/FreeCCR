#!/usr/bin/env python3
"""Monochrome RAW interpretation + single-channel export
(spec/monochrome-raw-mode.md).

Two halves:

* The global "interpret RAW as monochrome" toggle reads a RAW's whole visible
  mosaic as one luminance sample per photosite — full sensor resolution, no
  demosaic, the declared CFA ignored (for mono-converted bodies whose RAW still
  reports RGGB) — and such an image renders grey end-to-end.
* A monochrome image (that toggle, a true monochrome sensor, or Color Profile =
  Black & White) exports as a SINGLE-channel TIFF/JPEG with the metadata that
  says so, plus a grey ICC profile.
"""
import io
import os
import struct
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

import tifffile  # noqa: E402
from core import ccr_image as ccr_image_mod  # noqa: E402
from core import ccr_merge  # noqa: E402
from core import ccr_processor as cp  # noqa: E402
from core import color_management as cm  # noqa: E402
from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402

try:
    from PIL import Image as PILImage
    PIL_OK = True
except Exception:
    PIL_OK = False


@pytest.fixture(autouse=True)
def _clean_backend():
    saved = (getattr(ccr_backend, "mono_raw", False), ccr_backend.positive_mode)
    yield
    ccr_backend.images = []
    ccr_backend.file_paths = []
    ccr_backend.mono_raw, ccr_backend.positive_mode = saved


# --------------------------------------------------------------------------- #
# 1. The predicate
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("decoded_mono, profile, expected", [
    (False, "color", False),
    (True, "color", True),      # monochrome decode
    (False, "bw", True),        # per-image Black & White
    (True, "bw", True),
])
def test_renders_monochrome_truth_table(decoded_mono, profile, expected):
    img = SimpleNamespace(decoded_mono=decoded_mono, color_profile=profile)
    assert cp.renders_monochrome(img) is expected


def test_renders_monochrome_defaults_to_colour_for_a_bare_stub():
    """The export writer is called with partial stubs all over the suite; they
    must read as colour without carrying either attribute."""
    assert cp.renders_monochrome(SimpleNamespace()) is False
    assert cp.renders_monochrome(object()) is False


# --------------------------------------------------------------------------- #
# 2. Grey ICC profiles + the grey export transform
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("target", ["srgb", "prophoto"])
def test_gray_icc_is_a_valid_grey_profile(target):
    icc = cm.gray_icc_bytes(target)
    assert icc[36:40] == b"acsp"                     # ICC signature
    assert icc[16:20] == b"GRAY"                     # data colour space
    assert len(icc) == struct.unpack(">I", icc[0:4])[0]   # declared size is real
    tags = cm._read_tag_table(icc)
    assert b"kTRC" in tags and b"wtpt" in tags
    assert not {b"rXYZ", b"gXYZ", b"bXYZ"} & set(tags)   # no colorants in grey
    if PIL_OK:
        from PIL import ImageCms
        ImageCms.getOpenProfile(io.BytesIO(icc))     # raises if malformed


def test_gray_icc_bytes_selects_by_target():
    assert cm.gray_icc_bytes("srgb") == cm.GRAY_SRGB_ICC_BYTES
    assert cm.gray_icc_bytes("prophoto") == cm.GRAY_PROPHOTO_ICC_BYTES
    assert cm.GRAY_SRGB_ICC_BYTES != cm.GRAY_PROPHOTO_ICC_BYTES


def test_gray_icc_tone_curve_matches_its_rgb_sibling():
    """Same curve as the RGB profile, so grey and colour exports of the same
    tones are interpreted identically."""
    for gray, rgb in ((cm.GRAY_SRGB_ICC_BYTES, cm.SRGB_ICC_BYTES),
                      (cm.GRAY_PROPHOTO_ICC_BYTES, cm.PROPHOTO_ICC_BYTES)):
        g_tags, r_tags = cm._read_tag_table(gray), cm._read_tag_table(rgb)
        g_off, g_len = g_tags[b"kTRC"]
        r_off, r_len = r_tags[b"rTRC"]
        assert gray[g_off:g_off + g_len] == rgb[r_off:r_off + r_len]


def test_apply_export_colorspace_gray_srgb_is_identity():
    src = np.array([[0, 1, 1000, 30000, 65535]], dtype=np.uint16)
    out, icc = cm.apply_export_colorspace_gray(src, "srgb")
    np.testing.assert_array_equal(out, src)
    assert icc == cm.GRAY_SRGB_ICC_BYTES


def test_apply_export_colorspace_gray_prophoto_is_tone_only():
    src = np.arange(0, 65536, 257, dtype=np.uint16).reshape(1, -1)
    out, icc = cm.apply_export_colorspace_gray(src, "prophoto")
    assert icc == cm.GRAY_PROPHOTO_ICC_BYTES
    assert out.dtype == np.uint16 and out.shape == src.shape
    # Endpoints pinned, monotone, and exactly romm(srgb_decode(x)) — no matrix.
    assert out[0, 0] == 0 and out[0, -1] == 65535
    assert np.all(np.diff(out[0].astype(np.int32)) >= 0)
    expect = np.rint(cm.romm_encode(
        cm.srgb_decode(src.astype(np.float32) / np.float32(65535.0))) * 65535.0)
    assert np.abs(out.astype(np.float32) - expect).max() <= 1.0


# --------------------------------------------------------------------------- #
# 3. Single-channel writing
# --------------------------------------------------------------------------- #

def _rgb(h=4, w=6, value=20000):
    return np.full((h, w, 3), value, dtype=np.uint16)


def _tinted(h=4, w=6):
    img = np.zeros((h, w, 3), dtype=np.uint16)
    img[..., 0], img[..., 1], img[..., 2] = 10000, 20000, 5000
    return img


def _write(stub, rgb, path, jpg=False, target="srgb"):
    return cp.write_export_image(stub, rgb, path, jpg, 92, None,
                                 output_colorspace=target)


def _jpeg_components(data: bytes):
    """SOF0/1/2 component count of an encoded JPEG (1 = greyscale, 3 = colour)."""
    i = 2
    while i < len(data) - 1:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            return data[i + 9]
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    return None


@pytest.mark.parametrize("target", ["srgb", "prophoto"])
@pytest.mark.parametrize("mono_attrs", [
    {"decoded_mono": True, "color_profile": "color"},    # monochrome decode
    {"decoded_mono": False, "color_profile": "bw"},      # Black & White profile
])
def test_monochrome_tiff_is_single_channel_with_grey_metadata(
        tmp_path, target, mono_attrs):
    out = _write(SimpleNamespace(**mono_attrs), _rgb(),
                 str(tmp_path / f"m_{target}_{mono_attrs['color_profile']}.tiff"),
                 target=target)
    with tifffile.TiffFile(out) as tf:
        page = tf.pages[0]
        assert int(getattr(page.photometric, "value", page.photometric)) == 1
        assert int(page.samplesperpixel) == 1
        assert bytes(page.tags[34675].value) == cm.gray_icc_bytes(target)
        arr = tf.asarray()
    assert arr.ndim == 2 and arr.dtype == np.uint16
    if PIL_OK:
        assert PILImage.open(out).mode == "I;16"


def test_colour_tiff_is_unchanged(tmp_path):
    """Regression: a colour image still writes 3 channels + the RGB profile."""
    out = _write(SimpleNamespace(decoded_mono=False, color_profile="color"),
                 _tinted(), str(tmp_path / "c.tiff"))
    with tifffile.TiffFile(out) as tf:
        page = tf.pages[0]
        assert int(page.samplesperpixel) == 3
        assert bytes(page.tags[34675].value) == cm.SRGB_ICC_BYTES
        assert tf.asarray().shape == (4, 6, 3)


def test_monochrome_jpeg_has_one_component_and_grey_icc(tmp_path):
    out = _write(SimpleNamespace(decoded_mono=True, color_profile="color"),
                 _rgb(), str(tmp_path / "m.jpg"), jpg=True)
    data = open(out, "rb").read()
    assert _jpeg_components(data) == 1
    assert cm.GRAY_SRGB_ICC_BYTES in data            # APP2 payload is embedded
    if PIL_OK:
        im = PILImage.open(io.BytesIO(data))
        assert im.mode == "L"
        assert im.info.get("icc_profile") == cm.GRAY_SRGB_ICC_BYTES


def test_colour_jpeg_still_has_three_components(tmp_path):
    out = _write(SimpleNamespace(decoded_mono=False, color_profile="color"),
                 _tinted(), str(tmp_path / "c.jpg"), jpg=True)
    assert _jpeg_components(open(out, "rb").read()) == 3


def test_monochrome_write_preserves_the_neutral_value_exactly(tmp_path):
    """Rec.601 weights sum to 1.0, so collapsing equal channels is identity —
    the greyscale file carries the previewed tone, not a re-weighted one."""
    src = np.zeros((1, 4, 3), dtype=np.uint16)
    for i, v in enumerate((0, 999, 30000, 65535)):
        src[0, i] = (v, v, v)
    out = _write(SimpleNamespace(decoded_mono=True, color_profile="color"),
                 src, str(tmp_path / "exact.tiff"))
    np.testing.assert_array_equal(tifffile.imread(out), [[0, 999, 30000, 65535]])


def test_exported_monochrome_file_reimports_as_rgb(tmp_path):
    """Round trip: FreeCCR's own greyscale TIFF loads again as a neutral image."""
    out = _write(SimpleNamespace(decoded_mono=True, color_profile="color"),
                 _rgb(value=30000), str(tmp_path / "round.tiff"))
    img = CCRImage.__new__(CCRImage)
    img.source_ops = []
    img.input_transfer = None
    img.decoded_mono = False
    back = img.read_image(out, preview=False)
    assert back is not None and back.ndim == 3 and back.shape[2] == 3
    assert np.array_equal(back[..., 0], back[..., 1])
    assert img.decoded_mono is False          # a non-RAW read is never forced


# --------------------------------------------------------------------------- #
# 4. The monochrome RAW decode
# --------------------------------------------------------------------------- #

class _FakeSizes:
    def __init__(self, h, w):
        self.height, self.width = int(h), int(w)


class _FakeRaw:
    """Minimal rawpy stand-in: a 6x8 RGGB mosaic with a per-index black
    pedestal, plus a postprocess that returns a distinctly COLOURED frame so the
    two decode paths can never be confused."""

    def __init__(self, mosaic, colors, black, white=16383.0, mono_sensor=False):
        self.raw_image_visible = mosaic
        self.raw_colors_visible = colors
        self.black_level_per_channel = black
        self.white_level = white
        self.camera_whitebalance = [2.0, 1.0, 1.5, 0.0]
        self.sizes = _FakeSizes(*mosaic.shape[:2])
        self.num_colors = 1 if mono_sensor else 3
        self.color_desc = b"G" if mono_sensor else b"RGBG"
        self.raw_pattern = np.array([[0, 1], [3, 2]])
        self.postprocess_calls = []

    def postprocess(self, **kw):
        self.postprocess_calls.append(kw)
        h, w = self.raw_image_visible.shape[:2]
        if kw.get("half_size"):
            h, w = h // 2, w // 2
        out = np.zeros((h, w, 3), dtype=np.uint16)
        out[..., 0], out[..., 1], out[..., 2] = 3000, 2000, 1000
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


BLACK = [100, 200, 300, 400]


def _fake_raw(mono_sensor=False, mosaic_3d=False):
    colors = np.tile(np.array([[0, 1], [3, 2]]), (3, 4))        # 6x8 RGGB
    value = np.arange(48, dtype=np.float32).reshape(6, 8) * 100
    mosaic = (value + np.asarray(BLACK, dtype=np.float32)[colors]).astype(np.uint16)
    if mosaic_3d:
        mosaic = np.repeat(mosaic[..., None], 3, axis=2)        # e.g. linear DNG
    return _FakeRaw(mosaic, colors, BLACK, mono_sensor=mono_sensor), value


def _raw_stub():
    img = CCRImage.__new__(CCRImage)
    img.source_ops = []
    img.decoded_mono = False
    img.last_read_error = None
    return img


@pytest.fixture
def fake_rawpy(monkeypatch):
    """Patch rawpy.imread for the duration of a test; returns a setter."""
    holder = {}

    def _install(raw):
        holder["raw"] = raw
        monkeypatch.setattr(ccr_image_mod.rawpy, "imread",
                            lambda *a, **k: holder["raw"])
        return raw

    return _install


def test_mono_toggle_reads_the_whole_mosaic_at_full_resolution(fake_rawpy):
    raw, value = _fake_raw()
    fake_rawpy(raw)
    ccr_backend.mono_raw = True
    img = _raw_stub()

    out = img.read_image("frame.arw", preview=False)

    assert raw.postprocess_calls == []            # the mosaic read, not libraw
    assert out.shape == (6, 8, 3) and out.dtype == np.uint16
    assert img.decoded_mono is True
    assert img.original_full_size == (6, 8)
    # Three identical channels, each (mosaic - its black) * 65535/white_level.
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 0], out[..., 2])
    expect = np.clip(value * (65535.0 / raw.white_level), 0, 65535)
    assert np.abs(out[..., 0].astype(np.float32) - expect).max() <= 1.0
    # Exactly the shared ccr_merge read (one implementation, two features).
    plane = ccr_merge.mono_plane_from_mosaic(
        raw.raw_image_visible, raw.raw_colors_visible, BLACK)
    np.testing.assert_array_equal(plane, value)


def test_mono_preview_bins_2x2_but_reports_the_full_size(fake_rawpy):
    raw, _ = _fake_raw()
    fake_rawpy(raw)
    ccr_backend.mono_raw = True
    img = _raw_stub()

    out = img.read_image("frame.arw", preview=True)

    assert out.shape == (3, 4, 3)                  # binned
    assert img.original_full_size == (6, 8)        # canonical size unchanged
    assert img.decoded_mono is True


def test_toggle_off_keeps_the_colour_decode(fake_rawpy):
    raw, _ = _fake_raw()
    fake_rawpy(raw)
    ccr_backend.mono_raw = False
    img = _raw_stub()

    out = img.read_image("frame.arw", preview=False)

    assert raw.postprocess_calls, "the colour path must go through libraw"
    assert img.decoded_mono is False
    assert not np.array_equal(out[..., 0], out[..., 2])    # still colour


def test_true_mono_sensor_is_flagged_without_the_toggle(fake_rawpy):
    """A real monochrome sensor already decoded grey; it is now also DECLARED
    monochrome, so it renders grey and exports single-channel."""
    raw, _ = _fake_raw(mono_sensor=True)
    fake_rawpy(raw)
    ccr_backend.mono_raw = False
    img = _raw_stub()

    img.read_image("frame.arw", preview=False)

    assert img.decoded_mono is True
    assert raw.postprocess_calls                   # the existing mono path


def test_no_mosaic_falls_back_instead_of_failing(fake_rawpy):
    """A file with no 2-D sensor mosaic (linear DNG) must not break the import."""
    raw, _ = _fake_raw(mosaic_3d=True)
    fake_rawpy(raw)
    ccr_backend.mono_raw = True
    img = _raw_stub()

    out = img.read_image("linear.arw", preview=False)

    assert out is not None and out.ndim == 3
    assert raw.postprocess_calls                   # fell back to postprocess
    assert img.decoded_mono is True                # still rendered/written grey


# --------------------------------------------------------------------------- #
# 5. Forced-grey render
# --------------------------------------------------------------------------- #

def _render_stub(decoded_mono, settings=None, profile="color"):
    img = CCRImage.__new__(CCRImage)
    img.adjustment_settings = settings or {}
    img.color_profile = profile
    img.decoded_mono = decoded_mono
    img.contrast_base = 0
    img.temperature_base = 0
    img.brightness_base = 0
    img.exposure_base = 0
    img.converted = False
    img._ws_windowed = False
    img.tint_balance_factor = 1.0
    return img


def _is_neutral(img):
    return (np.array_equal(img[..., 0], img[..., 1])
            and np.array_equal(img[..., 1], img[..., 2]))


def test_mono_decode_renders_grey_even_with_colour_sliders():
    src = _tinted(8, 8)
    settings = {"temperature": 60, "saturation": 70, "tint": -40}
    out = _render_stub(True, settings).apply_adjustments(src)
    assert _is_neutral(out), "a monochrome decode must not render a tint"
    # The same adjustments on a colour image emphatically do tint it.
    assert not _is_neutral(_render_stub(False, settings).apply_adjustments(src))


def test_mono_decode_renders_grey_with_no_adjustments_at_all():
    """The no-adjustments early return has its own collapse — cover it too."""
    out = _render_stub(True).apply_adjustments(_tinted(8, 8))
    assert _is_neutral(out)


def test_renders_monochrome_method_matches_the_predicate():
    assert _render_stub(True).renders_monochrome() is True
    assert _render_stub(False).renders_monochrome() is False
    assert _render_stub(False, profile="bw").renders_monochrome() is True


# --------------------------------------------------------------------------- #
# 6. Settings dialog + MainWindow handler
# --------------------------------------------------------------------------- #

class _StubMW(QWidget):
    def __init__(self):
        super().__init__()
        self.mono_calls = []

    def on_mono_raw_toggled(self, checked):
        self.mono_calls.append(checked)


# Keep dialogs alive until exit — see the note in test_settings_dialog.py
# (collecting a parentless dialog mid-run corrupts the heap on Windows).
_LIVE_DIALOGS = []


def _dialog():
    from widgets.settings_dialog import SettingsDialog
    d = SettingsDialog(_StubMW())
    _LIVE_DIALOGS.append(d)
    return d


def test_dialog_seeds_the_checkbox_from_the_backend():
    ccr_backend.mono_raw = True
    assert _dialog()._cb_mono_raw.isChecked() is True
    ccr_backend.mono_raw = False
    assert _dialog()._cb_mono_raw.isChecked() is False


def test_dialog_applies_only_a_changed_checkbox():
    ccr_backend.mono_raw = False
    d = _dialog()
    d._apply_pending()
    assert d._mw.mono_calls == []                  # unchanged -> no call
    d._cb_mono_raw.setChecked(True)
    d._apply_pending()
    assert d._mw.mono_calls == [True]


def test_main_window_handler_sets_and_persists_the_flag():
    from ui.main_window import MainWindow
    stored = {}
    fake = SimpleNamespace(
        _settings=SimpleNamespace(setValue=stored.__setitem__),
        sliders_panel=SimpleNamespace(set_temporary_hint=lambda *a, **k: None))
    ccr_backend.images = []                        # no re-decode path to drive

    MainWindow.on_mono_raw_toggled(fake, True)
    assert ccr_backend.mono_raw is True
    assert stored == {"import/mono_raw": True}

    MainWindow.on_mono_raw_toggled(fake, False)
    assert ccr_backend.mono_raw is False
    assert stored["import/mono_raw"] is False


def test_backend_flag_defaults_off():
    assert isinstance(getattr(ccr_backend, "mono_raw"), bool)
    assert hasattr(ccr_backend, "reprocess_all_for_mono_raw_change")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

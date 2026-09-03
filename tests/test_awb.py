#!/usr/bin/env python3
"""Tests for Auto White Balance (spec/auto-white-balance.md).

AWB estimates the neutral (cast) color of the converted base with a classical,
learning-free algorithm and feeds it through compute_neutral_balance — the same
inverse the WB eyedropper uses — so the result lands on the R/G/B Balance
sliders (spec/channel-balance.md). Pure numpy core, runs headless.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # the apply-path tests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.awb import (  # noqa: E402
    AWB_ALGORITHMS, AWB_DEFAULT, AWB_LO, AWB_HI, AWB_TONE_LO, AWB_TONE_HI,
    estimate_neutral_rgb, compute_awb_balance,
)
from core.ccr_processor import (  # noqa: E402
    _balance_curve, apply_crop_to_image, compute_neutral_balance, encode_window,
)

CAST = (1.25, 1.0, 0.8)          # warm illuminant (R high, B low)


def _u16(d):
    """Full-range uint16 base from a display array in [0, 1]."""
    return (np.clip(np.asarray(d, dtype=np.float32), 0, 1) * 65535).astype(np.uint16)


def _cast_scene(rng, cast=CAST, shape=(64, 64), lo=0.15, hi=0.75):
    """A neutral random scene tinted by `cast` (channel means ∝ cast)."""
    g = rng.uniform(lo, hi, size=shape).astype(np.float32)
    return np.stack([g * c for c in cast], axis=-1)


def _ratios(triple):
    r, g, b = triple
    return r / g, b / g


# --- the estimators ---------------------------------------------------------

def test_gray_world_recovers_cast():
    d = _cast_scene(np.random.default_rng(0))
    est = estimate_neutral_rgb(_u16(d), ws_windowed=False, algorithm="gray_world")
    rg, bg = _ratios(est)
    assert rg == pytest.approx(CAST[0] / CAST[1], rel=0.01)
    assert bg == pytest.approx(CAST[2] / CAST[1], rel=0.01)


def test_white_patch_balances_on_brightest_content():
    """A bright cast-colored patch drives white_patch, not the dim scene mean."""
    cast = (1.1, 1.0, 0.85)
    d = np.full((100, 100, 3), 0.3, dtype=np.float32)
    d[..., 0] *= 0.9                                   # dim content: different cast
    patch = np.array([0.75 * c for c in cast], dtype=np.float32)   # inside the band
    d[:5, :, :] = patch                                # 5% brightest rows
    est = estimate_neutral_rgb(_u16(d), ws_windowed=False, algorithm="white_patch")
    rg, bg = _ratios(est)
    assert rg == pytest.approx(cast[0] / cast[1], rel=0.02)
    assert bg == pytest.approx(cast[2] / cast[1], rel=0.02)


def test_shades_of_gray_equals_gray_world_on_uniform():
    d = np.full((32, 32, 3), 0.5, dtype=np.float32) * np.array(
        [1.1, 1.0, 0.9], dtype=np.float32)
    gw = estimate_neutral_rgb(_u16(d), algorithm="gray_world")
    sog = estimate_neutral_rgb(_u16(d), algorithm="shades_of_gray")
    assert np.allclose(gw, sog, rtol=1e-4)


def test_gray_edge_recovers_cast_from_edges():
    """Luminance stripes tinted by the cast: every edge gradient carries the
    cast's channel ratios."""
    stripes = np.tile(
        np.repeat(np.array([0.3, 0.6], dtype=np.float32), 8), 4)   # 64 cols
    d = np.stack([np.tile(stripes * c, (64, 1)) for c in CAST], axis=-1)
    est = estimate_neutral_rgb(_u16(d), ws_windowed=False, algorithm="gray_edge")
    rg, bg = _ratios(est)
    assert rg == pytest.approx(CAST[0] / CAST[1], rel=0.03)
    assert bg == pytest.approx(CAST[2] / CAST[1], rel=0.03)


def test_unknown_algorithm_falls_back_to_gray_world():
    d = _cast_scene(np.random.default_rng(1))
    got = estimate_neutral_rgb(_u16(d), algorithm="not_a_real_algorithm")
    gw = estimate_neutral_rgb(_u16(d), algorithm="gray_world")
    assert got == gw


# --- masking ----------------------------------------------------------------

def test_out_of_bound_pixels_excluded():
    """Near-clip and near-black pixels must not skew the estimate: the dirty
    image's estimate equals the estimate over just its in-bound interior."""
    d = _cast_scene(np.random.default_rng(2), shape=(80, 80))
    dirty = d.copy()
    dirty[:4, :, :] = 0.995        # blown row block (> AWB_HI)
    dirty[-4:, :, :] = 0.005       # holder-black block (< AWB_LO)
    interior = estimate_neutral_rgb(_u16(d[4:-4]), algorithm="gray_world")
    got = estimate_neutral_rgb(_u16(dirty), algorithm="gray_world")
    assert np.allclose(interior, got, rtol=1e-4)


def _scan_frame(rng, cast=CAST, shape=(120, 120), band=20):
    """An UNCROPPED film scan: image content in the middle, the holder surround
    above it (masked to pure black in the scan → clipped white once inverted)
    and the clear film base / sprocket holes below (the scan's maximum → crushed
    black once inverted). Both carry scanner noise, so they straddle the
    extremes instead of sitting exactly on 1.0 / 0.0."""
    d = _cast_scene(rng, cast=cast, shape=shape)
    d[:band] = rng.uniform(0.95, 1.0, size=(band, shape[1], 1)).astype(np.float32)
    d[-band:] = rng.uniform(0.0, 0.05, size=(band, shape[1], 1)).astype(np.float32)
    return d


@pytest.mark.parametrize("algorithm",
                         ["gray_world", "white_patch", "shades_of_gray"])
def test_film_surround_never_votes(algorithm):
    """The user's case: on an uncropped scan the estimate must come from the
    image only — the pure-white holder and pure-black film base are film, not
    scene, and neither carries a cast. Noisy extremes (0.95-1.0 / 0.0-0.05) are
    the real-world shape of those regions and must be rejected wholesale."""
    d = _scan_frame(np.random.default_rng(20))
    full = estimate_neutral_rgb(_u16(d), algorithm=algorithm)
    content = estimate_neutral_rgb(_u16(d[20:-20]), algorithm=algorithm)
    assert full == content


def test_film_surround_leaves_the_cast_intact():
    """Stated as the property that matters: the surround must not drag the
    estimate toward neutral (its own color), for every algorithm."""
    d = _scan_frame(np.random.default_rng(21))
    for algorithm, _label in AWB_ALGORITHMS:
        rg, bg = _ratios(estimate_neutral_rgb(_u16(d), algorithm=algorithm))
        assert rg == pytest.approx(CAST[0] / CAST[1], rel=0.05), algorithm
        assert bg == pytest.approx(CAST[2] / CAST[1], rel=0.05), algorithm


def test_tone_band_excludes_extremes_that_pass_the_channel_gate():
    """The luminance band is a second, independent rejection: a bright (or
    deep) neutral surround whose channels all sit inside [AWB_LO, AWB_HI] is
    still outside the midtone band, so it does not vote."""
    d = _cast_scene(np.random.default_rng(22), shape=(100, 100))
    dirty = d.copy()
    dirty[:10] = 0.90        # lum 0.90 > AWB_TONE_HI, every channel < AWB_HI
    dirty[-10:] = 0.10       # lum 0.10 < AWB_TONE_LO, every channel > AWB_LO
    got = estimate_neutral_rgb(_u16(dirty), algorithm="gray_world")
    content = estimate_neutral_rgb(_u16(d[10:-10]), algorithm="gray_world")
    assert np.allclose(got, content, rtol=1e-4)


def test_insufficient_content_returns_none():
    d = np.full((100, 100, 3), 0.995, dtype=np.float32)   # all above AWB_HI
    assert estimate_neutral_rgb(_u16(d), algorithm="gray_world") is None


def test_windowed_and_full_range_agree():
    """The same display-domain scene, encoded windowed vs full-range, yields
    the same channel ratios (de-windowing is correct)."""
    d = _cast_scene(np.random.default_rng(3))
    ws = estimate_neutral_rgb(encode_window(d.copy()), ws_windowed=True,
                              algorithm="gray_world")
    full = estimate_neutral_rgb(_u16(d), ws_windowed=False,
                                algorithm="gray_world")
    assert _ratios(ws) == pytest.approx(_ratios(full), rel=5e-3)


# --- estimate → sliders → gains round-trip -----------------------------------

def test_roundtrip_neutralizes_the_cast():
    """Applying the AWB-computed Balance through the real Balance curve makes
    the estimated channel values meet (within int-slider quantization).

    The check is AT the estimated tone, which is what a tone-weighted control
    can neutralize — unlike the old flat WB gain, a Balance move does not
    neutralize every tone at once, by design."""
    d = _cast_scene(np.random.default_rng(4))
    est = estimate_neutral_rgb(_u16(d), algorithm="gray_world")
    vals = compute_neutral_balance(*est)
    got = [float(_balance_curve(s, v)) for s, v in zip(vals, est)]
    assert max(got) - min(got) < 0.005


def test_estimate_domain_is_normalised():
    """estimate_neutral_rgb must hand compute_neutral_balance NORMALISED [0,1]
    display units. Balance is tone-DEPENDENT, so (unlike the old ratio-only
    temp/tint inverse) the absolute scale is part of the answer."""
    est = estimate_neutral_rgb(_u16(_cast_scene(np.random.default_rng(5))),
                               algorithm="gray_world")
    assert all(0.0 <= v <= 1.0 for v in est)


# --- compute_awb_balance on an image-like object -----------------------------

class _StubImage:
    def __init__(self, base, ws=False, converted=True, crop=None, angle=0.0):
        self.resized_raw = base
        self._ws_windowed = ws
        self.converted = converted
        self.tint_balance_factor = 1.0
        self.adjustment_settings = {}
        self.crop_rect = crop
        self.crop_angle = angle


def test_compute_awb_balance_stub_image():
    d = _cast_scene(np.random.default_rng(5))
    res = compute_awb_balance(_StubImage(_u16(d)), algorithm="gray_world")
    assert res is not None
    br, bg, bb = res
    # Warm cast (R high, B low) → pull R down and push B up.
    assert br < 0 and bb > 0
    assert all(-100 <= v <= 100 for v in res)


def test_compute_awb_balance_no_base():
    assert compute_awb_balance(_StubImage(None), algorithm="gray_world") is None


# --- crop awareness -----------------------------------------------------------

def test_crop_region_drives_the_estimate():
    """Cast-A content inside the crop, strongly different cast-B junk outside:
    the cropped estimate matches the interior content, not the whole frame.

    Asserted on the ESTIMATE rather than the resulting slider values: the solve
    is now image-aware (Channel Levels + the Auto Gain offset run ahead of the
    tone-dependent Balance stage), and Auto Gain is measured from the whole
    base — as the render measures it — so a 128px frame and a bare 64px interior
    legitimately land on different sliders even from an identical estimate."""
    rng = np.random.default_rng(10)
    inside = _cast_scene(rng, cast=CAST, shape=(64, 64))
    d = _cast_scene(rng, cast=(0.7, 1.0, 1.3), shape=(128, 128))   # cool junk
    d[32:96, 32:96, :] = inside                                     # centered 50%
    crop = (0.25, 0.25, 0.75, 0.75)
    est_cropped = estimate_neutral_rgb(
        apply_crop_to_image(_u16(d), crop, 0.0), algorithm="gray_world")
    est_interior = estimate_neutral_rgb(_u16(inside), algorithm="gray_world")
    est_full = estimate_neutral_rgb(_u16(d), algorithm="gray_world")
    assert est_cropped == pytest.approx(est_interior, rel=1e-6)
    assert est_cropped != pytest.approx(est_full, rel=1e-3)
    # ...and the crop still changes what the user ends up with.
    assert (compute_awb_balance(_StubImage(_u16(d), crop=crop),
                                algorithm="gray_world")
            != compute_awb_balance(_StubImage(_u16(d)), algorithm="gray_world"))


def test_rotated_crop_black_fill_is_masked():
    """An angled crop samples outside the source (black fill); those pixels
    sit below AWB_LO and must not skew the estimate toward neutral-dark."""
    d = _cast_scene(np.random.default_rng(11), shape=(96, 96))
    straight = compute_awb_balance(
        _StubImage(_u16(d), crop=(0.1, 0.1, 0.9, 0.9)), algorithm="gray_world")
    angled = compute_awb_balance(
        _StubImage(_u16(d), crop=(0.1, 0.1, 0.9, 0.9), angle=8.0),
        algorithm="gray_world")
    # same scene statistics → within a slider unit of the straight crop
    assert abs(angled[0] - straight[0]) <= 1
    assert abs(angled[1] - straight[1]) <= 1


def test_stub_without_crop_attrs_still_works():
    """The hook may see images predating the crop feature — getattr defaults."""
    img = _StubImage(_u16(_cast_scene(np.random.default_rng(12))))
    del img.crop_rect, img.crop_angle
    assert compute_awb_balance(img, algorithm="gray_world") is not None


# --- the post-conversion hook (backend policy) --------------------------------

@pytest.fixture
def backend():
    from core.ccr_backend import ccr_backend
    saved = (ccr_backend.auto_awb, ccr_backend.awb_algorithm)
    yield ccr_backend
    ccr_backend.auto_awb, ccr_backend.awb_algorithm = saved


def test_hook_writes_when_unset(backend):
    backend.auto_awb = True
    backend.awb_algorithm = "gray_world"
    img = _StubImage(_u16(_cast_scene(np.random.default_rng(6))))
    backend.maybe_auto_awb(img)
    assert any(img.adjustment_settings.get(k, 0)
               for k in ("balance_r", "balance_g", "balance_b"))
    assert "balance_g" in img.adjustment_settings


@pytest.mark.parametrize("preset", [{"balance_r": 5}, {"balance_g": -3},
                                    {"balance_b": 2, "balance_r": 1}])
def test_hook_never_clobbers_saved_wb(backend, preset):
    backend.auto_awb = True
    img = _StubImage(_u16(_cast_scene(np.random.default_rng(7))))
    img.adjustment_settings = dict(preset)
    backend.maybe_auto_awb(img)
    assert img.adjustment_settings == preset


def test_hook_off_by_default_and_inert_when_off(backend):
    backend.auto_awb = False
    img = _StubImage(_u16(_cast_scene(np.random.default_rng(8))))
    backend.maybe_auto_awb(img)
    assert img.adjustment_settings == {}


def test_hook_skips_unconverted(backend):
    backend.auto_awb = True
    img = _StubImage(_u16(_cast_scene(np.random.default_rng(9))),
                     converted=False)
    backend.maybe_auto_awb(img)
    assert img.adjustment_settings == {}


# --- registry / defaults -------------------------------------------------------

def test_algorithm_registry():
    ids = [a for a, _ in AWB_ALGORITHMS]
    assert ids == ["gray_world", "white_patch", "shades_of_gray", "gray_edge"]
    assert AWB_DEFAULT in ids
    assert 0.0 < AWB_LO < AWB_HI < 1.0
    # The band sits strictly inside the channel gate: midtones + a slice of the
    # shadows and highlights, never the extremes an uncropped scan always has.
    assert AWB_LO < AWB_TONE_LO < AWB_TONE_HI < AWB_HI


def test_backend_defaults(backend):
    """Constructor defaults (the fixture restores any earlier mutation, so the
    singleton still carries them here): auto-AWB opt-in, Gray World."""
    assert backend.auto_awb is False
    assert backend.awb_algorithm == AWB_DEFAULT


# --- applying the result (the canvas must show it immediately) ----------------
# update_preview() paints the CACHED resized_preview and only regenerates it
# afterwards, so a discrete edit that merely queues the debounced reprocess
# leaves the canvas on the pre-edit render until something else redraws it.
# on_wb_sampled therefore has to regenerate FIRST, then display.

@pytest.fixture
def wb_panel(tmp_path):
    """A real SlidersPanel over one converted, cast-tinted image, hosted the way
    MainWindow hosts it (parent().parent() carries image_preview). Yields
    (panel, image, log) where log records render/show in the order they happen."""
    import cv2
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
    QApplication.instance() or QApplication(sys.argv[:1])
    from core.ccr_backend import ccr_backend
    from core.ccr_image import CCRImage
    from widgets.sliders_panel import SlidersPanel

    path = str(tmp_path / "scan.png")
    cv2.imwrite(path, np.full((40, 60, 3), 20000, np.uint16))
    img = CCRImage(path)
    img.converted = True
    img.adjustment_settings = {}
    img.resized_raw = _u16(_cast_scene(np.random.default_rng(30), shape=(40, 60)))

    log = []
    render = img.update_thumbnail_and_preview

    def _logged_render():
        log.append("render")
        return render()
    img.update_thumbnail_and_preview = _logged_render

    class _PreviewStub:
        def update_preview(self, idx):
            log.append("show")

    class _Host(QWidget):
        def __init__(self):
            super().__init__()
            self.image_preview = _PreviewStub()
            self.mid = QWidget(self)

    host = _Host()
    QVBoxLayout(host.mid).addWidget(SlidersPanel(host.mid))
    panel = host.mid.layout().itemAt(0).widget()
    saved = (ccr_backend.images, ccr_backend.file_paths)
    ccr_backend.images, ccr_backend.file_paths = [img], [img.file_path]
    panel.current_idx = 0
    log.clear()                      # ignore setup renders
    yield panel, img, log
    ccr_backend.images, ccr_backend.file_paths = saved


def test_wb_result_is_rendered_before_it_is_shown(wb_panel):
    """The eyedropper/AWB apply path: the new Balance values reach the settings
    AND the canvas is repainted from a fresh render, in that order."""
    panel, img, log = wb_panel
    panel.on_wb_sampled(-30, 12, 7)
    assert img.adjustment_settings["balance_r"] == -30
    assert img.adjustment_settings["balance_g"] == 12
    assert img.adjustment_settings["balance_b"] == 7
    assert "render" in log            # not just the stale cached preview
    assert log[-1] == "show"          # ...and the fresh render is what's shown


def test_wb_apply_leaves_no_pending_reprocess(wb_panel):
    """The debounced reprocess is cancelled — the final state was rendered now,
    so nothing is left queued to redo it."""
    panel, _img, _log = wb_panel
    panel.on_wb_sampled(-30, 12, 7)
    assert panel._pending_adjustment is None
    assert not panel._debounce_timer.isActive()


def test_auto_wb_button_applies_and_shows(wb_panel, backend):
    """End-to-end: the AWB button estimates the cast, writes the sliders and
    leaves a freshly rendered canvas."""
    panel, img, log = wb_panel
    backend.awb_algorithm = "gray_world"
    panel._on_auto_wb()
    # Warm cast (R high, B low) → R pulled down, B pushed up.
    assert img.adjustment_settings["balance_r"] < 0
    assert img.adjustment_settings["balance_b"] > 0
    assert "render" in log and log[-1] == "show"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

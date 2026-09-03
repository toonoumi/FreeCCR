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
    apply_crop_to_image, encode_window,
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

def test_estimate_domain_is_normalised():
    """estimate_neutral_rgb hands the solver NORMALISED [0,1] display units;
    compute_awb_balance encodes that back into the base domain as a 1-pixel
    sample area, so the closed loop can render it like any other pick."""
    est = estimate_neutral_rgb(_u16(_cast_scene(np.random.default_rng(5))),
                               algorithm="gray_world")
    assert all(0.0 <= v <= 1.0 for v in est)


# --- compute_awb_balance on an image-like object -----------------------------

# compute_awb_balance now finishes through CCRImage.solve_neutral_balance, which
# renders the sample area through the real pipeline — so these need a REAL image,
# not a duck-typed stub. The estimate half is still exercised directly against
# estimate_neutral_rgb above, where a bare array is the right input.

def _real_image(tmp_path, base, ws=False, converted=True, crop=None, angle=0.0,
                name="scan.png"):
    """A CCRImage carrying `base` as its converted base."""
    import cv2
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication(sys.argv[:1])
    from core.ccr_image import CCRImage
    path = str(tmp_path / name)
    cv2.imwrite(path, np.full((8, 8, 3), 20000, np.uint16))
    img = CCRImage(path)
    img.resized_raw = base
    img._ws_windowed = ws
    img.converted = converted
    img.adjustment_settings = {}
    img.crop_rect = crop
    img.crop_angle = angle
    return img


def test_compute_awb_balance_end_to_end(tmp_path):
    d = _cast_scene(np.random.default_rng(5))
    res = compute_awb_balance(_real_image(tmp_path, _u16(d)), algorithm="gray_world")
    assert res is not None
    br, bg, bb = res
    # Red is the anchor and never moves; a warm cast (R high, B low) is corrected
    # by raising green and blue onto red -- blue furthest, since it is furthest off.
    assert br == 0
    assert bg > 0 and bb > 0 and bb > bg
    assert all(-100 <= v <= 100 for v in res)


def test_compute_awb_balance_no_base(tmp_path):
    assert compute_awb_balance(_real_image(tmp_path, None),
                               algorithm="gray_world") is None


# --- crop awareness -----------------------------------------------------------

def test_crop_region_drives_the_estimate(tmp_path):
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
    assert (compute_awb_balance(_real_image(tmp_path, _u16(d), crop=crop, name="a.png"),
                                algorithm="gray_world")
            != compute_awb_balance(_real_image(tmp_path, _u16(d), name="b.png"),
                                   algorithm="gray_world"))


def test_rotated_crop_black_fill_is_masked(tmp_path):
    """An angled crop samples outside the source (black fill); those pixels
    sit below AWB_LO and must not skew the estimate toward neutral-dark."""
    d = _cast_scene(np.random.default_rng(11), shape=(96, 96))
    straight = compute_awb_balance(
        _real_image(tmp_path, _u16(d), crop=(0.1, 0.1, 0.9, 0.9), name="s.png"),
        algorithm="gray_world")
    angled = compute_awb_balance(
        _real_image(tmp_path, _u16(d), crop=(0.1, 0.1, 0.9, 0.9), angle=8.0,
                    name="ang.png"),
        algorithm="gray_world")
    # Same scene statistics, so the angled crop must land essentially where the
    # straight one did. Not bit-identical any more: the solve is a closed loop
    # resolving to INTEGER sliders, and rotation resamples the content, so a few
    # units of drift is expected — a skew from unmasked black fill would be a
    # collapse toward zero, not a nudge.
    assert abs(angled[1] - straight[1]) <= 5
    assert abs(angled[2] - straight[2]) <= 5
    assert any(abs(v) > 5 for v in angled), "black fill collapsed the correction"


def test_image_without_crop_attrs_still_works(tmp_path):
    """The hook may see images predating the crop feature — getattr defaults."""
    img = _real_image(tmp_path, _u16(_cast_scene(np.random.default_rng(12))))
    del img.crop_rect, img.crop_angle
    assert compute_awb_balance(img, algorithm="gray_world") is not None


# --- the post-conversion hook (backend policy) --------------------------------

@pytest.fixture
def backend():
    from core.ccr_backend import ccr_backend
    saved = (ccr_backend.auto_awb, ccr_backend.awb_algorithm)
    yield ccr_backend
    ccr_backend.auto_awb, ccr_backend.awb_algorithm = saved


def test_hook_writes_when_unset(backend, tmp_path):
    backend.auto_awb = True
    backend.awb_algorithm = "gray_world"
    img = _real_image(tmp_path, _u16(_cast_scene(np.random.default_rng(6))))
    backend.maybe_auto_awb(img)
    assert any(img.adjustment_settings.get(k, 0)
               for k in ("balance_r", "balance_g", "balance_b"))
    assert "balance_g" in img.adjustment_settings


@pytest.mark.parametrize("preset", [{"balance_r": 5}, {"balance_g": -3},
                                    {"balance_b": 2, "balance_r": 1}])
def test_hook_never_clobbers_saved_wb(backend, preset, tmp_path):
    backend.auto_awb = True
    img = _real_image(tmp_path, _u16(_cast_scene(np.random.default_rng(7))))
    img.adjustment_settings = dict(preset)
    backend.maybe_auto_awb(img)
    assert img.adjustment_settings == preset


def test_hook_off_by_default_and_inert_when_off(backend, tmp_path):
    backend.auto_awb = False
    img = _real_image(tmp_path, _u16(_cast_scene(np.random.default_rng(8))))
    backend.maybe_auto_awb(img)
    assert img.adjustment_settings == {}


def test_hook_skips_unconverted(backend, tmp_path):
    backend.auto_awb = True
    img = _real_image(tmp_path, _u16(_cast_scene(np.random.default_rng(9))),
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
    # Red is the anchor; a warm cast is corrected by raising green and blue.
    assert img.adjustment_settings["balance_r"] == 0
    assert img.adjustment_settings["balance_g"] > 0
    assert img.adjustment_settings["balance_b"] > 0
    assert "render" in log and log[-1] == "show"


# --- AWB as a whole-frame regression -----------------------------------------
#
# AWB no longer solves at one estimated pixel; it renders a downscaled copy of
# the frame through the real pipeline every iteration and drives the estimator's
# reading of THAT to grey. Two scale/direction bugs made the button silently
# dead, and both are pinned here.

def _frame(tmp_path, cast=(1.0, 0.93, 0.78), name="f.png"):
    import cv2
    from core.ccr_processor import encode_window
    rng = np.random.default_rng(5)
    lum = cv2.GaussianBlur(rng.normal(0.42, 0.16, size=(240, 360)).astype(np.float32),
                           (0, 0), 4)
    lum[:, -30:] = 0.92                                   # a highlight region
    lum = np.clip(lum, 0.03, 0.98)
    disp = np.clip(np.stack([lum * c for c in cast], axis=-1), 0, 1.2)
    img = _real_image(tmp_path, encode_window(disp), ws=True, name=name)
    return img


def _render_cast(img, settings):
    """In-gate midtone cast of the RENDER, as a percentage."""
    out = img.apply_adjustments(img.resized_raw.copy(), settings=settings)
    d = out.reshape(-1, 3).astype(np.float32) / 65535.0
    lum = d @ np.float32([0.299, 0.587, 0.114])
    m = (np.all((d >= AWB_LO) & (d <= AWB_HI), axis=1)
         & (lum >= AWB_TONE_LO) & (lum <= AWB_TONE_HI))
    assert m.sum() > 50
    e = d[m].mean(axis=0)
    return float((e.max() - e.min()) / e.max() * 100.0)


@pytest.mark.parametrize("algorithm", [a for a, _ in AWB_ALGORITHMS])
def test_every_algorithm_actually_corrects(tmp_path, algorithm):
    """The regression that mattered: the button must not silently do nothing.

    gray_edge got there two ways — its estimate is ~50x smaller than a pixel
    mean (an absolute convergence tolerance was already satisfied at the first
    measurement) and it reports gradient ENERGY, which FALLS as a channel is
    lifted, so a bisection assuming a rising response drove it backwards."""
    img = _frame(tmp_path, name=f"{algorithm}.png")
    res = compute_awb_balance(img, algorithm=algorithm)
    assert res is not None
    assert res != (0, 0, 0), f"{algorithm} did nothing"
    before = _render_cast(img, {})
    after = _render_cast(img, dict(balance_r=res[0], balance_g=res[1],
                                   balance_b=res[2]))
    assert before > 15.0
    assert after < 3.0, f"{algorithm}: {before:.1f}% -> {after:.1f}%"


@pytest.mark.parametrize("preset", [
    {},
    {"ch_r_shift": 15, "ch_g_gain": 12},
    {"ch_master_gain": 30},
    {"cineon_log": True},
])
def test_regression_holds_through_later_stages(tmp_path, preset):
    """It optimises the RENDER, so stages after Balance are accounted for."""
    img = _frame(tmp_path, name=f"p{abs(hash(str(preset))) % 9999}.png")
    img.adjustment_settings = dict(preset)
    res = compute_awb_balance(img, algorithm="gray_world")
    after = _render_cast(img, dict(preset, balance_r=res[0], balance_g=res[1],
                                   balance_b=res[2]))
    assert after < 3.0


def test_red_is_never_moved(tmp_path):
    img = _frame(tmp_path, name="red.png")
    assert compute_awb_balance(img, algorithm="gray_world")[0] == 0


def test_solve_sample_is_downscaled_but_spatially_intact(tmp_path):
    """gray_edge needs neighbours, so the whole-frame stand-in is a DOWNSCALE,
    not a scattered pixel sample."""
    from core.awb import downscale_for_solve, AWB_SOLVE_LONG_SIDE
    img = _frame(tmp_path, name="ds.png")
    small = downscale_for_solve(img.resized_raw)
    assert small.ndim == 3 and small.shape[2] == 3
    assert max(small.shape[:2]) == AWB_SOLVE_LONG_SIDE
    assert min(small.shape[:2]) > 1              # still a 2-D image, not a strip


def test_gray_edge_mask_selects_a_minority_of_pixels(tmp_path):
    """gray_edge is used as a pixel SELECTOR under the closed loop: its raw
    output is gradient energy, which is not a per-channel value statistic and
    cannot be driven to grey."""
    from core.awb import downscale_for_solve, gray_edge_pixel_mask
    img = _frame(tmp_path, name="ge.png")
    mask = gray_edge_pixel_mask(downscale_for_solve(img.resized_raw), True)
    assert mask is not None
    frac = mask.sum() / mask.size
    assert 0.01 < frac < 0.5


def test_result_is_idempotent(tmp_path):
    """The estimate runs on the base, so re-running AWB on an already-corrected
    image returns the same answer rather than compounding."""
    img = _frame(tmp_path, name="idem.png")
    first = compute_awb_balance(img, algorithm="gray_world")
    img.adjustment_settings = dict(balance_r=first[0], balance_g=first[1],
                                   balance_b=first[2])
    second = compute_awb_balance(img, algorithm="gray_world")
    assert second == pytest.approx(first, abs=2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

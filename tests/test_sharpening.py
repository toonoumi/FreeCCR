#!/usr/bin/env python3
"""Sharpening — the Details section (spec/sharpening.md).

A luma-only unsharp mask as the LAST stage of the adjustment chain, with a
VISIBLE non-zero default (25) so the user can see the grain and confirm focus.
The 1080 preview exaggerates the radius on purpose; export and zoom tiles are
exact.
"""
import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core import catalog  # noqa: E402
from core import ccr_image as ccr_image_mod  # noqa: E402
from core.ccr_processor import (SHARPEN_MAX_RADIUS, SHARPEN_RADIUS_DIV,  # noqa: E402
                                apply_sharpening)
from widgets.sliders_panel import SYNC_GROUPS, SlidersPanel  # noqa: E402


def _edge(v_lo=10000, v_hi=40000, n=64):
    img = np.full((n, n, 3), v_lo, dtype=np.uint16)
    img[:, n // 2:] = v_hi
    return img


# --- pure filter ---------------------------------------------------------------

def test_flat_field_is_returned_unchanged():
    """No high-pass energy to add -> nothing to do. This is what keeps
    sharpening from lifting noise in a smooth sky or a dense shadow."""
    flat = np.full((64, 64, 3), 20000, np.uint16)
    np.testing.assert_array_equal(apply_sharpening(flat, 25), flat)


def test_step_edge_gains_overshoot_on_both_sides():
    img = _edge()
    out = apply_sharpening(img, 50)
    n = img.shape[1] // 2
    assert out[32, n - 1, 0] < img[32, n - 1, 0]      # undershoot, dark side
    assert out[32, n, 0] > img[32, n, 0]              # overshoot, light side


def test_amount_zero_and_radius_zero_are_identity():
    img = _edge()
    np.testing.assert_array_equal(apply_sharpening(img, 0), img)
    np.testing.assert_array_equal(apply_sharpening(img, 50, radius=0), img)


def test_tiny_buffer_is_left_alone():
    tiny = np.full((3, 3, 3), 500, np.uint16)
    np.testing.assert_array_equal(apply_sharpening(tiny, 100), tiny)


# --- luma only -----------------------------------------------------------------

def test_sharpening_preserves_hue_and_saturation():
    """One scalar is added to R, G and B alike, so the channel DIFFERENCES that
    carry hue/saturation survive untouched. This is what stops an inverted
    negative's chroma noise being amplified in the stretched shadows."""
    img = np.zeros((64, 64, 3), np.uint16)
    img[:, :32] = [30000, 10000, 5000]
    img[:, 32:] = [12000, 4000, 2000]
    out = apply_sharpening(img, 80)
    before = img.astype(np.int32)
    after = out.astype(np.int32)
    assert not np.array_equal(out, img)                      # it did something
    np.testing.assert_array_equal(after[..., 0] - after[..., 1],
                                  before[..., 0] - before[..., 1])
    np.testing.assert_array_equal(after[..., 1] - after[..., 2],
                                  before[..., 1] - before[..., 2])


def test_monochrome_frame_stays_neutral():
    img = np.stack([_edge()[..., 0]] * 3, axis=-1)
    out = apply_sharpening(img, 70)
    np.testing.assert_array_equal(out[..., 0], out[..., 1])
    np.testing.assert_array_equal(out[..., 1], out[..., 2])


# --- radius scaling ------------------------------------------------------------

def _halo_width(out, img):
    return int((np.abs(out[32].astype(int) - img[32].astype(int)).sum(1) > 0).sum())


def test_preview_scale_widens_the_halo_and_native_is_exact():
    img = _edge()
    native = apply_sharpening(img, 50, scale=1.0)
    preview = apply_sharpening(img, 50, scale=5.55)
    assert _halo_width(preview, img) > _halo_width(native, img)
    # scale below 1 must never SHRINK below the native radius (clamped)
    np.testing.assert_array_equal(apply_sharpening(img, 50, scale=0.2), native)


def test_radius_is_clamped_so_a_small_buffer_cannot_halo_wildly():
    img = _edge(n=256)
    huge = apply_sharpening(img, 50, radius=100, scale=50.0)
    capped = apply_sharpening(img, 50,
                              radius=SHARPEN_MAX_RADIUS * SHARPEN_RADIUS_DIV,
                              scale=1.0)
    np.testing.assert_array_equal(huge, capped)


def test_radius_slider_25_is_one_native_pixel():
    assert 25.0 / SHARPEN_RADIUS_DIV == 1.0


# --- masking -------------------------------------------------------------------

def test_masking_spares_flat_areas_but_keeps_edges():
    rng = np.random.default_rng(0)
    img = _edge(n=128).astype(np.float32)
    img[:, :48] += rng.normal(0, 300, img[:, :48].shape)     # noisy flat region
    img = np.clip(img, 0, 65535).astype(np.uint16)
    n = img.shape[1] // 2

    off = apply_sharpening(img, 80, masking=0)
    on = apply_sharpening(img, 80, masking=100)

    def delta(out, cols):
        return float(np.abs(out[:, cols].astype(np.int32)
                            - img[:, cols].astype(np.int32)).mean())

    flat = slice(5, 40)
    edge = slice(n - 2, n + 2)
    assert delta(on, flat) < delta(off, flat) * 0.5    # flat area largely spared
    assert delta(on, edge) > delta(on, flat)           # the edge still sharpens


# --- defaults and backward compatibility ---------------------------------------

def test_slider_defaults_are_visible_25():
    assert SlidersPanel.SLIDER_DEFAULTS["sharpen_amount"] == 25
    assert SlidersPanel.SLIDER_DEFAULTS["sharpen_radius"] == 25


def test_keys_are_last_so_the_positional_zip_is_unchanged():
    keys = SlidersPanel.ADJUSTMENT_KEYS
    assert keys[-3:] == ["sharpen_amount", "sharpen_radius", "sharpen_masking"]
    assert keys[-4] == "band_feather"


def test_panel_has_one_slider_per_key_in_order():
    panel = SlidersPanel()
    assert len(panel.sliders) == len(SlidersPanel.ADJUSTMENT_KEYS)
    tail = dict(zip(SlidersPanel.ADJUSTMENT_KEYS, panel.sliders))
    assert tail["sharpen_amount"].value() == 25
    assert tail["sharpen_radius"].value() == 25
    assert tail["sharpen_masking"].value() == 0


def test_details_has_its_own_sync_group_and_keys_stay_partitioned():
    """Every adjustment key must belong to exactly one sync group. Sharpening
    gets its own rather than riding another: it is a per-capture judgement
    (focus, grain, how far the frame was downscaled), synced independently of
    colour."""
    groups = {gid: keys for gid, _label, keys in SYNC_GROUPS}
    assert tuple(groups["details"]) == ("sharpen_amount", "sharpen_radius",
                                        "sharpen_masking")
    grouped = [k for _gid, _label, keys in SYNC_GROUPS for k in keys]
    grouped = [k for k in grouped if k != "cineon_log"]
    assert sorted(grouped) == sorted(SlidersPanel.ADJUSTMENT_KEYS)
    assert len(grouped) == len(set(grouped))


def test_details_section_exists_and_is_collapsed_by_default():
    panel = SlidersPanel()
    assert hasattr(panel, "details_section")
    # CollapsibleSection defaults to collapsed: the toggle is unchecked and its
    # content widget is hidden.
    assert panel.details_section._toggle_btn.isChecked() is False
    assert panel.details_section._content.isVisible() is False
    assert "Details" in panel.details_section._toggle_btn.text()


def test_old_catalog_restores_with_sharpening_off(monkeypatch):
    """A catalog written before the Details section has no sharpen_* keys.
    Falling through to SLIDER_DEFAULTS would silently re-sharpen every scan ever
    made, so a pre-existing dict pins sharpening OFF."""
    captured = {}

    class _Stub:
        # _restore_image reads back a number of attributes it then re-assigns
        # (contrast_base and friends); a permissive stub keeps this test about
        # the sharpening compat rather than about CCRImage's full surface.
        def __init__(self, file_path, **kw):
            captured.update(kw.get("adjustment_settings") or {})

        def __getattr__(self, name):
            return lambda *a, **k: None      # value or no-op method, either way

    monkeypatch.setattr(ccr_image_mod, "CCRImage", _Stub)
    catalog._restore_image("x.arw", {"adjustment_settings": {"brightness": 7}})
    assert captured["brightness"] == 7
    assert captured["sharpen_amount"] == 0
    assert captured["sharpen_radius"] == 0


def test_catalog_round_trips_an_explicit_sharpening_value(monkeypatch):
    captured = {}

    class _Stub:
        # _restore_image reads back a number of attributes it then re-assigns
        # (contrast_base and friends); a permissive stub keeps this test about
        # the sharpening compat rather than about CCRImage's full surface.
        def __init__(self, file_path, **kw):
            captured.update(kw.get("adjustment_settings") or {})

        def __getattr__(self, name):
            return lambda *a, **k: None      # value or no-op method, either way

    monkeypatch.setattr(ccr_image_mod, "CCRImage", _Stub)
    catalog._restore_image("x.arw",
                           {"adjustment_settings": {"sharpen_amount": 60,
                                                    "sharpen_radius": 40}})
    assert captured["sharpen_amount"] == 60
    assert captured["sharpen_radius"] == 40


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

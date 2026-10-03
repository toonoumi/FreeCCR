#!/usr/bin/env python3
"""
Tests for full-resolution zoom (spec/full-res-zoom.md).

The zoom detail tile must be decoded at the resolution the CURRENT zoom needs —
the source's own at 100% — while never coming back softer than the legacy
half-size request, and without re-decoding when a sharp-enough tile is cached.
"""

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout,  # noqa: E402
                               QGraphicsPixmapItem)
from PySide6.QtGui import QPixmap, QColor  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from core.ccr_image import CCRImage  # noqa: E402
from widgets import image_preview as ip_mod  # noqa: E402
from widgets.image_preview import ImagePreview  # noqa: E402

CAP = ImagePreview.HIRES_MAX_LONG_SIDE          # 4500, the legacy cap/floor


# --------------------------------------------------------------------------- #
# The decode-request rule (pure: no Qt, no I/O)
# --------------------------------------------------------------------------- #
def _bare(full, path="frame.cr3", **kw):
    """A CCRImage carrying only what hires_decode_request reads."""
    s = CCRImage.__new__(CCRImage)
    s.original_full_size = full                 # (h, w)
    s.file_path = path
    s.is_merged = kw.get("is_merged", False)
    s.merge_sources = kw.get("merge_sources", None)
    s.merge_demosaic = kw.get("merge_demosaic", True)
    s.merge_mono = kw.get("merge_mono", False)
    return s


class TestDecodeRequest:
    """§3.2 — want_long -> (preview, max_long_side)."""

    def test_raw_at_100_percent_is_a_full_decode(self):
        img = _bare((5504, 8256))
        assert img.hires_decode_request(8256, CAP) == (False, 8256)

    def test_raw_at_fit_keeps_todays_resolution(self):
        # Very little is wanted (fitted view / dust mode), so the floor applies:
        # exactly the legacy half-size decode, never softer.
        img = _bare((5504, 8256))
        assert img.hires_decode_request(900, CAP) == (True, 4128)

    def test_raw_intermediate_zoom_needs_the_full_decode_downsized(self):
        img = _bare((5504, 8256))
        # 60%: more than the half decode carries, less than the whole source.
        assert img.hires_decode_request(4954, CAP) == (False, 4954)

    def test_raw_half_decode_still_chosen_at_exactly_half(self):
        img = _bare((5504, 8256))
        assert img.hires_decode_request(4128, CAP) == (True, 4128)

    def test_small_raw_at_100_percent(self):
        # A 1500px RAW halves to 750, so 100% needs the full decode.
        img = _bare((1000, 1500))
        assert img.hires_decode_request(1500, CAP) == (False, 1500)

    def test_non_raw_at_100_percent_lifts_the_legacy_cap(self):
        img = _bare((6667, 10000), path="scan.tif")
        # preview is ignored by the non-RAW decoder; the cap is what mattered.
        assert img.hires_decode_request(10000, CAP) == (True, 10000)

    def test_non_raw_at_fit_keeps_the_legacy_cap(self):
        img = _bare((6667, 10000), path="scan.tif")
        assert img.hires_decode_request(900, CAP) == (True, CAP)

    def test_non_raw_smaller_than_the_cap(self):
        img = _bare((2000, 3000), path="scan.tif")
        assert img.hires_decode_request(900, CAP) == (True, 3000)

    def test_zoom_past_100_percent_clamps_to_the_source(self):
        img = _bare((5504, 8256))
        # 400% cannot invent pixels.
        assert img.hires_decode_request(8256 * 4, CAP) == (False, 8256)

    def test_feature_off_is_the_legacy_request(self):
        img = _bare((5504, 8256))
        assert img.hires_decode_request(None, CAP) == (True, CAP)

    def test_unknown_source_size_is_the_legacy_request(self):
        img = _bare(None)
        assert img.hires_decode_request(8256, CAP) == (True, CAP)

    def test_merged_demosaic_honours_half_size(self):
        img = _bare((4000, 6000), is_merged=True, merge_sources=["a", "b", "c"])
        assert img.hires_decode_request(900, CAP) == (True, 3000)
        assert img.hires_decode_request(6000, CAP) == (False, 6000)

    def test_merged_photosite_read_has_no_half_size(self):
        # The single-photosite merge's half-sensor read IS its full resolution,
        # so there is no half tier below it — the floor is the legacy cap.
        img = _bare((4000, 6000), is_merged=True, merge_sources=["a", "b", "c"],
                    merge_demosaic=False)
        assert img.hires_decode_request(900, CAP) == (True, CAP)
        assert img.hires_decode_request(6000, CAP) == (True, 6000)


# --------------------------------------------------------------------------- #
# Headless ImagePreview: the zoom measurement and the request/cache logic
# --------------------------------------------------------------------------- #
class _StubImage:
    """Preview-state stub that uses the REAL decode-request rule."""

    hires_decode_request = CCRImage.hires_decode_request
    _honours_half_size = CCRImage._honours_half_size

    def __init__(self, source_w=8256, source_h=5504,
                 preview_w=1080, preview_h=720, path="frame.cr3"):
        self.original_full_size = (source_h, source_w)
        self.resized_raw = np.zeros((preview_h, preview_w, 3), np.uint16)
        self.converted = False
        self.file_path = path
        self.is_merged = False
        self.merge_sources = None
        self.merge_demosaic = True
        self.merge_mono = False
        # Read by _current_adj_sig
        self.adjustment_settings = {}
        self.contrast_base = 0
        self.temperature_base = 0
        self.brightness_base = 0
        self.exposure_base = 0.0
        self.base_curve = 0
        self.color_profile = "color"
        self.area_layers = []
        self.dust_spots = []
        self.crop_rect = None
        self.crop_angle = 0.0


class _Conn:
    def connect(self, *a, **kw):
        pass


class _FakeWorker:
    """Records the request instead of decoding anything."""

    instances = []

    def __init__(self, img, sig, adj_sig, base=None, max_long_side=None,
                 sprocket_alpha=None, parent=None, preview=True, res=None):
        self.req_img = img
        self.req_sig = sig
        self.req_adj_sig = adj_sig
        self.base = base
        self.cap = max_long_side
        self.preview = preview
        self.req_res = int(res if res is not None else (max_long_side or 0))
        self.finished_hires = _Conn()
        self.finished = _Conn()
        self._running = False
        _FakeWorker.instances.append(self)

    def isRunning(self):
        return self._running

    def start(self):
        self._running = True


class _StubPanel:
    def set_sliders_enabled(self, *a):
        pass


class _Host(QWidget):
    def __init__(self):
        super().__init__()
        self.sliders_panel = _StubPanel()
        self.mid = QWidget(self)


_HOSTS = []


@pytest.fixture
def fake_worker(monkeypatch):
    monkeypatch.setattr(ip_mod, "HiResDetailWorker", _FakeWorker)
    _FakeWorker.instances = []
    yield _FakeWorker
    _FakeWorker.instances = []


def _setup(source_w=8256, source_h=5504, preview_w=1080, preview_h=720,
           view_w=1300, view_h=900, path="frame.cr3"):
    """ImagePreview with a known preview pixmap, source size and viewport
    (the pattern from test_zoom_percent.py)."""
    ccr_backend.images = []
    host = _Host()
    _HOSTS.append(host)
    host_layout = QVBoxLayout(host)
    host_layout.setContentsMargins(0, 0, 0, 0)
    mid_layout = QVBoxLayout(host.mid)
    mid_layout.setContentsMargins(0, 0, 0, 0)
    host_layout.addWidget(host.mid)
    ip = ImagePreview(host.mid)
    mid_layout.addWidget(ip)
    host.resize(view_w, view_h)
    host.show()
    _app.processEvents()

    ccr_backend.images = [_StubImage(source_w, source_h, preview_w, preview_h, path)]
    ip.current_idx = 0
    ip._current_image_ref = ccr_backend.images[0]
    pm = QPixmap(preview_w, preview_h)
    pm.fill(QColor(128, 128, 128))
    ip.current_pixmap = pm
    ip.pixmap_item = QGraphicsPixmapItem(pm)
    ip.scene.addItem(ip.pixmap_item)
    ip._zoom = 1.0
    ip._fit_view_to_content()
    _app.processEvents()
    return ip


def _viewport_ok(ip):
    vp = ip.view.viewport().rect()
    return vp.width() > 50 and vp.height() > 50


class TestTargetFromZoom:
    """§3.1 — the measured target ties the derivation to the real geometry."""

    def test_100_percent_asks_for_the_whole_source(self, fake_worker):
        ip = _setup()
        if not _viewport_ok(ip):
            pytest.skip("offscreen viewport not sized")
        ip.zoom_to_percent(1.0)
        preview, target = ip._hires_target_long_side()
        assert preview is False
        assert target == pytest.approx(8256, abs=8)

    def test_fitted_view_asks_for_the_floor(self, fake_worker):
        ip = _setup()
        if not _viewport_ok(ip):
            pytest.skip("offscreen viewport not sized")
        ip.zoom_to_fit()
        assert ip._hires_target_long_side() == (True, 4128)

    def test_feature_off_is_the_legacy_request(self, fake_worker, monkeypatch):
        ip = _setup()
        if not _viewport_ok(ip):
            pytest.skip("offscreen viewport not sized")
        ip.zoom_to_percent(1.0)
        monkeypatch.setattr(ccr_backend, "full_res_zoom", False, raising=False)
        assert ip._hires_target_long_side() == (True, CAP)

    def test_unknown_source_size_is_the_legacy_request(self, fake_worker):
        ip = _setup()
        ccr_backend.images[0].original_full_size = None
        assert ip._hires_target_long_side() == (True, CAP)

    def test_image_without_the_rule_falls_back(self, fake_worker):
        ip = _setup()
        ccr_backend.images[0] = object()        # no hires_decode_request
        assert ip._hires_target_long_side() == (True, CAP)


class TestRequestAndCache:
    """§5.1-5.3 — escalate when the cached tile is too small, never otherwise."""

    @staticmethod
    def _zoom_to(ip, pct):
        ip.zoom_to_percent(pct)
        # Ignore the request the zoom itself made. The fakes never emit
        # `finished`, so they would otherwise sit in _hires_workers forever and
        # the in-flight dedup would suppress everything below (a real worker
        # discards itself on completion).
        _FakeWorker.instances = []
        ip._hires_workers.clear()

    @staticmethod
    def _cache(ip, res, with_base=True, adj_current=True):
        img = ccr_backend.images[0]
        pm = QPixmap(res, int(res * 2 / 3))
        pm.fill(QColor(64, 64, 64))
        ip._hires = {
            "img": img, "sig": ip._hires_signature(img),
            "adj_sig": ip._current_adj_sig() if adj_current else ("stale",),
            "base": np.zeros((4, 4, 3), np.uint16) if with_base else None,
            "sprocket_alpha": None, "full_pm": pm, "res": res,
            "display_pm": None, "crop_sig": None,
        }

    def test_tile_too_small_is_re_decoded(self, fake_worker):
        ip = _setup()
        if not _viewport_ok(ip):
            pytest.skip("offscreen viewport not sized")
        self._zoom_to(ip, 1.0)                  # target ~8256
        self._cache(ip, 4128)                   # only the half-size tile
        ip._maybe_request_hires()
        assert len(_FakeWorker.instances) == 1
        w = _FakeWorker.instances[0]
        assert w.base is None                   # the small base cannot be reused
        assert w.preview is False               # full decode
        assert w.cap == pytest.approx(8256, abs=8)

    def test_sharp_enough_tile_is_not_re_requested(self, fake_worker):
        ip = _setup()
        if not _viewport_ok(ip):
            pytest.skip("offscreen viewport not sized")
        self._zoom_to(ip, 1.0)
        self._cache(ip, 8256)
        ip._maybe_request_hires()
        assert _FakeWorker.instances == []

    def test_zoom_out_keeps_the_sharp_tile(self, fake_worker):
        ip = _setup()
        if not _viewport_ok(ip):
            pytest.skip("offscreen viewport not sized")
        self._zoom_to(ip, 1.0)
        self._cache(ip, 8256)
        self._zoom_to(ip, 0.5)                  # target drops to ~4128
        ip._maybe_request_hires()
        assert _FakeWorker.instances == []      # never downgraded

    def test_adjustment_change_reuses_the_big_base(self, fake_worker):
        ip = _setup()
        if not _viewport_ok(ip):
            pytest.skip("offscreen viewport not sized")
        self._zoom_to(ip, 1.0)
        self._cache(ip, 8256, adj_current=False)
        ip._maybe_request_hires()
        assert len(_FakeWorker.instances) == 1
        w = _FakeWorker.instances[0]
        assert w.base is not None               # re-adjust only, no decode
        assert w.req_res == 8256                # stands for the base's own res

    def test_matching_in_flight_render_suppresses_a_duplicate(self, fake_worker):
        ip = _setup()
        if not _viewport_ok(ip):
            pytest.skip("offscreen viewport not sized")
        self._zoom_to(ip, 1.0)
        img = ccr_backend.images[0]
        inflight = _FakeWorker(img, ip._hires_signature(img),
                               ip._current_adj_sig(), res=8256)
        inflight.start()
        ip._hires_workers.add(inflight)
        _FakeWorker.instances = []
        ip._maybe_request_hires()
        assert _FakeWorker.instances == []
        ip._hires_workers.discard(inflight)

    def test_in_flight_render_that_is_too_small_does_not_suppress(self, fake_worker):
        ip = _setup()
        if not _viewport_ok(ip):
            pytest.skip("offscreen viewport not sized")
        self._zoom_to(ip, 1.0)
        img = ccr_backend.images[0]
        inflight = _FakeWorker(img, ip._hires_signature(img),
                               ip._current_adj_sig(), res=4128)
        inflight.start()
        ip._hires_workers.add(inflight)
        _FakeWorker.instances = []
        ip._maybe_request_hires()
        assert len(_FakeWorker.instances) == 1
        ip._hires_workers.discard(inflight)


class TestRenderHiresBasePlumbing:
    """§4 — the preview flag reaches read_image, and defaults to half size."""

    @staticmethod
    def _stub():
        s = CCRImage.__new__(CCRImage)
        s.file_path = "frame.cr3"
        s.converted = False
        s.conversion_inputs = None
        s.seen = {}

        def _read(path, preview=True, max_long_side=None):
            s.seen = {"preview": preview, "cap": max_long_side}
            return np.zeros((8, 12, 3), np.uint16)

        s.read_image = _read
        return s

    def test_full_decode_is_forwarded(self):
        s = self._stub()
        out, alpha = s.render_hires_base(max_long_side=8256, preview=False)
        assert s.seen == {"preview": False, "cap": 8256}
        assert out is not None and alpha is None

    def test_default_stays_half_size(self):
        s = self._stub()
        s.render_hires_base(max_long_side=CAP)
        assert s.seen == {"preview": True, "cap": CAP}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

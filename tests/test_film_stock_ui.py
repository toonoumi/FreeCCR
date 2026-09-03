#!/usr/bin/env python3
"""
UI smoke tests for the film-stock slope selector in the sliders panel
(spec/film-stock-slopes.md §4): combo/save/delete gating per B/W-point state,
the slope-source label text, the save flow (name prompt → persist →
auto-select) and delete falling back to Default. Modal dialogs are
monkeypatched; the panel's QSettings is redirected to a temp ini so tests
never touch the user's real store.
"""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtWidgets import QApplication, QMessageBox, QInputDialog  # noqa: E402
from PySide6.QtCore import QSettings  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from core.film_stocks import decode_film_stocks  # noqa: E402
import widgets.sliders_panel as sliders_panel_mod  # noqa: E402
from widgets.sliders_panel import SlidersPanel, FILM_STOCK_DEFAULT_LABEL  # noqa: E402

BASE_BGR = (11385.0, 14569.0, 7022.0)
WHITE_BGR = (760.0, 420.0, 117.0)
PORTRA = {"name": "Portra 400", "slopes_bgr": [0.85, 0.65, 0.56],
          "black_bgr": None, "white_bgr": None, "created": ""}


@pytest.fixture
def panel(tmp_path):
    """A SlidersPanel whose film-stock QSettings lives in a temp ini, with a
    clean backend B/W state. Backend globals are restored afterwards."""
    p = SlidersPanel(None)
    p._settings = QSettings(str(tmp_path / "test.ini"), QSettings.IniFormat)
    p._film_stocks = []
    p._reload_film_stock_combo()
    ccr_backend.black_point_bgr = None
    ccr_backend.white_point_bgr = None
    ccr_backend.positive_mode = False
    ccr_backend.clear_film_stock()
    p._update_bwp_mode_label()
    yield p
    ccr_backend.black_point_bgr = None
    ccr_backend.white_point_bgr = None
    ccr_backend.clear_film_stock()


def _select_stock(panel, stocks, name):
    panel._film_stocks = list(stocks)
    panel._reload_film_stock_combo(select=name)


def test_default_only_combo_and_gating(panel):
    assert panel.film_stock_combo.count() == 1
    assert panel.film_stock_combo.itemText(0) == FILM_STOCK_DEFAULT_LABEL
    assert panel.film_stock_combo.isEnabled()          # selectable pre-sampling
    assert not panel.save_film_stock_btn.isEnabled()   # needs BOTH points
    assert not panel.delete_film_stock_btn.isEnabled() # Default not deletable
    # With no black point the label now names the no-anchor state rather than
    # showing nothing — converting unanchored is allowed (fixed-constant density
    # invert). See spec/no-anchor-convert.md.
    assert "density invert" in panel.bwp_mode_label.text().lower()


def test_black_only_states_and_label(panel):
    _select_stock(panel, [PORTRA], "Portra 400")
    ccr_backend.set_black_point(BASE_BGR)
    panel._update_bwp_mode_label()
    assert ccr_backend.film_stock_slopes == (0.85, 0.65, 0.56)
    # Name last: the label elides from the right (it sits in a fixed-width
    # panel), so the mode has to come first or a long stock name eats it.
    assert panel.bwp_mode_label.text() == \
        'Anchor source: film stock (black point only) — Portra 400'
    assert not panel.save_film_stock_btn.isEnabled()
    assert panel.delete_film_stock_btn.isEnabled()
    # Back to Default: backend cleared, label reverts, delete gated off.
    panel.film_stock_combo.setCurrentIndex(0)
    assert ccr_backend.film_stock_slopes is None
    assert panel.bwp_mode_label.text() == \
        "Anchor source: default slope (black point only)"
    assert not panel.delete_film_stock_btn.isEnabled()


def test_white_point_overrides_and_disables_selector(panel):
    _select_stock(panel, [PORTRA], "Portra 400")
    ccr_backend.set_black_point(BASE_BGR)
    ccr_backend.set_white_point(WHITE_BGR)
    panel._update_bwp_mode_label()
    assert panel.bwp_mode_label.text() == "Anchor source: white point (two-point)"
    assert not panel.film_stock_combo.isEnabled()
    assert panel.save_film_stock_btn.isEnabled()       # both points → can save
    assert not panel.delete_film_stock_btn.isEnabled()


def test_positive_mode_disables_everything(panel):
    ccr_backend.set_black_point(BASE_BGR)
    ccr_backend.set_white_point(WHITE_BGR)
    ccr_backend.positive_mode = True
    panel.set_negative_controls_enabled(False)
    assert not panel.film_stock_combo.isEnabled()
    assert not panel.save_film_stock_btn.isEnabled()
    assert not panel.delete_film_stock_btn.isEnabled()
    # Leaving positive mode re-applies the CONDITIONAL gating, not blanket-on.
    ccr_backend.positive_mode = False
    panel.set_negative_controls_enabled(True)
    assert not panel.film_stock_combo.isEnabled()      # white point still set
    assert panel.save_film_stock_btn.isEnabled()


def test_save_flow_persists_and_selects(panel, monkeypatch):
    ccr_backend.set_black_point(BASE_BGR)
    ccr_backend.set_white_point(WHITE_BGR)
    panel._update_bwp_mode_label()
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: ("Portra 400", True)))
    panel._on_save_film_stock()
    # Persisted (round-trips through the store) and auto-selected.
    stored = decode_film_stocks(
        panel._settings.value("convert/film_stocks", "", type=str))
    assert [s["name"] for s in stored] == ["Portra 400"]
    assert stored[0]["black_bgr"] == list(BASE_BGR)    # provenance kept
    assert panel.film_stock_combo.currentText() == "Portra 400"
    # The SELECTION is session-only (per roll) — never persisted; only the
    # preset list is. See spec §2 goals / §4.4.
    assert panel._settings.value("convert/film_stock_selected", "", type=str) \
        == ""
    # The saved slopes reproduce the pair (S = 1/log10(base/dense)).
    from core.ccr_processor import compute_density_slopes
    assert tuple(stored[0]["slopes_bgr"]) == \
        pytest.approx(compute_density_slopes(BASE_BGR, WHITE_BGR))
    # Clearing the white point activates the stock for the backend.
    ccr_backend.white_point_bgr = None
    panel._update_bwp_mode_label()
    assert panel.film_stock_combo.isEnabled()
    assert ccr_backend.film_stock_slopes == tuple(stored[0]["slopes_bgr"])


def test_save_rejects_unusable_pair(panel, monkeypatch):
    ccr_backend.set_black_point((100.0, 14569.0, 7022.0))   # B: base < dense
    ccr_backend.set_white_point(WHITE_BGR)
    warned = []
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: warned.append(a)))
    monkeypatch.setattr(QInputDialog, "getText",
                        staticmethod(lambda *a, **k: pytest.fail(
                            "name prompt reached with an unusable pair")))
    panel._on_save_film_stock()
    assert warned
    assert panel._film_stocks == []


def test_delete_falls_back_to_default(panel, monkeypatch):
    _select_stock(panel, [PORTRA], "Portra 400")
    ccr_backend.set_black_point(BASE_BGR)
    panel._update_bwp_mode_label()
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    panel._on_delete_film_stock()
    assert panel._film_stocks == []
    assert panel.film_stock_combo.currentText() == FILM_STOCK_DEFAULT_LABEL
    assert ccr_backend.film_stock_slopes is None
    assert decode_film_stocks(
        panel._settings.value("convert/film_stocks", "", type=str)) == []
    assert panel.bwp_mode_label.text() == \
        "Anchor source: default slope (black point only)"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

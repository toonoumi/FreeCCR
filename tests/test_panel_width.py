#!/usr/bin/env python3
"""
The right-hand side panels must FIT the width they are given.

Horizontal scrolling is off in these panels, so a row whose minimum width
exceeds the panel silently clips at the right edge — the failure mode that
produced this test (Convert All / Slice / the Color Profile dropdown were cut
off, and the longest slider label sat flush against the border). Minimum widths
depend on the system UI font, so these assertions are the guard rail: they fail
on the machine whose font no longer fits rather than shipping a clipped panel.

The theme's global QSS sets the control padding these widths come from, so the
panels are measured inside a real MainWindow with the theme applied — a bare
panel would report Qt's unstyled defaults instead. The window is never closed:
MainWindow.closeEvent raises a modal confirm-exit dialog that would hang.
"""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtGui import QFont, QFontMetrics  # noqa: E402
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel,  # noqa: E402
                               QScrollArea)

_app = QApplication.instance() or QApplication(sys.argv[:1])

from core.ccr_backend import ccr_backend  # noqa: E402
from ui import theme  # noqa: E402

theme.apply_theme(_app)

from ui.main_window import MainWindow  # noqa: E402
from widgets.sliders_panel import SlidersPanel  # noqa: E402

# The longest label sitting in a LABEL_COL_W gutter.
LONGEST_LABEL = "Subtracted Sat"

# How much wider a UI font the panels must absorb (see the stress test below).
FONT_STRESS = 1.15

PANEL_NAMES = ["sliders_panel", "dust_panel", "crop_panel"]


@pytest.fixture(scope="module")
def window():
    win = MainWindow()
    win.resize(1860, 1080)
    _app.processEvents()
    return win


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_panel_layout_fits_its_width(window, name):
    """A panel layout whose minimum exceeds the panel's fixed width clips."""
    panel = getattr(window, name)
    assert panel.width() == theme.PANEL_W
    need = panel.layout().minimumSize().width()
    assert need <= theme.PANEL_W, (
        f"{name} layout needs {need}px but the panel is {theme.PANEL_W}px wide "
        f"— its content will be clipped")


@pytest.mark.parametrize("name", PANEL_NAMES)
def test_scroll_content_fits_the_viewport(window, name):
    """Scroll content wider than its viewport is unreachable, not scrollable:
    these areas have the horizontal scrollbar switched off."""
    panel = getattr(window, name)
    for area in panel.findChildren(QScrollArea):
        content = area.widget()
        if content is None:
            continue
        need = content.minimumSizeHint().width()
        assert need <= area.viewport().width(), (
            f"{name} scroll content needs {need}px, viewport is "
            f"{area.viewport().width()}px")


def test_no_sliders_row_exceeds_the_content_budget(window):
    """Every row of the sliders panel must fit inside the panel minus its
    content margins."""
    budget = theme.PANEL_W - 2 * theme.GAP_PANEL
    layout = window.sliders_panel.findChild(QScrollArea).widget().layout()
    need, idx = max((layout.itemAt(i).minimumSize().width(), i)
                    for i in range(layout.count()))
    assert need <= budget, (
        f"row {idx} needs {need}px, the content budget is {budget}px")


def test_label_gutter_fits_the_longest_label():
    """LABEL_COL_W is a fixed width and QLabel clips rather than elides, so the
    longest label must fit with room to spare."""
    text_w = QFontMetrics(QLabel().font()).horizontalAdvance(LONGEST_LABEL)
    assert text_w <= theme.LABEL_COL_W - theme.GAP_TIGHT, (
        f"{LONGEST_LABEL!r} measures {text_w}px in this UI font; "
        f"LABEL_COL_W is {theme.LABEL_COL_W}px")


def test_panel_survives_a_wider_ui_font():
    """The headless test font is narrower than a real desktop UI font, so
    fitting it proves little on its own — this is what makes the guard bite.

    A panel sized to fit exactly one machine's font clips on the next one (how
    the original bug shipped). Re-measure with the UI font enlarged by
    FONT_STRESS and require everything to still fit.
    """
    base = _app.font()
    stressed = QFont(base)
    stressed.setPointSizeF(base.pointSizeF() * FONT_STRESS)
    try:
        _app.setFont(stressed)
        panel = SlidersPanel(None)
        panel.setFixedWidth(theme.PANEL_W)
        panel.resize(theme.PANEL_W, 1200)
        _app.processEvents()

        budget = theme.PANEL_W - 2 * theme.GAP_PANEL
        layout = panel.findChild(QScrollArea).widget().layout()
        need, idx = max((layout.itemAt(i).minimumSize().width(), i)
                        for i in range(layout.count()))
        assert need <= budget, (
            f"at {FONT_STRESS:.2f}x the UI font, row {idx} needs {need}px but "
            f"the content budget is {budget}px — PANEL_W has no slack")

        text_w = QFontMetrics(stressed).horizontalAdvance(LONGEST_LABEL)
        assert text_w <= theme.LABEL_COL_W, (
            f"at {FONT_STRESS:.2f}x the UI font {LONGEST_LABEL!r} measures "
            f"{text_w}px, LABEL_COL_W is {theme.LABEL_COL_W}px")
    finally:
        _app.setFont(base)


def test_long_combo_entry_does_not_widen_the_panel(window):
    """A user-named film stock is arbitrarily long; the combo must elide it
    instead of pushing the panel's content past its edge."""
    panel = window.sliders_panel
    combo = panel.film_stock_combo
    layout = panel.findChild(QScrollArea).widget().layout()
    before = layout.minimumSize().width()
    combo.addItem("Kodak Portra 400 pushed two stops, 2026 batch")
    combo.setCurrentIndex(combo.count() - 1)
    _app.processEvents()
    after = layout.minimumSize().width()
    combo.removeItem(combo.count() - 1)
    assert after == before <= theme.PANEL_W, (
        f"a long combo entry widened the content minimum from {before} to {after}")


@pytest.mark.parametrize("attr", ["film_stock_combo", "color_profile_combo"])
def test_panel_combos_are_shrinkable(window, attr):
    combo = getattr(window.sliders_panel, attr)
    assert combo.sizeAdjustPolicy() == QComboBox.AdjustToMinimumContentsLengthWithIcon

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

Two kinds of test live here, and only one of them can run everywhere:

* **Fit budgets** — "this row fits the panel" — are absolute pixel claims, so
  they are only meaningful when the widths come from a real UI font. Qt no
  longer ships fonts, and the offscreen platform on a machine with no
  fontconfig (Windows, typically) has *none*: QFontDatabase is empty and every
  measurement falls back to a last-resort face roughly twice as wide as a real
  one — "Subtracted Sat" measures 168px there against 76px in Segoe UI. Held to
  that, PANEL_W would have to exceed 500px to pass, for a font no user will
  ever see. These tests carry @needs_ui_font and skip where there is none.
* **Behavioural claims** — "a long name does not widen the panel", "the label
  elides" — hold under any font, including the fallback, and are ungated. They
  are what actually covers shrinkable_combo / ElidingLabel / ElidingComboBox,
  so the guard keeps biting on every platform.

To measure the fit budgets on Windows anyway, run this file against the real
platform: QT_QPA_PLATFORM=windows pytest tests/test_panel_width.py
"""
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from PySide6.QtGui import QFont, QFontDatabase, QFontMetrics  # noqa: E402
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

# An empty font database means Qt resolved every request to its last-resort
# fallback; nothing measured against it says anything about a real desktop.
needs_ui_font = pytest.mark.skipif(
    not QFontDatabase.families(),
    reason="no fonts installed on this platform (Qt ships none) — pixel budgets "
           "would be measured against a last-resort fallback face")


def _fitting_width(widget_font_source, text, slack):
    """A render width the text is guaranteed to FIT in, whatever the font.

    Hard-coding it assumes a font: under the fontless fallback the "short"
    strings below measure ~450px, so a fixed 324px box elides them and the
    fits-fine comparisons compare the wrong thing.
    """
    return QFontMetrics(widget_font_source.font()).horizontalAdvance(text) + slack


@pytest.fixture(scope="module")
def window():
    win = MainWindow()
    win.resize(1860, 1080)
    _app.processEvents()
    return win


@needs_ui_font
@pytest.mark.parametrize("name", PANEL_NAMES)
def test_panel_layout_fits_its_width(window, name):
    """A panel layout whose minimum exceeds the panel's fixed width clips."""
    panel = getattr(window, name)
    assert panel.width() == theme.PANEL_W
    need = panel.layout().minimumSize().width()
    assert need <= theme.PANEL_W, (
        f"{name} layout needs {need}px but the panel is {theme.PANEL_W}px wide "
        f"— its content will be clipped")


@needs_ui_font
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


@needs_ui_font
def test_no_sliders_row_exceeds_the_content_budget(window):
    """Every row of the sliders panel must fit inside the panel minus its
    content margins."""
    budget = theme.PANEL_W - 2 * theme.GAP_PANEL
    layout = window.sliders_panel.findChild(QScrollArea).widget().layout()
    need, idx = max((layout.itemAt(i).minimumSize().width(), i)
                    for i in range(layout.count()))
    assert need <= budget, (
        f"row {idx} needs {need}px, the content budget is {budget}px")


@needs_ui_font
def test_label_gutter_fits_the_longest_label():
    """LABEL_COL_W is a fixed width and QLabel clips rather than elides, so the
    longest label must fit with room to spare."""
    text_w = QFontMetrics(QLabel().font()).horizontalAdvance(LONGEST_LABEL)
    assert text_w <= theme.LABEL_COL_W - theme.GAP_TIGHT, (
        f"{LONGEST_LABEL!r} measures {text_w}px in this UI font; "
        f"LABEL_COL_W is {theme.LABEL_COL_W}px")


@needs_ui_font
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


def test_long_camera_profile_does_not_widen_the_sidebar(window):
    """Camera-profile entries are user-imported ICC/DCP filenames of any
    length. The sidebar is a fixed width, so the combo must elide rather than
    push the layout's minimum past it."""
    sidebar = window.thumbnail_list
    combo = sidebar.profile_combo
    before = sidebar.layout.minimumSize().width()
    # blockSignals: selecting an entry fires _on_profile_combo_changed, which
    # tries to activate the profile and raises a modal warning for a bogus
    # path — that hangs a headless run. refresh_profile_combo() blocks the same
    # way when it repopulates.
    combo.blockSignals(True)
    combo.addItem("Canon EOS R5 Adobe Standard v3 daylight  (DCP)", "x")
    combo.setCurrentIndex(combo.count() - 1)
    combo.blockSignals(False)
    _app.processEvents()
    after = sidebar.layout.minimumSize().width()
    combo.blockSignals(True)
    combo.removeItem(combo.count() - 1)
    combo.blockSignals(False)
    # The behavioural claim — adding the name changed nothing — holds under any
    # font. Only the "and it fits" tail below needs a real one.
    assert after == before, (
        f"a long camera-profile name widened the sidebar minimum from "
        f"{before} to {after}")
    if QFontDatabase.families():
        assert before <= sidebar.width(), (
            f"the sidebar minimum is {before}px but the sidebar is "
            f"{sidebar.width()}px")


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
    assert after == before, (
        f"a long combo entry widened the content minimum from {before} to {after}")
    if QFontDatabase.families():
        assert before <= theme.PANEL_W, (
            f"the content minimum is {before}px, the panel is {theme.PANEL_W}px")


@pytest.mark.parametrize("attr", ["film_stock_combo", "color_profile_combo"])
def test_panel_combos_are_shrinkable(window, attr):
    combo = getattr(window.sliders_panel, attr)
    assert combo.sizeAdjustPolicy() == QComboBox.AdjustToMinimumContentsLengthWithIcon


# --- eliding: the combos that carry user-supplied names ---------------------
# Width is only half the problem. Qt does not elide a non-editable QComboBox —
# it draws the label into the edit-field rect and lets the glyphs run under the
# arrow, so a long film-stock or camera-profile name reads as
# "Fuji Superia X-TRA 40" with nothing to say it was cut.

LONG_NAME = "Fuji Superia X-TRA 400 — expired 2019 batch"
SHORT_NAME = "Default"


# Wide enough for SHORT_NAME plus the frame and drop-down arrow the style puts
# around it, so "text that fits" is true by construction in any font.
_COMBO_CHROME = 60


def _render(cls, text, *, enabled=True, width=None):
    combo = cls()
    combo.addItem(text)
    combo.setEnabled(enabled)
    if width is None:
        width = _fitting_width(combo, SHORT_NAME, _COMBO_CHROME)
    combo.setFixedSize(width, theme.CONTROL_H)
    _app.processEvents()
    return combo.grab().toImage()


@pytest.mark.parametrize("enabled", [True, False])
def test_eliding_combo_matches_qt_exactly_for_text_that_fits(enabled):
    """Painting the label by hand must not change anything but the eliding.

    The custom paintEvent re-draws the label itself, so it can silently drift
    from Qt on placement (an inset shifts short labels out of line with the
    rest of the row) or on colour (losing the disabled group). Pixel equality
    in the fits-fine case is the guard, checked in both enabled states.
    """
    assert (_render(QComboBox, SHORT_NAME, enabled=enabled)
            == _render(theme.ElidingComboBox, SHORT_NAME, enabled=enabled)), (
        "text that fits must render exactly as Qt draws it")


@pytest.mark.parametrize("enabled", [True, False])
def test_eliding_combo_shortens_text_that_does_not_fit(enabled):
    assert (_render(QComboBox, LONG_NAME, enabled=enabled)
            != _render(theme.ElidingComboBox, LONG_NAME, enabled=enabled)), (
        "text too long for the box must be elided, not clipped like Qt's")


@pytest.mark.parametrize("owner,attr", [
    ("sliders_panel", "film_stock_combo"),      # user-named film stocks
    ("thumbnail_list", "profile_combo"),        # user-imported ICC/DCP names
])
def test_user_named_combos_elide(window, owner, attr):
    combo = getattr(getattr(window, owner), attr)
    assert isinstance(combo, theme.ElidingComboBox), (
        f"{owner}.{attr} holds arbitrary user text and must elide it")


def test_selecting_a_film_stock_does_not_widen_the_panel(window):
    """The status line under the combo NAMES the selected stock.

    Keeping the combo itself narrow is only half of it: a plain QLabel's
    minimum width is its whole text, so this line was enough on its own to push
    the scroll content ~200px past the panel and clip every row in it — the
    combo looked innocent because it already elides.
    """
    panel = window.sliders_panel
    layout = panel.findChild(QScrollArea).widget().layout()
    before = layout.minimumSize().width()
    saved = (ccr_backend.black_point_bgr, ccr_backend.white_point_bgr,
             ccr_backend.film_stock_name)
    try:
        ccr_backend.black_point_bgr = (100.0, 100.0, 100.0)
        ccr_backend.white_point_bgr = None
        ccr_backend.film_stock_name = "Kodak Portra 400 pushed two stops, 2026 batch"
        panel._update_bwp_mode_label()
        _app.processEvents()
        assert panel.bwp_mode_label.text(), "the status line should be showing"
        after = layout.minimumSize().width()
    finally:
        (ccr_backend.black_point_bgr, ccr_backend.white_point_bgr,
         ccr_backend.film_stock_name) = saved
        panel._update_bwp_mode_label()
    assert after == before, (
        f"naming a film stock widened the content minimum from {before} to "
        f"{after}")
    if QFontDatabase.families():
        assert before <= theme.PANEL_W, (
            f"the content minimum is {before}px, the panel is {theme.PANEL_W}px")


def test_long_hint_word_does_not_widen_the_panel(window):
    """The hint label wraps, which bounds it at its longest WORD — and a film
    stock named without spaces is a single word."""
    panel = window.sliders_panel
    before = panel.minimumSizeHint().width()
    try:
        panel.set_temporary_hint(
            'Film stock "Kodak_Portra_400_pushed_two_stops_2026_batch" deleted.',
            duration=60000)
        _app.processEvents()
        after = panel.minimumSizeHint().width()
    finally:
        panel.hint_label.setText("")
    assert after == before, (
        f"an unbreakable hint word widened the panel minimum from {before} to "
        f"{after}")
    if QFontDatabase.families():
        assert before <= theme.PANEL_W, (
            f"the panel minimum is {before}px, the panel is {theme.PANEL_W}px")


# The status line elides the same way the combos do, for the same reason: it
# carries a user-supplied name in a panel that cannot scroll sideways.

LONG_STATUS = ('Anchor source: film stock (black point only) — '
               'Kodak Portra 400 pushed two stops, 2026 batch')
SHORT_STATUS = "Anchor source: white point (two-point)"


def _render_label(cls, text, *, width=None):
    label = cls()
    label.setText(text)
    if width is None:
        width = _fitting_width(label, SHORT_STATUS, theme.GAP_TIGHT * 4)
    label.setFixedSize(width, theme.CONTROL_H)
    _app.processEvents()
    return label.grab().toImage()


def test_eliding_label_matches_qt_exactly_for_text_that_fits():
    """Drawing the text by hand must change nothing but the eliding — placement
    and colour both come from the label, and either can silently drift."""
    assert (_render_label(QLabel, SHORT_STATUS)
            == _render_label(theme.ElidingLabel, SHORT_STATUS)), (
        "text that fits must render exactly as QLabel draws it")


def test_eliding_label_shortens_text_that_does_not_fit():
    assert (_render_label(QLabel, LONG_STATUS)
            != _render_label(theme.ElidingLabel, LONG_STATUS)), (
        "text too long for the label must be elided, not clipped")


def test_eliding_label_keeps_the_full_text_in_its_tooltip():
    """Elision hides part of a name the user chose — the tooltip is the only
    place it stays readable."""
    label = theme.ElidingLabel()
    label.setText(LONG_STATUS)
    assert label.toolTip() == LONG_STATUS


def test_eliding_combo_leaves_the_popup_list_intact(window):
    """Elision is display-only: the dropdown must still offer full names."""
    combo = window.sliders_panel.film_stock_combo
    combo.blockSignals(True)
    combo.addItem(LONG_NAME)
    combo.blockSignals(False)
    assert combo.itemText(combo.count() - 1) == LONG_NAME
    combo.blockSignals(True)
    combo.removeItem(combo.count() - 1)
    combo.blockSignals(False)

"""Channel Levels panel placement (spec/channel-levels-pre-clamp.md §4).

The section moved from the bottom of the panel (below Curves and Subtractive
Saturations) to directly under the Convert row and above the WB/AWB row, so the
panel reads top-to-bottom in pipeline order — Channel Levels is now the FIRST
adjustment stage.

Moving the section widget must NOT disturb the positional ADJUSTMENT_KEYS zip:
placement and slider population are decoupled, and the last test here is the
guard on that.
"""
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication(sys.argv[:1])

from widgets.sliders_panel import SlidersPanel  # noqa: E402

CHANNEL_KEYS = (
    "ch_input_gain", "ch_master_shift", "ch_master_gain",
    "ch_r_shift", "ch_r_gain", "ch_r_blackpoint",
    "ch_g_shift", "ch_g_gain", "ch_g_blackpoint",
    "ch_b_shift", "ch_b_gain", "ch_b_blackpoint",
)

# Slider labels, in the create_slider() order the channel keys are zipped to.
CHANNEL_LABELS = (
    "Input Gain", "Master Shift", "Master Gain",
    "R Shift", "R Gain", "R Blackpoint",
    "G Shift", "G Gain", "G Blackpoint",
    "B Shift", "B Gain", "B Blackpoint",
)


@pytest.fixture(scope="module")
def panel():
    return SlidersPanel()


def _scroll_layout(panel):
    """The QVBoxLayout the sections and button rows are added to."""
    return panel.od_section.parentWidget().layout()


def _index_of(layout, widget):
    """Index in `layout` of the item that is `widget`, or of the nested row
    layout that contains it. -1 when absent."""
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget() is widget:
            return i
        sub = item.layout()
        if sub is not None:
            for j in range(sub.count()):
                if sub.itemAt(j).widget() is widget:
                    return i
    return -1


def test_section_sits_below_convert_and_above_awb(panel):
    layout = _scroll_layout(panel)
    convert = _index_of(layout, panel.convert_current_bwp_btn)
    section = _index_of(layout, panel.od_section)
    awb = _index_of(layout, panel.auto_wb_btn)

    assert convert >= 0 and section >= 0 and awb >= 0
    assert convert < section < awb


def test_section_is_above_curves_and_bands(panel):
    layout = _scroll_layout(panel)
    section = _index_of(layout, panel.od_section)
    assert section < _index_of(layout, panel.curves_section)
    assert section < _index_of(layout, panel.band_section)


def test_section_starts_collapsed(panel):
    # CollapsibleSection hides its content until the header is clicked.
    assert not panel.od_section._content.isVisible()
    assert not panel.od_section._toggle_btn.isChecked()


def test_every_channel_slider_defaults_to_zero(panel):
    for key in CHANNEL_KEYS:
        idx = panel.adjustment_keys.index(key)
        assert panel.sliders[idx].value() == 0, key
        assert panel._default_for(key) == 0, key


def _row_label_for(panel, slider):
    """The text of the label in `slider`'s row inside the Channel Levels
    section (create_slider puts the label first in the row)."""
    content = panel.od_section._content_layout
    for i in range(content.count()):
        row = content.itemAt(i).layout()
        if row is None:
            continue
        widgets = [row.itemAt(j).widget() for j in range(row.count())]
        if slider in widgets:
            return widgets[0].text()
    return None


def test_adjustment_keys_still_map_to_the_right_sliders(panel):
    """The positional zip survived the move: each channel key must still land on
    the slider carrying its label, and that slider must live in this section."""
    for key, label in zip(CHANNEL_KEYS, CHANNEL_LABELS):
        idx = panel.adjustment_keys.index(key)
        assert _row_label_for(panel, panel.sliders[idx]) == label, (key, label)


def test_channel_keys_precede_band_keys(panel):
    """Channel sliders must still be created before the band sliders."""
    last_channel = max(panel.adjustment_keys.index(k) for k in CHANNEL_KEYS)
    first_band = min(i for i, k in enumerate(panel.adjustment_keys)
                     if k.startswith("band_"))
    assert last_channel < first_band

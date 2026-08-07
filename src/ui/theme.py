"""FreeCCR UI theme — single source of truth for color, spacing, radius, type.

Neutral dark, color-critical (no hue cast so the UI never biases perception of
image color). Installed once at app start via ``apply_theme(app)`` (see
``src/main.py``). Pure PySide6 — imports no FreeCCR widgets, so any widget may
``from ui.theme import ...`` without an import cycle.

See ``spec/visual-redesign.md`` for the design rationale and token table.
"""

import os
import sys
import tempfile

from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QColor, QPalette, QIcon, QPixmap, QPainter, QPen
from PySide6.QtWidgets import QComboBox, QLabel, QWidget

# --------------------------------------------------------------------------- #
# 3.1 Surfaces (neutral grays — zero hue)
# --------------------------------------------------------------------------- #
WINDOW = "#1e1e1e"          # app background, menu bar
CANVAS = "#1a1a1a"          # image viewport behind the photo (QGraphicsView)
PANEL = "#2a2a2a"           # side panels, dialogs, group backgrounds
SURFACE = "#333333"         # inputs, combos, line edits, unchecked buttons
SURFACE_HOVER = "#3c3c3c"   # hover state
SURFACE_ACTIVE = "#454545"  # pressed / checked / selected background
BORDER = "#3c3c3c"          # 1px control borders, separators
BORDER_STRONG = "#5a5a5a"   # emphasized borders

# 3.2 Text
TEXT = "#e0e0e0"
TEXT_MUTED = "#9a9a9a"
TEXT_DISABLED = "#6a6a6a"
TEXT_ON_ACCENT = "#ffffff"

# 3.3 Accent / focus (neutral — no colored highlight)
ACCENT = "#6e6e6e"
SELECTION_BG = "#454545"
SELECTION_TEXT = "#ffffff"

# 3.4 Semantic colors (used sparingly; theme-independent)
CH_R = "#d06666"
CH_G = "#66aa66"
CH_B = "#6688d0"
# Slider track gradients (groove) — map the axis to its colour meaning.
TEMP_GRADIENT = ("#3d7fd1", "#e6c34d")   # Temperature: cool blue (low) -> warm amber (high)
TINT_GRADIENT = ("#5cb85c", "#c264c2")   # Tint: green (low) -> magenta (high)
SUCCESS = "#3a8f5a"
SUCCESS_HOVER = "#46a368"
SUCCESS_PRESSED = "#2f7449"
DANGER = "#d9534f"
DANGER_HOVER = "#c9302c"
DANGER_PRESSED = "#ac2925"
BAND_COLORS = {
    "red": "#c0392b", "skin": "#d8956b", "yellow": "#c8b900",
    "green": "#27ae60", "cyan": "#17a8b4", "blue": "#2f6fd0", "purple": "#8e44ad",
}

# 3.5 Tether banner (dark amber — was light-themed)
WARN_BG = "#3a2f12"
WARN_BORDER = "#5a4a1e"
WARN_TEXT = "#e0c060"
WARN_TEXT_MUTED = "#b89a48"

# 3.6 Icons
ICON = "#cfcfcf"
ICON_DISABLED = "#6a6a6a"

# 3.7 Spacing / radius / type
SPACE_XS, SPACE_SM, SPACE_MD, SPACE_LG, SPACE_XL = 2, 4, 6, 8, 12
RADIUS_SM, RADIUS_MD, RADIUS_LG = 3, 5, 8
FS_CAPTION, FS_BODY, FS_CONTROL, FS_HEADING = 11, 12, 13, 14

# 3.7b Layout rhythm — the spacing "design language". Use these intent-named gaps
# in Python layout code (setSpacing / setContentsMargins / addSpacing) so spacing
# is consistent, never an ad-hoc literal.
GAP_TIGHT = SPACE_XS      # 2  — within one control row (label | control | value)
GAP_ROW = SPACE_SM        # 4  — between rows inside a group (intra-group rhythm)
GAP_BTN = SPACE_LG        # 8  — between sibling buttons in a button row (canonical)
GAP_PANEL = SPACE_LG      # 8  — panel/dialog content margin; inter-column gutter
GAP_SECTION = SPACE_XL    # 12 — between functional groups (usually with a separator)

# Standard control geometry (one scale instead of a 22/24/28/30/42 zoo).
CONTROL_H = 28            # sliders, combos, labels, normal & secondary buttons, tabs
CONTROL_H_LG = 36         # weighty primary commits (Done, Export)
GLYPH_W = 28             # square glyph buttons (✕ remove / clear) — one width
LABEL_COL_W = 100         # right-aligned label gutter — the longest label
                          # ("Subtracted Sat") measures 89px in the macOS system
                          # font, so 90 left it flush against the panel border and
                          # one font step away from clipping. Keep slack here.
VALUE_COL_W = 40          # left-aligned value gutter in a 3-column row
PANEL_W = 340             # the right-hand side panels (sliders / dust / crop).
                          # Widest row is WB Picker|AWB|Crop|Slice at ~293px, so
                          # this must stay >= that + 2*GAP_PANEL, with slack for
                          # systems whose UI font is wider than the one it was
                          # measured on — a panel that is too narrow silently
                          # CLIPS its content (horizontal scrolling is off).
DIALOG_W_SM = 280         # progress / simple dialogs
DIALOG_W_MD = 440         # forms (export, update, sync)


class Paint:
    """3.8 UI-chrome paint constants (non-QSS; follow the dark theme)."""
    CURVE_BG = "#232323"
    CURVE_GRID = "#3d3d3d"
    CURVE_GRID_MINOR = "#333333"
    CURVE_IDENTITY = "#5a5a5a"
    CURVE_NODE = "#f0f0f0"
    CURVE_NODE_OUTLINE = "#222222"
    CURVE_NODE_DISABLED = "#777777"
    # Histogram (widgets/histogram_widget.py paints it with QPainter). RGB
    # tuples: the model stores raw counts, the widget builds QColors from these.
    HIST_BG = (42, 42, 42)            # == PANEL #2a2a2a (keep container in sync)
    HIST_GRID = (255, 255, 255, 16)   # quarter-tone gridlines — faint over HIST_BG
    HIST_R = (230, 80, 80)
    HIST_G = (90, 200, 90)
    HIST_B = (100, 140, 235)
    HIST_PEAK = (235, 235, 235)       # where all channels overlap / clip-all wedge
    # Scopes panel (widgets/scopes_panel.py): RGB parade + vectorscope under
    # the canvas. Darker than HIST_BG — the additive traces need contrast.
    SCOPE_BG = (26, 26, 26)
    SCOPE_GRID = (255, 255, 255, 30)
    SCOPE_LABEL = (255, 255, 255, 90)
    SCOPE_MARKER = (255, 255, 255, 235)
    SCOPE_MARKER_OUTLINE = (20, 20, 20, 220)
    SCOPE_REF = (225, 175, 80, 180)   # Cineon 95/685 reference lines (amber)
    SCOPE_SKIN = (230, 165, 145, 150)  # vectorscope skin-tone line


class Overlay:
    """3.8 On-image overlay constants (functional; values preserved verbatim).

    These sit over arbitrary photos and are tuned for contrast against image
    content, NOT the UI theme — do not re-tint to match the dark surfaces.
    """
    CURSOR_LIGHT = (255, 255, 255)
    CURSOR_DARK = (25, 25, 25)
    BWP_WHITE = (0, 100, 255, 140)
    BWP_DENSE = (255, 140, 0, 140)
    COMP_GRID = (0, 80, 0, 80)
    REF_FRAME = (255, 0, 0, 180)
    REF_FRAME_DRAG = (255, 0, 0, 120)
    SLICE = (0, 200, 255)            # alpha applied at call site (150 ghost / 235)
    DIM = (0, 0, 0, 80)
    DIM_CROP = (0, 0, 0, 110)
    BLACK_FILL = (0, 0, 0)
    DUST_STROKE = (255, 40, 40, 150)
    DUST_CURSOR = (255, 255, 255, 220)
    CROP_BORDER = (255, 255, 255, 220)
    HANDLE_FILL = (255, 255, 255, 235)
    HANDLE_OUTLINE = (30, 30, 30, 230)
    AREA_LINE = (255, 255, 255, 220)
    AREA_FEATHER = (255, 255, 255, 120)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def qcolor(v) -> QColor:
    """Coerce a hex string, (r,g,b[,a]) tuple, or QColor into a QColor."""
    if isinstance(v, QColor):
        return v
    if isinstance(v, (tuple, list)):
        return QColor(*v)
    return QColor(v)


def build_palette() -> QPalette:
    """A full dark Fusion palette derived from the tokens above."""
    pal = QPalette()
    pal.setColor(QPalette.Window, qcolor(WINDOW))
    pal.setColor(QPalette.WindowText, qcolor(TEXT))
    pal.setColor(QPalette.Base, qcolor(SURFACE))
    pal.setColor(QPalette.AlternateBase, qcolor(PANEL))
    pal.setColor(QPalette.ToolTipBase, qcolor(PANEL))
    pal.setColor(QPalette.ToolTipText, qcolor(TEXT))
    pal.setColor(QPalette.Text, qcolor(TEXT))
    pal.setColor(QPalette.Button, qcolor(SURFACE))
    pal.setColor(QPalette.ButtonText, qcolor(TEXT))
    pal.setColor(QPalette.BrightText, qcolor("#ffffff"))
    pal.setColor(QPalette.Highlight, qcolor(SELECTION_BG))
    pal.setColor(QPalette.HighlightedText, qcolor(SELECTION_TEXT))
    pal.setColor(QPalette.Link, qcolor("#9ab0c0"))
    try:
        pal.setColor(QPalette.PlaceholderText, qcolor(TEXT_MUTED))
    except AttributeError:
        pass  # older Qt without PlaceholderText role
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, qcolor(TEXT_DISABLED))
    pal.setColor(QPalette.Disabled, QPalette.Highlight, qcolor(SURFACE_ACTIVE))
    pal.setColor(QPalette.Disabled, QPalette.HighlightedText, qcolor(TEXT_DISABLED))
    return pal


# QSS uses { } literally, so format with %(name)s (QSS has no literal '%').
_QSS_TEMPLATE = """
QToolTip {
    color: %(text)s; background-color: %(panel)s;
    border: 1px solid %(border)s; padding: 3px 6px;
}

QPushButton {
    background-color: %(surface)s; color: %(text)s;
    border: 1px solid %(border)s; border-radius: %(r_sm)dpx;
    padding: 4px 10px;
}
QPushButton:hover { background-color: %(surface_hover)s; }
QPushButton:pressed, QPushButton:checked { background-color: %(surface_active)s; }
QPushButton:disabled { color: %(text_disabled)s; border-color: %(border)s; }

QToolBar { background-color: %(window)s; border: none; spacing: 2px; padding: 2px; }
QToolButton {
    background-color: transparent; color: %(text)s;
    border: none; border-radius: %(r_sm)dpx; padding: 3px;
}
QToolButton:hover { background-color: %(surface_hover)s; }
QToolButton:pressed, QToolButton:checked { background-color: %(surface_active)s; }
QToolButton:disabled { color: %(text_disabled)s; }

QComboBox {
    background-color: %(surface)s; color: %(text)s;
    border: 1px solid %(border)s; border-radius: %(r_sm)dpx; padding: 2px 6px;
}
QComboBox:hover { border-color: %(border_strong)s; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background-color: %(panel)s; color: %(text)s;
    border: 1px solid %(border)s;
    selection-background-color: %(selection_bg)s;
    selection-color: %(selection_text)s;
}

QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {
    background-color: %(surface)s; color: %(text)s;
    border: 1px solid %(border)s; border-radius: %(r_sm)dpx; padding: 2px 4px;
    selection-background-color: %(selection_bg)s;
    selection-color: %(selection_text)s;
}
QLineEdit:focus, QSpinBox:focus, QTextEdit:focus { border-color: %(accent)s; }

QGroupBox {
    border: 1px solid %(border)s; border-radius: %(r_md)dpx;
    margin-top: 8px; padding-top: 4px;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 8px; padding: 0 4px;
    color: %(text_muted)s;
}

QListWidget, QListView, QTreeView {
    background-color: %(panel)s; color: %(text)s;
    border: 1px solid %(border)s; border-radius: %(r_sm)dpx;
}
QListWidget::item:hover, QListView::item:hover { background-color: %(surface_hover)s; }
QListWidget::item:selected, QListView::item:selected {
    background-color: %(selection_bg)s; color: %(selection_text)s;
}

QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background-color: %(window)s; width: 12px; margin: 0; }
QScrollBar::handle:vertical {
    background-color: %(surface_active)s; min-height: 24px;
    border-radius: 6px; margin: 2px;
}
QScrollBar::handle:vertical:hover { background-color: %(border_strong)s; }
QScrollBar:horizontal { background-color: %(window)s; height: 12px; margin: 0; }
QScrollBar::handle:horizontal {
    background-color: %(surface_active)s; min-width: 24px;
    border-radius: 6px; margin: 2px;
}
QScrollBar::handle:horizontal:hover { background-color: %(border_strong)s; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QProgressBar {
    background-color: %(surface)s; color: %(text)s;
    border: 1px solid %(border)s; border-radius: %(r_sm)dpx;
    text-align: center;
}
QProgressBar::chunk { background-color: %(accent)s; border-radius: %(r_sm)dpx; }

QSlider::groove:horizontal {
    height: 4px; background-color: %(surface)s; border-radius: 2px;
}
QSlider::sub-page:horizontal { background-color: %(border_strong)s; border-radius: 2px; }
QSlider::handle:horizontal {
    background-color: %(text_muted)s; width: 12px; height: 12px;
    margin: -5px 0; border-radius: 6px;
}
QSlider::handle:horizontal:hover { background-color: %(text)s; }

QMenuBar { background-color: %(window)s; color: %(text)s; }
QMenuBar::item { background: transparent; padding: 4px 8px; }
QMenuBar::item:selected { background-color: %(surface_hover)s; }
QMenu { background-color: %(panel)s; color: %(text)s; border: 1px solid %(border)s; }
QMenu::item { padding: 4px 22px; }
QMenu::item:selected { background-color: %(selection_bg)s; color: %(selection_text)s; }
QMenu::item:disabled { color: %(text_disabled)s; }
QMenu::separator { height: 1px; background-color: %(border)s; margin: 4px 10px; }

QCheckBox, QRadioButton { color: %(text)s; spacing: 6px; }
QCheckBox:disabled, QRadioButton:disabled { color: %(text_disabled)s; }

QTabWidget::pane { border: 1px solid %(border)s; }
QTabBar::tab {
    background-color: %(surface)s; color: %(text_muted)s;
    padding: 4px 10px; border: 1px solid %(border)s; border-bottom: none;
}
QTabBar::tab:selected { background-color: %(panel)s; color: %(text)s; }

QGraphicsView { background-color: %(canvas)s; border: none; }

QLabel[caption="true"] { color: %(text_muted)s; font-size: %(fs_caption)dpx; }
"""


def global_qss(chevron_path=None) -> str:
    """The central stylesheet, built from the tokens above (one source of truth).

    ``chevron_path`` (from :func:`_ensure_chevron_icon`) adds the combo dropdown
    arrow; omitted (e.g. in tests) the base combo styling still applies.
    """
    qss = _QSS_TEMPLATE % {
        "window": WINDOW, "canvas": CANVAS, "panel": PANEL, "surface": SURFACE,
        "surface_hover": SURFACE_HOVER, "surface_active": SURFACE_ACTIVE,
        "border": BORDER, "border_strong": BORDER_STRONG,
        "text": TEXT, "text_muted": TEXT_MUTED, "text_disabled": TEXT_DISABLED,
        "accent": ACCENT, "selection_bg": SELECTION_BG, "selection_text": SELECTION_TEXT,
        "r_sm": RADIUS_SM, "r_md": RADIUS_MD, "fs_caption": FS_CAPTION,
    }
    if chevron_path:
        cp = str(chevron_path).replace("\\", "/")
        qss += (f"\nQComboBox::down-arrow {{ image: url({cp});"
                f" width: 12px; height: 12px; margin-right: 6px; }}\n")
    return qss


def load_tinted_icon(abs_path: str, color: str = ICON,
                     disabled_color: str = ICON_DISABLED) -> QIcon:
    """Load a monochrome PNG and recolor it using its alpha as a mask.

    The toolbar icons are pure-black line art on transparency (invisible on a
    dark theme); this paints them ``color`` (and a dimmer ``disabled_color`` for
    the disabled state) so they read on dark surfaces.
    """
    base = QPixmap(abs_path)
    if base.isNull():
        return QIcon(abs_path)

    def _tint(pm: QPixmap, col: str) -> QPixmap:
        out = QPixmap(pm.size())
        out.fill(Qt.transparent)
        p = QPainter(out)
        p.drawPixmap(0, 0, pm)
        p.setCompositionMode(QPainter.CompositionMode_SourceIn)
        p.fillRect(out.rect(), qcolor(col))
        p.end()
        return out

    icon = QIcon()
    icon.addPixmap(_tint(base, color), QIcon.Normal)
    icon.addPixmap(_tint(base, disabled_color), QIcon.Disabled)
    return icon


_BTN_ROLE_QSS = {
    # Primary = the weighty commit on a surface. NEUTRAL elevation (no hue) so it
    # leads without adding colour to a colour-critical UI.
    "primary": (
        "QPushButton { background-color: %(s_active)s; color: %(text)s;"
        " border: 1px solid %(border_strong)s; border-radius: %(r)dpx;"
        " padding: 5px 12px; font-weight: bold; }"
        " QPushButton:hover { background-color: %(p_hover)s; }"
        " QPushButton:pressed { background-color: %(border)s; }"
        " QPushButton:disabled { background-color: %(surface)s; color: %(disabled)s;"
        " border-color: %(border)s; }"
    ),
    # Danger = irreversible data loss. The only saturated colour in normal UI;
    # restrained at rest (red text + border), fills red on hover.
    "danger": (
        "QPushButton { background-color: %(surface)s; color: %(danger)s;"
        " border: 1px solid %(danger)s; border-radius: %(r)dpx; padding: 4px 12px; }"
        " QPushButton:hover { background-color: %(danger)s; color: #ffffff; }"
        " QPushButton:pressed { background-color: %(danger_pressed)s; color: #ffffff; }"
        " QPushButton:disabled { background-color: %(surface)s; color: %(disabled)s;"
        " border-color: %(border)s; }"
    ),
    # Glyph-only danger (e.g. a ✕ remove): no rest emphasis, red glyph, fills on hover.
    "danger_glyph": (
        "QPushButton { background-color: transparent; color: %(danger)s;"
        " border: 1px solid %(border)s; border-radius: %(r)dpx; padding: 4px 8px; }"
        " QPushButton:hover { background-color: %(danger)s; color: #ffffff;"
        " border-color: %(danger)s; }"
        " QPushButton:pressed { background-color: %(danger_pressed)s; color: #ffffff; }"
        " QPushButton:disabled { color: %(disabled)s; border-color: %(border)s; }"
    ),
}


def style_button(btn, role, *, default=False, glyph_only=False) -> None:
    """Give a QPushButton a semantic role so colour carries meaning, not decoration:

    - ``"primary"`` — the surface's weighty commit (neutral elevation, bold):
      Convert Current, Sync, Export, Done.
    - ``"danger"``  — irreversible data loss (red): Reset, Reset Curve, Clear all.
      Pass ``glyph_only=True`` for a glyph button like a ✕ remove.
    - ``"secondary"`` / ``None`` — the default neutral button (clears any role).

    ``default=True`` also makes it the dialog's default button (focus ring). Plain
    QPushButtons honour their own stylesheet over the global rule, so the role
    reliably takes effect.
    """
    role = (role or "").lower()
    vals = {
        "r": RADIUS_SM, "surface": SURFACE, "s_active": SURFACE_ACTIVE,
        "p_hover": "#505050", "border": BORDER, "border_strong": BORDER_STRONG,
        "text": TEXT, "disabled": TEXT_DISABLED,
        "danger": DANGER, "danger_pressed": DANGER_PRESSED,
    }
    if role == "primary":
        btn.setStyleSheet(_BTN_ROLE_QSS["primary"] % vals)
    elif role == "danger":
        btn.setStyleSheet(_BTN_ROLE_QSS["danger_glyph" if glyph_only else "danger"] % vals)
    else:
        btn.setStyleSheet("")
    if default:
        btn.setDefault(True)


def apply_panel_spacing(layout, *, margin=GAP_PANEL, spacing=GAP_ROW):
    """Conform a panel/dialog layout to the standard rhythm in one call."""
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(spacing)
    return layout


def apply_button_row(layout, *, spacing=GAP_BTN):
    """Conform a horizontal button row: canonical button gap, no extra margins."""
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    return layout


def shrinkable_combo(combo, *, min_chars=6):
    """Stop a QComboBox from dictating its panel's width.

    A QComboBox's default minimumSizeHint is wide enough for its LONGEST ITEM,
    which propagates all the way out as a hard minimum on the containing panel —
    adding one long preset name (a film stock, a colour profile) is then enough
    to push a fixed-width panel's content past its edge and clip it. Sizing to a
    short contents length instead lets the combo shrink to the space the row can
    spare, which is what a narrow side panel needs.

    This governs the combo's WIDTH only. Text too long for the resulting box is
    hard-clipped by Qt unless the combo also elides — see ElidingComboBox.
    """
    from PySide6.QtWidgets import QComboBox
    combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
    combo.setMinimumContentsLength(min_chars)
    return combo


class ElidingComboBox(QComboBox):
    """A combo whose closed-state text ends in "…" when it doesn't fit.

    Qt does not elide a non-editable QComboBox: it draws the label into the
    edit-field rect and lets the glyphs run under the arrow, so a long
    user-supplied name (a film stock, a camera profile) reads as
    ``Fuji Superia X-TRA 40`` with no sign that anything was cut. Nothing in the
    UI then says the selection is longer than what's shown.

    The frame and arrow are still painted by the active style, so the app's
    global QSS applies unchanged; only the label is drawn here, elided to the
    edit-field rect. The popup list is untouched, so the full names are always
    one click away. Tooltips are left alone — callers set their own.
    """

    def paintEvent(self, _event):
        from PySide6.QtWidgets import (QStyle, QStyleOptionComboBox,
                                       QStylePainter)
        painter = QStylePainter(self)
        # Pick the colour group explicitly: the global QSS has no
        # QComboBox:disabled rule, so a disabled combo's dimmed text comes from
        # the palette's Disabled group. Drawing the label ourselves would
        # otherwise render it at full strength and lose the disabled look.
        group = QPalette.Active if self.isEnabled() else QPalette.Disabled
        painter.setPen(self.palette().color(group, QPalette.Text))
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)
        full = opt.currentText
        # Blank the text so the style paints frame + arrow only; drawing the
        # label twice would leave the un-elided version showing underneath.
        opt.currentText = ""
        painter.drawComplexControl(QStyle.CC_ComboBox, opt)
        # The edit-field sub-control rect already carries the style's padding —
        # do NOT inset it further, or short labels shift a few px right of where
        # Qt would have drawn them and the combo stops matching its neighbours.
        rect = self.style().subControlRect(
            QStyle.CC_ComboBox, opt, QStyle.SC_ComboBoxEditField, self)
        painter.drawText(
            rect, Qt.AlignLeft | Qt.AlignVCenter,
            self.fontMetrics().elidedText(full, Qt.ElideRight, rect.width()))


def keep_out_of_minimum_width(widget):
    """Stop a widget's own text from setting its panel's minimum width.

    ``qSmartMinSize`` — what a layout asks each item for — skips the width
    entirely when the horizontal policy is Ignored, and only then. Mutate the
    policy already on the widget rather than building a fresh one: QLabel keeps
    its height-for-width bit in there, and dropping that stops a word-wrapped
    label from reporting the taller height its wrapped text needs.
    """
    from PySide6.QtWidgets import QSizePolicy
    policy = widget.sizePolicy()
    policy.setHorizontalPolicy(QSizePolicy.Ignored)
    widget.setSizePolicy(policy)
    return widget


class ElidingLabel(QLabel):
    """A single-line label that shortens its text rather than widening its panel.

    A QLabel's minimum width is the full width of its text, so a label carrying
    a user-supplied name inside a fixed-width panel does not merely overflow its
    own line — it raises the minimum width of everything laid out beside it and
    pushes the panel's content past the edge, where horizontal scrolling is off
    and the whole panel clips. Word wrap is not a fix on its own: it lowers the
    minimum only to the longest WORD, and "Kodak_Portra_400_pushed" is one word.

    Eliding happens at PAINT time, against the width the layout actually handed
    out — a label that is currently hidden has never been laid out, so its own
    width() at setText time is stale or still the default. The full text stays
    in the tooltip.
    """

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        keep_out_of_minimum_width(self)

    def setText(self, text):
        # Not a virtual in Qt, so this only covers Python callers — which is all
        # of them here. The elided form hides part of the name; the tooltip is
        # where it stays readable.
        super().setText(text)
        self.setToolTip(text)

    def paintEvent(self, _event):
        painter = QPainter(self)
        # QPainter starts on a black pen. The QSS `color:` reaches a QLabel as
        # its palette foreground role, so read it from there — otherwise muted
        # text paints black on the dark theme.
        painter.setPen(self.palette().color(self.foregroundRole()))
        rect = self.contentsRect()
        painter.drawText(
            rect, int(self.alignment()),
            self.fontMetrics().elidedText(
                self.text(), Qt.ElideRight, rect.width()))


def section_separator():
    """The one canonical horizontal rule between functional regions — a 1px line
    with section spacing above/below. A plain HLine coloured via ``color:`` is
    invisible on the dark theme, so the line is painted via ``background-color``.
    """
    from PySide6.QtWidgets import QFrame
    sep = QFrame()
    sep.setStyleSheet(
        f"QFrame {{ background-color: {BORDER_STRONG}; min-height: 1px; max-height: 1px;"
        f" margin-top: {SPACE_LG}px; margin-bottom: {SPACE_SM}px; }}")
    return sep


_CHEVRON_PATH = None


def _ensure_chevron_icon() -> str:
    """Generate the combo dropdown chevron PNG once; return its path for QSS
    ``image: url(...)``. Requires a running QApplication (uses QPixmap/QPainter).

    Qt suppresses the native combo arrow once the combo is stylesheet-styled and
    the QSS border-triangle trick renders as a blocky shape, so we paint a clean
    chevron and reference it as the down-arrow image.
    """
    global _CHEVRON_PATH
    if _CHEVRON_PATH and os.path.exists(_CHEVRON_PATH):
        return _CHEVRON_PATH
    path = os.path.join(tempfile.gettempdir(), "freeccr_chevron_down.png")
    pm = QPixmap(12, 12)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(qcolor(TEXT_MUTED), 1.6))
    p.drawLine(3, 4, 6, 8)   # ╲
    p.drawLine(6, 8, 9, 4)   # ╱  → a downward chevron ˅
    p.end()
    pm.save(path)
    _CHEVRON_PATH = path
    return path


def section_header_qss():
    """QSS for a group-anchor section header (bold caption, full-strength text)."""
    return (f"color: {TEXT}; font-size: {FS_CAPTION}px; font-weight: bold; "
            f"letter-spacing: 0.5px; margin-bottom: {SPACE_XS}px;")


def apply_theme(app, settings=None) -> None:
    """Install the global dark theme on the QApplication.

    Fusion is required: the native Windows/macOS style ignores QPalette for many
    controls; Fusion honors it fully. Call once, before building the main window.
    ``settings`` is reserved for a future theme preference; v1 always applies dark.
    """
    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setStyleSheet(global_qss(_ensure_chevron_icon()))


def apply_windows_dark_titlebar(widget) -> bool:
    """Make a top-level window's native title bar dark on Windows 10/11.

    Qt's palette/QSS don't reach the OS-drawn title bar (the non-client area), so
    it stays light against the dark theme. This sets the DWM immersive-dark-mode
    attribute on the window's HWND. Apply it to each top-level window (main window,
    custom dialogs) before showing it. No-op off Windows or if unsupported.
    Returns True if the attribute call was made.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        # winId() forces creation of the native window handle.
        hwnd = int(widget.winId())
        if not hwnd:
            return False
        value = ctypes.c_int(1)
        size = ctypes.sizeof(value)
        # DWMWA_USE_IMMERSIVE_DARK_MODE = 20 (Win10 20H1+/Win11); 19 on older builds.
        dwm = ctypes.windll.dwmapi
        if dwm.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), size) != 0:
            dwm.DwmSetWindowAttribute(hwnd, 19, ctypes.byref(value), size)
        return True
    except Exception:
        return False


class _DarkTitleBarFilter(QObject):
    """App-level filter that darkens each top-level window's title bar on show.

    Covers dialogs created on demand (message boxes, the export / loading / about
    dialogs) without touching every call site. Windows-only; inert elsewhere.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._seen = set()

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.Show
                and isinstance(obj, QWidget) and obj.isWindow()):
            key = id(obj)
            if key not in self._seen:
                self._seen.add(key)
                apply_windows_dark_titlebar(obj)
        return False


_dark_titlebar_filter = None


def install_dark_titlebar_filter(app) -> None:
    """Install an app-wide filter so every top-level window (dialogs, message
    boxes) gets a dark native title bar when shown.

    No-op off Windows or if already installed. Call once after ``apply_theme()``
    from the app entry point (kept out of ``apply_theme`` so the test suite,
    which themes a shared QApplication, isn't given a persistent filter).
    """
    global _dark_titlebar_filter
    if sys.platform != "win32" or _dark_titlebar_filter is not None:
        return
    _dark_titlebar_filter = _DarkTitleBarFilter(app)
    app.installEventFilter(_dark_titlebar_filter)


def apply_macos_srgb_colorspace(widget) -> bool:
    """Tag a top-level window's NSWindow as sRGB on macOS.

    Qt raster content is not color-managed (QTBUG-47660): the Cocoa backing
    store inherits the NSWindow's colorspace, which defaults to the display's
    own profile, so our sRGB-encoded preview pixels reach the screen
    un-matched — visibly oversaturated on the wide-gamut (Display P3) panels
    of every modern Mac, while exports (ICC-tagged sRGB/ProPhoto) display
    correctly in ColorSync-managed viewers. Tagging the NSWindow sRGB makes
    the WindowServer color-match the whole window (previews and UI alike) to
    whatever profile the display uses.

    Safe to call after show(): Qt observes
    NSWindowDidChangeBackingPropertiesNotification and re-tags its backing
    buffers on the change. No-op off macOS. Returns True if the colorspace
    call was made.
    """
    if sys.platform != "darwin":
        return False
    try:
        import ctypes
        import ctypes.util

        objc = ctypes.CDLL(ctypes.util.find_library("objc"))
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        # objc_msgSend is variadic — it must be cast to an exact prototype per
        # call signature (mandatory on arm64). Everything here is id-sized.
        send = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p))
        send1 = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p))

        def sel(name: bytes) -> ctypes.c_void_p:
            return ctypes.c_void_p(objc.sel_registerName(name))

        # On macOS winId() is the NSView* (and forces native window creation).
        nsview = ctypes.c_void_p(int(widget.winId()))
        nswindow = ctypes.c_void_p(send(nsview, sel(b"window")))
        if not nswindow:
            return False
        srgb = ctypes.c_void_p(send(
            ctypes.c_void_p(objc.objc_getClass(b"NSColorSpace")),
            sel(b"sRGBColorSpace")))
        if not srgb:
            return False
        send1(nswindow, sel(b"setColorSpace:"), srgb)
        return True
    except Exception:
        return False


class _MacSRGBColorSpaceFilter(QObject):
    """App-level filter that sRGB-tags each top-level window on show.

    Covers dialogs created on demand, like ``_DarkTitleBarFilter``. Re-applies
    on every Show (no seen-set): setColorSpace is cheap and idempotent, and a
    platform window Qt re-created in between (new winId) needs re-tagging.
    macOS-only; inert elsewhere.
    """

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.Type.Show
                and isinstance(obj, QWidget) and obj.isWindow()):
            apply_macos_srgb_colorspace(obj)
        return False


_macos_srgb_filter = None


def install_macos_srgb_filter(app) -> None:
    """Install an app-wide filter so every top-level window is tagged with the
    sRGB colorspace when shown (see ``apply_macos_srgb_colorspace``).

    No-op off macOS or if already installed. Call once from the app entry
    point, alongside ``install_dark_titlebar_filter``.
    """
    global _macos_srgb_filter
    if sys.platform != "darwin" or _macos_srgb_filter is not None:
        return
    _macos_srgb_filter = _MacSRGBColorSpaceFilter(app)
    app.installEventFilter(_macos_srgb_filter)

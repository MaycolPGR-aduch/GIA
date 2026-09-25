"""Sistema de diseño de GIA: paletas, tipografía y hoja de estilos Qt (QSS).

Todas las ventanas toman su estilo de aquí. El QSS se aplica una sola vez a
nivel de ``QApplication``; los widgets cambian de aspecto mediante propiedades
dinámicas (``variant``, ``role``, ``tone``) en lugar de llamar a
``setStyleSheet`` en caliente, que es costoso.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QApplication, QWidget


PALETTES = {
    "dark": {
        "bg": "#0b1120",
        "surface": "#111827",
        "surface_alt": "#1a2332",
        "surface_hover": "#243044",
        "border": "#253247",
        "border_strong": "#3a4a63",
        "text": "#f1f5f9",
        "text_muted": "#94a3b8",
        "text_subtle": "#64748b",
        "accent": "#3b82f6",
        "accent_hover": "#2563eb",
        "on_accent": "#ffffff",
        "success": "#22c55e",
        "warning": "#f59e0b",
        "danger": "#ef4444",
        "info": "#06b6d4",
        "neutral": "#64748b",
        "video_bg": "#020617",
        "focus": "#60a5fa",
        "border_width": 1,
    },
    "high_contrast": {
        "bg": "#000000",
        "surface": "#000000",
        "surface_alt": "#0d0d0d",
        "surface_hover": "#1f1f1f",
        "border": "#ffffff",
        "border_strong": "#ffffff",
        "text": "#ffffff",
        "text_muted": "#f0f0f0",
        "text_subtle": "#d4d4d4",
        "accent": "#ffd400",
        "accent_hover": "#ffe34d",
        "on_accent": "#000000",
        "success": "#00ff6a",
        "warning": "#ffb000",
        "danger": "#ff4d4d",
        "info": "#00e5ff",
        "neutral": "#bdbdbd",
        "video_bg": "#000000",
        "focus": "#00e5ff",
        "border_width": 2,
    },
}

# Tamaños tipográficos base (px) antes de aplicar la escala.
FONT_SIZES = {
    "small": 11,
    "body": 13,
    "h2": 15,
    "title": 18,
    "display": 26,
    "hero": 34,
}

FONT_FAMILY = "Segoe UI"
MIN_FONT_SCALE = 0.85
MAX_FONT_SCALE = 1.8
PRESENTATION_FONT_SCALE = 1.35


class _ThemeState(QObject):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.theme = "dark"
        self.font_scale = 1.0


_state: _ThemeState | None = None


def theme_state() -> _ThemeState:
    global _state
    if _state is None:
        _state = _ThemeState()
    return _state


def current_theme() -> str:
    return theme_state().theme


def current_font_scale() -> float:
    return theme_state().font_scale


def palette(theme: str | None = None) -> dict:
    return PALETTES.get(theme or current_theme(), PALETTES["dark"])


def color(token: str, theme: str | None = None) -> QColor:
    return QColor(palette(theme)[token])


def font_px(size_key: str, font_scale: float | None = None) -> int:
    scale = current_font_scale() if font_scale is None else font_scale
    return max(8, round(FONT_SIZES[size_key] * scale))


def font(size_key: str = "body", bold: bool = False) -> QFont:
    qfont = QFont(FONT_FAMILY)
    qfont.setPixelSize(font_px(size_key))
    qfont.setBold(bold)
    return qfont


# Tonos semánticos para estados del sistema.
TONES = ("accent", "success", "warning", "danger", "info", "neutral")


def tone_color(tone: str) -> QColor:
    return color(tone if tone in TONES else "neutral")


def refresh_style(widget: QWidget) -> None:
    """Re-evalúa el QSS tras cambiar una propiedad dinámica."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def set_widget_property(widget: QWidget, name: str, value) -> None:
    """Cambia una propiedad solo si varía, y repolariza el widget."""
    if widget.property(name) == value:
        return
    widget.setProperty(name, value)
    refresh_style(widget)


def build_stylesheet(theme: str = "dark", font_scale: float = 1.0) -> str:
    p = palette(theme)
    bw = p["border_width"]
    fs = {key: font_px(key, font_scale) for key in FONT_SIZES}
    return f"""
* {{
    font-family: '{FONT_FAMILY}', 'Inter', sans-serif;
}}
QWidget {{
    color: {p['text']};
    font-size: {fs['body']}px;
}}
QMainWindow, QDialog, QWidget#appRoot {{
    background-color: {p['bg']};
}}
QToolTip {{
    background-color: {p['surface_alt']};
    color: {p['text']};
    border: {bw}px solid {p['border_strong']};
    padding: 6px 8px;
    border-radius: 6px;
}}

/* ---------- Tipografía por rol ---------- */
QLabel[role="hero"] {{ font-size: {fs['hero']}px; font-weight: 700; }}
QLabel[role="display"] {{ font-size: {fs['display']}px; font-weight: 700; }}
QLabel[role="title"] {{ font-size: {fs['title']}px; font-weight: 700; }}
QLabel[role="h2"] {{ font-size: {fs['h2']}px; font-weight: 600; }}
QLabel[role="cardTitle"] {{
    font-size: {fs['small']}px; font-weight: 700; color: {p['text_muted']};
}}
QLabel[role="muted"] {{ color: {p['text_muted']}; }}
QLabel[role="subtle"] {{ color: {p['text_subtle']}; font-size: {fs['small']}px; }}
QLabel[role="small"] {{ font-size: {fs['small']}px; }}
QLabel[role="quote"] {{ font-size: {fs['title']}px; font-weight: 600; font-style: italic; }}
QLabel[role="logo"] {{
    background-color: {p['accent']}; color: {p['on_accent']};
    font-size: {fs['h2']}px; font-weight: 800; border-radius: 10px;
    padding: 6px 9px;
}}
QLabel[tone="success"] {{ color: {p['success']}; }}
QLabel[tone="warning"] {{ color: {p['warning']}; }}
QLabel[tone="danger"] {{ color: {p['danger']}; }}
QLabel[tone="info"] {{ color: {p['info']}; }}
QLabel[tone="accent"] {{ color: {p['accent']}; }}
QLabel[tone="muted"] {{ color: {p['text_muted']}; }}

/* ---------- Tarjetas ---------- */
QFrame[role="card"] {{
    background-color: {p['surface']};
    border: {bw}px solid {p['border']};
    border-radius: 14px;
}}
QFrame[role="header"] {{
    background-color: {p['surface']};
    border: {bw}px solid {p['border']};
    border-radius: 14px;
}}
QFrame[role="compact"] {{
    background-color: {p['surface']};
    border: {max(bw, 2)}px solid {p['border_strong']};
    border-radius: 18px;
}}
QFrame[role="divider"] {{
    background-color: {p['border']};
    max-height: {bw}px; min-height: {bw}px; border: none;
}}

/* ---------- Botones ---------- */
QPushButton, QToolButton {{
    background-color: {p['surface_alt']};
    border: {bw}px solid {p['border']};
    border-radius: 10px;
    padding: 9px 14px;
    font-weight: 600;
    color: {p['text']};
}}
QPushButton:hover, QToolButton:hover {{
    background-color: {p['surface_hover']};
    border-color: {p['border_strong']};
}}
QPushButton:pressed, QToolButton:pressed {{ background-color: {p['surface']}; }}
QPushButton:focus, QToolButton:focus {{ border: {max(bw, 2)}px solid {p['focus']}; }}
QPushButton:disabled, QToolButton:disabled {{
    color: {p['text_subtle']}; background-color: {p['surface']}; border-color: {p['border']};
}}
QPushButton[variant="primary"] {{
    background-color: {p['accent']}; border-color: {p['accent']}; color: {p['on_accent']};
}}
QPushButton[variant="primary"]:hover {{ background-color: {p['accent_hover']}; }}
QPushButton[variant="success"] {{
    background-color: {p['success']}; border-color: {p['success']}; color: #04210f;
}}
QPushButton[variant="warning"] {{
    background-color: {p['warning']}; border-color: {p['warning']}; color: #2a1800;
}}
QPushButton[variant="danger"] {{
    background-color: transparent; border: {max(bw, 1)}px solid {p['danger']}; color: {p['danger']};
}}
QPushButton[variant="danger"]:hover {{ background-color: {p['danger']}; color: #ffffff; }}
QPushButton[variant="ghost"], QToolButton[variant="ghost"] {{
    background-color: transparent; border-color: transparent;
}}
QPushButton[variant="ghost"]:hover, QToolButton[variant="ghost"]:hover {{
    background-color: {p['surface_hover']}; border-color: {p['border']};
}}
QPushButton[variant="ghost"]:checked, QToolButton[variant="ghost"]:checked {{
    background-color: {p['surface_hover']}; border-color: {p['accent']};
}}
QPushButton[size="large"] {{ padding: 12px 18px; font-size: {fs['h2']}px; }}
QToolButton[role="section"] {{
    background: transparent; border: none; padding: 6px 2px;
    font-size: {fs['small']}px; font-weight: 700; color: {p['text_muted']};
    text-align: left;
}}
QToolButton[role="section"]:hover {{ color: {p['text']}; }}

/* ---------- Entradas ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {p['surface_alt']};
    border: {bw}px solid {p['border']};
    border-radius: 8px;
    padding: 7px 10px;
    selection-background-color: {p['accent']};
    selection-color: {p['on_accent']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: {max(bw, 2)}px solid {p['focus']};
}}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox QAbstractItemView {{
    background-color: {p['surface_alt']};
    border: {bw}px solid {p['border_strong']};
    selection-background-color: {p['accent']};
    selection-color: {p['on_accent']};
    outline: none;
}}
QCheckBox {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator {{
    width: 18px; height: 18px; border-radius: 5px;
    border: {max(bw, 2)}px solid {p['border_strong']}; background: {p['surface_alt']};
}}
QCheckBox::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; }}
QCheckBox::indicator:focus {{ border-color: {p['focus']}; }}

QTextEdit, QPlainTextEdit {{
    background-color: {p['surface_alt']};
    border: {bw}px solid {p['border']};
    border-radius: 10px;
    padding: 6px 8px;
    selection-background-color: {p['accent']};
}}
QPlainTextEdit[role="log"] {{
    background-color: {p['video_bg']};
    font-family: 'Cascadia Mono', 'Consolas', monospace;
    font-size: {fs['small']}px;
}}

/* ---------- Progreso ---------- */
QProgressBar {{
    border: {bw}px solid {p['border']};
    border-radius: 7px;
    text-align: center;
    background-color: {p['surface_alt']};
    min-height: 14px;
    font-size: {fs['small']}px;
    font-weight: 600;
}}
QProgressBar::chunk {{ background-color: {p['accent']}; border-radius: 6px; }}
QProgressBar[tone="success"]::chunk {{ background-color: {p['success']}; }}
QProgressBar[tone="warning"]::chunk {{ background-color: {p['warning']}; }}

/* ---------- Listas, tablas y pestañas ---------- */
QListWidget {{
    background-color: transparent;
    border: none;
    outline: none;
}}
QListWidget::item {{
    background-color: {p['surface_alt']};
    border: {bw}px solid {p['border']};
    border-radius: 10px;
    margin: 3px 0px;
    padding: 4px;
}}
QListWidget::item:hover {{ border-color: {p['border_strong']}; }}
QListWidget::item:selected {{
    background-color: {p['surface_hover']};
    border: {max(bw, 2)}px solid {p['accent']};
    color: {p['text']};
}}
QListWidget#stepNav::item {{
    background-color: transparent; border: {bw}px solid transparent;
    padding: 10px 8px; margin: 2px 0px;
}}
QListWidget#stepNav::item:selected {{
    background-color: {p['surface_hover']}; border-color: {p['accent']};
}}
QListWidget#stepNav::item:disabled {{ color: {p['text_subtle']}; }}

QTableWidget {{
    background-color: {p['surface']};
    border: {bw}px solid {p['border']};
    border-radius: 10px;
    gridline-color: {p['border']};
    selection-background-color: {p['surface_hover']};
}}
QHeaderView::section {{
    background-color: {p['surface_alt']};
    color: {p['text_muted']};
    border: none;
    border-bottom: {bw}px solid {p['border']};
    padding: 6px 8px;
    font-weight: 700;
}}
QTableCornerButton::section {{ background-color: {p['surface_alt']}; border: none; }}

QTabWidget::pane {{
    border: {bw}px solid {p['border']};
    background-color: {p['surface']};
    border-radius: 10px;
    top: -1px;
}}
QTabBar::tab {{
    background-color: transparent;
    color: {p['text_muted']};
    border: none;
    border-bottom: 3px solid transparent;
    padding: 8px 16px;
    font-weight: 600;
}}
QTabBar::tab:selected {{ color: {p['text']}; border-bottom-color: {p['accent']}; }}
QTabBar::tab:hover {{ color: {p['text']}; }}

QGroupBox {{
    background-color: {p['surface']};
    border: {bw}px solid {p['border']};
    border-radius: 12px;
    margin-top: 14px;
    padding: 16px 10px 10px 10px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin; subcontrol-position: top left;
    left: 12px; padding: 0 6px; color: {p['text_muted']};
}}

/* ---------- Scroll ---------- */
QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p['border_strong']}; min-height: 24px; border-radius: 4px; }}
QScrollBar::handle:vertical:hover {{ background: {p['text_subtle']}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {p['border_strong']}; min-width: 24px; border-radius: 4px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ border: none; background: none; width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

QStatusBar {{ background-color: {p['surface']}; color: {p['text_muted']}; border-top: {bw}px solid {p['border']}; }}
QMessageBox QLabel {{ font-size: {fs['body']}px; }}
"""


def _apply(app: QApplication | None = None) -> None:
    app = app or QApplication.instance()
    if app is None:
        return
    state = theme_state()
    app.setStyleSheet(build_stylesheet(state.theme, state.font_scale))
    base_font = QFont(FONT_FAMILY)
    base_font.setPixelSize(font_px("body"))
    app.setFont(base_font)
    state.changed.emit()


def apply_theme(app: QApplication, settings: dict | None = None) -> None:
    """Aplica el tema guardado en ``settings`` (high_contrast, font_scale)."""
    settings = settings or {}
    state = theme_state()
    state.theme = "high_contrast" if settings.get("high_contrast") else "dark"
    state.font_scale = _clamp_scale(settings.get("font_scale", 1.0))
    _apply(app)


def _clamp_scale(value) -> float:
    try:
        scale = float(value)
    except (TypeError, ValueError):
        scale = 1.0
    return round(min(MAX_FONT_SCALE, max(MIN_FONT_SCALE, scale)), 2)


def _persist(settings: dict | None) -> None:
    if settings is None:
        return
    from .config_store import save_settings

    settings["high_contrast"] = current_theme() == "high_contrast"
    settings["font_scale"] = current_font_scale()
    try:
        save_settings(settings)
    except Exception:
        pass


def set_high_contrast(enabled: bool, settings: dict | None = None) -> None:
    theme_state().theme = "high_contrast" if enabled else "dark"
    _apply()
    _persist(settings)


def toggle_high_contrast(settings: dict | None = None) -> bool:
    enabled = current_theme() != "high_contrast"
    set_high_contrast(enabled, settings)
    return enabled


def set_font_scale(scale: float, settings: dict | None = None) -> float:
    theme_state().font_scale = _clamp_scale(scale)
    _apply()
    _persist(settings)
    return current_font_scale()


def step_font_scale(delta: float, settings: dict | None = None) -> float:
    return set_font_scale(current_font_scale() + delta, settings)

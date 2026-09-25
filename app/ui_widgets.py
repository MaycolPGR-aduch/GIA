"""Componentes visuales reutilizables de GIA (PySide6).

Los widgets pintados a mano (badges, video, medidores, fichas de gesto) leen la
paleta activa en cada ``paintEvent`` y se repintan cuando cambia el tema, así
que funcionan igual en modo oscuro y en alto contraste.
"""

from __future__ import annotations

import math
from datetime import datetime
from html import escape

import numpy as np
from PySide6.QtCore import QByteArray, QEasingCurve, QRectF, QSize, Qt, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QFontMetrics, QIcon, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import ui_theme as theme


# --------------------------------------------------------------------------
# Iconos SVG (24x24, trazo). Se colorean al renderizar.
# --------------------------------------------------------------------------
_ICON_PATHS = {
    "pause": '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>',
    "play": '<path d="M7 5l12 7-12 7z"/>',
    "cursor": '<path d="M5 3l14 7-6 2-2 6z"/>',
    "target": '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>',
    "compact": '<rect x="3" y="5" width="18" height="14" rx="2"/><rect x="12" y="12" width="7" height="5" rx="1"/>',
    "expand": '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    "help": '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6V14"/><path d="M12 17.5v.01"/>',
    "mic": '<rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5 11a7 7 0 0 0 14 0M12 18v3"/>',
    "power": '<path d="M12 3v8"/><path d="M6.3 7a8 8 0 1 0 11.4 0"/>',
    "camera": '<path d="M3 8h3l2-3h8l2 3h3v11H3z"/><circle cx="12" cy="13" r="3.5"/>',
    "check": '<path d="M5 12l5 5 9-10"/>',
    "alert": '<path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.01"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "monitor": '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/>',
    "contrast": '<circle cx="12" cy="12" r="9"/><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/>',
    "list": '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
    "sliders": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "chip": '<rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "face": '<circle cx="12" cy="12" r="9"/><path d="M9 10h.01M15 10h.01M8.5 14.5a4.5 4.5 0 0 0 7 0"/>',
    "book": '<path d="M4 4h6a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H4z"/><path d="M20 4h-6a2 2 0 0 0-2 2v14a2 2 0 0 1 2-2h6z"/>',
    "lock": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "plus": '<path d="M12 5v14M5 12h14"/>',
    "trash": '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>',
    "save": '<path d="M5 3h11l3 3v15H5z"/><path d="M8 3v5h7V3M8 21v-7h8v7"/>',
    "refresh": '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
    "record": '<circle cx="12" cy="12" r="6" fill="currentColor"/>',
    "stop": '<rect x="6" y="6" width="12" height="12" rx="2" fill="currentColor"/>',
    "x": '<path d="M6 6l12 12M18 6L6 18"/>',
    "chevron_right": '<path d="M9 6l6 6-6 6"/>',
    "chevron_down": '<path d="M6 9l6 6 6-6"/>',
    "globe": '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
    "text_size": '<path d="M4 20l5-14 5 14M6 15h6M15 20l3-8 3 8M16 17h4"/>',
}

_icon_cache: dict[tuple, QIcon] = {}


def _render_svg(name: str, hex_color: str, size: int) -> QPixmap:
    body = _ICON_PATHS[name].replace("currentColor", hex_color)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
        f'stroke="{hex_color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    ratio = 2
    pixmap = QPixmap(size * ratio, size * ratio)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    QSvgRenderer(QByteArray(svg.encode("utf-8"))).render(painter)
    painter.end()
    pixmap.setDevicePixelRatio(ratio)
    return pixmap


def icon(name: str, color: str | QColor | None = None, size: int = 20) -> QIcon:
    if name not in _ICON_PATHS:
        return QIcon()
    hex_color = QColor(color).name() if color is not None else theme.palette()["text"]
    key = (name, hex_color, size, theme.current_theme())
    cached = _icon_cache.get(key)
    if cached is not None:
        return cached
    qicon = QIcon()
    qicon.addPixmap(_render_svg(name, hex_color, size), QIcon.Normal)
    qicon.addPixmap(_render_svg(name, theme.palette()["text_subtle"], size), QIcon.Disabled)
    _icon_cache[key] = qicon
    return qicon


# --------------------------------------------------------------------------
# Helpers simples
# --------------------------------------------------------------------------
def make_label(text: str = "", role: str | None = None, tone: str | None = None, wrap: bool = False) -> QLabel:
    label = QLabel(text)
    if role:
        label.setProperty("role", role)
    if tone:
        label.setProperty("tone", tone)
    label.setWordWrap(wrap)
    return label


def set_label_text(label: QLabel, text: str) -> None:
    """Evita repintados cuando el texto no cambia."""
    if label.text() != text:
        label.setText(text)


class Button(QPushButton):
    """QPushButton con icono SVG que se recolorea según variante y tema."""

    _ICON_COLORS = {
        "primary": "on_accent",
        "danger": "danger",
    }
    _FIXED_ICON_COLORS = {"success": "#04210f", "warning": "#2a1800"}

    def __init__(self, text: str = "", icon_name: str | None = None, variant: str | None = None,
                 tooltip: str | None = None, size: str | None = None, checkable: bool = False, parent=None):
        super().__init__(text, parent)
        self._icon_name = icon_name
        self.setCursor(Qt.PointingHandCursor)
        self.setCheckable(checkable)
        if variant:
            self.setProperty("variant", variant)
        if size:
            self.setProperty("size", size)
        if tooltip:
            self.setToolTip(tooltip)
        self.setIconSize(QSize(18, 18))
        self._refresh_icon()
        theme.theme_state().changed.connect(self._refresh_icon)

    def set_variant(self, variant: str | None) -> None:
        theme.set_widget_property(self, "variant", variant or "")
        self._refresh_icon()

    def set_icon_name(self, icon_name: str | None) -> None:
        if icon_name != self._icon_name:
            self._icon_name = icon_name
            self._refresh_icon()

    def _refresh_icon(self) -> None:
        if not self._icon_name:
            self.setIcon(QIcon())
            return
        variant = self.property("variant") or ""
        palette = theme.palette()
        if variant in self._FIXED_ICON_COLORS:
            icon_color = self._FIXED_ICON_COLORS[variant]
        else:
            icon_color = palette[self._ICON_COLORS.get(variant, "text")]
        self.setIcon(icon(self._icon_name, icon_color))


class Card(QFrame):
    """Superficie con borde redondeado y título opcional."""

    def __init__(self, title: str | None = None, icon_name: str | None = None, parent=None, spacing: int = 10):
        super().__init__(parent)
        self.setProperty("role", "card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(16, 14, 16, 16)
        self._layout.setSpacing(spacing)
        self.header_layout = None
        if title:
            self.header_layout = QHBoxLayout()
            self.header_layout.setSpacing(8)
            if icon_name:
                self._icon_label = QLabel()
                self._icon_name = icon_name
                self._refresh_title_icon()
                theme.theme_state().changed.connect(self._refresh_title_icon)
                self.header_layout.addWidget(self._icon_label)
            self.title_label = make_label(title.upper(), role="cardTitle")
            self.header_layout.addWidget(self.title_label)
            self.header_layout.addStretch(1)
            self._layout.addLayout(self.header_layout)

    def _refresh_title_icon(self):
        self._icon_label.setPixmap(icon(self._icon_name, theme.palette()["text_muted"], 16).pixmap(16, 16))

    @property
    def body(self) -> QVBoxLayout:
        return self._layout


def divider() -> QFrame:
    line = QFrame()
    line.setProperty("role", "divider")
    return line


# --------------------------------------------------------------------------
# Badge de estado (píldora con punto de color)
# --------------------------------------------------------------------------
class StatusBadge(QWidget):
    def __init__(self, text: str = "", tone: str = "neutral", large: bool = False, parent=None):
        super().__init__(parent)
        self._text = text
        self._tone = tone
        self._large = large
        self._pulse_value = 1.0
        self._anim = QVariantAnimation(self)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setDuration(1100)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_pulse)
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        theme.theme_state().changed.connect(self._on_theme_changed)

    def _on_theme_changed(self):
        self.updateGeometry()
        self.update()

    def _font(self):
        return theme.font("h2" if self._large else "small", bold=True)

    def set_status(self, text: str, tone: str = "neutral", pulse: bool = False) -> None:
        changed = text != self._text or tone != self._tone
        self._text = text
        self._tone = tone
        if pulse and self._anim.state() != QVariantAnimation.Running:
            self._anim.start()
        elif not pulse and self._anim.state() == QVariantAnimation.Running:
            self._anim.stop()
            self._pulse_value = 1.0
            changed = True
        if changed:
            self.updateGeometry()
            self.update()

    def text(self) -> str:
        return self._text

    def _on_pulse(self, value):
        self._pulse_value = 0.3 + 0.7 * (1.0 - abs(2.0 * float(value) - 1.0))
        self.update()

    def sizeHint(self) -> QSize:
        fm = QFontMetrics(self._font())
        height = fm.height() + (16 if self._large else 10)
        width = fm.horizontalAdvance(self._text) + height + (22 if self._large else 14)
        return QSize(width, height)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        radius = rect.height() / 2
        tone = theme.tone_color(self._tone)
        high_contrast = theme.current_theme() == "high_contrast"

        fill = QColor(tone)
        fill.setAlpha(0 if high_contrast else 40)
        border = QColor(tone)
        border.setAlpha(255 if high_contrast else 150)
        painter.setPen(QPen(border, 2 if high_contrast else 1.2))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, radius, radius)

        dot_r = rect.height() * 0.17
        dot_x = rect.left() + radius
        dot = QColor(tone)
        dot.setAlphaF(self._pulse_value)
        painter.setPen(Qt.NoPen)
        painter.setBrush(dot)
        painter.drawEllipse(QRectF(dot_x - dot_r, rect.center().y() - dot_r, dot_r * 2, dot_r * 2))

        painter.setPen(theme.color("text"))
        painter.setFont(self._font())
        text_rect = rect.adjusted(radius + dot_r + 6, 0, -radius * 0.6, 0)
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self._text)


# --------------------------------------------------------------------------
# Vista de video pintada (más barata que escalar QPixmaps en un QLabel)
# --------------------------------------------------------------------------
class VideoView(QWidget):
    def __init__(self, placeholder: str = "Cámara inactiva", radius: int = 14,
                 min_size: tuple[int, int] = (480, 270), circular: bool = False, parent=None):
        super().__init__(parent)
        self._placeholder = placeholder
        self._radius = radius
        self._circular = circular
        self._array = None
        self._image = None
        self._overlay_text = ""
        self._overlay_tone = "accent"
        self._corner_text = ""
        self._corner_tone = "danger"
        self.setMinimumSize(*min_size)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAttribute(Qt.WA_OpaquePaintEvent, False)
        theme.theme_state().changed.connect(self.update)

    def has_frame(self) -> bool:
        return self._image is not None

    def set_frame(self, frame_rgb: np.ndarray | None) -> None:
        if frame_rgb is None:
            return
        frame = np.ascontiguousarray(frame_rgb)
        height, width = frame.shape[:2]
        self._array = frame  # mantiene viva la memoria que usa QImage
        self._image = QImage(frame.data, width, height, 3 * width, QImage.Format_RGB888)
        self.update()

    def clear(self, placeholder: str | None = None) -> None:
        self._array = None
        self._image = None
        if placeholder is not None:
            self._placeholder = placeholder
        self.update()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def set_overlay_text(self, text: str, tone: str = "accent") -> None:
        if text != self._overlay_text or tone != self._overlay_tone:
            self._overlay_text = text
            self._overlay_tone = tone
            self.update()

    def set_corner_badge(self, text: str, tone: str = "danger") -> None:
        if text != self._corner_text or tone != self._corner_tone:
            self._corner_text = text
            self._corner_tone = tone
            self.update()

    def _shape_path(self, rect: QRectF) -> QPainterPath:
        path = QPainterPath()
        if self._circular:
            side = min(rect.width(), rect.height())
            path.addEllipse(QRectF(rect.center().x() - side / 2, rect.center().y() - side / 2, side, side))
        else:
            path.addRoundedRect(rect, self._radius, self._radius)
        return path

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        path = self._shape_path(rect)
        painter.save()
        painter.setClipPath(path)
        painter.fillPath(path, theme.color("video_bg"))

        if self._image is not None:
            img_w, img_h = self._image.width(), self._image.height()
            box = path.boundingRect()
            if self._circular:
                scale = max(box.width() / img_w, box.height() / img_h)
            else:
                scale = min(box.width() / img_w, box.height() / img_h)
            target_w, target_h = img_w * scale, img_h * scale
            target = QRectF(
                box.center().x() - target_w / 2,
                box.center().y() - target_h / 2,
                target_w,
                target_h,
            )
            painter.drawImage(target, self._image)
        else:
            self._paint_placeholder(painter, rect)

        if self._overlay_text:
            shade = QColor(0, 0, 0, 120)
            painter.fillPath(path, shade)
            overlay_font = theme.font("hero", bold=True)
            if len(self._overlay_text) <= 2:
                overlay_font.setPixelSize(int(min(rect.height() * 0.42, 180)))
            painter.setFont(overlay_font)
            painter.setPen(theme.tone_color(self._overlay_tone))
            painter.drawText(rect, Qt.AlignCenter | Qt.TextWordWrap, self._overlay_text)
        painter.restore()

        if self._corner_text and not self._circular:
            self._paint_corner_badge(painter, rect)

        border = theme.color("border_strong" if theme.current_theme() == "high_contrast" else "border")
        painter.setPen(QPen(border, theme.palette()["border_width"]))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

    def _paint_placeholder(self, painter: QPainter, rect: QRectF):
        muted = theme.palette()["text_subtle"]
        icon_size = 44 if not self._circular else max(20, int(rect.height() * 0.35))
        pix = icon("camera", muted, icon_size).pixmap(icon_size, icon_size)
        has_text = bool(self._placeholder) and not self._circular
        cy = rect.center().y() - (16 if has_text else 0)
        painter.drawPixmap(int(rect.center().x() - icon_size / 2), int(cy - icon_size / 2), pix)
        if has_text:
            painter.setPen(QColor(muted))
            painter.setFont(theme.font("body"))
            text_rect = QRectF(rect.left(), cy + icon_size / 2 + 6, rect.width(), 40)
            painter.drawText(text_rect, Qt.AlignHCenter | Qt.AlignTop, self._placeholder)

    def _paint_corner_badge(self, painter: QPainter, rect: QRectF):
        painter.setFont(theme.font("small", bold=True))
        fm = painter.fontMetrics()
        pad = 10
        width = fm.horizontalAdvance(self._corner_text) + pad * 2 + 14
        height = fm.height() + 10
        badge = QRectF(rect.left() + 14, rect.top() + 14, width, height)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))
        painter.drawRoundedRect(badge, height / 2, height / 2)
        painter.setBrush(theme.tone_color(self._corner_tone))
        painter.drawEllipse(QRectF(badge.left() + pad, badge.center().y() - 4, 8, 8))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(badge.adjusted(pad + 14, 0, 0, 0), Qt.AlignVCenter | Qt.AlignLeft, self._corner_text)


# --------------------------------------------------------------------------
# Medidores
# --------------------------------------------------------------------------
class MeterBar(QWidget):
    """Barra horizontal pintada: nivel de micrófono o confianza."""

    def __init__(self, mode: str = "level", height: int = 10, parent=None):
        super().__init__(parent)
        self._mode = mode
        self._value = 0.0
        self._peak = None
        self._tone = "accent"
        self.setFixedHeight(height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        theme.theme_state().changed.connect(self.update)

    def set_value(self, value: float, peak: float | None = None, tone: str | None = None) -> None:
        value = max(0.0, min(100.0, float(value or 0)))
        peak = None if peak is None else max(0.0, min(100.0, float(peak)))
        tone = tone or self._tone
        if abs(value - self._value) < 0.5 and peak == self._peak and tone == self._tone:
            return
        self._value, self._peak, self._tone = value, peak, tone
        self.update()

    def _fill_color(self) -> QColor:
        if self._mode == "level":
            if self._value >= 70:
                return theme.color("danger")
            if self._value >= 35:
                return theme.color("warning")
            return theme.color("success")
        return theme.tone_color(self._tone)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        radius = rect.height() / 2
        painter.setPen(Qt.NoPen)
        painter.setBrush(theme.color("surface_hover"))
        painter.drawRoundedRect(rect, radius, radius)
        if self._value > 0:
            fill = QRectF(rect.left(), rect.top(), max(rect.height(), rect.width() * self._value / 100.0), rect.height())
            painter.setBrush(self._fill_color())
            painter.drawRoundedRect(fill, radius, radius)
        if self._peak:
            x = rect.left() + rect.width() * self._peak / 100.0
            painter.setPen(QPen(theme.color("text"), 2))
            painter.drawLine(int(x), int(rect.top() + 1), int(x), int(rect.bottom() - 1))


# --------------------------------------------------------------------------
# Fichas de gesto
# --------------------------------------------------------------------------
class GestureChip(QWidget):
    def __init__(self, gesture_id: str, title: str, action: str, parent=None):
        super().__init__(parent)
        self.gesture_id = gesture_id
        self._title = title
        self._action = action
        self._glow = 0.0
        self._candidate = False
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(1600)
        self._anim.setStartValue(1.0)
        self._anim.setKeyValueAt(0.35, 1.0)
        self._anim.setEndValue(0.0)
        self._anim.setEasingCurve(QEasingCurve.InOutQuad)
        self._anim.valueChanged.connect(self._on_glow)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip(f"{title} → {action}")
        theme.theme_state().changed.connect(self._on_theme_changed)

    def _on_theme_changed(self):
        self.updateGeometry()
        self.update()

    def _on_glow(self, value):
        self._glow = float(value)
        self.update()

    def flash(self) -> None:
        self._anim.stop()
        self._glow = 1.0
        self.update()
        self._anim.start()

    def set_candidate(self, active: bool) -> None:
        if active != self._candidate:
            self._candidate = active
            self.update()

    def sizeHint(self) -> QSize:
        title_fm = QFontMetrics(theme.font("body", bold=True))
        action_fm = QFontMetrics(theme.font("small"))
        width = max(title_fm.horizontalAdvance(self._title), action_fm.horizontalAdvance(self._action)) + 28
        height = title_fm.height() + action_fm.height() + 20
        return QSize(max(120, width), height)

    def minimumSizeHint(self) -> QSize:
        hint = self.sizeHint()
        return QSize(110, hint.height())

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = theme.palette()
        rect = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        base = QColor(palette["surface_alt"])
        success = QColor(palette["success"])
        glow = self._glow
        fill = QColor(
            int(base.red() + (success.red() - base.red()) * glow * 0.9),
            int(base.green() + (success.green() - base.green()) * glow * 0.9),
            int(base.blue() + (success.blue() - base.blue()) * glow * 0.9),
        )
        if glow > 0.02:
            border = success
            border_w = 2
        elif self._candidate:
            border = QColor(palette["accent"])
            border_w = 2
        else:
            border = QColor(palette["border"])
            border_w = palette["border_width"]
        painter.setPen(QPen(border, border_w))
        painter.setBrush(fill)
        painter.drawRoundedRect(rect, 12, 12)

        dark_text = glow > 0.5
        title_color = QColor("#04210f") if dark_text else QColor(palette["text"])
        action_color = QColor("#0b3d1f") if dark_text else QColor(palette["text_muted"])
        inner = rect.adjusted(12, 8, -12, -8)
        title_font = theme.font("body", bold=True)
        action_font = theme.font("small")
        title_h = QFontMetrics(title_font).height()
        painter.setFont(title_font)
        painter.setPen(title_color)
        elided = QFontMetrics(title_font).elidedText(self._title, Qt.ElideRight, int(inner.width()))
        painter.drawText(QRectF(inner.left(), inner.top(), inner.width(), title_h), Qt.AlignLeft | Qt.AlignVCenter, elided)
        painter.setFont(action_font)
        painter.setPen(action_color)
        elided_action = QFontMetrics(action_font).elidedText(self._action, Qt.ElideRight, int(inner.width()))
        painter.drawText(
            QRectF(inner.left(), inner.top() + title_h + 2, inner.width(), inner.height() - title_h),
            Qt.AlignLeft | Qt.AlignTop,
            elided_action,
        )


class GestureStrip(QWidget):
    """Rejilla con una ficha por gesto habilitado."""

    def __init__(self, gestures: list[dict], columns: int = 3, parent=None):
        super().__init__(parent)
        self.chips: dict[str, GestureChip] = {}
        self._candidate = None
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        for index, gesture in enumerate(gestures):
            chip = GestureChip(
                gesture["id"],
                gesture.get("short_title", gesture["title"]),
                gesture.get("short_action", gesture["action"]),
            )
            self.chips[gesture["id"]] = chip
            layout.addWidget(chip, index // columns, index % columns)

    def flash(self, gesture_id: str) -> bool:
        chip = self.chips.get(gesture_id)
        if chip is None:
            return False
        chip.flash()
        return True

    def set_candidate(self, gesture_id: str | None) -> None:
        if gesture_id == self._candidate:
            return
        if self._candidate in self.chips:
            self.chips[self._candidate].set_candidate(False)
        self._candidate = gesture_id if gesture_id in self.chips else None
        if self._candidate:
            self.chips[self._candidate].set_candidate(True)


# --------------------------------------------------------------------------
# Tarjeta de métrica
# --------------------------------------------------------------------------
class StatCard(QFrame):
    def __init__(self, caption: str, value: str = "-", tone: str | None = None, parent=None):
        super().__init__(parent)
        self.setProperty("role", "card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(2)
        self.value_label = make_label(value, role="display", tone=tone)
        self.caption_label = make_label(caption, role="muted", wrap=True)
        layout.addWidget(self.value_label)
        layout.addWidget(self.caption_label)

    def set_value(self, value: str, tone: str | None = None) -> None:
        set_label_text(self.value_label, value)
        theme.set_widget_property(self.value_label, "tone", tone or "")


# --------------------------------------------------------------------------
# Sección plegable
# --------------------------------------------------------------------------
class CollapsibleSection(QWidget):
    toggled = Signal(bool)

    def __init__(self, title: str, expanded: bool = False, parent=None):
        super().__init__(parent)
        self._title = title.upper()
        self.button = QToolButton()
        self.button.setProperty("role", "section")
        self.button.setCheckable(True)
        self.button.setChecked(expanded)
        self.button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.button.setCursor(Qt.PointingHandCursor)
        self.button.setText(self._title)
        self.button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.button.toggled.connect(self._on_toggled)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(4, 2, 4, 4)
        self.content_layout.setSpacing(6)
        self.content.setVisible(expanded)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.button)
        layout.addWidget(self.content)
        self._refresh_icon()
        theme.theme_state().changed.connect(self._refresh_icon)

    def _refresh_icon(self):
        name = "chevron_down" if self.button.isChecked() else "chevron_right"
        self.button.setIcon(icon(name, theme.palette()["text_muted"], 16))

    def _on_toggled(self, checked: bool):
        self.content.setVisible(checked)
        self._refresh_icon()
        self.toggled.emit(checked)

    def is_expanded(self) -> bool:
        return self.button.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self.button.setChecked(expanded)


# --------------------------------------------------------------------------
# Log de eventos
# --------------------------------------------------------------------------
class EventLog(QPlainTextEdit):
    MAX_LINES = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "log")
        self.setReadOnly(True)
        self.setMaximumBlockCount(self.MAX_LINES)
        self.setLineWrapMode(QPlainTextEdit.WidgetWidth)

    @staticmethod
    def _tone_for(text: str) -> str:
        lowered = text.lower()
        if "error" in lowered or "no se pudo" in lowered:
            return "danger"
        if "no reconocido" in lowered or "sin " in lowered or "rechaz" in lowered:
            return "warning"
        if "reconocido" in lowered or "ejecut" in lowered or "activad" in lowered or "iniciad" in lowered:
            return "success"
        if "escuch" in lowered or "texto" in lowered:
            return "info"
        return "text_muted"

    def add(self, text: str) -> None:
        palette = theme.palette()
        stamp = datetime.now().strftime("%H:%M:%S")
        tone = self._tone_for(text)
        tone_hex = palette[tone] if tone in palette else palette["text_muted"]
        self.appendHtml(
            f'<span style="color:{palette["text_subtle"]}">{stamp}</span>&nbsp;&nbsp;'
            f'<span style="color:{tone_hex}">{escape(text)}</span>'
        )


# --------------------------------------------------------------------------
# Tarjeta de perfil (para listas)
# --------------------------------------------------------------------------
def format_accuracy(value) -> str:
    try:
        return f"{float(value) * 100:.1f} %".replace(".", ",")
    except (TypeError, ValueError):
        return "-"


def format_date(value) -> str:
    if not value:
        return "-"
    try:
        return datetime.fromisoformat(str(value)).strftime("%d/%m/%Y")
    except ValueError:
        return str(value)[:10]


class ProfileCard(QWidget):
    def __init__(self, summary: dict, parent=None):
        super().__init__(parent)
        self.summary = summary
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(12)

        self.avatar = QLabel(summary["name"][:1].upper() or "?")
        self.avatar.setProperty("role", "logo")
        self.avatar.setAlignment(Qt.AlignCenter)
        self.avatar.setFixedSize(40, 40)
        layout.addWidget(self.avatar)

        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        name = make_label(summary["name"], role="h2")
        text_col.addWidget(name)
        if summary.get("model_version") is not None:
            meta = (
                f"Modelo v{summary['model_version']}  ·  "
                f"{format_accuracy(summary.get('validation_accuracy'))} precisión  ·  "
                f"{format_date(summary.get('trained_at'))}"
            )
        else:
            meta = "Sin modelo entrenado"
        text_col.addWidget(make_label(meta, role="subtle"))
        layout.addLayout(text_col, 1)

        badge = StatusBadge()
        if summary.get("calibrated"):
            badge.set_status("Calibrado", "success")
        else:
            badge.set_status("Sin calibrar", "warning")
        layout.addWidget(badge, 0, Qt.AlignVCenter)


def populate_profile_list(list_widget: QListWidget, summaries: list[dict], select_name: str | None = None) -> None:
    list_widget.clear()
    selected_row = 0
    for row, summary in enumerate(summaries):
        card = ProfileCard(summary)
        item = QListWidgetItem()
        item.setData(Qt.UserRole, summary["name"])
        card.ensurePolished()
        hint = card.sizeHint()
        item.setSizeHint(QSize(hint.width(), max(hint.height(), theme.font_px("h2") * 2 + 34) + 8))
        list_widget.addItem(item)
        list_widget.setItemWidget(item, card)
        if summary["name"] == select_name:
            selected_row = row
    if list_widget.count():
        list_widget.setCurrentRow(selected_row)


# --------------------------------------------------------------------------
# Matriz de confusión pintada (sigue el tema activo)
# --------------------------------------------------------------------------
class ConfusionMatrixView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._matrix: list[list[float]] = []
        self._labels: list[str] = []
        self.setMinimumSize(320, 280)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        theme.theme_state().changed.connect(self.update)

    def set_data(self, matrix, labels: list[str]) -> None:
        self._matrix = [list(map(float, row)) for row in (matrix or [])]
        self._labels = list(labels or [])
        self.update()

    def has_data(self) -> bool:
        return bool(self._matrix)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        palette = theme.palette()
        rect = QRectF(self.rect())
        count = len(self._matrix)
        if count == 0 or len(self._labels) != count:
            painter.setPen(QColor(palette["text_subtle"]))
            painter.setFont(theme.font("body"))
            painter.drawText(rect, Qt.AlignCenter, "Sin matriz de confusión disponible")
            return

        label_font = theme.font("small")
        fm = QFontMetrics(label_font)
        row_labels = [f"{index + 1}. {label}" for index, label in enumerate(self._labels)]
        label_w = min(max(fm.horizontalAdvance(label) for label in row_labels) + 12, rect.width() * 0.4)
        axis_h = fm.height() + 8
        available_w = rect.width() - label_w - 4
        available_h = rect.height() - axis_h * 2 - 6
        cell = max(12.0, min(available_w / count, available_h / count))
        grid_left = rect.left() + label_w
        grid_top = rect.top() + axis_h
        accent = QColor(palette["accent"])
        danger = QColor(palette["danger"])
        surface = QColor(palette["surface_alt"])

        value_font = theme.font("small", bold=True)
        for r, row in enumerate(self._matrix):
            row_total = sum(row) or 1.0
            for c, value in enumerate(row):
                ratio = value / row_total
                if value > 0 and r != c:
                    ratio = 0.35 + 0.65 * ratio  # los errores deben verse aunque sean pocos
                base = accent if r == c else danger
                fill = QColor(
                    int(surface.red() + (base.red() - surface.red()) * ratio),
                    int(surface.green() + (base.green() - surface.green()) * ratio),
                    int(surface.blue() + (base.blue() - surface.blue()) * ratio),
                )
                cell_rect = QRectF(grid_left + c * cell + 1, grid_top + r * cell + 1, cell - 2, cell - 2)
                painter.setPen(Qt.NoPen)
                painter.setBrush(fill)
                painter.drawRoundedRect(cell_rect, 4, 4)
                if value > 0 and cell >= 22:
                    painter.setFont(value_font)
                    painter.setPen(QColor("#ffffff") if ratio > 0.45 else QColor(palette["text"]))
                    painter.drawText(cell_rect, Qt.AlignCenter, str(int(value)))

        painter.setFont(label_font)
        painter.setPen(QColor(palette["text_muted"]))
        for index, label in enumerate(row_labels):
            row_rect = QRectF(rect.left(), grid_top + index * cell, label_w - 8, cell)
            painter.drawText(row_rect, Qt.AlignRight | Qt.AlignVCenter, fm.elidedText(label, Qt.ElideRight, int(label_w - 8)))
            # Etiqueta de columna: número corto para no saturar; la fila tiene el nombre.
            row_num = QRectF(grid_left + index * cell, rect.top(), cell, axis_h)
            painter.drawText(row_num, Qt.AlignHCenter | Qt.AlignBottom, str(index + 1))
        painter.setPen(QColor(palette["text_subtle"]))
        caption = "Filas: gesto real · Columnas: gesto predicho"
        painter.drawText(
            QRectF(rect.left(), grid_top + count * cell + 4, grid_left - rect.left() + count * cell, axis_h),
            Qt.AlignHCenter | Qt.AlignTop,
            caption,
        )

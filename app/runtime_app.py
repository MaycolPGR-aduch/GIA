from __future__ import annotations

import sys
import webbrowser
from html import escape

from PySide6.QtCore import QObject, QPoint, Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from app import ui_theme as theme
from app.gesture_catalog import get_enabled_gesture_catalog
from app.ui_widgets import (
    Button,
    Card,
    CollapsibleSection,
    EventLog,
    GestureStrip,
    MeterBar,
    StatusBadge,
    VideoView,
    format_accuracy,
    make_label,
    set_label_text,
)
from app.voice_router import get_voice_help_rows


# Textos y tonos de la tarjeta de voz según el estado que publica el controlador.
VOICE_STATES = {
    "en espera": ("Cierra ambos ojos para dar una orden", "muted"),
    "escuchando": ("Escuchando… habla ahora", "info"),
    "comando reconocido": ("Comando reconocido", "success"),
    "comando no reconocido": ("No se reconoció el comando", "warning"),
    "sin audio reconocido": ("No se escuchó ninguna frase", "warning"),
    "deshabilitada": ("Voz deshabilitada en este perfil", "muted"),
    "error": ("Error en el reconocimiento de voz", "danger"),
}

MIC_STATES = {
    "activo": "Micrófono activo",
    "muestreo en pausa": "Micrófono en uso por el reconocimiento",
    "sin señal": "Micrófono sin señal",
    "error": "Error de micrófono",
    "deshabilitado": "Monitor de micrófono deshabilitado",
}


def system_status(payload: dict) -> tuple[str, str, bool]:
    """Devuelve (texto, tono, pulso) del estado general del sistema."""
    mode = payload.get("mode", "")
    face = payload.get("face", "")
    runtime_state = payload.get("runtime_state", "")
    if face == "sin cámara" or mode == "error":
        return "Sin cámara", "danger", False
    if mode == "listening":
        return "Escuchando…", "info", True
    if mode == "paused":
        return "En pausa", "warning", False
    if mode == "stopped":
        return "Detenido", "neutral", False
    if face == "sin rostro":
        return "Buscando rostro", "neutral", True
    if runtime_state == "validando gesto":
        return "Validando gesto", "accent", False
    if face == "inestable":
        return "Rostro inestable", "warning", False
    return "Activo", "success", False


class RuntimeSignals(QObject):
    video_update = Signal(object, object)
    status_update = Signal(dict)
    event_update = Signal(str)
    ui_action = Signal(object)


class CompactWindow(QWidget):
    """Barra flotante mínima para usar el PC con la interfaz reducida."""

    def __init__(self, parent_gui: RuntimeMainWindow):
        super().__init__()
        self.parent_gui = parent_gui
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowTitle("GIA · Modo compacto")
        self.drag_position = QPoint()
        self.init_ui()

    def init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.container = QFrame()
        self.container.setProperty("role", "compact")
        outer.addWidget(self.container)

        layout = QHBoxLayout(self.container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)

        self.video_view = VideoView(placeholder="", circular=True, min_size=(92, 92))
        self.video_view.setFixedSize(92, 92)
        layout.addWidget(self.video_view)

        info = QVBoxLayout()
        info.setSpacing(6)
        badges = QHBoxLayout()
        badges.setSpacing(6)
        self.status_badge = StatusBadge("Iniciando", "neutral")
        self.cursor_badge = StatusBadge("Cursor congelado", "neutral")
        badges.addWidget(self.status_badge)
        badges.addWidget(self.cursor_badge)
        badges.addStretch(1)
        info.addLayout(badges)

        self.last_label = make_label("Listo", role="h2")
        info.addWidget(self.last_label)
        self.voice_label = make_label("Cierra ambos ojos para dar una orden", role="subtle")
        info.addWidget(self.voice_label)
        self.mic_meter = MeterBar("level", height=6)
        info.addWidget(self.mic_meter)
        layout.addLayout(info, 1)

        btn_restore = Button("", "expand", variant="primary", tooltip="Volver a la ventana completa (Ctrl+M)")
        btn_restore.setFixedSize(44, 44)
        btn_restore.clicked.connect(self.parent_gui.exit_compact_mode)
        layout.addWidget(btn_restore, 0, Qt.AlignVCenter)

        self.resize(560, 124)
        stored = getattr(self.parent_gui, "_compact_position", None)
        if stored is not None:
            self.move(stored)
        else:
            screen = QApplication.primaryScreen().availableGeometry()
            self.move(screen.x() + (screen.width() - self.width()) // 2, screen.y() + 16)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()

    def moveEvent(self, event):
        self.parent_gui._compact_position = self.pos()
        super().moveEvent(event)


class RuntimeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.settings = {}
        self.is_compact_mode = False
        self.compact_window = None
        self.frame_pending = False
        self.presentation_mode = False
        self._pre_presentation_scale = None
        self._compact_position = None
        self._last_payload: dict = {}
        self._last_mode = None
        self._last_cursor = None
        self.gesture_catalog = get_enabled_gesture_catalog()
        self.gesture_title_map = {gesture["id"]: gesture.get("short_title", gesture["title"]) for gesture in self.gesture_catalog}

        self.signals = RuntimeSignals()
        self.signals.video_update.connect(self.update_video_feed)
        self.signals.status_update.connect(self.update_status)
        self.signals.event_update.connect(self.append_event)
        self.signals.ui_action.connect(self._run_ui_action)

        self.init_ui()
        self._install_shortcuts()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        self.setWindowTitle("GIA · Asistente facial")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1180, int(screen.width() * 0.94)), min(720, int(screen.height() * 0.9)))

        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        main = QVBoxLayout(root)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(12)

        main.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        main.addLayout(body, 1)

        # Columna izquierda: cámara + gestos
        left = QVBoxLayout()
        left.setSpacing(10)
        self.video_view = VideoView(placeholder="Esperando la señal de la cámara…", min_size=(480, 270))
        left.addWidget(self.video_view, 1)
        self.gesture_strip = GestureStrip(self.gesture_catalog, columns=3)
        left.addWidget(self.gesture_strip)
        body.addLayout(left, 3)

        # Columna derecha con scroll para pantallas pequeñas
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        right_scroll.setMinimumWidth(340)
        right_panel = QWidget()
        right = QVBoxLayout(right_panel)
        right.setContentsMargins(0, 0, 4, 0)
        right.setSpacing(12)
        right_scroll.setWidget(right_panel)
        body.addWidget(right_scroll, 2)

        right.addWidget(self._build_gesture_card())
        right.addWidget(self._build_voice_card())
        right.addWidget(self._build_controls_card())
        right.addWidget(self._build_diagnostics())
        right.addStretch(1)

    def _build_header(self) -> QFrame:
        header = QFrame()
        header.setProperty("role", "header")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(12)

        logo = make_label("GIA", role="logo")
        layout.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        titles.addWidget(make_label("Asistente facial", role="title"))
        self.lbl_subtitle = make_label("Perfil: -", role="muted")
        titles.addWidget(self.lbl_subtitle)
        layout.addLayout(titles)
        layout.addStretch(1)

        self.status_badge = StatusBadge("Iniciando", "neutral", large=True)
        self.cursor_badge = StatusBadge("Cursor congelado", "neutral", large=True)
        layout.addWidget(self.status_badge)
        layout.addWidget(self.cursor_badge)

        layout.addSpacing(6)
        self.btn_contrast = Button("", "contrast", variant="ghost", tooltip="Alto contraste (Ctrl+H)", checkable=True)
        self.btn_contrast.setChecked(theme.current_theme() == "high_contrast")
        self.btn_contrast.clicked.connect(self._toggle_contrast)
        self.btn_presentation = Button("", "monitor", variant="ghost", tooltip="Modo presentación (F5)", checkable=True)
        self.btn_presentation.clicked.connect(self.toggle_presentation_mode)
        btn_help = Button("", "help", variant="ghost", tooltip="Guía rápida y atajos")
        btn_help.clicked.connect(self.show_guide_dialog)
        for button in (self.btn_contrast, self.btn_presentation, btn_help):
            button.setFixedSize(40, 40)
            layout.addWidget(button)
        return header

    def _build_gesture_card(self) -> Card:
        card = Card("Último gesto", "face")
        self.lbl_last_gesture = make_label("Esperando gesto…", role="display")
        self.lbl_last_action = make_label("Haz un gesto para ver aquí la acción ejecutada.", role="muted", wrap=True)
        confidence_row = QHBoxLayout()
        confidence_row.setSpacing(8)
        self.confidence_bar = MeterBar("confidence", height=8)
        self.lbl_confidence = make_label("—", role="small")
        self.lbl_confidence.setMinimumWidth(42)
        self.lbl_confidence.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        confidence_row.addWidget(make_label("Confianza", role="subtle"))
        confidence_row.addWidget(self.confidence_bar, 1)
        confidence_row.addWidget(self.lbl_confidence)
        card.body.addWidget(self.lbl_last_gesture)
        card.body.addWidget(self.lbl_last_action)
        card.body.addLayout(confidence_row)
        return card

    def _build_voice_card(self) -> Card:
        card = Card("Voz", "mic")
        self.lbl_voice_state = make_label("Cierra ambos ojos para dar una orden", role="h2", tone="muted", wrap=True)
        self.lbl_voice_text = make_label("", role="quote", wrap=True)
        self.lbl_voice_text.setVisible(False)
        self.lbl_voice_cmd = make_label("", role="muted", wrap=True)
        self.lbl_voice_cmd.setVisible(False)
        mic_row = QHBoxLayout()
        mic_row.setSpacing(8)
        self.mic_meter = MeterBar("level", height=8)
        mic_row.addWidget(make_label("Mic", role="subtle"))
        mic_row.addWidget(self.mic_meter, 1)
        card.body.addWidget(self.lbl_voice_state)
        card.body.addWidget(self.lbl_voice_text)
        card.body.addWidget(self.lbl_voice_cmd)
        card.body.addLayout(mic_row)
        return card

    def _build_controls_card(self) -> Card:
        card = Card("Controles", "sliders")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.btn_pause = Button("Pausar", "pause", variant="primary", tooltip="Pausar o reanudar el sistema (Ctrl+P · Ctrl+Alt+P desde cualquier app)")
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_cursor = Button("Cursor congelado", "cursor", tooltip="Activar o congelar el cursor (Ctrl+K)")
        self.btn_cursor.clicked.connect(self._toggle_cursor)
        btn_recenter = Button("Recentrar", "target", tooltip="Toma la posición actual de la nariz como centro (Ctrl+R)")
        btn_recenter.clicked.connect(self._recenter)
        btn_compact = Button("Modo compacto", "compact", tooltip="Reduce la interfaz a una barra flotante (Ctrl+M)")
        btn_compact.clicked.connect(self.enter_compact_mode)
        btn_guide = Button("Guía", "book", tooltip="Gestos, comandos y atajos")
        btn_guide.clicked.connect(self.show_guide_dialog)
        btn_voice_guide = Button("Comandos de voz", "list", tooltip="Frases disponibles")
        btn_voice_guide.clicked.connect(self.show_voice_commands_dialog)
        btn_voice_test = Button("Probar voz", "mic", tooltip="Escucha una orden sin usar el gesto de ojos cerrados")
        btn_voice_test.clicked.connect(self._test_voice)
        btn_close = Button("Cerrar", "power", variant="danger", tooltip="Terminar la sesión")
        btn_close.clicked.connect(self._confirm_quit)

        grid.addWidget(self.btn_pause, 0, 0)
        grid.addWidget(self.btn_cursor, 0, 1)
        grid.addWidget(btn_recenter, 1, 0)
        grid.addWidget(btn_compact, 1, 1)
        grid.addWidget(btn_guide, 2, 0)
        grid.addWidget(btn_voice_guide, 2, 1)
        grid.addWidget(btn_voice_test, 3, 0)
        grid.addWidget(btn_close, 3, 1)
        card.body.addLayout(grid)
        return card

    def _build_diagnostics(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.diag_section = CollapsibleSection("Diagnóstico técnico", expanded=False)
        diag_grid = QGridLayout()
        diag_grid.setHorizontalSpacing(12)
        diag_grid.setVerticalSpacing(4)
        self._diag_labels = {}
        rows = [
            ("fps", "FPS real"),
            ("face", "Rostro"),
            ("detail", "Detalle"),
            ("prediction", "Predicción"),
            ("rejection", "Rechazo técnico"),
            ("hint", "Pista"),
            ("mic", "Micrófono"),
            ("mic_activity", "Actividad mic."),
            ("model", "Modelo"),
        ]
        for row, (key, caption) in enumerate(rows):
            diag_grid.addWidget(make_label(caption, role="subtle"), row, 0, Qt.AlignTop)
            value = make_label("-", role="small", wrap=True)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            diag_grid.addWidget(value, row, 1)
            self._diag_labels[key] = value
        diag_grid.setColumnStretch(1, 1)
        self.diag_section.content_layout.addLayout(diag_grid)
        self.check_camera_metrics = QCheckBox("Mostrar métricas sobre la cámara")
        self.check_camera_metrics.toggled.connect(self._toggle_camera_metrics)
        self.diag_section.content_layout.addWidget(self.check_camera_metrics)
        layout.addWidget(self.diag_section)

        self.log_section = CollapsibleSection("Registro de eventos", expanded=False)
        self.event_log = EventLog()
        self.event_log.setMinimumHeight(140)
        self.log_section.content_layout.addWidget(self.event_log)
        layout.addWidget(self.log_section)
        return wrapper

    def _install_shortcuts(self):
        bindings = [
            ("Ctrl+P", self._toggle_pause),
            ("Ctrl+K", self._toggle_cursor),
            ("Ctrl+R", self._recenter),
            ("Ctrl+M", self._toggle_compact),
            ("Ctrl+D", lambda: self.diag_section.set_expanded(not self.diag_section.is_expanded())),
            ("Ctrl+H", self._toggle_contrast),
            ("Ctrl++", lambda: self._step_font(0.1)),
            ("Ctrl+=", lambda: self._step_font(0.1)),
            ("Ctrl+-", lambda: self._step_font(-0.1)),
            ("F5", self.toggle_presentation_mode),
            ("F11", self._toggle_fullscreen),
            ("Esc", self._exit_special_modes),
        ]
        self._shortcuts = []
        for sequence, callback in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    # ------------------------------------------------------------ controlador
    def set_controller(self, controller):
        self.controller = controller
        self.settings = dict(getattr(controller, "settings", {}) or {})
        if hasattr(self.controller, "set_camera_overlay_details"):
            self.controller.set_camera_overlay_details(self.check_camera_metrics.isChecked())
        profile_name = (getattr(controller, "profile", {}) or {}).get("name", "-")
        classifier = getattr(controller, "classifier", None)
        version = getattr(classifier, "active_version", None)
        summary = getattr(classifier, "training_summary", {}) or {}
        parts = [f"Perfil: {profile_name}"]
        if version is not None:
            parts.append(f"Modelo v{version}")
        if summary.get("validation_accuracy") is not None:
            parts.append(f"{format_accuracy(summary['validation_accuracy'])} precisión")
        self.lbl_subtitle.setText("  ·  ".join(parts))
        self._diag_labels["model"].setText(f"v{version}" if version is not None else "sin versión")

    @Slot(object)
    def _run_ui_action(self, callback):
        if callable(callback):
            callback()

    def run_on_ui_thread(self, callback):
        self.signals.ui_action.emit(callback)

    # ----------------------------------------------------------- actualización
    def update_video_feed(self, frame_rgb, compact_frame_rgb=None):
        try:
            if frame_rgb is not None and not self.is_compact_mode:
                self.video_view.set_frame(frame_rgb)
            if compact_frame_rgb is not None and self.compact_window is not None:
                self.compact_window.video_view.set_frame(compact_frame_rgb)
        except Exception as exc:
            self.append_event(f"Error en video: {exc}")
        finally:
            self.frame_pending = False

    def update_status(self, payload: dict):
        previous = self._last_payload
        self._last_payload = payload

        status_text, status_tone, pulse = system_status(payload)
        self.status_badge.set_status(status_text, status_tone, pulse)
        cursor_active = payload.get("cursor_active", payload.get("cursor") == "activo")
        cursor_text, cursor_tone = ("Cursor activo", "success") if cursor_active else ("Cursor congelado", "neutral")
        self.cursor_badge.set_status(cursor_text, cursor_tone)

        gesture_id = payload.get("gesture", "-")
        if payload.get("gesture_executed"):
            self._show_executed_gesture(gesture_id, payload)
        elif payload.get("runtime_state") == "validando gesto":
            self.gesture_strip.set_candidate(gesture_id)
        else:
            self.gesture_strip.set_candidate(None)

        self._update_voice(payload, previous)
        self._update_diagnostics(payload)
        self._update_stateful_buttons(payload)
        self._update_compact(payload, status_text, status_tone, pulse, cursor_text, cursor_tone)

    def _show_executed_gesture(self, gesture_id: str, payload: dict):
        self.gesture_strip.set_candidate(None)
        self.gesture_strip.flash(gesture_id)
        title = payload.get("gesture_title") or self._format_gesture_label(gesture_id)
        action = payload.get("gesture_action") or ""
        confidence = float(payload.get("confidence_value", 0.0) or 0.0)
        set_label_text(self.lbl_last_gesture, self._display_text(title))
        set_label_text(self.lbl_last_action, f"→ {self._display_text(action)}" if action else "Gesto ejecutado")
        self.confidence_bar.set_value(confidence * 100, tone="success" if confidence >= 0.6 else "warning")
        set_label_text(self.lbl_confidence, f"{confidence * 100:.0f} %")

    def _update_voice(self, payload: dict, previous: dict):
        voice_state = self._display_text(payload.get("voice_state", "en espera"))
        text, tone = VOICE_STATES.get(voice_state, (voice_state.capitalize(), "muted"))
        set_label_text(self.lbl_voice_state, text)
        theme.set_widget_property(self.lbl_voice_state, "tone", tone)

        voice_text = self._display_text(payload.get("voice_text", "-"))
        has_text = voice_text not in ("", "-")
        self.lbl_voice_text.setVisible(has_text)
        if has_text:
            set_label_text(self.lbl_voice_text, f"“{voice_text}”")
        voice_command = self._display_text(payload.get("voice_command", "-"))
        has_command = voice_command not in ("", "-")
        self.lbl_voice_cmd.setVisible(has_command)
        if has_command:
            set_label_text(self.lbl_voice_cmd, f"→ Comando: {voice_command}")
        self.mic_meter.set_value(payload.get("mic_level", 0), payload.get("mic_peak"))

    def _update_diagnostics(self, payload: dict):
        if not self.diag_section.is_expanded():
            return
        labels = self._diag_labels
        confidence = payload.get("confidence", "-")
        gesture_id = payload.get("gesture", "-")
        set_label_text(labels["fps"], f"{payload.get('fps_real', 0):.1f}")
        set_label_text(labels["face"], self._display_text(payload.get("face", "-")))
        set_label_text(labels["detail"], self._display_text(payload.get("status_text", "-")))
        set_label_text(labels["prediction"], f"{self._format_gesture_label(gesture_id)} ({confidence})")
        set_label_text(labels["rejection"], self._display_text(payload.get("rejection_reason", "-")) or "-")
        set_label_text(labels["hint"], self._display_text(payload.get("diagnostic_hint", "-")) or "-")
        mic_status = MIC_STATES.get(payload.get("mic_monitor_status", ""), self._display_text(payload.get("mic_monitor_status", "-")))
        set_label_text(labels["mic"], f"{mic_status} · {self._display_text(payload.get('mic_device', '-'))}")
        set_label_text(
            labels["mic_activity"],
            f"Nivel {int(payload.get('mic_level', 0) or 0)} % · Pico {int(payload.get('mic_peak', 0) or 0)} % · "
            f"Actividad {int(float(payload.get('mic_active_ratio', 0.0) or 0.0) * 100)} %",
        )

    def _update_compact(self, payload, status_text, status_tone, pulse, cursor_text, cursor_tone):
        compact = self.compact_window
        if compact is None:
            return
        compact.status_badge.set_status(status_text, status_tone, pulse)
        compact.cursor_badge.set_status(cursor_text, cursor_tone)
        if payload.get("gesture_executed"):
            title = payload.get("gesture_title") or self._format_gesture_label(payload.get("gesture"))
            action = payload.get("gesture_action") or ""
            set_label_text(compact.last_label, f"{title} → {action}" if action else title)
        voice_command = self._display_text(payload.get("voice_command", "-"))
        if payload.get("voice_state") == "comando reconocido" and voice_command not in ("", "-"):
            set_label_text(compact.last_label, f"Voz → {voice_command}")
        voice_text, _ = VOICE_STATES.get(payload.get("voice_state", ""), ("", "muted"))
        set_label_text(compact.voice_label, voice_text)
        compact.mic_meter.set_value(payload.get("mic_level", 0))

    def _format_gesture_label(self, gesture_id):
        if not gesture_id or gesture_id == "-":
            return "-"
        label = self.gesture_title_map.get(gesture_id, str(gesture_id).replace("_", " "))
        return self._display_text(label)

    def _update_stateful_buttons(self, payload: dict):
        mode = payload.get("mode", "-")
        cursor_active = payload.get("cursor_active", payload.get("cursor") == "activo")

        if mode != self._last_mode:
            self._last_mode = mode
            if mode == "paused":
                self.btn_pause.setText("Reanudar")
                self.btn_pause.set_icon_name("play")
                self.btn_pause.set_variant("warning")
            else:
                self.btn_pause.setText("Pausar")
                self.btn_pause.set_icon_name("pause")
                self.btn_pause.set_variant("primary")

        if cursor_active != self._last_cursor:
            self._last_cursor = cursor_active
            if cursor_active:
                self.btn_cursor.setText("Cursor activo")
                self.btn_cursor.set_variant("success")
                self.btn_cursor.setToolTip("El cursor sigue tu cabeza. Clic para congelarlo (Ctrl+K)")
            else:
                self.btn_cursor.setText("Cursor congelado")
                self.btn_cursor.set_variant("")
                self.btn_cursor.setToolTip("El cursor está quieto. Clic para activarlo (Ctrl+K)")

    def append_event(self, text: str):
        self.event_log.add(self._display_text(text))

    @staticmethod
    def _display_text(value) -> str:
        text = "" if value is None else str(value)
        if any(marker in text for marker in ("Ã", "Â", "â")):
            try:
                text = text.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        return text

    # --------------------------------------------------------------- diálogos
    def _build_help_dialog(self, title: str, size: tuple[int, int]) -> tuple[QDialog, QVBoxLayout]:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(*size)
        outer = QVBoxLayout(dialog)
        outer.setContentsMargins(0, 0, 0, 12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 12)
        layout.setSpacing(12)
        scroll.setWidget(content)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = Button("Cerrar", variant="primary")
        btn_close.clicked.connect(dialog.accept)
        close_row.addWidget(btn_close)
        close_row.setContentsMargins(20, 0, 20, 0)
        outer.addLayout(close_row)
        return dialog, layout

    def show_guide_dialog(self):
        dialog, layout = self._build_help_dialog("GIA · Guía rápida", (760, 640))
        layout.addWidget(make_label("Guía rápida de interacción", role="title"))
        layout.addWidget(make_label(
            "Mantén cada gesto el tiempo indicado. Las fichas bajo la cámara se iluminan en verde cuando "
            "un gesto se ejecuta.",
            role="muted",
            wrap=True,
        ))

        layout.addWidget(make_label("GESTOS FACIALES", role="cardTitle"))
        grid = QGridLayout()
        grid.setSpacing(10)
        for index, gesture in enumerate(self.gesture_catalog):
            card = Card()
            card.body.setSpacing(4)
            card.body.addWidget(make_label(self._display_text(gesture["title"]), role="h2", wrap=True))
            card.body.addWidget(make_label(f"→ {self._display_text(gesture['action'])}", tone="accent", wrap=True))
            card.body.addWidget(make_label(
                f"Mantener {gesture['duration_ms']} ms · {self._display_text(gesture['warning'])}",
                role="subtle",
                wrap=True,
            ))
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)

        layout.addWidget(make_label("ATAJOS DE TECLADO", role="cardTitle"))
        shortcuts = [
            ("Ctrl+P", "Pausar / reanudar"),
            ("Ctrl+Alt+P", "Pausa de emergencia (en cualquier app)"),
            ("Ctrl+K", "Activar / congelar cursor"),
            ("Ctrl+R", "Recentrar cursor"),
            ("Ctrl+M", "Modo compacto"),
            ("F5", "Modo presentación"),
            ("Ctrl+H", "Alto contraste"),
            ("Ctrl + / Ctrl −", "Tamaño de letra"),
            ("Ctrl+D", "Diagnóstico técnico"),
            ("F11", "Pantalla completa"),
        ]
        shortcut_card = Card()
        shortcut_grid = QGridLayout()
        shortcut_grid.setHorizontalSpacing(16)
        shortcut_grid.setVerticalSpacing(6)
        for index, (keys, description) in enumerate(shortcuts):
            shortcut_grid.addWidget(make_label(keys, tone="accent"), index // 2, (index % 2) * 2)
            shortcut_grid.addWidget(make_label(description, role="muted"), index // 2, (index % 2) * 2 + 1)
        shortcut_card.body.addLayout(shortcut_grid)
        layout.addWidget(shortcut_card)

        layout.addWidget(make_label("ACCESOS WEB", role="cardTitle"))
        web_row = QHBoxLayout()
        web_row.setSpacing(8)
        for name, url in [
            ("Gmail", "https://gmail.com"),
            ("Facebook", "https://facebook.com"),
            ("WhatsApp", "https://web.whatsapp.com"),
            ("YouTube", "https://youtube.com"),
        ]:
            button = Button(name, "globe")
            button.clicked.connect(lambda checked=False, link=url: webbrowser.open(link))
            web_row.addWidget(button)
        layout.addLayout(web_row)
        layout.addStretch(1)
        dialog.exec()

    def show_voice_commands_dialog(self):
        dialog, layout = self._build_help_dialog("GIA · Comandos de voz", (760, 660))
        layout.addWidget(make_label("Cómo dar una orden por voz", role="title"))
        steps = Card()
        for number, text in enumerate(
            [
                "Cierra ambos ojos de forma deliberada durante un momento.",
                "Espera a que el estado cambie a «Escuchando».",
                "Di una de las frases de la lista, de forma clara y corta.",
                "Revisa en la tarjeta «Voz» la frase reconocida y el comando ejecutado.",
            ],
            start=1,
        ):
            row = QHBoxLayout()
            badge = make_label(str(number), role="logo")
            badge.setFixedSize(28, 28)
            badge.setAlignment(Qt.AlignCenter)
            row.addWidget(badge)
            row.addWidget(make_label(text, wrap=True), 1)
            steps.body.addLayout(row)
        steps.body.addWidget(make_label(
            "Consejo: el botón «Probar voz» permite comprobar micrófono y reconocimiento sin usar el gesto.",
            role="subtle",
            wrap=True,
        ))
        layout.addWidget(steps)

        layout.addWidget(make_label("COMANDOS DISPONIBLES", role="cardTitle"))
        accent = theme.palette()["accent"]
        for row in get_voice_help_rows(self.settings):
            phrases = row.get("spoken_examples", [])[:5]
            card = Card(spacing=4)
            card.body.addWidget(make_label(self._display_text(row["label"]), role="h2"))
            card.body.addWidget(make_label(self._display_text(row["description"]), role="muted", wrap=True))
            if phrases:
                phrase_html = "  ·  ".join(
                    f'<span style="color:{accent}">“{escape(self._display_text(phrase))}”</span>' for phrase in phrases
                )
            else:
                phrase_html = "Sin frases configuradas"
            phrases_label = make_label(phrase_html, wrap=True)
            phrases_label.setTextFormat(Qt.RichText)
            card.body.addWidget(phrases_label)
            layout.addWidget(card)
        layout.addStretch(1)
        dialog.exec()

    # ------------------------------------------------------------------ modos
    def enter_compact_mode(self):
        if self.is_compact_mode:
            return
        self.compact_window = CompactWindow(self)
        self.is_compact_mode = True
        self.compact_window.show()
        self.hide()
        if self._last_payload:
            self.update_status(self._last_payload)
        self.append_event("Cambiado a modo compacto.")

    def exit_compact_mode(self):
        if not self.is_compact_mode:
            return
        self.is_compact_mode = False
        if self.compact_window:
            self.compact_window.close()
            self.compact_window = None
        self.show()
        self.append_event("Restaurada ventana completa.")

    def _toggle_compact(self):
        if self.is_compact_mode:
            self.exit_compact_mode()
        else:
            self.enter_compact_mode()

    def toggle_presentation_mode(self):
        self.presentation_mode = not self.presentation_mode
        if self.presentation_mode:
            self._pre_presentation_scale = theme.current_font_scale()
            theme.set_font_scale(max(theme.PRESENTATION_FONT_SCALE, self._pre_presentation_scale))
            self.diag_section.set_expanded(False)
            self.log_section.setVisible(False)
            self.diag_section.setVisible(False)
            self.showFullScreen()
            self.append_event("Modo presentación activado (F5 o Esc para salir).")
        else:
            theme.set_font_scale(self._pre_presentation_scale or 1.0)
            self.log_section.setVisible(True)
            self.diag_section.setVisible(True)
            self.showNormal()
            self.append_event("Modo presentación desactivado.")
        self.btn_presentation.setChecked(self.presentation_mode)

    def _exit_special_modes(self):
        if self.presentation_mode:
            self.toggle_presentation_mode()
        elif self.isFullScreen():
            self.showNormal()

    def _toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    def _toggle_contrast(self):
        enabled = theme.toggle_high_contrast(self._persisted_settings())
        self.btn_contrast.setChecked(enabled)

    def _step_font(self, delta: float):
        if self.presentation_mode:
            theme.step_font_scale(delta)
        else:
            theme.step_font_scale(delta, self._persisted_settings())

    def _persisted_settings(self) -> dict | None:
        if self.controller is not None and isinstance(getattr(self.controller, "settings", None), dict):
            return self.controller.settings
        return None

    # --------------------------------------------------------------- acciones
    def _toggle_pause(self):
        if self.controller:
            self.controller.toggle_pause()

    def _recenter(self):
        if self.controller:
            self.controller.recenter()

    def _toggle_cursor(self):
        if self.controller:
            self.controller.toggle_cursor_control()

    def _toggle_camera_metrics(self, checked):
        if self.controller:
            self.controller.set_camera_overlay_details(checked)

    def _test_voice(self):
        if self.controller and hasattr(self.controller, "start_manual_voice_test"):
            self.controller.start_manual_voice_test()

    def _confirm_quit(self):
        answer = QMessageBox.question(
            self,
            "Cerrar GIA",
            "¿Deseas terminar la sesión del asistente?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer == QMessageBox.Yes:
            self._quit()

    def _quit(self):
        if self.controller:
            self.controller.stop()
        if self.compact_window:
            self.compact_window.close()
        self.close()

    def closeEvent(self, event: QCloseEvent):
        if self.controller:
            self.controller.stop()
        if self.compact_window:
            self.compact_window.close()
        event.accept()


def run_qt_runtime(profile: dict, settings: dict):
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        theme.apply_theme(app, settings)

    window = RuntimeMainWindow()
    from app.assistive_controls import AssistiveController
    controller = AssistiveController(profile, settings, main_gui_interface=window)
    window.set_controller(controller)
    window.show()
    controller.start()
    sys.exit(app.exec())

from __future__ import annotations

import sys
import webbrowser
from html import escape
from PySide6.QtCore import Qt, Slot, QPoint, QObject, Signal
from PySide6.QtGui import QPixmap, QFont, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QCheckBox, QGroupBox, QListWidget,
    QFormLayout, QMessageBox, QDialog, QScrollArea, QGridLayout, QProgressBar
)

from app.gesture_catalog import get_enabled_gesture_catalog
from app.qt_helpers import numpy_to_qimage, MODERN_STYLE
from app.voice_router import get_voice_help_rows


class RuntimeSignals(QObject):
    video_update = Signal(object, object)
    status_update = Signal(dict)
    event_update = Signal(str)
    ui_action = Signal(object)


class CompactWindow(QWidget):
    def __init__(self, parent_gui: RuntimeMainWindow):
        super().__init__()
        self.parent_gui = parent_gui
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.drag_position = QPoint()

        self.init_ui()

    def init_ui(self):
        self.resize(980, 230)
        
        # Contenedor principal con estilo y bordes redondeados
        self.container = QWidget(self)
        self.container.setObjectName("compactContainer")
        self.container.setStyleSheet("""
            QWidget#compactContainer {
                background-color: #0f172a;
                border: 2px solid #1e293b;
                border-radius: 16px;
            }
            QLabel {
                color: #f8fafc;
            }
            QTextEdit {
                background-color: #111827;
                border: 1px solid #334155;
                border-radius: 8px;
                color: #f8fafc;
            }
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
        """)

        layout = QHBoxLayout(self.container)

        # Vista de cámara mini
        self.video_frame = QWidget()
        self.video_frame.setStyleSheet("background-color: #020617; border-radius: 12px; min-width: 240px; min-height: 135px; max-width: 240px; max-height: 135px;")
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.setContentsMargins(0, 0, 0, 0)
        self.video_label = QLabel("Feed")
        self.video_label.setFixedSize(240, 135)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #020617; border-radius: 12px;")
        video_layout.addWidget(self.video_label)
        layout.addWidget(self.video_frame)

        # Contenido derecho
        right_layout = QVBoxLayout()
        
        # Fila superior: Estado y botón restaurar
        top_row = QHBoxLayout()
        self.status_label = QLabel("Modo compacto activo")
        self.status_label.setFont(QFont("Segoe UI", 13, QFont.Bold))
        top_row.addWidget(self.status_label)
        
        btn_restore = QPushButton("Volver a ventana completa")
        btn_restore.clicked.connect(self.parent_gui.exit_compact_mode)
        top_row.addWidget(btn_restore)
        right_layout.addLayout(top_row)

        # Información de voz
        self.voice_label = QLabel("Voz: en espera | Texto: - | Comando: -")
        self.voice_label.setStyleSheet("color: #cbd5e1;")
        self.voice_label.setWordWrap(True)
        right_layout.addWidget(self.voice_label)

        self.mic_label = QLabel("Mic: sin datos")
        self.mic_label.setStyleSheet("color: #93c5fd;")
        self.mic_label.setWordWrap(True)
        right_layout.addWidget(self.mic_label)

        # Log de eventos
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setText("Comandos ejecutados:\n")
        self.event_log.setMaximumHeight(80)
        right_layout.addWidget(self.event_log)

        layout.addLayout(right_layout)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.container)

        # Posicionamiento inicial en la parte superior central de la pantalla
        screen = QApplication.primaryScreen().geometry()
        x = (screen.width() - self.width()) // 2
        y = 20
        self.move(x, y)

    # Permitir arrastrar la ventana sin marco
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()


class RuntimeMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.controller = None
        self.settings = {}
        self.is_compact_mode = False
        self.compact_window = None
        self.gesture_title_map = {gesture["id"]: gesture["title"] for gesture in get_enabled_gesture_catalog()}
        
        self.signals = RuntimeSignals()
        self.signals.video_update.connect(self.update_video_feed)
        self.signals.status_update.connect(self.update_status)
        self.signals.event_update.connect(self.append_event)
        self.signals.ui_action.connect(self._run_ui_action)
        
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("GIA v2 - Runtime Asistivo")
        self.resize(1200, 780)
        self.setStyleSheet(MODERN_STYLE)

        # Widget central
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Panel izquierdo: Feed de Video
        left_layout = QVBoxLayout()
        
        self.status_label = QLabel("Listo para iniciar sesión asistiva.")
        self.status_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        left_layout.addWidget(self.status_label)

        self.video_label = QLabel("Esperando señal de cámara...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #1e293b; background-color: #000000; border-radius: 12px; min-width: 640px; min-height: 480px;")
        left_layout.addWidget(self.video_label, 1)

        main_layout.addLayout(left_layout, 2)

        # Panel derecho: Métricas y acciones
        right_layout = QVBoxLayout()

        # Grupo de Métricas
        metrics_box = QGroupBox("Estado e Inferencia")
        metrics_layout = QGridLayout(metrics_box)

        self.lbl_gesture = QLabel("Gesto: -")
        self.lbl_gesture.setFont(QFont("Segoe UI", 12, QFont.Bold))
        metrics_layout.addWidget(self.lbl_gesture, 0, 0)

        self.lbl_confidence = QLabel("Confianza: -")
        metrics_layout.addWidget(self.lbl_confidence, 0, 1)

        self.lbl_face = QLabel("Rostro: -")
        metrics_layout.addWidget(self.lbl_face, 1, 0)

        self.lbl_mode = QLabel("Modo: -")
        metrics_layout.addWidget(self.lbl_mode, 1, 1)

        self.lbl_cursor = QLabel("Cursor: -")
        metrics_layout.addWidget(self.lbl_cursor, 2, 0)

        self.lbl_voice_state = QLabel("Voz: en espera")
        metrics_layout.addWidget(self.lbl_voice_state, 2, 1)

        self.lbl_voice_text = QLabel("Texto: -")
        self.lbl_voice_text.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_voice_text, 3, 0, 1, 2)

        self.lbl_voice_cmd = QLabel("Comando: -")
        self.lbl_voice_cmd.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_voice_cmd, 4, 0, 1, 2)

        self.lbl_mic_status = QLabel("Microfono: sin datos")
        self.lbl_mic_status.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_mic_status, 5, 0, 1, 2)

        self.mic_level_bar = QProgressBar()
        self.mic_level_bar.setRange(0, 100)
        self.mic_level_bar.setValue(0)
        self.mic_level_bar.setFormat("Nivel %p%")
        self.mic_level_bar.setStyleSheet(
            "QProgressBar { background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; text-align: center; color: #e2e8f0; }"
            "QProgressBar::chunk { background-color: #22c55e; border-radius: 5px; }"
        )
        metrics_layout.addWidget(self.mic_level_bar, 6, 0, 1, 2)

        self.lbl_mic_activity = QLabel("Actividad: 0% | Pico: 0%")
        metrics_layout.addWidget(self.lbl_mic_activity, 7, 0, 1, 2)

        self.lbl_rejection = QLabel("Rechazo técnico: -")
        self.lbl_rejection.setStyleSheet("color: #ef4444;")
        self.lbl_rejection.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_rejection, 8, 0, 1, 2)

        self.lbl_model_version = QLabel("Modelo: -")
        self.lbl_model_version.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_model_version, 9, 0, 1, 2)

        right_layout.addWidget(metrics_box)

        # Grupo de Acciones
        actions_box = QGroupBox("Controles Rápidos")
        actions_layout = QGridLayout(actions_box)

        self.btn_pause = QPushButton("Pausar sistema")
        self.btn_pause.setObjectName("accentButton")
        self.btn_pause.clicked.connect(self._toggle_pause)
        actions_layout.addWidget(self.btn_pause, 0, 0)

        btn_recenter = QPushButton("Recentrar")
        btn_recenter.clicked.connect(self._recenter)
        actions_layout.addWidget(btn_recenter, 0, 1)

        self.btn_cursor = QPushButton("Activar cursor")
        self.btn_cursor.setObjectName("dangerButton")
        self.btn_cursor.clicked.connect(self._toggle_cursor)
        actions_layout.addWidget(self.btn_cursor, 1, 0)

        btn_compact = QPushButton("Modo compacto")
        btn_compact.clicked.connect(self.enter_compact_mode)
        actions_layout.addWidget(btn_compact, 1, 1)

        btn_guide = QPushButton("Guia rapida")
        btn_guide.clicked.connect(self.show_guide_dialog)
        actions_layout.addWidget(btn_guide, 2, 0)

        btn_voice_guide = QPushButton("Comandos de voz")
        btn_voice_guide.clicked.connect(self.show_voice_commands_dialog)
        actions_layout.addWidget(btn_voice_guide, 2, 1)

        btn_voice_test = QPushButton("Probar voz")
        btn_voice_test.clicked.connect(self._test_voice)
        actions_layout.addWidget(btn_voice_test, 3, 0)

        btn_close = QPushButton("Cerrar sistema")
        btn_close.setObjectName("dangerButton")
        btn_close.clicked.connect(self._quit)
        actions_layout.addWidget(btn_close, 3, 1)

        self.check_camera_metrics = QCheckBox("Mostrar metricas sobre la camara")
        self.check_camera_metrics.toggled.connect(self._toggle_camera_metrics)
        actions_layout.addWidget(self.check_camera_metrics, 4, 0, 1, 2)

        right_layout.addWidget(actions_box)

        # Consola de Eventos
        self.event_log = QTextEdit()
        self.event_log.setReadOnly(True)
        self.event_log.setText("Eventos recientes:\n")
        self.event_log.setMaximumHeight(150)
        self.event_log.setStyleSheet("background-color: #020617; border: 1px solid #1e293b; color: #10b981;")
        right_layout.addWidget(self.event_log)

        main_layout.addLayout(right_layout, 1)

    def set_controller(self, controller):
        self.controller = controller
        self.settings = dict(getattr(controller, "settings", {}) or {})
        if hasattr(self.controller, "set_camera_overlay_details"):
            self.controller.set_camera_overlay_details(self.check_camera_metrics.isChecked())

    @Slot(object)
    def _run_ui_action(self, callback):
        if callable(callback):
            callback()

    def run_on_ui_thread(self, callback):
        self.signals.ui_action.emit(callback)

    def update_video_feed(self, frame_rgb, compact_frame_rgb=None):
        try:
            if not self.is_compact_mode:
                image = numpy_to_qimage(frame_rgb)
                pix = QPixmap.fromImage(image)
                scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled)

            if compact_frame_rgb is not None and self.compact_window is not None:
                image_c = numpy_to_qimage(compact_frame_rgb)
                pix_c = QPixmap.fromImage(image_c)
                scaled_compact = pix_c.scaled(
                    self.compact_window.video_label.size(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
                self.compact_window.video_label.setText("")
                self.compact_window.video_label.setPixmap(scaled_compact)
        except Exception as exc:
            self.append_event(f"Error en video: {exc}")

    def update_status(self, payload: dict):
        status_text = self._display_text(payload.get("status_text", "Sin estado"))
        self.status_label.setText(status_text)
        
        # Actualizar labels principales
        gesture_id = payload.get("gesture", "-")
        self.lbl_gesture.setText(f"Gesto: {self._format_gesture_label(gesture_id)}")
        self.lbl_confidence.setText(f"Confianza: {payload.get('confidence', '-')}")
        self.lbl_face.setText(f"Rostro: {self._display_text(payload.get('face', '-'))}")
        self.lbl_mode.setText(f"Modo: {self._display_text(payload.get('mode', '-'))}")
        self.lbl_cursor.setText(f"Cursor: {self._display_text(payload.get('cursor', '-'))}")
        self.lbl_voice_state.setText(f"Voz: {self._display_text(payload.get('voice_state', 'en espera'))}")
        self.lbl_voice_text.setText(f"Texto: {self._display_text(payload.get('voice_text', '-'))}")
        self.lbl_voice_cmd.setText(f"Comando: {self._display_text(payload.get('voice_command', '-'))}")
        mic_status = self._display_text(payload.get("mic_monitor_status", "sin datos"))
        mic_device = self._display_text(payload.get("mic_device", "Desconocido"))
        mic_level = int(payload.get("mic_level", 0) or 0)
        mic_peak = int(payload.get("mic_peak", 0) or 0)
        mic_activity = float(payload.get("mic_active_ratio", 0.0) or 0.0)
        self.lbl_mic_status.setText(f"Microfono: {mic_status} | {mic_device}")
        self.mic_level_bar.setValue(max(0, min(mic_level, 100)))
        self._update_mic_bar_style(mic_level)
        self.lbl_mic_activity.setText(f"Actividad: {int(mic_activity * 100)}% | Pico: {mic_peak}%")
        self.lbl_rejection.setText(f"Rechazo tecnico: {self._display_text(payload.get('rejection_reason', '-'))}")
        
        diag = self._display_text(payload.get('diagnostic_hint', '-'))
        self.lbl_model_version.setText(f"Modelo: v{payload.get('model_version', '-')} | Hint: {diag}")

        self._update_stateful_buttons(payload)

        # Actualizar compactos
        if self.compact_window is not None:
            self.compact_window.status_label.setText(status_text)
            self.compact_window.voice_label.setText(
                f"Voz: {self._display_text(payload.get('voice_state', 'en espera'))} | "
                f"Texto: {self._display_text(payload.get('voice_text', '-'))} | "
                f"Comando: {self._display_text(payload.get('voice_command', '-'))}"
            )
            self.compact_window.mic_label.setText(
                f"Mic: {mic_status} | Nivel: {mic_level}% | Pico: {mic_peak}%"
            )

    def _format_gesture_label(self, gesture_id):
        if not gesture_id or gesture_id == "-":
            return "-"
        label = self.gesture_title_map.get(gesture_id, str(gesture_id).replace("_", " "))
        return self._display_text(label)

    def _update_mic_bar_style(self, level: int):
        if level >= 70:
            color = "#ef4444"
        elif level >= 35:
            color = "#f59e0b"
        else:
            color = "#22c55e"
        self.mic_level_bar.setStyleSheet(
            "QProgressBar { background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; text-align: center; color: #e2e8f0; }"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 5px; }}"
        )

    def _update_stateful_buttons(self, payload: dict):
        mode = payload.get("mode", "-")
        cursor = payload.get("cursor", "-")

        if mode == "paused":
            self.btn_pause.setText("Reanudar sistema")
            self.btn_pause.setStyleSheet("background-color: #dc2626; border-color: #ef4444;")
        else:
            self.btn_pause.setText("Pausar sistema")
            self.btn_pause.setStyleSheet("background-color: #2563eb; border-color: #3b82f6;")

        if cursor == "activo":
            self.btn_cursor.setText("Congelar cursor")
            self.btn_cursor.setStyleSheet("background-color: #2563eb; border-color: #3b82f6;")
        else:
            self.btn_cursor.setText("Activar cursor")
            self.btn_cursor.setStyleSheet("background-color: #dc2626; border-color: #ef4444;")

    def append_event(self, text: str):
        # Insertar tanto en el log principal como en el compacto
        self.event_log.append(f"- {text}")
        if self.compact_window is not None:
            self.compact_window.event_log.append(f"- {text}")

    @staticmethod
    def _display_text(value) -> str:
        text = "" if value is None else str(value)
        if any(marker in text for marker in ("\u00c3", "\u00c2", "\u00e2")):
            try:
                text = text.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        return text

    def _add_help_label(self, layout: QVBoxLayout, text: str, object_name: str, rich_text: bool = False):
        label = QLabel(text)
        label.setObjectName(object_name)
        label.setWordWrap(True)
        label.setTextFormat(Qt.RichText if rich_text else Qt.AutoText)
        layout.addWidget(label)
        return label

    def _build_help_dialog(self, title: str, size: tuple[int, int]) -> tuple[QDialog, QVBoxLayout]:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(*size)
        dialog.setObjectName("helpDialog")
        dialog.setStyleSheet(
            MODERN_STYLE
            + """
QDialog#helpDialog {
    background-color: #0f172a;
}
QScrollArea#helpScroll {
    border: 1px solid #1e293b;
    border-radius: 8px;
    background-color: #0f172a;
}
QScrollArea#helpScroll QWidget#qt_scrollarea_viewport {
    background-color: #0f172a;
}
QWidget#helpScrollWidget {
    background-color: #0f172a;
}
QWidget#helpViewport {
    background-color: #0f172a;
}
QLabel#helpTitle {
    color: #f8fafc;
    font-size: 16px;
    font-weight: bold;
}
QLabel#helpSection {
    color: #93c5fd;
    font-size: 14px;
    font-weight: bold;
}
QLabel#helpBody {
    color: #e2e8f0;
    background-color: transparent;
}
"""
        )

        lay = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setObjectName("helpScroll")
        scroll.setWidgetResizable(True)
        scroll.viewport().setObjectName("helpViewport")
        lay.addWidget(scroll)

        scroll_widget = QWidget()
        scroll_widget.setObjectName("helpScrollWidget")
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(18, 18, 18, 18)
        scroll_layout.setSpacing(12)
        scroll.setWidget(scroll_widget)
        return dialog, scroll_layout

    def show_guide_dialog(self):
        dialog, scroll_layout = self._build_help_dialog("Guia de interaccion GIA", (640, 540))

        self._add_help_label(scroll_layout, "Guia rapida de interaccion", "helpTitle")
        self._add_help_label(
            scroll_layout,
            "Este panel resume los gestos activos, la duracion sugerida y las acciones disponibles.",
            "helpBody",
        )
        self._add_help_label(scroll_layout, "Gestos faciales", "helpSection")

        for gesture in get_enabled_gesture_catalog():
            title = escape(self._display_text(gesture["title"]))
            action = escape(self._display_text(gesture["action"]))
            warning = escape(self._display_text(gesture["warning"]))
            help_html = (
                f"<b>{title}</b> &rarr; {action}<br/>"
                f"<span style='color:#94a3b8;'>Duracion: {gesture['duration_ms']} ms | {warning}</span>"
            )
            self._add_help_label(scroll_layout, help_html, "helpBody", rich_text=True)

        self._add_help_label(scroll_layout, "Comandos de voz", "helpSection")
        self._add_help_label(
            scroll_layout,
            "Usa el boton <b>Comandos de voz</b> del panel lateral para ver las frases exactas, "
            "como activarlas y para que sirve cada una.",
            "helpBody",
            rich_text=True,
        )

        for row in get_voice_help_rows(self.settings):
            help_html = (
                f"<b>{escape(self._display_text(row['label']))}</b> &rarr; "
                f"{escape(self._display_text(row['description']))}"
            )
            self._add_help_label(scroll_layout, help_html, "helpBody", rich_text=True)

        self._add_help_label(scroll_layout, "Accesos rapidos web", "helpSection")

        for name, url in [
            ("Abrir Gmail", "https://gmail.com"),
            ("Abrir Facebook", "https://facebook.com"),
            ("Abrir WhatsApp", "https://web.whatsapp.com"),
            ("Abrir YouTube", "https://youtube.com"),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked=False, link=url: webbrowser.open(link))
            scroll_layout.addWidget(btn)

        dialog.exec()

    def show_voice_commands_dialog(self):
        dialog, scroll_layout = self._build_help_dialog("Comandos de voz activos", (720, 620))

        self._add_help_label(scroll_layout, "Como usar los comandos de voz", "helpTitle")
        intro = (
            "1. Manten ambos ojos cerrados de forma deliberada para activar la escucha.<br/>"
            "2. Espera el estado <b>Escuchando</b>.<br/>"
            "3. Di una de las frases activas de forma clara.<br/>"
            "4. Revisa en el panel derecho el texto reconocido y el comando resultante."
        )
        self._add_help_label(scroll_layout, intro, "helpBody", rich_text=True)
        self._add_help_label(
            scroll_layout,
            "<b>Consejo:</b> usa frases cortas y directas. El sistema funciona mejor con "
            "comandos cerrados que con lenguaje muy libre.",
            "helpBody",
            rich_text=True,
        )
        self._add_help_label(
            scroll_layout,
            "Tambien puedes usar el boton <b>Probar voz</b> del panel lateral para depurar "
            "microfono, transcripcion y comando sin depender del gesto de ojos cerrados.",
            "helpBody",
            rich_text=True,
        )
        self._add_help_label(scroll_layout, "Comandos disponibles", "helpSection")

        for row in get_voice_help_rows(self.settings):
            examples = ", ".join(f'"{alias}"' for alias in row.get("spoken_examples", [])[:4]) if row.get("spoken_examples") else "Sin aliases configurados"
            item = (
                f"<b>{escape(self._display_text(row['label']))}</b><br/>"
                f"<span style='color:#94a3b8;'>Que hace:</span> {escape(self._display_text(row['description']))}<br/>"
                f"<span style='color:#94a3b8;'>Frases activas:</span> "
                f"{escape(self._display_text(examples))}"
            )
            self._add_help_label(scroll_layout, item, "helpBody", rich_text=True)

        footer = (
            "<b>Observacion:</b> si una frase no se activa, revisa los campos "
            "<b>Texto</b> y <b>Comando</b> del panel derecho para ver que interpreto el sistema."
        )
        self._add_help_label(scroll_layout, footer, "helpBody", rich_text=True)

        dialog.exec()

    def enter_compact_mode(self):
        if self.is_compact_mode:
            return
        
        self.hide()
        self.compact_window = CompactWindow(self)
        self.compact_window.show()
        self.is_compact_mode = True
        self.append_event("Cambiado a modo compacto.")

    def exit_compact_mode(self):
        if not self.is_compact_mode:
            return
        
        if self.compact_window:
            self.compact_window.close()
            self.compact_window = None
            
        self.show()
        self.is_compact_mode = False
        self.append_event("Restaurada ventana completa.")

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
        
    window = RuntimeMainWindow()
    from app.assistive_controls import AssistiveController
    controller = AssistiveController(profile, settings, main_gui_interface=window)
    window.set_controller(controller)
    window.show()
    controller.start()
    sys.exit(app.exec())

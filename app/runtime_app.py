from __future__ import annotations

import sys
import webbrowser
from PySide6.QtCore import Qt, Slot, QPoint, QObject, Signal
from PySide6.QtGui import QImage, QPixmap, QFont, QCloseEvent, QPainter, QBrush, QPen
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QCheckBox, QGroupBox, QListWidget,
    QFormLayout, QMessageBox, QDialog, QScrollArea, QGridLayout
)

from app.gesture_catalog import GESTURE_CATALOG, VOICE_COMMAND_HELP
from app.qt_helpers import numpy_to_qimage, MODERN_STYLE


class RuntimeSignals(QObject):
    video_update = Signal(object, object)
    status_update = Signal(dict)
    event_update = Signal(str)


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
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border-radius: 12px;")
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
        self.is_compact_mode = False
        self.compact_window = None
        self.gesture_title_map = {gesture["id"]: gesture["title"] for gesture in GESTURE_CATALOG}
        
        self.signals = RuntimeSignals()
        self.signals.video_update.connect(self.update_video_feed)
        self.signals.status_update.connect(self.update_status)
        self.signals.event_update.connect(self.append_event)
        
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

        self.lbl_rejection = QLabel("Rechazo técnico: -")
        self.lbl_rejection.setStyleSheet("color: #ef4444;")
        self.lbl_rejection.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_rejection, 5, 0, 1, 2)

        self.lbl_model_version = QLabel("Modelo: -")
        self.lbl_model_version.setWordWrap(True)
        metrics_layout.addWidget(self.lbl_model_version, 6, 0, 1, 2)

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

        btn_guide = QPushButton("Guía rápida")
        btn_guide.clicked.connect(self.show_guide_dialog)
        actions_layout.addWidget(btn_guide, 2, 0)

        btn_close = QPushButton("Cerrar sistema")
        btn_close.setObjectName("dangerButton")
        btn_close.clicked.connect(self._quit)
        actions_layout.addWidget(btn_close, 2, 1)

        self.check_camera_metrics = QCheckBox("Mostrar métricas sobre la cámara")
        self.check_camera_metrics.toggled.connect(self._toggle_camera_metrics)
        actions_layout.addWidget(self.check_camera_metrics, 3, 0, 1, 2)

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
        if hasattr(self.controller, "set_camera_overlay_details"):
            self.controller.set_camera_overlay_details(self.check_camera_metrics.isChecked())

    def update_video_feed(self, frame_rgb, compact_frame_rgb=None):
        try:
            if not self.is_compact_mode:
                image = numpy_to_qimage(frame_rgb)
                pix = QPixmap.fromImage(image)
                scaled = pix.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.video_label.setPixmap(scaled)

            if compact_frame_rgb is not None and self.compact_window is not None:
                # Dibujar un feed de cámara recortado circular o bordes redondeados
                image_c = numpy_to_qimage(compact_frame_rgb)
                pix_c = QPixmap.fromImage(image_c)
                
                # Crear imagen de destino
                target = QPixmap(240, 135)
                target.fill(Qt.transparent)
                
                painter = QPainter(target)
                painter.setRenderHint(QPainter.Antialiasing)
                # Crear máscara redondeada
                path = QBrush(pix_c.scaled(240, 135, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
                painter.setBrush(path)
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(0, 0, 240, 135, 12, 12)
                painter.end()
                
                self.compact_window.video_label.setPixmap(target)
        except Exception as exc:
            self.append_event(f"Error en video: {exc}")

    def update_status(self, payload: dict):
        status_text = payload.get("status_text", "Sin estado")
        self.status_label.setText(status_text)
        
        # Actualizar labels principales
        gesture_id = payload.get("gesture", "-")
        self.lbl_gesture.setText(f"Gesto: {self._format_gesture_label(gesture_id)}")
        self.lbl_confidence.setText(f"Confianza: {payload.get('confidence', '-')}")
        self.lbl_face.setText(f"Rostro: {payload.get('face', '-')}")
        self.lbl_mode.setText(f"Modo: {payload.get('mode', '-')}")
        self.lbl_cursor.setText(f"Cursor: {payload.get('cursor', '-')}")
        self.lbl_voice_state.setText(f"Voz: {payload.get('voice_state', 'en espera')}")
        self.lbl_voice_text.setText(f"Texto: {payload.get('voice_text', '-')}")
        self.lbl_voice_cmd.setText(f"Comando: {payload.get('voice_command', '-')}")
        self.lbl_rejection.setText(f"Rechazo técnico: {payload.get('rejection_reason', '-')}")
        
        diag = payload.get('diagnostic_hint', '-')
        self.lbl_model_version.setText(f"Modelo: v{payload.get('model_version', '-')} | Hint: {diag}")

        self._update_stateful_buttons(payload)

        # Actualizar compactos
        if self.compact_window is not None:
            self.compact_window.status_label.setText(status_text)
            self.compact_window.voice_label.setText(
                f"Voz: {payload.get('voice_state', 'en espera')} | "
                f"Texto: {payload.get('voice_text', '-')} | "
                f"Comando: {payload.get('voice_command', '-')}"
            )

    def _format_gesture_label(self, gesture_id):
        if not gesture_id or gesture_id == "-":
            return "-"
        return self.gesture_title_map.get(gesture_id, str(gesture_id).replace("_", " "))

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

    def show_guide_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Guía de interacción GIA")
        dialog.resize(600, 500)
        dialog.setStyleSheet(MODERN_STYLE)

        lay = QVBoxLayout(dialog)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        lay.addWidget(scroll)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll.setWidget(scroll_widget)

        # Gestos
        title_g = QLabel("Gestos Faciales")
        title_g.setFont(QFont("Segoe UI", 12, QFont.Bold))
        scroll_layout.addWidget(title_g)

        for gesture in GESTURE_CATALOG:
            item = QLabel(f"<b>{gesture['title']}</b> &rarr; {gesture['action']}<br/>"
                          f"<font color='#94a3b8'>{gesture['duration_ms']}ms | {gesture['warning']}</font>")
            item.setWordWrap(True)
            scroll_layout.addWidget(item)

        # Comando de voz
        title_v = QLabel("<br/>Comandos de Voz")
        title_v.setFont(QFont("Segoe UI", 12, QFont.Bold))
        scroll_layout.addWidget(title_v)

        for command, desc in VOICE_COMMAND_HELP:
            item = QLabel(f"<b>\"{command}\"</b> &rarr; {desc}")
            item.setWordWrap(True)
            scroll_layout.addWidget(item)

        # Enlaces
        title_l = QLabel("<br/>Accesos Rápidos Web")
        title_l.setFont(QFont("Segoe UI", 12, QFont.Bold))
        scroll_layout.addWidget(title_l)

        for name, url in [
            ("Abrir Gmail", "https://gmail.com"),
            ("Abrir Facebook", "https://facebook.com"),
            ("Abrir WhatsApp", "https://web.whatsapp.com"),
            ("Abrir YouTube", "https://youtube.com")
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked=False, link=url: webbrowser.open(link))
            scroll_layout.addWidget(btn)

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

"""Ventana única de GIA: coordina las pantallas y garantiza un solo dueño de la cámara."""
from __future__ import annotations

import time

from PySide6.QtCore import QByteArray, Qt, QThread, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QStackedWidget, QVBoxLayout, QWidget

from app.app_runtime import ERROR_LOG_PATH, log_exception
from app.config_store import list_profiles, load_profile, save_settings
from app.home_page import HomePage
from app.voice_command_config_dialog import VoiceCommandConfigDialog

ERROR_NOTICE_MIN_INTERVAL_S = 15.0
SESSION_CLOSE_TIMEOUT_MS = 20000


class SessionCloser(QThread):
    """Libera la sesión fuera del hilo de la interfaz: espera a la cámara y exporta el registro."""

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller

    def run(self):
        self.controller.stop()


class GiaMainWindow(QMainWindow):
    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self.trainer = None
        self.assistant = None
        self.controller = None
        self._closer: SessionCloser | None = None
        self._starting_session = False
        self._closing = False
        self._last_error_notice = 0.0

        self.setWindowTitle("GIA")
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)

        self.home = HomePage(settings)
        self.home.start_requested.connect(self.start_assistant)
        self.home.train_requested.connect(self.open_trainer)
        self.home.update_requested.connect(lambda name: self.open_trainer(name, update_model=True))
        self.home.voice_commands_requested.connect(self.open_voice_commands)
        self.home.quit_requested.connect(self.close)
        self.stack.addWidget(self.home)
        self._restore_geometry()

    # --------------------------------------------------------------- arranque
    def begin(self, *, autostart: bool = True, train_profile: str | None = None, open_trainer: bool = False):
        """Lo primero que hace GIA al abrirse, ya con la ventana visible."""
        if open_trainer:
            names = list_profiles()
            name = train_profile or self.settings.get("last_profile")
            if name in names:
                self.open_trainer(name)
            return
        last_profile = self.settings.get("last_profile")
        if (
            autostart
            and self.settings.get("autostart_last_profile", True)
            and last_profile
            and self.home.selected_name() == last_profile
            and self.home.check_ready(last_profile)
        ):
            self.home.begin_autostart(last_profile)

    # -------------------------------------------------------------- navegación
    def _session_active(self) -> bool:
        return self._starting_session or self.controller is not None or self._closer is not None

    def _ensure_trainer(self):
        if self.trainer is None:
            from app.trainer_app import TrainerPage

            self.trainer = TrainerPage(self.settings)
            self.trainer.home_requested.connect(self.show_home)
            self.trainer.assistant_requested.connect(self.start_assistant)
            self.stack.addWidget(self.trainer)
        return self.trainer

    def _leave_trainer(self) -> bool:
        """Suelta la cámara del Entrenador. False si está entrenando y no se puede salir aún."""
        if self.trainer is None:
            return True
        if not self.trainer.can_leave():
            QMessageBox.information(self, "Entrenamiento en curso", "Espera a que termine el entrenamiento; tarda unos segundos.")
            return False
        self.trainer.release_camera()
        return True

    def show_home(self):
        if not self._leave_trainer():
            return
        self.home.refresh()
        self.stack.setCurrentWidget(self.home)
        self.setWindowTitle("GIA")

    def open_trainer(self, name: str, *, update_model: bool = False):
        self.home.cancel_autostart()
        if self._session_active():
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            trainer = self._ensure_trainer()
            trainer.open_profile(name)
            self.stack.setCurrentWidget(trainer)
            self.setWindowTitle(f"GIA · Calibrar y entrenar · {name}")
        except Exception as exc:  # noqa: BLE001 - un perfil dañado no debe cerrar la aplicación
            self._report_failure(f"No se pudo abrir el perfil «{name}»", exc)
            return
        finally:
            QApplication.restoreOverrideCursor()
        if update_model:
            trainer.go_to_training()
            trainer.run_model_upgrade()

    def open_voice_commands(self):
        self.home.cancel_autostart()
        dialog = VoiceCommandConfigDialog(self.settings, self)
        if dialog.exec():
            # Los ajustes se comparten por referencia entre pantallas: se actualizan en el sitio.
            updated = dialog.get_updated_settings()
            self.settings.clear()
            self.settings.update(updated)
            save_settings(self.settings)
            QMessageBox.information(self, "Configuración guardada", "Los comandos globales de voz se actualizaron.")

    # ------------------------------------------------------------------ sesión
    def start_assistant(self, name: str):
        self.home.cancel_autostart()
        if self._session_active() or not self._leave_trainer():
            return
        self._starting_session = True
        self.stack.setCurrentWidget(self.home)
        self.home.refresh(name)
        self.home.set_busy("Preparando el asistente…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        page = None
        controller = None
        try:
            profile = load_profile(name)
            from app.assistive_controls import AssistiveController
            from app.runtime_app import AssistantPage

            page = AssistantPage()
            controller = AssistiveController(profile, self.settings, main_gui_interface=page)
            page.set_controller(controller)
            page.session_end_requested.connect(self.end_session)
            self.stack.addWidget(page)
            self.stack.setCurrentWidget(page)
            controller.start()
        except Exception as exc:  # noqa: BLE001 - p. ej. un modelo dañado: se vuelve al inicio con el motivo
            if controller is not None:
                controller.stop()
            if page is not None:
                self.stack.removeWidget(page)
                page.deleteLater()
            self.stack.setCurrentWidget(self.home)
            self._report_failure(f"No se pudo iniciar el asistente con «{name}»", exc)
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.home.set_busy("")
            self._starting_session = False

        self.assistant = page
        self.controller = controller
        self.settings["last_profile"] = name
        save_settings(self.settings)
        self.setWindowTitle(f"GIA · Asistente · {name}")
        if self.settings.get("preflight_on_start", True):
            page.show_preflight_dialog()

    def end_session(self):
        """Vuelve al inicio de inmediato y libera la sesión en segundo plano."""
        if self.controller is None or self._closer is not None:
            return
        self.assistant.prepare_to_close()
        self.home.refresh()
        self.home.set_busy("Guardando la sesión…")
        self.stack.setCurrentWidget(self.home)
        self.setWindowTitle("GIA")
        self._closer = SessionCloser(self.controller, self)
        self._closer.finished.connect(self._on_session_closed)
        self._closer.start()

    def _on_session_closed(self):
        summary = getattr(self.controller, "_usage_summary_on_stop", None)
        self.stack.removeWidget(self.assistant)
        self.assistant.deleteLater()
        self.assistant = None
        self.controller = None
        self._closer.deleteLater()
        self._closer = None
        self.home.set_busy("")
        self.home.refresh()
        if summary and not self._closing:
            QMessageBox.information(
                self,
                "Grabación de uso",
                f"Se guardaron {summary['seconds'] / 60:.1f} minutos de la sesión en:\n{summary['path']}",
            )

    # ------------------------------------------------------------------ errores
    def _report_failure(self, title: str, exc: Exception):
        log_exception(type(exc), exc, exc.__traceback__)
        QMessageBox.warning(self, "GIA", f"{title}.\n\n{exc}")

    def show_unexpected_error(self, text: str):
        """Aviso de un error no capturado: GIA sigue abierta, sin repetir mensajes en cadena."""
        now = time.monotonic()
        if now - self._last_error_notice < ERROR_NOTICE_MIN_INTERVAL_S or self._closing:
            return
        self._last_error_notice = now
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("GIA · Algo salió mal")
        box.setText("Ocurrió un error inesperado. GIA sigue abierta, pero si algo deja de responder, ciérrala y vuelve a abrirla.")
        box.setInformativeText(f"{text}\n\nDetalles guardados en:\n{ERROR_LOG_PATH}")
        box.setModal(False)
        box.show()

    # ------------------------------------------------------------------ cierre
    def _restore_geometry(self):
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1180, int(screen.width() * 0.94)), min(760, int(screen.height() * 0.9)))
        stored = self.settings.get("window_geometry")
        if stored:
            self.restoreGeometry(QByteArray.fromHex(str(stored).encode("ascii")))

    def _save_geometry(self):
        if not self.isFullScreen():
            self.settings["window_geometry"] = bytes(self.saveGeometry().toHex()).decode("ascii")
            save_settings(self.settings)

    def closeEvent(self, event: QCloseEvent):
        self._closing = True
        self.home.shutdown()
        if self.assistant is not None:
            self.assistant.prepare_to_close()
        self._save_geometry()
        self.hide()
        if self.trainer is not None:
            self.trainer.shutdown()
        if self._closer is not None:
            self._closer.wait(SESSION_CLOSE_TIMEOUT_MS)
        elif self.controller is not None:
            self.controller.stop()
        event.accept()
        # El modo compacto oculta esta ventana: la salida de la aplicación se decide aquí, no por Qt.
        QTimer.singleShot(0, QApplication.quit)

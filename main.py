import os
import sys
import traceback

import absl.logging
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from app import ui_theme as theme
from app.config_store import ensure_app_dirs, get_profile_summary, list_profiles, load_profile, load_settings, save_settings
from app.runtime_app import run_qt_runtime
from app.ui_widgets import Button, make_label, populate_profile_list
from app.voice_command_config_dialog import VoiceCommandConfigDialog


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)


class ProfileSelector(QWidget):
    """Pantalla de inicio: elegir perfil y lanzar el asistente."""

    def __init__(self):
        super().__init__()
        ensure_app_dirs()
        self.settings = load_settings()
        self.selected_profile_data = None
        self.trainer_window = None
        self.init_ui()

    def init_ui(self):
        self.setObjectName("appRoot")
        self.setWindowTitle("GIA · Inicio")
        self.resize(560, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 22)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header.setSpacing(14)
        logo = make_label("GIA", role="logo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(56, 56)
        header.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        titles.addWidget(make_label("GIA · Asistente de movilidad", role="title"))
        titles.addWidget(make_label("Control del computador con gestos faciales y voz, sin conexión.", role="muted", wrap=True))
        header.addLayout(titles, 1)
        layout.addLayout(header)

        layout.addSpacing(6)
        layout.addWidget(make_label("ELIGE TU PERFIL", role="cardTitle"))
        self.cb_profiles = QListWidget()
        self.cb_profiles.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.cb_profiles.itemDoubleClicked.connect(lambda _item: self.on_start_clicked())
        layout.addWidget(self.cb_profiles, 1)

        btn_start = Button("Iniciar asistente", "play", variant="primary", size="large")
        btn_start.setDefault(True)
        btn_start.clicked.connect(self.on_start_clicked)
        layout.addWidget(btn_start)

        secondary = QHBoxLayout()
        secondary.setSpacing(8)
        btn_trainer = Button("Entrenador", "chip", tooltip="Crear perfiles, calibrar y entrenar gestos")
        btn_trainer.clicked.connect(self.open_trainer)
        btn_voice_config = Button("Comandos de voz", "mic", tooltip="Configurar frases y macros globales")
        btn_voice_config.clicked.connect(self.open_voice_command_settings)
        self.btn_contrast = Button("", "contrast", variant="ghost", tooltip="Alto contraste", checkable=True)
        self.btn_contrast.setChecked(theme.current_theme() == "high_contrast")
        self.btn_contrast.setFixedWidth(44)
        self.btn_contrast.clicked.connect(self._toggle_contrast)
        btn_quit = Button("Salir", variant="ghost")
        btn_quit.clicked.connect(self.close)
        secondary.addWidget(btn_trainer)
        secondary.addWidget(btn_voice_config)
        secondary.addStretch(1)
        secondary.addWidget(self.btn_contrast)
        secondary.addWidget(btn_quit)
        layout.addLayout(secondary)

        self.refresh_profiles()

    def refresh_profiles(self):
        summaries = [get_profile_summary(name) for name in list_profiles()]
        # Primero los perfiles calibrados, luego alfabético.
        summaries.sort(key=lambda summary: (not summary["calibrated"], summary["name"].lower()))
        populate_profile_list(self.cb_profiles, summaries, self.settings.get("last_profile"))

    def _selected_name(self) -> str:
        item = self.cb_profiles.currentItem()
        return item.data(Qt.UserRole) if item is not None else ""

    def on_start_clicked(self):
        name = self._selected_name()
        if not name:
            QMessageBox.warning(self, "Sin perfiles", "No hay perfiles creados. Abre el Entrenador para crear y calibrar uno.")
            return

        profile = load_profile(name)
        if not profile.get("calibration", {}).get("completed", False):
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Perfil sin calibrar")
            box.setText(f"El perfil «{name}» todavía no tiene un modelo de gestos entrenado.")
            box.setInformativeText("Abre el Entrenador para capturar tus gestos y entrenar el modelo.")
            open_button = box.addButton("Abrir entrenador", QMessageBox.AcceptRole)
            box.addButton("Cancelar", QMessageBox.RejectRole)
            box.exec()
            if box.clickedButton() is open_button:
                self.open_trainer()
            return

        self.settings["last_profile"] = name
        save_settings(self.settings)
        self.selected_profile_data = profile
        if self.trainer_window is not None:
            self.trainer_window.close()
        self.close()

    def open_trainer(self):
        from app.trainer_app import TrainerMainWindow

        if self.trainer_window is None:
            self.trainer_window = TrainerMainWindow()
            self.trainer_window.closed.connect(self._on_trainer_closed)
        self.trainer_window.show()
        self.trainer_window.raise_()
        self.trainer_window.activateWindow()

    def _on_trainer_closed(self):
        self.trainer_window = None
        self.settings = load_settings()
        self.refresh_profiles()

    def open_voice_command_settings(self):
        dialog = VoiceCommandConfigDialog(self.settings, self)
        if dialog.exec():
            self.settings = dialog.get_updated_settings()
            save_settings(self.settings)
            QMessageBox.information(self, "Configuración guardada", "Los comandos globales de voz se actualizaron.")

    def _toggle_contrast(self):
        enabled = theme.toggle_high_contrast(self.settings)
        self.btn_contrast.setChecked(enabled)


def main():
    app = QApplication(sys.argv)
    theme.apply_theme(app, load_settings())

    selector = ProfileSelector()
    selector.show()
    app.exec()

    if selector.selected_profile_data is not None:
        # Lanzar el runtime asistivo principal con el perfil cargado
        run_qt_runtime(selector.selected_profile_data, selector.settings)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MAIN: Error no capturado en main: {exc}")
        traceback.print_exc()
    finally:
        print("MAIN: Programa terminado.")

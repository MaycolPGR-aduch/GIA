import os
import sys
import traceback
import absl.logging
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox
)

from app.config_store import ensure_app_dirs, load_settings, list_profiles, load_profile
from app.qt_helpers import MODERN_STYLE
from app.runtime_app import run_qt_runtime


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)


class ProfileSelector(QWidget):
    def __init__(self):
        super().__init__()
        ensure_app_dirs()
        self.settings = load_settings()
        self.selected_profile_data = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("GIA v2 - Inicio de Sesión")
        self.resize(400, 220)
        self.setStyleSheet(MODERN_STYLE)

        layout = QVBoxLayout(self)

        lbl_title = QLabel("GIA v2 - Asistente de Movilidad")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #3b82f6;")
        lbl_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(lbl_title)

        layout.addWidget(QLabel("Selecciona tu perfil de usuario para iniciar:"))

        self.cb_profiles = QComboBox()
        layout.addWidget(self.cb_profiles)

        btn_start = QPushButton("Iniciar Asistente")
        btn_start.setObjectName("accentButton")
        btn_start.clicked.connect(self.on_start_clicked)
        layout.addWidget(btn_start)

        btn_quit = QPushButton("Cerrar")
        btn_quit.clicked.connect(self.close)
        layout.addWidget(btn_quit)

        self.refresh_profiles()

    def refresh_profiles(self):
        self.cb_profiles.clear()
        profiles = list_profiles()
        self.cb_profiles.addItems(profiles)

    def on_start_clicked(self):
        name = self.cb_profiles.currentText()
        if not name:
            QMessageBox.warning(self, "Error", "No hay perfiles creados. Ejecuta 'main_trainer.py' para crear y calibrar un perfil.")
            return

        profile = load_profile(name)
        calibration = profile.get("calibration", {})
        
        if not calibration.get("completed", False):
            QMessageBox.warning(
                self, "Perfil no calibrado",
                f"El perfil '{name}' no tiene calibración completa ni modelo entrenado.\n\n"
                "Por favor, ejecuta 'main_trainer.py' primero para calibrar tus gestos."
            )
            return

        self.selected_profile_data = profile
        self.close()


def main():
    app = QApplication(sys.argv)
    
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

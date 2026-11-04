import argparse
import os
import sys

import absl.logging
from PySide6.QtWidgets import QApplication, QMessageBox

from app import ui_theme as theme
from app.app_runtime import ErrorReporter, acquire_single_instance_lock
from app.config_store import load_settings
from app.main_window import GiaMainWindow


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GIA: control del computador con gestos faciales y voz.")
    parser.add_argument("--train", nargs="?", const="", default=None, metavar="PERFIL", help="abre Calibrar y entrenar")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    app = QApplication(sys.argv[:1])
    # La salida la decide la ventana principal: el modo compacto la oculta sin terminar GIA.
    app.setQuitOnLastWindowClosed(False)

    lock = acquire_single_instance_lock()
    if lock is None:
        theme.apply_theme(app, load_settings())
        QMessageBox.information(None, "GIA", "GIA ya está abierta. Búscala en la barra de tareas.")
        return 0

    reporter = ErrorReporter()
    reporter.install()
    settings = load_settings()
    theme.apply_theme(app, settings)

    window = GiaMainWindow(settings)
    reporter.error_occurred.connect(window.show_unexpected_error)
    window.show()
    if arguments.train is not None:
        window.open_trainer_at_launch(arguments.train or None)
    code = app.exec()
    lock.unlock()
    return code


if __name__ == "__main__":
    sys.exit(main())

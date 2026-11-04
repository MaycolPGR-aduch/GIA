"""Servicios de proceso: una sola instancia de GIA y registro de errores inesperados."""
from __future__ import annotations

import faulthandler
import sys
import threading
import traceback
from datetime import datetime

from PySide6.QtCore import QLockFile, QObject, Signal

from .config_store import DATA_DIR, LOG_DIR

ERROR_LOG_PATH = LOG_DIR / "errores_gia.log"
LOCK_PATH = DATA_DIR / ".gia.lock"


def acquire_single_instance_lock() -> QLockFile | None:
    """Devuelve el candado si esta es la única GIA abierta; None si ya hay otra.

    Hay que conservar el objeto mientras dure el programa. Si otra GIA se cerró de golpe, Qt
    detecta que su proceso ya no existe y recupera el candado.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    lock = QLockFile(str(LOCK_PATH))
    lock.setStaleLockTime(0)
    return lock if lock.tryLock(100) else None


def log_exception(exc_type, exc_value, exc_traceback) -> None:
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    print(text, file=sys.stderr)
    try:
        with open(ERROR_LOG_PATH, "a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}]\n{text}")
    except OSError:
        pass


class ErrorReporter(QObject):
    """Registra los errores no capturados y avisa a la interfaz en su propio hilo."""

    error_occurred = Signal(str)

    def __init__(self):
        super().__init__()
        self._fault_file = None

    def install(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        sys.excepthook = self._on_exception
        threading.excepthook = self._on_thread_exception
        # Los fallos nativos (cámara, MediaPipe, Qt) no pasan por Python: quedan en el mismo archivo.
        self._fault_file = open(ERROR_LOG_PATH, "a", encoding="utf-8")
        faulthandler.enable(self._fault_file)

    def _on_exception(self, exc_type, exc_value, exc_traceback) -> None:
        self._report(exc_type, exc_value, exc_traceback)

    def _on_thread_exception(self, args) -> None:
        if args.exc_type is SystemExit:
            return
        self._report(args.exc_type, args.exc_value, args.exc_traceback)

    def _report(self, exc_type, exc_value, exc_traceback) -> None:
        log_exception(exc_type, exc_value, exc_traceback)
        self.error_occurred.emit(f"{exc_type.__name__}: {exc_value}")

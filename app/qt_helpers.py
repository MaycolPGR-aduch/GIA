from __future__ import annotations

import threading
import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from app.landmark_provider import LandmarkProvider
from app.ui_theme import build_stylesheet


class CameraThread(QThread):
    # Emite (frame_rgb, face_sample)
    frame_processed = Signal(object, object)
    camera_error = Signal(str)

    # La cámara puede seguir ocupada un momento si otra pantalla acaba de soltarla.
    OPEN_ATTEMPTS = 4
    OPEN_RETRY_S = 1.5
    STOP_TIMEOUT_MS = 10000

    def __init__(self, camera_index: int = 0, fps: int = 15, width: int = 640, height: int = 480):
        super().__init__()
        self.camera_index = camera_index
        self.fps = fps
        self.width = width
        self.height = height
        self._stop_requested = threading.Event()

    @property
    def running(self) -> bool:
        return self.isRunning() and not self._stop_requested.is_set()

    def _open_camera(self):
        for attempt in range(self.OPEN_ATTEMPTS):
            if self._stop_requested.is_set():
                return None
            cap = cv2.VideoCapture(self.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if cap.isOpened():
                return cap
            cap.release()
            if attempt + 1 < self.OPEN_ATTEMPTS:
                self._stop_requested.wait(self.OPEN_RETRY_S)
        if not self._stop_requested.is_set():
            self.camera_error.emit("No se pudo abrir la cámara. Revisa que esté conectada y que ninguna otra aplicación la use.")
        return None

    def run(self):
        provider = None
        cap = None
        try:
            provider = LandmarkProvider(fps=self.fps)
            cap = self._open_camera()
            if cap is None:
                return

            frame_delay = 1.0 / max(self.fps, 1)

            while not self._stop_requested.is_set():
                start_time = time.monotonic()
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue

                snapshot = provider.process(frame_bgr)
                self.frame_processed.emit(snapshot.frame_rgb, snapshot.face_sample)

                elapsed = time.monotonic() - start_time
                if elapsed < frame_delay:
                    time.sleep(frame_delay - elapsed)
        except Exception as exc:
            self.camera_error.emit(str(exc))
        finally:
            if cap and cap.isOpened():
                cap.release()
            if provider:
                provider.close()

    def stop(self):
        """Detiene la captura. Se puede llamar en cualquier momento, incluso mientras la cámara abre."""
        self._stop_requested.set()
        if not self.wait(self.STOP_TIMEOUT_MS):
            print("CameraThread: la cámara no respondió al detenerla.")


def numpy_to_qimage(frame_rgb: np.ndarray) -> QImage:
    height, width, channel = frame_rgb.shape
    bytes_per_line = channel * width
    return QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)


# Compatibilidad: el estilo vive ahora en app/ui_theme.py y se aplica a nivel de
# QApplication con apply_theme(). Este alias se mantiene para importaciones antiguas.
MODERN_STYLE = build_stylesheet("dark", 1.0)

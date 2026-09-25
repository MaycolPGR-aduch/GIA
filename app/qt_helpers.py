from __future__ import annotations

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

    def __init__(self, camera_index: int = 0, fps: int = 15, width: int = 640, height: int = 480):
        super().__init__()
        self.camera_index = camera_index
        self.fps = fps
        self.width = width
        self.height = height
        self.running = False

    def run(self):
        provider = None
        cap = None
        try:
            provider = LandmarkProvider(fps=self.fps)
            cap = cv2.VideoCapture(self.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            if not cap.isOpened():
                self.camera_error.emit("No se pudo abrir la cámara.")
                return

            self.running = True
            frame_delay = 1.0 / max(self.fps, 1)

            while self.running:
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
        self.running = False
        self.wait()


def numpy_to_qimage(frame_rgb: np.ndarray) -> QImage:
    height, width, channel = frame_rgb.shape
    bytes_per_line = channel * width
    return QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)


# Compatibilidad: el estilo vive ahora en app/ui_theme.py y se aplica a nivel de
# QApplication con apply_theme(). Este alias se mantiene para importaciones antiguas.
MODERN_STYLE = build_stylesheet("dark", 1.0)

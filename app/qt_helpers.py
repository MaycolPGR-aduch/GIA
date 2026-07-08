from __future__ import annotations

import time
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage, QPixmap
from app.landmark_provider import LandmarkProvider


class CameraThread(QThread):
    # Emite (frame_bgr, face_sample)
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
                self.frame_processed.emit(snapshot.frame_bgr, snapshot.face_sample)

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


# Estilos CSS de Qt (QSS) para lograr una estética moderna y premium
MODERN_STYLE = """
QMainWindow {
    background-color: #0f172a; /* Slate 900 */
}

QWidget {
    color: #f8fafc; /* Slate 50 */
    font-family: 'Segoe UI', 'Outfit', 'Inter', sans-serif;
    font-size: 13px;
}

QTabWidget::pane {
    border: 1px solid #1e293b;
    background-color: #0f172a;
    border-radius: 8px;
}

QTabBar::tab {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    margin-right: 4px;
}

QTabBar::tab:selected {
    background-color: #2563eb; /* Royal Blue */
    color: #ffffff;
    border-color: #2563eb;
}

QPushButton {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
}

QPushButton:pressed {
    background-color: #0f172a;
}

QPushButton#accentButton {
    background-color: #2563eb;
    border-color: #3b82f6;
    color: #ffffff;
}

QPushButton#accentButton:hover {
    background-color: #1d4ed8;
}

QPushButton#dangerButton {
    background-color: #dc2626;
    border-color: #ef4444;
    color: #ffffff;
}

QPushButton#dangerButton:hover {
    background-color: #b91c1c;
}

QProgressBar {
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    background-color: #1e293b;
}

QProgressBar::chunk {
    background-color: #10b981; /* Emerald 500 */
    border-radius: 3px;
}

QComboBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 12px;
}

QComboBox::drop-down {
    border: none;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    border: 1px solid #334155;
    selection-background-color: #2563eb;
}

QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 12px;
}

QTextEdit {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 8px 10px;
    color: #f8fafc;
    selection-background-color: #2563eb;
}

QTextEdit[readOnly="true"] {
    background-color: #0f172a;
}

QGroupBox {
    border: 1px solid #1e293b;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
    color: #3b82f6;
}

QListWidget {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 6px;
}

QListWidget::item {
    padding: 8px;
    border-radius: 4px;
}

QListWidget::item:hover {
    background-color: #334155;
}

QListWidget::item:selected {
    background-color: #2563eb;
    color: #ffffff;
}

QScrollBar:vertical {
    border: none;
    background: #0f172a;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    border: none;
    background: none;
}
"""

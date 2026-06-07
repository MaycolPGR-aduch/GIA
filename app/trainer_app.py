from __future__ import annotations

import sys
import time
import numpy as np
import cv2
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QComboBox, QPushButton, QLineEdit,
    QGroupBox, QProgressBar, QListWidget, QListWidgetItem,
    QFormLayout, QMessageBox, QDoubleSpinBox, QCheckBox
)

from app.config_store import (
    ensure_app_dirs, load_settings, list_profiles, load_profile,
    save_profile, build_default_profile, delete_profile, save_settings
)
from app.audio_utils import list_input_devices
from app.calibration_service import CalibrationService
from app.training_service import TrainingService
from app.gesture_catalog import NEUTRAL_GESTURE_META, get_enabled_gesture_catalog
from app.qt_helpers import CameraThread, numpy_to_qimage, MODERN_STYLE


class TrainerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_app_dirs()
        self.settings = load_settings()
        self.window_size = int(self.settings.get("gesture_window_size", 12))
        
        self.current_profile_name = ""
        self.profile = {}
        self.calibration_service = None
        self.training_service = None
        
        self.camera_thread = None
        self.capture_gesture_id = None
        self.countdown_val = 0
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.on_countdown_tick)
        self.is_capturing = False

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("GIA v2 - Entrenador y Calibrador Asistivo")
        self.resize(1000, 700)
        self.setStyleSheet(MODERN_STYLE)

        # Tab Widget Principal
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Crear las diferentes pestañas
        self.create_profiles_tab()
        self.create_diagnostics_tab()
        self.create_guide_tab()
        self.create_calibration_tab()
        self.create_training_tab()

        self.tabs.setTabEnabled(1, False)
        self.tabs.setTabEnabled(3, False)
        self.tabs.setTabEnabled(4, False)

        # Cargar lista de perfiles inicial
        self.refresh_profile_list()
        self.refresh_audio_device_selector()

    def create_profiles_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Panel izquierdo: Cargar perfil existente
        left_box = QGroupBox("Seleccionar Perfil Existente")
        left_layout = QVBoxLayout(left_box)

        self.profile_list = QComboBox()
        left_layout.addWidget(self.profile_list)

        btn_load = QPushButton("Cargar Perfil")
        btn_load.setObjectName("accentButton")
        btn_load.clicked.connect(self.load_selected_profile)
        left_layout.addWidget(btn_load)

        btn_delete = QPushButton("Eliminar Perfil")
        btn_delete.setObjectName("dangerButton")
        btn_delete.clicked.connect(self.delete_selected_profile)
        left_layout.addWidget(btn_delete)

        left_layout.addStretch()
        layout.addWidget(left_box)

        # Panel derecho: Crear nuevo perfil
        right_box = QGroupBox("Crear Nuevo Perfil")
        right_layout = QFormLayout(right_box)

        self.txt_new_name = QLineEdit()
        self.txt_new_name.setPlaceholderText("Ej. Juan")
        right_layout.addRow("Nombre del Perfil:", self.txt_new_name)

        self.check_voice = QCheckBox("Habilitar comandos de voz")
        self.check_voice.setChecked(True)
        right_layout.addRow(self.check_voice)

        self.check_tts = QCheckBox("Habilitar retroalimentación por voz (TTS)")
        self.check_tts.setChecked(False)
        right_layout.addRow(self.check_tts)

        btn_create = QPushButton("Crear Perfil")
        btn_create.setObjectName("accentButton")
        btn_create.clicked.connect(self.create_new_profile)
        right_layout.addRow(btn_create)

        layout.addWidget(right_box)
        self.tabs.addTab(tab, "1. Perfiles")

    def create_diagnostics_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Control de cámara y feed
        left_layout = QVBoxLayout()
        self.diag_cam_label = QLabel("La cámara está inactiva.")
        self.diag_cam_label.setAlignment(Qt.AlignCenter)
        self.diag_cam_label.setStyleSheet("border: 2px solid #1e293b; background-color: #020617; border-radius: 8px; min-width: 480px; min-height: 360px;")
        left_layout.addWidget(self.diag_cam_label)

        self.btn_toggle_diag_cam = QPushButton("Iniciar Prueba de Cámara")
        self.btn_toggle_diag_cam.clicked.connect(self.toggle_diag_camera)
        left_layout.addWidget(self.btn_toggle_diag_cam)
        layout.addLayout(left_layout)

        # Ajustes rápidos
        right_box = QGroupBox("Ajustes Rápidos del Perfil")
        right_layout = QFormLayout(right_box)

        self.spin_sensitivity = QDoubleSpinBox()
        self.spin_sensitivity.setRange(0.5, 5.0)
        self.spin_sensitivity.setSingleStep(0.1)
        right_layout.addRow("Sensibilidad del Cursor:", self.spin_sensitivity)

        self.spin_dead_zone = QDoubleSpinBox()
        self.spin_dead_zone.setRange(2.0, 30.0)
        self.spin_dead_zone.setSingleStep(1.0)
        right_layout.addRow("Zona Muerta (px):", self.spin_dead_zone)

        self.spin_smoothing = QDoubleSpinBox()
        self.spin_smoothing.setRange(0.1, 0.95)
        self.spin_smoothing.setSingleStep(0.05)
        right_layout.addRow("Suavizado:", self.spin_smoothing)

        self.check_mirror = QCheckBox("Vista Espejo de Cámara")
        right_layout.addRow(self.check_mirror)

        self.check_invert_x = QCheckBox("Invertir Eje X")
        right_layout.addRow(self.check_invert_x)

        self.cb_audio_input = QComboBox()
        right_layout.addRow("Microfono de voz:", self.cb_audio_input)

        btn_refresh_audio = QPushButton("Actualizar microfonos")
        btn_refresh_audio.clicked.connect(self.refresh_audio_device_selector)
        right_layout.addRow(btn_refresh_audio)

        btn_save_settings = QPushButton("Guardar Ajustes")
        btn_save_settings.setObjectName("accentButton")
        btn_save_settings.clicked.connect(self.save_quick_settings)
        right_layout.addRow(btn_save_settings)

        layout.addWidget(right_box)
        self.tabs.addTab(tab, "2. Diagnóstico")

    def create_guide_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        guide_list = QListWidget()
        # Agregar neutral
        item_text = f"Postura neutral - {NEUTRAL_GESTURE_META['training_hint']}\nAcción: {NEUTRAL_GESTURE_META['action']}"
        guide_list.addItem(QListWidgetItem(item_text))

        # Agregar gestos
        for gesture in get_enabled_gesture_catalog():
            item_text = f"{gesture['title']} - {gesture['training_hint']}\nAcción: {gesture['action']} | Error común: {gesture['common_failure_hint']}"
            guide_list.addItem(QListWidgetItem(item_text))

        layout.addWidget(QLabel("Guía rápida de gestos disponibles en el sistema:"))
        layout.addWidget(guide_list)
        self.tabs.addTab(tab, "3. Guía de Gestos")

    def create_calibration_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Panel izquierdo: Feed de Cámara
        left_layout = QVBoxLayout()
        self.calib_cam_label = QLabel("Inicia la cámara para capturar muestras.")
        self.calib_cam_label.setAlignment(Qt.AlignCenter)
        self.calib_cam_label.setStyleSheet("border: 2px solid #1e293b; background-color: #020617; border-radius: 8px; min-width: 480px; min-height: 360px;")
        left_layout.addWidget(self.calib_cam_label)

        self.lbl_calib_status = QLabel("Selecciona un gesto y haz clic en Capturar.")
        self.lbl_calib_status.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.lbl_calib_status.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(self.lbl_calib_status)

        self.calib_progress = QProgressBar()
        left_layout.addWidget(self.calib_progress)

        self.btn_capture = QPushButton("Iniciar Captura (3s Countdown)")
        self.btn_capture.setObjectName("accentButton")
        self.btn_capture.clicked.connect(self.on_capture_click)
        left_layout.addWidget(self.btn_capture)

        self.btn_reset_gesture = QPushButton("Limpiar Muestras del Gesto Seleccionado")
        self.btn_reset_gesture.setObjectName("dangerButton")
        self.btn_reset_gesture.clicked.connect(self.reset_gesture_samples)
        left_layout.addWidget(self.btn_reset_gesture)

        layout.addLayout(left_layout)

        # Panel derecho: Lista de gestos a calibrar
        right_box = QGroupBox("Gestos del Perfil")
        right_layout = QVBoxLayout(right_box)

        self.gesture_list = QListWidget()
        self.gesture_list.currentItemChanged.connect(self.on_gesture_selected_changed)
        right_layout.addWidget(self.gesture_list)

        self.lbl_gesture_desc = QLabel("Selecciona un gesto para ver sus detalles.")
        self.lbl_gesture_desc.setWordWrap(True)
        right_layout.addWidget(self.lbl_gesture_desc)

        layout.addWidget(right_box)
        self.tabs.addTab(tab, "4. Calibración")

    def create_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.lbl_train_title = QLabel("Entrenamiento del Clasificador de Gestos")
        self.lbl_train_title.setFont(QFont("Segoe UI", 14, QFont.Bold))
        layout.addWidget(self.lbl_train_title)

        self.txt_train_status = QLabel("Listo para entrenar una vez completada la calibración.")
        self.txt_train_status.setWordWrap(True)
        layout.addWidget(self.txt_train_status)

        self.btn_train = QPushButton("Entrenar Modelo Híbrido (GRU + Regresión Logística)")
        self.btn_train.setObjectName("accentButton")
        self.btn_train.clicked.connect(self.run_training)
        layout.addWidget(self.btn_train)

        self.txt_train_results = QLabel("")
        self.txt_train_results.setFont(QFont("Courier New", 10))
        self.txt_train_results.setStyleSheet("background-color: #1e293b; border-radius: 6px; padding: 12px;")
        self.txt_train_results.setWordWrap(True)
        layout.addWidget(self.txt_train_results)

        layout.addStretch()
        self.tabs.addTab(tab, "5. Entrenamiento")

    def refresh_profile_list(self):
        self.profile_list.clear()
        profiles = list_profiles()
        self.profile_list.addItems(profiles)

    def load_selected_profile(self):
        name = self.profile_list.currentText()
        if not name:
            return
        
        self.current_profile_name = name
        self.profile = load_profile(name)
        self.calibration_service = CalibrationService(self.window_size)
        self.training_service = TrainingService(name, self.window_size)

        # Cargar dataset si existe
        self.training_service.load_classifier()
        pending = self.training_service.load_pending_samples()
        if pending is not None:
            samples, quality = pending
            self.calibration_service.load_samples(samples, quality)
        else:
            samples = self.training_service.load_active_dataset()
            self.calibration_service.load_samples(samples, self.profile.get("calibration", {}).get("capture_quality"))

        # Actualizar inputs de ajustes rápidos
        self.spin_sensitivity.setValue(self.profile.get("cursor_sensitivity", 1.35))
        self.spin_dead_zone.setValue(self.profile.get("dead_zone_px", 8.0))
        self.spin_smoothing.setValue(self.profile.get("smoothing_factor", 0.62))
        self.check_mirror.setChecked(self.profile.get("mirror_preview", True))
        self.check_invert_x.setChecked(self.profile.get("invert_x", True))
        self.refresh_audio_device_selector()

        # Habilitar pestañas
        self.tabs.setTabEnabled(1, True)
        self.tabs.setTabEnabled(3, True)
        self.tabs.setTabEnabled(4, True)

        self.refresh_gestures_list_widget()
        self.tabs.setCurrentIndex(1)
        QMessageBox.information(self, "Perfil Cargado", f"Perfil '{name}' cargado con éxito.")

    def delete_selected_profile(self):
        name = self.profile_list.currentText()
        if not name:
            return
        
        confirm = QMessageBox.question(
            self, "Confirmar eliminación",
            f"¿Estás seguro de que deseas eliminar permanentemente el perfil '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            delete_profile(name)
            self.refresh_profile_list()
            QMessageBox.information(self, "Eliminado", f"Perfil '{name}' eliminado.")

    def create_new_profile(self):
        name = self.txt_new_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Introduce un nombre válido.")
            return
        
        # Crear perfil
        profile = build_default_profile(name)
        profile["voice_enabled"] = self.check_voice.isChecked()
        profile["tts_enabled"] = self.check_tts.isChecked()
        save_profile(profile)
        
        self.txt_new_name.clear()
        self.refresh_profile_list()
        
        # Cargar directamente
        index = self.profile_list.findText(name)
        if index >= 0:
            self.profile_list.setCurrentIndex(index)
        self.load_selected_profile()

    def save_quick_settings(self):
        if not self.profile:
            return
        self.profile["cursor_sensitivity"] = self.spin_sensitivity.value()
        self.profile["dead_zone_px"] = self.spin_dead_zone.value()
        self.profile["smoothing_factor"] = self.spin_smoothing.value()
        self.profile["mirror_preview"] = self.check_mirror.isChecked()
        self.profile["invert_x"] = self.check_invert_x.isChecked()

        save_profile(self.profile)
        self.settings["audio_input_device"] = self.cb_audio_input.currentData()
        save_settings(self.settings)
        QMessageBox.information(self, "Ajustes Guardados", "Los ajustes del perfil y del audio se actualizaron.")

    def refresh_audio_device_selector(self):
        selected_device = self.settings.get("audio_input_device")
        self.cb_audio_input.blockSignals(True)
        self.cb_audio_input.clear()
        self.cb_audio_input.addItem("Predeterminado del sistema", None)
        devices = list_input_devices()
        selected_index = 0
        for position, device in enumerate(devices, start=1):
            self.cb_audio_input.addItem(device["label"], device["id"])
            if selected_device == device["id"]:
                selected_index = position
        if selected_device not in (None, "") and selected_index == 0:
            self.cb_audio_input.addItem(f"Dispositivo no disponible ({selected_device})", selected_device)
            selected_index = self.cb_audio_input.count() - 1
        self.cb_audio_input.setCurrentIndex(selected_index)
        self.cb_audio_input.blockSignals(False)

    # Control de Cámara para Diagnóstico
    def toggle_diag_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_diag_camera()
        else:
            self.start_diag_camera()

    def start_diag_camera(self):
        camera_idx = int(self.settings.get("camera_index", 0))
        fps = int(self.settings.get("fps", 15))
        self.camera_thread = CameraThread(camera_index=camera_idx, fps=fps)
        self.camera_thread.frame_processed.connect(self.on_diag_frame)
        self.camera_thread.camera_error.connect(self.on_camera_error)
        self.camera_thread.start()
        self.btn_toggle_diag_cam.setText("Detener Prueba de Cámara")

    def stop_diag_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        self.diag_cam_label.clear()
        self.diag_cam_label.setText("La cámara está inactiva.")
        self.btn_toggle_diag_cam.setText("Iniciar Prueba de Cámara")

    @Slot(object, object)
    def on_diag_frame(self, frame_rgb, face_sample):
        # Dibujar landmarks básicos de diagnóstico
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if face_sample is not None:
            for pt in face_sample.points_px.values():
                cv2.circle(frame_bgr, pt, 1, (60, 255, 80), -1)
            cv2.circle(frame_bgr, face_sample.nose_px, 4, (255, 0, 255), -1)
            
            # Dibujar el rectángulo de cara
            sc = int(face_sample.face_scale_px)
            fc = face_sample.face_center_px
            cv2.rectangle(frame_bgr, (fc[0] - sc//2, fc[1] - sc//2), (fc[0] + sc//2, fc[1] + sc//2), (255, 180, 0), 1)

        if self.profile.get("mirror_preview", True):
            frame_bgr = cv2.flip(frame_bgr, 1)

        qimg = numpy_to_qimage(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        self.diag_cam_label.setPixmap(QPixmap.fromImage(qimg).scaled(480, 360, Qt.KeepAspectRatio))

    # Control de Cámara para Calibración
    def refresh_gestures_list_widget(self):
        self.gesture_list.clear()
        if not self.calibration_service:
            return
        
        statuses = self.calibration_service.refresh_statuses()
        for idx_id in self.calibration_service.gesture_order:
            status = statuses[idx_id]
            title = status.title
            readiness = status.readiness.upper()
            frames = f"{status.valid_frames} frames"
            item = QListWidgetItem(f"{title} ({readiness}) - {frames}")
            item.setData(Qt.UserRole, idx_id)
            self.gesture_list.addItem(item)

    def on_gesture_selected_changed(self, current, previous):
        if not current:
            self.lbl_gesture_desc.setText("")
            self.btn_capture.setEnabled(False)
            return
        
        gesture_id = current.data(Qt.UserRole)
        self.capture_gesture_id = gesture_id
        meta = self.calibration_service.gesture_meta[gesture_id]
        
        desc = (
            f"<b>Instrucción</b>: {meta['training_hint']}<br/>"
            f"<b>Fallas comunes</b>: {meta.get('common_failure_hint', 'Ninguna')}<br/><br/>"
            f"<i>Recomendado: {meta['recommended_min_frames']} frames mínimos.</i>"
        )
        self.lbl_gesture_desc.setText(desc)
        self.btn_capture.setEnabled(True)

        # Actualizar progreso
        status = self.calibration_service.refresh_statuses()[gesture_id]
        self.calib_progress.setMaximum(max(meta["recommended_min_frames"], status.valid_frames))
        self.calib_progress.setValue(status.valid_frames)

    def on_capture_click(self):
        if self.is_capturing:
            self.stop_capture_and_camera()
        else:
            self.start_countdown()

    def start_countdown(self):
        if not self.capture_gesture_id:
            return
        
        self.btn_capture.setEnabled(False)
        self.gesture_list.setEnabled(False)
        
        # Iniciar cámara si no está encendida
        camera_idx = int(self.settings.get("camera_index", 0))
        fps = int(self.settings.get("fps", 15))
        
        if self.camera_thread:
            self.camera_thread.stop()
            
        self.camera_thread = CameraThread(camera_index=camera_idx, fps=fps)
        self.camera_thread.frame_processed.connect(self.on_calibration_frame)
        self.camera_thread.camera_error.connect(self.on_camera_error)
        self.camera_thread.start()
        
        self.countdown_val = 3
        self.lbl_calib_status.setText(f"Prepárate: {self.countdown_val}...")
        self.countdown_timer.start(1000)

    def on_countdown_tick(self):
        self.countdown_val -= 1
        if self.countdown_val > 0:
            self.lbl_calib_status.setText(f"Prepárate: {self.countdown_val}...")
        else:
            self.countdown_timer.stop()
            self.lbl_calib_status.setText("¡MANTÉN EL GESTO AHORA!")
            # Iniciar grabación conectando la captura de muestras
            self.camera_thread.frame_processed.disconnect(self.on_calibration_frame)
            self.camera_thread.frame_processed.connect(self.on_capture_sample_frame)
            self.is_capturing = True
            self.btn_capture.setText("Detener Captura")
            self.btn_capture.setObjectName("dangerButton")
            self.btn_capture.style().unpolish(self.btn_capture)
            self.btn_capture.style().polish(self.btn_capture)
            self.btn_capture.update()
            self.btn_capture.setEnabled(True)

    @Slot(object, object)
    def on_calibration_frame(self, frame_rgb, face_sample):
        # Renderiza el feed con espejo
        frame_render = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if self.profile.get("mirror_preview", True):
            frame_render = cv2.flip(frame_render, 1)
        qimg = numpy_to_qimage(cv2.cvtColor(frame_render, cv2.COLOR_BGR2RGB))
        self.calib_cam_label.setPixmap(QPixmap.fromImage(qimg).scaled(480, 360, Qt.KeepAspectRatio))

    @Slot(object, object)
    def on_capture_sample_frame(self, frame_rgb, face_sample):
        # Registrar muestra
        is_registered, feedback = self.calibration_service.register_sample(self.capture_gesture_id, face_sample)
        
        # Renderizar en UI
        frame_render = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Mostrar retroalimentación de MediaPipe
        if face_sample is not None:
            for pt in face_sample.points_px.values():
                cv2.circle(frame_render, pt, 1, (0, 255, 0), -1)
        
        if self.profile.get("mirror_preview", True):
            frame_render = cv2.flip(frame_render, 1)
            
        qimg = numpy_to_qimage(cv2.cvtColor(frame_render, cv2.COLOR_BGR2RGB))
        self.calib_cam_label.setPixmap(QPixmap.fromImage(qimg).scaled(480, 360, Qt.KeepAspectRatio))

        # Actualizar progreso
        status = self.calibration_service.refresh_statuses()[self.capture_gesture_id]
        meta = self.calibration_service.gesture_meta[self.capture_gesture_id]
        
        self.calib_progress.setMaximum(max(meta["recommended_min_frames"], status.valid_frames))
        self.calib_progress.setValue(status.valid_frames)
        self.lbl_calib_status.setText(
            f"Muestras: {status.valid_frames} (mínimo recomendado: {meta['recommended_min_frames']})\n"
            f"{feedback}\nHaz clic en 'Detener Captura' cuando termines."
        )

    def stop_capture_and_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        
        self.is_capturing = False
        self.btn_capture.setText("Iniciar Captura (3s Countdown)")
        self.btn_capture.setObjectName("accentButton")
        self.btn_capture.style().unpolish(self.btn_capture)
        self.btn_capture.style().polish(self.btn_capture)
        self.btn_capture.update()

        self.lbl_calib_status.setText("Captura finalizada.")
        self.btn_capture.setEnabled(True)
        self.gesture_list.setEnabled(True)
        self.refresh_gestures_list_widget()
        
        # Guardar persistencia temporal de muestras registradas
        if self.calibration_service:
            samples = self.calibration_service.export_samples()
            quality = self.calibration_service.export_capture_quality()
            self.training_service.save_pending_samples(samples, quality)
        
        # Recargar el item seleccionado
        items = self.gesture_list.findItems(self.capture_gesture_id, Qt.MatchContains)
        # Buscar por data
        for i in range(self.gesture_list.count()):
            item = self.gesture_list.item(i)
            if item.data(Qt.UserRole) == self.capture_gesture_id:
                self.gesture_list.setCurrentItem(item)
                break

    def reset_gesture_samples(self):
        if not self.capture_gesture_id or not self.calibration_service:
            return
        
        confirm = QMessageBox.question(
            self, "Confirmar limpieza",
            f"¿Estás seguro de que deseas limpiar todas las muestras del gesto '{self.capture_gesture_id}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if confirm == QMessageBox.Yes:
            self.calibration_service.reset_gesture(self.capture_gesture_id)
            self.refresh_gestures_list_widget()
            meta = self.calibration_service.gesture_meta[self.capture_gesture_id]
            self.calib_progress.setMaximum(meta["recommended_min_frames"])
            self.calib_progress.setValue(0)
            self.lbl_calib_status.setText("Muestras eliminadas.")
            
            # Guardar el cambio tras resetear
            if self.calibration_service:
                samples = self.calibration_service.export_samples()
                quality = self.calibration_service.export_capture_quality()
                self.training_service.save_pending_samples(samples, quality)

    def on_camera_error(self, err_msg):
        self.stop_capture_and_camera()
        QMessageBox.critical(self, "Error de Cámara", f"Ocurrió un error en el dispositivo de vídeo: {err_msg}")

    # Entrenamiento del Adaptador Local
    def run_training(self):
        if not self.calibration_service:
            return
        
        can_train, blockers, statuses = self.calibration_service.can_train()
        if not can_train:
            blockers_str = ", ".join(blockers)
            QMessageBox.warning(
                self, "No se puede entrenar",
                f"Aún faltan gestos por calibrar de forma robusta:\n{blockers_str}"
            )
            return

        self.btn_train.setEnabled(False)
        self.txt_train_status.setText("Entrenando clasificador GRU local...")
        QApplication.processEvents()

        try:
            samples = self.calibration_service.export_samples()
            quality = self.calibration_service.export_capture_quality()
            
            classifier = self.training_service.train(samples, capture_quality_summary=quality)
            self.calibration_service.mark_trained()
            self.training_service.clear_pending_samples()
            
            # Guardar estado de calibración en el perfil
            self.profile = self.training_service.build_profile_training_state(
                self.profile, samples, quality, self.calibration_service.refresh_statuses()
            )
            save_profile(self.profile)

            # Mostrar resultados de entrenamiento
            summary = classifier.training_summary
            report = summary.get("class_metrics", {})
            
            results_text = (
                f"<b>Modelo entrenado con éxito (Versión {summary['version']})</b><br/>"
                f"Fecha: {summary['trained_at']}<br/>"
                f"Muestras totales: {summary['training_windows']} ventanas útiles.<br/>"
                f"Precisión de validación: <b>{summary['validation_accuracy']:.2%}</b><br/><br/>"
                f"<b>Métricas por clase:</b><br/>"
            )
            for gesture_id in summary["classes"]:
                class_info = report.get(gesture_id, {})
                f1 = class_info.get("f1-score", 0.0)
                rec = class_info.get("recall", 0.0)
                results_text += f"- <b>{gesture_id}</b>: F1={f1:.2f} | Recall={rec:.2f}<br/>"
                
            self.txt_train_results.setText(results_text)
            self.txt_train_status.setText("Modelo listo para interactuar en runtime.")
            self.refresh_gestures_list_widget()
            
            QMessageBox.information(
                self, "Entrenamiento Completado",
                f"El modelo de gestos se entrenó con una precisión de {summary['validation_accuracy']:.1%}."
            )
        except Exception as exc:
            self.txt_train_status.setText(f"Error en entrenamiento: {exc}")
            QMessageBox.critical(self, "Error", f"Ocurrió un error al entrenar el modelo: {exc}")
        finally:
            self.btn_train.setEnabled(True)

    def closeEvent(self, event):
        # Apagar cámaras y salir
        if self.camera_thread:
            self.camera_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = TrainerMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

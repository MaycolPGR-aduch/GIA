from __future__ import annotations

import sys

import cv2
from PySide6.QtCore import QSize, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import ui_theme as theme
from app.audio_utils import list_input_devices
from app.calibration_service import CalibrationService
from app.config_store import (
    build_default_profile,
    delete_profile,
    ensure_app_dirs,
    get_profile_summary,
    list_profiles,
    load_profile,
    load_settings,
    save_profile,
    save_settings,
)
from app.gesture_catalog import GESTURE_CATALOG, NEUTRAL_GESTURE_META, get_enabled_gesture_catalog
from app.qt_helpers import CameraThread
from app.training_service import TrainingService
from app.ui_widgets import (
    Button,
    Card,
    ConfusionMatrixView,
    StatCard,
    StatusBadge,
    VideoView,
    format_accuracy,
    icon,
    make_label,
    populate_profile_list,
    set_label_text,
)


STEPS = [
    ("Perfiles", "user", "Crea un perfil nuevo o carga uno existente."),
    ("Diagnóstico", "camera", "Comprueba la cámara y el micrófono, y ajusta el cursor."),
    ("Guía de gestos", "book", "Conoce cada gesto antes de calibrarlo."),
    ("Calibración", "record", "Captura muestras de cada gesto con la cámara."),
    ("Entrenamiento", "chip", "Entrena el modelo de gestos de este perfil."),
]
STEP_PROFILES, STEP_DIAGNOSTICS, STEP_GUIDE, STEP_CALIBRATION, STEP_TRAINING = range(5)

READINESS = {
    "sin iniciar": ("Falta", "neutral"),
    "insuficiente": ("Parcial", "warning"),
    "listo para entrenar": ("Listo", "success"),
    "entrenado": ("Entrenado", "accent"),
}

GESTURE_SHORT_TITLES = {gesture["id"]: gesture.get("short_title", gesture["title"]) for gesture in GESTURE_CATALOG}
GESTURE_SHORT_TITLES["neutral"] = NEUTRAL_GESTURE_META.get("short_title", "Neutral")


class TrainingWorker(QThread):
    """Entrena el modelo fuera del hilo de la interfaz."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, training_service: TrainingService, samples: dict, quality: dict):
        super().__init__()
        self.training_service = training_service
        self.samples = samples
        self.quality = quality

    def run(self):
        try:
            classifier = self.training_service.train(self.samples, capture_quality_summary=self.quality)
            self.succeeded.emit(classifier)
        except Exception as exc:  # noqa: BLE001 - se muestra al usuario
            self.failed.emit(str(exc))


class GestureProgressItem(QWidget):
    """Fila de la lista de calibración: nombre, estado y progreso."""

    def __init__(self, status, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        top = QHBoxLayout()
        self.title = make_label(GESTURE_SHORT_TITLES.get(status.gesture_id, status.title), role="h2")
        self.badge = StatusBadge()
        top.addWidget(self.title, 1)
        top.addWidget(self.badge)
        layout.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        layout.addWidget(self.progress)
        self.meta = make_label("", role="subtle")
        layout.addWidget(self.meta)
        self.set_status(status)

    def set_status(self, status):
        text, tone = READINESS.get(status.readiness, (status.readiness.capitalize(), "neutral"))
        self.badge.set_status(text, tone)
        self.progress.setMaximum(max(status.recommended_min_frames, status.valid_frames, 1))
        self.progress.setValue(status.valid_frames)
        theme.set_widget_property(self.progress, "tone", "success" if tone in ("success", "accent") else "warning")
        self.meta.setText(f"{status.valid_frames} / {status.recommended_min_frames} fotogramas · {status.action}")


class TrainerMainWindow(QMainWindow):
    closed = Signal()

    def __init__(self):
        super().__init__()
        ensure_app_dirs()
        self.settings = load_settings()
        self.window_size = int(self.settings.get("gesture_window_size", 12))

        self.current_profile_name = ""
        self.profile = {}
        self.calibration_service = None
        self.training_service = None
        self.training_worker = None

        self.camera_thread = None
        self.capture_gesture_id = None
        self.countdown_val = 0
        self.countdown_timer = QTimer(self)
        self.countdown_timer.timeout.connect(self.on_countdown_tick)
        self.is_capturing = False
        self._last_face_present = None

        self.init_ui()

    # ------------------------------------------------------------------ UI
    def init_ui(self):
        self.setWindowTitle("GIA · Entrenador y calibrador")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1180, int(screen.width() * 0.94)), min(740, int(screen.height() * 0.9)))

        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        main = QHBoxLayout(root)
        main.setContentsMargins(14, 14, 14, 14)
        main.setSpacing(14)

        main.addWidget(self._build_sidebar())

        content = QVBoxLayout()
        content.setSpacing(12)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.lbl_step_title = make_label("", role="title")
        self.lbl_step_desc = make_label("", role="muted")
        titles.addWidget(self.lbl_step_title)
        titles.addWidget(self.lbl_step_desc)
        header.addLayout(titles, 1)
        self.profile_badge = StatusBadge("Sin perfil cargado", "neutral", large=True)
        header.addWidget(self.profile_badge, 0, Qt.AlignVCenter)
        content.addLayout(header)

        self.stack = QStackedWidget()
        content.addWidget(self.stack, 1)
        main.addLayout(content, 1)

        # Crear las diferentes páginas (mismo orden que STEPS)
        self.create_profiles_tab()
        self.create_diagnostics_tab()
        self.create_guide_tab()
        self.create_calibration_tab()
        self.create_training_tab()

        self.statusBar().showMessage("Carga o crea un perfil para empezar.")

        self._set_step_enabled(STEP_DIAGNOSTICS, False)
        self._set_step_enabled(STEP_CALIBRATION, False)
        self._set_step_enabled(STEP_TRAINING, False)
        self.step_list.setCurrentRow(STEP_PROFILES)
        theme.theme_state().changed.connect(self._refresh_step_icons)

        # Cargar lista de perfiles inicial
        self.refresh_profile_list()
        self.refresh_audio_device_selector()

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setProperty("role", "card")
        sidebar.setFixedWidth(236)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(10)

        brand = QHBoxLayout()
        brand.setSpacing(10)
        brand.addWidget(make_label("GIA", role="logo"))
        brand_titles = QVBoxLayout()
        brand_titles.setSpacing(0)
        brand_titles.addWidget(make_label("Entrenador", role="h2"))
        brand_titles.addWidget(make_label("Calibración de gestos", role="subtle"))
        brand.addLayout(brand_titles, 1)
        layout.addLayout(brand)
        layout.addSpacing(6)

        self.step_list = QListWidget()
        self.step_list.setObjectName("stepNav")
        self.step_list.setIconSize(QSize(20, 20))
        self.step_list.setFocusPolicy(Qt.StrongFocus)
        for index, (title, _icon_name, _desc) in enumerate(STEPS):
            item = QListWidgetItem(f"{index + 1}. {title}")
            item.setSizeHint(QSize(0, 44))
            self.step_list.addItem(item)
        self.step_list.currentRowChanged.connect(self._on_step_changed)
        layout.addWidget(self.step_list, 1)

        self.btn_contrast = Button("Alto contraste", "contrast", variant="ghost", checkable=True)
        self.btn_contrast.setChecked(theme.current_theme() == "high_contrast")
        self.btn_contrast.clicked.connect(self._toggle_contrast)
        layout.addWidget(self.btn_contrast)
        return sidebar

    # ------------------------------------------------------------ navegación
    def _set_step_enabled(self, index: int, enabled: bool):
        item = self.step_list.item(index)
        flags = item.flags()
        item.setFlags(flags | Qt.ItemIsEnabled | Qt.ItemIsSelectable if enabled else flags & ~Qt.ItemIsEnabled & ~Qt.ItemIsSelectable)
        self._refresh_step_icons()

    def _step_done(self, index: int) -> bool:
        if index == STEP_PROFILES:
            return bool(self.current_profile_name)
        if index == STEP_CALIBRATION and self.calibration_service is not None:
            return self.calibration_service.can_train()[0]
        if index == STEP_TRAINING:
            return bool(self.profile.get("calibration", {}).get("completed"))
        return False

    def _refresh_step_icons(self):
        palette = theme.palette()
        for index, (_title, icon_name, _desc) in enumerate(STEPS):
            item = self.step_list.item(index)
            if not item.flags() & Qt.ItemIsEnabled:
                item.setIcon(icon("lock", palette["text_subtle"]))
                item.setToolTip("Carga un perfil para habilitar este paso.")
            elif self._step_done(index):
                item.setIcon(icon("check", palette["success"]))
                item.setToolTip("Completado")
            else:
                item.setIcon(icon(icon_name, palette["text_muted"]))
                item.setToolTip("")

    def _go_to_step(self, index: int):
        self.step_list.setCurrentRow(index)

    def _on_step_changed(self, index: int):
        if index < 0:
            return
        previous = self.stack.currentIndex()
        if previous == STEP_DIAGNOSTICS and index != STEP_DIAGNOSTICS:
            self.stop_diag_camera()
        if previous == STEP_CALIBRATION and index != STEP_CALIBRATION and (self.is_capturing or self.countdown_timer.isActive()):
            self.countdown_timer.stop()
            self.stop_capture_and_camera()
        self.stack.setCurrentIndex(index)
        title, _icon_name, desc = STEPS[index]
        self.lbl_step_title.setText(title)
        self.lbl_step_desc.setText(desc)

    def _toggle_contrast(self):
        enabled = theme.toggle_high_contrast(self.settings)
        self.btn_contrast.setChecked(enabled)

    # --------------------------------------------------------------- páginas
    @staticmethod
    def _page() -> tuple[QWidget, QHBoxLayout]:
        page = QWidget()
        layout = QHBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        return page, layout

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(widget)
        return scroll

    def create_profiles_tab(self):
        page, layout = self._page()

        existing = Card("Perfiles existentes", "user")
        self.profile_list = QListWidget()
        self.profile_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.profile_list.itemDoubleClicked.connect(lambda _item: self.load_selected_profile())
        existing.body.addWidget(self.profile_list, 1)
        row = QHBoxLayout()
        btn_load = Button("Cargar perfil", "check", variant="primary")
        btn_load.clicked.connect(self.load_selected_profile)
        btn_delete = Button("Eliminar", "trash", variant="danger")
        btn_delete.clicked.connect(self.delete_selected_profile)
        row.addWidget(btn_load, 1)
        row.addWidget(btn_delete)
        existing.body.addLayout(row)
        layout.addWidget(existing, 3)

        create = Card("Crear perfil nuevo", "plus")
        create.body.addWidget(make_label("Nombre del perfil", role="h2"))
        self.txt_new_name = QLineEdit()
        self.txt_new_name.setPlaceholderText("Ej. Juan")
        self.txt_new_name.returnPressed.connect(self.create_new_profile)
        create.body.addWidget(self.txt_new_name)
        create.body.addSpacing(6)

        self.check_voice = QCheckBox("Habilitar comandos de voz")
        self.check_voice.setChecked(True)
        create.body.addWidget(self.check_voice)
        create.body.addWidget(make_label("Permite dar órdenes habladas cerrando ambos ojos.", role="subtle", wrap=True))

        self.check_tts = QCheckBox("Habilitar respuesta hablada (TTS)")
        self.check_tts.setChecked(False)
        create.body.addWidget(self.check_tts)
        create.body.addWidget(make_label("El sistema confirma en voz alta cada acción.", role="subtle", wrap=True))
        create.body.addStretch(1)

        btn_create = Button("Crear y cargar perfil", "plus", variant="primary")
        btn_create.clicked.connect(self.create_new_profile)
        create.body.addWidget(btn_create)
        layout.addWidget(create, 2)
        self.stack.addWidget(page)

    def create_diagnostics_tab(self):
        page, layout = self._page()

        left = QVBoxLayout()
        left.setSpacing(10)
        self.diag_view = VideoView(placeholder="La cámara está inactiva", min_size=(420, 260))
        left.addWidget(self.diag_view, 1)
        controls = QHBoxLayout()
        self.face_badge = StatusBadge("Cámara apagada", "neutral", large=True)
        controls.addWidget(self.face_badge)
        controls.addStretch(1)
        self.btn_toggle_diag_cam = Button("Iniciar prueba de cámara", "camera", variant="primary")
        self.btn_toggle_diag_cam.clicked.connect(self.toggle_diag_camera)
        controls.addWidget(self.btn_toggle_diag_cam)
        left.addLayout(controls)
        layout.addLayout(left, 3)

        settings_panel = QWidget()
        panel = QVBoxLayout(settings_panel)
        panel.setContentsMargins(0, 0, 4, 0)
        panel.setSpacing(12)

        cursor_card = Card("Ajustes del cursor", "cursor")
        self.spin_sensitivity = QDoubleSpinBox()
        self.spin_sensitivity.setRange(0.5, 5.0)
        self.spin_sensitivity.setSingleStep(0.1)
        self._add_setting(cursor_card, "Sensibilidad", self.spin_sensitivity, "Cuánto se mueve el cursor por cada movimiento de cabeza.")

        self.spin_dead_zone = QDoubleSpinBox()
        self.spin_dead_zone.setRange(2.0, 30.0)
        self.spin_dead_zone.setSingleStep(1.0)
        self.spin_dead_zone.setSuffix(" px")
        self._add_setting(cursor_card, "Zona muerta", self.spin_dead_zone, "Movimientos menores a este valor se ignoran (evita temblores).")

        self.spin_smoothing = QDoubleSpinBox()
        self.spin_smoothing.setRange(0.1, 0.95)
        self.spin_smoothing.setSingleStep(0.05)
        self._add_setting(cursor_card, "Suavizado", self.spin_smoothing, "Valores altos dan un movimiento más suave pero más lento.")

        self.check_mirror = QCheckBox("Vista espejo de la cámara")
        self.check_invert_x = QCheckBox("Invertir eje horizontal")
        cursor_card.body.addWidget(self.check_mirror)
        cursor_card.body.addWidget(self.check_invert_x)
        panel.addWidget(cursor_card)

        audio_card = Card("Micrófono", "mic")
        self.cb_audio_input = QComboBox()
        audio_card.body.addWidget(self.cb_audio_input)
        btn_refresh_audio = Button("Actualizar micrófonos", "refresh")
        btn_refresh_audio.clicked.connect(self.refresh_audio_device_selector)
        audio_card.body.addWidget(btn_refresh_audio)
        panel.addWidget(audio_card)

        btn_save_settings = Button("Guardar ajustes", "save", variant="primary")
        btn_save_settings.clicked.connect(self.save_quick_settings)
        panel.addWidget(btn_save_settings)
        panel.addStretch(1)

        scroll = self._scroll(settings_panel)
        scroll.setMinimumWidth(330)
        layout.addWidget(scroll, 2)
        self.stack.addWidget(page)

    @staticmethod
    def _add_setting(card: Card, title: str, widget: QWidget, help_text: str):
        row = QHBoxLayout()
        row.addWidget(make_label(title, role="h2"), 1)
        widget.setMinimumWidth(110)
        row.addWidget(widget)
        card.body.addLayout(row)
        card.body.addWidget(make_label(help_text, role="subtle", wrap=True))

    def create_guide_tab(self):
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(0, 0, 4, 0)
        grid.setSpacing(12)
        gestures = [NEUTRAL_GESTURE_META] + get_enabled_gesture_catalog()
        for index, gesture in enumerate(gestures):
            card = Card(spacing=6)
            header = QHBoxLayout()
            header.addWidget(make_label(gesture["title"], role="h2", wrap=True), 1)
            if gesture.get("duration_ms"):
                header.addWidget(make_label(f"{gesture['duration_ms']} ms", role="subtle"))
            card.body.addLayout(header)
            card.body.addWidget(make_label(f"→ {gesture['action']}", tone="accent", wrap=True))
            card.body.addWidget(make_label(f"<b>Cómo hacerlo:</b> {gesture['training_hint']}", wrap=True))
            card.body.addWidget(make_label(f"⚠ {gesture['common_failure_hint']}", role="small", tone="warning", wrap=True))
            card.body.addStretch(1)
            grid.addWidget(card, index // 2, index % 2)
        grid.setRowStretch((len(gestures) + 1) // 2, 1)
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._scroll(content))
        self.stack.addWidget(page)

    def create_calibration_tab(self):
        page, layout = self._page()

        left = QVBoxLayout()
        left.setSpacing(10)
        self.calib_view = VideoView(placeholder="La cámara se enciende al iniciar la captura", min_size=(420, 260))
        left.addWidget(self.calib_view, 1)

        self.lbl_calib_status = make_label("Selecciona un gesto de la lista y pulsa «Iniciar captura».", role="h2", wrap=True)
        self.lbl_calib_status.setAlignment(Qt.AlignCenter)
        left.addWidget(self.lbl_calib_status)

        self.calib_progress = QProgressBar()
        self.calib_progress.setFormat("")
        self.calib_progress.setFixedHeight(22)
        left.addWidget(self.calib_progress)

        buttons = QHBoxLayout()
        self.btn_capture = Button("Iniciar captura (cuenta regresiva de 3 s)", "record", variant="primary", size="large")
        self.btn_capture.clicked.connect(self.on_capture_click)
        self.btn_capture.setEnabled(False)
        buttons.addWidget(self.btn_capture, 1)
        self.btn_reset_gesture = Button("Limpiar muestras", "trash", variant="danger", tooltip="Borra las muestras del gesto seleccionado")
        self.btn_reset_gesture.clicked.connect(self.reset_gesture_samples)
        buttons.addWidget(self.btn_reset_gesture)
        left.addLayout(buttons)
        layout.addLayout(left, 3)

        right = Card("Gestos del perfil", "list")
        self.gesture_list = QListWidget()
        self.gesture_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.gesture_list.currentItemChanged.connect(self.on_gesture_selected_changed)
        right.body.addWidget(self.gesture_list, 1)
        self.lbl_gesture_desc = make_label("Selecciona un gesto para ver sus instrucciones.", role="muted", wrap=True)
        self.lbl_gesture_desc.setTextFormat(Qt.RichText)
        right.body.addWidget(self.lbl_gesture_desc)
        right.setMinimumWidth(330)
        layout.addWidget(right, 2)
        self.stack.addWidget(page)

    def create_training_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        action_card = Card()
        action_row = QHBoxLayout()
        texts = QVBoxLayout()
        texts.setSpacing(2)
        self.lbl_train_title = make_label("Modelo híbrido GRU + Regresión logística", role="h2")
        self.txt_train_status = make_label("Completa la calibración de todos los gestos para entrenar.", role="muted", wrap=True)
        texts.addWidget(self.lbl_train_title)
        texts.addWidget(self.txt_train_status)
        action_row.addLayout(texts, 1)
        self.btn_train = Button("Entrenar modelo", "chip", variant="primary", size="large")
        self.btn_train.clicked.connect(self.run_training)
        action_row.addWidget(self.btn_train)
        action_card.body.addLayout(action_row)
        self.train_busy = QProgressBar()
        self.train_busy.setRange(0, 0)
        self.train_busy.setFixedHeight(8)
        self.train_busy.setTextVisible(False)
        self.train_busy.setVisible(False)
        action_card.body.addWidget(self.train_busy)
        layout.addWidget(action_card)

        stats = QHBoxLayout()
        stats.setSpacing(12)
        self.stat_accuracy = StatCard("Precisión de validación")
        self.stat_train = StatCard("Ventanas de entrenamiento")
        self.stat_val = StatCard("Ventanas de validación")
        self.stat_classes = StatCard("Clases (gestos + neutral)")
        for card in (self.stat_accuracy, self.stat_train, self.stat_val, self.stat_classes):
            stats.addWidget(card)
        layout.addLayout(stats)

        details = QHBoxLayout()
        details.setSpacing(12)
        metrics_card = Card("Métricas por clase", "list")
        self.metrics_table = QTableWidget(0, 5)
        self.metrics_table.setHorizontalHeaderLabels(["Gesto", "Precisión", "Recall", "F1", "Muestras"])
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.metrics_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.metrics_table.setShowGrid(False)
        header = self.metrics_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        metrics_card.body.addWidget(self.metrics_table, 1)
        details.addWidget(metrics_card, 1)

        confusion_card = Card("Matriz de confusión (validación)", "target")
        self.confusion_view = ConfusionMatrixView()
        confusion_card.body.addWidget(self.confusion_view, 1)
        details.addWidget(confusion_card, 1)
        layout.addLayout(details, 1)

        self.txt_train_results = make_label("", role="subtle", wrap=True)
        layout.addWidget(self.txt_train_results)
        self._clear_training_results()
        self.stack.addWidget(page)

    # --------------------------------------------------------------- perfiles
    def refresh_profile_list(self):
        summaries = [get_profile_summary(name) for name in list_profiles()]
        populate_profile_list(self.profile_list, summaries, self.current_profile_name or None)

    def _selected_profile_name(self) -> str:
        item = self.profile_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else ""

    def load_selected_profile(self):
        name = self._selected_profile_name()
        if not name:
            return

        self.current_profile_name = name
        self.profile = load_profile(name)
        self.calibration_service = CalibrationService(self.window_size)
        self.training_service = TrainingService(name, self.window_size)

        # Cargar dataset si existe
        has_model = self.training_service.load_classifier()
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

        # Habilitar pasos
        self._set_step_enabled(STEP_DIAGNOSTICS, True)
        self._set_step_enabled(STEP_CALIBRATION, True)
        self._set_step_enabled(STEP_TRAINING, True)

        calibrated = bool(self.profile.get("calibration", {}).get("completed"))
        self.profile_badge.set_status(f"Perfil: {name}", "success" if calibrated else "warning")
        self.refresh_gestures_list_widget()
        if has_model and self.training_service.classifier.training_summary:
            self._show_training_summary(self.training_service.classifier.training_summary)
        else:
            self._clear_training_results()
        self._update_train_status()
        self._refresh_step_icons()
        self._go_to_step(STEP_DIAGNOSTICS)
        self.statusBar().showMessage(f"Perfil «{name}» cargado.", 5000)

    def delete_selected_profile(self):
        name = self._selected_profile_name()
        if not name:
            return

        confirm = QMessageBox.question(
            self,
            "Confirmar eliminación",
            f"¿Seguro que deseas eliminar permanentemente el perfil «{name}» y sus modelos?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            delete_profile(name)
            if name == self.current_profile_name:
                self.current_profile_name = ""
                self.profile = {}
                self.calibration_service = None
                self.training_service = None
                self.profile_badge.set_status("Sin perfil cargado", "neutral")
                for step in (STEP_DIAGNOSTICS, STEP_CALIBRATION, STEP_TRAINING):
                    self._set_step_enabled(step, False)
            self.refresh_profile_list()
            self.statusBar().showMessage(f"Perfil «{name}» eliminado.", 5000)

    def create_new_profile(self):
        name = self.txt_new_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Nombre requerido", "Introduce un nombre para el perfil.")
            return
        if name in list_profiles():
            QMessageBox.warning(self, "Perfil existente", f"Ya existe un perfil llamado «{name}».")
            return

        # Crear perfil
        profile = build_default_profile(name)
        profile["voice_enabled"] = self.check_voice.isChecked()
        profile["tts_enabled"] = self.check_tts.isChecked()
        save_profile(profile)

        self.txt_new_name.clear()
        self.current_profile_name = name
        self.refresh_profile_list()
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
        self.statusBar().showMessage("Ajustes del perfil y del micrófono guardados.", 5000)

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

    # ----------------------------------------------- cámara de diagnóstico
    def toggle_diag_camera(self):
        if self.camera_thread and self.camera_thread.isRunning():
            self.stop_diag_camera()
        else:
            self.start_diag_camera()

    def start_diag_camera(self):
        camera_idx = int(self.settings.get("camera_index", 0))
        fps = int(self.settings.get("fps", 15))
        if self.camera_thread:
            self.camera_thread.stop()
        self.camera_thread = CameraThread(camera_index=camera_idx, fps=fps)
        self.camera_thread.frame_processed.connect(self.on_diag_frame)
        self.camera_thread.camera_error.connect(self.on_camera_error)
        self.camera_thread.start()
        self._last_face_present = None
        self.face_badge.set_status("Encendiendo cámara…", "info", pulse=True)
        self.btn_toggle_diag_cam.setText("Detener prueba de cámara")
        self.btn_toggle_diag_cam.set_icon_name("stop")
        self.btn_toggle_diag_cam.set_variant("")

    def stop_diag_camera(self):
        if self.camera_thread and not self.is_capturing and not self.countdown_timer.isActive():
            self.camera_thread.stop()
            self.camera_thread = None
        self.diag_view.clear("La cámara está inactiva")
        self.face_badge.set_status("Cámara apagada", "neutral")
        self.btn_toggle_diag_cam.setText("Iniciar prueba de cámara")
        self.btn_toggle_diag_cam.set_icon_name("camera")
        self.btn_toggle_diag_cam.set_variant("primary")

    @Slot(object, object)
    def on_diag_frame(self, frame_rgb, face_sample):
        # Dibuja landmarks de diagnóstico directamente sobre el RGB.
        frame = frame_rgb.copy()
        face_present = face_sample is not None
        if face_present:
            for pt in face_sample.points_px.values():
                cv2.circle(frame, pt, 1, (80, 255, 60), -1)
            cv2.circle(frame, face_sample.nose_px, 4, (255, 0, 255), -1)
            sc = int(face_sample.face_scale_px)
            fc = face_sample.face_center_px
            cv2.rectangle(frame, (fc[0] - sc // 2, fc[1] - sc // 2), (fc[0] + sc // 2, fc[1] + sc // 2), (0, 180, 255), 1)

        if self.profile.get("mirror_preview", True):
            frame = cv2.flip(frame, 1)
        self.diag_view.set_frame(frame)

        if face_present != self._last_face_present:
            self._last_face_present = face_present
            if face_present:
                self.face_badge.set_status("Rostro detectado", "success")
            else:
                self.face_badge.set_status("Sin rostro", "warning", pulse=True)

    # --------------------------------------------------------- calibración
    def refresh_gestures_list_widget(self):
        selected_id = self.capture_gesture_id
        self.gesture_list.blockSignals(True)
        self.gesture_list.clear()
        if not self.calibration_service:
            self.gesture_list.blockSignals(False)
            return

        statuses = self.calibration_service.refresh_statuses()
        selected_row = -1
        for row, gesture_id in enumerate(self.calibration_service.gesture_order):
            status = statuses[gesture_id]
            widget = GestureProgressItem(status)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, gesture_id)
            hint = widget.sizeHint()
            item.setSizeHint(QSize(hint.width(), hint.height() + 6))
            self.gesture_list.addItem(item)
            self.gesture_list.setItemWidget(item, widget)
            if gesture_id == selected_id:
                selected_row = row
        self.gesture_list.blockSignals(False)
        if selected_row >= 0:
            self.gesture_list.setCurrentRow(selected_row)
        self._refresh_step_icons()
        self._update_train_status()

    def on_gesture_selected_changed(self, current, previous):
        if not current:
            self.lbl_gesture_desc.setText("")
            self.btn_capture.setEnabled(False)
            return

        gesture_id = current.data(Qt.UserRole)
        self.capture_gesture_id = gesture_id
        meta = self.calibration_service.gesture_meta[gesture_id]
        warning = theme.palette()["warning"]

        desc = (
            f"<b>{meta['title']}</b><br/>"
            f"{meta['training_hint']}<br/>"
            f"<span style='color:{warning}'>⚠ {meta.get('common_failure_hint', '-')}</span><br/>"
            f"<i>Mínimo recomendado: {meta['recommended_min_frames']} fotogramas.</i>"
        )
        self.lbl_gesture_desc.setText(desc)
        self.btn_capture.setEnabled(True)

        # Actualizar progreso
        status = self.calibration_service.refresh_statuses()[gesture_id]
        self._set_calib_progress(status.valid_frames, meta["recommended_min_frames"])
        self._set_calib_status(f"Listo para capturar: {GESTURE_SHORT_TITLES.get(gesture_id, meta['title'])}")

    def _set_calib_progress(self, valid_frames: int, minimum: int):
        self.calib_progress.setMaximum(max(minimum, valid_frames, 1))
        self.calib_progress.setValue(valid_frames)
        self.calib_progress.setFormat(f"{valid_frames} / {minimum} fotogramas")
        theme.set_widget_property(self.calib_progress, "tone", "success" if valid_frames >= minimum else "")

    def _set_calib_status(self, text: str, tone: str = ""):
        set_label_text(self.lbl_calib_status, text)
        theme.set_widget_property(self.lbl_calib_status, "tone", tone)

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
        self.calib_view.set_overlay_text(str(self.countdown_val))
        self._set_calib_status("Prepárate…", "info")
        self.countdown_timer.start(1000)

    def on_countdown_tick(self):
        self.countdown_val -= 1
        if self.countdown_val > 0:
            self.calib_view.set_overlay_text(str(self.countdown_val))
        else:
            self.countdown_timer.stop()
            self.calib_view.set_overlay_text("¡Mantén el gesto!", "success")
            QTimer.singleShot(900, lambda: self.calib_view.set_overlay_text(""))
            self.calib_view.set_corner_badge("GRABANDO", "danger")
            self._set_calib_status("¡Mantén el gesto ahora!", "success")
            # Iniciar grabación conectando la captura de muestras
            self.camera_thread.frame_processed.disconnect(self.on_calibration_frame)
            self.camera_thread.frame_processed.connect(self.on_capture_sample_frame)
            self.is_capturing = True
            self.btn_capture.setText("Detener captura")
            self.btn_capture.set_icon_name("stop")
            self.btn_capture.set_variant("danger")
            self.btn_capture.setEnabled(True)

    def _render_calibration_frame(self, frame_rgb, face_sample=None):
        frame = frame_rgb.copy() if face_sample is not None else frame_rgb
        if face_sample is not None:
            for pt in face_sample.points_px.values():
                cv2.circle(frame, pt, 1, (80, 255, 60), -1)
        if self.profile.get("mirror_preview", True):
            frame = cv2.flip(frame, 1)
        self.calib_view.set_frame(frame)

    @Slot(object, object)
    def on_calibration_frame(self, frame_rgb, face_sample):
        self._render_calibration_frame(frame_rgb)

    @Slot(object, object)
    def on_capture_sample_frame(self, frame_rgb, face_sample):
        # Registrar muestra
        is_registered, feedback = self.calibration_service.register_sample(self.capture_gesture_id, face_sample)
        self._render_calibration_frame(frame_rgb, face_sample)

        # Actualizar progreso
        status = self.calibration_service.refresh_statuses()[self.capture_gesture_id]
        meta = self.calibration_service.gesture_meta[self.capture_gesture_id]
        minimum = meta["recommended_min_frames"]

        self._set_calib_progress(status.valid_frames, minimum)
        if status.valid_frames >= minimum:
            self._set_calib_status("✓ Mínimo alcanzado. Puedes detener la captura.", "success")
        else:
            self._set_calib_status(feedback or "Mantén el gesto…", "" if is_registered else "warning")

    def stop_capture_and_camera(self):
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None

        was_capturing = self.is_capturing
        self.is_capturing = False
        self.calib_view.set_overlay_text("")
        self.calib_view.set_corner_badge("")
        self.btn_capture.setText("Iniciar captura (cuenta regresiva de 3 s)")
        self.btn_capture.set_icon_name("record")
        self.btn_capture.set_variant("primary")

        self._set_calib_status("Captura finalizada." if was_capturing else "Captura cancelada.")
        self.btn_capture.setEnabled(bool(self.capture_gesture_id))
        self.gesture_list.setEnabled(True)

        # Guardar persistencia temporal de muestras registradas
        if self.calibration_service:
            samples = self.calibration_service.export_samples()
            quality = self.calibration_service.export_capture_quality()
            self.training_service.save_pending_samples(samples, quality)

        # Recargar la lista manteniendo el gesto seleccionado
        self.refresh_gestures_list_widget()

    def reset_gesture_samples(self):
        if not self.capture_gesture_id or not self.calibration_service:
            return

        title = GESTURE_SHORT_TITLES.get(self.capture_gesture_id, self.capture_gesture_id)
        confirm = QMessageBox.question(
            self,
            "Confirmar limpieza",
            f"¿Seguro que deseas borrar todas las muestras del gesto «{title}»?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm == QMessageBox.Yes:
            self.calibration_service.reset_gesture(self.capture_gesture_id)
            meta = self.calibration_service.gesture_meta[self.capture_gesture_id]
            self._set_calib_progress(0, meta["recommended_min_frames"])
            self._set_calib_status("Muestras eliminadas.", "warning")

            # Guardar el cambio tras resetear
            samples = self.calibration_service.export_samples()
            quality = self.calibration_service.export_capture_quality()
            self.training_service.save_pending_samples(samples, quality)
            self.refresh_gestures_list_widget()

    def on_camera_error(self, err_msg):
        self.countdown_timer.stop()
        self.stop_capture_and_camera()
        self.stop_diag_camera()
        QMessageBox.critical(self, "Error de cámara", f"Ocurrió un error en el dispositivo de vídeo:\n{err_msg}")

    # -------------------------------------------------------- entrenamiento
    def _update_train_status(self):
        if not self.calibration_service:
            return
        can_train, blockers, _statuses = self.calibration_service.can_train()
        if self.training_worker is not None:
            return
        if can_train:
            self.txt_train_status.setText("Todos los gestos tienen muestras suficientes. Ya puedes entrenar.")
            theme.set_widget_property(self.txt_train_status, "tone", "success")
        else:
            pending = ", ".join(GESTURE_SHORT_TITLES.get(blocker, blocker) for blocker in blockers)
            self.txt_train_status.setText(f"Faltan muestras de: {pending}.")
            theme.set_widget_property(self.txt_train_status, "tone", "warning")

    def run_training(self):
        if not self.calibration_service or self.training_worker is not None:
            return

        can_train, blockers, statuses = self.calibration_service.can_train()
        if not can_train:
            pending = ", ".join(GESTURE_SHORT_TITLES.get(blocker, blocker) for blocker in blockers)
            QMessageBox.warning(
                self,
                "No se puede entrenar",
                f"Aún faltan gestos por calibrar con suficientes muestras:\n{pending}",
            )
            return

        self.btn_train.setEnabled(False)
        self.train_busy.setVisible(True)
        self.txt_train_status.setText("Entrenando el clasificador… la interfaz sigue disponible.")
        theme.set_widget_property(self.txt_train_status, "tone", "info")

        samples = self.calibration_service.export_samples()
        quality = self.calibration_service.export_capture_quality()
        self._training_inputs = (samples, quality)
        self.training_worker = TrainingWorker(self.training_service, samples, quality)
        self.training_worker.succeeded.connect(self._on_training_succeeded)
        self.training_worker.failed.connect(self._on_training_failed)
        self.training_worker.finished.connect(self._on_training_finished)
        self.training_worker.start()

    def _on_training_succeeded(self, classifier):
        samples, quality = self._training_inputs
        self.calibration_service.mark_trained()
        self.training_service.clear_pending_samples()

        # Guardar estado de calibración en el perfil
        self.profile = self.training_service.build_profile_training_state(
            self.profile, samples, quality, self.calibration_service.refresh_statuses()
        )
        save_profile(self.profile)

        summary = classifier.training_summary
        self._show_training_summary(summary)
        self.txt_train_status.setText(
            f"Modelo v{summary['version']} listo para usar en el asistente "
            f"({format_accuracy(summary['validation_accuracy'])} de precisión en validación)."
        )
        theme.set_widget_property(self.txt_train_status, "tone", "success")
        self.profile_badge.set_status(f"Perfil: {self.current_profile_name}", "success")
        self.refresh_gestures_list_widget()
        self.refresh_profile_list()
        self.statusBar().showMessage("Entrenamiento completado.", 6000)

    def _on_training_failed(self, message: str):
        self.txt_train_status.setText(f"Error en el entrenamiento: {message}")
        theme.set_widget_property(self.txt_train_status, "tone", "danger")
        QMessageBox.critical(self, "Error", f"Ocurrió un error al entrenar el modelo:\n{message}")

    def _on_training_finished(self):
        self.training_worker = None
        self.btn_train.setEnabled(True)
        self.train_busy.setVisible(False)
        self._refresh_step_icons()

    def _clear_training_results(self):
        for card in (self.stat_accuracy, self.stat_train, self.stat_val, self.stat_classes):
            card.set_value("—")
        self.metrics_table.setRowCount(0)
        self.confusion_view.set_data([], [])
        self.txt_train_results.setText("Aún no hay un modelo entrenado para este perfil.")

    def _show_training_summary(self, summary: dict):
        accuracy = summary.get("validation_accuracy")
        tone = None
        if accuracy is not None:
            tone = "success" if accuracy >= 0.9 else "warning" if accuracy >= 0.75 else "danger"
        self.stat_accuracy.set_value(format_accuracy(accuracy), tone)
        self.stat_train.set_value(str(summary.get("train_windows", summary.get("training_windows", "—"))))
        self.stat_val.set_value(str(summary.get("validation_windows", "—")))
        classes = list(summary.get("classes", []))
        self.stat_classes.set_value(str(len(classes)) if classes else "—")

        report = summary.get("class_metrics", {}) or {}
        self.metrics_table.setRowCount(len(classes))
        for row, gesture_id in enumerate(classes):
            info = report.get(gesture_id, {})
            values = [
                GESTURE_SHORT_TITLES.get(gesture_id, gesture_id),
                f"{info.get('precision', 0.0):.2f}",
                f"{info.get('recall', 0.0):.2f}",
                f"{info.get('f1-score', 0.0):.2f}",
                str(int(info.get("support", 0))),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column:
                    item.setTextAlignment(Qt.AlignCenter)
                if column == 3:
                    f1 = info.get("f1-score", 0.0)
                    item.setForeground(theme.color("success" if f1 >= 0.9 else "warning" if f1 >= 0.75 else "danger"))
                self.metrics_table.setItem(row, column, item)
        self.metrics_table.resizeRowsToContents()

        labels = [GESTURE_SHORT_TITLES.get(gesture_id, gesture_id) for gesture_id in classes]
        self.confusion_view.set_data(summary.get("confusion_matrix", []), labels)
        trained_at = str(summary.get("trained_at", ""))[:16].replace("T", " ")
        self.txt_train_results.setText(
            f"Modelo v{summary.get('version', '-')} · entrenado {trained_at} UTC · "
            f"ventana temporal de {summary.get('window_size', '-')} fotogramas · "
            f"{summary.get('model_type', 'LogisticRegression')} sobre embeddings GRU."
        )

    # -------------------------------------------------------------- cierre
    def closeEvent(self, event):
        # Apagar cámaras y salir
        self.countdown_timer.stop()
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        if self.training_worker is not None:
            self.training_worker.wait(15000)
        event.accept()
        self.closed.emit()


def main():
    app = QApplication(sys.argv)
    theme.apply_theme(app, load_settings())
    window = TrainerMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

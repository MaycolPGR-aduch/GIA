"""Pantalla de inicio: perfiles, su estado y qué hacer con ellos."""
from __future__ import annotations

import re

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QVBoxLayout,
    QWidget,
)

from app import ui_theme as theme
from app.config_store import (
    build_default_profile,
    delete_profile,
    get_profile_summary,
    list_profiles,
    load_profile,
    save_profile,
    save_settings,
)
from app.ui_widgets import Button, Card, StatusBadge, format_accuracy, format_date, make_label, populate_profile_list

AUTOSTART_SECONDS = 5
PROFILE_NAME_PATTERN = re.compile(r"^[\w][\w .()-]{0,39}$", re.UNICODE)


def validate_profile_name(name: str, existing: list[str]) -> str:
    """Devuelve el problema del nombre, o una cadena vacía si es válido (también se usa como carpeta)."""
    clean = name.strip()
    if not clean:
        return "Introduce un nombre para el perfil."
    if not PROFILE_NAME_PATTERN.match(clean) or clean.endswith("."):
        return "Usa solo letras, números, espacios y . ( ) - (hasta 40 caracteres)."
    if clean.casefold() in {item.casefold() for item in existing}:
        return f"Ya existe un perfil llamado «{clean}»."
    return ""


def primary_action(summary: dict) -> str:
    """Lo más útil que se puede hacer con el perfil: "start" si está listo, "calibrate" si no."""
    return "start" if summary.get("calibrated") and summary.get("model_version") is not None else "calibrate"


class ModelInspector(QThread):
    """Lee el modelo fuera del hilo de la interfaz: la primera vez importa scikit-learn."""

    inspected = Signal(str, dict)

    def __init__(self, profile_name: str, parent=None):
        super().__init__(parent)
        self.profile_name = profile_name

    def run(self):
        info = {"needs_refresh": False, "mixed_resolution": False}
        try:
            from app.gesture_ml import GestureClassifier

            classifier = GestureClassifier(self.profile_name)
            if classifier.load():
                info["needs_refresh"] = classifier.needs_refresh
                info["mixed_resolution"] = bool(classifier.training_summary.get("mixed_capture_resolution"))
        except Exception as exc:  # noqa: BLE001 - un modelo dañado no debe impedir usar la pantalla
            info["error"] = str(exc)
        self.inspected.emit(self.profile_name, info)


class NewProfileDialog(QDialog):
    def __init__(self, existing: list[str], parent=None):
        super().__init__(parent)
        self.existing = existing
        self.setWindowTitle("GIA · Nuevo perfil")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 18)
        layout.setSpacing(10)
        layout.addWidget(make_label("Nuevo perfil", role="title"))
        layout.addWidget(make_label("Cada persona necesita su propio perfil, porque el sistema aprende sus gestos.", role="muted", wrap=True))
        layout.addWidget(make_label("Nombre", role="h2"))
        self.txt_name = QLineEdit()
        self.txt_name.setPlaceholderText("Ej. Juan")
        self.txt_name.returnPressed.connect(self._accept_if_valid)
        self.txt_name.textChanged.connect(lambda _text: self.lbl_error.setText(""))
        layout.addWidget(self.txt_name)
        self.lbl_error = make_label("", tone="danger", wrap=True)
        layout.addWidget(self.lbl_error)
        self.check_voice = QCheckBox("Comandos de voz")
        self.check_voice.setChecked(True)
        layout.addWidget(self.check_voice)
        layout.addWidget(make_label("Permite dar órdenes habladas cerrando ambos ojos.", role="subtle", wrap=True))
        self.check_tts = QCheckBox("Respuesta hablada")
        layout.addWidget(self.check_tts)
        layout.addWidget(make_label("El sistema confirma en voz alta cada acción.", role="subtle", wrap=True))
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        btn_cancel = Button("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        btn_create = Button("Crear perfil", "plus", variant="primary")
        btn_create.clicked.connect(self._accept_if_valid)
        buttons.addWidget(btn_cancel)
        buttons.addWidget(btn_create)
        layout.addLayout(buttons)

    def _accept_if_valid(self):
        problem = validate_profile_name(self.txt_name.text(), self.existing)
        if problem:
            self.lbl_error.setText(problem)
            return
        self.accept()

    def build_profile(self) -> dict:
        profile = build_default_profile(self.txt_name.text().strip())
        profile["voice_enabled"] = self.check_voice.isChecked()
        profile["tts_enabled"] = self.check_tts.isChecked()
        return profile


class HomePage(QWidget):
    start_requested = Signal(str)
    train_requested = Signal(str)
    update_requested = Signal(str)
    voice_commands_requested = Signal()
    quit_requested = Signal()

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._summary: dict | None = None
        self._inspectors: list[ModelInspector] = []
        self._busy = False
        self._loading_details = False
        self._countdown = 0
        self._autostart_name = ""
        self._autostart_timer = QTimer(self)
        self._autostart_timer.setInterval(1000)
        self._autostart_timer.timeout.connect(self._on_autostart_tick)
        self._build_ui()
        shortcut = QShortcut(QKeySequence("Esc"), self)
        shortcut.activated.connect(self.cancel_autostart)
        self.refresh(settings.get("last_profile"))

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
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
        titles.addWidget(make_label("Control del computador con gestos faciales y voz, sin conexión.", role="muted"))
        header.addLayout(titles, 1)
        self.btn_contrast = Button("", "contrast", variant="ghost", tooltip="Alto contraste", checkable=True)
        self.btn_contrast.setChecked(theme.current_theme() == "high_contrast")
        self.btn_contrast.setFixedSize(44, 44)
        self.btn_contrast.clicked.connect(self._toggle_contrast)
        header.addWidget(self.btn_contrast)
        btn_quit = Button("Salir", "power", variant="ghost")
        btn_quit.clicked.connect(self.quit_requested)
        header.addWidget(btn_quit)
        layout.addLayout(header)

        self.autostart_banner = Card(spacing=8)
        banner_row = QHBoxLayout()
        self.lbl_autostart = make_label("", role="h2", tone="accent", wrap=True)
        banner_row.addWidget(self.lbl_autostart, 1)
        btn_start_now = Button("Iniciar ahora", "play", variant="primary")
        btn_start_now.clicked.connect(self._start_autostart_now)
        banner_row.addWidget(btn_start_now)
        btn_cancel_auto = Button("Cancelar (Esc)", "x")
        btn_cancel_auto.clicked.connect(self.cancel_autostart)
        banner_row.addWidget(btn_cancel_auto)
        self.autostart_banner.body.addLayout(banner_row)
        self.autostart_banner.setVisible(False)
        layout.addWidget(self.autostart_banner)

        body = QHBoxLayout()
        body.setSpacing(14)
        layout.addLayout(body, 1)

        profiles = Card("Perfiles", "user")
        self.profile_list = QListWidget()
        self.profile_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.profile_list.currentItemChanged.connect(self._on_user_selected_profile)
        self.profile_list.itemPressed.connect(lambda _item: self.cancel_autostart())
        self.profile_list.itemDoubleClicked.connect(lambda _item: self._run_primary_action())
        profiles.body.addWidget(self.profile_list, 1)
        self.btn_new = Button("Nuevo perfil", "plus")
        self.btn_new.clicked.connect(self.create_profile)
        profiles.body.addWidget(self.btn_new)
        profiles.setMinimumWidth(380)
        body.addWidget(profiles, 3)

        self.details = Card(spacing=12)
        self.lbl_name = make_label("", role="display", wrap=True)
        self.details.body.addWidget(self.lbl_name)
        self.status_badge = StatusBadge("", "neutral", large=True)
        badge_row = QHBoxLayout()
        badge_row.addWidget(self.status_badge)
        badge_row.addStretch(1)
        self.details.body.addLayout(badge_row)
        self.lbl_model = make_label("", role="muted", wrap=True)
        self.details.body.addWidget(self.lbl_model)
        self.lbl_notice = make_label("", tone="warning", wrap=True)
        self.lbl_notice.setVisible(False)
        self.details.body.addWidget(self.lbl_notice)
        self.btn_update = Button("Actualizar modelo", "refresh", tooltip="Reentrena con los datos guardados; no hay que recalibrar")
        self.btn_update.setVisible(False)
        self.btn_update.clicked.connect(self._on_update_clicked)
        self.details.body.addWidget(self.btn_update)

        self.btn_primary = Button("Iniciar asistente", "play", variant="primary", size="large")
        self.btn_primary.clicked.connect(self._run_primary_action)
        self.details.body.addWidget(self.btn_primary)
        self.btn_train = Button("Calibrar y entrenar", "chip", size="large", tooltip="Diagnóstico, guía, calibración y entrenamiento")
        self.btn_train.clicked.connect(self._on_train_clicked)
        self.details.body.addWidget(self.btn_train)

        self.check_voice = QCheckBox("Comandos de voz")
        self.check_voice.toggled.connect(lambda value: self._save_profile_option("voice_enabled", value))
        self.details.body.addWidget(self.check_voice)
        self.check_tts = QCheckBox("Respuesta hablada")
        self.check_tts.toggled.connect(lambda value: self._save_profile_option("tts_enabled", value))
        self.details.body.addWidget(self.check_tts)
        self.details.body.addStretch(1)

        secondary = QHBoxLayout()
        self.btn_voice = Button("Frases de voz", "mic", tooltip="Configurar frases y macros, comunes a todos los perfiles")
        self.btn_voice.clicked.connect(self.voice_commands_requested)
        secondary.addWidget(self.btn_voice)
        secondary.addStretch(1)
        self.btn_delete = Button("Eliminar perfil", "trash", variant="danger")
        self.btn_delete.clicked.connect(self.delete_selected_profile)
        secondary.addWidget(self.btn_delete)
        self.details.body.addLayout(secondary)
        body.addWidget(self.details, 2)

        self.empty_state = Card(spacing=12)
        self.empty_state.body.addStretch(1)
        self.empty_state.body.addWidget(make_label("Bienvenido a GIA", role="display", wrap=True))
        self.empty_state.body.addWidget(make_label(
            "Crea tu perfil para empezar. Después el sistema te guiará para comprobar la cámara, "
            "aprender tus gestos y empezar a usarlo.",
            role="muted",
            wrap=True,
        ))
        btn_first = Button("Crear mi perfil", "plus", variant="primary", size="large")
        btn_first.clicked.connect(self.create_profile)
        self.empty_state.body.addWidget(btn_first)
        self.empty_state.body.addStretch(1)
        body.addWidget(self.empty_state, 2)

        footer = QHBoxLayout()
        self.check_autostart = QCheckBox("Iniciar el asistente al abrir GIA con el último perfil usado")
        self.check_autostart.setChecked(bool(self.settings.get("autostart_last_profile", True)))
        self.check_autostart.toggled.connect(self._on_autostart_toggled)
        footer.addWidget(self.check_autostart)
        footer.addStretch(1)
        self.lbl_busy = make_label("", role="h2", tone="info")
        footer.addWidget(self.lbl_busy)
        layout.addLayout(footer)

    # ------------------------------------------------------------- perfiles
    def refresh(self, select_name: str | None = None):
        """Vuelve a leer los perfiles del disco, sin disparar ninguna acción."""
        current = select_name or self.selected_name()
        self.btn_contrast.setChecked(theme.current_theme() == "high_contrast")
        summaries = [get_profile_summary(name) for name in list_profiles()]
        # Primero los perfiles calibrados, luego alfabético.
        summaries.sort(key=lambda summary: (not summary["calibrated"], summary["name"].lower()))
        self.profile_list.blockSignals(True)
        populate_profile_list(self.profile_list, summaries, current)
        self.profile_list.blockSignals(False)
        self._on_profile_selected()

    def check_ready(self, name: str) -> bool:
        return primary_action(get_profile_summary(name)) == "start"

    def selected_name(self) -> str:
        item = self.profile_list.currentItem()
        return item.data(Qt.UserRole) if item is not None else ""

    def _on_user_selected_profile(self, *_args):
        self.cancel_autostart()
        self._on_profile_selected()

    def _on_profile_selected(self, *_args):
        name = self.selected_name()
        has_profile = bool(name)
        self.details.setVisible(has_profile)
        self.empty_state.setVisible(not has_profile)
        if not has_profile:
            self._summary = None
            return
        self._summary = get_profile_summary(name)
        self.lbl_name.setText(name)
        self.lbl_notice.setVisible(False)
        self.btn_update.setVisible(False)
        if primary_action(self._summary) == "start":
            self.status_badge.set_status("Listo para usar", "success")
            self.lbl_model.setText(
                f"Modelo v{self._summary['model_version']}  ·  "
                f"{format_accuracy(self._summary.get('validation_accuracy'))} de precisión  ·  "
                f"entrenado {format_date(self._summary.get('trained_at'))}"
            )
            self.btn_primary.setText("Iniciar asistente")
            self.btn_primary.set_icon_name("play")
            self.btn_train.setVisible(True)
            self._inspect_model(name)
        else:
            self.status_badge.set_status("Sin calibrar", "warning")
            self.lbl_model.setText("El sistema todavía no conoce tus gestos. La calibración tarda unos minutos.")
            self.btn_primary.setText("Calibrar y entrenar")
            self.btn_primary.set_icon_name("chip")
            self.btn_train.setVisible(False)
        self._load_profile_options(name)
        self._apply_busy_state()

    def _load_profile_options(self, name: str):
        self._loading_details = True
        try:
            profile = load_profile(name)
            self.check_voice.setChecked(bool(profile.get("voice_enabled", True)))
            self.check_tts.setChecked(bool(profile.get("tts_enabled", False)))
        finally:
            self._loading_details = False

    def _save_profile_option(self, key: str, value: bool):
        name = self.selected_name()
        if self._loading_details or not name:
            return
        self.cancel_autostart()
        profile = load_profile(name)
        profile[key] = bool(value)
        save_profile(profile)

    def _inspect_model(self, name: str):
        inspector = ModelInspector(name, self)
        inspector.inspected.connect(self._on_model_inspected)
        inspector.finished.connect(lambda: self._forget_inspector(inspector))
        self._inspectors.append(inspector)
        inspector.start()

    def _forget_inspector(self, inspector: ModelInspector):
        if inspector in self._inspectors:
            self._inspectors.remove(inspector)
        inspector.deleteLater()

    def _on_model_inspected(self, name: str, info: dict):
        if name != self.selected_name():
            return
        notices = []
        if info.get("error"):
            notices.append("No se pudo leer el modelo de este perfil: habrá que volver a entrenarlo.")
            self.status_badge.set_status("Modelo dañado", "danger")
        if info.get("mixed_resolution"):
            notices.append("La calibración mezcla resoluciones de cámara: conviene recapturar todos los gestos.")
        if info.get("needs_refresh"):
            notices.append("Hay una mejora disponible para este modelo. No hace falta recalibrar.")
        self.btn_update.setVisible(bool(info.get("needs_refresh")))
        self.lbl_notice.setText("\n".join(notices))
        self.lbl_notice.setVisible(bool(notices))
        self._apply_busy_state()

    def create_profile(self):
        self.cancel_autostart()
        dialog = NewProfileDialog(list_profiles(), self)
        if dialog.exec() != QDialog.Accepted:
            return
        profile = dialog.build_profile()
        save_profile(profile)
        self.refresh(profile["name"])
        self.train_requested.emit(profile["name"])

    def delete_selected_profile(self):
        name = self.selected_name()
        if not name:
            return
        self.cancel_autostart()
        confirm = QMessageBox.question(
            self,
            "Eliminar perfil",
            f"¿Eliminar permanentemente el perfil «{name}», sus calibraciones y sus modelos?\nNo se puede deshacer.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        delete_profile(name)
        if self.settings.get("last_profile") == name:
            self.settings["last_profile"] = None
            save_settings(self.settings)
        self.refresh()

    # --------------------------------------------------------------- acciones
    def _run_primary_action(self):
        if self._busy or self._summary is None:
            return
        self.cancel_autostart()
        name = self._summary["name"]
        if primary_action(self._summary) == "start":
            self.start_requested.emit(name)
        else:
            self.train_requested.emit(name)

    def _on_train_clicked(self):
        self.cancel_autostart()
        if self.selected_name():
            self.train_requested.emit(self.selected_name())

    def _on_update_clicked(self):
        self.cancel_autostart()
        if self.selected_name():
            self.update_requested.emit(self.selected_name())

    def _toggle_contrast(self):
        self.cancel_autostart()
        self.btn_contrast.setChecked(theme.toggle_high_contrast(self.settings))

    def set_busy(self, text: str = ""):
        """Bloquea las acciones mientras se prepara o se cierra una sesión."""
        self._busy = bool(text)
        self.lbl_busy.setText(text)
        self._apply_busy_state()

    def _apply_busy_state(self):
        for widget in (self.btn_primary, self.btn_train, self.btn_update, self.btn_new, self.btn_delete, self.btn_voice):
            widget.setEnabled(not self._busy)
        self.profile_list.setEnabled(not self._busy)

    # ------------------------------------------------------- inicio automático
    def begin_autostart(self, name: str, seconds: int = AUTOSTART_SECONDS):
        """Cuenta atrás cancelable para que no haga falta el ratón para empezar."""
        self._autostart_name = name
        self._countdown = seconds
        self.autostart_banner.setVisible(True)
        self._update_autostart_text()
        self._autostart_timer.start()

    @property
    def autostart_active(self) -> bool:
        return self._autostart_timer.isActive()

    def cancel_autostart(self):
        self._autostart_timer.stop()
        self.autostart_banner.setVisible(False)

    def _update_autostart_text(self):
        self.lbl_autostart.setText(f"Iniciando el asistente con «{self._autostart_name}» en {self._countdown} s…")

    def _on_autostart_tick(self):
        self._countdown -= 1
        if self._countdown <= 0:
            self._start_autostart_now()
        else:
            self._update_autostart_text()

    def _start_autostart_now(self):
        self.cancel_autostart()
        self.start_requested.emit(self._autostart_name)

    def _on_autostart_toggled(self, enabled: bool):
        self.settings["autostart_last_profile"] = bool(enabled)
        save_settings(self.settings)
        if not enabled:
            self.cancel_autostart()

    def shutdown(self):
        self.cancel_autostart()
        for inspector in list(self._inspectors):
            inspector.wait(5000)

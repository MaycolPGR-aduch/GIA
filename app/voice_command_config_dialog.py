from __future__ import annotations

from copy import deepcopy

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
)

from .qt_helpers import MODERN_STYLE
from .voice_router import (
    BUILTIN_COMMAND_SPECS,
    DIRECT_ACTION_SPEC_BY_ID,
    build_custom_command_id,
    find_phrase_conflicts,
    format_macro_keys,
    get_direct_action_specs,
    normalize_phrase_list,
    normalize_voice_settings,
    parse_macro_keys,
    split_phrases_text,
)


class VoiceCommandConfigDialog(QDialog):
    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.settings = normalize_voice_settings(deepcopy(settings))
        self.current_builtin_id = None
        self.current_custom_id = None
        self.setWindowTitle("Configuracion global de comandos de voz")
        self.resize(980, 720)
        self.setStyleSheet(MODERN_STYLE)
        self._build_ui()
        self._refresh_builtin_list()
        self._refresh_custom_list()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel("Configuracion global de comandos de voz")
        header.setStyleSheet("font-size: 18px; font-weight: bold; color: #3b82f6;")
        layout.addWidget(header)

        self.check_voice_enabled = QCheckBox("Habilitar comandos de voz globales")
        self.check_voice_enabled.setChecked(bool(self.settings.get("voice_commands_enabled", True)))
        layout.addWidget(self.check_voice_enabled)

        model_note = QLabel("Modelo ASR activo: Whisper small")
        model_note.setStyleSheet("color: #94a3b8;")
        layout.addWidget(model_note)

        tabs = QTabWidget()
        tabs.addTab(self._build_builtin_tab(), "Comandos base")
        tabs.addTab(self._build_custom_tab(), "Personalizados")
        layout.addWidget(tabs, 1)

        actions = QHBoxLayout()
        btn_save = QPushButton("Guardar configuracion")
        btn_save.setObjectName("accentButton")
        btn_save.clicked.connect(self._save_and_close)
        actions.addWidget(btn_save)

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.clicked.connect(self.reject)
        actions.addWidget(btn_cancel)
        layout.addLayout(actions)

    def _build_builtin_tab(self):
        container = QDialog()
        layout = QHBoxLayout(container)

        self.builtin_list = QListWidget()
        self.builtin_list.currentItemChanged.connect(self._on_builtin_selected)
        layout.addWidget(self.builtin_list, 1)

        editor_box = QGroupBox("Editar aliases")
        editor_layout = QFormLayout(editor_box)
        self.lbl_builtin_desc = QLabel("Selecciona un comando base.")
        self.lbl_builtin_desc.setWordWrap(True)
        editor_layout.addRow("Descripcion:", self.lbl_builtin_desc)

        helper = QLabel(
            "Estas frases son exactamente las opciones que puedes decir para activar el comando base seleccionado. "
            "Escribe una frase por linea."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #94a3b8;")
        editor_layout.addRow(helper)

        self.txt_builtin_phrases = QTextEdit()
        self.txt_builtin_phrases.setPlaceholderText("Una frase por linea")
        self.txt_builtin_phrases.setMaximumHeight(180)
        editor_layout.addRow("Frases activas\n(que debes decir):", self.txt_builtin_phrases)

        btn_builtin_save = QPushButton("Guardar aliases")
        btn_builtin_save.clicked.connect(self._save_builtin_aliases)
        editor_layout.addRow(btn_builtin_save)

        btn_builtin_reset = QPushButton("Restaurar defaults")
        btn_builtin_reset.clicked.connect(self._reset_builtin_aliases)
        editor_layout.addRow(btn_builtin_reset)
        layout.addWidget(editor_box, 2)
        return container

    def _build_custom_tab(self):
        container = QDialog()
        layout = QHBoxLayout(container)

        self.custom_list = QListWidget()
        self.custom_list.currentItemChanged.connect(self._on_custom_selected)
        layout.addWidget(self.custom_list, 1)

        editor_box = QGroupBox("Editar comando personalizado")
        editor_layout = QFormLayout(editor_box)

        custom_help = QLabel(
            "Un comando personalizado tiene dos partes: la frase que diras y la accion que quieres ejecutar. "
            "Ejemplo: si escribes la frase 'copiar' y defines la macro 'Ctrl + C', cuando digas 'copiar' el sistema enviara Ctrl + C."
        )
        custom_help.setWordWrap(True)
        custom_help.setStyleSheet("color: #94a3b8;")
        editor_layout.addRow(custom_help)

        self.txt_custom_label = QLineEdit()
        self.txt_custom_label.setPlaceholderText("Nombre interno visible, por ejemplo: Copiar")
        editor_layout.addRow("Nombre visible:", self.txt_custom_label)

        self.txt_custom_phrases = QTextEdit()
        self.txt_custom_phrases.setPlaceholderText("Escribe aqui lo que vas a decir, por ejemplo:\ncopiar")
        self.txt_custom_phrases.setMaximumHeight(150)
        editor_layout.addRow("Frases activas\n(que debes decir):", self.txt_custom_phrases)

        self.cb_action_type = QComboBox()
        self.cb_action_type.addItem("Macro de teclado", "macro")
        self.cb_action_type.addItem("Accion directa", "direct_action")
        self.cb_action_type.currentIndexChanged.connect(self._update_custom_action_widgets)
        editor_layout.addRow("Tipo de accion:", self.cb_action_type)

        self.txt_macro_keys = QLineEdit()
        self.txt_macro_keys.setPlaceholderText("Ej. Ctrl + C")
        editor_layout.addRow("Macro\n(accion a ejecutar):", self.txt_macro_keys)

        self.cb_direct_action = QComboBox()
        for spec in get_direct_action_specs():
            self.cb_direct_action.addItem(spec["label"], spec["id"])
        editor_layout.addRow("Accion directa\n(accion a ejecutar):", self.cb_direct_action)

        self.lbl_custom_action_help = QLabel("")
        self.lbl_custom_action_help.setWordWrap(True)
        self.lbl_custom_action_help.setStyleSheet("color: #94a3b8;")
        editor_layout.addRow(self.lbl_custom_action_help)

        self.check_custom_enabled = QCheckBox("Comando habilitado")
        self.check_custom_enabled.setChecked(True)
        editor_layout.addRow(self.check_custom_enabled)

        buttons = QHBoxLayout()
        btn_new = QPushButton("Nuevo")
        btn_new.clicked.connect(self._new_custom_command)
        buttons.addWidget(btn_new)

        btn_save = QPushButton("Guardar comando")
        btn_save.clicked.connect(self._save_custom_command)
        buttons.addWidget(btn_save)

        btn_delete = QPushButton("Eliminar")
        btn_delete.setObjectName("dangerButton")
        btn_delete.clicked.connect(self._delete_custom_command)
        buttons.addWidget(btn_delete)
        editor_layout.addRow(buttons)

        layout.addWidget(editor_box, 2)
        self._update_custom_action_widgets()
        return container

    def _refresh_builtin_list(self):
        self.builtin_list.clear()
        overrides = self.settings.get("voice_builtin_alias_overrides", {})
        for spec in BUILTIN_COMMAND_SPECS:
            item = QListWidgetItem(spec["label"])
            item.setData(Qt.UserRole, spec["id"])
            if spec["id"] in overrides:
                item.setText(f"{spec['label']} (personalizado)")
            self.builtin_list.addItem(item)
        if self.builtin_list.count() and self.builtin_list.currentItem() is None:
            self.builtin_list.setCurrentRow(0)

    def _refresh_custom_list(self, select_id: str | None = None):
        self.custom_list.clear()
        selected_row = 0
        commands = self.settings.get("voice_custom_commands", [])
        for index, command in enumerate(commands):
            suffix = "" if command.get("enabled", True) else " (deshabilitado)"
            item = QListWidgetItem(f"{command['label']}{suffix}")
            item.setData(Qt.UserRole, command["id"])
            self.custom_list.addItem(item)
            if command["id"] == select_id:
                selected_row = index
        if self.custom_list.count():
            self.custom_list.setCurrentRow(selected_row)
        else:
            self._new_custom_command()

    def _on_builtin_selected(self, current, previous):
        if not current:
            self.current_builtin_id = None
            return
        command_id = current.data(Qt.UserRole)
        self.current_builtin_id = command_id
        spec = next(spec for spec in BUILTIN_COMMAND_SPECS if spec["id"] == command_id)
        phrases = self.settings.get("voice_builtin_alias_overrides", {}).get(command_id, spec["default_phrases"])
        self.lbl_builtin_desc.setText(spec["description"])
        self.txt_builtin_phrases.setPlainText("\n".join(phrases))

    def _save_builtin_aliases(self):
        if not self.current_builtin_id:
            return
        phrases = split_phrases_text(self.txt_builtin_phrases.toPlainText())
        if not phrases:
            QMessageBox.warning(self, "Alias invalidos", "Debes definir al menos una frase.")
            return
        conflicts = find_phrase_conflicts(phrases, self.settings, skip_kind="builtin", skip_id=self.current_builtin_id)
        if conflicts:
            QMessageBox.warning(
                self,
                "Frases duplicadas",
                "Estas frases ya pertenecen a otros comandos: " + ", ".join(conflicts),
            )
            return
        default_phrases = next(spec["default_phrases"] for spec in BUILTIN_COMMAND_SPECS if spec["id"] == self.current_builtin_id)
        overrides = self.settings.setdefault("voice_builtin_alias_overrides", {})
        if normalize_phrase_list(default_phrases) == phrases:
            overrides.pop(self.current_builtin_id, None)
        else:
            overrides[self.current_builtin_id] = phrases
        self._refresh_builtin_list()

    def _reset_builtin_aliases(self):
        if not self.current_builtin_id:
            return
        self.settings.setdefault("voice_builtin_alias_overrides", {}).pop(self.current_builtin_id, None)
        self._refresh_builtin_list()
        self._on_builtin_selected(self.builtin_list.currentItem(), None)

    def _new_custom_command(self):
        self.current_custom_id = None
        self.txt_custom_label.clear()
        self.txt_custom_phrases.clear()
        self.cb_action_type.setCurrentIndex(0)
        self.txt_macro_keys.clear()
        self.cb_direct_action.setCurrentIndex(0)
        self.check_custom_enabled.setChecked(True)

    def _on_custom_selected(self, current, previous):
        if not current:
            self._new_custom_command()
            return
        command_id = current.data(Qt.UserRole)
        self.current_custom_id = command_id
        command = next(item for item in self.settings.get("voice_custom_commands", []) if item["id"] == command_id)
        self.txt_custom_label.setText(command["label"])
        self.txt_custom_phrases.setPlainText("\n".join(command.get("spoken_phrases", [])))
        self.cb_action_type.setCurrentIndex(0 if command["action_type"] == "macro" else 1)
        self.txt_macro_keys.setText(format_macro_keys(command.get("macro_keys", [])))
        if command.get("direct_action_id") in DIRECT_ACTION_SPEC_BY_ID:
            self.cb_direct_action.setCurrentIndex(self.cb_direct_action.findData(command["direct_action_id"]))
        else:
            self.cb_direct_action.setCurrentIndex(0)
        self.check_custom_enabled.setChecked(bool(command.get("enabled", True)))
        self._update_custom_action_widgets()

    def _update_custom_action_widgets(self):
        is_macro = self.cb_action_type.currentData() == "macro"
        self.txt_macro_keys.setEnabled(is_macro)
        self.cb_direct_action.setEnabled(not is_macro)
        self.txt_macro_keys.setVisible(is_macro)
        self.cb_direct_action.setVisible(not is_macro)
        if is_macro:
            self.lbl_custom_action_help.setText(
                "Modo macro: escribe la combinacion de teclas que se ejecutara cuando digas una de las frases activas."
            )
        else:
            self.lbl_custom_action_help.setText(
                "Modo accion directa: selecciona una accion interna del sistema para que se ejecute cuando digas una de las frases activas."
            )

    def _save_custom_command(self):
        label = self.txt_custom_label.text().strip()
        if not label:
            QMessageBox.warning(self, "Nombre requerido", "Debes escribir un nombre visible para el comando.")
            return
        phrases = split_phrases_text(self.txt_custom_phrases.toPlainText())
        if not phrases:
            QMessageBox.warning(self, "Frases requeridas", "Debes escribir al menos una frase.")
            return
        conflicts = find_phrase_conflicts(phrases, self.settings, skip_kind="custom", skip_id=self.current_custom_id)
        if conflicts:
            QMessageBox.warning(
                self,
                "Frases duplicadas",
                "Estas frases ya pertenecen a otros comandos: " + ", ".join(conflicts),
            )
            return

        action_type = self.cb_action_type.currentData()
        macro_keys = []
        direct_action_id = None
        if action_type == "macro":
            macro_keys = parse_macro_keys(self.txt_macro_keys.text())
            if not macro_keys:
                QMessageBox.warning(self, "Macro invalida", "Debes definir una combinacion de teclas valida.")
                return
        else:
            direct_action_id = self.cb_direct_action.currentData()
            if direct_action_id not in DIRECT_ACTION_SPEC_BY_ID:
                QMessageBox.warning(self, "Accion invalida", "Debes seleccionar una accion directa valida.")
                return

        commands = self.settings.setdefault("voice_custom_commands", [])
        existing_ids = {command["id"] for command in commands if command["id"] != self.current_custom_id}
        command_id = self.current_custom_id or build_custom_command_id(label, existing_ids)
        payload = {
            "id": command_id,
            "label": label,
            "spoken_phrases": phrases,
            "action_type": action_type,
            "macro_keys": macro_keys,
            "direct_action_id": direct_action_id,
            "enabled": self.check_custom_enabled.isChecked(),
        }

        replaced = False
        for index, command in enumerate(commands):
            if command["id"] == command_id:
                commands[index] = payload
                replaced = True
                break
        if not replaced:
            commands.append(payload)
        self.settings = normalize_voice_settings(self.settings)
        self.current_custom_id = command_id
        self._refresh_custom_list(select_id=command_id)

    def _delete_custom_command(self):
        if not self.current_custom_id:
            self._new_custom_command()
            return
        commands = [
            command
            for command in self.settings.get("voice_custom_commands", [])
            if command["id"] != self.current_custom_id
        ]
        self.settings["voice_custom_commands"] = commands
        self.current_custom_id = None
        self._refresh_custom_list()

    def _save_and_close(self):
        self.settings["voice_commands_enabled"] = self.check_voice_enabled.isChecked()
        self.settings = normalize_voice_settings(self.settings)
        self.accept()

    def get_updated_settings(self) -> dict:
        return normalize_voice_settings(self.settings)

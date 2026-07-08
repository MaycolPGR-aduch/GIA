from __future__ import annotations

import webbrowser

import pyautogui

from .voice_router import DIRECT_ACTION_SPEC_BY_ID, format_macro_keys


class ActionService:
    def __init__(
        self,
        logger,
        *,
        on_voice_listener,
        on_toggle_cursor,
        on_recenter,
        on_toggle_pause,
        on_set_cursor,
        on_quit,
        main_gui_interface,
        audio_endpoint,
    ):
        self.logger = logger
        self.on_voice_listener = on_voice_listener
        self.on_toggle_cursor = on_toggle_cursor
        self.on_recenter = on_recenter
        self.on_toggle_pause = on_toggle_pause
        self.on_set_cursor = on_set_cursor
        self.on_quit = on_quit
        self.main_gui_interface = main_gui_interface
        self.audio_endpoint = audio_endpoint

    def execute_gesture_action(self, gesture_id: str, app_state) -> None:
        if gesture_id == "left_blink_intent" and app_state == "ready":
            pyautogui.click()
        elif gesture_id == "right_blink_intent" and app_state == "ready":
            pyautogui.click(button="right")
        elif gesture_id == "both_eyes_closed_intent":
            self.on_voice_listener()
        elif gesture_id == "mouth_open_hold":
            self.on_toggle_cursor()
        elif gesture_id == "smile":
            self.on_recenter()
        elif gesture_id == "brows_up":
            self.logger.log("gesture", "Gesto deshabilitado ignorado", gesture_id=gesture_id)
        elif gesture_id == "confirm":
            self.logger.log("gesture", "Confirmacion detectada", gesture_id=gesture_id)

    def execute_voice_command(self, command_id: str, app_state: str) -> None:
        self.execute_builtin_voice_command(command_id, app_state)

    def execute_builtin_voice_command(self, command_id: str, app_state: str) -> None:
        self.execute_direct_action(command_id, app_state)

    def execute_custom_voice_command(self, command_config: dict, app_state: str) -> None:
        label = command_config.get("label", command_config.get("id", "Comando personalizado"))
        action_type = command_config.get("action_type")
        if action_type == "macro":
            macro_keys = command_config.get("macro_keys", [])
            self._execute_macro(label, macro_keys)
        elif action_type == "direct_action":
            direct_action_id = command_config.get("direct_action_id")
            if direct_action_id:
                self.logger.log(
                    "voice_custom",
                    "Accion directa personalizada ejecutada",
                    label=label,
                    direct_action_id=direct_action_id,
                )
                self.execute_direct_action(direct_action_id, app_state)

    def execute_direct_action(self, direct_action_id: str, app_state: str) -> None:
        if direct_action_id == "pause" and app_state != "paused":
            self.on_toggle_pause()
        elif direct_action_id == "resume" and app_state == "paused":
            self.on_toggle_pause()
        elif direct_action_id == "recenter":
            self.on_recenter()
        elif direct_action_id == "cursor_on":
            self.on_set_cursor(True)
        elif direct_action_id == "cursor_off":
            self.on_set_cursor(False)
        elif direct_action_id == "guide" and self.main_gui_interface:
            self.main_gui_interface.show_guide_dialog()
        elif direct_action_id == "compact_ui" and self.main_gui_interface:
            self._invoke_ui(self.main_gui_interface.enter_compact_mode)
        elif direct_action_id == "expand_ui" and self.main_gui_interface:
            self._invoke_ui(self.main_gui_interface.exit_compact_mode)
        elif direct_action_id == "quit":
            self.on_quit()
        elif direct_action_id.startswith("volume_"):
            self._set_volume(int(direct_action_id.split("_")[1]))
        elif direct_action_id == "open_gmail":
            webbrowser.open("https://gmail.com")
        elif direct_action_id == "open_facebook":
            webbrowser.open("https://facebook.com")
        elif direct_action_id == "open_whatsapp":
            webbrowser.open("https://web.whatsapp.com")
        elif direct_action_id == "open_youtube":
            webbrowser.open("https://youtube.com")
        elif direct_action_id == "start_dictation":
            self._execute_macro("Modo dictado", ["win", "h"])

    def _invoke_ui(self, callback):
        if hasattr(self.main_gui_interface, "run_on_ui_thread"):
            self.main_gui_interface.run_on_ui_thread(callback)
        else:
            from PySide6.QtCore import QTimer

            QTimer.singleShot(0, callback)

    def _execute_macro(self, label: str, macro_keys: list[str]) -> None:
        cleaned = [str(key).strip().lower() for key in macro_keys if str(key).strip()]
        if not cleaned:
            self.logger.log("voice_custom", "Macro ignorada por falta de teclas", label=label)
            return
        pyautogui.hotkey(*cleaned)
        self.logger.log(
            "voice_custom",
            "Macro ejecutada",
            label=label,
            macro_keys=cleaned,
            macro_display=format_macro_keys(cleaned),
        )

    def _set_volume(self, percent: int) -> None:
        if self.audio_endpoint is None:
            self.logger.log("audio", "Backend de volumen no disponible", percent=percent)
            return
        try:
            self.audio_endpoint.SetMasterVolumeLevelScalar(percent / 100.0, None)
            self.logger.log("audio", "Cambio de volumen", percent=percent)
        except Exception as exc:
            self.logger.log("error", "No se pudo cambiar volumen", error=str(exc))


def get_direct_action_label(direct_action_id: str) -> str:
    spec = DIRECT_ACTION_SPEC_BY_ID.get(direct_action_id)
    return spec["label"] if spec else direct_action_id

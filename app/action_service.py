from __future__ import annotations

import webbrowser


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
            import pyautogui

            pyautogui.click()
        elif gesture_id == "right_blink_intent" and app_state == "ready":
            import pyautogui

            pyautogui.click(button="right")
        elif gesture_id == "both_eyes_closed_intent":
            self.on_voice_listener()
        elif gesture_id == "mouth_open_hold":
            self.on_toggle_cursor()
        elif gesture_id == "smile":
            self.on_recenter()
        elif gesture_id == "brows_up":
            self.on_toggle_pause()
        elif gesture_id == "confirm":
            self.logger.log("gesture", "Confirmación detectada", gesture_id=gesture_id)

    def execute_voice_command(self, command_id: str, app_state: str) -> None:
        if command_id == "pause" and app_state != "paused":
            self.on_toggle_pause()
        elif command_id == "resume" and app_state == "paused":
            self.on_toggle_pause()
        elif command_id == "recenter":
            self.on_recenter()
        elif command_id == "cursor_on":
            self.on_set_cursor(True)
        elif command_id == "cursor_off":
            self.on_set_cursor(False)
        elif command_id == "guide" and self.main_gui_interface:
            self.main_gui_interface.show_guide_dialog()
        elif command_id == "compact_ui" and self.main_gui_interface:
            if hasattr(self.main_gui_interface, "root"):
                self.main_gui_interface.root.after(0, self.main_gui_interface.enter_compact_mode)
            else:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, self.main_gui_interface.enter_compact_mode)
        elif command_id == "expand_ui" and self.main_gui_interface:
            if hasattr(self.main_gui_interface, "root"):
                self.main_gui_interface.root.after(0, self.main_gui_interface.exit_compact_mode)
            else:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, self.main_gui_interface.exit_compact_mode)
        elif command_id == "quit":
            self.on_quit()
        elif command_id.startswith("volume_"):
            self._set_volume(int(command_id.split("_")[1]))
        elif command_id == "open_gmail":
            webbrowser.open("https://gmail.com")
        elif command_id == "open_facebook":
            webbrowser.open("https://facebook.com")
        elif command_id == "open_whatsapp":
            webbrowser.open("https://web.whatsapp.com")
        elif command_id == "open_youtube":
            webbrowser.open("https://youtube.com")

    def _set_volume(self, percent: int) -> None:
        if self.audio_endpoint is None:
            self.logger.log("audio", "Backend de volumen no disponible", percent=percent)
            return
        try:
            self.audio_endpoint.SetMasterVolumeLevelScalar(percent / 100.0, None)
            self.logger.log("audio", "Cambio de volumen", percent=percent)
        except Exception as exc:
            self.logger.log("error", "No se pudo cambiar volumen", error=str(exc))

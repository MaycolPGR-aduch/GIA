import os
import traceback

import absl.logging
import customtkinter as ctk

from app.assistive_controls import AssistiveController
from app.config_store import ensure_app_dirs
from app.launcher_gui import LauncherGUI
from app.runtime_gui import RuntimeGUI


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
absl.logging.set_verbosity(absl.logging.ERROR)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class GIAApplication:
    def __init__(self):
        ensure_app_dirs()
        self.root = ctk.CTk()
        self.controller = None
        self.runtime_gui = None
        self.launcher = None

    def run(self):
        self.show_launcher()
        self.root.mainloop()

    def show_launcher(self):
        self.launcher = LauncherGUI(self.root, self.start_runtime)

    def start_runtime(self, profile: dict, settings: dict):
        for child in self.root.winfo_children():
            child.destroy()
        self.runtime_gui = RuntimeGUI(self.root)
        self.controller = AssistiveController(profile, settings, main_gui_interface=self.runtime_gui)
        self.runtime_gui.set_controller(self.controller)
        self.controller.start()


def main():
    app = GIAApplication()
    app.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MAIN: Error no capturado en main: {exc}")
        traceback.print_exc()
    finally:
        print("MAIN: Programa terminado.")

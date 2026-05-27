from __future__ import annotations

import tkinter as tk
import webbrowser

import customtkinter as ctk
from PIL import Image, ImageTk

from .gesture_catalog import GESTURE_CATALOG, VOICE_COMMAND_HELP


class RuntimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("GIA v2 - Runtime Asistivo")
        self.root.geometry("1280x840")
        self.root.minsize(1100, 720)
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.controller = None
        self.image_ref = None
        self.compact_image_ref = None
        self.is_compact_mode = False
        self.expanded_geometry = "1280x840"
        self.compact_window = None
        self.compact_status_label = None
        self.compact_voice_label = None
        self.compact_event_log = None
        self.compact_video_label = None

        self._build_layout()

    def set_controller(self, controller):
        self.controller = controller

    def _build_layout(self):
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self.header = ctk.CTkFrame(self.root, corner_radius=0, fg_color="#183153")
        self.header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.header.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            self.header,
            text="GIA v2 - Control Facial Asistivo",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="white",
        )
        self.title_label.grid(row=0, column=0, padx=24, pady=16, sticky="w")

        self.state_badge = ctk.CTkLabel(
            self.header,
            text="Estado: launcher",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#183153",
            fg_color="#f4d35e",
            corner_radius=14,
            padx=16,
            pady=8,
        )
        self.state_badge.grid(row=0, column=1, padx=16, pady=16, sticky="e")

        self.left_frame = ctk.CTkFrame(self.root, fg_color="#101820")
        self.left_frame.grid(row=1, column=0, sticky="nsew", padx=(18, 10), pady=18)
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            self.left_frame,
            text="Listo para iniciar sesion asistiva.",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#f5f7fa",
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))

        self.video_label = tk.Label(self.left_frame, bg="#000000", bd=0, highlightthickness=0)
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))

        self.right_frame = ctk.CTkFrame(self.root, fg_color="#f4f6f8")
        self.right_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 18), pady=18)
        self.right_frame.grid_rowconfigure(3, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)

        self.metric_frame = ctk.CTkFrame(self.right_frame, fg_color="white")
        self.metric_frame.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))
        self.metric_frame.grid_columnconfigure((0, 1), weight=1)

        self.gesture_label = ctk.CTkLabel(
            self.metric_frame,
            text="Gesto: -",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#183153",
        )
        self.gesture_label.grid(row=0, column=0, padx=14, pady=(14, 4), sticky="w")

        self.confidence_label = ctk.CTkLabel(
            self.metric_frame,
            text="Confianza: -",
            font=ctk.CTkFont(size=16),
            text_color="#5c677d",
        )
        self.confidence_label.grid(row=1, column=0, padx=14, pady=(0, 10), sticky="w")

        self.face_label = ctk.CTkLabel(
            self.metric_frame,
            text="Rostro: -",
            font=ctk.CTkFont(size=16),
            text_color="#5c677d",
        )
        self.face_label.grid(row=0, column=1, padx=14, pady=(14, 4), sticky="w")

        self.mode_label = ctk.CTkLabel(
            self.metric_frame,
            text="Modo: -",
            font=ctk.CTkFont(size=16),
            text_color="#5c677d",
        )
        self.mode_label.grid(row=1, column=1, padx=14, pady=(0, 10), sticky="w")

        self.cursor_label = ctk.CTkLabel(
            self.metric_frame,
            text="Cursor: -",
            font=ctk.CTkFont(size=16),
            text_color="#5c677d",
        )
        self.cursor_label.grid(row=2, column=0, padx=14, pady=(0, 10), sticky="w")

        self.voice_state_label = ctk.CTkLabel(
            self.metric_frame,
            text="Voz: en espera",
            font=ctk.CTkFont(size=16),
            text_color="#5c677d",
        )
        self.voice_state_label.grid(row=2, column=1, padx=14, pady=(0, 10), sticky="w")

        self.voice_text_label = ctk.CTkLabel(
            self.metric_frame,
            text="Texto reconocido: -",
            font=ctk.CTkFont(size=13),
            text_color="#334e68",
            justify="left",
            wraplength=320,
        )
        self.voice_text_label.grid(row=3, column=0, columnspan=2, padx=14, pady=(0, 6), sticky="w")

        self.voice_command_label = ctk.CTkLabel(
            self.metric_frame,
            text="Comando interpretado: -",
            font=ctk.CTkFont(size=13),
            text_color="#334e68",
            justify="left",
            wraplength=320,
        )
        self.voice_command_label.grid(row=4, column=0, columnspan=2, padx=14, pady=(0, 14), sticky="w")

        self.actions_frame = ctk.CTkFrame(self.right_frame, fg_color="white")
        self.actions_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=10)
        self.actions_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(self.actions_frame, text="Pausar / Reanudar", command=self._toggle_pause, height=48).grid(
            row=0, column=0, padx=10, pady=10, sticky="ew"
        )
        ctk.CTkButton(self.actions_frame, text="Recentrar", command=self._recenter, height=48).grid(
            row=0, column=1, padx=10, pady=10, sticky="ew"
        )
        ctk.CTkButton(
            self.actions_frame,
            text="Activar / Congelar cursor",
            command=self._toggle_cursor,
            height=48,
        ).grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
        ctk.CTkButton(self.actions_frame, text="Modo compacto", command=self.enter_compact_mode, height=48).grid(
            row=1, column=1, padx=10, pady=(0, 10), sticky="ew"
        )
        ctk.CTkButton(self.actions_frame, text="Guia rapida", command=self.show_guide_dialog, height=48).grid(
            row=2, column=0, padx=10, pady=(0, 10), sticky="ew"
        )
        ctk.CTkButton(self.actions_frame, text="Cerrar sistema", command=self._quit, height=48, fg_color="#aa2e25").grid(
            row=2, column=1, padx=10, pady=(0, 10), sticky="ew"
        )

        self.event_log = ctk.CTkTextbox(self.right_frame, height=160, fg_color="#0d1b2a", text_color="#f1faee")
        self.event_log.grid(row=2, column=0, sticky="ew", padx=16, pady=10)
        self.event_log.insert("end", "Eventos recientes:\n")
        self.event_log.configure(state="disabled")

        self.guide_frame = ctk.CTkScrollableFrame(self.right_frame, fg_color="white")
        self.guide_frame.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 16))
        ctk.CTkLabel(
            self.guide_frame,
            text="Resumen de gestos y voz",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#183153",
        ).pack(anchor="w", padx=8, pady=(10, 6))

        for gesture in GESTURE_CATALOG:
            ctk.CTkLabel(
                self.guide_frame,
                text=f"{gesture['title']} -> {gesture['action']}",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#1f2933",
                anchor="w",
            ).pack(fill="x", padx=8, pady=(8, 0))
            ctk.CTkLabel(
                self.guide_frame,
                text=f"{gesture['duration_ms']} ms | {gesture['warning']}",
                font=ctk.CTkFont(size=12),
                text_color="#52606d",
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=8, pady=(0, 2))

        ctk.CTkLabel(
            self.guide_frame,
            text="Comandos de voz",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#183153",
        ).pack(anchor="w", padx=8, pady=(12, 6))
        for command, description in VOICE_COMMAND_HELP:
            ctk.CTkLabel(
                self.guide_frame,
                text=f"{command}: {description}",
                font=ctk.CTkFont(size=12),
                text_color="#52606d",
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=8, pady=2)

    def update_video_feed(self, frame_rgb, compact_frame_rgb=None):
        try:
            if not self.is_compact_mode:
                image = Image.fromarray(frame_rgb)
                image.thumbnail((860, 640))
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=photo)
                self.image_ref = photo

            if compact_frame_rgb is not None and self.compact_video_label is not None:
                compact_image = Image.fromarray(compact_frame_rgb)
                compact_image.thumbnail((240, 135))
                compact_photo = ImageTk.PhotoImage(image=compact_image)
                self.compact_video_label.configure(image=compact_photo)
                self.compact_image_ref = compact_photo
        except Exception as exc:
            self.append_event(f"Error actualizando video: {exc}")

    def update_status(self, payload: dict):
        status_text = payload.get("status_text", "Sin estado")
        self.status_label.configure(text=status_text)
        self.state_badge.configure(text=f"Estado: {payload.get('state', '-')}")
        self.gesture_label.configure(text=f"Gesto: {payload.get('gesture', '-')}")
        self.confidence_label.configure(text=f"Confianza: {payload.get('confidence', '-')}")
        self.face_label.configure(text=f"Rostro: {payload.get('face', '-')}")
        self.mode_label.configure(text=f"Modo: {payload.get('mode', '-')}")
        self.cursor_label.configure(text=f"Cursor: {payload.get('cursor', '-')}")
        self.voice_state_label.configure(text=f"Voz: {payload.get('voice_state', 'en espera')}")
        self.voice_text_label.configure(text=f"Texto reconocido: {payload.get('voice_text', '-')}")
        self.voice_command_label.configure(text=f"Comando interpretado: {payload.get('voice_command', '-')}")
        if self.compact_status_label is not None:
            self.compact_status_label.configure(text=status_text)
        if self.compact_voice_label is not None:
            self.compact_voice_label.configure(
                text=(
                    f"Voz: {payload.get('voice_state', 'en espera')} | "
                    f"Texto: {payload.get('voice_text', '-')} | "
                    f"Comando: {payload.get('voice_command', '-')}"
                )
            )

    def append_event(self, text: str):
        widgets = [self.event_log]
        if self.compact_event_log is not None:
            widgets.append(self.compact_event_log)
        for widget in widgets:
            widget.configure(state="normal")
            widget.insert("end", f"- {text}\n")
            widget.see("end")
            widget.configure(state="disabled")

    def show_guide_dialog(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Guia rapida de GIA")
        dialog.geometry("720x620")
        scroll = ctk.CTkScrollableFrame(dialog)
        scroll.pack(fill="both", expand=True, padx=18, pady=18)
        for gesture in GESTURE_CATALOG:
            ctk.CTkLabel(
                scroll,
                text=gesture["title"],
                font=ctk.CTkFont(size=17, weight="bold"),
                anchor="w",
            ).pack(fill="x", pady=(10, 0))
            ctk.CTkLabel(
                scroll,
                text=f"Accion: {gesture['action']}\nDuracion: {gesture['duration_ms']} ms\nAdvertencia: {gesture['warning']}",
                justify="left",
                anchor="w",
            ).pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(
            scroll,
            text="Accesos web integrados",
            font=ctk.CTkFont(size=17, weight="bold"),
        ).pack(anchor="w", pady=(14, 4))
        for name, url in [
            ("Gmail", "https://gmail.com"),
            ("Facebook", "https://facebook.com"),
            ("WhatsApp", "https://web.whatsapp.com"),
            ("YouTube", "https://youtube.com"),
        ]:
            ctk.CTkButton(scroll, text=name, command=lambda link=url: webbrowser.open(link)).pack(
                anchor="w", pady=4
            )

    def enter_compact_mode(self):
        if self.is_compact_mode:
            return
        self.expanded_geometry = self.root.geometry()
        self.root.withdraw()
        self._create_compact_window()
        self.is_compact_mode = True
        self.append_event("Interfaz cambiada a modo compacto.")

    def exit_compact_mode(self):
        if not self.is_compact_mode:
            return
        self._destroy_compact_window()
        self.root.deiconify()
        self.root.geometry(self.expanded_geometry)
        self.is_compact_mode = False
        self.append_event("Interfaz restaurada a modo completo.")

    def _create_compact_window(self):
        if self.compact_window is not None and self.compact_window.winfo_exists():
            return

        compact_window = ctk.CTkToplevel(self.root)
        compact_window.title("GIA v2 - Compacto")
        compact_window.resizable(False, False)
        compact_window.attributes("-topmost", True)
        compact_window.protocol("WM_DELETE_WINDOW", self.exit_compact_mode)
        compact_window.grid_columnconfigure(0, weight=1)

        width = 980
        height = 230
        x = max((self.root.winfo_screenwidth() - width) // 2, 0)
        y = 8
        compact_window.geometry(f"{width}x{height}+{x}+{y}")

        shell = ctk.CTkFrame(compact_window, fg_color="#0f172a", corner_radius=16)
        shell.pack(fill="both", expand=True, padx=8, pady=8)
        shell.grid_columnconfigure(0, weight=0)
        shell.grid_columnconfigure(1, weight=1)
        shell.grid_columnconfigure(2, weight=0)

        preview_frame = ctk.CTkFrame(shell, fg_color="#020617", corner_radius=12)
        preview_frame.grid(row=0, column=0, rowspan=3, sticky="nsw", padx=(16, 8), pady=12)
        self.compact_video_label = tk.Label(preview_frame, bg="#000000", bd=0, highlightthickness=0)
        self.compact_video_label.pack(padx=8, pady=8)

        self.compact_status_label = ctk.CTkLabel(
            shell,
            text="Modo compacto activo",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="white",
        )
        self.compact_status_label.grid(row=0, column=1, sticky="w", padx=(8, 16), pady=(12, 4))

        ctk.CTkButton(
            shell,
            text="Volver a interfaz grande",
            command=self.exit_compact_mode,
            width=210,
        ).grid(row=0, column=2, sticky="e", padx=16, pady=(12, 4))

        self.compact_voice_label = ctk.CTkLabel(
            shell,
            text="Voz: en espera | Texto: - | Comando: -",
            font=ctk.CTkFont(size=13),
            text_color="#cbd5e1",
            justify="left",
            wraplength=640,
        )
        self.compact_voice_label.grid(row=1, column=1, columnspan=2, sticky="w", padx=(8, 16), pady=(0, 8))

        self.compact_event_log = ctk.CTkTextbox(
            shell,
            height=92,
            fg_color="#111827",
            text_color="#f8fafc",
        )
        self.compact_event_log.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(8, 16), pady=(0, 12))
        self.compact_event_log.insert("end", "Comandos ejecutados:\n")
        self.compact_event_log.configure(state="disabled")

        compact_window.update_idletasks()
        self.compact_window = compact_window

    def _destroy_compact_window(self):
        if self.compact_window is not None and self.compact_window.winfo_exists():
            self.compact_window.destroy()
        self.compact_window = None
        self.compact_status_label = None
        self.compact_voice_label = None
        self.compact_event_log = None
        self.compact_video_label = None
        self.compact_image_ref = None

    def _toggle_pause(self):
        if self.controller:
            self.controller.toggle_pause()

    def _recenter(self):
        if self.controller:
            self.controller.recenter()

    def _toggle_cursor(self):
        if self.controller:
            self.controller.toggle_cursor_control()

    def _quit(self):
        if self.controller:
            self.controller.stop()
        self._destroy_compact_window()
        self.root.quit()
        self.root.destroy()

    def _on_closing(self):
        self._quit()

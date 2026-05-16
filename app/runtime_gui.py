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

        self._build_layout()

    def set_controller(self, controller):
        self.controller = controller

    def _build_layout(self):
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self.root, corner_radius=0, fg_color="#183153")
        header.grid(row=0, column=0, columnspan=2, sticky="nsew")
        header.grid_columnconfigure(0, weight=1)

        self.title_label = ctk.CTkLabel(
            header,
            text="GIA v2 - Control Facial Asistivo",
            font=ctk.CTkFont(size=26, weight="bold"),
            text_color="white",
        )
        self.title_label.grid(row=0, column=0, padx=24, pady=16, sticky="w")

        self.state_badge = ctk.CTkLabel(
            header,
            text="Estado: launcher",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color="#183153",
            fg_color="#f4d35e",
            corner_radius=14,
            padx=16,
            pady=8,
        )
        self.state_badge.grid(row=0, column=1, padx=16, pady=16, sticky="e")

        left = ctk.CTkFrame(self.root, fg_color="#101820")
        left.grid(row=1, column=0, sticky="nsew", padx=(18, 10), pady=18)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            left,
            text="Listo para iniciar sesión asistiva.",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color="#f5f7fa",
            anchor="w",
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))

        self.video_label = tk.Label(left, bg="#000000", bd=0, highlightthickness=0)
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))

        right = ctk.CTkFrame(self.root, fg_color="#f4f6f8")
        right.grid(row=1, column=1, sticky="nsew", padx=(10, 18), pady=18)
        right.grid_rowconfigure(3, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.metric_frame = ctk.CTkFrame(right, fg_color="white")
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
        self.confidence_label.grid(row=1, column=0, padx=14, pady=(0, 14), sticky="w")

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
        self.mode_label.grid(row=1, column=1, padx=14, pady=(0, 14), sticky="w")

        actions = ctk.CTkFrame(right, fg_color="white")
        actions.grid(row=1, column=0, sticky="ew", padx=16, pady=10)
        actions.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(actions, text="Pausar / Reanudar", command=self._toggle_pause, height=48).grid(
            row=0, column=0, padx=10, pady=10, sticky="ew"
        )
        ctk.CTkButton(actions, text="Recentrar", command=self._recenter, height=48).grid(
            row=0, column=1, padx=10, pady=10, sticky="ew"
        )
        ctk.CTkButton(actions, text="Guía rápida", command=self.show_guide_dialog, height=48).grid(
            row=1, column=0, padx=10, pady=(0, 10), sticky="ew"
        )
        ctk.CTkButton(actions, text="Cerrar sistema", command=self._quit, height=48, fg_color="#aa2e25").grid(
            row=1, column=1, padx=10, pady=(0, 10), sticky="ew"
        )

        self.event_log = ctk.CTkTextbox(right, height=160, fg_color="#0d1b2a", text_color="#f1faee")
        self.event_log.grid(row=2, column=0, sticky="ew", padx=16, pady=10)
        self.event_log.insert("end", "Eventos recientes:\n")
        self.event_log.configure(state="disabled")

        guide = ctk.CTkScrollableFrame(right, fg_color="white")
        guide.grid(row=3, column=0, sticky="nsew", padx=16, pady=(0, 16))
        ctk.CTkLabel(
            guide,
            text="Resumen de gestos y voz",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#183153",
        ).pack(anchor="w", padx=8, pady=(10, 6))

        for gesture in GESTURE_CATALOG:
            ctk.CTkLabel(
                guide,
                text=f"{gesture['title']} -> {gesture['action']}",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color="#1f2933",
                anchor="w",
            ).pack(fill="x", padx=8, pady=(8, 0))
            ctk.CTkLabel(
                guide,
                text=f"{gesture['duration_ms']} ms | {gesture['warning']}",
                font=ctk.CTkFont(size=12),
                text_color="#52606d",
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=8, pady=(0, 2))

        ctk.CTkLabel(
            guide,
            text="Comandos de voz",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#183153",
        ).pack(anchor="w", padx=8, pady=(12, 6))
        for command, description in VOICE_COMMAND_HELP:
            ctk.CTkLabel(
                guide,
                text=f"{command}: {description}",
                font=ctk.CTkFont(size=12),
                text_color="#52606d",
                anchor="w",
                justify="left",
            ).pack(fill="x", padx=8, pady=2)

    def update_video_feed(self, frame_rgb):
        try:
            image = Image.fromarray(frame_rgb)
            image.thumbnail((860, 640))
            photo = ImageTk.PhotoImage(image=image)
            self.video_label.configure(image=photo)
            self.image_ref = photo
        except Exception as exc:
            self.append_event(f"Error actualizando video: {exc}")

    def update_status(self, payload: dict):
        self.status_label.configure(text=payload.get("status_text", "Sin estado"))
        self.state_badge.configure(text=f"Estado: {payload.get('state', '-')}")
        self.gesture_label.configure(text=f"Gesto: {payload.get('gesture', '-')}")
        self.confidence_label.configure(text=f"Confianza: {payload.get('confidence', '-')}")
        self.face_label.configure(text=f"Rostro: {payload.get('face', '-')}")
        self.mode_label.configure(text=f"Modo: {payload.get('mode', '-')}")

    def append_event(self, text: str):
        self.event_log.configure(state="normal")
        self.event_log.insert("end", f"- {text}\n")
        self.event_log.see("end")
        self.event_log.configure(state="disabled")

    def show_guide_dialog(self):
        dialog = ctk.CTkToplevel(self.root)
        dialog.title("Guía rápida de GIA")
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
                text=f"Acción: {gesture['action']}\nDuración: {gesture['duration_ms']} ms\nAdvertencia: {gesture['warning']}",
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

    def _toggle_pause(self):
        if self.controller:
            self.controller.toggle_pause()

    def _recenter(self):
        if self.controller:
            self.controller.recenter()

    def _quit(self):
        if self.controller:
            self.controller.stop()
        self.root.quit()
        self.root.destroy()

    def _on_closing(self):
        self._quit()

from __future__ import annotations

import time

import customtkinter as ctk
import cv2
import sounddevice as sd

from .calibration_service import CalibrationService
from .config_store import (
    build_default_profile,
    list_profiles,
    load_profile,
    load_settings,
    parse_camera_resolution,
    save_profile,
    save_settings,
)
from .gesture_catalog import GESTURE_CATALOG, NEUTRAL_GESTURE_META, VOICE_COMMAND_HELP
from .landmark_provider import LandmarkProvider
from .training_service import TrainingService


READINESS_COLORS = {
    "sin iniciar": "#94a3b8",
    "insuficiente": "#f59e0b",
    "listo para entrenar": "#10b981",
    "entrenado": "#2563eb",
}

CAMERA_RESOLUTION_OPTIONS = [
    "640x480",
    "800x600",
    "960x540",
    "1280x720",
    "1920x1080",
]


class LauncherGUI:
    CAPTURE_PREP_SECONDS = 3.0

    def __init__(self, root, on_start_callback):
        self.root = root
        self.on_start_callback = on_start_callback
        self.settings = load_settings()
        self.profile_names = list_profiles()
        self.current_profile_name = self.profile_names[0]
        self.profile = load_profile(self.current_profile_name)
        self.window_size = int(self.settings.get("gesture_window_size", 12))
        self.training_service = TrainingService(self.current_profile_name, self.window_size)
        self.calibration_service = CalibrationService(self.window_size)

        self.capture_target: str | None = None
        self.capture_phase = "idle"
        self.capture_countdown_deadline = 0.0
        self.capture_deadline = 0.0
        self.capture_duration_seconds = 0.0
        self.latest_sample = None
        self.training_cards: dict[str, dict] = {}
        self.training_statuses = {}

        self.root.title("GIA v2 - Launcher")
        self.root.geometry("1400x920")
        self.root.minsize(1240, 780)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.cap = cv2.VideoCapture(int(self.settings.get("camera_index", 0)))
        self._apply_camera_resolution(self.cap)
        self.provider = LandmarkProvider(fps=int(self.settings.get("fps", 15)))

        self._build_layout()
        self._populate_guide()
        self._switch_profile(self.current_profile_name, refresh_menu=False)
        self._schedule_preview()

    def _build_layout(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        shell = ctk.CTkFrame(self.root, fg_color="#0f172a")
        shell.grid(row=0, column=0, sticky="nsew", padx=18, pady=18)
        shell.grid_rowconfigure(3, weight=1)
        shell.grid_columnconfigure(0, weight=1)

        hero = ctk.CTkFrame(shell, fg_color="#10233f", corner_radius=18)
        hero.grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 12))
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            hero,
            text="GIA v2 Launcher",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="white",
        ).grid(row=0, column=0, sticky="w", padx=20, pady=(18, 6))
        ctk.CTkLabel(
            hero,
            text="Configura el perfil, revisa el estado del sistema y entrena los gestos antes de iniciar el control facial asistido.",
            font=ctk.CTkFont(size=15),
            text_color="#cbd5e1",
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=20, pady=(0, 14))

        self.capture_monitor_label = ctk.CTkLabel(
            hero,
            text="Cámara en espera. La captura se activará solo cuando inicies un gesto.",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#fde68a",
            justify="left",
            wraplength=420,
        )
        self.capture_monitor_label.grid(row=0, column=1, rowspan=2, sticky="e", padx=20, pady=(18, 14))

        summary_strip = ctk.CTkFrame(shell, fg_color="#0f172a")
        summary_strip.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))
        summary_strip.grid_columnconfigure((0, 1, 2), weight=1)

        self.profile_summary_card = self._build_summary_card(
            summary_strip,
            column=0,
            title="Perfil activo",
            text="Sin perfil cargado",
        )
        self.camera_summary_card = self._build_summary_card(
            summary_strip,
            column=1,
            title="Cámara",
            text="Resolución pendiente",
        )
        self.flow_summary_card = self._build_summary_card(
            summary_strip,
            column=2,
            title="Flujo recomendado",
            text="1. Perfil  2. Ajustes  3. Diagnóstico  4. Calibración  5. Iniciar",
        )

        self.tabview = ctk.CTkTabview(shell, fg_color="#e5e7eb")
        self.tabview.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        for tab_name in ("Perfiles", "Guia", "Diagnostico", "Ajustes", "Calibracion"):
            self.tabview.add(tab_name)

        self._build_profile_tab()
        self._build_diagnostics_tab()
        self._build_settings_tab()
        self._build_calibration_tab()

        start_bar = ctk.CTkFrame(shell, fg_color="#111827")
        start_bar.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))
        self.launch_status = ctk.CTkLabel(
            start_bar,
            text="Carga o captura un dataset suficiente antes de iniciar.",
            text_color="#f8fafc",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.launch_status.pack(side="left", padx=14, pady=12)
        self.start_button = ctk.CTkButton(
            start_bar,
            text="Iniciar sistema asistivo",
            command=self._start_runtime,
            state="disabled",
            width=220,
            height=42,
        )
        self.start_button.pack(side="right", padx=14, pady=12)

    def _build_summary_card(self, parent, *, column: int, title: str, text: str):
        card = ctk.CTkFrame(parent, fg_color="#f8fafc")
        card.grid(row=0, column=column, sticky="ew", padx=6)
        card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            card,
            text=title,
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#183153",
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(12, 4))
        body = ctk.CTkLabel(
            card,
            text=text,
            font=ctk.CTkFont(size=13),
            text_color="#475569",
            justify="left",
            wraplength=320,
        )
        body.grid(row=1, column=0, sticky="w", padx=14, pady=(0, 12))
        return body

    def _build_section_hint(self, parent, *, row: int, text: str, text_color: str = "#475569"):
        hint = ctk.CTkFrame(parent, fg_color="#f8fafc")
        hint.grid(row=row, column=0, columnspan=3, sticky="ew", padx=12, pady=(12 if row == 0 else 0, 10))
        hint.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            hint,
            text=text,
            justify="left",
            wraplength=980,
            text_color=text_color,
        ).grid(row=0, column=0, sticky="w", padx=12, pady=10)
        return hint

    def _build_profile_tab(self):
        tab = self.tabview.tab("Perfiles")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(5, weight=1)
        self._build_section_hint(
            tab,
            row=0,
            text="Paso 1. Selecciona un perfil existente o crea uno nuevo. Cada perfil guarda su calibración, modelo y ajustes del cursor.",
        )
        ctk.CTkLabel(tab, text="Perfil actual", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=1, column=0, sticky="w", padx=12, pady=(4, 6)
        )
        self.profile_menu = ctk.CTkOptionMenu(tab, values=self.profile_names, command=self._switch_profile)
        self.profile_menu.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        self.new_profile_entry = ctk.CTkEntry(tab, placeholder_text="Nuevo perfil")
        self.new_profile_entry.grid(row=3, column=0, sticky="ew", padx=12, pady=6)
        ctk.CTkButton(tab, text="Crear perfil", command=self._create_profile).grid(
            row=4, column=0, sticky="ew", padx=12, pady=6
        )
        self.profile_status_label = ctk.CTkLabel(
            tab,
            text="Estado del perfil: pendiente de carga.",
            justify="left",
            text_color="#475569",
        )
        self.profile_status_label.grid(row=5, column=0, sticky="w", padx=12, pady=(6, 4))
        self.profile_info = ctk.CTkTextbox(tab, height=280)
        self.profile_info.grid(row=6, column=0, sticky="nsew", padx=12, pady=(6, 12))
        self.profile_info.configure(state="disabled")

    def _build_diagnostics_tab(self):
        tab = self.tabview.tab("Diagnostico")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(2, weight=1)
        self._build_section_hint(
            tab,
            row=0,
            text="Paso 3. Verifica aquí si la cámara, el audio y el modelo del perfil están listos antes de entrar al runtime.",
        )
        ctk.CTkButton(tab, text="Ejecutar diagnostico", command=self._run_diagnostics).grid(
            row=1, column=0, sticky="ew", padx=12, pady=(0, 12)
        )
        self.diagnostics_box = ctk.CTkTextbox(tab, height=360)
        self.diagnostics_box.grid(row=2, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _build_settings_tab(self):
        tab = self.tabview.tab("Ajustes")
        tab.grid_columnconfigure(1, weight=1)
        self._build_section_hint(
            tab,
            row=0,
            text="Paso 2. Ajusta sensibilidad, velocidad y resolución antes de calibrar. Si dudas, empieza con 1280x720 o 960x540.",
        )
        self.setting_controls = {}
        self.bool_controls = {}
        controls = [
            ("cursor_sensitivity", "Sensibilidad cursor", 0.4, 3.0),
            ("dead_zone_px", "Zona muerta (px)", 0, 30),
            ("smoothing_factor", "Suavizado", 0.1, 0.95),
            ("max_cursor_speed_px", "Velocidad max. cursor", 6, 80),
        ]
        for row, (key, label, min_v, max_v) in enumerate(controls):
            row += 1
            ctk.CTkLabel(tab, text=label).grid(row=row, column=0, sticky="w", padx=12, pady=10)
            slider = ctk.CTkSlider(tab, from_=min_v, to=max_v)
            slider.grid(row=row, column=1, sticky="ew", padx=12, pady=10)
            value_label = ctk.CTkLabel(tab, text="-")
            value_label.grid(row=row, column=2, sticky="e", padx=12)
            slider.configure(command=lambda value, k=key, l=value_label: self._on_setting_change(k, value, l))
            self.setting_controls[key] = (slider, value_label)
        bool_row = len(controls) + 1
        for index, (key, label) in enumerate(
            [
                ("mirror_preview", "Vista espejo"),
                ("invert_x", "Invertir eje X del cursor"),
                ("invert_y", "Invertir eje Y del cursor"),
                ("cursor_starts_active", "Iniciar cursor activo"),
            ]
        ):
            switch = ctk.CTkSwitch(tab, text=label, command=lambda k=key: self._on_bool_setting_change(k))
            switch.grid(row=bool_row + index, column=0, columnspan=3, sticky="w", padx=12, pady=8)
            self.bool_controls[key] = switch
        resolution_row = bool_row + len(self.bool_controls)
        ctk.CTkLabel(tab, text="Resolución de cámara").grid(row=resolution_row, column=0, sticky="w", padx=12, pady=10)
        self.camera_resolution_menu = ctk.CTkOptionMenu(
            tab,
            values=CAMERA_RESOLUTION_OPTIONS,
            command=self._on_camera_resolution_change,
        )
        self.camera_resolution_menu.grid(row=resolution_row, column=1, sticky="ew", padx=12, pady=10)
        self.camera_resolution_hint = ctk.CTkLabel(tab, text="Usa 1280x720 o 960x540 para un frame amplio sin recortes.")
        self.camera_resolution_hint.grid(row=resolution_row, column=2, sticky="e", padx=12)
        ctk.CTkButton(tab, text="Guardar ajustes del perfil", command=self._save_profile_settings).grid(
            row=resolution_row + 1, column=0, columnspan=3, sticky="ew", padx=12, pady=12
        )

    def _build_calibration_tab(self):
        tab = self.tabview.tab("Calibracion")
        tab.grid_columnconfigure(0, weight=1)
        tab.grid_rowconfigure(5, weight=1)

        self._build_section_hint(
            tab,
            row=0,
            text="Paso 4. Captura primero la postura neutral y luego cada gesto. El entrenamiento se habilita solo cuando el dataset sea suficiente.",
        )

        summary = ctk.CTkFrame(tab, fg_color="white")
        summary.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        summary.grid_columnconfigure(0, weight=1)

        self.calibration_status = ctk.CTkLabel(
            summary,
            text="Captura la postura neutral primero y luego cada gesto con el rostro visible.",
            justify="left",
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.calibration_status.grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        self.calibration_hint = ctk.CTkLabel(
            summary,
            text="El entrenamiento se habilita solo cuando todos los gestos tienen suficientes frames válidos y ventanas útiles.",
            justify="left",
            text_color="#475569",
        )
        self.calibration_hint.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 8))
        self.training_summary_label = ctk.CTkLabel(
            summary,
            text="Resumen del dataset: sin datos.",
            justify="left",
            text_color="#183153",
        )
        self.training_summary_label.grid(row=2, column=0, sticky="w", padx=12, pady=(0, 4))
        self.training_blockers_label = ctk.CTkLabel(
            summary,
            text="Bloqueos: captura neutral y los gestos restantes.",
            justify="left",
            text_color="#b45309",
        )
        self.training_blockers_label.grid(row=3, column=0, sticky="w", padx=12, pady=(0, 12))

        self.capture_progress = ctk.CTkProgressBar(tab)
        self.capture_progress.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 4))
        self.capture_progress.set(0.0)
        self.capture_progress_label = ctk.CTkLabel(
            tab,
            text="Progreso de captura: 0%",
            justify="left",
            text_color="#475569",
        )
        self.capture_progress_label.grid(row=3, column=0, sticky="w", padx=12, pady=(0, 8))

        self.train_button = ctk.CTkButton(
            tab,
            text="Entrenar modelo",
            command=self._train_classifier,
            state="disabled",
        )
        self.train_button.grid(row=4, column=0, sticky="ew", padx=12, pady=(0, 8))

        self.cards_frame = ctk.CTkScrollableFrame(tab, fg_color="#eef2f7")
        self.cards_frame.grid(row=5, column=0, sticky="nsew", padx=12, pady=(0, 12))
        self.cards_frame.grid_columnconfigure(0, weight=1)

        for index, gesture_meta in enumerate([NEUTRAL_GESTURE_META] + GESTURE_CATALOG):
            card = ctk.CTkFrame(self.cards_frame, fg_color="white")
            card.grid(row=index, column=0, sticky="ew", padx=4, pady=6)
            card.grid_columnconfigure(0, weight=1)

            title = ctk.CTkLabel(card, text=gesture_meta["title"], font=ctk.CTkFont(size=16, weight="bold"))
            title.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 2))
            badge = ctk.CTkLabel(card, text="sin iniciar", corner_radius=10, fg_color="#94a3b8", text_color="white", padx=10)
            badge.grid(row=0, column=1, sticky="e", padx=12, pady=(10, 2))
            action = ctk.CTkLabel(card, text=f"Acción: {gesture_meta['action']}", text_color="#334155")
            action.grid(row=1, column=0, sticky="w", padx=12, pady=(0, 2))
            counts = ctk.CTkLabel(card, text="Frames capturados: 0 | válidos: 0 | inválidos: 0 | ventanas ML: 0")
            counts.grid(row=2, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 2))
            minimums = ctk.CTkLabel(
                card,
                text=(
                    f"Mínimo recomendado: {gesture_meta['recommended_min_frames']} frames y "
                    f"{gesture_meta['recommended_min_windows']} ventanas"
                ),
                text_color="#475569",
            )
            minimums.grid(row=3, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 2))
            help_label = ctk.CTkLabel(card, text=gesture_meta["training_hint"], justify="left", wraplength=720)
            help_label.grid(row=4, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 6))
            blocker = ctk.CTkLabel(card, text="Bloqueo: captura pendiente.", text_color="#b45309", justify="left", wraplength=720)
            blocker.grid(row=5, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 10))
            button = ctk.CTkButton(
                card,
                text="Capturar" if gesture_meta["id"] == "neutral" else "Capturar / Recapturar",
                command=lambda gesture_id=gesture_meta["id"]: self._begin_capture(gesture_id),
                width=180,
            )
            button.grid(row=0, column=2, rowspan=2, sticky="e", padx=12, pady=12)
            self.training_cards[gesture_meta["id"]] = {
                "badge": badge,
                "counts": counts,
                "minimums": minimums,
                "help": help_label,
                "blocker": blocker,
                "button": button,
            }

    def _populate_guide(self):
        guide_tab = self.tabview.tab("Guia")
        guide = ctk.CTkScrollableFrame(guide_tab)
        guide.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(
            guide,
            text="Orden recomendado de uso: 1. Perfil  2. Ajustes  3. Diagnóstico  4. Calibración  5. Iniciar sistema asistivo",
            justify="left",
            wraplength=980,
            text_color="#475569",
        ).pack(fill="x", padx=10, pady=(8, 6))
        ctk.CTkLabel(
            guide,
            text="Gestos faciales implementados",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))
        for gesture in [NEUTRAL_GESTURE_META] + GESTURE_CATALOG:
            ctk.CTkLabel(
                guide,
                text=f"{gesture['title']} -> {gesture['action']}",
                font=ctk.CTkFont(size=14, weight="bold"),
                anchor="w",
            ).pack(fill="x", padx=10, pady=(8, 0))
            ctk.CTkLabel(
                guide,
                text=(
                    f"Duracion minima de uso: {gesture['duration_ms']} ms\n"
                    f"Entrenamiento guiado: {self._capture_seconds_for(gesture['id']):.1f} s\n"
                    f"Cómo hacerlo: {gesture['training_hint']}\n"
                    f"Error común: {gesture['common_failure_hint']}\n"
                    f"{gesture['warning']}"
                ),
                justify="left",
                anchor="w",
            ).pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkLabel(
            guide,
            text="Comandos de voz disponibles",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(12, 4))
        for command, description in VOICE_COMMAND_HELP:
            ctk.CTkLabel(
                guide,
                text=f"{command}: {description}",
                justify="left",
                anchor="w",
            ).pack(fill="x", padx=10, pady=3)

    def _schedule_preview(self):
        if self.cap is None or self.provider is None or not self.root.winfo_exists():
            return
        if self.capture_target is None:
            self.root.after(120, self._schedule_preview)
            return
        if not self.cap.isOpened():
            self.capture_monitor_label.configure(text="No se pudo abrir la cámara para capturar el gesto.")
            self.root.after(200, self._schedule_preview)
            return

        ok, frame = self.cap.read()
        if ok:
            snapshot = self.provider.process(frame)
            self.latest_sample = snapshot.face_sample
            self._update_capture_state(snapshot.face_sample)

        self.root.after(66, self._schedule_preview)

    def _update_capture_state(self, sample):
        if self.capture_target is None:
            return

        now = time.monotonic()
        if self.capture_phase == "countdown":
            remaining = max(0.0, self.capture_countdown_deadline - now)
            self.capture_monitor_label.configure(
                text=f"Preparando captura de {self._gesture_title(self.capture_target)}: {remaining:.1f} s"
            )
            progress = max(0.0, min(1.0, 1 - (remaining / self.CAPTURE_PREP_SECONDS)))
            self.capture_progress.set(progress)
            self.capture_progress_label.configure(text=f"Preparacion: {int(progress * 100)}%")
            if now >= self.capture_countdown_deadline:
                self.capture_phase = "capturing"
                gesture_name = self._gesture_title(self.capture_target)
                self.calibration_status.configure(
                    text=f"Capturando {gesture_name}. Mantén el gesto durante {self.capture_duration_seconds:.1f} s."
                )
                self.capture_monitor_label.configure(
                    text=f"Captura activa: mantén {gesture_name} durante {self.capture_duration_seconds:.1f} s."
                )
            return

        accepted, feedback = self.calibration_service.register_sample(self.capture_target, sample)
        remaining = max(0.0, self.capture_deadline - now)
        gesture_name = self._gesture_title(self.capture_target)
        self.capture_monitor_label.configure(
            text=f"Capturando {gesture_name}: {remaining:.1f} s restantes. {feedback}"
        )
        self.calibration_hint.configure(text=feedback)
        if not accepted:
            self.calibration_status.configure(text=f"Captura en curso para {gesture_name}. {feedback}")
        progress = max(0.0, min(1.0, 1 - (remaining / max(self.capture_duration_seconds, 0.1))))
        self.capture_progress.set(progress)
        self.capture_progress_label.configure(text=f"Captura: {int(progress * 100)}%")
        self._refresh_training_dashboard()

        if now >= self.capture_deadline:
            statuses = self.calibration_service.refresh_statuses()
            status = statuses[self.capture_target]
            self.capture_monitor_label.configure(
                text=f"Captura finalizada para {gesture_name}. Frames válidos: {status.valid_frames}."
            )
            self.calibration_status.configure(
                text=(
                    f"Captura completada para {gesture_name}. "
                    f"Frames válidos: {status.valid_frames}. Ventanas ML: {status.training_windows}."
                )
            )
            self.capture_target = None
            self.capture_phase = "idle"
            self.capture_progress.set(1.0)
            self.capture_progress_label.configure(text="Captura completada: 100%")
            self._refresh_training_dashboard()
            self._sync_runtime_ready_state()

    def _begin_capture(self, gesture_id: str):
        capture_seconds = self._capture_seconds_for(gesture_id)
        gesture_meta = self._gesture_meta(gesture_id)
        self.calibration_service.reset_gesture(gesture_id)
        self.capture_target = gesture_id
        self.capture_phase = "countdown"
        self.capture_duration_seconds = capture_seconds
        self.capture_countdown_deadline = time.monotonic() + self.CAPTURE_PREP_SECONDS
        self.capture_deadline = self.capture_countdown_deadline + capture_seconds
        self.capture_progress.set(0.0)
        self.capture_progress_label.configure(text="Progreso de captura: 0%")
        self.capture_monitor_label.configure(
            text=f"Preparando captura de {gesture_meta['title']}. Colócate frente a la cámara y espera la cuenta regresiva."
        )
        self.calibration_status.configure(
            text=(
                f"Preparate para {gesture_meta['title']}. Tendrás {self.CAPTURE_PREP_SECONDS:.0f} s antes de la captura "
                f"y luego deberás mantener el gesto {capture_seconds:.1f} s."
            )
        )
        self.calibration_hint.configure(
            text=f"Cómo hacerlo: {gesture_meta['training_hint']} | Error común: {gesture_meta['common_failure_hint']}"
        )
        self._refresh_training_dashboard()
        self._sync_runtime_ready_state()

    def _train_classifier(self):
        ready, blockers, statuses = self.calibration_service.can_train()
        if not ready:
            self.calibration_status.configure(
                text=f"No se puede entrenar todavía. Faltan: {', '.join(blockers)}."
            )
            self._refresh_training_dashboard()
            return

        try:
            samples = self.calibration_service.export_samples()
            capture_quality = self.calibration_service.export_capture_quality()
            classifier = self.training_service.train(samples, capture_quality_summary=capture_quality)

            neutral = samples.get("neutral", [])
            if neutral:
                self.profile["neutral_nose"] = [
                    sum(sample.nose_px[0] for sample in neutral) / len(neutral),
                    sum(sample.nose_px[1] for sample in neutral) / len(neutral),
                ]
                self.profile["neutral_face_scale_px"] = sum(sample.face_scale_px for sample in neutral) / len(neutral)
                self.profile["neutral_face_center"] = [
                    sum(sample.face_center_px[0] for sample in neutral) / len(neutral),
                    sum(sample.face_center_px[1] for sample in neutral) / len(neutral),
                ]

            self.profile = self.training_service.build_profile_training_state(
                self.profile,
                samples,
                capture_quality,
                statuses,
            )
            self.calibration_service.mark_trained()
            self.training_statuses = self.calibration_service.refresh_statuses()
            self.profile["calibration"]["gesture_readiness"] = {
                gesture_id: status.readiness for gesture_id, status in self.training_statuses.items()
            }
            self.profile["calibration"]["capture_quality"] = self.calibration_service.export_capture_quality()
            save_profile(self.profile)

            accuracy = classifier.training_summary.get("validation_accuracy")
            accuracy_text = f"{accuracy:.3f}" if isinstance(accuracy, float) else "sin validación"
            self.launch_status.configure(text="Modelo entrenado. Ya puedes iniciar el sistema o seguir refinando capturas.")
            self.calibration_status.configure(
                text=(
                    "Modelo entrenado y perfil guardado. "
                    f"Accuracy validación: {accuracy_text}. "
                    f"Versión activa: {classifier.active_version}."
                )
            )
            self.calibration_hint.configure(
                text="Thresholds recomendados y métricas por clase ya fueron persistidos en el perfil."
            )
            self._refresh_profile_info()
            self._refresh_training_dashboard()
            self._refresh_launcher_summary()
            self._sync_runtime_ready_state()
        except Exception as exc:
            self.calibration_status.configure(text=f"Error entrenando modelo: {exc}")

    def _refresh_training_dashboard(self):
        statuses = self.calibration_service.refresh_statuses()
        self.training_statuses = statuses
        summary = self.calibration_service.build_summary()

        blockers_text = ", ".join(summary["blockers"]) if summary["blockers"] else "ninguno"
        dataset_state = "pendiente de reentrenar" if summary["pending_retraining"] else "sin cambios pendientes"
        self.training_summary_label.configure(
            text=(
                f"Resumen del dataset: {summary['ready_count']}/{summary['total_gestures']} gestos listos | "
                f"Estado del dataset: {dataset_state}"
            )
        )
        self.training_blockers_label.configure(text=f"Bloqueos actuales: {blockers_text}")
        self.train_button.configure(state="normal" if summary["ready_to_train"] else "disabled")
        self._refresh_launcher_summary()

        for gesture_id, status in statuses.items():
            widgets = self.training_cards[gesture_id]
            color = READINESS_COLORS.get(status.readiness, "#94a3b8")
            widgets["badge"].configure(text=status.readiness, fg_color=color)
            widgets["counts"].configure(
                text=(
                    f"Frames capturados: {status.captured_frames} | válidos: {status.valid_frames} | "
                    f"inválidos: {status.invalid_frames} | ventanas ML: {status.training_windows}"
                )
            )
            widgets["help"].configure(
                text=f"Ayuda: {status.help_message}\nÉxito esperado: {status.success_message}"
            )
            blocker_text = status.blocker or "Listo para entrenar."
            widgets["blocker"].configure(
                text=f"Bloqueo / estado: {blocker_text}",
                text_color="#b45309" if status.blocker else "#047857",
            )

    def _run_diagnostics(self):
        self.diagnostics_box.delete("1.0", "end")
        camera_status = "OK" if self.cap and self.cap.isOpened() else "Fallo"
        self.diagnostics_box.insert("end", f"Camara: {camera_status}\n")
        if self.cap and self.cap.isOpened():
            self.diagnostics_box.insert(
                "end",
                (
                    f"Resolución pedida: {self.settings.get('camera_resolution')} | "
                    f"Resolución activa: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}\n"
                ),
            )
        try:
            devices = sd.query_devices()
            self.diagnostics_box.insert("end", f"Microfono / audio: OK ({len(devices)} dispositivos)\n")
        except Exception as exc:
            self.diagnostics_box.insert("end", f"Microfono / audio: error {exc}\n")
        model_status = self.training_service.load_classifier()
        self.diagnostics_box.insert("end", f"Modelo ML del perfil: {'OK' if model_status else 'No entrenado'}\n")
        self.diagnostics_box.insert("end", f"Version activa: {self.training_service.classifier.active_version}\n")
        self.diagnostics_box.insert("end", f"Versiones disponibles: {[entry['version'] for entry in self.training_service.list_versions()]}\n")
        self.diagnostics_box.insert("end", f"Thresholds actuales: {self.profile.get('gesture_confidence', {})}\n")
        self.diagnostics_box.insert("end", f"Pending retraining: {self.profile['calibration'].get('pending_retraining')}\n")
        self.diagnostics_box.insert("end", f"Gesture readiness: {self.profile['calibration'].get('gesture_readiness', {})}\n")
        self.diagnostics_box.insert("end", f"Capture quality: {self.profile['calibration'].get('capture_quality', {})}\n")
        self.diagnostics_box.insert("end", f"Ultimas metricas de entrenamiento: {self.profile['calibration'].get('last_training_metrics', {})}\n")
        self.diagnostics_box.insert("end", "TTS: opcional; el runtime continuara aunque falle.\n")

    def _switch_profile(self, profile_name: str, refresh_menu: bool = True):
        self.current_profile_name = profile_name
        self.profile = load_profile(profile_name)
        self.training_service = TrainingService(profile_name, self.window_size)
        self.calibration_service = CalibrationService(self.window_size)
        self._apply_profile_to_controls()
        self._load_existing_dataset()
        self._refresh_profile_info()
        self._refresh_training_dashboard()
        self._refresh_launcher_summary()
        self._sync_runtime_ready_state()
        if refresh_menu:
            self.profile_menu.set(profile_name)

    def _create_profile(self):
        name = self.new_profile_entry.get().strip()
        if not name:
            return
        if name in self.profile_names:
            self.launch_status.configure(text=f"El perfil {name} ya existe.")
            return
        save_profile(build_default_profile(name))
        self.profile_names = list_profiles()
        self.profile_menu.configure(values=self.profile_names)
        self.new_profile_entry.delete(0, "end")
        self.launch_status.configure(text=f"Perfil {name} creado.")
        self._switch_profile(name)

    def _apply_profile_to_controls(self):
        for key, (slider, label) in self.setting_controls.items():
            value = float(self.profile.get(key, self.settings.get(key, 0)))
            slider.set(value)
            label.configure(text=f"{value:.2f}")
        for key, switch in self.bool_controls.items():
            if bool(self.profile.get(key, False)):
                switch.select()
            else:
                switch.deselect()
        self.camera_resolution_menu.set(self.settings.get("camera_resolution", "1280x720"))

    def _refresh_profile_info(self):
        calibration = self.profile["calibration"]
        readiness_summary = self.calibration_service.build_summary()
        active_version = calibration.get("active_model_version")
        active_version_text = f"v{active_version}" if active_version else "sin modelo activo"
        pending_text = "sí" if calibration.get("pending_retraining") else "no"
        readiness_text = f"{readiness_summary['ready_count']}/{readiness_summary['total_gestures']}"
        self.profile_status_label.configure(
            text=(
                f"Estado del perfil: {active_version_text} | Gestos listos: {readiness_text} | "
                f"Reentrenamiento pendiente: {pending_text}"
            )
        )
        self.profile_info.configure(state="normal")
        self.profile_info.delete("1.0", "end")
        self.profile_info.insert(
            "end",
            (
                f"Perfil: {self.profile['name']}\n"
                f"Estado general: {'listo para iniciar' if bool(self.profile.get('launcher_completed')) and active_version else 'requiere completar flujo'}\n"
                f"Modelo activo: {active_version_text}\n"
                f"Gestos listos para entrenar: {readiness_text}\n"
                f"Reentrenamiento pendiente: {pending_text}\n"
                f"Última calibración: {calibration['last_calibrated_at']}\n\n"
                f"Ajustes principales\n"
                f"- Sensibilidad cursor: {self.profile['cursor_sensitivity']}\n"
                f"- Zona muerta: {self.profile['dead_zone_px']}\n"
                f"- Suavizado: {self.profile['smoothing_factor']}\n"
                f"- Velocidad máxima: {self.profile['max_cursor_speed_px']}\n"
                f"- Vista espejo: {self.profile['mirror_preview']}\n"
                f"- Invertir X: {self.profile['invert_x']}\n"
                f"- Invertir Y: {self.profile['invert_y']}\n"
                f"- Cursor inicia activo: {self.profile['cursor_starts_active']}\n\n"
                f"Detalle técnico\n"
                f"- Versiones guardadas: {calibration['model_versions']}\n"
                f"- Readiness por gesto: {calibration.get('gesture_readiness')}\n"
                f"- Modelo perfil: {calibration['active_model_path']}\n"
                f"- Dataset activo: {calibration['active_dataset_path']}\n"
            ),
        )
        self.profile_info.configure(state="disabled")
        self._refresh_launcher_summary()

    def _on_setting_change(self, key, value, value_label):
        self.profile[key] = float(value)
        value_label.configure(text=f"{float(value):.2f}")

    def _on_bool_setting_change(self, key):
        switch = self.bool_controls[key]
        self.profile[key] = bool(switch.get())

    def _save_profile_settings(self):
        save_profile(self.profile)
        self.launch_status.configure(text="Ajustes del perfil guardados.")
        self._refresh_profile_info()
        self._refresh_launcher_summary()

    def _start_runtime(self):
        if not self.training_service.load_classifier():
            self.launch_status.configure(text="Este perfil aun no tiene un modelo entrenado. Completa calibracion.")
            self.start_button.configure(state="disabled")
            return
        self.profile["calibration"]["pending_retraining"] = self.calibration_service.pending_retraining
        self.profile["calibration"]["gesture_readiness"] = {
            gesture_id: status.readiness for gesture_id, status in self.training_statuses.items()
        }
        self.profile["calibration"]["capture_quality"] = self.calibration_service.export_capture_quality()
        save_profile(self.profile)
        save_settings(self.settings)
        self.shutdown()
        self.on_start_callback(self.profile, self.settings)

    def _sync_runtime_ready_state(self):
        model_ready = self.training_service.load_classifier()
        self.profile["calibration"]["pending_retraining"] = self.calibration_service.pending_retraining
        if model_ready:
            self.profile["calibration"]["active_model_version"] = self.training_service.classifier.active_version
            self.profile["calibration"]["active_model_path"] = str(self.training_service.classifier.model_path)
            self.profile["calibration"]["active_dataset_path"] = self.training_service.classifier.active_dataset_path
            self.profile["calibration"]["model_versions"] = [
                entry["version"] for entry in self.training_service.list_versions()
            ]
            self.profile["calibration"]["last_training_metrics"] = self.training_service.classifier.training_summary
        launcher_ready = bool(self.profile.get("launcher_completed")) and model_ready
        self.start_button.configure(state="normal" if launcher_ready else "disabled")
        if launcher_ready and self.calibration_service.pending_retraining:
            self.launch_status.configure(
                text="Hay nuevas capturas pendientes. Puedes usar el modelo activo o reentrenar antes de iniciar."
            )
        elif launcher_ready:
            self.launch_status.configure(text="Perfil calibrado. Puedes iniciar el sistema asistivo.")
        else:
            self.launch_status.configure(text="Completa un entrenamiento válido antes de iniciar.")
        self._refresh_launcher_summary()

    def _load_existing_dataset(self):
        loaded = self.training_service.load_active_dataset()
        stored_quality = self.profile["calibration"].get("capture_quality")
        self.calibration_service.load_samples(loaded, stored_capture_quality=stored_quality)
        if self.profile["calibration"].get("pending_retraining"):
            self.calibration_service.pending_retraining = True
        self.training_statuses = self.calibration_service.refresh_statuses()

    def _gesture_meta(self, gesture_id: str) -> dict:
        if gesture_id == "neutral":
            return NEUTRAL_GESTURE_META
        for gesture in GESTURE_CATALOG:
            if gesture["id"] == gesture_id:
                return gesture
        return NEUTRAL_GESTURE_META

    def _gesture_title(self, gesture_id: str) -> str:
        return self._gesture_meta(gesture_id)["title"]

    def _capture_seconds_for(self, gesture_id: str) -> float:
        if gesture_id == "neutral":
            return 3.0
        duration_ms = float(self._gesture_meta(gesture_id).get("duration_ms", 700))
        return max(2.5, (duration_ms / 1000.0) + 1.6)

    def shutdown(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if self.provider:
            self.provider.close()
        self.cap = None
        self.provider = None

    def _on_close(self):
        self.shutdown()
        self.root.quit()
        self.root.destroy()

    def _on_camera_resolution_change(self, value: str):
        self.settings["camera_resolution"] = value
        if self.cap and self.cap.isOpened():
            self._apply_camera_resolution(self.cap)
        self._refresh_launcher_summary()
        self.launch_status.configure(text=f"Resolución de cámara cambiada a {value}.")

    def _apply_camera_resolution(self, cap):
        width, height = parse_camera_resolution(self.settings.get("camera_resolution"))
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except Exception:
            pass

    def _refresh_launcher_summary(self):
        ready_summary = self.calibration_service.build_summary()
        self.profile_summary_card.configure(
            text=(
                f"{self.profile['name']}\n"
                f"Modelo activo: v{self.profile['calibration'].get('active_model_version') or '-'}\n"
                f"Gestos listos: {ready_summary['ready_count']}/{ready_summary['total_gestures']}"
            )
        )
        self.camera_summary_card.configure(
            text=(
                f"Índice: {self.settings.get('camera_index', 0)}\n"
                f"Resolución: {self.settings.get('camera_resolution', '1280x720')}\n"
                f"FPS objetivo: {self.settings.get('fps', 15)}"
            )
        )
        capture_state = "pendiente de reentrenar" if ready_summary["pending_retraining"] else "sin cambios pendientes"
        self.flow_summary_card.configure(
            text=(
                "1. Elige perfil\n"
                "2. Ajusta cámara y sensibilidad\n"
                "3. Ejecuta diagnóstico\n"
                "4. Captura y entrena\n"
                f"Estado actual: {capture_state}"
            )
        )

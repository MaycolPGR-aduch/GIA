from __future__ import annotations

import time
import tkinter as tk

import customtkinter as ctk
import cv2
import sounddevice as sd
from PIL import Image, ImageTk

from .config_store import (
    build_default_profile,
    list_profiles,
    load_profile,
    load_settings,
    save_profile,
    save_settings,
)
from .gesture_catalog import GESTURE_CATALOG, VOICE_COMMAND_HELP
from .gesture_ml import GestureClassifier
from .landmark_provider import LandmarkProvider


class LauncherGUI:
    CAPTURE_PREP_SECONDS = 3.0

    def __init__(self, root, on_start_callback):
        self.root = root
        self.on_start_callback = on_start_callback
        self.settings = load_settings()
        self.profile_names = list_profiles()
        self.current_profile_name = self.profile_names[0]
        self.profile = load_profile(self.current_profile_name)
        self.classifier = GestureClassifier(
            self.current_profile_name,
            window_size=int(self.settings.get("gesture_window_size", 12)),
        )

        self.calibration_samples: dict[str, list] = {gesture["id"]: [] for gesture in GESTURE_CATALOG}
        self.calibration_samples["neutral"] = []
        self.capture_target: str | None = None
        self.capture_phase = "idle"
        self.capture_countdown_deadline = 0.0
        self.capture_deadline = 0.0
        self.capture_duration_seconds = 0.0
        self.preview_image = None
        self.latest_sample = None

        self.root.title("GIA v2 - Launcher")
        self.root.geometry("1320x880")
        self.root.minsize(1180, 760)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.cap = cv2.VideoCapture(int(self.settings.get("camera_index", 0)))
        self.provider = LandmarkProvider(fps=int(self.settings.get("fps", 15)))

        self._build_layout()
        self._populate_guide()
        self._apply_profile_to_controls()
        self.profile_menu.set(self.current_profile_name)
        self._refresh_sample_counts()
        self._sync_runtime_ready_state()
        self._schedule_preview()

    def _build_layout(self):
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        left = ctk.CTkFrame(self.root, fg_color="#0f172a")
        left.grid(row=0, column=0, sticky="nsew", padx=(18, 9), pady=18)
        left.grid_rowconfigure(4, weight=1)
        left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left,
            text="GIA v2 Launcher",
            font=ctk.CTkFont(size=30, weight="bold"),
            text_color="white",
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(18, 6))
        ctk.CTkLabel(
            left,
            text="Perfil, guia, diagnostico y calibracion obligatoria antes de iniciar.",
            font=ctk.CTkFont(size=15),
            text_color="#cbd5e1",
            justify="left",
        ).grid(row=1, column=0, sticky="w", padx=18, pady=(0, 18))

        self.tabview = ctk.CTkTabview(left, fg_color="#e5e7eb")
        self.tabview.grid(row=2, column=0, sticky="nsew", padx=18, pady=(0, 18))
        for tab_name in ("Perfiles", "Guia", "Diagnostico", "Ajustes", "Calibracion"):
            self.tabview.add(tab_name)

        self._build_profile_tab()
        self._build_diagnostics_tab()
        self._build_settings_tab()
        self._build_calibration_tab()

        start_bar = ctk.CTkFrame(left, fg_color="#111827")
        start_bar.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))
        self.launch_status = ctk.CTkLabel(
            start_bar,
            text="Completa una calibracion y entrenamiento antes de iniciar.",
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

        right = ctk.CTkFrame(self.root, fg_color="#f8fafc")
        right.grid(row=0, column=1, sticky="nsew", padx=(9, 18), pady=18)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self.preview_status = ctk.CTkLabel(
            right,
            text="Vista previa y captura",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#183153",
        )
        self.preview_status.grid(row=0, column=0, sticky="w", padx=18, pady=(18, 8))

        self.preview_label = tk.Label(right, bg="#000000", bd=0, highlightthickness=0)
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))

    def _build_profile_tab(self):
        tab = self.tabview.tab("Perfiles")
        tab.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(tab, text="Perfil actual", font=ctk.CTkFont(size=18, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=12, pady=(14, 6)
        )
        self.profile_menu = ctk.CTkOptionMenu(tab, values=self.profile_names, command=self._switch_profile)
        self.profile_menu.grid(row=1, column=0, sticky="ew", padx=12, pady=6)
        self.new_profile_entry = ctk.CTkEntry(tab, placeholder_text="Nuevo perfil")
        self.new_profile_entry.grid(row=2, column=0, sticky="ew", padx=12, pady=6)
        ctk.CTkButton(tab, text="Crear perfil", command=self._create_profile).grid(
            row=3, column=0, sticky="ew", padx=12, pady=6
        )
        self.profile_info = ctk.CTkTextbox(tab, height=260)
        self.profile_info.grid(row=4, column=0, sticky="nsew", padx=12, pady=(10, 12))
        self.profile_info.configure(state="disabled")

    def _build_diagnostics_tab(self):
        tab = self.tabview.tab("Diagnostico")
        tab.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(tab, text="Ejecutar diagnostico", command=self._run_diagnostics).grid(
            row=0, column=0, sticky="ew", padx=12, pady=12
        )
        self.diagnostics_box = ctk.CTkTextbox(tab, height=320)
        self.diagnostics_box.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _build_settings_tab(self):
        tab = self.tabview.tab("Ajustes")
        tab.grid_columnconfigure(1, weight=1)
        self.setting_controls = {}
        controls = [
            ("cursor_sensitivity", "Sensibilidad cursor", 0.4, 3.0),
            ("dead_zone_px", "Zona muerta (px)", 0, 30),
            ("smoothing_factor", "Suavizado", 0.1, 0.95),
            ("max_cursor_speed_px", "Velocidad max. cursor", 6, 80),
        ]
        for row, (key, label, min_v, max_v) in enumerate(controls):
            ctk.CTkLabel(tab, text=label).grid(row=row, column=0, sticky="w", padx=12, pady=10)
            slider = ctk.CTkSlider(tab, from_=min_v, to=max_v)
            slider.grid(row=row, column=1, sticky="ew", padx=12, pady=10)
            value_label = ctk.CTkLabel(tab, text="-")
            value_label.grid(row=row, column=2, sticky="e", padx=12)
            slider.configure(command=lambda value, k=key, l=value_label: self._on_setting_change(k, value, l))
            self.setting_controls[key] = (slider, value_label)
        ctk.CTkButton(tab, text="Guardar ajustes del perfil", command=self._save_profile_settings).grid(
            row=len(controls), column=0, columnspan=3, sticky="ew", padx=12, pady=12
        )

    def _build_calibration_tab(self):
        tab = self.tabview.tab("Calibracion")
        tab.grid_columnconfigure((0, 1), weight=1)
        self.calibration_status = ctk.CTkLabel(
            tab,
            text="Captura la postura neutral primero y luego cada gesto con el rostro visible.",
            justify="left",
        )
        self.calibration_status.grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(12, 8))
        self.calibration_hint = ctk.CTkLabel(
            tab,
            text="Flujo: 3 s de preparacion -> captura guiada -> entrenamiento del modelo del perfil.",
            justify="left",
            text_color="#475569",
        )
        self.calibration_hint.grid(row=1, column=0, columnspan=2, sticky="w", padx=12, pady=(0, 8))

        ctk.CTkButton(tab, text="Capturar neutral", command=lambda: self._begin_capture("neutral")).grid(
            row=2, column=0, sticky="ew", padx=12, pady=6
        )
        ctk.CTkButton(tab, text="Entrenar modelo", command=self._train_classifier).grid(
            row=2, column=1, sticky="ew", padx=12, pady=6
        )
        self.sample_count_box = ctk.CTkTextbox(tab, height=320)
        self.sample_count_box.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=12, pady=(8, 12))

        for index, gesture in enumerate(GESTURE_CATALOG, start=4):
            ctk.CTkButton(
                tab,
                text=f"Capturar {gesture['title']}",
                command=lambda gesture_id=gesture["id"]: self._begin_capture(gesture_id),
            ).grid(row=index, column=0, sticky="ew", padx=12, pady=4)
            ctk.CTkLabel(
                tab,
                text=(
                    f"{gesture['action']} | mantener {gesture['duration_ms'] / 1000:.2f} s "
                    f"| grabacion {self._capture_seconds_for(gesture['id']):.1f} s"
                ),
                justify="left",
            ).grid(row=index, column=1, sticky="w", padx=12, pady=4)

    def _populate_guide(self):
        guide_tab = self.tabview.tab("Guia")
        guide = ctk.CTkScrollableFrame(guide_tab)
        guide.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(
            guide,
            text="Gestos faciales implementados",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(anchor="w", padx=10, pady=(8, 4))
        for gesture in GESTURE_CATALOG:
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
        if not self.cap.isOpened():
            self.preview_status.configure(text="No se pudo abrir la camara")
            return

        ok, frame = self.cap.read()
        if ok:
            snapshot = self.provider.process(frame)
            self.latest_sample = snapshot.face_sample
            preview_frame = snapshot.frame_rgb.copy()
            if snapshot.face_sample is not None:
                for point in snapshot.face_sample.points_px.values():
                    cv2.circle(preview_frame, point, 2, (0, 255, 0), -1)
                cv2.circle(preview_frame, snapshot.face_sample.nose_px, 5, (255, 80, 0), -1)
            self._update_capture_state(snapshot.face_sample)
            self._draw_capture_overlay(preview_frame)
            image = Image.fromarray(preview_frame)
            image.thumbnail((760, 620))
            photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=photo)
            self.preview_image = photo

        self.root.after(66, self._schedule_preview)

    def _update_capture_state(self, sample):
        if self.capture_target is None:
            return

        now = time.monotonic()
        if self.capture_phase == "countdown":
            remaining = max(0.0, self.capture_countdown_deadline - now)
            self.preview_status.configure(text=f"Preparate: {remaining:.1f} s")
            if now >= self.capture_countdown_deadline:
                self.capture_phase = "capturing"
                gesture_name = self._gesture_title(self.capture_target)
                self.calibration_status.configure(
                    text=f"Capturando {gesture_name}. Manten el gesto durante {self.capture_duration_seconds:.1f} s."
                )
            return

        if sample is not None:
            self.calibration_samples[self.capture_target].append(sample)

        remaining = max(0.0, self.capture_deadline - now)
        gesture_name = self._gesture_title(self.capture_target)
        self.preview_status.configure(text=f"Capturando {gesture_name}: {remaining:.1f} s restantes")
        if now >= self.capture_deadline:
            captured = len(self.calibration_samples[self.capture_target])
            self.preview_status.configure(text=f"Captura finalizada: {gesture_name}")
            self.calibration_status.configure(
                text=f"Captura completada para {gesture_name}. Se guardaron {captured} frames de landmarks."
            )
            self.capture_target = None
            self.capture_phase = "idle"
            self._refresh_sample_counts()

    def _begin_capture(self, gesture_id: str):
        capture_seconds = self._capture_seconds_for(gesture_id)
        gesture_name = self._gesture_title(gesture_id)
        self.calibration_status.configure(
            text=(
                f"Preparate para {gesture_name}. Tendras {self.CAPTURE_PREP_SECONDS:.0f} s antes de la captura "
                f"y luego deberas mantener el gesto {capture_seconds:.1f} s."
            )
        )
        self.preview_status.configure(text=f"Preparando captura de {gesture_name}...")
        self.calibration_samples[gesture_id] = []
        self.capture_target = gesture_id
        self.capture_phase = "countdown"
        self.capture_duration_seconds = capture_seconds
        self.capture_countdown_deadline = time.monotonic() + self.CAPTURE_PREP_SECONDS
        self.capture_deadline = self.capture_countdown_deadline + capture_seconds

    def _train_classifier(self):
        try:
            samples = {name: values[:] for name, values in self.calibration_samples.items() if values}
            self.classifier = GestureClassifier(
                self.current_profile_name,
                window_size=int(self.settings.get("gesture_window_size", 12)),
            )
            self.classifier.fit(samples)

            self.profile["launcher_completed"] = True
            self.profile["calibration"]["completed"] = True
            self.profile["calibration"]["last_calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self.profile["calibration"]["samples_per_gesture"] = {
                name: len(values) for name, values in self.calibration_samples.items()
            }

            self.classifier.save_samples(
                {
                    "profile": self.current_profile_name,
                    "model_path": str(self.classifier.model_path),
                    "samples_path": str(self.classifier.samples_path),
                    "training_summary": self.classifier.training_summary,
                    "sample_counts": self.profile["calibration"]["samples_per_gesture"],
                }
            )

            neutral = self.calibration_samples.get("neutral", [])
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

            save_profile(self.profile)
            self.launch_status.configure(text="Calibracion lista. Ya puedes iniciar el sistema.")
            self.calibration_status.configure(
                text=(
                    "Modelo entrenado y perfil guardado. "
                    f"Clases: {', '.join(self.classifier.training_summary.get('classes', []))}."
                )
            )
            self._refresh_profile_info()
            self._sync_runtime_ready_state()
        except Exception as exc:
            self.calibration_status.configure(text=f"Error entrenando modelo: {exc}")

    def _refresh_sample_counts(self):
        self.sample_count_box.delete("1.0", "end")
        self.sample_count_box.insert("end", "Muestras por gesto:\n")
        for name, values in self.calibration_samples.items():
            self.sample_count_box.insert(
                "end",
                f"- {name}: {len(values)} frames | recomendado: {self._capture_seconds_for(name):.1f} s\n",
            )

    def _run_diagnostics(self):
        self.diagnostics_box.delete("1.0", "end")
        camera_status = "OK" if self.cap and self.cap.isOpened() else "Fallo"
        self.diagnostics_box.insert("end", f"Camara: {camera_status}\n")
        try:
            devices = sd.query_devices()
            self.diagnostics_box.insert("end", f"Microfono / audio: OK ({len(devices)} dispositivos)\n")
        except Exception as exc:
            self.diagnostics_box.insert("end", f"Microfono / audio: error {exc}\n")
        model_status = "OK" if self.classifier.load() else "No entrenado todavia"
        self.diagnostics_box.insert("end", f"Modelo ML del perfil: {model_status}\n")
        self.diagnostics_box.insert("end", f"Ruta modelo: {self.classifier.model_path}\n")
        self.diagnostics_box.insert("end", f"Ruta resumen entrenamiento: {self.classifier.samples_path}\n")
        if self.classifier.training_summary:
            self.diagnostics_box.insert("end", f"Resumen entrenamiento: {self.classifier.training_summary}\n")
        self.diagnostics_box.insert("end", "TTS: opcional; el runtime continuara aunque falle.\n")

    def _switch_profile(self, profile_name: str):
        self.current_profile_name = profile_name
        self.profile = load_profile(profile_name)
        self.classifier = GestureClassifier(profile_name, window_size=int(self.settings.get("gesture_window_size", 12)))
        self._apply_profile_to_controls()
        self._refresh_profile_info()
        self._sync_runtime_ready_state()

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
        self.profile_menu.set(name)
        self._switch_profile(name)
        self.new_profile_entry.delete(0, "end")
        self.launch_status.configure(text=f"Perfil {name} creado.")

    def _apply_profile_to_controls(self):
        for key, (slider, label) in self.setting_controls.items():
            value = float(self.profile.get(key, self.settings.get(key, 0)))
            slider.set(value)
            label.configure(text=f"{value:.2f}")
        self._refresh_profile_info()

    def _refresh_profile_info(self):
        self.profile_info.configure(state="normal")
        self.profile_info.delete("1.0", "end")
        self.profile_info.insert(
            "end",
            (
                f"Perfil: {self.profile['name']}\n"
                f"Calibrado: {self.profile['calibration']['completed']}\n"
                f"Ultima calibracion: {self.profile['calibration']['last_calibrated_at']}\n"
                f"Sensibilidad: {self.profile['cursor_sensitivity']}\n"
                f"Zona muerta: {self.profile['dead_zone_px']}\n"
                f"Suavizado: {self.profile['smoothing_factor']}\n"
                f"Velocidad max.: {self.profile['max_cursor_speed_px']}\n"
                f"Modelo perfil: {self.classifier.model_path}\n"
                f"Resumen entrenamiento: {self.classifier.samples_path}\n"
            ),
        )
        self.profile_info.configure(state="disabled")

    def _on_setting_change(self, key, value, value_label):
        self.profile[key] = float(value)
        value_label.configure(text=f"{float(value):.2f}")

    def _save_profile_settings(self):
        save_profile(self.profile)
        self.launch_status.configure(text="Ajustes del perfil guardados.")
        self._refresh_profile_info()

    def _start_runtime(self):
        if not self.classifier.load():
            self.launch_status.configure(text="Este perfil aun no tiene un modelo entrenado. Completa calibracion.")
            self.start_button.configure(state="disabled")
            return
        save_profile(self.profile)
        save_settings(self.settings)
        self.shutdown()
        self.on_start_callback(self.profile, self.settings)

    def _sync_runtime_ready_state(self):
        model_ready = self.classifier.load()
        launcher_ready = bool(self.profile.get("launcher_completed")) and model_ready
        self.start_button.configure(state="normal" if launcher_ready else "disabled")
        if launcher_ready:
            self.launch_status.configure(text="Perfil calibrado. Puedes iniciar el sistema asistivo.")
        elif self.profile.get("launcher_completed") and not model_ready:
            self.launch_status.configure(text="Falta el modelo del perfil. Reentrena desde la calibracion.")

    def _gesture_meta(self, gesture_id: str) -> dict:
        if gesture_id == "neutral":
            return {
                "id": "neutral",
                "title": "postura neutral",
                "action": "referencia base",
                "duration_ms": 0,
                "warning": "Manten una postura relajada mirando al frente.",
            }
        for gesture in GESTURE_CATALOG:
            if gesture["id"] == gesture_id:
                return gesture
        return {
            "id": gesture_id,
            "title": gesture_id,
            "action": "-",
            "duration_ms": 0,
            "warning": "",
        }

    def _gesture_title(self, gesture_id: str) -> str:
        return self._gesture_meta(gesture_id)["title"]

    def _capture_seconds_for(self, gesture_id: str) -> float:
        if gesture_id == "neutral":
            return 3.0
        duration_ms = float(self._gesture_meta(gesture_id).get("duration_ms", 700))
        return max(2.5, (duration_ms / 1000.0) + 1.6)

    def _draw_capture_overlay(self, frame_rgb):
        if self.capture_target is None:
            return
        now = time.monotonic()
        gesture_name = self._gesture_title(self.capture_target)
        if self.capture_phase == "countdown":
            remaining = max(0.0, self.capture_countdown_deadline - now)
            title = f"Preparate: {remaining:.1f} s"
            subtitle = f"Siguiente captura: {gesture_name}"
        else:
            remaining = max(0.0, self.capture_deadline - now)
            title = f"Capturando: {remaining:.1f} s"
            subtitle = f"Manten {gesture_name} durante {self.capture_duration_seconds:.1f} s"

        cv2.rectangle(frame_rgb, (24, 24), (660, 122), (15, 23, 42), -1)
        cv2.putText(frame_rgb, title, (42, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2)
        cv2.putText(frame_rgb, subtitle, (42, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (203, 213, 225), 2)

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

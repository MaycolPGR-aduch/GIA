from __future__ import annotations

import os
import threading
import time
import webbrowser
from collections import deque

import cv2
import numpy as np
import pyautogui
import pyttsx3
import sounddevice as sd
from faster_whisper import WhisperModel
from pycaw.pycaw import AudioUtilities

from .gesture_catalog import GESTURE_CATALOG
from .gesture_ml import GestureClassifier
from .heuristic_engine import HeuristicEngine
from .landmark_provider import LandmarkProvider
from .models import AppState, FaceSample
from .session_logger import SessionLogger
from .voice_router import resolve_command


AUDIO_RECORDING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "grabacion.wav",
)


class AssistiveController:
    def __init__(self, profile: dict, settings: dict, main_gui_interface=None):
        self.profile = profile
        self.settings = settings
        self.main_gui_interface = main_gui_interface
        self.state = AppState.READY
        self.running = False
        self.control_thread = None
        self.voice_thread = None

        self.logger = SessionLogger(profile["name"])
        self.heuristics = HeuristicEngine(profile)
        self.provider = LandmarkProvider(fps=int(settings.get("fps", 15)))
        self.classifier = GestureClassifier(
            profile["name"],
            window_size=int(settings.get("gesture_window_size", 12)),
        )
        self.classifier.load()

        self.window_size = int(settings.get("gesture_window_size", 12))
        self.gesture_window: deque[FaceSample] = deque(maxlen=self.window_size)
        self.gesture_thresholds = profile.get("gesture_confidence", {})
        self.gesture_durations = profile.get("gesture_duration_ms", {})
        self.gesture_cooldowns = profile.get("gesture_cooldown_ms", {})
        self.current_gesture = None
        self.current_gesture_started_at = 0.0
        self.last_action_at: dict[str, float] = {}

        pyautogui.FAILSAFE = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.current_cursor_x, self.current_cursor_y = pyautogui.position()

        self.audio_endpoint = self._configure_audio()
        self.tts_enabled = bool(profile.get("tts_enabled", False))
        self.voice_enabled = bool(profile.get("voice_enabled", True))
        self.tts = self._configure_tts() if self.tts_enabled else None
        self.whisper = WhisperModel("base", device="cpu", compute_type="int8") if self.voice_enabled else None

        self.cap = None

    def _ui_available(self) -> bool:
        if self.main_gui_interface is None:
            return False
        try:
            return bool(self.main_gui_interface.root.winfo_exists())
        except Exception:
            return False

    def _configure_audio(self):
        try:
            speakers = AudioUtilities.GetSpeakers()
            return getattr(speakers, "EndpointVolume", None)
        except Exception as exc:
            print(f"Error configurando audio: {exc}")
            return None

    def _configure_tts(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            return engine
        except Exception as exc:
            print(f"Error configurando sintesis de voz: {exc}")
            return None

    def start(self):
        if self.running:
            return
        self.running = True
        self.logger.log("session", "Inicio de sesión", state=self.state.value)
        self.control_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.control_thread.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        self.state = AppState.STOPPED
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=3)
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.provider.close()
        self.logger.log("session", "Fin de sesión", state=self.state.value)
        try:
            self.logger.export_excel()
        except Exception as exc:
            self.logger.log("error", "No se pudo exportar log Excel", error=str(exc))

    def hablar(self, text: str):
        if self.main_gui_interface:
            self.main_gui_interface.append_event(text)
        if not self.tts:
            return
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as exc:
            print(f"Error reproduciendo TTS: {exc}")

    def toggle_pause(self):
        self.state = AppState.READY if self.state == AppState.PAUSED else AppState.PAUSED
        self.logger.log("state", "Cambio de pausa", state=self.state.value)
        self.hablar("Sistema reanudado." if self.state == AppState.READY else "Sistema en pausa.")

    def recenter(self):
        sample = self.gesture_window[-1] if self.gesture_window else None
        self.heuristics.recenter(sample)
        self.logger.log("heuristic", "Cursor recentrado")
        self.hablar("Cursor recentrado.")

    def _main_loop(self):
        self.cap = cv2.VideoCapture(int(self.settings.get("camera_index", 0)))
        if not self.cap.isOpened():
            self.state = AppState.ERROR
            self._push_status("No se pudo abrir la cámara.", "-", 0.0, face="sin cámara")
            return

        self.hablar("Controles asistenciales iniciados.")
        self.logger.log("runtime", "Runtime iniciado", profile=self.profile["name"])
        frame_delay = 1.0 / max(int(self.settings.get("fps", 15)), 1)

        while self.running:
            loop_start = time.monotonic()
            ok, frame_bgr = self.cap.read()
            if not ok:
                self.logger.log("error", "No se pudo leer frame")
                continue

            snapshot = self.provider.process(frame_bgr)
            control_state = self.heuristics.update(snapshot.face_sample)
            prediction = self._evaluate_gesture(snapshot.face_sample, control_state)
            self._apply_continuous_control(control_state)
            frame_rgb = self._render_overlay(snapshot.frame_rgb, snapshot.face_sample, control_state, prediction)
            self._push_gui_frame(frame_rgb)

            elapsed = time.monotonic() - loop_start
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    def _evaluate_gesture(self, sample: FaceSample | None, control_state):
        if sample is None:
            self.current_gesture = None
            return None

        self.gesture_window.append(sample)
        prediction = self.classifier.predict(self.gesture_window)
        if prediction is None:
            self._push_status(control_state.status_text, "-", 0.0, face="rostro detectado", mode=self.state.value)
            return None

        if prediction.gesture_id == "neutral":
            self.current_gesture = None
            self.current_gesture_started_at = 0.0
            prediction.reason = "Estado neutral"
            self._push_status(
                "Listo para gesto.",
                "neutral",
                prediction.confidence,
                face="estable" if control_state.face_stable else "inestable",
                mode=self.state.value,
            )
            return prediction

        threshold = float(self.gesture_thresholds.get(prediction.gesture_id, 0.8))
        now = time.monotonic()

        if not control_state.face_stable:
            prediction.reason = "Rostro inestable"
            self._push_status(
                "Rostro detectado pero aún inestable.",
                prediction.gesture_id,
                prediction.confidence,
                face="inestable",
                mode=self.state.value,
            )
            return prediction

        if prediction.confidence < threshold:
            prediction.reason = "Confianza baja"
            self._push_status(
                "Gesto rechazado por baja confianza.",
                prediction.gesture_id,
                prediction.confidence,
                face="estable",
                mode=self.state.value,
            )
            return prediction

        if prediction.gesture_id != self.current_gesture:
            self.current_gesture = prediction.gesture_id
            self.current_gesture_started_at = now
            prediction.reason = "Iniciando validación temporal"
            self._push_status(
                "Gesto detectado. Validando duración...",
                prediction.gesture_id,
                prediction.confidence,
                face="estable",
                mode=self.state.value,
            )
            return prediction

        required_duration = self.gesture_durations.get(prediction.gesture_id, 700) / 1000.0
        if now - self.current_gesture_started_at < required_duration:
            prediction.reason = "Esperando duración mínima"
            self._push_status(
                "Gesto detectado. Esperando duración mínima...",
                prediction.gesture_id,
                prediction.confidence,
                face="estable",
                mode=self.state.value,
            )
            return prediction

        cooldown = self.gesture_cooldowns.get(prediction.gesture_id, 1500) / 1000.0
        last_action = self.last_action_at.get(prediction.gesture_id, 0.0)
        if now - last_action < cooldown:
            prediction.reason = "En cooldown"
            self._push_status(
                "Gesto reconocido pero en cooldown.",
                prediction.gesture_id,
                prediction.confidence,
                face="estable",
                mode=self.state.value,
            )
            return prediction

        if self.state == AppState.PAUSED and prediction.gesture_id not in {"brows_up", "mouth_open_hold"}:
            prediction.reason = "Sistema en pausa"
            self._push_status(
                "Sistema en pausa. Solo gestos seguros están activos.",
                prediction.gesture_id,
                prediction.confidence,
                face="estable",
                mode=self.state.value,
            )
            return prediction

        prediction.accepted = True
        self.last_action_at[prediction.gesture_id] = now
        self._execute_gesture_action(prediction.gesture_id)
        self.logger.log(
            "gesture",
            "Gesto aceptado",
            gesture_id=prediction.gesture_id,
            confidence=round(prediction.confidence, 3),
        )
        self._push_status(
            f"Gesto ejecutado: {prediction.gesture_id}",
            prediction.gesture_id,
            prediction.confidence,
            face="estable",
            mode=self.state.value,
        )
        return prediction

    def _apply_continuous_control(self, control_state):
        if self.state != AppState.READY or not control_state.face_present:
            return
        if abs(control_state.smoothed_dx) < 0.5 and abs(control_state.smoothed_dy) < 0.5:
            return
        self.current_cursor_x = int(np.clip(self.current_cursor_x + control_state.smoothed_dx, 0, self.screen_width - 1))
        self.current_cursor_y = int(np.clip(self.current_cursor_y + control_state.smoothed_dy, 0, self.screen_height - 1))
        pyautogui.moveTo(self.current_cursor_x, self.current_cursor_y, duration=0)

    def _execute_gesture_action(self, gesture_id: str):
        if gesture_id == "left_blink_intent" and self.state == AppState.READY:
            pyautogui.click()
        elif gesture_id == "right_blink_intent" and self.state == AppState.READY:
            pyautogui.click(button="right")
        elif gesture_id == "both_eyes_closed_intent":
            self._start_voice_listener()
        elif gesture_id == "mouth_open_hold":
            if self.main_gui_interface:
                self.main_gui_interface.show_guide_dialog()
        elif gesture_id == "smile":
            self.recenter()
        elif gesture_id == "brows_up":
            self.toggle_pause()
        elif gesture_id == "confirm":
            self.logger.log("gesture", "Confirmación detectada")

    def _start_voice_listener(self):
        if not self.voice_enabled or self.whisper is None:
            self._push_status("Voz deshabilitada en este perfil.", "-", 0.0, face="estable", mode=self.state.value)
            return
        if self.voice_thread and self.voice_thread.is_alive():
            return
        self.voice_thread = threading.Thread(target=self._listen_once, daemon=True)
        self.voice_thread.start()

    def _listen_once(self):
        self.state = AppState.LISTENING
        self.hablar("Escuchando.")
        self.logger.log("voice", "Inicio escucha")
        try:
            recording = sd.rec(int(3 * 44100), samplerate=44100, channels=1, dtype="int16")
            sd.wait()
            from scipy.io.wavfile import write

            write(AUDIO_RECORDING_PATH, 44100, recording)
            segments, _ = self.whisper.transcribe(AUDIO_RECORDING_PATH, language="es")
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if not text:
                self.logger.log("voice", "Sin audio reconocido")
                self._push_status("No se escuchó un comando.", "-", 0.0, face="estable", mode=self.state.value)
                return
            command_id, confidence = resolve_command(text)
            self.logger.log("voice", "Comando transcrito", text=text, command_id=command_id, confidence=confidence)
            if command_id and confidence >= 0.68:
                self._execute_voice_command(command_id)
                self._push_status(
                    f"Comando de voz: {command_id}",
                    command_id,
                    confidence,
                    face="estable",
                    mode=self.state.value,
                )
            else:
                self._push_status("Comando de voz no reconocido.", "-", confidence, face="estable", mode=self.state.value)
        except Exception as exc:
            self.logger.log("error", "Fallo en voz", error=str(exc))
            self._push_status(f"Error en voz: {exc}", "-", 0.0, face="estable", mode=self.state.value)
        finally:
            if self.state == AppState.LISTENING:
                self.state = AppState.READY

    def _execute_voice_command(self, command_id: str):
        if command_id == "pause":
            if self.state != AppState.PAUSED:
                self.toggle_pause()
        elif command_id == "resume":
            if self.state == AppState.PAUSED:
                self.toggle_pause()
        elif command_id == "recenter":
            self.recenter()
        elif command_id == "guide" and self.main_gui_interface:
            self.main_gui_interface.show_guide_dialog()
        elif command_id == "quit":
            self.stop()
            if self.main_gui_interface:
                self.main_gui_interface.root.after(0, self.main_gui_interface._quit)
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

    def _set_volume(self, percent: int):
        if self.audio_endpoint is None:
            return
        try:
            self.audio_endpoint.SetMasterVolumeLevelScalar(percent / 100.0, None)
            self.logger.log("audio", "Cambio de volumen", percent=percent)
        except Exception as exc:
            self.logger.log("error", "No se pudo cambiar volumen", error=str(exc))

    def _render_overlay(self, frame_rgb, sample, control_state, prediction):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        if sample is not None:
            for point in sample.points_px.values():
                cv2.circle(frame_bgr, point, 1, (60, 255, 80), -1)
            cv2.circle(frame_bgr, sample.nose_px, 5, (0, 128, 255), -1)

        cv2.putText(frame_bgr, f"Estado: {self.state.value}", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_bgr, f"{control_state.status_text}", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if prediction is not None:
            cv2.putText(
                frame_bgr,
                f"Gesto: {prediction.gesture_id} ({prediction.confidence:.2f})",
                (16, 84),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 220, 140),
                2,
            )
        metric_y = 116
        for key, value in control_state.debug_metrics.items():
            cv2.putText(
                frame_bgr,
                f"{key}: {value:.3f}",
                (16, metric_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 255, 200),
                1,
            )
            metric_y += 22
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _push_gui_frame(self, frame_rgb):
        if not self._ui_available():
            return
        try:
            self.main_gui_interface.root.after(0, lambda: self.main_gui_interface.update_video_feed(frame_rgb))
        except Exception:
            pass

    def _push_status(self, status_text, gesture, confidence, *, face="-", mode=None):
        payload = {
            "status_text": status_text,
            "gesture": gesture,
            "confidence": f"{confidence:.2f}" if isinstance(confidence, (int, float)) else confidence,
            "face": face,
            "mode": mode or self.state.value,
            "state": self.state.value,
        }
        if not self._ui_available():
            return
        try:
            self.main_gui_interface.root.after(0, lambda: self.main_gui_interface.update_status(payload))
        except Exception:
            pass

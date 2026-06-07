from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import pyautogui
import pyttsx3
from faster_whisper import WhisperModel
from PySide6.QtCore import QTimer
from pycaw.pycaw import AudioUtilities

from .action_service import ActionService
from .audio_utils import AudioLevelMonitor, capture_voice_clip, save_audio_clip
from .config_store import parse_camera_resolution
from .gesture_ml import GestureClassifier
from .heuristic_engine import HeuristicEngine
from .inference_service import InferenceService
from .landmark_provider import LandmarkProvider
from .models import AppState, FaceSample, GesturePrediction
from .session_logger import SessionLogger
from .voice_router import resolve_command_entry


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
        self.window_size = int(settings.get("gesture_window_size", 12))
        self.classifier = GestureClassifier(profile["name"], window_size=self.window_size)
        self.classifier.load()
        if self.classifier.window_size != self.window_size:
            self.window_size = self.classifier.window_size
        self.inference = InferenceService(profile, self.window_size)

        pyautogui.FAILSAFE = False
        self.screen_width, self.screen_height = pyautogui.size()
        self.current_cursor_x, self.current_cursor_y = pyautogui.position()
        self.cursor_active = bool(profile.get("cursor_starts_active", False))
        self.mirror_preview = bool(profile.get("mirror_preview", True))

        self.audio_endpoint = self._configure_audio()
        self.tts_enabled = bool(profile.get("tts_enabled", False))
        self.voice_commands_enabled = bool(settings.get("voice_commands_enabled", True))
        self.voice_enabled = bool(profile.get("voice_enabled", True)) and self.voice_commands_enabled
        self.audio_input_device = settings.get("audio_input_device")
        self.voice_sample_rate = int(settings.get("voice_sample_rate", 16000))
        self.voice_record_seconds = float(settings.get("voice_record_seconds", 3.0))
        self.voice_model_size = str(settings.get("voice_model_size", "small"))
        self.tts = self._configure_tts() if self.tts_enabled else None
        self.whisper = WhisperModel(self.voice_model_size, device="cpu", compute_type="int8") if self.voice_enabled else None
        self.audio_level_monitor = (
            AudioLevelMonitor(self.voice_sample_rate, device=self.audio_input_device)
            if self.voice_commands_enabled
            else None
        )
        self.action_service = ActionService(
            self.logger,
            on_voice_listener=self._start_voice_listener,
            on_toggle_cursor=self.toggle_cursor_control,
            on_recenter=self.recenter,
            on_toggle_pause=self.toggle_pause,
            on_set_cursor=self.set_cursor_control,
            on_quit=self._quit_from_action,
            main_gui_interface=self.main_gui_interface,
            audio_endpoint=self.audio_endpoint,
        )

        self.cap = None
        self.last_voice_text = "-"
        self.last_voice_command = "-"
        self.last_voice_state = "en espera"
        self.last_rejection_reason = "-"
        self.last_diagnostic_hint = "-"
        self.show_camera_overlay_details = False
        self.hybrid_eye_closed_confidence_floor = 0.18
        self.hybrid_eye_closed_ratio_max = 0.19
        self.hybrid_eye_closed_asymmetry_max = 0.08
        self.heuristic_blink_closed_ratio = 0.17
        self.heuristic_blink_open_ratio = 0.20
        self.heuristic_blink_asymmetry_min = 0.04
        self.heuristic_blink_confidence = 0.72

    def _ui_available(self) -> bool:
        if self.main_gui_interface is None:
            return False
        if hasattr(self.main_gui_interface, "root"):
            try:
                return bool(self.main_gui_interface.root.winfo_exists())
            except Exception:
                return False
        try:
            if self.main_gui_interface.isVisible():
                return True
            compact_window = getattr(self.main_gui_interface, "compact_window", None)
            return bool(compact_window is not None and compact_window.isVisible())
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
        if self.audio_level_monitor is not None:
            started = self.audio_level_monitor.start()
            if not started:
                snapshot = self.audio_level_monitor.snapshot()
                self.logger.log(
                    "voice_audio",
                    "No se pudo iniciar monitor de microfono",
                    device=snapshot.get("device_label"),
                    error=snapshot.get("error", ""),
                )
        self.logger.log(
            "session",
            "Inicio de sesión",
            state=self.state.value,
            model_version=self.classifier.active_version,
        )
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
        if self.audio_level_monitor is not None:
            self.audio_level_monitor.stop()
        self.provider.close()
        self.logger.log("session", "Fin de sesión", state=self.state.value)
        try:
            self.logger.export_excel()
        except Exception as exc:
            self.logger.log("error", "No se pudo exportar log Excel", error=str(exc))

    def hablar(self, text: str):
        self._publish_event(text)
        if not self.tts:
            return
        try:
            self.tts.say(text)
            self.tts.runAndWait()
        except Exception as exc:
            print(f"Error reproduciendo TTS: {exc}")

    def _publish_event(self, text: str):
        if self.main_gui_interface:
            if hasattr(self.main_gui_interface, "signals"):
                self.main_gui_interface.signals.event_update.emit(text)
            else:
                self.main_gui_interface.append_event(text)


    def toggle_cursor_control(self):
        self.cursor_active = not self.cursor_active
        self.logger.log("cursor", "Cambio de clutch", cursor_active=self.cursor_active)
        self.hablar("Cursor activado." if self.cursor_active else "Cursor congelado.")

    def set_cursor_control(self, enabled: bool):
        if self.cursor_active == enabled:
            return
        self.cursor_active = enabled
        self.logger.log("cursor", "Cursor configurado", cursor_active=self.cursor_active)
        self.hablar("Cursor activado." if self.cursor_active else "Cursor congelado.")

    def set_camera_overlay_details(self, enabled: bool):
        self.show_camera_overlay_details = bool(enabled)

    def toggle_pause(self):
        self.state = AppState.READY if self.state == AppState.PAUSED else AppState.PAUSED
        self.logger.log("state", "Cambio de pausa", state=self.state.value)
        self.hablar("Sistema reanudado." if self.state == AppState.READY else "Sistema en pausa.")

    def recenter(self):
        sample = self.inference.gesture_window[-1] if self.inference.gesture_window else None
        self.heuristics.recenter(sample)
        self.logger.log("heuristic", "Cursor recentrado")
        self.hablar("Cursor recentrado.")

    def _main_loop(self):
        self.cap = cv2.VideoCapture(int(self.settings.get("camera_index", 0)))
        self._apply_camera_resolution(self.cap)
        if not self.cap.isOpened():
            self.state = AppState.ERROR
            self._push_status("No se pudo abrir la cámara.", "-", 0.0, face="sin cámara")
            return

        self.hablar("Controles asistenciales iniciados.")
        self.logger.log(
            "runtime",
            "Runtime iniciado",
            profile=self.profile["name"],
            model_version=self.classifier.active_version,
        )
        frame_delay = 1.0 / max(int(self.settings.get("fps", 15)), 1)

        while self.running:
            loop_start = time.monotonic()
            ok, frame_bgr = self.cap.read()
            if not ok:
                self.logger.log("error", "No se pudo leer frame")
                continue

            snapshot = self.provider.process(frame_bgr)
            control_state = self.heuristics.update(snapshot.face_sample)
            prediction, inference_result = self._evaluate_gesture(snapshot.face_sample, control_state)
            self._apply_continuous_control(control_state)
            frame_rgb = self._render_overlay(snapshot.frame_rgb, snapshot.face_sample, control_state, prediction, compact=False)
            compact_frame_rgb = self._render_overlay(snapshot.frame_rgb, snapshot.face_sample, control_state, prediction, compact=True)
            self._push_gui_frame(frame_rgb, compact_frame_rgb)

            elapsed = time.monotonic() - loop_start
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    def _evaluate_gesture(self, sample: FaceSample | None, control_state):
        if sample is not None:
            self.inference.append_sample(sample)
        prediction = self.classifier.predict(self.inference.gesture_window)
        heuristic_prediction = self._build_heuristic_blink_prediction(sample)
        if self._should_use_heuristic_blink(prediction, heuristic_prediction):
            prediction = heuristic_prediction

        allow_low_confidence = bool(prediction and self._passes_hybrid_gesture_validation(prediction, sample))
        result = self.inference.evaluate(
            sample,
            control_state,
            prediction,
            self.state,
            self._cursor_status_text(),
            allow_low_confidence=allow_low_confidence,
        )
        self.last_rejection_reason = result.rejection_reason or "-"
        self.last_diagnostic_hint = result.diagnostic_hint or "-"

        if result.executed and result.prediction is not None:
            self.action_service.execute_gesture_action(result.prediction.gesture_id, self.state.value)
            self.logger.log(
                "gesture",
                "Gesto aceptado",
                gesture_id=result.prediction.gesture_id,
                prediction_confidence=round(result.prediction.confidence, 3),
                rejection_reason="",
                model_version=self.classifier.active_version,
                stability=control_state.face_stable,
            )
        elif result.should_log and result.prediction is not None:
            self.logger.log(
                "gesture_rejected",
                "Gesto rechazado",
                gesture_id=result.prediction.gesture_id,
                prediction_confidence=round(result.prediction.confidence, 3),
                rejection_reason=result.rejection_reason,
                model_version=self.classifier.active_version,
                stability=control_state.face_stable,
            )

        self._push_status(
            result.status_text,
            result.prediction.gesture_id if result.prediction else "-",
            result.prediction.confidence if result.prediction else 0.0,
            face=result.face_label,
            mode=result.mode_label,
            cursor=result.cursor_label,
            rejection_reason=result.rejection_reason or result.prediction.reason if result.prediction else result.rejection_reason,
            diagnostic_hint=result.diagnostic_hint,
        )
        return prediction, result

    def _apply_continuous_control(self, control_state):
        if self.state != AppState.READY or not control_state.face_present or not self.cursor_active:
            return
        if abs(control_state.smoothed_dx) < 0.5 and abs(control_state.smoothed_dy) < 0.5:
            return
        self.current_cursor_x = int(np.clip(self.current_cursor_x + control_state.smoothed_dx, 0, self.screen_width - 1))
        self.current_cursor_y = int(np.clip(self.current_cursor_y + control_state.smoothed_dy, 0, self.screen_height - 1))
        pyautogui.moveTo(self.current_cursor_x, self.current_cursor_y, duration=0)

    def start_manual_voice_test(self):
        self._start_voice_listener(trigger_source="manual")

    def _start_voice_listener(self, trigger_source: str = "gesture"):
        if not self.voice_enabled or self.whisper is None:
            self.last_voice_state = "deshabilitada"
            self._push_status(
                "Voz deshabilitada en este perfil.",
                "-",
                0.0,
                face="estable",
                mode=self.state.value,
                cursor=self._cursor_status_text(),
            )
            return
        if self.voice_thread and self.voice_thread.is_alive():
            self._publish_event("Ya hay una escucha de voz en progreso.")
            return
        self.voice_thread = threading.Thread(
            target=self._listen_once,
            kwargs={"trigger_source": trigger_source},
            daemon=True,
        )
        self.voice_thread.start()

    def _get_mic_status_payload(self) -> dict:
        if self.audio_level_monitor is None:
            return {
                "mic_device": "Monitor deshabilitado",
                "mic_level": 0,
                "mic_peak": 0,
                "mic_active_ratio": 0.0,
                "mic_monitor_status": "deshabilitado",
                "mic_monitor_error": "",
            }
        snapshot = self.audio_level_monitor.snapshot()
        if not snapshot.get("available"):
            status = "error" if snapshot.get("error") else "sin señal"
        elif snapshot.get("suspended"):
            status = "muestreo en pausa"
        else:
            status = "activo"
        return {
            "mic_device": snapshot.get("device_label", "Desconocido"),
            "mic_level": int(snapshot.get("level_percent", 0)),
            "mic_peak": int(snapshot.get("peak_percent", 0)),
            "mic_active_ratio": float(snapshot.get("active_ratio", 0.0)),
            "mic_monitor_status": status,
            "mic_monitor_error": snapshot.get("error", ""),
        }

    def _listen_once(self, trigger_source: str = "gesture"):
        previous_state = self.state
        self.state = AppState.LISTENING
        self.last_voice_state = "escuchando"
        self.last_voice_text = "-"
        self.last_voice_command = "-"
        self._push_status(
            "Escuchando comando de voz...",
            "-",
            0.0,
            face="estable",
            mode=self.state.value,
            cursor=self._cursor_status_text(),
        )
        self.hablar("Escuchando.")
        self.logger.log(
            "voice",
            "Inicio escucha",
            trigger_source=trigger_source,
            device=self.audio_input_device,
            duration_s=self.voice_record_seconds,
            sample_rate=self.voice_sample_rate,
        )
        try:
            if self.audio_level_monitor is not None:
                self.audio_level_monitor.suspend()
            recording, audio_diag = capture_voice_clip(
                self.voice_record_seconds,
                self.voice_sample_rate,
                device=self.audio_input_device,
            )
            save_audio_clip(AUDIO_RECORDING_PATH, recording, self.voice_sample_rate)
            self.logger.log("voice_audio", "Audio capturado", **audio_diag)
            self._publish_event(
                "Audio capturado | "
                f"Mic: {audio_diag['device_label']} | "
                f"RMS: {audio_diag['rms']:.4f} | "
                f"Pico: {audio_diag['peak']:.4f}"
            )
            segments, _ = self.whisper.transcribe(
                AUDIO_RECORDING_PATH,
                language="es",
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if not text:
                self.logger.log("voice", "Sin audio reconocido")
                self.last_voice_state = "sin audio reconocido"
                self.last_voice_text = "(sin audio)"
                self.last_voice_command = "-"
                self._publish_event(
                    "Voz sin transcripcion reconocible. "
                    f"Mic: {audio_diag['device_label']} | "
                    f"RMS: {audio_diag['rms']:.4f}"
                )
                self._push_status(
                    "No se escucho un comando.",
                    "-",
                    0.0,
                    face="estable",
                    mode=previous_state.value,
                    cursor=self._cursor_status_text(),
                    rejection_reason="sin transcripcion",
                    diagnostic_hint=(
                        f"Mic: {audio_diag['device_label']} | "
                        f"RMS: {audio_diag['rms']:.4f} | "
                        f"Actividad: {audio_diag['active_ratio']:.4f}"
                    ),
                )
                return
            command_entry, confidence = resolve_command_entry(text, self.settings)
            command_id = command_entry["id"] if command_entry else None
            self.last_voice_text = text
            self.last_voice_command = command_entry["label"] if command_entry else "-"
            self._publish_event(f'Texto transcrito: "{text}"')
            self.logger.log(
                "voice",
                "Comando transcrito",
                text=text,
                command_id=command_id,
                command_kind=command_entry["kind"] if command_entry else None,
                confidence=confidence,
                model_version=self.classifier.active_version,
            )
            if command_entry and confidence >= 0.68:
                self.last_voice_state = "comando reconocido"
                self._publish_event(f"Comando reconocido: {command_entry['label']} ({confidence:.2f})")
                if command_entry["kind"] == "builtin":
                    self.action_service.execute_builtin_voice_command(command_id, previous_state.value)
                else:
                    self.action_service.execute_custom_voice_command(command_entry, previous_state.value)
                self._push_status(
                    f"Comando de voz: {command_entry['label']}",
                    command_entry["label"],
                    confidence,
                    face="estable",
                    mode=self.state.value,
                    cursor=self._cursor_status_text(),
                    diagnostic_hint=f"Mic: {audio_diag['device_label']} | ASR: {self.voice_model_size}",
                )
            else:
                self.last_voice_state = "comando no reconocido"
                self._publish_event(f'Comando no reconocido ({confidence:.2f}): "{text}"')
                self._push_status(
                    "Comando de voz no reconocido.",
                    "-",
                    confidence,
                    face="estable",
                    mode=previous_state.value,
                    cursor=self._cursor_status_text(),
                    rejection_reason="voz no reconocida",
                    diagnostic_hint="El texto se transcribio pero no coincidió con aliases confiables.",
                )
        except Exception as exc:
            self.logger.log("error", "Fallo en voz", error=str(exc))
            self.last_voice_state = "error"
            self.last_voice_text = str(exc)
            self.last_voice_command = "-"
            self._publish_event(f"Error en voz: {exc}")
            self._push_status(
                f"Error en voz: {exc}",
                "-",
                0.0,
                face="estable",
                mode=previous_state.value,
                cursor=self._cursor_status_text(),
                rejection_reason="error de voz",
                diagnostic_hint=str(exc),
            )
        finally:
            if self.audio_level_monitor is not None:
                self.audio_level_monitor.resume()
            if self.state == AppState.LISTENING:
                self.state = previous_state if previous_state != AppState.LISTENING else AppState.READY
            if self.last_voice_state == "escuchando":
                self.last_voice_state = "en espera"

    def _quit_from_action(self):
        self.stop()
        if self.main_gui_interface:
            if hasattr(self.main_gui_interface, "_quit"):
                QTimer.singleShot(0, self.main_gui_interface._quit)
            elif hasattr(self.main_gui_interface, "close"):
                QTimer.singleShot(0, self.main_gui_interface.close)

    def _render_overlay(self, frame_rgb, sample, control_state, prediction, compact: bool = False):
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_height, frame_width = frame_bgr.shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)
        neutral_point = (
            int(self.heuristics.neutral_nose[0]) if self.heuristics.neutral_nose else frame_center[0],
            int(self.heuristics.neutral_nose[1]) if self.heuristics.neutral_nose else frame_center[1],
        )

        cv2.line(frame_bgr, (frame_center[0] - 18, frame_center[1]), (frame_center[0] + 18, frame_center[1]), (255, 180, 0), 1)
        cv2.line(frame_bgr, (frame_center[0], frame_center[1] - 18), (frame_center[0], frame_center[1] + 18), (255, 180, 0), 1)
        cv2.circle(frame_bgr, frame_center, 4, (255, 180, 0), 1)
        cv2.circle(frame_bgr, neutral_point, 8, (255, 0, 255), 2)
        cv2.line(frame_bgr, (neutral_point[0] - 10, neutral_point[1]), (neutral_point[0] + 10, neutral_point[1]), (255, 0, 255), 2)
        cv2.line(frame_bgr, (neutral_point[0], neutral_point[1] - 10), (neutral_point[0], neutral_point[1] + 10), (255, 0, 255), 2)

        if sample is not None and not compact:
            for point in sample.points_px.values():
                cv2.circle(frame_bgr, point, 1, (60, 255, 80), -1)
            cv2.circle(frame_bgr, sample.nose_px, 5, (0, 128, 255), -1)
        if self.mirror_preview:
            frame_bgr = cv2.flip(frame_bgr, 1)

        if compact:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        cv2.putText(frame_bgr, "Centro camara", (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)
        cv2.putText(frame_bgr, "Referencia cursor", (16, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)

        if prediction is not None:
            overlay = frame_bgr.copy()
            badge_left = 18
            badge_top = 82
            badge_right = min(frame_width - 18, 440)
            badge_bottom = 132
            cv2.rectangle(overlay, (badge_left, badge_top), (badge_right, badge_bottom), (22, 163, 74), -1)
            frame_bgr = cv2.addWeighted(overlay, 0.22, frame_bgr, 0.78, 0)
            cv2.putText(
                frame_bgr,
                prediction.gesture_id,
                (34, 116),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.82,
                (210, 255, 220),
                2,
            )

        if self.show_camera_overlay_details:
            detail_y = 178 if prediction is not None else 92
            cv2.putText(frame_bgr, f"Estado: {self.state.value}", (16, detail_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(
                frame_bgr,
                f"{control_state.status_text} | cursor: {self._cursor_status_text()}",
                (16, detail_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        if prediction is not None and self.show_camera_overlay_details:
            cv2.putText(
                frame_bgr,
                f"Gesto: {prediction.gesture_id} ({prediction.confidence:.2f})",
                (16, detail_y + 54),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 220, 140),
                2,
            )
            cv2.putText(
                frame_bgr,
                f"Motivo: {prediction.reason}",
                (16, detail_y + 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 140),
                1,
            )
        if self.show_camera_overlay_details:
            metric_y = (detail_y + 110) if prediction is not None else 202
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

    def _cursor_status_text(self) -> str:
        return "activo" if self.cursor_active else "congelado"

    def _build_heuristic_blink_prediction(self, sample: FaceSample | None) -> GesturePrediction | None:
        if sample is None:
            return None

        left_eye_ratio = float(sample.metrics.get("left_eye_ratio", 1.0))
        right_eye_ratio = float(sample.metrics.get("right_eye_ratio", 1.0))
        asymmetry = abs(left_eye_ratio - right_eye_ratio)

        if left_eye_ratio <= self.hybrid_eye_closed_ratio_max and right_eye_ratio <= self.hybrid_eye_closed_ratio_max:
            return None

        if (
            left_eye_ratio <= self.heuristic_blink_closed_ratio
            and right_eye_ratio >= self.heuristic_blink_open_ratio
            and (right_eye_ratio - left_eye_ratio) >= self.heuristic_blink_asymmetry_min
            and asymmetry >= self.heuristic_blink_asymmetry_min
        ):
            return GesturePrediction(
                gesture_id="left_blink_intent",
                confidence=self.heuristic_blink_confidence,
                accepted=False,
                reason="Fallback heuristico ocular",
            )

        if (
            right_eye_ratio <= self.heuristic_blink_closed_ratio
            and left_eye_ratio >= self.heuristic_blink_open_ratio
            and (left_eye_ratio - right_eye_ratio) >= self.heuristic_blink_asymmetry_min
            and asymmetry >= self.heuristic_blink_asymmetry_min
        ):
            return GesturePrediction(
                gesture_id="right_blink_intent",
                confidence=self.heuristic_blink_confidence,
                accepted=False,
                reason="Fallback heuristico ocular",
            )
        return None

    def _should_use_heuristic_blink(self, prediction: GesturePrediction | None, heuristic_prediction: GesturePrediction | None) -> bool:
        if heuristic_prediction is None:
            return False
        if prediction is None:
            return True
        if prediction.gesture_id == "neutral":
            return True
        if prediction.gesture_id not in {"left_blink_intent", "right_blink_intent"}:
            return False
        threshold = float(self.profile.get("gesture_confidence", {}).get(prediction.gesture_id, 0.5))
        return prediction.confidence < threshold

    def _passes_hybrid_gesture_validation(self, prediction, sample: FaceSample | None) -> bool:
        if sample is None:
            return False
        if prediction.gesture_id != "both_eyes_closed_intent":
            return False
        left_eye_ratio = float(sample.metrics.get("left_eye_ratio", 1.0))
        right_eye_ratio = float(sample.metrics.get("right_eye_ratio", 1.0))
        mean_ratio = (left_eye_ratio + right_eye_ratio) / 2.0
        asymmetry = abs(left_eye_ratio - right_eye_ratio)
        return (
            prediction.confidence >= self.hybrid_eye_closed_confidence_floor
            and left_eye_ratio <= self.hybrid_eye_closed_ratio_max
            and right_eye_ratio <= self.hybrid_eye_closed_ratio_max
            and mean_ratio <= self.hybrid_eye_closed_ratio_max
            and asymmetry <= self.hybrid_eye_closed_asymmetry_max
        )

    def _push_gui_frame(self, frame_rgb, compact_frame_rgb=None):
        if not self._ui_available():
            return
        try:
            if hasattr(self.main_gui_interface, "signals"):
                self.main_gui_interface.signals.video_update.emit(frame_rgb, compact_frame_rgb)
            elif hasattr(self.main_gui_interface, "root"):
                self.main_gui_interface.root.after(
                    0,
                    lambda: self.main_gui_interface.update_video_feed(frame_rgb, compact_frame_rgb),
                )
        except Exception:
            pass

    def _push_status(
        self,
        status_text,
        gesture,
        confidence,
        *,
        face="-",
        mode=None,
        cursor=None,
        voice_state=None,
        voice_text=None,
        voice_command=None,
        rejection_reason=None,
        diagnostic_hint=None,
    ):
        mic_payload = self._get_mic_status_payload()
        payload = {
            "status_text": status_text,
            "gesture": gesture,
            "confidence": f"{confidence:.2f}" if isinstance(confidence, (int, float)) else confidence,
            "face": face,
            "mode": mode or self.state.value,
            "cursor": cursor or self._cursor_status_text(),
            "voice_state": voice_state or self.last_voice_state,
            "voice_text": voice_text or self.last_voice_text,
            "voice_command": voice_command or self.last_voice_command,
            "state": self.state.value,
            "rejection_reason": rejection_reason if rejection_reason is not None else self.last_rejection_reason,
            "diagnostic_hint": diagnostic_hint if diagnostic_hint is not None else self.last_diagnostic_hint,
            "model_version": self.classifier.active_version if self.classifier.active_version is not None else "-",
            **mic_payload,
        }
        if not self._ui_available():
            return
        try:
            if hasattr(self.main_gui_interface, "signals"):
                self.main_gui_interface.signals.status_update.emit(payload)
            elif hasattr(self.main_gui_interface, "root"):
                self.main_gui_interface.root.after(0, lambda: self.main_gui_interface.update_status(payload))
        except Exception:
            pass


    def _apply_camera_resolution(self, cap):
        width, height = parse_camera_resolution(self.settings.get("camera_resolution"))
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except Exception:
            pass

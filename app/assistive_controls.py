from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import pyautogui
from faster_whisper import WhisperModel
from pycaw.pycaw import AudioUtilities

from .action_service import ActionService
from .audio_utils import AudioLevelMonitor, capture_voice_clip, friendly_audio_error, save_audio_clip
from .config_store import parse_camera_resolution
from .gesture_catalog import GESTURE_CATALOG, clamp_confidence_threshold
from .gesture_ml import GestureClassifier, ProbabilitySmoother
from .global_hotkey import (
    EMERGENCY_HOTKEY_LABEL,
    MARK_ERROR_HOTKEY_LABEL,
    MARK_SEGMENT_HOTKEY_LABEL,
    GlobalHotkey,
)
from .heuristic_engine import HeuristicEngine
from .inference_service import InferenceService
from .landmark_provider import LandmarkProvider
from .models import AppState, FaceSample, GesturePrediction
from .session_logger import SessionLogger
from .speech import SpeechQueue
from .usage_recorder import UsageRecorder
from .voice_router import resolve_command_entry


# Ancho máximo del fotograma que se envía a la interfaz (se reduce en el hilo de trabajo).
GUI_FRAME_WIDTH = 960
COMPACT_FRAME_WIDTH = 200
STATUS_PUSH_INTERVAL_S = 0.1
# Lecturas fallidas seguidas (~1 s a 15 FPS) antes de dar la cámara por perdida.
CAMERA_MAX_READ_FAILURES = 15
CAMERA_RETRY_INTERVAL_S = 2.0
CONTROL_THREAD_STOP_TIMEOUT_S = 8.0
# Las velocidades del cursor del perfil están ajustadas a 15 FPS; se escalan por el tiempo
# real entre fotogramas para que la velocidad no dependa de los FPS.
REFERENCE_CURSOR_FPS = 15.0
PRECISION_SPEED_FACTOR = 0.35
# Gestos que hacen clic en la posición del cursor: lo congelan mientras se validan.
POSITIONAL_GESTURES = {"left_blink_intent", "right_blink_intent"}
BROW_SIGNAL_CONFIDENCE = 0.9
# Aviso de cara fuera de la zona calibrada: tolera movimientos breves para no parpadear.
ZONE_HINT_ON_S = 1.0
ZONE_HINT_OFF_S = 0.5
ZONE_HINTS = {"pose": "Mira al frente", "far": "Acércate a la cámara", "near": "Aléjate un poco"}
GESTURE_LABELS = {
    gesture["id"]: (gesture.get("short_title", gesture["title"]), gesture.get("short_action", gesture["action"]))
    for gesture in GESTURE_CATALOG
}

AUDIO_RECORDING_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "grabacion.wav",
)


_whisper_models: dict[str, WhisperModel] = {}
_whisper_lock = threading.Lock()


def get_shared_whisper(model_size: str) -> WhisperModel:
    """Un solo modelo por tamaño en todo el proceso: cargarlo tarda ~10 s y no cambia entre sesiones."""
    with _whisper_lock:
        if model_size not in _whisper_models:
            _whisper_models[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
        return _whisper_models[model_size]


class AssistiveController:
    def __init__(self, profile: dict, settings: dict, main_gui_interface=None):
        self.profile = profile
        self.settings = settings
        self.main_gui_interface = main_gui_interface
        self.state = AppState.READY
        self.running = False
        self.control_thread = None
        self.voice_thread = None
        self._stopped = False

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
        # Por defecto pyautogui duerme 100 ms tras cada llamada: con el cursor moviéndose
        # frenaba todo el bucle de cámara a 8 FPS.
        pyautogui.PAUSE = 0
        self.screen_width, self.screen_height = pyautogui.size()
        self.current_cursor_x, self.current_cursor_y = pyautogui.position()
        # Posición en coma flotante para no perder los desplazamientos de menos de 1 px.
        self._cursor_fx, self._cursor_fy = float(self.current_cursor_x), float(self.current_cursor_y)
        self.precision_mode = False
        self._frame_dt = None
        self._click_anchor = None
        self.prediction_smoother = ProbabilitySmoother(float(settings.get("gesture_smoothing_alpha", 0.5)))
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
        # Whisper se carga en segundo plano al iniciar (start) para no congelar la interfaz.
        self.whisper = None
        self.voice_model_state = "cargando" if self.voice_enabled else "deshabilitada"
        self.voice_model_error = ""
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
            on_set_precision=self.set_precision_mode,
        )

        self.emergency_hotkey = GlobalHotkey(self._emergency_toggle_pause)
        self.usage_recorder = None
        self._usage_hotkeys = []
        self.zone_hint = ""
        self._zone_issue_since = None
        self._zone_clear_since = None
        self.cap = None
        self.camera_state = "abriendo"
        self._camera_ok_since = None
        self._state_before_camera_loss = None
        self._last_face_seen_at = None
        self.last_voice_text = "-"
        self.last_voice_command = "-"
        self.last_voice_state = "en espera"
        self.last_rejection_reason = "-"
        self.last_diagnostic_hint = "-"
        self.show_camera_overlay_details = False
        self.fps_real = 0.0
        self._last_frame_time = None
        self._last_status_push = 0.0
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
        return SpeechQueue()

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
                self._publish_event(friendly_audio_error(snapshot.get("error", "")))
        if self.emergency_hotkey.start():
            self._publish_event(f"Pausa de emergencia disponible en cualquier app: {EMERGENCY_HOTKEY_LABEL}")
        else:
            self._publish_event(f"No se pudo registrar {EMERGENCY_HOTKEY_LABEL} (¿lo usa otra aplicación?).")
        self.logger.log(
            "session",
            "Inicio de sesión",
            state=self.state.value,
            model_version=self.classifier.active_version,
        )
        self.control_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.control_thread.start()
        if self.voice_enabled and self.whisper is None:
            threading.Thread(target=self._load_voice_model, name="gia-whisper", daemon=True).start()

    def _load_voice_model(self):
        try:
            model = get_shared_whisper(self.voice_model_size)
        except Exception as exc:
            self.voice_model_error = str(exc)
            self.voice_model_state = "error"
            self.logger.log("error", "No se pudo cargar el modelo de voz", error=str(exc))
            self._publish_event(f"Reconocimiento de voz no disponible: {exc}")
            return
        self.whisper = model
        self.voice_model_state = "lista"
        self._publish_event("Reconocimiento de voz listo.")

    def preflight_snapshot(self) -> dict:
        """Foto del estado para la verificación previa (se lee desde la interfaz)."""
        now = time.monotonic()
        monitor = self.audio_level_monitor.snapshot() if self.audio_level_monitor is not None else {}
        summary = self.classifier.training_summary or {}
        return {
            "camera_state": self.camera_state,
            "camera_ok_for_s": now - self._camera_ok_since if self._camera_ok_since else 0.0,
            "fps_real": float(self.fps_real or 0.0),
            "face_seen_ago_s": now - self._last_face_seen_at if self._last_face_seen_at else None,
            "mic_monitor_enabled": self.audio_level_monitor is not None,
            "mic_available": bool(monitor.get("available")),
            "mic_error": monitor.get("error", ""),
            "mic_device": monitor.get("device_label", ""),
            "mic_peak": float(monitor.get("peak", 0.0) or 0.0),
            "voice_model_state": self.voice_model_state,
            "voice_model_error": self.voice_model_error,
            "model_loaded": self.classifier.model is not None,
            "model_version": self.classifier.active_version,
            "model_feature_version": self.classifier.feature_version,
            "model_needs_refresh": self.classifier.needs_refresh,
            "zone_hint": self.zone_hint,
            "model_accuracy": summary.get("validation_accuracy"),
            "hotkey_registered": bool(self.emergency_hotkey.registered),
        }

    def stop(self):
        """Libera todos los recursos. Es idempotente y sirve aunque nunca se haya iniciado."""
        if self._stopped:
            return
        self._stopped = True
        self.running = False
        self.state = AppState.STOPPED
        self.emergency_hotkey.stop()
        self._usage_summary_on_stop = self.stop_usage_recording()
        # El hilo de control es el dueño de la cámara y de MediaPipe: hasta que termine no se tocan.
        loop_finished = True
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=CONTROL_THREAD_STOP_TIMEOUT_S)
            loop_finished = not self.control_thread.is_alive()
        if self.audio_level_monitor is not None:
            self.audio_level_monitor.stop()
        if loop_finished:
            self.provider.close()
        else:
            self.logger.log("error", "El bucle de cámara no terminó a tiempo")
        if self.tts:
            self.tts.stop()
        self.logger.log("session", "Fin de sesión", state=self.state.value)
        try:
            self.logger.export_excel()
        except Exception as exc:
            self.logger.log("error", "No se pudo exportar log Excel", error=str(exc))

    def hablar(self, text: str, *, wait: bool = False):
        """Publica el aviso y lo dice en voz alta sin bloquear (wait=True espera a terminar)."""
        self._publish_event(text)
        if self.tts:
            self.tts.say(text, wait=wait)

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

    def set_precision_mode(self, enabled: bool):
        if self.precision_mode == enabled:
            return
        self.precision_mode = enabled
        self.logger.log("cursor", "Modo preciso", precision_mode=enabled)
        self.hablar("Modo preciso activado." if enabled else "Velocidad normal.")

    def toggle_precision_mode(self):
        self.set_precision_mode(not self.precision_mode)

    def set_camera_overlay_details(self, enabled: bool):
        self.show_camera_overlay_details = bool(enabled)

    def toggle_pause(self):
        self.state = AppState.READY if self.state == AppState.PAUSED else AppState.PAUSED
        self.logger.log("state", "Cambio de pausa", state=self.state.value)
        self.hablar("Sistema reanudado." if self.state == AppState.READY else "Sistema en pausa.")

    def _emergency_toggle_pause(self):
        self.logger.log("state", "Atajo global de pausa", hotkey=EMERGENCY_HOTKEY_LABEL)
        self.toggle_pause()

    def recenter(self):
        sample = self.inference.gesture_window[-1] if self.inference.gesture_window else None
        self.heuristics.recenter(sample)
        self.logger.log("heuristic", "Cursor recentrado")
        self.hablar("Cursor recentrado.")

    def _main_loop(self):
        try:
            self._run_camera_loop()
        finally:
            if self.cap is not None:
                self.cap.release()

    def _run_camera_loop(self):
        frame_delay = 1.0 / max(int(self.settings.get("fps", 15)), 1)
        announced = False
        read_failures = 0
        if not self._open_camera():
            self._camera_lost("No se pudo abrir la cámara.")

        while self.running:
            if self.camera_state != "ok" and not self._reconnect_camera():
                continue
            loop_start = time.monotonic()
            ok, frame_bgr = self.cap.read()
            if not ok:
                read_failures += 1
                if read_failures >= CAMERA_MAX_READ_FAILURES:
                    read_failures = 0
                    self._camera_lost("Se perdió la señal de la cámara.")
                else:
                    time.sleep(0.05)
                continue
            read_failures = 0

            if not announced:
                announced = True
                self.hablar("Controles asistenciales iniciados.")
                self.logger.log(
                    "runtime",
                    "Runtime iniciado",
                    profile=self.profile["name"],
                    model_version=self.classifier.active_version,
                )

            self._update_fps(loop_start)
            snapshot = self.provider.process(frame_bgr)
            if snapshot.face_sample is not None:
                self._last_face_seen_at = time.monotonic()
            control_state = self.heuristics.update(snapshot.face_sample)
            prediction, inference_result = self._evaluate_gesture(snapshot.face_sample, control_state)
            self._update_zone_hint(snapshot.face_sample, control_state)
            self._record_usage(snapshot.face_sample, control_state, prediction, inference_result)
            self._apply_continuous_control(control_state)
            if self._gui_wants_frame():
                compact = self._gui_is_compact()
                frame_rgb = self._render_overlay(
                    snapshot.frame_rgb, snapshot.face_sample, control_state, prediction, compact=compact
                )
                if compact:
                    self._push_gui_frame(None, frame_rgb)
                else:
                    self._push_gui_frame(frame_rgb, None)

            elapsed = time.monotonic() - loop_start
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

    def _open_camera(self) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(int(self.settings.get("camera_index", 0)))
        self._apply_camera_resolution(self.cap)
        if not self.cap.isOpened():
            return False
        self.camera_state = "ok"
        self._camera_ok_since = time.monotonic()
        return True

    def _camera_lost(self, message: str) -> None:
        """Pasa a modo sin cámara: sin acciones, y reintentos periódicos de reconexión."""
        if self.cap is not None:
            self.cap.release()
        self.camera_state = "reconectando"
        self._camera_ok_since = None
        self._last_frame_time = None
        self.fps_real = 0.0
        if self.state != AppState.ERROR:
            self._state_before_camera_loss = self.state
            self.state = AppState.ERROR
        self.logger.log("error", "Cámara no disponible", reason=message)
        self._publish_event(f"{message} Reintentando cada {CAMERA_RETRY_INTERVAL_S:.0f} s…")
        self._push_status(f"{message} Reintentando…", "-", 0.0, face="sin cámara")

    def _reconnect_camera(self) -> bool:
        deadline = time.monotonic() + CAMERA_RETRY_INTERVAL_S
        while self.running and time.monotonic() < deadline:
            time.sleep(0.1)  # en pasos cortos para que stop() no espere
        if not self.running or not self._open_camera():
            self.camera_state = "reconectando"
            self._push_status("Cámara no disponible. Reintentando…", "-", 0.0, face="sin cámara")
            return False
        if self.state == AppState.ERROR:
            self.state = self._state_before_camera_loss or AppState.READY
        self.logger.log("runtime", "Cámara reconectada")
        self._publish_event("Cámara reconectada.")
        return True

    def _update_fps(self, now: float) -> None:
        last = getattr(self, "_last_frame_time", None)
        self._last_frame_time = now
        if last is None:
            return
        delta = now - last
        if delta <= 0:
            return
        self._frame_dt = delta
        instant = 1.0 / delta
        current = getattr(self, "fps_real", 0.0) or instant
        self.fps_real = current * 0.9 + instant * 0.1

    def _gui_is_compact(self) -> bool:
        return bool(getattr(self.main_gui_interface, "is_compact_mode", False))

    def _gui_wants_frame(self) -> bool:
        """Descarta el fotograma si la interfaz aún no pintó el anterior."""
        if not self._ui_available():
            return False
        return not bool(getattr(self.main_gui_interface, "frame_pending", False))

    def _evaluate_gesture(self, sample: FaceSample | None, control_state):
        if sample is not None:
            self.inference.append_sample(sample)
        else:
            self.prediction_smoother.reset()
        prediction = self.classifier.predict(self.inference.gesture_window, smoother=self.prediction_smoother)
        heuristic_prediction = self._build_heuristic_blink_prediction(sample)
        if self._should_use_heuristic_blink(prediction, heuristic_prediction):
            prediction = heuristic_prediction

        prediction = self._apply_brow_signal(prediction, sample)

        allow_low_confidence = bool(prediction and self._passes_hybrid_gesture_validation(prediction, sample))
        result = self.inference.evaluate(
            sample,
            control_state,
            prediction,
            self.state,
            self._cursor_status_text(),
            allow_low_confidence=allow_low_confidence,
            cursor_active=self.cursor_active,
            signal_verified=self.classifier.verify_signal(prediction.gesture_id, sample) if prediction else None,
        )
        self.last_rejection_reason = result.rejection_reason or "-"
        self.last_diagnostic_hint = result.diagnostic_hint or "-"

        anchor = self._update_click_anchor()
        if result.executed and result.prediction is not None:
            position = anchor[1] if anchor and anchor[0] == result.prediction.gesture_id else None
            self.action_service.execute_gesture_action(result.prediction.gesture_id, self.state.value, position=position)
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
            runtime_state=result.runtime_state,
            gesture_executed=bool(result.executed),
            hold_progress=result.hold_progress,
            throttle=not result.executed,
        )
        return prediction, result

    # ---------------------------------------------------------- grabación de uso
    def start_usage_recording(self) -> str:
        """Empieza a grabar puntos faciales y decisiones (sin video ni audio)."""
        if self.usage_recorder is not None:
            return str(self.usage_recorder.path)
        summary = self.classifier.training_summary or {}
        recorder = UsageRecorder(
            self.profile["name"],
            {
                "model_version": self.classifier.active_version,
                "feature_version": self.classifier.feature_version,
                "metric_source": self.classifier.extractor.metric_source,
                "validation_accuracy": summary.get("validation_accuracy"),
                "thresholds": self.profile.get("gesture_confidence", {}),
                "durations_ms": self.profile.get("gesture_duration_ms", {}),
                "cooldowns_ms": self.profile.get("gesture_cooldown_ms", {}),
                "smoothing_alpha": self.prediction_smoother.alpha,
                "screen": [self.screen_width, self.screen_height],
            },
        )
        path = recorder.start()
        self.usage_recorder = recorder
        self._usage_hotkeys = [
            GlobalHotkey(lambda: self.mark_usage("error"), key="E"),
            GlobalHotkey(lambda: self.mark_usage("segment"), key="N"),
        ]
        for hotkey in self._usage_hotkeys:
            hotkey.start()
        self.logger.log("usage", "Grabación de uso iniciada", path=str(path))
        self._publish_event(
            f"Grabación de uso iniciada. {MARK_ERROR_HOTKEY_LABEL} marca un error; "
            f"{MARK_SEGMENT_HOTKEY_LABEL} marca el cambio de tramo."
        )
        return str(path)

    def stop_usage_recording(self) -> dict | None:
        recorder = getattr(self, "usage_recorder", None)
        if recorder is None:
            return None
        self.usage_recorder = None
        for hotkey in self._usage_hotkeys:
            hotkey.stop()
        self._usage_hotkeys = []
        summary = recorder.stop()
        self.logger.log("usage", "Grabación de uso finalizada", **summary)
        self._publish_event(f"Grabación de uso guardada: {summary['path']}")
        return summary

    def mark_usage(self, kind: str) -> None:
        recorder = self.usage_recorder
        if recorder is None:
            return
        recorder.mark(kind)
        label = "error" if kind == "error" else "cambio de tramo"
        self._publish_event(f"Marcado {label} ({recorder.markers[kind]}) en {recorder.elapsed_s:.0f} s.")

    def _record_usage(self, sample, control_state, prediction, result) -> None:
        recorder = self.usage_recorder
        if recorder is None:
            return
        recorder.record_frame(
            sample,
            state=self.state.value,
            cursor_active=bool(self.cursor_active),
            precision=bool(self.precision_mode),
            steering=bool(control_state.steering),
            face_stable=bool(control_state.face_stable),
            gesture=prediction.gesture_id if prediction else None,
            confidence=round(float(prediction.confidence), 4) if prediction else None,
            runtime_state=result.runtime_state,
            rejection=result.rejection_reason or "",
            executed=bool(result.executed),
            zone_hint=getattr(self, "zone_hint", ""),
        )

    def _update_zone_hint(self, sample, control_state, now: float | None = None) -> None:
        """Pide volver a la posición calibrada; girar la cabeza para guiar el cursor no cuenta."""
        now = time.monotonic() if now is None else now
        issue = self.classifier.zone_issue(sample)
        if issue == "pose" and self.cursor_active and control_state.steering:
            issue = None
        if issue is None:
            self._zone_issue_since = None
            if sample is None:
                self.zone_hint = ""
            elif self.zone_hint:
                if self._zone_clear_since is None:
                    self._zone_clear_since = now
                if now - self._zone_clear_since >= ZONE_HINT_OFF_S:
                    self.zone_hint = ""
            return
        self._zone_clear_since = None
        if self._zone_issue_since is None:
            self._zone_issue_since = now
        if now - self._zone_issue_since >= ZONE_HINT_ON_S or self.zone_hint:
            self.zone_hint = ZONE_HINTS[issue]

    def _apply_brow_signal(self, prediction, sample):
        """Con las cejas arriba según MediaPipe, cuenta como "cejas" salvo otro gesto claro del modelo.

        El modelo por sí solo confunde las cejas con inclinar la cabeza y a veces no las ve.
        """
        if self.classifier.verify_signal("confirm", sample) is not True:
            return prediction
        model_confidence = 0.0
        if prediction is not None:
            if prediction.gesture_id == "confirm":
                model_confidence = prediction.confidence
            elif prediction.gesture_id != "neutral":
                threshold = clamp_confidence_threshold(self.profile.get("gesture_confidence", {}).get(prediction.gesture_id, 0.6))
                if prediction.confidence >= threshold and self.classifier.verify_signal(prediction.gesture_id, sample) is not False:
                    return prediction
        return GesturePrediction("confirm", max(model_confidence, BROW_SIGNAL_CONFIDENCE), False, "Cejas por señal")

    def _update_click_anchor(self):
        """Guarda dónde estaba el cursor al empezar un guiño: el clic se hace ahí."""
        gesture_id = self.inference.gesture_in_progress()
        if gesture_id not in POSITIONAL_GESTURES:
            self._click_anchor = None
        elif self._click_anchor is None or self._click_anchor[0] != gesture_id:
            self._click_anchor = (gesture_id, tuple(pyautogui.position()))
        return self._click_anchor

    def _apply_continuous_control(self, control_state):
        if self.state != AppState.READY or not control_state.face_present or not self.cursor_active:
            return
        if self._click_anchor is not None:
            return  # cursor congelado mientras se valida un guiño
        frame_dt = self._frame_dt or (1.0 / REFERENCE_CURSOR_FPS)
        scale = float(np.clip(frame_dt * REFERENCE_CURSOR_FPS, 0.5, 2.0))
        if self.precision_mode:
            scale *= PRECISION_SPEED_FACTOR
        self._cursor_fx = float(np.clip(self._cursor_fx + control_state.smoothed_dx * scale, 0, self.screen_width - 1))
        self._cursor_fy = float(np.clip(self._cursor_fy + control_state.smoothed_dy * scale, 0, self.screen_height - 1))
        target_x, target_y = int(round(self._cursor_fx)), int(round(self._cursor_fy))
        if (target_x, target_y) == (self.current_cursor_x, self.current_cursor_y):
            return
        self.current_cursor_x, self.current_cursor_y = target_x, target_y
        pyautogui.moveTo(self.current_cursor_x, self.current_cursor_y, duration=0)

    def start_manual_voice_test(self):
        self._start_voice_listener(trigger_source="manual")

    def _start_voice_listener(self, trigger_source: str = "gesture"):
        if self.voice_enabled and self.whisper is None:
            if getattr(self, "voice_model_state", "") == "cargando":
                message = "El reconocimiento de voz aún se está cargando. Inténtalo en unos segundos."
            else:
                message = f"Reconocimiento de voz no disponible: {getattr(self, 'voice_model_error', '')}"
            self._publish_event(message)
            self._push_status(message, "-", 0.0, face="estable", mode=self.state.value, cursor=self._cursor_status_text())
            return
        if not self.voice_enabled:
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
        # Espera a terminar de hablar para que el micrófono no grabe el propio aviso.
        self.hablar("Escuchando.", wait=True)
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
            if audio_diag["peak"] < 1e-4:
                raise RuntimeError(
                    "El micrófono no entrega señal (silencio absoluto). Revisa que esté conectado, que no esté "
                    "silenciado y que Windows lo permita en Privacidad > Micrófono."
                )
            segments, _ = self.whisper.transcribe(
                AUDIO_RECORDING_PATH,
                language="es",
                vad_filter=True,
            )
            text = " ".join(segment.text.strip() for segment in segments).strip()
            if not self.running:
                return  # la sesión terminó mientras se transcribía: no ejecutar nada
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
            message = str(exc) if isinstance(exc, RuntimeError) else friendly_audio_error(exc)
            self.logger.log("error", "Fallo en voz", error=str(exc))
            self.last_voice_state = "error"
            self.last_voice_text = message
            self.last_voice_command = "-"
            self._publish_event(f"Error en voz: {message}")
            self._push_status(
                f"Error en voz: {message}",
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
        """El comando de voz "cerrar sistema" termina la sesión; quien la cierra es la interfaz."""
        gui = self.main_gui_interface
        if gui is not None and hasattr(gui, "run_on_ui_thread"):
            gui.run_on_ui_thread(gui.request_end_session)

    def _render_overlay(self, frame_rgb, sample, control_state, prediction, compact: bool = False):
        """Dibuja la guía visual sobre una copia reducida del fotograma (RGB)."""
        source_h, source_w = frame_rgb.shape[:2]
        target_w = COMPACT_FRAME_WIDTH if compact else GUI_FRAME_WIDTH
        scale = min(1.0, target_w / float(max(source_w, 1)))
        if scale < 1.0:
            frame = cv2.resize(frame_rgb, (int(source_w * scale), int(source_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            frame = frame_rgb.copy()
        frame_height, frame_width = frame.shape[:2]

        def scaled(point):
            return int(point[0] * scale), int(point[1] * scale)

        # Colores en RGB: el fotograma ya viene en RGB, no hace falta convertir a BGR.
        amber = (255, 180, 0)
        magenta = (255, 0, 255)
        green = (80, 255, 60)
        orange = (255, 128, 0)
        white = (255, 255, 255)
        info = (255, 220, 140)

        frame_center = (frame_width // 2, frame_height // 2)
        neutral = getattr(self.heuristics, "neutral_nose", None)
        if neutral and tuple(neutral) != (0.0, 0.0):
            neutral_point = scaled(neutral)
        else:
            neutral_point = frame_center

        if not compact:
            cv2.line(frame, (frame_center[0] - 14, frame_center[1]), (frame_center[0] + 14, frame_center[1]), amber, 1)
            cv2.line(frame, (frame_center[0], frame_center[1] - 14), (frame_center[0], frame_center[1] + 14), amber, 1)
        cv2.circle(frame, neutral_point, 7 if not compact else 4, magenta, 2)

        if sample is not None and not compact:
            for point in sample.points_px.values():
                cv2.circle(frame, scaled(point), 1, green, -1)
            cv2.circle(frame, scaled(sample.nose_px), 4, orange, -1)
        if self.mirror_preview:
            frame = cv2.flip(frame, 1)

        if compact or not self.show_camera_overlay_details:
            return frame

        detail_y = 28
        cv2.putText(frame, f"Estado: {self.state.value}", (14, detail_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, white, 2)
        cv2.putText(
            frame,
            f"{control_state.status_text} | cursor: {self._cursor_status_text()}",
            (14, detail_y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            white,
            1,
        )
        metric_y = detail_y + 48
        if prediction is not None:
            cv2.putText(
                frame,
                f"Gesto: {prediction.gesture_id} ({prediction.confidence:.2f})",
                (14, metric_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                info,
                1,
            )
            metric_y += 22
            cv2.putText(frame, f"Motivo: {prediction.reason}", (14, metric_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, info, 1)
            metric_y += 26
        for key, value in control_state.debug_metrics.items():
            cv2.putText(frame, f"{key}: {value:.3f}", (14, metric_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            metric_y += 20
        return frame

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
            # Las métricas siguen la imagen sin espejar: "left_eye_ratio" es el ojo derecho de la persona.
            return GesturePrediction(
                gesture_id="right_blink_intent",
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
                gesture_id="left_blink_intent",
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
        threshold = clamp_confidence_threshold(self.profile.get("gesture_confidence", {}).get(prediction.gesture_id, 0.5))
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
                self.main_gui_interface.frame_pending = True
                self.main_gui_interface.signals.video_update.emit(frame_rgb, compact_frame_rgb)
            elif hasattr(self.main_gui_interface, "root"):
                self.main_gui_interface.root.after(
                    0,
                    lambda: self.main_gui_interface.update_video_feed(frame_rgb, compact_frame_rgb),
                )
        except Exception:
            pass

    def _build_status_payload(
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
        runtime_state=None,
        gesture_executed=False,
        hold_progress=0.0,
    ) -> dict:
        mic_payload = self._get_mic_status_payload()
        gesture_title, gesture_action = GESTURE_LABELS.get(gesture, (gesture, "")) if gesture else ("-", "")
        return {
            "status_text": status_text,
            "gesture": gesture,
            "gesture_title": gesture_title,
            "gesture_action": gesture_action,
            "gesture_executed": bool(gesture_executed),
            "confidence": f"{confidence:.2f}" if isinstance(confidence, (int, float)) else confidence,
            "confidence_value": float(confidence) if isinstance(confidence, (int, float)) else 0.0,
            "face": face,
            "mode": mode or self.state.value,
            "cursor": cursor or self._cursor_status_text(),
            "cursor_active": bool(self.cursor_active),
            "voice_state": voice_state or self.last_voice_state,
            "voice_text": voice_text or self.last_voice_text,
            "voice_command": voice_command or self.last_voice_command,
            "state": self.state.value,
            "runtime_state": runtime_state or "",
            "rejection_reason": rejection_reason if rejection_reason is not None else self.last_rejection_reason,
            "diagnostic_hint": diagnostic_hint if diagnostic_hint is not None else self.last_diagnostic_hint,
            "model_version": self.classifier.active_version if self.classifier.active_version is not None else "-",
            "fps_real": round(float(getattr(self, "fps_real", 0.0) or 0.0), 1),
            "hold_progress": round(float(hold_progress or 0.0), 3),
            "precision_mode": bool(getattr(self, "precision_mode", False)),
            "usage_recording_s": recorder.elapsed_s if (recorder := getattr(self, "usage_recorder", None)) else None,
            "zone_hint": getattr(self, "zone_hint", ""),
            "voice_model_state": getattr(self, "voice_model_state", "lista"),
            **mic_payload,
        }

    def _push_status(self, status_text, gesture, confidence, *, throttle: bool = False, **kwargs):
        """Envía el estado a la interfaz.

        Con ``throttle=True`` (actualizaciones por fotograma) se limita a unas 10
        por segundo; los eventos importantes (gesto ejecutado, voz) pasan siempre.
        """
        now = time.monotonic()
        if throttle and now - getattr(self, "_last_status_push", 0.0) < STATUS_PUSH_INTERVAL_S:
            return
        if not self._ui_available():
            return
        self._last_status_push = now
        payload = self._build_status_payload(status_text, gesture, confidence, **kwargs)
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

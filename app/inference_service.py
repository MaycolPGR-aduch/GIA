from __future__ import annotations

import time
from collections import deque

from .gesture_catalog import is_gesture_enabled
from .models import AppState, FaceSample, GestureInferenceResult, GesturePrediction


class InferenceService:
    def __init__(self, profile: dict, window_size: int):
        self.window_size = window_size
        self.gesture_window: deque[FaceSample] = deque(maxlen=window_size)
        self.gesture_thresholds = profile.get("gesture_confidence", {})
        self.gesture_durations = profile.get("gesture_duration_ms", {})
        self.gesture_cooldowns = profile.get("gesture_cooldown_ms", {})
        self.current_gesture = None
        self.current_gesture_started_at = 0.0
        self.latched_gesture = None
        self.last_action_at: dict[str, float] = {}
        self.last_reason_signature = None

    def append_sample(self, sample: FaceSample) -> None:
        self.gesture_window.append(sample)

    def evaluate(
        self,
        sample: FaceSample | None,
        control_state,
        prediction: GesturePrediction | None,
        app_state: AppState,
        cursor_label: str,
        allow_low_confidence: bool = False,
    ) -> GestureInferenceResult:
        if sample is None:
            self.current_gesture = None
            self.current_gesture_started_at = 0.0
            self.latched_gesture = None
            return GestureInferenceResult(
                prediction=None,
                status_text="Sin rostro detectado.",
                face_label="sin rostro",
                mode_label=app_state.value,
                cursor_label=cursor_label,
                runtime_state="sin rostro",
            )

        if prediction is None:
            self.latched_gesture = None
            return GestureInferenceResult(
                prediction=None,
                status_text=control_state.status_text,
                face_label="estable" if control_state.face_stable else "inestable",
                mode_label=app_state.value,
                cursor_label=cursor_label,
                runtime_state="listo" if control_state.face_stable else "rostro inestable",
            )

        if prediction.gesture_id == "neutral":
            self.current_gesture = None
            self.current_gesture_started_at = 0.0
            self.latched_gesture = None
            prediction.reason = "Estado neutral"
            return GestureInferenceResult(
                prediction=prediction,
                status_text="Listo para gesto.",
                face_label="estable" if control_state.face_stable else "inestable",
                mode_label=app_state.value,
                cursor_label=cursor_label,
                runtime_state="listo",
                diagnostic_hint="El modelo detecta un estado neutral estable.",
            )

        if not is_gesture_enabled(prediction.gesture_id):
            return self._reject(
                prediction,
                "Gesto deshabilitado",
                "El gesto detectado está deshabilitado en esta versión.",
                app_state,
                cursor_label,
                should_log=False,
            )

        if prediction.gesture_id == self.latched_gesture:
            return self._reject(
                prediction,
                "Esperando liberación",
                "Gesto sostenido. Vuelve a neutral antes de repetirlo.",
                app_state,
                cursor_label,
                should_log=False,
            )

        threshold = float(self.gesture_thresholds.get(prediction.gesture_id, 0.6))
        now = time.monotonic()

        if not control_state.face_stable:
            return self._reject(
                prediction,
                "Rostro inestable",
                "Rostro detectado pero aún inestable.",
                app_state,
                cursor_label,
                should_log=True,
            )

        if prediction.confidence < threshold and not allow_low_confidence:
            return self._reject(
                prediction,
                "Confianza baja",
                f"Gesto rechazado por baja confianza (< {threshold:.2f}).",
                app_state,
                cursor_label,
                should_log=True,
            )
        if prediction.confidence < threshold and allow_low_confidence:
            prediction.reason = "Aceptado por validación híbrida"

        if prediction.gesture_id != self.current_gesture:
            self.current_gesture = prediction.gesture_id
            self.current_gesture_started_at = now
            return GestureInferenceResult(
                prediction=prediction,
                status_text="Gesto detectado. Validando duración...",
                face_label="estable",
                mode_label=app_state.value,
                cursor_label=cursor_label,
                runtime_state="validando gesto",
                rejection_reason="duracion insuficiente",
                diagnostic_hint="El gesto superó confianza pero aún no completó la duración mínima.",
            )

        required_duration = self.gesture_durations.get(prediction.gesture_id, 700) / 1000.0
        if now - self.current_gesture_started_at < required_duration:
            return GestureInferenceResult(
                prediction=prediction,
                status_text="Gesto detectado. Esperando duración mínima...",
                face_label="estable",
                mode_label=app_state.value,
                cursor_label=cursor_label,
                runtime_state="validando gesto",
                rejection_reason="duracion insuficiente",
                diagnostic_hint="Mantén el gesto un poco más para evitar falsos positivos.",
            )

        cooldown = self.gesture_cooldowns.get(prediction.gesture_id, 1500) / 1000.0
        last_action = self.last_action_at.get(prediction.gesture_id, 0.0)
        if now - last_action < cooldown:
            return self._reject(
                prediction,
                "En cooldown",
                "Gesto reconocido pero en cooldown.",
                app_state,
                cursor_label,
                should_log=True,
            )

        if app_state == AppState.PAUSED and prediction.gesture_id not in {"both_eyes_closed_intent", "mouth_open_hold"}:
            return self._reject(
                prediction,
                "Sistema en pausa",
                "Sistema en pausa. Solo gestos seguros están activos.",
                app_state,
                cursor_label,
                should_log=True,
            )

        prediction.accepted = True
        prediction.reason = "Gesto aceptado"
        self.latched_gesture = prediction.gesture_id
        self.last_action_at[prediction.gesture_id] = now
        self.last_reason_signature = None
        return GestureInferenceResult(
            prediction=prediction,
            status_text=f"Gesto ejecutado: {prediction.gesture_id}",
            face_label="estable",
            mode_label=app_state.value,
            cursor_label=cursor_label,
            runtime_state="ejecutado",
            executed=True,
            should_log=True,
            diagnostic_hint="El gesto superó confianza, duración y cooldown.",
        )

    def _reject(
        self,
        prediction: GesturePrediction,
        reason: str,
        status_text: str,
        app_state: AppState,
        cursor_label: str,
        *,
        should_log: bool,
    ) -> GestureInferenceResult:
        prediction.reason = reason
        signature = (prediction.gesture_id, reason)
        log_now = should_log and signature != self.last_reason_signature
        self.last_reason_signature = signature
        return GestureInferenceResult(
            prediction=prediction,
            status_text=status_text,
            face_label="estable" if reason != "Rostro inestable" else "inestable",
            mode_label=app_state.value,
            cursor_label=cursor_label,
            runtime_state="rechazado" if reason != "Rostro inestable" else "rostro inestable",
            rejection_reason=reason.lower(),
            should_log=log_now,
            diagnostic_hint=f"Motivo de rechazo: {reason}.",
        )

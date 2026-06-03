from __future__ import annotations

from copy import deepcopy

from .gesture_catalog import GESTURE_CATALOG, NEUTRAL_GESTURE_META
from .models import FaceSample, GestureTrainingStatus


MIN_FACE_SCALE_PX = 90.0


class CalibrationService:
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.gesture_order = ["neutral"] + [gesture["id"] for gesture in GESTURE_CATALOG]
        self.gesture_meta = {"neutral": deepcopy(NEUTRAL_GESTURE_META)}
        self.gesture_meta.update({gesture["id"]: deepcopy(gesture) for gesture in GESTURE_CATALOG})
        self.samples_by_gesture: dict[str, list[FaceSample]] = {gesture_id: [] for gesture_id in self.gesture_order}
        self.capture_quality: dict[str, dict] = {gesture_id: self._blank_quality(gesture_id) for gesture_id in self.gesture_order}
        self.pending_retraining = False

    def _blank_quality(self, gesture_id: str) -> dict:
        meta = self.gesture_meta[gesture_id]
        return {
            "captured_frames": 0,
            "valid_frames": 0,
            "invalid_frames": 0,
            "training_windows": 0,
            "trained": False,
            "readiness": "sin iniciar",
            "help_message": meta["training_hint"],
            "success_message": meta["success_hint"],
            "blocker": "Aún no se inició la captura.",
            "last_feedback": meta["training_hint"],
        }

    def load_samples(self, samples_by_gesture: dict[str, list[FaceSample]], stored_capture_quality: dict | None = None) -> None:
        self.samples_by_gesture = {gesture_id: list(samples_by_gesture.get(gesture_id, [])) for gesture_id in self.gesture_order}
        self.capture_quality = {gesture_id: self._blank_quality(gesture_id) for gesture_id in self.gesture_order}
        for gesture_id, samples in self.samples_by_gesture.items():
            quality = self.capture_quality[gesture_id]
            quality["captured_frames"] = len(samples)
            quality["valid_frames"] = len(samples)
            quality["invalid_frames"] = 0
            quality["training_windows"] = self._window_count(len(samples))
        if stored_capture_quality:
            for gesture_id, payload in stored_capture_quality.items():
                if gesture_id in self.capture_quality:
                    self.capture_quality[gesture_id].update(payload)
        self.pending_retraining = False
        self.refresh_statuses()

    def reset_gesture(self, gesture_id: str) -> None:
        self.samples_by_gesture[gesture_id] = []
        self.capture_quality[gesture_id] = self._blank_quality(gesture_id)
        self.pending_retraining = True

    def register_sample(self, gesture_id: str, sample: FaceSample | None) -> tuple[bool, str]:
        quality = self.capture_quality[gesture_id]
        quality["captured_frames"] += 1
        quality["trained"] = False
        meta = self.gesture_meta[gesture_id]

        if sample is None:
            quality["invalid_frames"] += 1
            quality["last_feedback"] = "No se detectó rostro. Mantén la cara frente a la cámara."
            self.refresh_statuses()
            return False, quality["last_feedback"]

        if float(sample.face_scale_px) < MIN_FACE_SCALE_PX:
            quality["invalid_frames"] += 1
            quality["last_feedback"] = "Acércate un poco a la cámara para capturar mejor el gesto."
            self.refresh_statuses()
            return False, quality["last_feedback"]

        self.samples_by_gesture[gesture_id].append(sample)
        quality["valid_frames"] = len(self.samples_by_gesture[gesture_id])
        quality["training_windows"] = self._window_count(quality["valid_frames"])
        self.pending_retraining = True

        if quality["valid_frames"] >= meta["recommended_min_frames"]:
            quality["last_feedback"] = meta["success_hint"]
        else:
            quality["last_feedback"] = meta["training_hint"]
        self.refresh_statuses()
        return True, quality["last_feedback"]

    def _window_count(self, valid_frames: int) -> int:
        return max(0, valid_frames - self.window_size + 1)

    def refresh_statuses(self) -> dict[str, GestureTrainingStatus]:
        statuses = {}
        for gesture_id in self.gesture_order:
            meta = self.gesture_meta[gesture_id]
            quality = self.capture_quality[gesture_id]
            valid_frames = len(self.samples_by_gesture[gesture_id])
            quality["valid_frames"] = valid_frames
            quality["training_windows"] = self._window_count(valid_frames)

            readiness = "sin iniciar"
            blocker = quality.get("blocker", "")
            if valid_frames == 0:
                blocker = "Captura pendiente."
            elif valid_frames < meta["recommended_min_frames"] or quality["training_windows"] < meta["recommended_min_windows"]:
                readiness = "insuficiente"
                blocker = (
                    f"Faltan frames o ventanas útiles. "
                    f"Objetivo: {meta['recommended_min_frames']} frames y {meta['recommended_min_windows']} ventanas."
                )
            elif quality.get("trained", False):
                readiness = "entrenado"
                blocker = ""
            else:
                readiness = "listo para entrenar"
                blocker = ""

            quality["readiness"] = readiness
            quality["help_message"] = quality.get("last_feedback") or meta["training_hint"]
            quality["success_message"] = meta["success_hint"]
            quality["blocker"] = blocker

            statuses[gesture_id] = GestureTrainingStatus(
                gesture_id=gesture_id,
                title=meta["title"],
                action=meta["action"],
                captured_frames=quality["captured_frames"],
                valid_frames=valid_frames,
                invalid_frames=quality["invalid_frames"],
                training_windows=quality["training_windows"],
                recommended_min_frames=meta["recommended_min_frames"],
                recommended_min_windows=meta["recommended_min_windows"],
                readiness=readiness,
                help_message=quality["help_message"],
                success_message=meta["success_hint"],
                blocker=blocker,
            )
        return statuses

    def can_train(self) -> tuple[bool, list[str], dict[str, GestureTrainingStatus]]:
        statuses = self.refresh_statuses()
        blockers = [
            status.title
            for status in statuses.values()
            if status.readiness not in {"listo para entrenar", "entrenado"}
        ]
        ready = len(blockers) == 0 and len([samples for samples in self.samples_by_gesture.values() if samples]) >= 2
        return ready, blockers, statuses

    def build_summary(self) -> dict:
        ready, blockers, statuses = self.can_train()
        ready_count = sum(
            1 for status in statuses.values() if status.readiness in {"listo para entrenar", "entrenado"}
        )
        return {
            "ready_to_train": ready,
            "ready_count": ready_count,
            "total_gestures": len(statuses),
            "blockers": blockers,
            "pending_retraining": self.pending_retraining,
        }

    def export_capture_quality(self) -> dict:
        self.refresh_statuses()
        return deepcopy(self.capture_quality)

    def export_samples(self) -> dict[str, list[FaceSample]]:
        return {gesture_id: list(samples) for gesture_id, samples in self.samples_by_gesture.items() if samples}

    def mark_trained(self) -> None:
        self.pending_retraining = False
        statuses = self.refresh_statuses()
        for gesture_id, status in statuses.items():
            if status.readiness == "listo para entrenar":
                self.capture_quality[gesture_id]["trained"] = True
                self.capture_quality[gesture_id]["readiness"] = "entrenado"
                self.capture_quality[gesture_id]["help_message"] = status.success_message

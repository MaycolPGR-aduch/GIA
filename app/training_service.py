from __future__ import annotations

import time

from .gesture_ml import GestureClassifier, LegacyModelError


class TrainingService:
    def __init__(self, profile_name: str, window_size: int):
        self.profile_name = profile_name
        self.window_size = window_size
        self.classifier = GestureClassifier(profile_name, window_size=window_size)
        self.load_error: str | None = None

    def load_classifier(self) -> bool:
        self.load_error = None
        try:
            return self.classifier.load()
        except LegacyModelError as exc:
            self.load_error = str(exc)
            return False

    def load_active_dataset(self) -> dict:
        return self.classifier.load_active_dataset()

    def list_versions(self) -> list[dict]:
        return self.classifier.list_versions()

    def save_pending_samples(self, samples_by_gesture: dict, capture_quality: dict) -> None:
        from .config_store import profile_calibration_dir
        from .gesture_ml import serialize_samples_by_gesture
        import json

        path = profile_calibration_dir(self.profile_name) / "pending_samples.json"
        payload = {
            "samples_by_gesture": serialize_samples_by_gesture(samples_by_gesture),
            "capture_quality": capture_quality
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def load_pending_samples(self) -> tuple[dict, dict] | None:
        from .config_store import profile_calibration_dir
        from .gesture_ml import deserialize_samples_by_gesture
        import json

        path = profile_calibration_dir(self.profile_name) / "pending_samples.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            samples = deserialize_samples_by_gesture(payload.get("samples_by_gesture", {}))
            quality = payload.get("capture_quality", {})
            return samples, quality
        except Exception:
            return None

    def clear_pending_samples(self) -> None:
        from .config_store import profile_calibration_dir
        path = profile_calibration_dir(self.profile_name) / "pending_samples.json"
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass

    def train(self, samples_by_gesture: dict, capture_quality_summary: dict | None = None) -> GestureClassifier:
        self.classifier = GestureClassifier(self.profile_name, window_size=self.window_size)
        self.classifier.fit(samples_by_gesture, capture_quality_summary=capture_quality_summary)
        return self.classifier

    def build_profile_training_state(self, profile: dict, samples_by_gesture: dict, capture_quality_summary: dict, statuses: dict) -> dict:
        profile["launcher_completed"] = True
        profile["calibration"]["completed"] = True
        profile["calibration"]["last_calibrated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        profile["calibration"]["samples_per_gesture"] = {
            name: len(values) for name, values in samples_by_gesture.items()
        }
        profile["calibration"]["gesture_readiness"] = {
            gesture_id: status.readiness for gesture_id, status in statuses.items()
        }
        profile["calibration"]["capture_quality"] = capture_quality_summary
        profile["calibration"]["pending_retraining"] = False
        profile["calibration"]["last_training_metrics"] = self.classifier.training_summary
        profile["calibration"]["active_model_version"] = self.classifier.active_version
        profile["calibration"]["active_model_path"] = str(self.classifier.model_path)
        profile["calibration"]["active_dataset_path"] = str(self.classifier.active_dataset_path)
        profile["calibration"]["model_versions"] = [
            entry["version"] for entry in self.classifier.list_versions()
        ]
        profile["gesture_confidence"].update(self.classifier.training_summary.get("recommended_thresholds", {}))
        return profile

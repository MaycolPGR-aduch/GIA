from __future__ import annotations

import json
from collections import deque
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config_store import (
    active_dataset_summary_path,
    legacy_dataset_summary_path,
    legacy_model_path,
    model_registry_path,
    profile_calibration_dir,
    versioned_dataset_path,
    versioned_model_path,
)
from .models import FaceSample, GesturePrediction


class GestureFeatureExtractor:
    def __init__(self, window_size: int = 12):
        self.window_size = window_size

    def extract(self, samples: list[FaceSample]) -> np.ndarray:
        window = samples[-self.window_size :]
        landmark_stack = np.stack([sample.normalized_landmarks.flatten() for sample in window])
        metric_order = sorted(window[0].metrics.keys())
        metric_stack = np.stack([[sample.metrics[name] for name in metric_order] for sample in window])

        feature_vector = np.concatenate(
            [
                landmark_stack.flatten(),
                metric_stack.flatten(),
                landmark_stack.mean(axis=0),
                landmark_stack.std(axis=0),
                metric_stack.mean(axis=0),
                metric_stack.std(axis=0),
                (landmark_stack[-1] - landmark_stack[0]),
                (metric_stack[-1] - metric_stack[0]),
            ]
        )
        return feature_vector.astype(np.float32)


def serialize_face_sample(sample: FaceSample) -> dict:
    return {
        "timestamp_ms": int(sample.timestamp_ms),
        "normalized_landmarks": sample.normalized_landmarks.tolist(),
        "metrics": {key: float(value) for key, value in sample.metrics.items()},
        "points_px": {str(key): [int(value[0]), int(value[1])] for key, value in sample.points_px.items()},
        "nose_px": [int(sample.nose_px[0]), int(sample.nose_px[1])],
        "face_scale_px": float(sample.face_scale_px),
        "face_center_px": [int(sample.face_center_px[0]), int(sample.face_center_px[1])],
    }


def deserialize_face_sample(payload: dict) -> FaceSample:
    return FaceSample(
        timestamp_ms=int(payload["timestamp_ms"]),
        normalized_landmarks=np.array(payload["normalized_landmarks"], dtype=np.float32),
        metrics={key: float(value) for key, value in payload["metrics"].items()},
        points_px={int(key): (int(value[0]), int(value[1])) for key, value in payload["points_px"].items()},
        nose_px=(int(payload["nose_px"][0]), int(payload["nose_px"][1])),
        face_scale_px=float(payload["face_scale_px"]),
        face_center_px=(int(payload["face_center_px"][0]), int(payload["face_center_px"][1])),
    )


def serialize_samples_by_gesture(samples_by_gesture: dict[str, list[FaceSample]]) -> dict:
    return {
        gesture_id: [serialize_face_sample(sample) for sample in samples]
        for gesture_id, samples in samples_by_gesture.items()
    }


def deserialize_samples_by_gesture(payload: dict) -> dict[str, list[FaceSample]]:
    return {
        gesture_id: [deserialize_face_sample(sample_payload) for sample_payload in samples]
        for gesture_id, samples in payload.items()
    }


class GestureClassifier:
    def __init__(self, profile_name: str, window_size: int = 12):
        self.profile_name = profile_name
        self.window_size = window_size
        self.extractor = GestureFeatureExtractor(window_size=window_size)
        self.profile_dir = profile_calibration_dir(profile_name)
        self.registry_path = model_registry_path(profile_name)
        self.summary_path = active_dataset_summary_path(profile_name)
        self.model_path = legacy_model_path(profile_name)
        self.samples_path = legacy_dataset_summary_path(profile_name)
        self.model: RandomForestClassifier | None = None
        self.labels: list[str] = []
        self.training_summary: dict = {}
        self.active_version: int | None = None
        self.active_dataset_path = None

    def _load_registry(self) -> dict:
        if not self.registry_path.exists():
            return {"profile": self.profile_name, "active_version": None, "versions": []}
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save_registry(self, payload: dict) -> None:
        self.registry_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, version: int | None = None) -> bool:
        registry = self._load_registry()
        target_version = version if version is not None else registry.get("active_version")
        versions = {entry["version"]: entry for entry in registry.get("versions", [])}

        if target_version in versions:
            version_entry = versions[target_version]
            self.model_path = versioned_model_path(self.profile_name, target_version)
            self.samples_path = self.summary_path
            dataset_name = version_entry.get("dataset_path")
            self.active_dataset_path = str(self.profile_dir / dataset_name) if dataset_name else None
            payload = joblib.load(self.model_path)
            self.model = payload["model"]
            self.labels = payload["labels"]
            self.training_summary = payload.get("training_summary", {})
            self.active_version = target_version
            return True

        legacy_path = legacy_model_path(self.profile_name)
        if legacy_path.exists():
            payload = joblib.load(legacy_path)
            self.model_path = legacy_path
            self.samples_path = legacy_dataset_summary_path(self.profile_name)
            self.model = payload["model"]
            self.labels = payload["labels"]
            self.training_summary = payload.get("training_summary", {})
            self.active_version = None
            self.active_dataset_path = None
            return True

        return False

    def save(self) -> None:
        if self.model is None:
            return
        joblib.dump(
            {
                "model": self.model,
                "labels": self.labels,
                "training_summary": self.training_summary,
            },
            self.model_path,
        )

    def save_samples(self, serialized_samples: dict) -> None:
        self.samples_path.write_text(json.dumps(serialized_samples, indent=2, ensure_ascii=False), encoding="utf-8")

    def list_versions(self) -> list[dict]:
        return self._load_registry().get("versions", [])

    def load_active_dataset(self) -> dict[str, list[FaceSample]]:
        registry = self._load_registry()
        active_version = registry.get("active_version")
        if active_version is not None:
            versions = {entry["version"]: entry for entry in registry.get("versions", [])}
            version_entry = versions.get(active_version)
            if version_entry:
                dataset_path = version_entry.get("dataset_path")
                if dataset_path:
                    dataset_file = self.profile_dir / dataset_path
                    if dataset_file.exists():
                        payload = json.loads(dataset_file.read_text(encoding="utf-8"))
                        return deserialize_samples_by_gesture(payload.get("samples_by_gesture", {}))
        return {}

    def fit(self, samples_by_gesture: dict[str, list[FaceSample]]) -> None:
        features = []
        labels = []
        class_windows: dict[str, int] = {}
        for gesture_id, samples in samples_by_gesture.items():
            if len(samples) < self.window_size:
                continue
            class_windows[gesture_id] = 0
            for index in range(self.window_size, len(samples) + 1):
                window = samples[index - self.window_size : index]
                features.append(self.extractor.extract(window))
                labels.append(gesture_id)
                class_windows[gesture_id] += 1

        if len(set(labels)) < 2:
            raise ValueError("Se necesitan al menos dos clases con muestras suficientes para entrenar.")

        registry = self._load_registry()
        existing_versions = [entry["version"] for entry in registry.get("versions", [])]
        next_version = (max(existing_versions) + 1) if existing_versions else 1

        feature_matrix = np.vstack(features)
        self.model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        )
        self.model.fit(feature_matrix, np.array(labels))
        self.labels = sorted(set(labels))
        self.active_version = next_version
        self.model_path = versioned_model_path(self.profile_name, next_version)
        dataset_path = versioned_dataset_path(self.profile_name, next_version)
        self.active_dataset_path = str(dataset_path)
        self.samples_path = self.summary_path

        self.training_summary = {
            "profile": self.profile_name,
            "version": next_version,
            "trained_at": datetime.utcnow().isoformat(),
            "window_size": self.window_size,
            "feature_dim": int(feature_matrix.shape[1]),
            "training_windows": int(feature_matrix.shape[0]),
            "classes": self.labels,
            "windows_per_class": class_windows,
            "model_type": "RandomForestClassifier",
            "n_estimators": 300,
            "model_path": str(self.model_path),
            "dataset_path": str(dataset_path),
        }
        self.save()

        dataset_payload = {
            "profile": self.profile_name,
            "version": next_version,
            "saved_at": self.training_summary["trained_at"],
            "window_size": self.window_size,
            "samples_by_gesture": serialize_samples_by_gesture(samples_by_gesture),
        }
        dataset_path.write_text(json.dumps(dataset_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        registry_entry = {
            "version": next_version,
            "trained_at": self.training_summary["trained_at"],
            "model_path": self.model_path.name,
            "dataset_path": dataset_path.name,
            "classes": self.labels,
            "training_windows": int(feature_matrix.shape[0]),
        }
        registry.setdefault("versions", []).append(registry_entry)
        registry["profile"] = self.profile_name
        registry["active_version"] = next_version
        self._save_registry(registry)

        self.save_samples(
            {
                "profile": self.profile_name,
                "active_version": next_version,
                "registry_path": str(self.registry_path),
                "versions": registry.get("versions", []),
                "training_summary": self.training_summary,
                "sample_counts": {name: len(values) for name, values in samples_by_gesture.items()},
            }
        )

    def predict(self, window: deque[FaceSample]) -> GesturePrediction | None:
        if self.model is None or len(window) < self.window_size:
            return None

        vector = self.extractor.extract(list(window))
        probabilities = self.model.predict_proba(vector.reshape(1, -1))[0]
        best_index = int(np.argmax(probabilities))
        gesture_id = self.model.classes_[best_index]
        confidence = float(probabilities[best_index])
        return GesturePrediction(
            gesture_id=gesture_id,
            confidence=confidence,
            accepted=False,
            reason="Predicción generada",
        )

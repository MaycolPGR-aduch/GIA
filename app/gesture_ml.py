from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .config_store import calibration_samples_path, profile_model_path
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


class GestureClassifier:
    def __init__(self, profile_name: str, window_size: int = 12):
        self.profile_name = profile_name
        self.window_size = window_size
        self.extractor = GestureFeatureExtractor(window_size=window_size)
        self.model_path = profile_model_path(profile_name)
        self.samples_path = calibration_samples_path(profile_name)
        self.model: RandomForestClassifier | None = None
        self.labels: list[str] = []
        self.training_summary: dict = {}

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.labels = payload["labels"]
        self.training_summary = payload.get("training_summary", {})
        return True

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

        feature_matrix = np.vstack(features)
        self.model = RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced_subsample",
        )
        self.model.fit(feature_matrix, np.array(labels))
        self.labels = sorted(set(labels))
        self.training_summary = {
            "profile": self.profile_name,
            "window_size": self.window_size,
            "feature_dim": int(feature_matrix.shape[1]),
            "training_windows": int(feature_matrix.shape[0]),
            "classes": self.labels,
            "windows_per_class": class_windows,
            "model_type": "RandomForestClassifier",
            "n_estimators": 300,
        }
        self.save()

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

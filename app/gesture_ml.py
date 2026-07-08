from __future__ import annotations

import gzip
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .config_store import (
    active_dataset_summary_path,
    model_registry_path,
    profile_calibration_dir,
    versioned_dataset_path,
    versioned_model_path,
)
from .models import FaceSample, GesturePrediction


MODEL_FORMAT_VERSION = 2

# Orden explícito de métricas de entrada. Los modelos guardados dependen de este
# orden; si se agrega una métrica nueva hay que crear un extractor nuevo (no
# modificar este listado) para no invalidar silenciosamente modelos existentes.
FEATURE_METRIC_ORDER = [
    "brow_raise_ratio",
    "left_eye_ratio",
    "mouth_open_ratio",
    "right_eye_ratio",
    "smile_ratio",
]

STAT_NAMES = ["mean", "std", "min", "max", "delta", "mean_abs_velocity"]

MODEL_VERSIONS_TO_KEEP = 3


class LegacyModelError(RuntimeError):
    """Modelo entrenado con un formato anterior; requiere reentrenamiento."""


def _metric_matrix(samples: list[FaceSample]) -> np.ndarray:
    return np.stack(
        [[sample.metrics[name] for name in FEATURE_METRIC_ORDER] for sample in samples]
    ).astype(np.float32)


class DeterministicGRUEncoder:
    def __init__(self, input_dim: int = 5, hidden_dim: int = 32, seed: int = 42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        rng = np.random.default_rng(seed)

        # Update gate weights
        self.W_z = rng.normal(0.0, 0.1, (input_dim, hidden_dim)).astype(np.float32)
        self.U_z = rng.normal(0.0, 0.1, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_z = np.zeros(hidden_dim, dtype=np.float32)

        # Reset gate weights
        self.W_r = rng.normal(0.0, 0.1, (input_dim, hidden_dim)).astype(np.float32)
        self.U_r = rng.normal(0.0, 0.1, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_r = np.zeros(hidden_dim, dtype=np.float32)

        # Candidate hidden state weights
        self.W_h = rng.normal(0.0, 0.1, (input_dim, hidden_dim)).astype(np.float32)
        self.U_h = rng.normal(0.0, 0.1, (hidden_dim, hidden_dim)).astype(np.float32)
        self.b_h = np.zeros(hidden_dim, dtype=np.float32)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

    def encode(self, sequence: np.ndarray) -> np.ndarray:
        h = np.zeros(self.hidden_dim, dtype=np.float32)
        for t in range(sequence.shape[0]):
            x_t = sequence[t]
            z_t = self._sigmoid(np.dot(x_t, self.W_z) + np.dot(h, self.U_z) + self.b_z)
            r_t = self._sigmoid(np.dot(x_t, self.W_r) + np.dot(h, self.U_r) + self.b_r)
            h_tilde = np.tanh(np.dot(x_t, self.W_h) + np.dot(r_t * h, self.U_h) + self.b_h)
            h = (1.0 - z_t) * h + z_t * h_tilde
        return h


class GRUFeatureExtractor:
    """Proyección recurrente con pesos aleatorios fijos (no entrenados).

    Se conserva como candidato de la comparación A/B: matemáticamente es una
    proyección aleatoria no lineal de la secuencia de métricas, no un GRU
    aprendido.
    """

    name = "gru_random"

    def __init__(self, window_size: int = 12):
        self.window_size = window_size
        self.gru = DeterministicGRUEncoder(
            input_dim=len(FEATURE_METRIC_ORDER), hidden_dim=32, seed=42
        )

    def feature_names(self) -> list[str]:
        return [f"gru_h{index:02d}" for index in range(self.gru.hidden_dim)]

    def extract(self, samples: list[FaceSample]) -> np.ndarray:
        window = samples[-self.window_size :]
        sequence = _metric_matrix(window)
        return self.gru.encode(sequence).astype(np.float32)


# Alias de compatibilidad con el nombre previo del extractor.
GestureFeatureExtractor = GRUFeatureExtractor


class StatisticalFeatureExtractor:
    """Estadísticos interpretables por métrica sobre la ventana temporal."""

    name = "stats_v1"

    def __init__(self, window_size: int = 12):
        self.window_size = window_size

    def feature_names(self) -> list[str]:
        return [f"{metric}_{stat}" for metric in FEATURE_METRIC_ORDER for stat in STAT_NAMES]

    def extract(self, samples: list[FaceSample]) -> np.ndarray:
        window = samples[-self.window_size :]
        matrix = _metric_matrix(window)
        features: list[float] = []
        for column in range(matrix.shape[1]):
            series = matrix[:, column]
            velocity = np.abs(np.diff(series)) if series.size > 1 else np.zeros(1, dtype=np.float32)
            features.extend(
                [
                    float(series.mean()),
                    float(series.std()),
                    float(series.min()),
                    float(series.max()),
                    float(series[-1] - series[0]),
                    float(velocity.mean()),
                ]
            )
        return np.array(features, dtype=np.float32)


EXTRACTOR_REGISTRY = {
    GRUFeatureExtractor.name: GRUFeatureExtractor,
    StatisticalFeatureExtractor.name: StatisticalFeatureExtractor,
}


def build_extractor(name: str, window_size: int):
    if name not in EXTRACTOR_REGISTRY:
        raise LegacyModelError(
            f"Extractor de features desconocido: '{name}'. Reentrena el modelo desde el Entrenador."
        )
    return EXTRACTOR_REGISTRY[name](window_size=window_size)


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


def serialize_face_sample_compact(sample: FaceSample) -> dict:
    # El reentrenamiento solo consume metrics + timestamp; el resto del
    # FaceSample no se persiste para mantener los datasets livianos.
    return {
        "timestamp_ms": int(sample.timestamp_ms),
        "metrics": {key: float(value) for key, value in sample.metrics.items()},
        "face_scale_px": float(sample.face_scale_px),
    }


def deserialize_face_sample(payload: dict) -> FaceSample:
    landmarks_payload = payload.get("normalized_landmarks")
    if landmarks_payload is not None:
        normalized_landmarks = np.array(landmarks_payload, dtype=np.float32)
    else:
        normalized_landmarks = np.zeros((20, 2), dtype=np.float32)
    points_px = {
        int(key): (int(value[0]), int(value[1]))
        for key, value in payload.get("points_px", {}).items()
    }
    nose_px = payload.get("nose_px", [0, 0])
    face_center_px = payload.get("face_center_px", [0, 0])
    return FaceSample(
        timestamp_ms=int(payload["timestamp_ms"]),
        normalized_landmarks=normalized_landmarks,
        metrics={key: float(value) for key, value in payload["metrics"].items()},
        points_px=points_px,
        nose_px=(int(nose_px[0]), int(nose_px[1])),
        face_scale_px=float(payload.get("face_scale_px", 1.0)),
        face_center_px=(int(face_center_px[0]), int(face_center_px[1])),
    )


def serialize_samples_by_gesture(samples_by_gesture: dict[str, list[FaceSample]]) -> dict:
    return {
        gesture_id: [serialize_face_sample_compact(sample) for sample in samples]
        for gesture_id, samples in samples_by_gesture.items()
    }


def deserialize_samples_by_gesture(payload: dict) -> dict[str, list[FaceSample]]:
    return {
        gesture_id: [deserialize_face_sample(sample_payload) for sample_payload in samples]
        for gesture_id, samples in payload.items()
    }


def _read_dataset_payload(path: Path) -> dict:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


class GestureClassifier:
    def __init__(self, profile_name: str, window_size: int = 12):
        self.profile_name = profile_name
        self.window_size = window_size
        self.extractor = StatisticalFeatureExtractor(window_size=window_size)
        self.profile_dir = profile_calibration_dir(profile_name)
        self.registry_path = model_registry_path(profile_name)
        self.summary_path = active_dataset_summary_path(profile_name)
        self.model_path: Path | None = None
        self.samples_path = self.summary_path
        self.model: LogisticRegression | None = None
        self.labels: list[str] = []
        self.training_summary: dict = {}
        self.active_version: int | None = None
        self.active_dataset_path = None

    def _split_windows_temporally(
        self, windows_by_gesture: dict[str, list], gap: int | None = None
    ) -> tuple[list, list[str], list, list[str]]:
        """Split temporal con gap entre train y validación.

        Las ventanas se generan con stride 1, por lo que ventanas adyacentes
        comparten window_size - 1 frames. Descartar `gap` ventanas entre ambos
        conjuntos garantiza que no compartan ningún frame (sin fuga de datos).
        """
        gap = self.window_size if gap is None else gap
        train_items: list = []
        train_labels: list[str] = []
        validation_items: list = []
        validation_labels: list[str] = []
        for gesture_id, windows in windows_by_gesture.items():
            validation_count = 0
            if len(windows) >= gap + 6:
                validation_count = max(1, int(round((len(windows) - gap) * 0.25)))
            if validation_count:
                train_cutoff = len(windows) - validation_count - gap
                train_slice = windows[:train_cutoff]
                validation_slice = windows[-validation_count:]
            else:
                train_slice = windows
                validation_slice = []
            for item in train_slice:
                train_items.append(item)
                train_labels.append(gesture_id)
            for item in validation_slice:
                validation_items.append(item)
                validation_labels.append(gesture_id)
        return train_items, train_labels, validation_items, validation_labels

    def _recommended_thresholds(
        self,
        true_labels: list[str],
        predicted_labels: np.ndarray,
        probabilities: np.ndarray,
        classes: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, float]]:
        thresholds = {}
        rejection_rates = {}
        for class_index, class_name in enumerate(classes):
            confidences = [
                float(probabilities[row_index][class_index])
                for row_index, true_label in enumerate(true_labels)
                if true_label == class_name and predicted_labels[row_index] == class_name
            ]
            if confidences:
                threshold = float(np.clip(np.quantile(confidences, 0.25), 0.25, 0.90))
            else:
                threshold = 0.30
            thresholds[str(class_name)] = round(threshold, 3)
            total_rows = sum(1 for true_label in true_labels if true_label == class_name)
            if total_rows == 0:
                rejection_rates[str(class_name)] = 0.0
            else:
                rejected = sum(
                    1
                    for row_index, true_label in enumerate(true_labels)
                    if true_label == class_name and float(np.max(probabilities[row_index])) < threshold
                )
                rejection_rates[str(class_name)] = round(rejected / total_rows, 3)
        return thresholds, rejection_rates

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

        if target_version not in versions:
            return False

        version_entry = versions[target_version]
        model_path = self.profile_dir / version_entry.get("model_path", "")
        if not model_path.exists():
            return False
        payload = joblib.load(model_path)
        if not isinstance(payload, dict) or payload.get("format_version") != MODEL_FORMAT_VERSION:
            raise LegacyModelError(
                f"El modelo v{target_version} del perfil '{self.profile_name}' usa un formato "
                "antiguo sin metadatos de extractor. Reentrena desde el Entrenador (pestaña 5. "
                "Entrenamiento); tus datasets versionados permiten reentrenar sin recapturar gestos."
            )

        self.window_size = int(payload.get("window_size", self.window_size))
        self.extractor = build_extractor(payload["extractor_name"], self.window_size)
        expected_features = payload.get("feature_names", [])
        if expected_features and expected_features != self.extractor.feature_names():
            raise LegacyModelError(
                f"El modelo v{target_version} del perfil '{self.profile_name}' fue entrenado con "
                "features que ya no coinciden con las actuales. Reentrena desde el Entrenador."
            )

        self.model_path = model_path
        self.samples_path = self.summary_path
        dataset_name = version_entry.get("dataset_path")
        self.active_dataset_path = str(self.profile_dir / dataset_name) if dataset_name else None
        self.model = payload["model"]
        self.labels = payload["labels"]
        self.training_summary = payload.get("training_summary", {})
        self.active_version = target_version
        return True

    def save(self) -> None:
        if self.model is None or self.model_path is None:
            return
        joblib.dump(
            {
                "format_version": MODEL_FORMAT_VERSION,
                "model": self.model,
                "labels": self.labels,
                "extractor_name": self.extractor.name,
                "feature_names": self.extractor.feature_names(),
                "metric_order": list(FEATURE_METRIC_ORDER),
                "window_size": self.window_size,
                "training_summary": self.training_summary,
            },
            self.model_path,
        )

    def save_samples(self, serialized_samples: dict) -> None:
        self.samples_path.write_text(
            json.dumps(serialized_samples, ensure_ascii=False), encoding="utf-8"
        )

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
                        payload = _read_dataset_payload(dataset_file)
                        return deserialize_samples_by_gesture(payload.get("samples_by_gesture", {}))
        return {}

    def _fit_candidate(
        self,
        extractor,
        train_windows: list[list[FaceSample]],
        train_labels: list[str],
        validation_windows: list[list[FaceSample]],
        validation_labels: list[str],
    ) -> dict:
        feature_matrix = np.vstack([extractor.extract(window) for window in train_windows])
        model = LogisticRegression(
            C=20.0,
            penalty="l2",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
        model.fit(feature_matrix, np.array(train_labels))

        labels = sorted(set(train_labels))
        candidate = {
            "extractor": extractor,
            "model": model,
            "feature_dim": int(feature_matrix.shape[1]),
            "train_windows": int(feature_matrix.shape[0]),
            "validation_accuracy": None,
            "class_metrics": {},
            "confusion_matrix": [],
            "recommended_thresholds": {label: 0.6 for label in labels},
            "low_confidence_rejection_rate": {label: 0.0 for label in labels},
        }
        if validation_windows:
            validation_matrix = np.vstack([extractor.extract(window) for window in validation_windows])
            predicted_labels = model.predict(validation_matrix)
            probabilities = model.predict_proba(validation_matrix)
            candidate["validation_accuracy"] = float(accuracy_score(validation_labels, predicted_labels))
            candidate["class_metrics"] = classification_report(
                validation_labels,
                predicted_labels,
                output_dict=True,
                zero_division=0,
            )
            candidate["confusion_matrix"] = confusion_matrix(
                validation_labels,
                predicted_labels,
                labels=list(model.classes_),
            ).tolist()
            thresholds, rejection_rates = self._recommended_thresholds(
                validation_labels,
                predicted_labels,
                probabilities,
                model.classes_,
            )
            candidate["recommended_thresholds"] = thresholds
            candidate["low_confidence_rejection_rate"] = rejection_rates
        return candidate

    def fit(self, samples_by_gesture: dict[str, list[FaceSample]], capture_quality_summary: dict | None = None) -> None:
        sample_windows_by_gesture: dict[str, list[list[FaceSample]]] = {}
        class_windows: dict[str, int] = {}
        for gesture_id, samples in samples_by_gesture.items():
            if len(samples) < self.window_size:
                continue
            windows = [
                samples[index - self.window_size : index]
                for index in range(self.window_size, len(samples) + 1)
            ]
            sample_windows_by_gesture[gesture_id] = windows
            class_windows[gesture_id] = len(windows)

        labels = sorted(sample_windows_by_gesture.keys())
        if len(labels) < 2:
            raise ValueError("Se necesitan al menos dos clases con muestras suficientes para entrenar.")
        if any(count < 3 for count in class_windows.values()):
            raise ValueError("Cada gesto necesita al menos 3 ventanas útiles para entrenar de forma robusta.")

        # Split único sobre ventanas de muestras crudas: ambos candidatos usan
        # exactamente los mismos índices de train/validación.
        train_windows, train_labels, validation_windows, validation_labels = (
            self._split_windows_temporally(sample_windows_by_gesture)
        )
        if len(set(train_labels)) < 2:
            raise ValueError("Se necesitan al menos dos clases en el conjunto de entrenamiento.")

        registry = self._load_registry()
        existing_versions = [entry["version"] for entry in registry.get("versions", [])]
        next_version = (max(existing_versions) + 1) if existing_versions else 1

        candidates = {
            name: self._fit_candidate(
                build_extractor(name, self.window_size),
                train_windows,
                train_labels,
                validation_windows,
                validation_labels,
            )
            for name in (GRUFeatureExtractor.name, StatisticalFeatureExtractor.name)
        }

        def _selection_score(candidate: dict) -> float:
            accuracy = candidate["validation_accuracy"]
            return -1.0 if accuracy is None else accuracy

        # Empate (incluida la ausencia de validación) → gana stats_v1 por ser
        # interpretable y de menor dimensión.
        stats_name = StatisticalFeatureExtractor.name
        gru_name = GRUFeatureExtractor.name
        selected_name = (
            stats_name
            if _selection_score(candidates[stats_name]) >= _selection_score(candidates[gru_name])
            else gru_name
        )
        winner = candidates[selected_name]

        self.extractor = winner["extractor"]
        self.model = winner["model"]
        self.labels = sorted(set(train_labels))
        self.active_version = next_version
        self.model_path = versioned_model_path(self.profile_name, next_version)
        dataset_path = versioned_dataset_path(self.profile_name, next_version)
        self.active_dataset_path = str(dataset_path)
        self.samples_path = self.summary_path

        trained_at = datetime.now(timezone.utc).isoformat()
        extractor_selection = {
            "selected": selected_name,
            "split": {"policy": "temporal_with_gap", "gap_windows": self.window_size},
            "candidates": {
                name: {
                    "validation_accuracy": candidate["validation_accuracy"],
                    "feature_dim": candidate["feature_dim"],
                    "class_metrics": candidate["class_metrics"],
                }
                for name, candidate in candidates.items()
            },
        }

        self.training_summary = {
            "profile": self.profile_name,
            "version": next_version,
            "trained_at": trained_at,
            "window_size": self.window_size,
            "extractor_name": selected_name,
            "feature_names": self.extractor.feature_names(),
            "feature_dim": winner["feature_dim"],
            "train_windows": winner["train_windows"],
            "validation_windows": int(len(validation_windows)),
            "training_windows": int(winner["train_windows"] + len(validation_windows)),
            "classes": self.labels,
            "windows_per_class": class_windows,
            "model_type": "LogisticRegression",
            "extractor_selection": extractor_selection,
            "model_path": str(self.model_path),
            "dataset_path": str(dataset_path),
            "validation_accuracy": winner["validation_accuracy"],
            "class_metrics": winner["class_metrics"],
            "confusion_matrix": winner["confusion_matrix"],
            "recommended_thresholds": winner["recommended_thresholds"],
            "low_confidence_rejection_rate": winner["low_confidence_rejection_rate"],
            "capture_quality_summary": capture_quality_summary or {},
        }
        self.save()

        dataset_payload = {
            "profile": self.profile_name,
            "version": next_version,
            "saved_at": trained_at,
            "window_size": self.window_size,
            "samples_by_gesture": serialize_samples_by_gesture(samples_by_gesture),
        }
        with gzip.open(dataset_path, "wt", encoding="utf-8") as handle:
            json.dump(dataset_payload, handle, ensure_ascii=False, separators=(",", ":"))

        registry_entry = {
            "version": next_version,
            "trained_at": trained_at,
            "model_path": self.model_path.name,
            "dataset_path": dataset_path.name,
            "classes": self.labels,
            "extractor_name": selected_name,
            "training_windows": int(winner["train_windows"] + len(validation_windows)),
            "train_windows": winner["train_windows"],
            "validation_windows": int(len(validation_windows)),
            "validation_accuracy": winner["validation_accuracy"],
            "recommended_thresholds": winner["recommended_thresholds"],
        }
        registry.setdefault("versions", []).append(registry_entry)
        registry["profile"] = self.profile_name
        registry["active_version"] = next_version
        registry = self._prune_old_versions(registry, keep=MODEL_VERSIONS_TO_KEEP)
        self._save_registry(registry)

        self.save_samples(
            {
                "profile": self.profile_name,
                "active_version": next_version,
                "registry_path": str(self.registry_path),
                "versions": registry.get("versions", []),
                "training_summary": self.training_summary,
                "sample_counts": {name: len(values) for name, values in samples_by_gesture.items()},
                "capture_quality_summary": capture_quality_summary or {},
            }
        )

    def _prune_old_versions(self, registry: dict, keep: int = MODEL_VERSIONS_TO_KEEP) -> dict:
        versions = sorted(registry.get("versions", []), key=lambda entry: entry["version"])
        kept = versions[-keep:] if keep > 0 else versions
        dropped = versions[: len(versions) - len(kept)]

        referenced_names = set()
        for entry in kept:
            for key in ("model_path", "dataset_path"):
                name = entry.get(key)
                if name:
                    referenced_names.add(name)

        for entry in dropped:
            for key in ("model_path", "dataset_path"):
                name = entry.get(key)
                if not name:
                    continue
                try:
                    (self.profile_dir / name).unlink(missing_ok=True)
                except OSError:
                    pass

        # Archivos versionados huérfanos (no referenciados por el registry).
        orphan_patterns = (
            f"{self.profile_name}_gesture_model_v*.pkl",
            f"{self.profile_name}_landmark_dataset_v*.json",
            f"{self.profile_name}_landmark_dataset_v*.json.gz",
        )
        for pattern in orphan_patterns:
            for path in self.profile_dir.glob(pattern):
                if path.name not in referenced_names:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass

        registry["versions"] = kept
        return registry

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

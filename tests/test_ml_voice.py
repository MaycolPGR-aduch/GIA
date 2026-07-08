import json
import time
import unittest
import uuid
from collections import deque
from shutil import rmtree

import joblib
import numpy as np

from app.assistive_controls import AssistiveController
from app.audio_utils import SilenceEndpointer
from app.config_store import model_registry_path, profile_calibration_dir
from app.gesture_ml import (
    FEATURE_METRIC_ORDER,
    GestureClassifier,
    LegacyModelError,
    StatisticalFeatureExtractor,
    deserialize_face_sample,
    serialize_face_sample_compact,
)
from app.models import AppState, FaceSample
from app.voice_router import resolve_command


def make_sample(ts: int, metrics: dict | None = None) -> FaceSample:
    base_metrics = {
        "left_eye_ratio": 0.22 + ts * 0.001,
        "right_eye_ratio": 0.21 + ts * 0.001,
        "mouth_open_ratio": 0.10 + ts * 0.001,
        "smile_ratio": 0.30 + ts * 0.001,
        "brow_raise_ratio": 0.08 + ts * 0.001,
    }
    if metrics:
        base_metrics.update(metrics)
    return FaceSample(
        timestamp_ms=ts,
        normalized_landmarks=np.full((20, 2), fill_value=0.1 * ts, dtype=np.float32),
        metrics=base_metrics,
        points_px={1: (100, 100), 152: (100, 200)},
        nose_px=(100, 100),
        face_scale_px=120.0,
        face_center_px=(100, 120),
    )


def make_shifted_sample(ts: int, shift: float) -> FaceSample:
    sample = make_sample(ts)
    sample.metrics = {key: value + shift for key, value in sample.metrics.items()}
    return sample


class TemporalSplitTests(unittest.TestCase):
    def test_split_with_gap_has_no_overlap(self):
        profile_name = f"split_profile_{uuid.uuid4().hex[:8]}"
        profile_dir = profile_calibration_dir(profile_name)
        try:
            classifier = GestureClassifier(profile_name, window_size=4)
            windows = {"gesture_a": list(range(30)), "gesture_b": list(range(5))}
            train_items, train_labels, val_items, val_labels = classifier._split_windows_temporally(windows)

            a_train = [item for item, label in zip(train_items, train_labels) if label == "gesture_a"]
            a_val = [item for item, label in zip(val_items, val_labels) if label == "gesture_a"]
            self.assertTrue(a_val, "La clase grande debe tener ventanas de validación")
            # Gap de window_size ventanas: con stride 1 garantiza cero frames compartidos.
            self.assertLess(max(a_train) + classifier.window_size, min(a_val))
            gap_indices = set(range(max(a_train) + 1, min(a_val)))
            self.assertTrue(gap_indices.isdisjoint(a_train))
            self.assertTrue(gap_indices.isdisjoint(a_val))

            # Clase pequeña: va completa a entrenamiento, sin validación.
            b_train = [item for item, label in zip(train_items, train_labels) if label == "gesture_b"]
            b_val = [item for item, label in zip(val_items, val_labels) if label == "gesture_b"]
            self.assertEqual(b_train, list(range(5)))
            self.assertEqual(b_val, [])
        finally:
            if profile_dir.exists():
                rmtree(profile_dir)


class StatisticalExtractorTests(unittest.TestCase):
    def test_known_ramp_statistics(self):
        window_size = 12
        ramp = [round(0.1 * i, 6) for i in range(window_size)]  # 0.0 .. 1.1
        samples = [
            make_sample(i, metrics={"left_eye_ratio": ramp[i]})
            for i in range(window_size)
        ]
        extractor = StatisticalFeatureExtractor(window_size=window_size)
        vector = extractor.extract(samples)
        names = extractor.feature_names()

        self.assertEqual(vector.shape, (30,))
        self.assertEqual(len(names), 30)
        self.assertEqual(len(set(names)), 30)

        base = FEATURE_METRIC_ORDER.index("left_eye_ratio") * 6
        self.assertEqual(names[base], "left_eye_ratio_mean")
        self.assertAlmostEqual(vector[base], 0.55, places=5)          # mean
        self.assertAlmostEqual(vector[base + 2], 0.0, places=5)       # min
        self.assertAlmostEqual(vector[base + 3], 1.1, places=5)       # max
        self.assertAlmostEqual(vector[base + 4], 1.1, places=5)       # delta
        self.assertAlmostEqual(vector[base + 5], 0.1, places=5)       # mean_abs_velocity


class ExtractorSelectionTests(unittest.TestCase):
    def test_ab_selection_and_roundtrip(self):
        profile_name = f"ab_profile_{uuid.uuid4().hex[:8]}"
        profile_dir = profile_calibration_dir(profile_name)
        try:
            classifier = GestureClassifier(profile_name, window_size=4)
            samples = {
                "neutral": [make_sample(i) for i in range(20)],
                "smile": [make_shifted_sample(i, 0.4) for i in range(20)],
            }
            classifier.fit(samples)

            selection = classifier.training_summary["extractor_selection"]
            self.assertEqual(set(selection["candidates"].keys()), {"gru_random", "stats_v1"})
            for candidate in selection["candidates"].values():
                self.assertIsInstance(candidate["validation_accuracy"], float)
            self.assertEqual(classifier.extractor.name, selection["selected"])
            self.assertEqual(
                classifier.training_summary["feature_names"],
                classifier.extractor.feature_names(),
            )

            reloaded = GestureClassifier(profile_name, window_size=4)
            self.assertTrue(reloaded.load())
            self.assertEqual(reloaded.extractor.name, selection["selected"])
            window = deque([make_sample(i) for i in range(4)], maxlen=4)
            prediction = reloaded.predict(window)
            self.assertIsNotNone(prediction)
            self.assertIn(prediction.gesture_id, {"neutral", "smile"})
        finally:
            if profile_dir.exists():
                rmtree(profile_dir)

    def test_legacy_model_is_rejected(self):
        profile_name = f"legacy_profile_{uuid.uuid4().hex[:8]}"
        profile_dir = profile_calibration_dir(profile_name)
        try:
            model_name = f"{profile_name}_gesture_model_v1.pkl"
            joblib.dump({"model": None, "labels": ["neutral"]}, profile_dir / model_name)
            registry = {
                "profile": profile_name,
                "active_version": 1,
                "versions": [{"version": 1, "model_path": model_name, "dataset_path": None}],
            }
            model_registry_path(profile_name).write_text(
                json.dumps(registry), encoding="utf-8"
            )
            classifier = GestureClassifier(profile_name, window_size=4)
            with self.assertRaises(LegacyModelError) as context:
                classifier.load()
            self.assertIn("Reentrena", str(context.exception))
        finally:
            if profile_dir.exists():
                rmtree(profile_dir)

    def test_compact_roundtrip_and_version_pruning(self):
        sample = make_sample(3)
        restored = deserialize_face_sample(serialize_face_sample_compact(sample))
        self.assertEqual(restored.timestamp_ms, sample.timestamp_ms)
        self.assertEqual(restored.metrics, sample.metrics)
        self.assertAlmostEqual(restored.face_scale_px, sample.face_scale_px)

        profile_name = f"prune_profile_{uuid.uuid4().hex[:8]}"
        profile_dir = profile_calibration_dir(profile_name)
        try:
            classifier = GestureClassifier(profile_name, window_size=4)
            samples = {
                "neutral": [make_sample(i) for i in range(8)],
                "smile": [make_shifted_sample(i, 0.4) for i in range(8)],
            }
            for _ in range(4):
                classifier.fit(samples)

            versions = classifier.list_versions()
            self.assertEqual([entry["version"] for entry in versions], [2, 3, 4])
            self.assertFalse((profile_dir / f"{profile_name}_gesture_model_v1.pkl").exists())
            self.assertFalse((profile_dir / f"{profile_name}_landmark_dataset_v1.json.gz").exists())
            self.assertTrue((profile_dir / f"{profile_name}_gesture_model_v4.pkl").exists())

            loaded_samples = classifier.load_active_dataset()
            self.assertEqual(len(loaded_samples["neutral"]), 8)
        finally:
            if profile_dir.exists():
                rmtree(profile_dir)


class VoiceMatchingTests(unittest.TestCase):
    def test_exact_command_matches(self):
        command, score = resolve_command("cerrar sistema")
        self.assertEqual(command, "quit")
        self.assertGreaterEqual(score, 0.95)
        command, score = resolve_command("pausa")
        self.assertEqual(command, "pause")
        self.assertGreaterEqual(score, 0.95)

    def test_negation_vetoes_command(self):
        _, score = resolve_command("no quiero cerrar sistema")
        self.assertLess(score, 0.68)
        _, score = resolve_command("cancela cerrar sistema")
        self.assertLess(score, 0.68)

    def test_low_coverage_gets_no_boost(self):
        _, score = resolve_command("hoy estuve pensando que tal vez deberia cerrar sistema")
        self.assertLess(score, 0.68)


class SilenceEndpointerTests(unittest.TestCase):
    SAMPLE_RATE = 16000

    def _quiet_block(self) -> np.ndarray:
        return np.full(int(self.SAMPLE_RATE * 0.1), 0.001, dtype=np.float32)

    def _speech_block(self) -> np.ndarray:
        t = np.arange(int(self.SAMPLE_RATE * 0.1)) / self.SAMPLE_RATE
        return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    def test_silence_after_speech_cuts_recording(self):
        endpointer = SilenceEndpointer(self.SAMPLE_RATE)
        for _ in range(3):
            self.assertIsNone(endpointer.feed(self._quiet_block()))
        for _ in range(10):
            self.assertIsNone(endpointer.feed(self._speech_block()))
        self.assertTrue(endpointer.speech_detected)
        reason = None
        blocks = 0
        while reason is None and blocks < 20:
            reason = endpointer.feed(np.zeros(int(self.SAMPLE_RATE * 0.1), dtype=np.float32))
            blocks += 1
        self.assertEqual(reason, "silence")
        self.assertEqual(blocks, 8)  # 0.8 s de silencio en bloques de 0.1 s

    def test_start_timeout_without_speech(self):
        endpointer = SilenceEndpointer(self.SAMPLE_RATE)
        reason = None
        blocks = 0
        while reason is None and blocks < 40:
            reason = endpointer.feed(self._quiet_block())
            blocks += 1
        self.assertEqual(reason, "start_timeout")
        self.assertFalse(endpointer.speech_detected)

    def test_max_duration_with_continuous_speech(self):
        endpointer = SilenceEndpointer(self.SAMPLE_RATE)
        reason = None
        blocks = 0
        while reason is None and blocks < 100:
            reason = endpointer.feed(self._speech_block())
            blocks += 1
        self.assertEqual(reason, "max_duration")


class QuitConfirmationTests(unittest.TestCase):
    def _make_controller(self, executed: list):
        controller = AssistiveController.__new__(AssistiveController)
        controller._pending_quit_deadline = 0.0
        controller.hablar = lambda text: None
        controller._publish_event = lambda text: None
        controller.action_service = type(
            "ActionServiceStub",
            (),
            {
                "execute_builtin_voice_command": lambda self, command_id, app_state: executed.append(
                    (command_id, app_state)
                ),
                "execute_custom_voice_command": lambda self, entry, app_state: executed.append(
                    (entry["id"], app_state)
                ),
            },
        )()
        return controller

    QUIT_ENTRY = {"id": "quit", "kind": "builtin", "label": "Cerrar sistema"}
    PAUSE_ENTRY = {"id": "pause", "kind": "builtin", "label": "Pausar sistema"}

    def test_first_quit_requires_confirmation(self):
        executed = []
        controller = self._make_controller(executed)
        handled, status = controller._dispatch_voice_command(
            self.QUIT_ENTRY, 0.95, "cerrar sistema", AppState.READY
        )
        self.assertTrue(handled)
        self.assertEqual(executed, [])
        self.assertGreater(controller._pending_quit_deadline, time.monotonic())
        self.assertIn("confirmaci", status.lower())

    def test_second_quit_executes(self):
        executed = []
        controller = self._make_controller(executed)
        controller._dispatch_voice_command(self.QUIT_ENTRY, 0.95, "cerrar sistema", AppState.READY)
        handled, _ = controller._dispatch_voice_command(
            self.QUIT_ENTRY, 0.95, "cerrar sistema", AppState.READY
        )
        self.assertTrue(handled)
        self.assertEqual(executed, [("quit", "ready")])
        self.assertEqual(controller._pending_quit_deadline, 0.0)

    def test_confirmation_phrase_executes_pending_quit(self):
        executed = []
        controller = self._make_controller(executed)
        controller._dispatch_voice_command(self.QUIT_ENTRY, 0.95, "cerrar sistema", AppState.READY)
        handled, _ = controller._dispatch_voice_command(None, 0.2, "confirmar", AppState.READY)
        self.assertTrue(handled)
        self.assertEqual(executed, [("quit", "ready")])

    def test_other_command_cancels_pending_quit(self):
        executed = []
        controller = self._make_controller(executed)
        controller._dispatch_voice_command(self.QUIT_ENTRY, 0.95, "cerrar sistema", AppState.READY)
        handled, _ = controller._dispatch_voice_command(
            self.PAUSE_ENTRY, 0.95, "pausar sistema", AppState.READY
        )
        self.assertTrue(handled)
        self.assertEqual(executed, [("pause", "ready")])
        self.assertEqual(controller._pending_quit_deadline, 0.0)


if __name__ == "__main__":
    unittest.main()

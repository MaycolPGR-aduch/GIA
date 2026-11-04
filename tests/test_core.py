import json
import joblib
import threading
import time
import unittest
import uuid
from unittest import mock
from pathlib import Path
from shutil import rmtree

import numpy as np

import app.assistive_controls as assistive_controls_module
import app.action_service as action_service_module
from app.assistive_controls import AssistiveController
from app.action_service import ActionService
from app.calibration_service import CalibrationService
from app.config_store import build_default_profile, load_profile, load_settings, profile_calibration_dir, save_profile
from app.gesture_ml import (
    GestureClassifier,
    ProbabilitySmoother,
    GestureFeatureExtractor,
    deserialize_face_sample,
    serialize_face_sample,
)
from app.heuristic_engine import HeuristicEngine
from app.inference_service import InferenceService
from app.models import AppState, FaceSample, GesturePrediction
from app.voice_router import find_phrase_conflicts, normalize_voice_settings, resolve_command, resolve_command_entry
from app.config_store import CALIBRATION_DIR, get_profile_summary
from app.ui_theme import PALETTES, build_stylesheet
from app.audio_utils import pick_fallback_input_device
from app.gesture_catalog import CONFIDENCE_THRESHOLD_MAX, CONFIDENCE_THRESHOLD_MIN, NEUTRAL_DYNAMIC_ID
from app.gesture_ml import BLENDSHAPE_NAMES, compute_signal_limits
from app.landmark_provider import head_pose_from_matrix
from app.face_metrics import BROW_LEFT, BROW_RIGHT, FACE_WIDTH, LEFT_EYE, MOUTH_OPEN, RIGHT_EYE, feature_metrics, legacy_metrics
from app.gesture_catalog import GESTURE_CATALOG
from sklearn.linear_model import LogisticRegression
from app.speech import SpeechQueue
from app.usage_recorder import UsageRecorder, read_usage_recording
from app.preflight import evaluate_preflight, summarize_preflight
from app.runtime_app import voice_status


def make_sample(ts: int) -> FaceSample:
    landmarks = np.full((20, 2), fill_value=0.1 * ts, dtype=np.float32)
    metrics = {
        "left_eye_ratio": 0.22 + ts * 0.001,
        "right_eye_ratio": 0.21 + ts * 0.001,
        "mouth_open_ratio": 0.10 + ts * 0.001,
        "smile_ratio": 0.30 + ts * 0.001,
        "brow_raise_ratio": 0.08 + ts * 0.001,
    }
    return FaceSample(
        timestamp_ms=ts,
        normalized_landmarks=landmarks,
        metrics=metrics,
        points_px={1: (100, 100), 152: (100, 200)},
        nose_px=(100, 100),
        face_scale_px=120.0,
        face_center_px=(100, 120),
    )


def make_shifted_sample(ts: int, shift: float) -> FaceSample:
    sample = make_sample(ts)
    sample.normalized_landmarks = sample.normalized_landmarks + shift
    sample.metrics = {key: value + shift for key, value in sample.metrics.items()}
    return sample


class CoreTests(unittest.TestCase):
    def test_default_profile_shape(self):
        profile = build_default_profile("tester")
        self.assertEqual(profile["name"], "tester")
        self.assertIn("gesture_confidence", profile)
        self.assertIn("calibration", profile)

    def test_settings_load(self):
        settings = load_settings()
        self.assertIn("fps", settings)
        self.assertIn("gesture_window_size", settings)
        self.assertIn("audio_input_device", settings)
        self.assertIn("voice_sample_rate", settings)
        self.assertIn("voice_record_seconds", settings)
        self.assertIn("voice_model_size", settings)
        self.assertIn("voice_commands_enabled", settings)
        self.assertIn("voice_builtin_alias_overrides", settings)
        self.assertIn("voice_custom_commands", settings)

    def test_profile_confidence_preserves_explicit_thresholds(self):
        profile_name = f"threshold_profile_{uuid.uuid4().hex[:8]}"
        profile = build_default_profile(profile_name)
        profile["gesture_confidence"]["both_eyes_closed_intent"] = 0.82
        save_profile(profile)
        try:
            loaded = load_profile(profile_name)
            self.assertEqual(loaded["gesture_confidence"]["both_eyes_closed_intent"], 0.82)
        finally:
            profile_path = Path("config") / "user_profiles" / f"{profile_name}.json"
            if profile_path.exists():
                profile_path.unlink()

    def test_load_profile_does_not_mutate_file_on_read(self):
        profile_name = f"read_only_profile_{uuid.uuid4().hex[:8]}"
        profile = build_default_profile(profile_name)
        profile["gesture_confidence"] = {"smile": 0.77}
        save_profile(profile)
        profile_path = Path("config") / "user_profiles" / f"{profile_name}.json"
        try:
            original_text = profile_path.read_text(encoding="utf-8")
            loaded = load_profile(profile_name)
            reloaded_text = profile_path.read_text(encoding="utf-8")
            self.assertEqual(original_text, reloaded_text)
            self.assertEqual(loaded["gesture_confidence"]["smile"], 0.77)
            self.assertIn("neutral", loaded["gesture_confidence"])
            self.assertIn("left_blink_intent", loaded["gesture_duration_ms"])
            self.assertIn("right_blink_intent", loaded["gesture_cooldown_ms"])
        finally:
            if profile_path.exists():
                profile_path.unlink()

    def test_voice_router(self):
        command, score = resolve_command("pausar sistema")
        self.assertEqual(command, "pause")
        self.assertGreaterEqual(score, 0.9)
        command, score = resolve_command("activar cursor")
        self.assertEqual(command, "cursor_on")
        self.assertGreaterEqual(score, 0.9)
        command, score = resolve_command("estamos listos")
        self.assertEqual(command, "compact_ui")
        self.assertGreaterEqual(score, 0.9)
        command, score = resolve_command("volvamos")
        self.assertEqual(command, "expand_ui")
        self.assertGreaterEqual(score, 0.9)

    def test_voice_router_builtin_alias_override(self):
        settings = normalize_voice_settings(
            {
                "voice_builtin_alias_overrides": {
                    "pause": ["detener sistema"],
                }
            }
        )
        command, score = resolve_command("detener sistema", settings)
        self.assertEqual(command, "pause")
        self.assertGreaterEqual(score, 0.9)

    def test_voice_router_custom_macro_command(self):
        settings = normalize_voice_settings(
            {
                "voice_custom_commands": [
                    {
                        "id": "copiar",
                        "label": "Copiar",
                        "spoken_phrases": ["copiar"],
                        "action_type": "macro",
                        "macro_keys": ["ctrl", "c"],
                        "enabled": True,
                    }
                ]
            }
        )
        entry, score = resolve_command_entry("copiar", settings)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["kind"], "custom")
        self.assertEqual(entry["action_type"], "macro")
        self.assertEqual(entry["macro_keys"], ["ctrl", "c"])
        self.assertGreaterEqual(score, 0.9)

    def test_voice_router_custom_direct_action(self):
        settings = normalize_voice_settings(
            {
                "voice_custom_commands": [
                    {
                        "id": "dictado",
                        "label": "Modo dictado",
                        "spoken_phrases": ["modo dictado"],
                        "action_type": "direct_action",
                        "direct_action_id": "start_dictation",
                        "enabled": True,
                    }
                ]
            }
        )
        entry, score = resolve_command_entry("modo dictado", settings)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["direct_action_id"], "start_dictation")
        self.assertGreaterEqual(score, 0.9)

    def test_voice_router_rejects_duplicate_phrases(self):
        settings = normalize_voice_settings(
            {
                "voice_custom_commands": [
                    {
                        "id": "copiar",
                        "label": "Copiar",
                        "spoken_phrases": ["copiar"],
                        "action_type": "macro",
                        "macro_keys": ["ctrl", "c"],
                        "enabled": False,
                    }
                ]
            }
        )
        conflicts = find_phrase_conflicts(["copiar"], settings)
        self.assertEqual(conflicts, ["Copiar"])

    def test_feature_extractor(self):
        extractor = GestureFeatureExtractor(window_size=4)
        vector = extractor.extract([make_sample(i) for i in range(4)])
        self.assertTrue(vector.size > 0)
        self.assertEqual(vector.ndim, 1)

    def test_heuristic_engine_velocity_mode(self):
        profile = build_default_profile("engine_tester")
        profile["invert_x"] = True
        profile["invert_y"] = False
        engine = HeuristicEngine(profile)
        neutral = make_sample(0)
        engine.recenter(neutral)
        moved = make_sample(1)
        moved.nose_px = (neutral.nose_px[0] + 24, neutral.nose_px[1])
        state = engine.update(moved)
        self.assertLess(state.smoothed_dx, 0.0)
        self.assertEqual(state.face_present, True)

    def test_face_sample_roundtrip(self):
        sample = make_sample(3)
        restored = deserialize_face_sample(serialize_face_sample(sample))
        self.assertEqual(restored.timestamp_ms, sample.timestamp_ms)
        self.assertEqual(restored.nose_px, sample.nose_px)
        self.assertTrue(np.allclose(restored.normalized_landmarks, sample.normalized_landmarks))

    def test_classifier_versioning_and_dataset_reload(self):
        profile_name = f"test_versioning_profile_{uuid.uuid4().hex[:8]}"
        profile_dir = profile_calibration_dir(profile_name)
        if profile_dir.exists():
            rmtree(profile_dir)
        try:
            classifier = GestureClassifier(profile_name, window_size=4)
            samples = {
                "neutral": [make_sample(i) for i in range(8)],
                "left_blink_intent": [make_shifted_sample(i, 0.4) for i in range(8)],
            }
            classifier.fit(samples, capture_quality_summary={"neutral": {"valid_frames": 8}})
            self.assertEqual(classifier.active_version, 1)
            self.assertTrue(classifier.model_path.exists())
            loaded_samples = classifier.load_active_dataset()
            self.assertIn("neutral", loaded_samples)
            self.assertEqual(len(loaded_samples["neutral"]), 8)
            self.assertIn("recommended_thresholds", classifier.training_summary)
            self.assertIn("class_metrics", classifier.training_summary)

            classifier.fit(samples)
            self.assertEqual(classifier.active_version, 2)
            self.assertEqual(len(classifier.list_versions()), 2)
        finally:
            if profile_dir.exists():
                rmtree(profile_dir)

    def test_calibration_service_tracks_readiness_and_pending_retraining(self):
        service = CalibrationService(window_size=4)
        self.assertNotIn("brows_up", service.gesture_order)
        samples = {
            "neutral": [make_sample(i) for i in range(6)],
            "left_blink_intent": [make_shifted_sample(i, 0.4) for i in range(6)],
        }
        service.load_samples(samples)
        ready, blockers, statuses = service.can_train()
        self.assertFalse(ready)
        self.assertIn("Postura neutral", blockers)
        self.assertEqual(statuses["neutral"].readiness, "insuficiente")

        for gesture_id in service.gesture_order:
            min_frames = service.gesture_meta[gesture_id]["recommended_min_frames"]
            service.samples_by_gesture[gesture_id] = [make_shifted_sample(i, 0.3) for i in range(min_frames)]
        ready, blockers, statuses = service.can_train()
        self.assertTrue(ready)
        self.assertEqual(blockers, [])
        self.assertEqual(statuses["neutral"].readiness, "listo para entrenar")

    def test_inference_service_reports_low_confidence_rejection(self):
        profile = build_default_profile("inference_profile")
        service = InferenceService(profile, window_size=4)
        sample = make_sample(1)
        state = type("State", (), {"face_stable": True, "status_text": "ok"})()
        prediction = GesturePrediction("smile", 0.2, False)
        result = service.evaluate(sample, state, prediction, AppState.READY, "activo")
        self.assertEqual(result.rejection_reason, "confianza baja")
        self.assertEqual(result.runtime_state, "rechazado")

    def test_inference_service_accepts_after_duration(self):
        profile = build_default_profile("inference_accept")
        profile["gesture_duration_ms"]["smile"] = 0
        service = InferenceService(profile, window_size=4)
        sample = make_sample(1)
        state = type("State", (), {"face_stable": True, "status_text": "ok"})()
        prediction = GesturePrediction("smile", 0.91, False)
        first = service.evaluate(sample, state, prediction, AppState.READY, "activo")
        self.assertEqual(first.runtime_state, "validando gesto")
        second = service.evaluate(sample, state, prediction, AppState.READY, "activo")
        self.assertTrue(second.executed)

    def test_inference_service_rejects_disabled_brows_up(self):
        profile = build_default_profile("disabled_brows")
        service = InferenceService(profile, window_size=4)
        sample = make_sample(1)
        state = type("State", (), {"face_stable": True, "status_text": "ok"})()
        prediction = GesturePrediction("brows_up", 0.95, False)
        result = service.evaluate(sample, state, prediction, AppState.READY, "activo")
        self.assertEqual(result.rejection_reason, "gesto deshabilitado")
        self.assertEqual(result.runtime_state, "rechazado")

    def test_inference_service_allows_voice_listener_while_paused(self):
        profile = build_default_profile("paused_voice")
        profile["gesture_duration_ms"]["both_eyes_closed_intent"] = 0
        service = InferenceService(profile, window_size=4)
        sample = make_sample(1)
        sample.metrics["left_eye_ratio"] = 0.16
        sample.metrics["right_eye_ratio"] = 0.16
        state = type("State", (), {"face_stable": True, "status_text": "ok"})()
        prediction = GesturePrediction("both_eyes_closed_intent", 0.91, False)
        first = service.evaluate(sample, state, prediction, AppState.PAUSED, "activo")
        second = service.evaluate(sample, state, prediction, AppState.PAUSED, "activo")
        self.assertEqual(first.runtime_state, "validando gesto")
        self.assertTrue(second.executed)

    def test_hybrid_eye_closed_validation(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.hybrid_eye_closed_confidence_floor = 0.18
        controller.hybrid_eye_closed_ratio_max = 0.19
        controller.hybrid_eye_closed_asymmetry_max = 0.08
        sample = make_sample(1)
        sample.metrics["left_eye_ratio"] = 0.16
        sample.metrics["right_eye_ratio"] = 0.11

        class Prediction:
            gesture_id = "both_eyes_closed_intent"
            confidence = 0.21

        self.assertTrue(controller._passes_hybrid_gesture_validation(Prediction(), sample))

    def test_ui_available_when_only_compact_window_is_visible(self):
        controller = AssistiveController.__new__(AssistiveController)

        class CompactWindow:
            def isVisible(self):
                return True

        class MainWindow:
            compact_window = CompactWindow()

            def isVisible(self):
                return False

        controller.main_gui_interface = MainWindow()
        self.assertTrue(controller._ui_available())

    def test_heuristic_blink_prediction_left(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.hybrid_eye_closed_ratio_max = 0.19
        controller.heuristic_blink_closed_ratio = 0.17
        controller.heuristic_blink_open_ratio = 0.20
        controller.heuristic_blink_asymmetry_min = 0.04
        controller.heuristic_blink_confidence = 0.72
        sample = make_sample(2)
        sample.metrics["left_eye_ratio"] = 0.14
        sample.metrics["right_eye_ratio"] = 0.24

        prediction = controller._build_heuristic_blink_prediction(sample)
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.gesture_id, "left_blink_intent")
        self.assertAlmostEqual(prediction.confidence, 0.72)

    def test_heuristic_blink_prediction_ignores_both_eyes_closed(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.hybrid_eye_closed_ratio_max = 0.19
        controller.heuristic_blink_closed_ratio = 0.17
        controller.heuristic_blink_open_ratio = 0.20
        controller.heuristic_blink_asymmetry_min = 0.04
        controller.heuristic_blink_confidence = 0.72
        controller.gesture_thresholds = {
            "left_blink_intent": 0.5,
            "right_blink_intent": 0.5,
        }
        sample = make_sample(2)
        sample.metrics["left_eye_ratio"] = 0.13
        sample.metrics["right_eye_ratio"] = 0.11

        prediction = controller._build_heuristic_blink_prediction(sample)
        self.assertIsNone(prediction)

        should_use = controller._should_use_heuristic_blink(
            GesturePrediction("left_blink_intent", 0.61, False),
            None,
        )
        self.assertFalse(should_use)

    def test_voice_listener_uses_previous_state_for_resume_command(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.state = AppState.PAUSED
        controller.last_voice_state = "-"
        controller.last_voice_text = "-"
        controller.last_voice_command = "-"
        controller.settings = {}
        controller.voice_record_seconds = 1.0
        controller.voice_sample_rate = 16000
        controller.audio_input_device = None
        controller.audio_level_monitor = None
        controller.voice_model_size = "small"
        controller._cursor_status_text = lambda: "congelado"
        controller._push_status = lambda *args, **kwargs: None
        controller._publish_event = lambda text: None
        controller.hablar = lambda text, **kwargs: None
        controller.logger = type("Logger", (), {"log": lambda self, *args, **kwargs: None})()
        controller.classifier = type("Classifier", (), {"active_version": 2})()
        executed = []
        controller.action_service = type(
            "ActionServiceStub",
            (),
            {
                "execute_builtin_voice_command": lambda self, command_id, app_state: executed.append(
                    (command_id, app_state)
                ),
                "execute_custom_voice_command": lambda self, command_entry, app_state: executed.append(
                    (command_entry["id"], app_state)
                ),
            },
        )()
        controller.whisper = type(
            "WhisperStub",
            (),
            {"transcribe": lambda self, *args, **kwargs: ([type("Segment", (), {"text": "reanudar sistema"})()], None)},
        )()

        original_capture = assistive_controls_module.capture_voice_clip
        original_save = assistive_controls_module.save_audio_clip
        assistive_controls_module.capture_voice_clip = lambda *args, **kwargs: (
            np.zeros(16000, dtype=np.float32),
            {
                "device_label": "Predeterminado del sistema",
                "sample_rate": 16000,
                "duration_s": 1.0,
                "rms": 0.05,
                "peak": 0.40,
                "active_ratio": 0.30,
                "clip_ratio": 0.0,
            },
        )
        assistive_controls_module.save_audio_clip = lambda *args, **kwargs: None
        try:
            controller._listen_once(trigger_source="manual")
        finally:
            assistive_controls_module.capture_voice_clip = original_capture
            assistive_controls_module.save_audio_clip = original_save

        self.assertEqual(executed, [("resume", "paused")])
        self.assertEqual(controller.state, AppState.PAUSED)
        self.assertEqual(controller.last_voice_state, "comando reconocido")

    def test_mic_status_payload_uses_monitor_snapshot(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.audio_level_monitor = type(
            "AudioMonitorStub",
            (),
            {
                "snapshot": lambda self: {
                    "device_label": "Mic USB",
                    "available": True,
                    "suspended": False,
                    "level_percent": 42,
                    "peak_percent": 87,
                    "active_ratio": 0.33,
                    "error": "",
                }
            },
        )()

        payload = controller._get_mic_status_payload()

        self.assertEqual(payload["mic_device"], "Mic USB")
        self.assertEqual(payload["mic_level"], 42)
        self.assertEqual(payload["mic_peak"], 87)
        self.assertAlmostEqual(payload["mic_active_ratio"], 0.33)
        self.assertEqual(payload["mic_monitor_status"], "activo")

    def test_action_service_start_dictation_uses_win_h(self):
        logs = []
        service = ActionService(
            type("Logger", (), {"log": lambda self, *args, **kwargs: logs.append((args, kwargs))})(),
            on_voice_listener=lambda: None,
            on_toggle_cursor=lambda: None,
            on_recenter=lambda: None,
            on_toggle_pause=lambda: None,
            on_set_cursor=lambda value: None,
            on_quit=lambda: None,
            main_gui_interface=None,
            audio_endpoint=None,
        )
        original_hotkey = action_service_module.pyautogui.hotkey
        pressed = []
        action_service_module.pyautogui.hotkey = lambda *keys: pressed.append(keys)
        try:
            service.execute_direct_action("start_dictation", "ready")
        finally:
            action_service_module.pyautogui.hotkey = original_hotkey
        self.assertEqual(pressed, [("win", "h")])

    def test_action_service_compact_ui_uses_ui_thread_runner(self):
        calls = []
        ui_calls = []

        class UiStub:
            def run_on_ui_thread(self, callback):
                ui_calls.append("queued")
                callback()

            def enter_compact_mode(self):
                calls.append("compact")

        service = ActionService(
            type("Logger", (), {"log": lambda self, *args, **kwargs: None})(),
            on_voice_listener=lambda: None,
            on_toggle_cursor=lambda: None,
            on_recenter=lambda: None,
            on_toggle_pause=lambda: None,
            on_set_cursor=lambda value: None,
            on_quit=lambda: None,
            main_gui_interface=UiStub(),
            audio_endpoint=None,
        )

        service.execute_direct_action("compact_ui", "ready")

        self.assertEqual(ui_calls, ["queued"])
        self.assertEqual(calls, ["compact"])

    def test_build_stylesheet_supports_themes_and_font_scale(self):
        dark = build_stylesheet("dark", 1.0)
        high_contrast = build_stylesheet("high_contrast", 1.0)
        scaled = build_stylesheet("dark", 1.5)
        self.assertIn(PALETTES["dark"]["bg"], dark)
        self.assertIn(PALETTES["high_contrast"]["accent"], high_contrast)
        self.assertNotEqual(dark, high_contrast)
        self.assertIn("font-size: 13px", dark)
        self.assertIn("font-size: 20px", scaled)

    def test_profile_summary_reads_active_model_without_creating_dirs(self):
        profile_name = f"summary_profile_{uuid.uuid4().hex[:8]}"
        profile = build_default_profile(profile_name)
        profile["calibration"]["completed"] = True
        save_profile(profile)
        calibration_dir = CALIBRATION_DIR / profile_name
        profile_path = Path("config") / "user_profiles" / f"{profile_name}.json"
        try:
            summary = get_profile_summary(profile_name)
            self.assertTrue(summary["calibrated"])
            self.assertIsNone(summary["model_version"])
            self.assertFalse(calibration_dir.exists())

            calibration_dir.mkdir(parents=True)
            (calibration_dir / "model_registry.json").write_text(
                json.dumps({
                    "active_version": 2,
                    "versions": [
                        {"version": 1, "validation_accuracy": 0.5, "trained_at": "2026-01-01T00:00:00", "classes": ["a"]},
                        {"version": 2, "validation_accuracy": 0.95, "trained_at": "2026-02-01T00:00:00", "classes": ["a", "b"]},
                    ],
                }),
                encoding="utf-8",
            )
            summary = get_profile_summary(profile_name)
            self.assertEqual(summary["model_version"], 2)
            self.assertAlmostEqual(summary["validation_accuracy"], 0.95)
            self.assertEqual(summary["classes"], ["a", "b"])
        finally:
            if profile_path.exists():
                profile_path.unlink()
            if calibration_dir.exists():
                rmtree(calibration_dir)

    def _payload_controller(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.audio_level_monitor = None
        controller.state = AppState.READY
        controller.cursor_active = True
        controller.last_voice_state = "en espera"
        controller.last_voice_text = "-"
        controller.last_voice_command = "-"
        controller.last_rejection_reason = "-"
        controller.last_diagnostic_hint = "-"
        controller.fps_real = 14.24
        controller.classifier = type("Classifier", (), {"active_version": 3})()
        return controller

    def test_status_payload_includes_execution_fields(self):
        controller = self._payload_controller()
        payload = controller._build_status_payload(
            "Gesto ejecutado",
            "left_blink_intent",
            0.91,
            runtime_state="ejecutado",
            gesture_executed=True,
        )
        self.assertTrue(payload["gesture_executed"])
        self.assertEqual(payload["gesture_title"], "Guiño izquierdo")
        self.assertEqual(payload["gesture_action"], "Clic izquierdo")
        self.assertEqual(payload["runtime_state"], "ejecutado")
        self.assertTrue(payload["cursor_active"])
        self.assertEqual(payload["fps_real"], 14.2)
        self.assertEqual(payload["model_version"], 3)
        self.assertAlmostEqual(payload["confidence_value"], 0.91)

    def test_push_status_throttles_only_frame_updates(self):
        controller = self._payload_controller()
        emitted = []

        class Signal:
            def emit(self, payload):
                emitted.append(payload)

        class Gui:
            signals = type("Signals", (), {"status_update": Signal()})()

            def isVisible(self):
                return True

        controller.main_gui_interface = Gui()
        controller._last_status_push = 0.0
        controller._push_status("a", "-", 0.0, throttle=True)
        controller._push_status("b", "-", 0.0, throttle=True)
        controller._push_status("c", "left_blink_intent", 0.9, gesture_executed=True)
        self.assertEqual([payload["status_text"] for payload in emitted], ["a", "c"])


    def _run_smile_sequence(self, profile, steps):
        """Evalúa (segundo, confianza) de "smile" con un reloj simulado."""
        service = InferenceService(profile, window_size=4)
        sample = make_sample(1)
        state = type("State", (), {"face_stable": True, "status_text": "ok"})()
        clock = {"now": 100.0}
        fake_time = type("FakeTime", (), {"monotonic": staticmethod(lambda: clock["now"])})
        results = []
        with mock.patch("app.inference_service.time", fake_time):
            for at, confidence in steps:
                clock["now"] = 100.0 + at
                prediction = GesturePrediction("smile", confidence, False)
                results.append(service.evaluate(sample, state, prediction, AppState.READY, "activo"))
        return results

    def test_inference_service_requires_continuous_duration(self):
        profile = build_default_profile("continuity")
        profile["gesture_confidence"]["smile"] = 0.6
        profile["gesture_duration_ms"]["smile"] = 600
        profile["gesture_cooldown_ms"]["smile"] = 0
        # Dos fotogramas confiables separados por 650 ms de confianza baja: no debe ejecutar.
        steps = [(0.0, 0.7)] + [(0.1 * i, 0.3) for i in range(1, 7)] + [(0.65, 0.7)]
        self.assertFalse(any(result.executed for result in self._run_smile_sequence(profile, steps)))
        # Sostenido de verdad (un fotograma débil aislado se tolera): sí ejecuta.
        steps = [(0.07 * i, 0.3 if i == 4 else 0.7) for i in range(10)]
        self.assertTrue(self._run_smile_sequence(profile, steps)[-1].executed)

    def test_inference_service_clamps_profile_thresholds(self):
        profile = build_default_profile("clamped")
        profile["gesture_confidence"]["smile"] = 0.40
        profile["gesture_duration_ms"]["smile"] = 0
        rejected = self._run_smile_sequence(profile, [(0.0, 0.50)])[0]
        self.assertEqual(rejected.rejection_reason, "confianza baja")
        profile["gesture_confidence"]["mouth_open_hold"] = 0.95
        service = InferenceService(profile, window_size=4)
        state = type("State", (), {"face_stable": True, "status_text": "ok"})()
        prediction = GesturePrediction("mouth_open_hold", 0.88, False)
        result = service.evaluate(make_sample(1), state, prediction, AppState.READY, "activo")
        self.assertEqual(result.runtime_state, "validando gesto")

    def test_recommended_thresholds_stay_in_range(self):
        classifier = GestureClassifier.__new__(GestureClassifier)
        classes = np.array(["neutral", "smile"])
        labels = ["smile", "smile", "neutral", "neutral"]
        predicted = np.array(["smile", "smile", "neutral", "neutral"])
        probabilities = np.array([[0.55, 0.45], [0.6, 0.4], [0.98, 0.02], [0.99, 0.01]])
        thresholds, _ = classifier._recommended_thresholds(labels, predicted, probabilities, classes)
        self.assertEqual(thresholds["smile"], CONFIDENCE_THRESHOLD_MIN)
        self.assertEqual(thresholds["neutral"], CONFIDENCE_THRESHOLD_MAX)

    def test_speech_queue_does_not_block_and_drops_stale_messages(self):
        spoken = []
        speaking = threading.Event()
        release = threading.Event()

        class SlowEngine:
            def say(self, text):
                spoken.append(text)
                speaking.set()

            def runAndWait(self):
                release.wait(2.0)

        queue = SpeechQueue(engine_factory=SlowEngine, max_pending=1)
        try:
            started = time.monotonic()
            queue.say("uno")
            self.assertTrue(speaking.wait(2.0))  # el motor queda ocupado diciendo "uno"
            queue.say("dos")
            queue.say("tres")
            self.assertLess(time.monotonic() - started, 0.5)
            queue.say("cuatro")  # descarta "tres" mientras el motor sigue ocupado
            release.set()
            deadline = time.monotonic() + 2.0
            while len(spoken) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            queue.stop()
        self.assertEqual(spoken, ["uno", "cuatro"])

    def test_fallback_input_device_prefers_real_microphone(self):
        devices = [
            {"id": 7, "name": "Altavoz de PC (Realtek output)", "hostapi": "Windows WDM-KS"},
            {"id": 9, "name": "Mezcla estéreo (Realtek)", "hostapi": "MME"},
            {"id": 8, "name": "Micrófono (Realtek HD Audio Mic input)", "hostapi": "Windows WDM-KS"},
            {"id": 2, "name": "Micrófono (Realtek)", "hostapi": "MME"},
        ]
        self.assertEqual(pick_fallback_input_device(devices)["id"], 2)
        self.assertIsNone(pick_fallback_input_device(devices[:2]))

    def test_confirm_gesture_presses_enter_only_when_ready(self):
        service = ActionService(
            type("Logger", (), {"log": lambda self, *args, **kwargs: None})(),
            on_voice_listener=lambda: None,
            on_toggle_cursor=lambda: None,
            on_recenter=lambda: None,
            on_toggle_pause=lambda: None,
            on_set_cursor=lambda value: None,
            on_quit=lambda: None,
            main_gui_interface=None,
            audio_endpoint=None,
        )
        pressed = []
        with mock.patch.object(action_service_module.pyautogui, "press", lambda key: pressed.append(key)):
            service.execute_gesture_action("confirm", "paused")
            service.execute_gesture_action("confirm", "ready")
        self.assertEqual(pressed, ["enter"])

    def _preflight_snapshot(self, **overrides):
        snapshot = {
            "camera_state": "ok",
            "camera_ok_for_s": 5.0,
            "fps_real": 14.6,
            "face_seen_ago_s": 0.1,
            "mic_monitor_enabled": True,
            "mic_available": True,
            "mic_error": "",
            "mic_device": "Micrófono",
            "voice_model_state": "lista",
            "voice_model_error": "",
            "model_loaded": True,
            "model_version": 3,
            "model_accuracy": 0.98,
            "hotkey_registered": True,
        }
        snapshot.update(overrides)
        return snapshot

    def test_preflight_all_ok(self):
        checks = evaluate_preflight(self._preflight_snapshot(), elapsed_s=2.0, mic_peak_seen=0.3)
        self.assertEqual({check.status for check in checks}, {"ok"})
        self.assertEqual(summarize_preflight(checks), ("Todo listo", "success"))

    def test_preflight_waits_then_reports_problems(self):
        snapshot = self._preflight_snapshot(camera_state="abriendo", face_seen_ago_s=None, voice_model_state="cargando")
        early = {check.key: check.status for check in evaluate_preflight(snapshot, elapsed_s=1.0)}
        self.assertEqual(early["camera"], "pending")
        self.assertEqual(early["face"], "pending")
        self.assertEqual(early["mic"], "pending")
        self.assertEqual(early["voice"], "pending")

        late = {check.key: check for check in evaluate_preflight(snapshot, elapsed_s=12.0, mic_peak_seen=0.0)}
        self.assertEqual(late["camera"].status, "error")
        self.assertEqual(late["mic"].status, "error")
        self.assertIn("Privacidad", late["mic"].detail)

    def test_preflight_flags_missing_model_mic_error_and_hotkey(self):
        snapshot = self._preflight_snapshot(
            model_loaded=False,
            mic_available=False,
            mic_error="Error querying device -1",
            hotkey_registered=False,
            camera_ok_for_s=10.0,
            face_seen_ago_s=None,
        )
        checks = {check.key: check for check in evaluate_preflight(snapshot, elapsed_s=12.0, mic_peak_seen=0.0)}
        self.assertEqual(checks["model"].status, "error")
        self.assertIn("No se encontró un micrófono", checks["mic"].detail)
        self.assertEqual(checks["hotkey"].status, "warning")
        self.assertEqual(checks["face"].status, "warning")
        self.assertEqual(summarize_preflight(list(checks.values()))[1], "danger")

    def test_voice_status_reports_model_loading(self):
        self.assertEqual(voice_status({"voice_state": "en espera", "voice_model_state": "cargando"})[1], "info")
        self.assertEqual(voice_status({"voice_state": "en espera", "voice_model_state": "error"})[1], "danger")
        self.assertEqual(voice_status({"voice_state": "escuchando", "voice_model_state": "lista"})[1], "info")

    def test_voice_listener_waits_for_model_load(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.voice_enabled = True
        controller.whisper = None
        controller.voice_model_state = "cargando"
        controller.voice_thread = None
        controller.state = AppState.READY
        controller._cursor_status_text = lambda: "congelado"
        events = []
        controller._publish_event = events.append
        controller._push_status = lambda *args, **kwargs: None
        controller._start_voice_listener()
        self.assertIsNone(controller.voice_thread)
        self.assertIn("cargando", events[0])

    def test_camera_reconnects_and_restores_state(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.running = True
        controller.settings = {"camera_index": 0, "camera_resolution": "640x480"}
        controller.state = AppState.PAUSED
        controller.cap = None
        controller.camera_state = "ok"
        controller._camera_ok_since = 1.0
        controller._state_before_camera_loss = None
        controller.fps_real = 15.0
        # Misma firma que SessionLogger.log para detectar choques de nombres de argumentos.
        controller.logger = type("Logger", (), {"log": lambda self, event_type, detail, **payload: None})()
        events = []
        controller._publish_event = events.append
        controller._push_status = lambda *args, **kwargs: None

        opened = {"value": False}

        class FakeCapture:
            def __init__(self, index):
                pass

            def set(self, *args):
                return True

            def isOpened(self):
                return opened["value"]

            def release(self):
                pass

        with mock.patch.object(assistive_controls_module.cv2, "VideoCapture", FakeCapture), mock.patch.object(
            assistive_controls_module, "CAMERA_RETRY_INTERVAL_S", 0.0
        ):
            controller._camera_lost("Se perdió la señal de la cámara.")
            self.assertEqual(controller.state, AppState.ERROR)
            self.assertFalse(controller._reconnect_camera())
            self.assertEqual(controller.camera_state, "reconectando")
            opened["value"] = True
            self.assertTrue(controller._reconnect_camera())
        self.assertEqual(controller.camera_state, "ok")
        self.assertEqual(controller.state, AppState.PAUSED)
        self.assertIn("Cámara reconectada.", events)

    def _make_action_service(self, **overrides):
        kwargs = dict(
            on_voice_listener=lambda: None,
            on_toggle_cursor=lambda: None,
            on_recenter=lambda: None,
            on_toggle_pause=lambda: None,
            on_set_cursor=lambda value: None,
            on_quit=lambda: None,
            main_gui_interface=None,
            audio_endpoint=None,
        )
        kwargs.update(overrides)
        return ActionService(type("Logger", (), {"log": lambda self, *args, **kwargs: None})(), **kwargs)

    def test_heuristic_engine_reports_steering_outside_fine_zone(self):
        profile = build_default_profile("steering")
        profile["neutral_nose"] = [100.0, 100.0]
        engine = HeuristicEngine(profile)
        centered = make_sample(1)
        centered.nose_px = (101, 100)
        centered.face_scale_px = 200.0
        self.assertFalse(engine.update(centered).steering)
        turned = make_sample(2)
        turned.nose_px = (140, 100)  # 0.20 del ancho de la cara
        turned.face_scale_px = 200.0
        self.assertTrue(engine.update(turned).steering)

    def test_click_gestures_require_stopped_cursor(self):
        profile = build_default_profile("click_gate")
        service = InferenceService(profile, window_size=4)
        steering = type("State", (), {"face_stable": True, "status_text": "ok", "steering": True})()
        wink = GesturePrediction("left_blink_intent", 0.95, False)
        result = service.evaluate(make_sample(1), steering, wink, AppState.READY, "activo", cursor_active=True)
        self.assertEqual(result.rejection_reason, "cursor en movimiento")
        # Con el cursor congelado (clutch apagado) el guiño sí se valida.
        result = service.evaluate(make_sample(1), steering, wink, AppState.READY, "congelado", cursor_active=False)
        self.assertEqual(result.runtime_state, "validando gesto")
        # Los gestos que no hacen clic no se bloquean al mover el cursor.
        smile = GesturePrediction("smile", 0.95, False)
        result = service.evaluate(make_sample(1), steering, smile, AppState.READY, "activo", cursor_active=True)
        self.assertEqual(result.runtime_state, "validando gesto")

    def test_inference_reports_hold_progress_and_gesture_in_progress(self):
        profile = build_default_profile("progress")
        profile["gesture_confidence"]["smile"] = 0.6
        profile["gesture_duration_ms"]["smile"] = 600
        results = self._run_smile_sequence(profile, [(0.0, 0.9), (0.15, 0.9), (0.30, 0.9), (0.45, 0.9), (0.61, 0.9)])
        progress = [round(result.hold_progress, 2) for result in results]
        self.assertEqual(progress, [0.0, 0.25, 0.5, 0.75, 1.0])
        self.assertTrue(results[-1].executed)

        service = InferenceService(profile, window_size=4)
        state = type("State", (), {"face_stable": True, "status_text": "ok"})()
        service.evaluate(make_sample(1), state, GesturePrediction("smile", 0.9, False), AppState.READY, "activo")
        self.assertEqual(service.gesture_in_progress(), "smile")
        self.assertIsNone(service.gesture_in_progress(time.monotonic() + 1.0))

    def test_probability_smoother_damps_single_frame_spikes(self):
        smoother = ProbabilitySmoother(alpha=0.5)
        smoother.update(np.array([0.9, 0.1]))
        spike = smoother.update(np.array([0.1, 0.9]))
        self.assertAlmostEqual(spike[1], 0.5)
        smoother.reset()
        self.assertAlmostEqual(smoother.update(np.array([0.1, 0.9]))[1], 0.9)

    def test_clicks_land_on_given_position(self):
        service = self._make_action_service()
        clicks = []
        with mock.patch.object(action_service_module.pyautogui, "click", lambda **kwargs: clicks.append(kwargs)):
            service.execute_gesture_action("left_blink_intent", "ready", position=(120, 340))
            service.execute_gesture_action("right_blink_intent", "ready")
        self.assertEqual(clicks, [{"x": 120, "y": 340}, {"button": "right"}])

    def test_precision_voice_commands_toggle_precision(self):
        calls = []
        service = self._make_action_service(on_set_precision=calls.append)
        service.execute_direct_action("precision_on", "ready")
        service.execute_direct_action("precision_off", "ready")
        self.assertEqual(calls, [True, False])
        self.assertEqual(resolve_command_entry("modo preciso", normalize_voice_settings({}))[0]["id"], "precision_on")

    def _cursor_controller(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.state = AppState.READY
        controller.cursor_active = True
        controller.precision_mode = False
        controller._click_anchor = None
        controller.screen_width, controller.screen_height = 1920, 1080
        controller.current_cursor_x, controller.current_cursor_y = 500, 500
        controller._cursor_fx, controller._cursor_fy = 500.0, 500.0
        return controller

    def _drive_cursor(self, controller, dx, frames, fps):
        controller._frame_dt = 1.0 / fps
        state = type("State", (), {"face_present": True, "smoothed_dx": dx, "smoothed_dy": 0.0})()
        moves = []
        with mock.patch.object(assistive_controls_module.pyautogui, "moveTo", lambda x, y, duration=0: moves.append((x, y))):
            for _ in range(frames):
                controller._apply_continuous_control(state)
        return moves

    def test_cursor_speed_is_fps_independent_and_subpixel(self):
        at_15 = self._cursor_controller()
        self._drive_cursor(at_15, 10.0, 15, fps=15)  # 1 s a 15 FPS
        at_10 = self._cursor_controller()
        self._drive_cursor(at_10, 10.0, 10, fps=10)  # 1 s a 10 FPS
        self.assertEqual(at_15.current_cursor_x, 650)
        self.assertEqual(at_10.current_cursor_x, 650)
        # 0,3 px por fotograma: antes se perdía por el redondeo; ahora avanza 3 px en 10 fotogramas.
        slow = self._cursor_controller()
        self._drive_cursor(slow, 0.3, 10, fps=15)
        self.assertEqual(slow.current_cursor_x, 503)

    def test_precision_mode_slows_cursor_and_wink_freezes_it(self):
        precise = self._cursor_controller()
        precise.precision_mode = True
        self._drive_cursor(precise, 10.0, 15, fps=15)
        self.assertAlmostEqual(precise._cursor_fx, 552.5)  # 150 px * 0.35
        frozen = self._cursor_controller()
        frozen._click_anchor = ("left_blink_intent", (500, 500))
        self.assertEqual(self._drive_cursor(frozen, 10.0, 5, fps=15), [])

    def test_click_anchor_follows_wink_in_progress(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller._click_anchor = None
        current = {"gesture": "left_blink_intent"}
        controller.inference = type("Inference", (), {"gesture_in_progress": lambda self: current["gesture"]})()
        positions = iter([(10, 20), (99, 99)])
        with mock.patch.object(assistive_controls_module.pyautogui, "position", lambda: next(positions)):
            self.assertEqual(controller._update_click_anchor(), ("left_blink_intent", (10, 20)))
            self.assertEqual(controller._update_click_anchor(), ("left_blink_intent", (10, 20)))
            current["gesture"] = "smile"
            self.assertIsNone(controller._update_click_anchor())

    def test_head_pose_from_matrix(self):
        self.assertEqual(head_pose_from_matrix(np.eye(4)), (0.0, 0.0, 0.0))
        angle = np.radians(30)
        yaw_matrix = np.eye(4)
        yaw_matrix[:3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
        pitch, yaw, roll = head_pose_from_matrix(yaw_matrix)
        self.assertAlmostEqual(yaw, 30.0, places=4)
        self.assertAlmostEqual(pitch, 0.0, places=4)
        self.assertIsNone(head_pose_from_matrix("no es una matriz"))

    def test_enriched_sample_roundtrip_and_old_datasets(self):
        sample = make_sample(5)
        sample.points_precise = {idx: (x + 0.37, y + 0.81) for idx, (x, y) in sample.points_px.items()}
        sample.blendshapes = {name: index / 100 for index, name in enumerate(BLENDSHAPE_NAMES)}
        sample.head_pose = (1.234, -5.678, 0.5)
        payload = serialize_face_sample(sample)
        self.assertIsInstance(payload["blendshapes"], list)  # sin repetir los 52 nombres
        restored = deserialize_face_sample(json.loads(json.dumps(payload)))
        self.assertAlmostEqual(restored.blendshapes["jawOpen"], sample.blendshapes["jawOpen"], places=3)
        self.assertEqual(restored.head_pose, (1.23, -5.68, 0.5))
        first = next(iter(sample.points_precise))
        self.assertAlmostEqual(restored.points_precise[first][0], sample.points_precise[first][0], places=1)

        old_payload = serialize_face_sample(make_sample(6))
        self.assertNotIn("blendshapes", old_payload)
        old = deserialize_face_sample(old_payload)
        self.assertEqual((old.blendshapes, old.points_precise, old.head_pose), ({}, {}, None))

    def test_calibration_requires_neutral_dynamic_step(self):
        service = CalibrationService(window_size=4)
        self.assertEqual(service.gesture_order[:2], ["neutral", NEUTRAL_DYNAMIC_ID])
        for gesture_id in service.gesture_order:
            if gesture_id == NEUTRAL_DYNAMIC_ID:
                continue
            minimum = service.gesture_meta[gesture_id]["recommended_min_frames"]
            service.samples_by_gesture[gesture_id] = [make_shifted_sample(i, 0.3) for i in range(minimum)]
        ready, blockers, _ = service.can_train()
        self.assertFalse(ready)
        self.assertEqual(blockers, ["Neutral en movimiento"])

    def test_neutral_dynamic_trains_as_neutral_and_measures_false_alarms(self):
        profile_name = f"test_dynamic_{uuid.uuid4().hex[:8]}"
        profile_dir = profile_calibration_dir(profile_name)
        try:
            classifier = GestureClassifier(profile_name, window_size=4)
            samples = {
                "neutral": [make_sample(i) for i in range(10)],
                NEUTRAL_DYNAMIC_ID: [make_shifted_sample(i, 0.05) for i in range(10)],
                "left_blink_intent": [make_shifted_sample(i, 0.4) for i in range(10)],
            }
            classifier.fit(samples)
            summary = classifier.training_summary
            self.assertEqual(summary["classes"], ["left_blink_intent", "neutral"])
            self.assertEqual(summary["windows_per_step"][NEUTRAL_DYNAMIC_ID], 7)
            self.assertEqual(summary["windows_per_class"]["neutral"], 14)
            self.assertIsNotNone(summary["hard_negative_false_alarm_rate"])
            self.assertIn(NEUTRAL_DYNAMIC_ID, classifier.load_active_dataset())

            del samples[NEUTRAL_DYNAMIC_ID]
            classifier.fit(samples)
            self.assertIsNone(classifier.training_summary["hard_negative_false_alarm_rate"])
        finally:
            if profile_dir.exists():
                rmtree(profile_dir)

    @staticmethod
    def _face_points(brow_lift: float = 0.0):
        points = {idx: (float(100 + i * 3), 100.0) for i, idx in enumerate(LEFT_EYE + RIGHT_EYE + MOUTH_OPEN)}
        points.update({1: (150.0, 120.0), 152: (150.0, 220.0), FACE_WIDTH[0]: (50.0, 120.0), FACE_WIDTH[1]: (250.0, 120.0)})
        points.update({159: (110.0, 90.0), 386: (190.0, 90.0)})
        points.update({
            BROW_LEFT[0]: (110.0, 70.0 - brow_lift), BROW_LEFT[1]: (120.0, 72.0 - brow_lift),
            BROW_RIGHT[0]: (190.0, 70.0 - brow_lift), BROW_RIGHT[1]: (180.0, 72.0 - brow_lift),
        })
        return points

    def test_brow_eye_metric_sees_raised_brows(self):
        resting, raised = self._face_points(), self._face_points(brow_lift=12.0)
        # La métrica antigua mide dos puntos de la misma ceja: no cambia al levantarla.
        self.assertEqual(legacy_metrics(resting)["brow_raise_ratio"], legacy_metrics(raised)["brow_raise_ratio"])
        self.assertAlmostEqual(feature_metrics(resting)[0], 20 / 200)
        self.assertAlmostEqual(feature_metrics(raised)[0], 32 / 200)

    def test_feature_metrics_fall_back_when_points_are_missing(self):
        sample = make_sample(1)
        values = feature_metrics(sample.points_px, sample.metrics)
        self.assertEqual(values[1], sample.metrics["left_eye_ratio"])
        self.assertEqual(values[0], sample.metrics["brow_raise_ratio"])

    def test_new_models_normalize_and_roundtrip(self):
        profile_name = f"test_v2_{uuid.uuid4().hex[:8]}"
        profile_dir = profile_calibration_dir(profile_name)
        try:
            classifier = GestureClassifier(profile_name, window_size=4)
            samples = {
                "neutral": [make_sample(i) for i in range(20)],
                "smile": [make_shifted_sample(i, 0.4) for i in range(20)],
            }
            classifier.fit(samples)
            self.assertEqual(classifier.feature_version, 2)
            self.assertEqual(classifier.extractor.metric_source, "int")
            self.assertTrue((classifier.extractor.neutral_std >= 0.05).all())
            # 17 ventanas por paso: 4 de validación y una purga de 4 antes de ellas.
            self.assertEqual(classifier.training_summary["validation_windows"], 8)
            self.assertEqual(classifier.training_summary["train_windows"], 18)

            window = samples["smile"][-4:]
            loaded = GestureClassifier(profile_name, window_size=4)
            self.assertTrue(loaded.load())
            self.assertEqual(loaded.feature_version, 2)
            self.assertEqual(loaded.predict(window).gesture_id, "smile")
            self.assertAlmostEqual(loaded.predict(window).confidence, classifier.predict(window).confidence, places=5)
        finally:
            if profile_dir.exists():
                rmtree(profile_dir)

    def test_old_model_files_keep_working_without_normalization(self):
        profile_name = f"test_v1_{uuid.uuid4().hex[:8]}"
        profile_dir = profile_calibration_dir(profile_name)
        try:
            extractor = GestureFeatureExtractor(window_size=4)
            X = [extractor.extract([make_sample(i + j) for i in range(4)]) for j in range(6)]
            X += [extractor.extract([make_shifted_sample(i + j, 0.4) for i in range(4)]) for j in range(6)]
            model = LogisticRegression(C=20.0).fit(np.array(X), ["neutral"] * 6 + ["smile"] * 6)
            joblib.dump({"model": model, "labels": ["neutral", "smile"], "training_summary": {"window_size": 4}}, GestureClassifier(profile_name).model_path)
            loaded = GestureClassifier(profile_name, window_size=4)
            self.assertTrue(loaded.load())
            self.assertEqual(loaded.feature_version, 1)
            self.assertIsNone(loaded.extractor.neutral_mean)
            self.assertEqual(loaded.predict([make_shifted_sample(i, 0.4) for i in range(4)]).gesture_id, "smile")
        finally:
            GestureClassifier(profile_name).model_path.unlink(missing_ok=True)
            if profile_dir.exists():
                rmtree(profile_dir)

    def test_preflight_warns_about_old_model(self):
        checks = {c.key: c for c in evaluate_preflight(self._preflight_snapshot(model_feature_version=1), 2.0, 0.3)}
        self.assertEqual(checks["model"].status, "warning")
        self.assertIn("Actualizar modelo", checks["model"].detail)

    def test_confirm_gesture_is_defined_as_raised_brows(self):
        confirm = next(gesture for gesture in GESTURE_CATALOG if gesture["id"] == "confirm")
        self.assertIn("cejas", confirm["training_hint"].lower())
        self.assertEqual(confirm["short_action"], "Enter")

    def test_usage_recorder_roundtrip_and_markers(self):
        directory = Path(profile_calibration_dir(f"test_usage_{uuid.uuid4().hex[:8]}"))
        try:
            recorder = UsageRecorder("Perfil Prueba", {"model_version": 3}, directory=directory)
            path = recorder.start()
            recorder.record_frame(make_sample(1), gesture="smile", confidence=0.9, executed=True)
            recorder.record_frame(None, gesture=None, confidence=None, executed=False)
            recorder.mark("error")
            summary = recorder.stop()
            self.assertEqual(summary["frames"], 2)
            self.assertEqual(summary["markers"], {"error": 1})
            self.assertEqual(summary["error"], "")
            recorder.record_frame(make_sample(2), gesture="smile")  # tras detener se ignora
            self.assertEqual(recorder.frames, 2)

            rows = list(read_usage_recording(path))
            self.assertEqual([row["type"] for row in rows], ["header", "frame", "frame", "marker", "footer"])
            self.assertEqual(rows[0]["model_version"], 3)
            self.assertEqual(rows[1]["sample"].metrics, make_sample(1).metrics)
            self.assertTrue(rows[1]["executed"])
            self.assertIsNone(rows[2]["sample"])
            self.assertEqual(rows[3]["kind"], "error")
        finally:
            if directory.exists():
                rmtree(directory)

    def test_usage_recorder_drops_frames_instead_of_blocking(self):
        directory = Path(profile_calibration_dir(f"test_usage_{uuid.uuid4().hex[:8]}"))
        try:
            recorder = UsageRecorder("P", directory=directory)
            recorder.active = True  # sin hilo escritor: la cola se llena
            recorder._queue = __import__("queue").Queue(maxsize=2)
            started = time.monotonic()
            for ts in range(5):
                recorder.record_frame(make_sample(ts))
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual((recorder.frames, recorder.dropped), (2, 3))
        finally:
            if directory.exists():
                rmtree(directory)

    def test_controller_records_usage_decisions(self):
        controller = AssistiveController.__new__(AssistiveController)
        controller.state = AppState.READY
        controller.cursor_active = True
        controller.precision_mode = False
        frames = []
        controller.usage_recorder = type("Rec", (), {"record_frame": lambda self, sample, **d: frames.append((sample, d))})()
        control = type("C", (), {"steering": True, "face_stable": False})()
        result = type("R", (), {"runtime_state": "rechazado", "rejection_reason": "confianza baja", "executed": False})()
        sample = make_sample(1)
        controller._record_usage(sample, control, GesturePrediction("smile", 0.41234567, False), result)
        controller._record_usage(None, control, None, result)
        self.assertIs(frames[0][0], sample)
        self.assertEqual(frames[0][1]["confidence"], 0.4123)
        self.assertEqual(frames[0][1]["rejection"], "confianza baja")
        self.assertTrue(frames[0][1]["steering"])
        self.assertIsNone(frames[1][1]["gesture"])
        controller.usage_recorder = None
        controller._record_usage(sample, control, None, result)  # sin grabación no hace nada
        self.assertEqual(len(frames), 2)

    @staticmethod
    def _metric_rows(brow=0.12, left=0.26, mouth=0.0, right=0.28, smile=0.6, count=5):
        return np.array([[brow, left, mouth, right, smile]] * count, dtype=np.float32)

    def test_signal_limits_choose_the_closing_eye(self):
        raw = {
            "neutral": self._metric_rows(),
            # El guiño derecho cierra la métrica del ojo izquierdo de la imagen.
            "right_blink_intent": self._metric_rows(left=0.14),
            "left_blink_intent": self._metric_rows(right=0.27),  # casi sin cierre: no se verifica
        }
        limits = compute_signal_limits(raw, {})
        self.assertEqual(limits["right_blink_intent"], {"metric": "left_eye_ratio", "max": 0.2})
        self.assertNotIn("left_blink_intent", limits)
        self.assertEqual(compute_signal_limits({"smile": self._metric_rows()}, {}), {})

    def test_signal_limits_for_smile_need_blendshapes(self):
        def with_smile(value):
            sample = make_sample(1)
            sample.blendshapes = {"mouthSmileLeft": value, "mouthSmileRight": value}
            return sample

        raw = {"neutral": self._metric_rows()}
        samples = {"neutral": [with_smile(0.04)] * 3, "smile": [with_smile(0.86)] * 3}
        self.assertEqual(compute_signal_limits(raw, samples)["smile"], {"blendshape": "smile", "min": 0.45})
        self.assertNotIn("smile", compute_signal_limits(raw, {"neutral": [make_sample(1)], "smile": [make_sample(2)]}))

    def test_verify_signal_confirms_or_rejects(self):
        classifier = GestureClassifier(f"test_signal_{uuid.uuid4().hex[:8]}")
        rmtree(classifier.profile_dir, ignore_errors=True)
        classifier.signal_limits = {
            "right_blink_intent": {"metric": "left_eye_ratio", "max": 0.2},
            "smile": {"blendshape": "smile", "min": 0.45},
        }
        open_eye, closed_eye = make_sample(1), make_sample(1)
        open_eye.metrics["left_eye_ratio"], closed_eye.metrics["left_eye_ratio"] = 0.25, 0.1
        self.assertFalse(classifier.verify_signal("right_blink_intent", open_eye))
        self.assertTrue(classifier.verify_signal("right_blink_intent", closed_eye))
        self.assertIsNone(classifier.verify_signal("confirm", open_eye))
        self.assertIsNone(classifier.verify_signal("right_blink_intent", None))
        self.assertIsNone(classifier.verify_signal("smile", open_eye))  # sin blendshapes no se puede verificar
        open_eye.blendshapes = {"mouthSmileLeft": 0.05, "mouthSmileRight": 0.05}
        self.assertFalse(classifier.verify_signal("smile", open_eye))

    def test_signal_check_rejects_and_lets_verified_clicks_pass_while_steering(self):
        profile = build_default_profile("signal_gate")
        steering = type("State", (), {"face_stable": True, "status_text": "ok", "steering": True})()
        wink = lambda: GesturePrediction("left_blink_intent", 0.95, False)
        evaluate = lambda service, **kw: service.evaluate(make_sample(1), steering, wink(), AppState.READY, "activo", cursor_active=True, **kw)
        result = evaluate(InferenceService(profile, 4), signal_verified=False)
        self.assertEqual(result.rejection_reason, "sin señal del gesto")
        self.assertEqual(evaluate(InferenceService(profile, 4)).rejection_reason, "cursor en movimiento")
        self.assertEqual(evaluate(InferenceService(profile, 4), signal_verified=True).runtime_state, "validando gesto")

    def test_preflight_warns_when_model_has_no_signal_check(self):
        checks = {c.key: c for c in evaluate_preflight(self._preflight_snapshot(model_signal_check=False), 2.0, 0.3)}
        self.assertEqual(checks["model"].status, "warning")

    def test_fit_saves_signal_limits_and_old_models_have_none(self):
        profile_name = f"test_limits_{uuid.uuid4().hex[:8]}"
        profile_dir = profile_calibration_dir(profile_name)
        try:
            classifier = GestureClassifier(profile_name, window_size=4)
            self.assertIsNone(classifier.signal_limits)
            samples = {
                "neutral": [make_sample(i) for i in range(10)],
                "right_blink_intent": [make_sample(i) for i in range(10)],
            }
            for sample in samples["right_blink_intent"]:
                sample.metrics["left_eye_ratio"] = 0.05
            classifier.fit(samples)
            self.assertEqual(classifier.signal_limits["right_blink_intent"]["metric"], "left_eye_ratio")
            loaded = GestureClassifier(profile_name, window_size=4)
            loaded.load()
            self.assertEqual(loaded.signal_limits, classifier.signal_limits)
        finally:
            if profile_dir.exists():
                rmtree(profile_dir)

if __name__ == "__main__":
    unittest.main()

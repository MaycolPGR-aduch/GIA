import unittest
import uuid
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
    GestureFeatureExtractor,
    deserialize_face_sample,
    serialize_face_sample,
)
from app.heuristic_engine import HeuristicEngine
from app.inference_service import InferenceService
from app.models import AppState, FaceSample, GesturePrediction
from app.voice_router import find_phrase_conflicts, normalize_voice_settings, resolve_command, resolve_command_entry


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
        controller.hablar = lambda text: None
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


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from app.config_store import build_default_profile, load_settings
from app.gesture_ml import GestureFeatureExtractor
from app.models import FaceSample
from app.voice_router import resolve_command


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

    def test_voice_router(self):
        command, score = resolve_command("pausar sistema")
        self.assertEqual(command, "pause")
        self.assertGreaterEqual(score, 0.9)

    def test_feature_extractor(self):
        extractor = GestureFeatureExtractor(window_size=4)
        vector = extractor.extract([make_sample(i) for i in range(4)])
        self.assertTrue(vector.size > 0)
        self.assertEqual(vector.ndim, 1)


if __name__ == "__main__":
    unittest.main()

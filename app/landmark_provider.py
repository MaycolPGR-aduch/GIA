from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np

from .config_store import FACE_LANDMARKER_MODEL_PATH
from .models import FaceSample


SELECTED_LANDMARKS = [
    33,
    133,
    159,
    145,
    362,
    263,
    386,
    374,
    61,
    291,
    13,
    14,
    78,
    308,
    55,
    285,
    70,
    300,
    1,
    152,
]

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_OPEN = [13, 14, 78, 308]
BROW_LEFT = [105, 66]
BROW_RIGHT = [334, 296]
FACE_WIDTH = [234, 454]
NOSE_IDX = 1


@dataclass(slots=True)
class DetectionSnapshot:
    frame_rgb: np.ndarray
    face_sample: FaceSample | None


class LandmarkProvider:
    def __init__(self, fps: int = 15):
        self.fps = fps
        self.timestamp_ms = 0
        base_options = mp.tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL_PATH))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    def process(self, frame_bgr: np.ndarray) -> DetectionSnapshot:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self.timestamp_ms += max(1, int(1000 / max(self.fps, 1)))
        result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        if not result.face_landmarks:
            return DetectionSnapshot(frame_rgb=frame_rgb, face_sample=None)

        landmarks = result.face_landmarks[0]
        height, width = frame_bgr.shape[:2]
        points_px = {
            idx: (int(landmarks[idx].x * width), int(landmarks[idx].y * height))
            for idx in set(SELECTED_LANDMARKS + LEFT_EYE + RIGHT_EYE + MOUTH_OPEN + BROW_LEFT + BROW_RIGHT + FACE_WIDTH)
        }

        center = points_px[NOSE_IDX]
        face_width_px = max(
            float(abs(points_px[FACE_WIDTH[1]][0] - points_px[FACE_WIDTH[0]][0])),
            1.0,
        )

        normalized_landmarks = np.array(
            [
                [
                    (points_px[idx][0] - center[0]) / face_width_px,
                    (points_px[idx][1] - center[1]) / face_width_px,
                ]
                for idx in SELECTED_LANDMARKS
            ],
            dtype=np.float32,
        )

        metrics = {
            "left_eye_ratio": self._aspect_ratio(LEFT_EYE, points_px),
            "right_eye_ratio": self._aspect_ratio(RIGHT_EYE, points_px),
            "mouth_open_ratio": self._mouth_open_ratio(points_px),
            "smile_ratio": self._smile_ratio(points_px),
            "brow_raise_ratio": self._brow_raise_ratio(points_px),
        }

        sample = FaceSample(
            timestamp_ms=self.timestamp_ms,
            normalized_landmarks=normalized_landmarks,
            metrics=metrics,
            points_px=points_px,
            nose_px=points_px[NOSE_IDX],
            face_scale_px=face_width_px,
            face_center_px=center,
        )
        return DetectionSnapshot(frame_rgb=frame_rgb, face_sample=sample)

    def _aspect_ratio(self, indices: Iterable[int], points_px: dict[int, tuple[int, int]]) -> float:
        p1, p2, p3, p4, p5, p6 = [points_px[idx] for idx in indices]
        vertical_1 = abs(p2[1] - p6[1])
        vertical_2 = abs(p3[1] - p5[1])
        horizontal = max(abs(p1[0] - p4[0]), 1)
        return float((vertical_1 + vertical_2) / (2.0 * horizontal))

    def _mouth_open_ratio(self, points_px: dict[int, tuple[int, int]]) -> float:
        upper = points_px[MOUTH_OPEN[0]]
        lower = points_px[MOUTH_OPEN[1]]
        left = points_px[MOUTH_OPEN[2]]
        right = points_px[MOUTH_OPEN[3]]
        vertical = abs(upper[1] - lower[1])
        horizontal = max(abs(left[0] - right[0]), 1)
        return float(vertical / horizontal)

    def _smile_ratio(self, points_px: dict[int, tuple[int, int]]) -> float:
        left = points_px[MOUTH_OPEN[2]]
        right = points_px[MOUTH_OPEN[3]]
        chin = points_px[152]
        nose = points_px[NOSE_IDX]
        width = abs(left[0] - right[0])
        height = max(abs(chin[1] - nose[1]), 1)
        return float(width / height)

    def _brow_raise_ratio(self, points_px: dict[int, tuple[int, int]]) -> float:
        brow_dist_left = abs(points_px[BROW_LEFT[0]][1] - points_px[BROW_LEFT[1]][1])
        brow_dist_right = abs(points_px[BROW_RIGHT[0]][1] - points_px[BROW_RIGHT[1]][1])
        face_width = max(abs(points_px[FACE_WIDTH[1]][0] - points_px[FACE_WIDTH[0]][0]), 1)
        return float((brow_dist_left + brow_dist_right) / (2.0 * face_width))

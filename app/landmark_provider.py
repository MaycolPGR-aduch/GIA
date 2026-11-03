from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np

from .config_store import FACE_LANDMARKER_MODEL_PATH
from .face_metrics import BROW_LEFT, BROW_RIGHT, FACE_WIDTH, LEFT_EYE, MOUTH_OPEN, NOSE_IDX, RIGHT_EYE, legacy_metrics
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



def head_pose_from_matrix(matrix) -> tuple[float, float, float] | None:
    """(pitch, yaw, roll) en grados a partir de la matriz de transformación facial 4x4."""
    try:
        rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    except Exception:
        return None
    if rotation.shape != (3, 3):
        return None
    pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
    yaw = math.degrees(math.asin(max(-1.0, min(1.0, -rotation[2, 0]))))
    roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
    return (pitch, yaw, roll)


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
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
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
        indices = set(SELECTED_LANDMARKS + LEFT_EYE + RIGHT_EYE + MOUTH_OPEN + BROW_LEFT + BROW_RIGHT + FACE_WIDTH)
        points_precise = {idx: (landmarks[idx].x * width, landmarks[idx].y * height) for idx in indices}
        # Las métricas y el dibujo siguen usando píxeles enteros para no cambiar el
        # comportamiento de los modelos ya entrenados.
        points_px = {idx: (int(x), int(y)) for idx, (x, y) in points_precise.items()}
        blendshapes = {}
        if getattr(result, "face_blendshapes", None):
            blendshapes = {category.category_name: float(category.score) for category in result.face_blendshapes[0]}
        head_pose = None
        if getattr(result, "facial_transformation_matrixes", None):
            head_pose = head_pose_from_matrix(result.facial_transformation_matrixes[0])

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

        metrics = legacy_metrics(points_px)

        sample = FaceSample(
            timestamp_ms=self.timestamp_ms,
            normalized_landmarks=normalized_landmarks,
            metrics=metrics,
            points_px=points_px,
            nose_px=points_px[NOSE_IDX],
            face_scale_px=face_width_px,
            face_center_px=center,
            points_precise=points_precise,
            blendshapes=blendshapes,
            head_pose=head_pose,
        )
        return DetectionSnapshot(frame_rgb=frame_rgb, face_sample=sample)

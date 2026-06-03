from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class AppState(str, Enum):
    LAUNCHER = "launcher"
    CALIBRATING = "calibrating"
    TESTING = "testing"
    READY = "ready"
    PAUSED = "paused"
    LISTENING = "listening"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(slots=True)
class FaceSample:
    timestamp_ms: int
    normalized_landmarks: np.ndarray
    metrics: dict[str, float]
    points_px: dict[int, tuple[int, int]]
    nose_px: tuple[int, int]
    face_scale_px: float
    face_center_px: tuple[int, int]


@dataclass(slots=True)
class ContinuousControlState:
    face_present: bool
    face_stable: bool
    cursor_dx: float = 0.0
    cursor_dy: float = 0.0
    smoothed_dx: float = 0.0
    smoothed_dy: float = 0.0
    status_text: str = "Sin rostro"
    debug_metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class GesturePrediction:
    gesture_id: str
    confidence: float
    accepted: bool
    reason: str = ""


@dataclass(slots=True)
class GestureTrainingStatus:
    gesture_id: str
    title: str
    action: str
    captured_frames: int
    valid_frames: int
    invalid_frames: int
    training_windows: int
    recommended_min_frames: int
    recommended_min_windows: int
    readiness: str
    help_message: str
    success_message: str
    blocker: str = ""


@dataclass(slots=True)
class GestureInferenceResult:
    prediction: GesturePrediction | None
    status_text: str
    face_label: str
    mode_label: str
    cursor_label: str
    runtime_state: str
    rejection_reason: str = ""
    executed: bool = False
    should_log: bool = False
    diagnostic_hint: str = ""


@dataclass(slots=True)
class SessionEvent:
    timestamp: str
    event_type: str
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)

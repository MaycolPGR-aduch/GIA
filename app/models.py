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
class SessionEvent:
    timestamp: str
    event_type: str
    detail: str
    payload: dict[str, Any] = field(default_factory=dict)

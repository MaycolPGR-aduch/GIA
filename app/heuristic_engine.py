from __future__ import annotations

import math

from .models import ContinuousControlState, FaceSample


class HeuristicEngine:
    def __init__(self, profile: dict):
        self.profile = profile
        self.neutral_nose = tuple(profile.get("neutral_nose", [0.0, 0.0]))
        self.face_stability_threshold = 0.035
        self.sensitivity = float(profile.get("cursor_sensitivity", 1.35))
        self.dead_zone_px = float(profile.get("dead_zone_px", 8))
        self.smoothing_factor = float(profile.get("smoothing_factor", 0.62))
        self.max_cursor_speed_px = float(profile.get("max_cursor_speed_px", 36))
        self.prev_dx = 0.0
        self.prev_dy = 0.0
        self.prev_nose = None

    def update_profile(self, profile: dict) -> None:
        self.__init__(profile)

    def recenter(self, sample: FaceSample | None) -> None:
        if sample is None:
            return
        self.neutral_nose = sample.nose_px
        self.prev_dx = 0.0
        self.prev_dy = 0.0

    def update(self, sample: FaceSample | None) -> ContinuousControlState:
        if sample is None:
            self.prev_nose = None
            return ContinuousControlState(
                face_present=False,
                face_stable=False,
                status_text="Sin rostro detectado",
            )

        if self.neutral_nose == (0.0, 0.0):
            self.recenter(sample)

        dx = (sample.nose_px[0] - self.neutral_nose[0]) * self.sensitivity
        dy = (sample.nose_px[1] - self.neutral_nose[1]) * self.sensitivity

        if abs(dx) < self.dead_zone_px:
            dx = 0.0
        if abs(dy) < self.dead_zone_px:
            dy = 0.0

        smoothed_dx = self.prev_dx * self.smoothing_factor + dx * (1 - self.smoothing_factor)
        smoothed_dy = self.prev_dy * self.smoothing_factor + dy * (1 - self.smoothing_factor)
        smoothed_dx = max(-self.max_cursor_speed_px, min(self.max_cursor_speed_px, smoothed_dx))
        smoothed_dy = max(-self.max_cursor_speed_px, min(self.max_cursor_speed_px, smoothed_dy))
        self.prev_dx = smoothed_dx
        self.prev_dy = smoothed_dy

        stable = True
        if self.prev_nose is not None:
            delta_x = abs(sample.nose_px[0] - self.prev_nose[0]) / max(sample.face_scale_px, 1.0)
            delta_y = abs(sample.nose_px[1] - self.prev_nose[1]) / max(sample.face_scale_px, 1.0)
            stable = math.hypot(delta_x, delta_y) <= self.face_stability_threshold or (abs(dx) + abs(dy) > 0)
        self.prev_nose = sample.nose_px

        status = "Rostro estable" if stable else "Ajustando estabilidad facial"
        return ContinuousControlState(
            face_present=True,
            face_stable=stable,
            cursor_dx=dx,
            cursor_dy=dy,
            smoothed_dx=smoothed_dx,
            smoothed_dy=smoothed_dy,
            status_text=status,
            debug_metrics=sample.metrics,
        )

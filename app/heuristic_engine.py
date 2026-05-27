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
        self.invert_x = bool(profile.get("invert_x", True))
        self.invert_y = bool(profile.get("invert_y", False))
        self.prev_speed_x = 0.0
        self.prev_speed_y = 0.0
        self.prev_nose = None

    def update_profile(self, profile: dict) -> None:
        self.__init__(profile)

    def recenter(self, sample: FaceSample | None) -> None:
        if sample is None:
            return
        self.neutral_nose = sample.nose_px
        self.prev_speed_x = 0.0
        self.prev_speed_y = 0.0

    def update(self, sample: FaceSample | None) -> ContinuousControlState:
        if sample is None:
            self.prev_nose = None
            self.prev_speed_x = 0.0
            self.prev_speed_y = 0.0
            return ContinuousControlState(
                face_present=False,
                face_stable=False,
                status_text="Sin rostro detectado",
            )

        if self.neutral_nose == (0.0, 0.0):
            self.recenter(sample)

        norm_x = (sample.nose_px[0] - self.neutral_nose[0]) / max(sample.face_scale_px, 1.0)
        norm_y = (sample.nose_px[1] - self.neutral_nose[1]) / max(sample.face_scale_px, 1.0)
        if self.invert_x:
            norm_x *= -1.0
        if self.invert_y:
            norm_y *= -1.0

        target_speed_x = self._map_axis_to_speed(norm_x)
        target_speed_y = self._map_axis_to_speed(norm_y)

        smoothed_speed_x = self._smooth_axis(self.prev_speed_x, target_speed_x)
        smoothed_speed_y = self._smooth_axis(self.prev_speed_y, target_speed_y)
        self.prev_speed_x = smoothed_speed_x
        self.prev_speed_y = smoothed_speed_y

        stable = True
        if self.prev_nose is not None:
            delta_x = abs(sample.nose_px[0] - self.prev_nose[0]) / max(sample.face_scale_px, 1.0)
            delta_y = abs(sample.nose_px[1] - self.prev_nose[1]) / max(sample.face_scale_px, 1.0)
            stable = math.hypot(delta_x, delta_y) <= self.face_stability_threshold or (
                abs(target_speed_x) + abs(target_speed_y) > 0
            )
        self.prev_nose = sample.nose_px

        intensity = math.hypot(norm_x, norm_y)
        if intensity <= self._dead_zone_ratio(sample.face_scale_px):
            status = "Rostro estable - zona neutral"
        else:
            status = "Rostro estable - control de cursor listo" if stable else "Ajustando estabilidad facial"

        debug_metrics = dict(sample.metrics)
        debug_metrics.update(
            {
                "control_norm_x": norm_x,
                "control_norm_y": norm_y,
                "cursor_speed_x": smoothed_speed_x,
                "cursor_speed_y": smoothed_speed_y,
            }
        )

        return ContinuousControlState(
            face_present=True,
            face_stable=stable,
            cursor_dx=target_speed_x,
            cursor_dy=target_speed_y,
            smoothed_dx=smoothed_speed_x,
            smoothed_dy=smoothed_speed_y,
            status_text=status,
            debug_metrics=debug_metrics,
        )

    def _dead_zone_ratio(self, face_scale_px: float) -> float:
        return max(self.dead_zone_px / max(face_scale_px, 1.0), 0.045)

    def _map_axis_to_speed(self, axis_value: float) -> float:
        sign = -1.0 if axis_value < 0 else 1.0
        magnitude = abs(axis_value)
        dead_zone = max(self.dead_zone_px / 140.0, 0.045)
        fine_zone = dead_zone + 0.06
        medium_zone = dead_zone + 0.17
        high_zone = dead_zone + 0.32

        if magnitude <= dead_zone:
            return 0.0
        if magnitude <= fine_zone:
            ratio = (magnitude - dead_zone) / max(fine_zone - dead_zone, 1e-6)
            speed_ratio = 0.22 * ratio
        elif magnitude <= medium_zone:
            ratio = (magnitude - fine_zone) / max(medium_zone - fine_zone, 1e-6)
            speed_ratio = 0.22 + (0.55 - 0.22) * ratio
        elif magnitude <= high_zone:
            ratio = (magnitude - medium_zone) / max(high_zone - medium_zone, 1e-6)
            speed_ratio = 0.55 + (1.0 - 0.55) * ratio
        else:
            speed_ratio = 1.0

        return sign * speed_ratio * self.max_cursor_speed_px * self.sensitivity

    def _smooth_axis(self, previous_speed: float, target_speed: float) -> float:
        adaptive_alpha = self.smoothing_factor
        if abs(target_speed) < self.max_cursor_speed_px * 0.2:
            adaptive_alpha = min(0.92, self.smoothing_factor + 0.18)
        elif abs(target_speed) > self.max_cursor_speed_px * 0.7:
            adaptive_alpha = max(0.30, self.smoothing_factor - 0.18)

        smoothed = previous_speed * adaptive_alpha + target_speed * (1 - adaptive_alpha)
        return max(-self.max_cursor_speed_px, min(self.max_cursor_speed_px, smoothed))

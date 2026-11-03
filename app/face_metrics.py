from __future__ import annotations

from typing import Iterable, Mapping

# Índices de MediaPipe Face Mesh usados por las métricas.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_OPEN = [13, 14, 78, 308]
BROW_LEFT = [105, 66]
BROW_RIGHT = [334, 296]
FACE_WIDTH = [234, 454]
NOSE_IDX = 1
CHIN_IDX = 152
LEFT_UPPER_LID = 159
RIGHT_UPPER_LID = 386

# Métricas del modelo actual (feature_version 2), en orden fijo.
FEATURE_METRICS = ("brow_eye_ratio", "left_eye_ratio", "mouth_open_ratio", "right_eye_ratio", "smile_ratio")

Points = Mapping[int, tuple[float, float]]


def _aspect_ratio(indices: Iterable[int], points: Points) -> float:
    p1, p2, p3, p4, p5, p6 = [points[idx] for idx in indices]
    vertical_1 = abs(p2[1] - p6[1])
    vertical_2 = abs(p3[1] - p5[1])
    horizontal = max(abs(p1[0] - p4[0]), 1)
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def _mouth_open_ratio(points: Points) -> float:
    upper, lower, left, right = [points[idx] for idx in MOUTH_OPEN]
    return float(abs(upper[1] - lower[1]) / max(abs(left[0] - right[0]), 1))


def _smile_ratio(points: Points) -> float:
    left, right = points[MOUTH_OPEN[2]], points[MOUTH_OPEN[3]]
    height = max(abs(points[CHIN_IDX][1] - points[NOSE_IDX][1]), 1)
    return float(abs(left[0] - right[0]) / height)


def _face_width(points: Points) -> float:
    return max(abs(points[FACE_WIDTH[1]][0] - points[FACE_WIDTH[0]][0]), 1)


def legacy_metrics(points: Points) -> dict[str, float]:
    """Métricas de los modelos antiguos (feature_version 1); no deben cambiar."""
    brow_left = abs(points[BROW_LEFT[0]][1] - points[BROW_LEFT[1]][1])
    brow_right = abs(points[BROW_RIGHT[0]][1] - points[BROW_RIGHT[1]][1])
    return {
        "left_eye_ratio": _aspect_ratio(LEFT_EYE, points),
        "right_eye_ratio": _aspect_ratio(RIGHT_EYE, points),
        "mouth_open_ratio": _mouth_open_ratio(points),
        "smile_ratio": _smile_ratio(points),
        "brow_raise_ratio": float((brow_left + brow_right) / (2.0 * _face_width(points))),
    }


def _brow_eye_ratio(points: Points) -> float:
    """Distancia ceja-párpado: a diferencia de brow_raise_ratio, sí crece al levantar las cejas."""
    left = abs(points[BROW_LEFT[0]][1] - points[LEFT_UPPER_LID][1])
    right = abs(points[BROW_RIGHT[0]][1] - points[RIGHT_UPPER_LID][1])
    return float((left + right) / (2.0 * _face_width(points)))


def feature_metrics(points: Points, fallback: Mapping[str, float] | None = None) -> list[float]:
    """Vector de métricas en el orden de FEATURE_METRICS.

    Algunos datasets antiguos no guardaron todos los puntos: en ese caso usa las
    métricas ya calculadas de la muestra (``fallback``).
    """
    try:
        return [
            _brow_eye_ratio(points),
            _aspect_ratio(LEFT_EYE, points),
            _mouth_open_ratio(points),
            _aspect_ratio(RIGHT_EYE, points),
            _smile_ratio(points),
        ]
    except KeyError:
        fallback = fallback or {}
        return [
            float(fallback.get("brow_raise_ratio", 0.0)),
            float(fallback.get("left_eye_ratio", 0.0)),
            float(fallback.get("mouth_open_ratio", 0.0)),
            float(fallback.get("right_eye_ratio", 0.0)),
            float(fallback.get("smile_ratio", 0.0)),
        ]

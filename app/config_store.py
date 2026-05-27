from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
PROFILE_DIR = CONFIG_DIR / "user_profiles"
DATA_DIR = BASE_DIR / "data"
CALIBRATION_DIR = DATA_DIR / "calibration"
LOG_DIR = DATA_DIR / "logs"
MODEL_DIR = DATA_DIR / "models"

SETTINGS_PATH = CONFIG_DIR / "settings.json"
FACE_LANDMARKER_MODEL_PATH = MODEL_DIR / "face_landmarker.task"


DEFAULT_SETTINGS = {
    "camera_index": 0,
    "fps": 15,
    "cursor_sensitivity": 1.35,
    "dead_zone_px": 8,
    "smoothing_factor": 0.62,
    "max_cursor_speed_px": 36,
    "face_stability_threshold": 0.035,
    "voice_enabled": True,
    "tts_enabled": False,
    "high_contrast": False,
    "font_scale": 1.0,
    "gesture_window_size": 12,
}

DEFAULT_PROFILE = {
    "name": "default",
    "created_at": None,
    "updated_at": None,
    "launcher_completed": False,
    "neutral_nose": [0.0, 0.0],
    "neutral_face_center": [0.0, 0.0],
    "neutral_face_scale_px": 1.0,
    "cursor_sensitivity": 1.35,
    "dead_zone_px": 8,
    "smoothing_factor": 0.62,
    "max_cursor_speed_px": 36,
    "invert_x": True,
    "invert_y": False,
    "mirror_preview": True,
    "cursor_starts_active": False,
    "voice_enabled": True,
    "tts_enabled": False,
    "gesture_confidence": {
        "neutral": 0.50,
        "left_blink_intent": 0.50,
        "right_blink_intent": 0.50,
        "both_eyes_closed_intent": 0.50,
        "mouth_open_hold": 0.50,
        "smile": 0.50,
        "brows_up": 0.50,
        "confirm": 0.50,
    },
    "gesture_duration_ms": {
        "left_blink_intent": 350,
        "right_blink_intent": 350,
        "both_eyes_closed_intent": 900,
        "mouth_open_hold": 700,
        "smile": 600,
        "brows_up": 700,
        "confirm": 650,
    },
    "gesture_cooldown_ms": {
        "left_blink_intent": 1200,
        "right_blink_intent": 1200,
        "both_eyes_closed_intent": 2200,
        "mouth_open_hold": 1800,
        "smile": 1800,
        "brows_up": 1500,
        "confirm": 1500,
    },
    "calibration": {
        "completed": False,
        "last_calibrated_at": None,
        "samples_per_gesture": {},
        "active_model_version": None,
        "active_model_path": None,
        "active_dataset_path": None,
        "model_versions": [],
    },
}


def ensure_app_dirs() -> None:
    for path in (CONFIG_DIR, PROFILE_DIR, DATA_DIR, CALIBRATION_DIR, LOG_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_settings() -> dict:
    ensure_app_dirs()
    merged = deepcopy(DEFAULT_SETTINGS)
    merged.update(_read_json(SETTINGS_PATH))
    save_settings(merged)
    return merged


def save_settings(settings: dict) -> None:
    ensure_app_dirs()
    _write_json(SETTINGS_PATH, settings)


def list_profiles() -> list[str]:
    ensure_app_dirs()
    profiles = sorted(path.stem for path in PROFILE_DIR.glob("*.json"))
    if not profiles:
        save_profile(deepcopy(DEFAULT_PROFILE))
        return ["default"]
    return profiles


def load_profile(profile_name: str) -> dict:
    ensure_app_dirs()
    profile_path = PROFILE_DIR / f"{profile_name}.json"
    profile = deepcopy(DEFAULT_PROFILE)
    profile["name"] = profile_name
    profile.update(_read_json(profile_path))
    profile["calibration"] = {**deepcopy(DEFAULT_PROFILE["calibration"]), **profile.get("calibration", {})}
    profile["gesture_confidence"] = {
        key: min(0.50, float(value))
        for key, value in {**deepcopy(DEFAULT_PROFILE["gesture_confidence"]), **profile.get("gesture_confidence", {})}.items()
    }
    if not profile.get("created_at"):
        profile["created_at"] = datetime.utcnow().isoformat()
    profile["updated_at"] = datetime.utcnow().isoformat()
    save_profile(profile)
    return profile


def save_profile(profile: dict) -> None:
    ensure_app_dirs()
    payload = deepcopy(DEFAULT_PROFILE)
    payload["name"] = profile.get("name", "default")
    payload.update(profile)
    payload["calibration"] = {**deepcopy(DEFAULT_PROFILE["calibration"]), **payload.get("calibration", {})}
    payload["updated_at"] = datetime.utcnow().isoformat()
    if not payload.get("created_at"):
        payload["created_at"] = payload["updated_at"]
    _write_json(PROFILE_DIR / f"{payload['name']}.json", payload)


def build_default_profile(profile_name: str) -> dict:
    profile = deepcopy(DEFAULT_PROFILE)
    profile["name"] = profile_name
    profile["created_at"] = datetime.utcnow().isoformat()
    profile["updated_at"] = profile["created_at"]
    return profile


def profile_calibration_dir(profile_name: str) -> Path:
    path = CALIBRATION_DIR / profile_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_registry_path(profile_name: str) -> Path:
    return profile_calibration_dir(profile_name) / "model_registry.json"


def versioned_model_path(profile_name: str, version: int) -> Path:
    return profile_calibration_dir(profile_name) / f"{profile_name}_gesture_model_v{version}.pkl"


def versioned_dataset_path(profile_name: str, version: int) -> Path:
    return profile_calibration_dir(profile_name) / f"{profile_name}_landmark_dataset_v{version}.json"


def active_dataset_summary_path(profile_name: str) -> Path:
    return profile_calibration_dir(profile_name) / f"{profile_name}_dataset_summary.json"


def legacy_model_path(profile_name: str) -> Path:
    return CALIBRATION_DIR / f"{profile_name}_gesture_model.pkl"


def legacy_dataset_summary_path(profile_name: str) -> Path:
    return CALIBRATION_DIR / f"{profile_name}_samples.json"


def session_log_path(profile_name: str) -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return LOG_DIR / f"{profile_name}_{stamp}.jsonl"

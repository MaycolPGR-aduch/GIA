from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write


def list_input_devices() -> list[dict]:
    try:
        hostapis = sd.query_hostapis()
        default_input = None
        try:
            default_input = sd.default.device[0]
        except Exception:
            default_input = None

        devices = []
        for index, raw_device in enumerate(sd.query_devices()):
            max_input_channels = int(raw_device.get("max_input_channels", 0))
            if max_input_channels < 1:
                continue
            hostapi_name = "Unknown"
            try:
                hostapi_name = hostapis[int(raw_device["hostapi"])]["name"]
            except Exception:
                pass
            is_default = index == default_input
            label = f"{index}: {raw_device['name']} [{hostapi_name}]"
            if is_default:
                label += " (predeterminado)"
            devices.append(
                {
                    "id": index,
                    "name": str(raw_device["name"]),
                    "hostapi": hostapi_name,
                    "label": label,
                    "is_default": is_default,
                    "max_input_channels": max_input_channels,
                }
            )
        return devices
    except Exception:
        return []


def resolve_input_device(device_setting):
    if device_setting in (None, "", "default"):
        return None
    try:
        return int(device_setting)
    except (TypeError, ValueError):
        return device_setting


def describe_input_device(device_setting) -> str:
    resolved = resolve_input_device(device_setting)
    devices = list_input_devices()
    if resolved is None:
        for device in devices:
            if device["is_default"]:
                return device["label"]
        return "Predeterminado del sistema"
    for device in devices:
        if device["id"] == resolved or device["name"] == resolved:
            return device["label"]
    return f"Dispositivo configurado: {resolved}"


def analyze_audio(audio: np.ndarray, sample_rate: int, *, device_label: str) -> dict:
    mono = np.asarray(audio, dtype=np.float32).reshape(-1)
    abs_audio = np.abs(mono)
    peak = float(abs_audio.max()) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float64))))) if mono.size else 0.0
    activity_threshold = max(peak * 0.08, 0.01)
    active_ratio = float(np.mean(abs_audio >= activity_threshold)) if mono.size else 0.0
    clip_ratio = float(np.mean(abs_audio >= 0.98)) if mono.size else 0.0
    return {
        "device_label": device_label,
        "sample_rate": int(sample_rate),
        "duration_s": round(mono.size / float(sample_rate), 3) if sample_rate else 0.0,
        "rms": round(rms, 4),
        "peak": round(peak, 4),
        "active_ratio": round(active_ratio, 4),
        "clip_ratio": round(clip_ratio, 4),
    }


def capture_voice_clip(duration_s: float, sample_rate: int, *, device=None) -> tuple[np.ndarray, dict]:
    sample_count = max(1, int(duration_s * sample_rate))
    resolved_device = resolve_input_device(device)
    recording = sd.rec(
        sample_count,
        samplerate=int(sample_rate),
        channels=1,
        dtype="float32",
        device=resolved_device,
    )
    sd.wait()
    mono = np.asarray(recording, dtype=np.float32).reshape(-1)
    diagnostics = analyze_audio(
        mono,
        int(sample_rate),
        device_label=describe_input_device(resolved_device),
    )
    return mono, diagnostics


def prepare_audio_for_asr(audio: np.ndarray) -> np.ndarray:
    prepared = np.asarray(audio, dtype=np.float32).copy().reshape(-1)
    peak = float(np.max(np.abs(prepared))) if prepared.size else 0.0
    if peak > 1e-6:
        scale = min(8.0, 0.85 / peak)
        prepared *= scale
    return np.clip(prepared, -1.0, 1.0)


def save_audio_clip(path: Path | str, audio: np.ndarray, sample_rate: int) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prepared = prepare_audio_for_asr(audio)
    write(destination, int(sample_rate), (prepared * 32767).astype(np.int16))
    return destination


class AudioLevelMonitor:
    def __init__(self, sample_rate: int, *, device=None, block_duration_s: float = 0.12):
        self.sample_rate = int(sample_rate)
        self.device_setting = device
        self.device = resolve_input_device(device)
        self.device_label = describe_input_device(self.device)
        self.block_duration_s = max(0.05, float(block_duration_s))
        self._lock = threading.Lock()
        self._stream = None
        self._running = False
        self._suspended = False
        self._last_update = 0.0
        self._last_error = ""
        self._rms = 0.0
        self._peak = 0.0
        self._active_ratio = 0.0

    def _callback(self, indata, frames, time_info, status):
        if self._suspended:
            return
        if status:
            with self._lock:
                self._last_error = str(status)
        mono = np.asarray(indata, dtype=np.float32).reshape(-1)
        diag = analyze_audio(mono, self.sample_rate, device_label=self.device_label)
        now = time.monotonic()
        with self._lock:
            self._rms = float(diag["rms"])
            self._peak = float(diag["peak"])
            self._active_ratio = float(diag["active_ratio"])
            self._last_update = now

    def start(self) -> bool:
        if self._running:
            return True
        try:
            blocksize = max(256, int(self.sample_rate * self.block_duration_s))
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                device=self.device,
                blocksize=blocksize,
                callback=self._callback,
            )
            self._stream.start()
            self._running = True
            self._last_error = ""
            return True
        except Exception as exc:
            with self._lock:
                self._last_error = str(exc)
            self._stream = None
            self._running = False
            return False

    def stop(self) -> None:
        stream = self._stream
        self._stream = None
        self._running = False
        self._suspended = False
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def suspend(self) -> None:
        self._suspended = True

    def resume(self) -> None:
        self._suspended = False

    def snapshot(self) -> dict:
        with self._lock:
            age_s = time.monotonic() - self._last_update if self._last_update else None
            return {
                "device_label": self.device_label,
                "available": self._running,
                "suspended": self._suspended,
                "rms": round(self._rms, 4),
                "peak": round(self._peak, 4),
                "active_ratio": round(self._active_ratio, 4),
                "level_percent": int(max(0.0, min(self._rms * 260.0, 100.0))),
                "peak_percent": int(max(0.0, min(self._peak * 100.0, 100.0))),
                "age_s": round(age_s, 3) if age_s is not None else None,
                "error": self._last_error,
            }

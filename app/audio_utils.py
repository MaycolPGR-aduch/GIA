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


def _system_default_input() -> int | None:
    try:
        index = sd.default.device[0]
    except Exception:
        return None
    return int(index) if index is not None and int(index) >= 0 else None


_HOSTAPI_PREFERENCE = ("Windows WASAPI", "MME", "Windows DirectSound", "Windows WDM-KS")
_NOT_A_MICROPHONE = ("mezcla", "stereo mix", "altavoz", "speaker", "output", "loopback")


def pick_fallback_input_device(devices: list[dict]) -> dict | None:
    """Elige el micrófono más probable cuando Windows no tiene uno predeterminado."""
    candidates = [
        device for device in devices
        if not any(word in device["name"].lower() for word in _NOT_A_MICROPHONE)
    ]
    if not candidates:
        return None

    def rank(device: dict) -> tuple[int, int, int]:
        name = device["name"].lower()
        looks_like_mic = 0 if ("mic" in name or "micr" in name) else 1
        hostapi = device.get("hostapi", "")
        api_rank = _HOSTAPI_PREFERENCE.index(hostapi) if hostapi in _HOSTAPI_PREFERENCE else len(_HOSTAPI_PREFERENCE)
        return looks_like_mic, api_rank, int(device["id"])

    return min(candidates, key=rank)


def _device_native_rate(device) -> int | None:
    try:
        info = sd.query_devices(device, "input")
        return int(info["default_samplerate"])
    except Exception:
        return None


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    factor = gcd(int(source_rate), int(target_rate))
    return resample_poly(audio, int(target_rate) // factor, int(source_rate) // factor).astype(np.float32)


def resolve_input_device(device_setting):
    """Traduce el ajuste a un dispositivo de sounddevice.

    Con "predeterminado" devuelve None (PortAudio usa el del sistema). Si Windows no
    tiene micrófono predeterminado (índice -1), usa el primer dispositivo de entrada.
    """
    if device_setting in (None, "", "default"):
        if _system_default_input() is not None:
            return None
        fallback = pick_fallback_input_device(list_input_devices())
        return fallback["id"] if fallback else None
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


def friendly_audio_error(exc: Exception | str) -> str:
    """Convierte los errores de PortAudio en un mensaje que se entienda en la demo."""
    text = str(exc)
    lowered = text.lower()
    if "device -1" in lowered or "no default input" in lowered or "invalid device" in lowered:
        return "No se encontró un micrófono. Conéctalo o elígelo en Configuración de voz y reinicia GIA."
    if "unanticipated host error" in lowered or "access" in lowered:
        return "Windows bloqueó el micrófono. Revisa Privacidad > Micrófono y que ninguna otra app lo use."
    return f"Error de micrófono: {text}"


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
    resolved_device = resolve_input_device(device)
    capture_rate = int(sample_rate)
    try:
        sd.check_input_settings(device=resolved_device, samplerate=capture_rate, channels=1, dtype="float32")
    except Exception:
        # Algunos controladores (p. ej. WDM-KS) no remuestrean: se graba a su frecuencia nativa.
        capture_rate = _device_native_rate(resolved_device) or capture_rate
    recording = sd.rec(
        max(1, int(duration_s * capture_rate)),
        samplerate=capture_rate,
        channels=1,
        dtype="float32",
        device=resolved_device,
    )
    sd.wait()
    mono = _resample(np.asarray(recording, dtype=np.float32).reshape(-1), capture_rate, int(sample_rate))
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
            stream_rate = self.sample_rate
            try:
                sd.check_input_settings(device=self.device, samplerate=stream_rate, channels=1, dtype="float32")
            except Exception:
                stream_rate = _device_native_rate(self.device) or stream_rate
            blocksize = max(256, int(stream_rate * self.block_duration_s))
            self._stream = sd.InputStream(
                samplerate=stream_rate,
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

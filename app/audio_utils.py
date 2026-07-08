from __future__ import annotations

import queue
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


class SilenceEndpointer:
    """Decide cuándo cortar una captura de voz. Puro y sin I/O: se alimenta
    con bloques mono float32 y devuelve el motivo de corte o None (seguir).

    Motivos de corte:
    - "silence": hubo voz y luego silencio sostenido (fin de habla).
    - "start_timeout": nunca se detectó voz dentro del tiempo de arranque.
    - "max_duration": se alcanzó la duración máxima permitida.
    """

    NOISE_CALIBRATION_BLOCKS = 3
    NOISE_RMS_CAP = 0.05

    def __init__(
        self,
        sample_rate: int,
        *,
        silence_duration_s: float = 0.8,
        min_duration_s: float = 1.0,
        max_duration_s: float = 6.0,
        start_timeout_s: float = 2.5,
        base_rms_threshold: float = 0.012,
    ):
        self.sample_rate = int(sample_rate)
        self.silence_duration_s = float(silence_duration_s)
        self.min_duration_s = float(min_duration_s)
        self.max_duration_s = float(max_duration_s)
        self.start_timeout_s = float(start_timeout_s)
        self.base_rms_threshold = float(base_rms_threshold)
        self.speech_detected = False
        self.elapsed_s = 0.0
        self._silence_s = 0.0
        self._noise_rms: list[float] = []

    def _voice_threshold(self) -> float:
        if not self._noise_rms:
            return self.base_rms_threshold
        noise_floor = float(np.median(self._noise_rms))
        return max(self.base_rms_threshold, noise_floor * 3.0)

    def feed(self, block: np.ndarray) -> str | None:
        mono = np.asarray(block, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return None
        block_s = mono.size / float(self.sample_rate)
        rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float64)))))

        if len(self._noise_rms) < self.NOISE_CALIBRATION_BLOCKS:
            # El piso de ruido se estima con los primeros bloques; se acota
            # para que hablar de inmediato no infle el umbral.
            self._noise_rms.append(min(rms, self.NOISE_RMS_CAP))

        self.elapsed_s += block_s
        if rms >= self._voice_threshold():
            self.speech_detected = True
            self._silence_s = 0.0
        elif self.speech_detected:
            self._silence_s += block_s

        # Épsilon para que la acumulación en flotante de bloques (p. ej. 8 x 0.1 s)
        # alcance los umbrales exactos.
        epsilon = 1e-6
        if self.elapsed_s + epsilon >= self.max_duration_s:
            return "max_duration"
        if not self.speech_detected and self.elapsed_s + epsilon >= self.start_timeout_s:
            return "start_timeout"
        if (
            self.speech_detected
            and self.elapsed_s + epsilon >= self.min_duration_s
            and self._silence_s + epsilon >= self.silence_duration_s
        ):
            return "silence"
        return None


def capture_voice_clip_until_silence(
    sample_rate: int,
    *,
    device=None,
    max_duration_s: float = 6.0,
    min_duration_s: float = 1.0,
    silence_duration_s: float = 0.8,
    block_duration_s: float = 0.1,
) -> tuple[np.ndarray, dict]:
    """Graba desde el micrófono hasta detectar fin de habla (o timeouts)."""
    resolved_device = resolve_input_device(device)
    endpointer = SilenceEndpointer(
        sample_rate,
        silence_duration_s=silence_duration_s,
        min_duration_s=min_duration_s,
        max_duration_s=max_duration_s,
    )
    block_queue: queue.Queue = queue.Queue()
    blocks: list[np.ndarray] = []
    blocksize = max(256, int(sample_rate * block_duration_s))

    def _callback(indata, frames, time_info, status):
        block_queue.put(indata.copy())

    stopped_reason = "max_duration"
    stream = sd.InputStream(
        samplerate=int(sample_rate),
        channels=1,
        dtype="float32",
        device=resolved_device,
        blocksize=blocksize,
        callback=_callback,
    )
    with stream:
        stall_deadline = time.monotonic() + max_duration_s + 2.0
        while True:
            try:
                block = block_queue.get(timeout=0.5)
            except queue.Empty:
                if time.monotonic() >= stall_deadline:
                    stopped_reason = "stream_stalled"
                    break
                continue
            mono = np.asarray(block, dtype=np.float32).reshape(-1)
            blocks.append(mono)
            reason = endpointer.feed(mono)
            if reason:
                stopped_reason = reason
                break

    recording = np.concatenate(blocks) if blocks else np.zeros(1, dtype=np.float32)
    diagnostics = analyze_audio(
        recording,
        int(sample_rate),
        device_label=describe_input_device(resolved_device),
    )
    diagnostics["stopped_reason"] = stopped_reason
    diagnostics["speech_detected"] = endpointer.speech_detected
    return recording, diagnostics


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

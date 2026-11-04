from __future__ import annotations

import collections
import gzip
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .config_store import USAGE_RECORDING_DIR
from .gesture_ml import deserialize_face_sample, serialize_face_sample
from .models import FaceSample

MAX_PENDING_FRAMES = 600
FLUSH_EVERY_FRAMES = 150


class UsageRecorder:
    """Graba los puntos faciales y las decisiones durante el uso real (sin video ni audio).

    La escritura ocurre en un hilo propio: si el disco se atrasa se descartan fotogramas
    (y se cuentan) en lugar de frenar el bucle de cámara.
    """

    def __init__(self, profile_name: str, metadata: dict | None = None, directory: Path | str | None = None):
        self.profile_name = profile_name
        self.metadata = metadata or {}
        self.directory = Path(directory) if directory is not None else USAGE_RECORDING_DIR
        self.path: Path | None = None
        self.active = False
        self.frames = 0
        self.dropped = 0
        self.error = ""
        self.markers: collections.Counter = collections.Counter()
        self._started_at = 0.0
        self._queue: queue.Queue = queue.Queue(maxsize=MAX_PENDING_FRAMES)
        self._thread: threading.Thread | None = None

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._started_at if self._started_at else 0.0

    def start(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = self.directory / f"{self.profile_name}_{stamp}.jsonl.gz"
        stream = gzip.open(self.path, "wt", encoding="utf-8")
        header = {"type": "header", "profile": self.profile_name, "started_at": datetime.now().isoformat(), **self.metadata}
        stream.write(json.dumps(header, ensure_ascii=False) + "\n")
        self._started_at = time.monotonic()
        self.active = True
        self._thread = threading.Thread(target=self._run, args=(stream,), name="gia-usage-recorder", daemon=True)
        self._thread.start()
        return self.path

    def record_frame(self, sample: FaceSample | None, **decision) -> None:
        if not self.active:
            return
        try:
            self._queue.put_nowait(("frame", self.elapsed_s, sample, decision))
            self.frames += 1
        except queue.Full:
            self.dropped += 1

    def mark(self, kind: str) -> None:
        if not self.active:
            return
        self.markers[kind] += 1
        try:
            self._queue.put_nowait(("marker", self.elapsed_s, kind, None))
        except queue.Full:
            self.dropped += 1

    def stop(self, timeout: float = 15.0) -> dict:
        seconds = self.elapsed_s
        if self.active:
            self.active = False
            self._queue.put(None)
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout)
        return {
            "path": str(self.path) if self.path else "",
            "seconds": seconds,
            "frames": self.frames,
            "dropped": self.dropped,
            "markers": dict(self.markers),
            "error": self.error,
        }

    def _run(self, stream) -> None:
        written = 0
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                kind, elapsed, payload, decision = item
                if kind == "marker":
                    row = {"type": "marker", "t": round(elapsed, 3), "kind": payload}
                else:
                    row = {
                        "type": "frame",
                        "t": round(elapsed, 3),
                        "sample": serialize_face_sample(payload) if payload is not None else None,
                        **decision,
                    }
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
                if written % FLUSH_EVERY_FRAMES == 0:
                    stream.flush()  # que un cierre brusco no pierda toda la grabación
            stream.write(
                json.dumps({"type": "footer", "frames": self.frames, "dropped": self.dropped, "markers": dict(self.markers)}) + "\n"
            )
        except Exception as exc:  # noqa: BLE001 - se informa al detener la grabación
            self.error = str(exc)
            self.active = False
        finally:
            try:
                stream.close()
            except Exception as exc:  # noqa: BLE001
                self.error = self.error or str(exc)


def read_usage_recording(path: Path | str) -> Iterator[dict]:
    """Lee una grabación; tolera un archivo truncado por un cierre brusco."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("type") == "frame" and row.get("sample") is not None:
                    row["sample"] = deserialize_face_sample(row["sample"])
                yield row
    except (EOFError, gzip.BadGzipFile):
        return

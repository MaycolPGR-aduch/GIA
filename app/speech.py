from __future__ import annotations

import threading
from collections import deque


def create_tts_engine(rate: int = 150):
    """Crea el motor pyttsx3 en el hilo que lo va a usar (SAPI exige COM inicializado ahí)."""
    try:
        import comtypes

        comtypes.CoInitialize()
    except Exception:
        pass
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    return engine


class SpeechQueue:
    """Síntesis de voz en un hilo propio para no bloquear el bucle de cámara.

    Solo conserva los últimos ``max_pending`` mensajes: si se acumulan avisos, los viejos
    se descartan porque ya no describen el estado actual.
    """

    def __init__(self, engine_factory=create_tts_engine, max_pending: int = 2):
        self._engine_factory = engine_factory
        self._max_pending = max(1, int(max_pending))
        self._pending: deque[tuple[str, threading.Event]] = deque()
        self._cond = threading.Condition()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="gia-tts", daemon=True)
        self._thread.start()

    def say(self, text: str, *, wait: bool = False, timeout: float = 4.0) -> None:
        done = threading.Event()
        with self._cond:
            if not self._running:
                return
            while len(self._pending) >= self._max_pending:
                _, dropped = self._pending.popleft()
                dropped.set()
            self._pending.append((text, done))
            self._cond.notify()
        if wait:
            done.wait(timeout)

    def stop(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._running = False
            while self._pending:
                _, dropped = self._pending.popleft()
                dropped.set()
            self._cond.notify()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout)

    def _run(self) -> None:
        try:
            engine = self._engine_factory()
        except Exception as exc:
            print(f"Error configurando sintesis de voz: {exc}")
            engine = None
        while True:
            with self._cond:
                while self._running and not self._pending:
                    self._cond.wait()
                if not self._running:
                    return
                text, done = self._pending.popleft()
            try:
                if engine is not None:
                    engine.say(text)
                    engine.runAndWait()
            except Exception as exc:
                print(f"Error reproduciendo TTS: {exc}")
            finally:
                done.set()

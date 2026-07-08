from __future__ import annotations

import queue
import threading


class TtsWorker:
    """Síntesis de voz en un hilo dedicado.

    pyttsx3 se inicializa DENTRO del hilo worker (requisito COM en Windows) y
    `speak()` solo encola, por lo que nunca bloquea el hilo de video ni la UI.
    """

    _SENTINEL = None

    def __init__(self, rate: int = 150):
        self.rate = rate
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def speak(self, text: str) -> None:
        if not text:
            return
        self._queue.put(text)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=2)
        self._thread = None

    def _run(self) -> None:
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
        except Exception as exc:
            print(f"Error configurando sintesis de voz: {exc}")
            engine = None

        while True:
            text = self._queue.get()
            if text is self._SENTINEL:
                break
            if engine is None:
                continue
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:
                print(f"Error reproduciendo TTS: {exc}")

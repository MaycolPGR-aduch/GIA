from __future__ import annotations

import sys
import threading

# Atajo global de emergencia: funciona aunque GIA no tenga el foco (p. ej. tras un clic
# facial en otra aplicación), algo que los atajos de Qt no pueden hacer.
EMERGENCY_HOTKEY_LABEL = "Ctrl+Alt+P"

_MOD_ALT = 0x0001
_MOD_CONTROL = 0x0002
_MOD_NOREPEAT = 0x4000
_WM_HOTKEY = 0x0312
_WM_QUIT = 0x0012
_HOTKEY_ID = 0x6A1


class GlobalHotkey:
    """Registra un atajo de teclado del sistema (RegisterHotKey) en un hilo propio."""

    def __init__(self, callback, *, modifiers: int = _MOD_CONTROL | _MOD_ALT, key: str = "P"):
        self.callback = callback
        self.modifiers = modifiers | _MOD_NOREPEAT
        self.virtual_key = ord(key.upper())
        self.registered = False
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._ready = threading.Event()

    def start(self, timeout: float = 2.0) -> bool:
        if sys.platform != "win32":
            return False
        if self._thread is not None:
            return self.registered
        self._thread = threading.Thread(target=self._run, name="gia-hotkey", daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        return self.registered

    def stop(self, timeout: float = 2.0) -> None:
        thread = self._thread
        if thread is None:
            return
        if self._thread_id is not None:
            import ctypes

            ctypes.windll.user32.PostThreadMessageW(self._thread_id, _WM_QUIT, 0, 0)
        if thread is not threading.current_thread():
            thread.join(timeout)
        self._thread = None

    def _run(self) -> None:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        self.registered = bool(user32.RegisterHotKey(None, _HOTKEY_ID, self.modifiers, self.virtual_key))
        self._ready.set()
        if not self.registered:
            return
        try:
            msg = wintypes.MSG()
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == _WM_HOTKEY and msg.wParam == _HOTKEY_ID:
                    try:
                        self.callback()
                    except Exception as exc:
                        print(f"Error en atajo global: {exc}")
        finally:
            user32.UnregisterHotKey(None, _HOTKEY_ID)
            self.registered = False

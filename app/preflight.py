from __future__ import annotations

from dataclasses import dataclass

from .audio_utils import friendly_audio_error
from .global_hotkey import EMERGENCY_HOTKEY_LABEL


# Tiempos de gracia antes de dar un chequeo por fallido.
CAMERA_OPEN_GRACE_S = 10.0
FACE_GRACE_S = 4.0
MIC_GRACE_S = 8.0
MIC_OK_PEAK = 0.02
MIC_SILENT_PEAK = 1e-4
FACE_RECENT_S = 1.0


@dataclass(slots=True)
class PreflightCheck:
    key: str
    title: str
    status: str  # "pending" | "ok" | "warning" | "error" | "skipped"
    detail: str


def _camera_check(snapshot: dict, elapsed_s: float) -> PreflightCheck:
    state = snapshot.get("camera_state", "abriendo")
    fps = float(snapshot.get("fps_real", 0.0) or 0.0)
    if state == "ok":
        return PreflightCheck("camera", "Cámara", "ok", f"Funcionando a {fps:.0f} FPS" if fps else "Funcionando")
    if state == "abriendo" and elapsed_s < CAMERA_OPEN_GRACE_S:
        return PreflightCheck("camera", "Cámara", "pending", "Abriendo la cámara…")
    return PreflightCheck(
        "camera",
        "Cámara",
        "error",
        "No hay imagen. Revisa que esté conectada y que ninguna otra app (Zoom, Teams, el Entrenador) "
        "la esté usando. GIA reintenta solo.",
    )


def _face_check(snapshot: dict) -> PreflightCheck:
    if snapshot.get("camera_state") != "ok":
        return PreflightCheck("face", "Rostro", "pending", "Esperando a la cámara…")
    since_seen = snapshot.get("face_seen_ago_s")
    if since_seen is not None and since_seen <= FACE_RECENT_S:
        return PreflightCheck("face", "Rostro", "ok", "Rostro detectado")
    if float(snapshot.get("camera_ok_for_s", 0.0) or 0.0) < FACE_GRACE_S:
        return PreflightCheck("face", "Rostro", "pending", "Buscando rostro…")
    return PreflightCheck(
        "face", "Rostro", "warning", "No se detecta un rostro. Colócate de frente a la cámara, con buena luz."
    )


def _mic_check(snapshot: dict, elapsed_s: float, mic_peak_seen: float) -> PreflightCheck:
    if not snapshot.get("mic_monitor_enabled", False):
        return PreflightCheck("mic", "Micrófono", "skipped", "Comandos de voz desactivados en la configuración.")
    if not snapshot.get("mic_available", False):
        return PreflightCheck("mic", "Micrófono", "error", friendly_audio_error(snapshot.get("mic_error", "")))
    device = snapshot.get("mic_device", "")
    if mic_peak_seen >= MIC_OK_PEAK:
        return PreflightCheck("mic", "Micrófono", "ok", f"Señal recibida · {device}")
    if elapsed_s < MIC_GRACE_S:
        return PreflightCheck("mic", "Micrófono", "pending", "Di algo en voz alta para probarlo…")
    if mic_peak_seen < MIC_SILENT_PEAK:
        return PreflightCheck(
            "mic",
            "Micrófono",
            "error",
            "No entrega señal (silencio absoluto). Revisa que no esté silenciado y que Windows lo permita "
            "en Privacidad > Micrófono.",
        )
    return PreflightCheck(
        "mic",
        "Micrófono",
        "warning",
        f"Señal muy baja (pico {mic_peak_seen * 100:.0f} %). Acércate o sube su volumen de entrada en Windows.",
    )


def _voice_model_check(snapshot: dict) -> PreflightCheck:
    state = snapshot.get("voice_model_state", "deshabilitada")
    if state == "lista":
        return PreflightCheck("voice", "Reconocimiento de voz", "ok", "Modelo Whisper cargado")
    if state == "cargando":
        return PreflightCheck("voice", "Reconocimiento de voz", "pending", "Cargando el modelo Whisper…")
    if state == "error":
        detail = snapshot.get("voice_model_error") or "error desconocido"
        return PreflightCheck("voice", "Reconocimiento de voz", "error", f"No se pudo cargar: {detail}")
    return PreflightCheck("voice", "Reconocimiento de voz", "skipped", "Voz desactivada en este perfil.")


def _gesture_model_check(snapshot: dict) -> PreflightCheck:
    if not snapshot.get("model_loaded", False):
        return PreflightCheck(
            "model", "Modelo de gestos", "error", "Este perfil no tiene un modelo entrenado. Abre el Entrenador."
        )
    parts = []
    if snapshot.get("model_version") is not None:
        parts.append(f"Versión {snapshot['model_version']}")
    accuracy = snapshot.get("model_accuracy")
    if accuracy is not None:
        parts.append(f"{float(accuracy) * 100:.0f} % en validación")
    return PreflightCheck("model", "Modelo de gestos", "ok", " · ".join(parts) or "Cargado")


def _hotkey_check(snapshot: dict) -> PreflightCheck:
    if snapshot.get("hotkey_registered", False):
        return PreflightCheck(
            "hotkey", "Pausa de emergencia", "ok", f"{EMERGENCY_HOTKEY_LABEL} funciona desde cualquier aplicación"
        )
    return PreflightCheck(
        "hotkey",
        "Pausa de emergencia",
        "warning",
        f"{EMERGENCY_HOTKEY_LABEL} no está disponible (otra app lo usa). Usa Ctrl+P con GIA en primer plano.",
    )


def evaluate_preflight(snapshot: dict, elapsed_s: float, mic_peak_seen: float = 0.0) -> list[PreflightCheck]:
    """Evalúa los chequeos previos a partir de una foto del estado del controlador."""
    return [
        _camera_check(snapshot, elapsed_s),
        _face_check(snapshot),
        _mic_check(snapshot, elapsed_s, mic_peak_seen),
        _voice_model_check(snapshot),
        _gesture_model_check(snapshot),
        _hotkey_check(snapshot),
    ]


def summarize_preflight(checks: list[PreflightCheck]) -> tuple[str, str]:
    """Devuelve (texto, tono) del resumen general."""
    errors = sum(check.status == "error" for check in checks)
    warnings = sum(check.status == "warning" for check in checks)
    pending = sum(check.status == "pending" for check in checks)
    if errors:
        return f"{errors} problema{'s' if errors > 1 else ''} por resolver", "danger"
    if pending:
        return "Verificando…", "info"
    if warnings:
        return f"Listo, con {warnings} aviso{'s' if warnings > 1 else ''}", "warning"
    return "Todo listo", "success"

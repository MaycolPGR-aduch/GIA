GESTURE_CATALOG = [
    {
        "id": "left_blink_intent",
        "enabled": True,
        "title": "Guiño izquierdo intencional",
        "short_title": "Guiño izquierdo",
        "short_action": "Clic izquierdo",
        "action": "Clic izquierdo",
        "duration_ms": 350,
        "warning": "Debe mantenerse levemente más que un parpadeo natural.",
        "training_hint": "Mantén el ojo izquierdo cerrado de forma deliberada mientras el derecho sigue abierto.",
        "common_failure_hint": "No gires la cabeza ni cierres ambos ojos al mismo tiempo.",
        "success_hint": "Se verá listo cuando acumule suficientes ventanas útiles para ML.",
        "recommended_min_frames": 28,
        "recommended_min_windows": 12,
    },
    {
        "id": "right_blink_intent",
        "enabled": True,
        "title": "Guiño derecho intencional",
        "short_title": "Guiño derecho",
        "short_action": "Clic derecho",
        "action": "Clic derecho",
        "duration_ms": 350,
        "warning": "No se ejecuta si la confianza es baja o el sistema no está estable.",
        "training_hint": "Mantén el ojo derecho cerrado de forma deliberada mientras el izquierdo sigue abierto.",
        "common_failure_hint": "Evita sonreír o mover la cabeza durante la captura.",
        "success_hint": "Necesita ventanas claras para diferenciarse del parpadeo natural.",
        "recommended_min_frames": 28,
        "recommended_min_windows": 12,
    },
    {
        "id": "both_eyes_closed_intent",
        "enabled": True,
        "title": "Ambos ojos cerrados",
        "short_title": "Ojos cerrados",
        "short_action": "Escuchar voz",
        "action": "Escuchar comando de voz",
        "duration_ms": 900,
        "warning": "Mantén ambos ojos cerrados de forma deliberada.",
        "training_hint": "Cierra ambos ojos con intención y mantenlos sin apretar el resto del rostro.",
        "common_failure_hint": "Si se mezcla con guiños asimétricos el modelo se vuelve confuso.",
        "success_hint": "Debe sentirse como un gesto sostenido, no como un parpadeo largo.",
        "recommended_min_frames": 32,
        "recommended_min_windows": 14,
    },
    {
        "id": "mouth_open_hold",
        "enabled": True,
        "title": "Boca en O sostenida",
        "short_title": "Boca en O",
        "short_action": "Activar/congelar cursor",
        "action": "Activar o congelar cursor",
        "duration_ms": 700,
        "warning": "Se usa como clutch del cursor. Debe formar una O clara y deliberada.",
        "training_hint": "Forma una O clara con la boca y mantenla estable hasta que termine la captura.",
        "common_failure_hint": "Evita hablar durante esta captura; eso mete ruido en el gesto.",
        "success_hint": "Mientras más estable sea la apertura, mejor separará este gesto de sonrisa y neutral.",
        "recommended_min_frames": 30,
        "recommended_min_windows": 13,
    },
    {
        "id": "smile",
        "enabled": True,
        "title": "Sonrisa",
        "short_title": "Sonrisa",
        "short_action": "Recentrar cursor",
        "action": "Recentrar cursor",
        "duration_ms": 600,
        "warning": "Úsala cuando el control se desvíe.",
        "training_hint": "Haz una sonrisa clara pero estable, sin inclinar la cabeza.",
        "common_failure_hint": "Una media sonrisa suele confundirse con neutral o confirmación.",
        "success_hint": "La captura es buena cuando la sonrisa se mantiene sin cambios bruscos.",
        "recommended_min_frames": 30,
        "recommended_min_windows": 13,
    },
    {
        "id": "brows_up",
        "enabled": False,
        "title": "Cejas levantadas",
        "short_title": "Cejas arriba",
        "short_action": "Pausar/reanudar",
        "action": "Pausar o reanudar",
        "duration_ms": 700,
        "warning": "Es el gesto de seguridad principal.",
        "training_hint": "Eleva ambas cejas manteniendo ojos y boca lo más neutros posible.",
        "common_failure_hint": "No combines este gesto con sorpresa grande o apertura de boca.",
        "success_hint": "Debe sentirse como un gesto de seguridad, claro y simétrico.",
        "recommended_min_frames": 30,
        "recommended_min_windows": 13,
    },
    {
        "id": "confirm",
        "enabled": True,
        "title": "Gesto de confirmación",
        "short_title": "Confirmación",
        "short_action": "Confirmar",
        "action": "Confirmar selección actual",
        "duration_ms": 650,
        "warning": "Reservado para flujos guiados y extensiones futuras.",
        "training_hint": "Usa siempre la misma expresión de confirmación para que el perfil sea consistente.",
        "common_failure_hint": "Si cambia mucho entre capturas, el modelo no aprenderá una firma clara.",
        "success_hint": "Aunque hoy se use poco, conviene entrenarlo con buena consistencia.",
        "recommended_min_frames": 30,
        "recommended_min_windows": 13,
    },
]


def get_enabled_gesture_catalog() -> list[dict]:
    return [gesture for gesture in GESTURE_CATALOG if gesture.get("enabled", True)]


def is_gesture_enabled(gesture_id: str) -> bool:
    if gesture_id == "neutral":
        return True
    for gesture in GESTURE_CATALOG:
        if gesture["id"] == gesture_id:
            return bool(gesture.get("enabled", True))
    return True


NEUTRAL_GESTURE_META = {
    "id": "neutral",
    "title": "Postura neutral",
    "short_title": "Neutral",
    "short_action": "Referencia base",
    "action": "Referencia base",
    "duration_ms": 0,
    "warning": "Mantén una postura relajada mirando al frente.",
    "training_hint": "Mira al frente y mantén una expresión relajada y estable.",
    "common_failure_hint": "Si te mueves mucho, el neutral deja de ser una referencia confiable.",
    "success_hint": "La captura es buena cuando nariz, ojos y boca permanecen estables.",
    "recommended_min_frames": 36,
    "recommended_min_windows": 18,
}


VOICE_COMMAND_HELP = [
    ("Pausar sistema", "Pone el runtime en modo seguro sin clics ni voz."),
    ("Reanudar sistema", "Saca el runtime del modo pausa."),
    ("Activar cursor", "Activa el movimiento del cursor sin tocar la pausa global."),
    ("Congelar cursor", "Desactiva temporalmente el movimiento del cursor."),
    ("Centrar cursor", "Recalibra la referencia actual de nariz."),
    ("Abrir guia", "Muestra la guia rapida de gestos."),
    ("Estamos listos", "Activa el modo compacto para usar la PC con minima interfaz."),
    ("Volvamos", "Restaura la interfaz grande con la camara visible."),
    ("Cerrar sistema", "Solicita terminar la sesion."),
    ("Volumen 25/50/75/100", "Ajusta el volumen del sistema si el backend de audio esta disponible."),
    ("Abrir Gmail/Facebook/WhatsApp/YouTube", "Abre accesos rapidos web integrados."),
]

GESTURE_CATALOG = [
    {
        "id": "left_blink_intent",
        "title": "Guiño izquierdo intencional",
        "action": "Clic izquierdo",
        "duration_ms": 350,
        "warning": "Debe mantenerse levemente más que un parpadeo natural.",
    },
    {
        "id": "right_blink_intent",
        "title": "Guiño derecho intencional",
        "action": "Clic derecho",
        "duration_ms": 350,
        "warning": "No se ejecuta si la confianza es baja o el sistema no está estable.",
    },
    {
        "id": "both_eyes_closed_intent",
        "title": "Ambos ojos cerrados",
        "action": "Escuchar comando de voz",
        "duration_ms": 900,
        "warning": "Mantén ambos ojos cerrados de forma deliberada.",
    },
    {
        "id": "mouth_open_hold",
        "title": "Boca abierta sostenida",
        "action": "Abrir guía rápida",
        "duration_ms": 700,
        "warning": "No cierra el sistema; se usa como gesto seguro de ayuda.",
    },
    {
        "id": "smile",
        "title": "Sonrisa",
        "action": "Recentrar cursor",
        "duration_ms": 600,
        "warning": "Úsala cuando el control se desvíe.",
    },
    {
        "id": "brows_up",
        "title": "Cejas levantadas",
        "action": "Pausar o reanudar",
        "duration_ms": 700,
        "warning": "Es el gesto de seguridad principal.",
    },
    {
        "id": "confirm",
        "title": "Gesto de confirmación",
        "action": "Confirmar selección actual",
        "duration_ms": 650,
        "warning": "Reservado para flujos guiados y extensiones futuras.",
    },
]


VOICE_COMMAND_HELP = [
    ("Pausar sistema", "Pone el runtime en modo seguro sin clics ni voz."),
    ("Reanudar sistema", "Saca el runtime del modo pausa."),
    ("Centrar cursor", "Recalibra la referencia actual de nariz."),
    ("Abrir guía", "Muestra la guía rápida de gestos."),
    ("Cerrar sistema", "Solicita terminar la sesión."),
    ("Volumen 25/50/75/100", "Ajusta el volumen del sistema si el backend de audio está disponible."),
    ("Abrir Gmail/Facebook/WhatsApp/YouTube", "Abre accesos rápidos web integrados."),
]

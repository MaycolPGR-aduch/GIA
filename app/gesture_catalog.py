GESTURE_CATALOG = [
    {
        "id": "left_blink_intent",
        "title": "Guino izquierdo intencional",
        "action": "Clic izquierdo",
        "duration_ms": 350,
        "warning": "Debe mantenerse levemente mas que un parpadeo natural.",
    },
    {
        "id": "right_blink_intent",
        "title": "Guino derecho intencional",
        "action": "Clic derecho",
        "duration_ms": 350,
        "warning": "No se ejecuta si la confianza es baja o el sistema no esta estable.",
    },
    {
        "id": "both_eyes_closed_intent",
        "title": "Ambos ojos cerrados",
        "action": "Escuchar comando de voz",
        "duration_ms": 900,
        "warning": "Manten ambos ojos cerrados de forma deliberada.",
    },
    {
        "id": "mouth_open_hold",
        "title": "Boca en O sostenida",
        "action": "Activar o congelar cursor",
        "duration_ms": 700,
        "warning": "Se usa como clutch del cursor. Debe formar una O clara y deliberada.",
    },
    {
        "id": "smile",
        "title": "Sonrisa",
        "action": "Recentrar cursor",
        "duration_ms": 600,
        "warning": "Usala cuando el control se desvie.",
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
        "title": "Gesto de confirmacion",
        "action": "Confirmar seleccion actual",
        "duration_ms": 650,
        "warning": "Reservado para flujos guiados y extensiones futuras.",
    },
]


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

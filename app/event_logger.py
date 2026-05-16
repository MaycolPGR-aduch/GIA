from .config_store import ensure_app_dirs


def inicializar_log():
    ensure_app_dirs()


def registrar_evento(accion, detalle, tiempo_respuesta=0.0):
    _ = (accion, detalle, tiempo_respuesta)
    ensure_app_dirs()

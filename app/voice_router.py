from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


VOICE_ALIASES = {
    "pause": [
        "pausar sistema",
        "pausa",
        "modo pausa",
    ],
    "resume": [
        "reanudar sistema",
        "continuar sistema",
        "reanudar",
    ],
    "recenter": [
        "centrar cursor",
        "recalibrar cursor",
        "recentrar cursor",
    ],
    "guide": [
        "abrir guia",
        "mostrar guia",
        "ver guia",
    ],
    "quit": [
        "cerrar sistema",
        "terminar ejecucion",
        "salir del sistema",
    ],
    "volume_25": ["volumen 25", "volumen al 25", "volumen 25 por ciento"],
    "volume_50": ["volumen 50", "volumen al 50", "volumen 50 por ciento"],
    "volume_75": ["volumen 75", "volumen al 75", "volumen 75 por ciento"],
    "volume_100": ["volumen 100", "volumen al 100", "volumen 100 por ciento"],
    "open_gmail": ["abrir gmail", "gmail"],
    "open_facebook": ["abrir facebook", "facebook"],
    "open_whatsapp": ["abrir whatsapp", "whatsapp"],
    "open_youtube": ["abrir youtube", "youtube"],
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9% ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def resolve_command(text: str) -> tuple[str | None, float]:
    normalized = normalize_text(text)
    best_command = None
    best_score = 0.0

    for command_id, aliases in VOICE_ALIASES.items():
        for alias in aliases:
            alias_norm = normalize_text(alias)
            score = SequenceMatcher(None, normalized, alias_norm).ratio()
            if alias_norm in normalized:
                score = max(score, 0.95)
            if score > best_score:
                best_command = command_id
                best_score = score
    return best_command, best_score

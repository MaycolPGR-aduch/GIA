from __future__ import annotations

import re
import unicodedata
from copy import deepcopy
from difflib import SequenceMatcher


BUILTIN_COMMAND_SPECS = [
    {
        "id": "pause",
        "label": "Pausar sistema",
        "description": "Pone el runtime en modo seguro sin clics ni voz.",
        "default_phrases": ["pausar sistema", "pausa", "modo pausa"],
        "direct_action_id": "pause",
    },
    {
        "id": "resume",
        "label": "Reanudar sistema",
        "description": "Saca el runtime del modo pausa.",
        "default_phrases": ["reanudar sistema", "continuar sistema", "reanudar"],
        "direct_action_id": "resume",
    },
    {
        "id": "recenter",
        "label": "Centrar cursor",
        "description": "Recalibra la referencia actual de nariz.",
        "default_phrases": ["centrar cursor", "recalibrar cursor", "recentrar cursor"],
        "direct_action_id": "recenter",
    },
    {
        "id": "cursor_on",
        "label": "Activar cursor",
        "description": "Activa el movimiento del cursor sin tocar la pausa global.",
        "default_phrases": ["activar cursor", "encender cursor", "habilitar cursor", "mover cursor"],
        "direct_action_id": "cursor_on",
    },
    {
        "id": "cursor_off",
        "label": "Congelar cursor",
        "description": "Desactiva temporalmente el movimiento del cursor.",
        "default_phrases": ["congelar cursor", "desactivar cursor", "detener cursor", "apagar cursor"],
        "direct_action_id": "cursor_off",
    },
    {
        "id": "guide",
        "label": "Abrir guia",
        "description": "Muestra la guia rapida de gestos.",
        "default_phrases": ["abrir guia", "mostrar guia", "ver guia"],
        "direct_action_id": "guide",
    },
    {
        "id": "compact_ui",
        "label": "Estamos listos",
        "description": "Activa el modo compacto para usar la PC con minima interfaz.",
        "default_phrases": ["estamos listos", "modo compacto", "minimizar interfaz", "comenzar a usar la pc"],
        "direct_action_id": "compact_ui",
    },
    {
        "id": "expand_ui",
        "label": "Volvamos",
        "description": "Restaura la interfaz grande con la camara visible.",
        "default_phrases": ["volvamos", "restaurar interfaz", "volver a la interfaz", "mostrar interfaz"],
        "direct_action_id": "expand_ui",
    },
    {
        "id": "quit",
        "label": "Cerrar sistema",
        "description": "Solicita terminar la sesion.",
        "default_phrases": ["cerrar sistema", "terminar ejecucion", "salir del sistema"],
        "direct_action_id": "quit",
    },
    {
        "id": "volume_25",
        "label": "Volumen 25",
        "description": "Ajusta el volumen del sistema al 25 por ciento.",
        "default_phrases": ["volumen 25", "volumen al 25", "volumen 25 por ciento"],
        "direct_action_id": "volume_25",
    },
    {
        "id": "volume_50",
        "label": "Volumen 50",
        "description": "Ajusta el volumen del sistema al 50 por ciento.",
        "default_phrases": ["volumen 50", "volumen al 50", "volumen 50 por ciento"],
        "direct_action_id": "volume_50",
    },
    {
        "id": "volume_75",
        "label": "Volumen 75",
        "description": "Ajusta el volumen del sistema al 75 por ciento.",
        "default_phrases": ["volumen 75", "volumen al 75", "volumen 75 por ciento"],
        "direct_action_id": "volume_75",
    },
    {
        "id": "volume_100",
        "label": "Volumen 100",
        "description": "Ajusta el volumen del sistema al 100 por ciento.",
        "default_phrases": ["volumen 100", "volumen al 100", "volumen 100 por ciento"],
        "direct_action_id": "volume_100",
    },
    {
        "id": "open_gmail",
        "label": "Abrir Gmail",
        "description": "Abre Gmail en el navegador.",
        "default_phrases": ["abrir gmail", "gmail"],
        "direct_action_id": "open_gmail",
    },
    {
        "id": "open_facebook",
        "label": "Abrir Facebook",
        "description": "Abre Facebook en el navegador.",
        "default_phrases": ["abrir facebook", "facebook"],
        "direct_action_id": "open_facebook",
    },
    {
        "id": "open_whatsapp",
        "label": "Abrir WhatsApp",
        "description": "Abre WhatsApp Web en el navegador.",
        "default_phrases": ["abrir whatsapp", "whatsapp"],
        "direct_action_id": "open_whatsapp",
    },
    {
        "id": "open_youtube",
        "label": "Abrir YouTube",
        "description": "Abre YouTube en el navegador.",
        "default_phrases": ["abrir youtube", "youtube"],
        "direct_action_id": "open_youtube",
    },
]


VOICE_ALIASES = {spec["id"]: list(spec["default_phrases"]) for spec in BUILTIN_COMMAND_SPECS}
BUILTIN_COMMAND_SPEC_BY_ID = {spec["id"]: spec for spec in BUILTIN_COMMAND_SPECS}

DIRECT_ACTION_SPECS = [
    {"id": spec["direct_action_id"], "label": spec["label"], "description": spec["description"]}
    for spec in BUILTIN_COMMAND_SPECS
] + [
    {
        "id": "start_dictation",
        "label": "Modo dictado",
        "description": "Activa el dictado nativo de Windows con Win+H.",
    }
]
DIRECT_ACTION_SPEC_BY_ID = {spec["id"]: spec for spec in DIRECT_ACTION_SPECS}

DEFAULT_VOICE_SETTINGS = {
    "voice_model_size": "small",
    "voice_commands_enabled": True,
    "voice_builtin_alias_overrides": {},
    "voice_custom_commands": [],
}

# Palabras que indican negación o cancelación: si aparecen en la transcripción,
# el comando NO debe dispararse ("no quiero cerrar sistema" no debe cerrar).
NEGATION_TOKENS = {
    "no",
    "nunca",
    "jamas",
    "cancela",
    "cancelar",
    "deten",
    "detente",
    "olvida",
    "olvidalo",
}

# Un alias solo recibe el boost de coincidencia exacta si cubre al menos esta
# fracción de las palabras de la transcripción (evita disparos dentro de
# frases largas no dirigidas al sistema).
MIN_ALIAS_WORD_COVERAGE = 0.5

MACRO_KEY_ALIASES = {
    "control": "ctrl",
    "ctl": "ctrl",
    "windows": "win",
    "super": "win",
    "command": "win",
    "return": "enter",
    "escape": "esc",
    "option": "alt",
    "del": "delete",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9% ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_phrase_list(values) -> list[str]:
    phrases = []
    seen = set()
    for raw in values or []:
        phrase = str(raw).strip()
        if not phrase:
            continue
        normalized = normalize_text(phrase)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        phrases.append(phrase)
    return phrases


def split_phrases_text(text: str) -> list[str]:
    parts = re.split(r"[\n,;]+", text or "")
    return normalize_phrase_list(parts)


def normalize_macro_keys(keys) -> list[str]:
    normalized = []
    for raw in keys or []:
        key = str(raw).strip().lower()
        if not key:
            continue
        key = key.replace(" ", "")
        key = MACRO_KEY_ALIASES.get(key, key)
        normalized.append(key)
    return normalized


def parse_macro_keys(text: str) -> list[str]:
    parts = re.split(r"\s*\+\s*|[\n,;]+", text or "")
    return normalize_macro_keys(parts)


def format_macro_keys(keys: list[str]) -> str:
    display_parts = []
    for key in keys or []:
        if key == "ctrl":
            display_parts.append("Ctrl")
        elif key == "alt":
            display_parts.append("Alt")
        elif key == "win":
            display_parts.append("Win")
        elif len(key) == 1:
            display_parts.append(key.upper())
        else:
            display_parts.append(key.title())
    return " + ".join(display_parts)


def slugify_command_id(label: str) -> str:
    base = normalize_text(label).replace(" ", "_")
    base = re.sub(r"_+", "_", base).strip("_")
    return base or "custom_command"


def build_custom_command_id(label: str, existing_ids: set[str], preferred: str | None = None) -> str:
    candidate = slugify_command_id(preferred or label)
    if candidate not in existing_ids:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing_ids:
        suffix += 1
    return f"{candidate}_{suffix}"


def normalize_voice_settings(settings: dict | None) -> dict:
    merged = deepcopy(DEFAULT_VOICE_SETTINGS)
    if settings:
        for key, value in settings.items():
            if key not in DEFAULT_VOICE_SETTINGS:
                merged[key] = value
            elif key not in {"voice_builtin_alias_overrides", "voice_custom_commands"}:
                merged[key] = value

    merged["voice_model_size"] = str(merged.get("voice_model_size") or "small").strip().lower() or "small"
    merged["voice_commands_enabled"] = bool(merged.get("voice_commands_enabled", True))

    overrides = {}
    raw_overrides = settings.get("voice_builtin_alias_overrides", {}) if settings else {}
    if isinstance(raw_overrides, dict):
        for command_id, phrases in raw_overrides.items():
            if command_id in BUILTIN_COMMAND_SPEC_BY_ID:
                cleaned = normalize_phrase_list(phrases)
                if cleaned:
                    overrides[command_id] = cleaned
    merged["voice_builtin_alias_overrides"] = overrides

    custom_commands = []
    existing_ids = set()
    raw_custom = settings.get("voice_custom_commands", []) if settings else []
    if isinstance(raw_custom, list):
        for item in raw_custom:
            normalized = _normalize_custom_command(item or {}, existing_ids)
            custom_commands.append(normalized)
            existing_ids.add(normalized["id"])
    merged["voice_custom_commands"] = custom_commands
    return merged


def _normalize_custom_command(command: dict, existing_ids: set[str]) -> dict:
    label = str(command.get("label") or "Nuevo comando").strip()
    action_type = str(command.get("action_type") or "macro").strip().lower()
    if action_type not in {"macro", "direct_action"}:
        action_type = "macro"
    preferred_id = command.get("id")
    command_id = build_custom_command_id(label, existing_ids, preferred=str(preferred_id) if preferred_id else None)
    spoken_phrases = normalize_phrase_list(command.get("spoken_phrases", []))
    macro_keys = normalize_macro_keys(command.get("macro_keys", []))
    direct_action_id = command.get("direct_action_id")
    if direct_action_id not in DIRECT_ACTION_SPEC_BY_ID:
        direct_action_id = None
    return {
        "id": command_id,
        "label": label,
        "spoken_phrases": spoken_phrases,
        "action_type": action_type,
        "macro_keys": macro_keys if action_type == "macro" else [],
        "direct_action_id": direct_action_id if action_type == "direct_action" else None,
        "enabled": bool(command.get("enabled", True)),
    }


def get_builtin_aliases(settings: dict | None = None) -> dict[str, list[str]]:
    normalized_settings = normalize_voice_settings(settings or {})
    aliases = {}
    overrides = normalized_settings.get("voice_builtin_alias_overrides", {})
    for spec in BUILTIN_COMMAND_SPECS:
        aliases[spec["id"]] = normalize_phrase_list(overrides.get(spec["id"], spec["default_phrases"]))
    return aliases


def get_builtin_command_entries(settings: dict | None = None) -> list[dict]:
    aliases = get_builtin_aliases(settings)
    entries = []
    for spec in BUILTIN_COMMAND_SPECS:
        entries.append(
            {
                "kind": "builtin",
                "id": spec["id"],
                "label": spec["label"],
                "description": spec["description"],
                "spoken_phrases": aliases[spec["id"]],
                "action_type": "direct_action",
                "macro_keys": [],
                "direct_action_id": spec["direct_action_id"],
                "enabled": True,
            }
        )
    return entries


def get_custom_command_entries(settings: dict | None = None, *, enabled_only: bool = False) -> list[dict]:
    normalized_settings = normalize_voice_settings(settings or {})
    entries = []
    for command in normalized_settings.get("voice_custom_commands", []):
        if enabled_only and not command.get("enabled", True):
            continue
        entry = deepcopy(command)
        entry["kind"] = "custom"
        if entry["action_type"] == "direct_action" and entry.get("direct_action_id") in DIRECT_ACTION_SPEC_BY_ID:
            entry["description"] = DIRECT_ACTION_SPEC_BY_ID[entry["direct_action_id"]]["description"]
        else:
            entry["description"] = f"Macro de teclado: {format_macro_keys(entry.get('macro_keys', []))}"
        entries.append(entry)
    return entries


def get_active_command_entries(settings: dict | None = None) -> list[dict]:
    normalized_settings = normalize_voice_settings(settings or {})
    entries = get_builtin_command_entries(normalized_settings)
    if not normalized_settings.get("voice_commands_enabled", True):
        return entries
    entries.extend(get_custom_command_entries(normalized_settings, enabled_only=True))
    return entries


def get_all_command_entries(settings: dict | None = None) -> list[dict]:
    normalized_settings = normalize_voice_settings(settings or {})
    entries = get_builtin_command_entries(normalized_settings)
    entries.extend(get_custom_command_entries(normalized_settings, enabled_only=False))
    return entries


def find_phrase_conflicts(
    phrases: list[str],
    settings: dict | None = None,
    *,
    skip_kind: str | None = None,
    skip_id: str | None = None,
) -> list[str]:
    candidates = {normalize_text(phrase): phrase for phrase in normalize_phrase_list(phrases)}
    conflicts = []
    for entry in get_all_command_entries(settings):
        if skip_kind == entry["kind"] and skip_id == entry["id"]:
            continue
        for phrase in entry.get("spoken_phrases", []):
            normalized = normalize_text(phrase)
            if normalized in candidates:
                conflicts.append(entry["label"])
                break
    return sorted(set(conflicts))


def _alias_match_score(normalized_text: str, alias_norm: str) -> float:
    if not alias_norm or not normalized_text:
        return 0.0
    if normalized_text == alias_norm:
        return 1.0
    score = SequenceMatcher(None, normalized_text, alias_norm).ratio()
    text_words = normalized_text.split()
    if NEGATION_TOKENS & set(text_words):
        return min(score, 0.30)
    if re.search(rf"(?:^|\s){re.escape(alias_norm)}(?:\s|$)", normalized_text):
        coverage = len(alias_norm.split()) / max(len(text_words), 1)
        if coverage >= MIN_ALIAS_WORD_COVERAGE:
            return max(score, 0.95)
    return score


def resolve_command_entry(text: str, settings: dict | None = None) -> tuple[dict | None, float]:
    normalized = normalize_text(text)
    best_entry = None
    best_score = 0.0
    for entry in get_active_command_entries(settings):
        for alias in entry.get("spoken_phrases", []):
            score = _alias_match_score(normalized, normalize_text(alias))
            if score > best_score:
                best_entry = entry
                best_score = score
    return best_entry, best_score


def resolve_command(text: str, settings: dict | None = None) -> tuple[str | None, float]:
    entry, score = resolve_command_entry(text, settings)
    return (entry["id"] if entry else None), score


def get_direct_action_specs() -> list[dict]:
    return deepcopy(DIRECT_ACTION_SPECS)


def get_voice_help_rows(settings: dict | None = None) -> list[dict]:
    rows = []
    for entry in get_active_command_entries(settings):
        row = deepcopy(entry)
        row["spoken_examples"] = entry.get("spoken_phrases", [])
        if row["action_type"] == "macro":
            row["description"] = f"Macro: {format_macro_keys(row.get('macro_keys', []))}"
        rows.append(row)
    return rows

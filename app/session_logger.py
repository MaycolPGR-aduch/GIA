from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config_store import session_log_path
from .models import SessionEvent


class SessionLogger:
    def __init__(self, profile_name: str):
        self.profile_name = profile_name
        self.log_path = session_log_path(profile_name)
        self.events: list[dict] = []

    def log(self, event_type: str, detail: str, **payload) -> None:
        event = SessionEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            detail=detail,
            payload=payload,
        )
        event_dict = {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "detail": event.detail,
            **event.payload,
        }
        self.events.append(event_dict)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event_dict, ensure_ascii=False) + "\n")

    def export_excel(self, destination: Path | None = None) -> Path | None:
        if not self.events:
            return None
        destination = destination or self.log_path.with_suffix(".xlsx")
        df = pd.DataFrame(self.events)
        df.to_excel(destination, index=False)
        return destination

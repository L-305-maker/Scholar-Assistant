from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from scholar_assistant.schemas.events import RunEvent


class EventSink:
    def __init__(self, events_path: Path | None = None) -> None:
        self.events_path = events_path
        self.events: list[RunEvent] = []
        if self.events_path:
            self.events_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: RunEvent) -> RunEvent:
        self.events.append(event)
        if self.events_path:
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
        return event

    def extend(self, events: Iterable[RunEvent]) -> None:
        for event in events:
            self.emit(event)


def event_to_json_line(event: RunEvent) -> str:
    return json.dumps(event.model_dump(mode="json"), ensure_ascii=False)

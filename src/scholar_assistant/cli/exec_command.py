from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scholar_assistant.core.events import event_to_json_line
from scholar_assistant.core.orchestrator import ResearchOrchestrator
from scholar_assistant.schemas.events import RunEvent, RunEventType

RESEARCH_TERMS = ["research", "search", "paper", "论文", "文献", "调研", "检索", "研究"]


async def execute_task(
    task: str,
    *,
    project_path: Path,
    json_mode: bool = False,
    output_schema_path: Path | None = None,
    ephemeral: bool = False,
    no_embeddings: bool = False,
) -> tuple[int, str]:
    if output_schema_path:
        schema = json.loads(output_schema_path.read_text(encoding="utf-8"))
    else:
        schema = None

    if ephemeral:
        events = _ephemeral_events(
            task,
            {
                "mode": "ephemeral",
                "result": "Ephemeral execution does not write project state in the MVP.",
                "output_schema": schema,
            },
        )
        if json_mode:
            return 0, "\n".join(event_to_json_line(event) for event in events) + "\n"
        return 0, "Ephemeral result: no project state was written.\n"

    if _looks_like_research_task(task):
        orchestrator = ResearchOrchestrator(project_path)
        result = await orchestrator.run_research(task, no_embeddings=no_embeddings)
        if json_mode:
            events = (result.run_path / "events.jsonl").read_text(encoding="utf-8")
            return 0, events
        payload: dict[str, Any] = result.summary()
        if schema is not None:
            payload = {"result": payload, "requested_output_schema": schema}
        return 0, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    payload = {
        "task": task,
        "result": "MVP exec handled this as a local project-analysis task.",
        "project_path": str(project_path.resolve()),
        "output_schema": schema,
        "note": "No arbitrary shell execution is allowed.",
    }
    if json_mode:
        events = _ephemeral_events(task, payload)
        return 0, "\n".join(event_to_json_line(event) for event in events) + "\n"
    return 0, json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _looks_like_research_task(task: str) -> bool:
    lower = task.lower()
    return any(term in lower for term in RESEARCH_TERMS)


def _ephemeral_events(task: str, result_payload: dict[str, Any]) -> list[RunEvent]:
    run = RunEvent.new(RunEventType.RUN_STARTED, payload={"task": task})
    return [
        run,
        RunEvent.new(
            RunEventType.TASK_STARTED,
            run_id=run.run_id,
            task_id="exec",
            payload={"task_type": "exec", "assigned_role": "Exec"},
        ),
        RunEvent.new(
            RunEventType.TASK_COMPLETED,
            run_id=run.run_id,
            task_id="exec",
            payload={"task_type": "exec", "assigned_role": "Exec", "task": task},
        ),
        RunEvent.new(
            RunEventType.RUN_COMPLETED,
            run_id=run.run_id,
            payload={"task": task, **result_payload},
        ),
    ]

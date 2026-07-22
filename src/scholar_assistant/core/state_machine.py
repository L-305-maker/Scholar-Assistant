from __future__ import annotations

from enum import StrEnum


class ResearchState(StrEnum):
    CREATED = "CREATED"
    SCOPING = "SCOPING"
    SEARCH_PLANNING = "SEARCH_PLANNING"
    SEARCHING = "SEARCHING"
    SCREENING = "SCREENING"
    READING = "READING"
    ANALYZING = "ANALYZING"
    VERIFYING = "VERIFYING"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIALLY_COMPLETED = "PARTIALLY_COMPLETED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    BLOCKED = "BLOCKED"


ALLOWED_TRANSITIONS: dict[ResearchState, set[ResearchState]] = {
    ResearchState.CREATED: {ResearchState.SCOPING, ResearchState.FAILED},
    ResearchState.SCOPING: {ResearchState.SEARCH_PLANNING, ResearchState.FAILED},
    ResearchState.SEARCH_PLANNING: {ResearchState.SEARCHING, ResearchState.FAILED},
    ResearchState.SEARCHING: {
        ResearchState.SCREENING,
        ResearchState.PARTIALLY_COMPLETED,
        ResearchState.FAILED,
        ResearchState.BUDGET_EXHAUSTED,
    },
    ResearchState.SCREENING: {ResearchState.READING, ResearchState.PARTIALLY_COMPLETED},
    ResearchState.READING: {ResearchState.ANALYZING, ResearchState.PARTIALLY_COMPLETED},
    ResearchState.ANALYZING: {ResearchState.VERIFYING, ResearchState.PARTIALLY_COMPLETED},
    ResearchState.VERIFYING: {ResearchState.REPORTING, ResearchState.PARTIALLY_COMPLETED},
    ResearchState.REPORTING: {ResearchState.COMPLETED, ResearchState.PARTIALLY_COMPLETED},
    ResearchState.COMPLETED: set(),
    ResearchState.FAILED: set(),
    ResearchState.PARTIALLY_COMPLETED: set(),
    ResearchState.BUDGET_EXHAUSTED: set(),
    ResearchState.BLOCKED: set(),
}


class InvalidTransitionError(ValueError):
    pass


class ResearchStateMachine:
    def __init__(self, state: ResearchState = ResearchState.CREATED) -> None:
        self.state = state
        self.history: list[ResearchState] = [state]

    def transition(self, next_state: ResearchState) -> ResearchState:
        allowed = ALLOWED_TRANSITIONS[self.state]
        if next_state not in allowed:
            msg = f"Invalid transition from {self.state} to {next_state}"
            raise InvalidTransitionError(msg)
        self.state = next_state
        self.history.append(next_state)
        return self.state

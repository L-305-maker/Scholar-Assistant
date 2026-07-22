"""Pydantic schemas used across Scholar-Assistant."""

from scholar_assistant.schemas.evidence import Claim, EvidenceUnit, Hypothesis
from scholar_assistant.schemas.paper import Paper, PaperVersion, ScholarlyWork
from scholar_assistant.schemas.research import ResearchProject
from scholar_assistant.schemas.task import Task

__all__ = [
    "Claim",
    "EvidenceUnit",
    "Hypothesis",
    "Paper",
    "PaperVersion",
    "ResearchProject",
    "ScholarlyWork",
    "Task",
]

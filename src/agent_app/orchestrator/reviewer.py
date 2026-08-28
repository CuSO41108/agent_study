from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from agent_app.types import ToolResult


ReviewDecision = Literal["accepted", "rejected", "blocked"]


@dataclass(frozen=True, slots=True)
class WorkerReviewEvidence:
    """Bounded, read-only evidence supplied to a worker reviewer."""

    task: str
    success_criteria: str
    child_session_id: str
    worker_success: bool
    final_text: str | None
    stop_reason: str | None
    tool_runs: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ReviewResult:
    decision: ReviewDecision
    feedback: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    summary: str = ""

    @property
    def accepted(self) -> bool:
        return self.decision == "accepted"


class WorkerReviewer:
    """Run a read-only model review over bounded Worker evidence.

    The reviewer receives no Function Calling tools and therefore cannot edit
    files, run commands, or approve side effects. Invalid reviewer output is
    fail-closed as ``blocked``.
    """

    def __init__(self, model_client: Any) -> None:
        self._model_client = model_client

    def review(self, evidence: WorkerReviewEvidence) -> ReviewResult:
        response = self._model_client.generate(
            system_prompt=(
                "You are a read-only coding-task reviewer. Return only one JSON object "
                "with decision, feedback, evidence_refs, and summary. decision must be "
                "accepted, rejected, or blocked. Accept only when the Worker evidence "
                "satisfies every success criterion. You have no tools and must not invent "
                "files, commands, tests, or results. Use blocked when the evidence is "
                "insufficient or contradictory."
            ),
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(_evidence_payload(evidence), ensure_ascii=False),
                }
            ],
            tools=[],
        )
        if getattr(response, "error_type", None):
            return ReviewResult(
                decision="blocked",
                feedback=(f"Reviewer model failed: {response.error_type}",),
            )
        text = getattr(response, "assistant_text", None)
        if not isinstance(text, str) or not text.strip():
            return ReviewResult(decision="blocked", feedback=("Reviewer returned no decision.",))
        try:
            payload = _decode_json_object(text)
            return _parse_review_result(payload)
        except (TypeError, ValueError, KeyError) as exc:
            return ReviewResult(decision="blocked", feedback=(f"Invalid reviewer output: {exc}",))


def _evidence_payload(evidence: WorkerReviewEvidence) -> dict[str, Any]:
    return {
        "task": evidence.task,
        "success_criteria": evidence.success_criteria,
        "child_session_id": evidence.child_session_id,
        "worker_success": evidence.worker_success,
        "final_text": evidence.final_text,
        "stop_reason": evidence.stop_reason,
        "tool_runs": list(evidence.tool_runs),
    }


def tool_results_to_evidence(tool_runs: Sequence[ToolResult]) -> tuple[dict[str, Any], ...]:
    """Bound tool outputs before they are sent to the reviewer model."""

    return tuple(
        {
            "tool_call_id": result.tool_call_id,
            "tool_name": result.tool_name,
            "success": result.success,
            "content_preview": result.content[:1000],
            "error": result.error[:500] if result.error else None,
        }
        for result in tool_runs
    )


def _parse_review_result(payload: Any) -> ReviewResult:
    if not isinstance(payload, dict):
        raise TypeError("review must be an object")
    decision = payload.get("decision")
    if decision not in {"accepted", "rejected", "blocked"}:
        raise ValueError("decision must be accepted, rejected, or blocked")
    feedback = _string_tuple(payload.get("feedback", []), field="feedback")
    evidence_refs = _string_tuple(payload.get("evidence_refs", []), field="evidence_refs")
    summary = payload.get("summary", "")
    if not isinstance(summary, str):
        raise TypeError("summary must be a string")
    return ReviewResult(
        decision=decision,
        feedback=feedback,
        evidence_refs=evidence_refs,
        summary=summary.strip(),
    )


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise TypeError(f"{field} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _decode_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    decoder = json.JSONDecoder()
    for index, character in enumerate(candidate):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("response does not contain a JSON object")

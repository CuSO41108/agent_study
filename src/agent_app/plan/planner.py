from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from agent_app.plan.graph import PlanGraph, parse_plan_graph, plan_graph_to_dict
from agent_app.plan.store import PlanRevision


class PlanPlanningError(ValueError):
    """Raised when the model does not return a safe structured plan."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str | None = None,
        detail: str | None = None,
        attempts: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.detail = detail
        self.attempts = attempts


PlannerAttemptHook = Callable[[dict[str, Any]], None]


class PlanPlanner:
    """Turn a user goal into a validated PlanGraph without executing tools."""

    def __init__(
        self,
        model_client: Any,
        *,
        max_request_retries: int = 2,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 4.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_request_retries < 0:
            raise ValueError("max_request_retries cannot be negative.")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay cannot be negative.")
        if retry_max_delay < 0:
            raise ValueError("retry_max_delay cannot be negative.")
        self._model_client = model_client
        self._max_request_retries = max_request_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._sleep = sleep

    @property
    def max_request_retries(self) -> int:
        return self._max_request_retries

    @property
    def max_request_attempts(self) -> int:
        return self._max_request_retries + 1

    def create_plan(
        self,
        goal: str,
        *,
        on_attempt: PlannerAttemptHook | None = None,
    ) -> PlanGraph:
        if not goal.strip():
            raise ValueError("Plan goal cannot be empty.")
        response, attempts = self._generate_with_retry(
            phase="initial_plan",
            on_attempt=on_attempt,
            system_prompt=(
                "You are a conservative coding-task planner. Return only one JSON object, "
                "with no Markdown and no commentary. The object must contain id, revision=1, "
                "goal, and nodes. Each node must use exactly one kind: inspect, edit, run, "
                "or verify. Each node must include objective, depends_on, allowed_tools, "
                "acceptance, and status=pending. It may include resources, an array of "
                "{key,mode} claims where key is workspace or file:<workspace-relative-path> "
                "and mode is read, write, or exclusive. Only declare a narrower resource "
                "when it is known; omitted resources use a conservative kind-based fallback. "
                "Use a static acyclic dependency graph. "
                "Allowed tools by kind are: inspect=file_read,code_search,web_search,skill_list,"
                "skill_load,skill_read_resource; edit=file_read,code_search,replace_in_file,file_write; "
                "run=shell; verify=file_read,code_search,shell. Keep the plan as small as possible."
            ),
            messages=[{"role": "user", "content": goal}],
            tools=[],
        )
        if getattr(response, "error_type", None):
            raise _model_failure("Planner", response, attempts=attempts)
        if getattr(response, "tool_calls", None):
            raise PlanPlanningError(
                "Planner must return JSON without tool calls.",
                error_type="unexpected_tool_calls",
                attempts=attempts,
            )
        text = getattr(response, "assistant_text", None)
        if not isinstance(text, str) or not text.strip():
            raise PlanPlanningError(
                "Planner returned no JSON plan.",
                error_type="empty_response",
                attempts=attempts,
            )

        payload = _decode_json_object(text, attempts=attempts)
        payload.setdefault("id", f"plan-{uuid4().hex}")
        payload.setdefault("revision", 1)
        payload["goal"] = goal.strip()
        nodes = payload.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, dict):
                    node.setdefault("status", "pending")
        try:
            return parse_plan_graph(payload)
        except (TypeError, KeyError, ValueError) as exc:
            raise PlanPlanningError(
                str(exc),
                error_type="invalid_plan",
                attempts=attempts,
            ) from exc

    def create_replan(
        self,
        *,
        current: PlanRevision,
        reason: str,
        on_attempt: PlannerAttemptHook | None = None,
    ) -> PlanGraph:
        """Ask the model for only a successor graph; completed nodes are preserved by PlanStore."""

        response, attempts = self._generate_with_retry(
            phase="replan",
            on_attempt=on_attempt,
            system_prompt=(
                "You are revising a failed coding-task plan. Return only one JSON object. "
                "Keep completed nodes unchanged, repair or replace only unfinished work, "
                "and use a static acyclic graph with the same plan id and a higher revision. "
                "Use only node kinds inspect, edit, run, verify and their allowed tool boundaries. "
                "Preserve or conservatively update optional resource claims using {key,mode}; "
                "never infer parallel safety for an unknown side effect."
            ),
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "goal": current.graph.goal,
                            "current_plan": plan_graph_to_dict(current.graph),
                            "node_results": current.node_results,
                            "reason": reason,
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            tools=[],
        )
        if getattr(response, "error_type", None):
            raise _model_failure("Replanner", response, attempts=attempts)
        if getattr(response, "tool_calls", None):
            raise PlanPlanningError(
                "Replanner must return JSON without tool calls.",
                error_type="unexpected_tool_calls",
                attempts=attempts,
            )
        text = getattr(response, "assistant_text", None)
        if not isinstance(text, str) or not text.strip():
            raise PlanPlanningError(
                "Replanner returned no JSON plan.",
                error_type="empty_response",
                attempts=attempts,
            )
        payload = _decode_json_object(text, attempts=attempts)
        payload["id"] = current.graph.id
        payload["revision"] = current.graph.revision + 1
        payload["goal"] = current.graph.goal
        nodes = payload.get("nodes")
        if isinstance(nodes, list):
            for node in nodes:
                if isinstance(node, dict):
                    node.setdefault("status", "pending")
        try:
            return parse_plan_graph(payload)
        except (TypeError, KeyError, ValueError) as exc:
            raise PlanPlanningError(
                str(exc),
                error_type="invalid_plan",
                attempts=attempts,
            ) from exc

    def _generate_with_retry(
        self,
        *,
        phase: str,
        on_attempt: PlannerAttemptHook | None,
        **request: Any,
    ) -> tuple[Any, int]:
        max_attempts = self.max_request_attempts
        for attempt in range(1, max_attempts + 1):
            response = self._model_client.generate(**request)
            error_type = getattr(response, "error_type", None)
            if not error_type:
                if on_attempt is not None:
                    on_attempt(
                        {
                            "phase": phase,
                            "attempt": attempt,
                            "max_attempts": max_attempts,
                            "request_status": "succeeded",
                            "error_type": None,
                            "error_detail": None,
                            "retryable": False,
                            "retry_delay_seconds": 0.0,
                        }
                    )
                return response, attempt

            detail = _response_error_detail(response)
            retryable = error_type == "request_error" and attempt < max_attempts
            delay = self._retry_delay(attempt) if retryable else 0.0
            if on_attempt is not None:
                on_attempt(
                    {
                        "phase": phase,
                        "attempt": attempt,
                        "max_attempts": max_attempts,
                        "request_status": "retrying" if retryable else "failed",
                        "error_type": str(error_type),
                        "error_detail": detail,
                        "retryable": retryable,
                        "retry_delay_seconds": delay,
                    }
                )
            if not retryable:
                return response, attempt
            self._sleep(delay)
        raise AssertionError("Planner retry loop exhausted without a response.")

    def _retry_delay(self, attempt: int) -> float:
        return min(
            self._retry_max_delay,
            self._retry_base_delay * (2 ** (attempt - 1)),
        )


def _decode_json_object(text: str, *, attempts: int | None = None) -> dict[str, Any]:
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
    raise PlanPlanningError(
        "Planner response does not contain a JSON object.",
        error_type="invalid_plan",
        attempts=attempts,
    )


_SECRET_PATTERNS = (
    re.compile(
        r'''(?i)(["']?authorization["']?\s*[:=]\s*["']?bearer\s+)[^"'\s,;}]+'''
    ),
    re.compile(
        r'''(?i)(["']?(?:api[_-]?key|token|password|secret)["']?\s*[:=]\s*["']?)[^"'\s,;}]+'''
    ),
    re.compile(r"\bsk-[A-Za-z0-9_-]+\b"),
)


def _response_error_detail(response: Any) -> str | None:
    raw_response = getattr(response, "raw_response", None)
    if not isinstance(raw_response, dict):
        return None
    detail = raw_response.get("detail")
    if detail is None:
        status = raw_response.get("status")
        body = raw_response.get("body")
        if status is not None or body is not None:
            detail = f"HTTP {status}: {body}" if status is not None else body
    if detail is None:
        return None
    sanitized = str(detail).strip()
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub(r"\1[REDACTED]" if pattern.groups else "[REDACTED]", sanitized)
    if len(sanitized) > 500:
        sanitized = sanitized[:497] + "..."
    return sanitized or None


def _model_failure(label: str, response: Any, *, attempts: int) -> PlanPlanningError:
    error_type = str(getattr(response, "error_type", "model_error"))
    detail = _response_error_detail(response)
    message = f"{label} model failed: {error_type}"
    if detail:
        message += f" ({detail})"
    message += f" after {attempts} attempt{'s' if attempts != 1 else ''}."
    return PlanPlanningError(
        message,
        error_type=error_type,
        detail=detail,
        attempts=attempts,
    )

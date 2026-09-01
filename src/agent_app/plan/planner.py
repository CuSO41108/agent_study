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
        recovery_task_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.detail = detail
        self.attempts = attempts
        self.recovery_task_id = recovery_task_id


PlannerAttemptHook = Callable[[dict[str, Any]], None]


_ARRAY_CONTRACT = (
    "The top-level nodes field must be a non-empty JSON array. For every node, "
    "depends_on and allowed_tools must be JSON arrays, and acceptance must be a "
    "non-empty JSON array of non-empty strings even when there is only one condition. "
    "Never return acceptance as a string. If resources is present, it must be a JSON array. "
)


class PlanPlanner:
    """Turn a user goal into a validated PlanGraph without executing tools."""

    def __init__(
        self,
        model_client: Any,
        *,
        max_request_retries: int = 2,
        max_format_repairs: int = 1,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 4.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_request_retries < 0:
            raise ValueError("max_request_retries cannot be negative.")
        if max_format_repairs < 0:
            raise ValueError("max_format_repairs cannot be negative.")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay cannot be negative.")
        if retry_max_delay < 0:
            raise ValueError("retry_max_delay cannot be negative.")
        self._model_client = model_client
        self._max_request_retries = max_request_retries
        self._max_format_repairs = max_format_repairs
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._sleep = sleep

    @property
    def max_request_retries(self) -> int:
        return self._max_request_retries

    @property
    def max_request_attempts(self) -> int:
        return self._max_request_retries + 1

    @property
    def max_attempts(self) -> int:
        """Maximum model calls across transient retries and bounded format repair."""

        return self.max_request_attempts + self._max_format_repairs

    def create_plan(
        self,
        goal: str,
        *,
        on_attempt: PlannerAttemptHook | None = None,
    ) -> PlanGraph:
        if not goal.strip():
            raise ValueError("Plan goal cannot be empty.")
        normalized_goal = goal.strip()

        def prepare_payload(payload: dict[str, Any]) -> None:
            payload.setdefault("id", f"plan-{uuid4().hex}")
            payload.setdefault("revision", 1)
            payload["goal"] = normalized_goal
            _default_pending_node_statuses(payload)

        return self._create_validated_plan(
            label="Planner",
            phase="initial_plan",
            on_attempt=on_attempt,
            system_prompt=(
                "You are a conservative coding-task planner. Return only one JSON object, "
                "with no Markdown and no commentary. The object must contain id, revision=1, "
                "goal, and nodes. Each node must use exactly one kind: inspect, edit, run, "
                "or verify. Each node must include objective, depends_on, allowed_tools, "
                "acceptance, and status=pending. "
                f"{_ARRAY_CONTRACT}"
                "It may include resources, an array of "
                "{key,mode} claims where key is workspace or file:<workspace-relative-path> "
                "and mode is read, write, or exclusive. Only declare a narrower resource "
                "when it is known; omitted resources use a conservative kind-based fallback. "
                "Use a static acyclic dependency graph. "
                "Allowed tools by kind are: inspect=file_read,code_search,web_search,skill_list,"
                "skill_load,skill_read_resource; edit=file_read,code_search,replace_in_file,file_write; "
                "run=shell; verify=file_read,code_search,shell. Keep the plan as small as possible."
            ),
            messages=[{"role": "user", "content": normalized_goal}],
            prepare_payload=prepare_payload,
        )

    def create_replan(
        self,
        *,
        current: PlanRevision,
        reason: str,
        on_attempt: PlannerAttemptHook | None = None,
    ) -> PlanGraph:
        """Ask the model for only a successor graph; completed nodes are preserved by PlanStore."""

        def prepare_payload(payload: dict[str, Any]) -> None:
            payload["id"] = current.graph.id
            payload["revision"] = current.graph.revision + 1
            payload["goal"] = current.graph.goal
            _default_pending_node_statuses(payload)

        return self._create_validated_plan(
            label="Replanner",
            phase="replan",
            on_attempt=on_attempt,
            system_prompt=(
                "You are revising a failed coding-task plan. Return only one JSON object. "
                "Keep completed nodes unchanged, repair or replace only unfinished work, "
                "and use a static acyclic graph with the same plan id and a higher revision. "
                "Use only node kinds inspect, edit, run, verify and their allowed tool boundaries. "
                f"{_ARRAY_CONTRACT}"
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
            prepare_payload=prepare_payload,
        )

    def _create_validated_plan(
        self,
        *,
        label: str,
        phase: str,
        on_attempt: PlannerAttemptHook | None,
        system_prompt: str,
        messages: list[dict[str, str]],
        prepare_payload: Callable[[dict[str, Any]], None],
    ) -> PlanGraph:
        active_messages = list(messages)
        attempts = 0
        request_retries_used = 0
        format_repairs_used = 0

        while True:
            response, attempts, request_retries_used = self._generate_with_retry(
                label=label,
                phase=phase,
                on_attempt=on_attempt,
                attempt_offset=attempts,
                request_retries_used=request_retries_used,
                system_prompt=system_prompt,
                messages=active_messages,
                tools=[],
            )
            text = getattr(response, "assistant_text", None)
            try:
                graph = self._parse_plan_response(
                    label=label,
                    response=response,
                    attempts=attempts,
                    prepare_payload=prepare_payload,
                )
            except PlanPlanningError as exc:
                can_repair = (
                    exc.error_type == "invalid_plan"
                    and isinstance(text, str)
                    and bool(text.strip())
                    and format_repairs_used < self._max_format_repairs
                )
                self._notify_attempt(
                    on_attempt,
                    phase=phase,
                    attempt=attempts,
                    request_status="repairing" if can_repair else "failed",
                    error_type=exc.error_type,
                    error_detail=str(exc),
                    retryable=can_repair,
                )
                if not can_repair:
                    raise PlanPlanningError(
                        str(exc),
                        error_type=exc.error_type,
                        detail=exc.detail,
                        attempts=attempts,
                    ) from exc

                format_repairs_used += 1
                active_messages.extend(
                    [
                        {"role": "assistant", "content": text},
                        {
                            "role": "user",
                            "content": (
                                "The previous PlanGraph JSON failed validation:\n"
                                f"{exc}\n"
                                "Return the complete corrected JSON object only. Preserve the goal "
                                "and intended node semantics. Ensure nodes is an array; depends_on and "
                                "allowed_tools are arrays; acceptance is a non-empty array of strings, "
                                "never a string; and resources, when present, is an array."
                            ),
                        },
                    ]
                )
                continue

            self._notify_attempt(
                on_attempt,
                phase=phase,
                attempt=attempts,
                request_status="succeeded",
                error_type=None,
                error_detail=None,
                retryable=False,
            )
            return graph

    @staticmethod
    def _parse_plan_response(
        *,
        label: str,
        response: Any,
        attempts: int,
        prepare_payload: Callable[[dict[str, Any]], None],
    ) -> PlanGraph:
        if getattr(response, "tool_calls", None):
            raise PlanPlanningError(
                f"{label} must return JSON without tool calls.",
                error_type="unexpected_tool_calls",
                attempts=attempts,
            )
        text = getattr(response, "assistant_text", None)
        if not isinstance(text, str) or not text.strip():
            raise PlanPlanningError(
                f"{label} returned no JSON plan.",
                error_type="empty_response",
                attempts=attempts,
            )
        payload = _decode_json_object(text, attempts=attempts)
        prepare_payload(payload)
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
        label: str,
        phase: str,
        on_attempt: PlannerAttemptHook | None,
        attempt_offset: int,
        request_retries_used: int,
        **request: Any,
    ) -> tuple[Any, int, int]:
        remaining_retries = self._max_request_retries - request_retries_used
        for local_attempt in range(1, remaining_retries + 2):
            attempt = attempt_offset + local_attempt
            response = self._model_client.generate(**request)
            error_type = getattr(response, "error_type", None)
            if not error_type:
                return response, attempt, request_retries_used

            detail = _response_error_detail(response)
            retryable = (
                error_type == "request_error"
                and request_retries_used < self._max_request_retries
            )
            if retryable:
                request_retries_used += 1
            delay = self._retry_delay(request_retries_used) if retryable else 0.0
            self._notify_attempt(
                on_attempt,
                phase=phase,
                attempt=attempt,
                request_status="retrying" if retryable else "failed",
                error_type=str(error_type),
                error_detail=detail,
                retryable=retryable,
                retry_delay_seconds=delay,
            )
            if not retryable:
                raise _model_failure(label, response, attempts=attempt)
            self._sleep(delay)
        raise AssertionError("Planner retry loop exhausted without a response.")

    def _notify_attempt(
        self,
        on_attempt: PlannerAttemptHook | None,
        *,
        phase: str,
        attempt: int,
        request_status: str,
        error_type: str | None,
        error_detail: str | None,
        retryable: bool,
        retry_delay_seconds: float = 0.0,
    ) -> None:
        if on_attempt is None:
            return
        on_attempt(
            {
                "phase": phase,
                "attempt": attempt,
                "max_attempts": self.max_attempts,
                "request_status": request_status,
                "error_type": error_type,
                "error_detail": error_detail,
                "retryable": retryable,
                "retry_delay_seconds": retry_delay_seconds,
            }
        )

    def _retry_delay(self, attempt: int) -> float:
        return min(
            self._retry_max_delay,
            self._retry_base_delay * (2 ** (attempt - 1)),
        )


def _default_pending_node_statuses(payload: dict[str, Any]) -> None:
    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if isinstance(node, dict):
            node.setdefault("status", "pending")


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

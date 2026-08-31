from __future__ import annotations

import json
from typing import Any

from agent_app.state.session_service import SessionService


TRACE_SCHEMA_VERSION = 1
REPLAY_SCHEMA_VERSION = 1
_REPLAY_MODES = {"audit", "dry"}
_TRANSIENT_ACTION_STATUSES = {"prepared", "executing", "uncertain"}


class ReplayModeError(ValueError):
    """Raised when a replay mode is unsupported or would be unsafe."""


def export_task_trace(sessions: SessionService, task_id: str) -> dict[str, Any]:
    task = sessions.get_task(task_id)
    if task is None:
        raise KeyError(task_id)
    return {
        "schema_version": TRACE_SCHEMA_VERSION,
        "trace_id": task.id,
        "session_id": task.session_id,
        "task": {
            "goal": task.goal,
            "status": task.status,
            "stop_reason": task.stop_reason,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "budget": task.budget,
        },
        "events": [
            {
                "event_id": trace.id,
                "event_type": trace.trace_type,
                "created_at": trace.created_at,
                "attributes": trace.payload,
            }
            for trace in sessions.list_task_traces(task_id)
        ],
    }


def replay_task_trace(
    sessions: SessionService,
    task_id: str,
    *,
    mode: str = "audit",
) -> dict[str, Any]:
    """Audit or dry-replay one persisted task without invoking any runtime.

    ``audit`` checks the durable trace against persisted side-effect actions.
    ``dry`` produces the same deterministic event walk while explicitly
    marking every tool operation as skipped.  Neither mode calls a model,
    invokes a Tool, or writes to SQLite.  Live reruns are deliberately not a
    replay operation: callers must use the explicit recovery/approval paths.
    """

    normalized_mode = mode.strip().casefold()
    if normalized_mode not in _REPLAY_MODES:
        raise ReplayModeError(
            f"Unsupported replay mode '{mode}'. Use 'audit' or 'dry'; "
            "live reruns require an explicit recovery and approval flow."
        )

    trace = export_task_trace(sessions, task_id)
    actions = [
        action
        for action in sessions.list_tool_actions(trace["session_id"])
        if action.task_id == task_id
    ]
    findings = _audit_replay_facts(trace, actions)
    tool_attempts = [
        event
        for event in trace["events"]
        if event["event_type"] == "tool_attempt"
    ]
    steps = _build_replay_steps(trace["events"], mode=normalized_mode)
    side_effect_count = sum(
        1
        for event in tool_attempts
        if bool(event["attributes"].get("side_effect"))
    )
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "trace_schema_version": trace["schema_version"],
        "replay_mode": normalized_mode,
        "task_id": task_id,
        "session_id": trace["session_id"],
        "task_status": trace["task"]["status"],
        "read_only": True,
        "execution_performed": False,
        "result": "passed" if not findings else "attention_required",
        "event_count": len(trace["events"]),
        "tool_attempt_count": len(tool_attempts),
        "persisted_action_count": len(actions),
        "side_effect_attempt_count": side_effect_count,
        "findings": findings,
        "steps": steps,
        "actions": [_replay_action_summary(action) for action in actions],
    }


def _audit_replay_facts(trace: dict[str, Any], actions: list[Any]) -> list[str]:
    findings: list[str] = []
    events = trace["events"]
    event_ids = [event.get("event_id") for event in events]
    if event_ids != sorted(event_ids):
        findings.append("TaskTrace event ids are not monotonically ordered.")
    if len(event_ids) != len(set(event_ids)):
        findings.append("TaskTrace contains duplicate event ids.")

    actions_by_call_id = {action.tool_call_id: action for action in actions}
    attempted_call_ids: set[str] = set()
    for event in events:
        if not isinstance(event.get("attributes"), dict):
            findings.append(f"Event {event.get('event_id', '?')} has invalid attributes.")
            continue
        if event.get("event_type") != "tool_attempt":
            continue
        attributes = event["attributes"]
        tool_name = attributes.get("tool")
        if not isinstance(tool_name, str) or not tool_name.strip():
            findings.append(f"Tool attempt event {event.get('event_id', '?')} has no tool name.")
        call_id = attributes.get("tool_call_id")
        if isinstance(call_id, str) and call_id:
            attempted_call_ids.add(call_id)
            if attributes.get("side_effect") and call_id not in actions_by_call_id:
                findings.append(
                    f"Side-effect tool attempt '{call_id}' has no persisted ToolAction."
                )

    for action in actions:
        side_effect = bool(action.recovery_metadata.get("side_effect"))
        if side_effect and action.status in _TRANSIENT_ACTION_STATUSES:
            findings.append(
                f"Side-effect ToolAction '{action.id}' remains {action.status}; "
                "manual resolution is required before resume."
            )
        if side_effect and action.tool_call_id not in attempted_call_ids:
            findings.append(
                f"Persisted side-effect ToolAction '{action.id}' has no matching trace attempt."
            )
    return findings


def _build_replay_steps(events: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    execution_state = "observed" if mode == "audit" else "skipped"
    reason = (
        "Recorded event inspected without re-execution."
        if mode == "audit"
        else "Dry replay never invokes models or tools."
    )
    steps: list[dict[str, Any]] = []
    for sequence, event in enumerate(events, start=1):
        attributes = event.get("attributes") or {}
        step = {
            "sequence": sequence,
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "execution": execution_state,
            "reason": reason,
        }
        if event.get("event_type") == "tool_attempt":
            step.update(
                {
                    "tool": attributes.get("tool"),
                    "tool_call_id": attributes.get("tool_call_id"),
                    "outcome": "succeeded" if attributes.get("success") else "failed",
                    "side_effect": bool(attributes.get("side_effect")),
                    "attempt": attributes.get("attempt", 1),
                }
            )
        steps.append(step)
    return steps


def _replay_action_summary(action: Any) -> dict[str, Any]:
    resolution = action.resolution
    return {
        "action_id": action.id,
        "tool": action.tool_name,
        "tool_call_id": action.tool_call_id,
        "status": action.status,
        "attempt": action.attempt,
        "retry_of": action.retry_of,
        "side_effect": bool(action.recovery_metadata.get("side_effect")),
        "plan_node_id": action.recovery_metadata.get("plan_node_id"),
        "has_result": action.result is not None,
        "resolution": None
        if resolution is None
        else {
            "outcome": resolution.outcome,
            "previous_status": resolution.previous_status,
            "resolved_by": resolution.resolved_by,
        },
    }


def render_task_timeline(trace: dict[str, Any]) -> str:
    task = trace["task"]
    lines = [
        f"Trace: {trace['trace_id']}",
        f"Session: {trace['session_id']}",
        f"Task: {task['goal']}",
        f"Status: {task['status']}" + (f" ({task['stop_reason']})" if task["stop_reason"] else ""),
        "Timeline:",
    ]
    lines.extend(render_trace_events(trace["events"]))
    return "\n".join(lines)


def render_trace_events(events: list[dict[str, Any]]) -> list[str]:
    return [
        f"{_short_time(event['created_at'])}  {event['event_type']:<18} {_event_summary(event['event_type'], event['attributes'])}"
        for event in events
    ]


def _short_time(value: str) -> str:
    return value[11:19] if len(value) >= 19 else value


def _event_summary(event_type: str, attributes: dict[str, Any]) -> str:
    if event_type == "state_transition":
        return f"{attributes.get('from', '∅')} → {attributes.get('to', '?')}"
    if event_type == "model_call":
        return f"{attributes.get('phase', 'model')} / {attributes.get('model', 'unknown')} / {attributes.get('total_tokens', 0)} tokens / {attributes.get('duration_ms', 0)} ms"
    if event_type == "checkpoint" and attributes.get("phase") == "planning":
        detail = (
            f"planning / {attributes.get('request_status', '?')} / "
            f"attempt {attributes.get('attempt', '?')}/{attributes.get('max_attempts', '?')}"
        )
        error_detail = attributes.get("error_detail")
        return f"{detail} / {error_detail}" if error_detail else detail
    if event_type == "decision":
        return f"tool calls: {len(attributes.get('tool_calls', []))}"
    if event_type == "approval":
        return f"{attributes.get('tool', 'action')} / {attributes.get('decision', 'pending')}"
    if event_type == "plan_created":
        return f"revision {attributes.get('revision', '?')} / {len(attributes.get('nodes', []))} nodes"
    if event_type == "plan_node_transition":
        return (
            f"{attributes.get('node_id', 'node')} / "
            f"{attributes.get('from_status', '?')} → {attributes.get('to_status', '?')}"
        )
    if event_type == "plan_node_approval":
        return (
            f"{attributes.get('node_id', 'node')} / "
            f"{attributes.get('decision', 'pending')} / "
            f"{attributes.get('to_status', '?')}"
        )
    if event_type == "plan_node_user_message":
        return (
            f"{attributes.get('node_id', 'node')} / "
            f"user answer / {attributes.get('to_status', '?')}"
        )
    if event_type == "plan_execution":
        return (
            f"revision {attributes.get('revision', '?')} / "
            f"{attributes.get('status', '?')} / "
            f"nodes: {len(attributes.get('executed_node_ids', []))}"
        )
    if event_type == "plan_failure":
        return (
            f"{attributes.get('failure_reason', 'unknown failure')} / "
            f"failed: {', '.join(attributes.get('failed_node_ids', [])) or 'none'}"
        )
    if event_type == "plan_replan":
        return (
            f"revision {attributes.get('from_revision', '?')} → "
            f"{attributes.get('to_revision', '?')} / "
            f"{attributes.get('reason', 'replan')}"
        )
    if event_type == "plan_replan_failed":
        return (
            f"{attributes.get('error_type', 'error')} / "
            f"{attributes.get('error', 'automatic replan failed')}"
        )
    if event_type == "tool_attempt":
        outcome = "success" if attributes.get("success") else attributes.get("error_type", "failed")
        return f"{attributes.get('tool', 'tool')} / {outcome} / {attributes.get('duration_ms', 0)} ms"
    if event_type == "budget":
        budget = attributes.get("budget", {})
        return f"tools {budget.get('used_tool_calls', 0)}/{budget.get('max_tool_calls', '?')}, tokens {budget.get('used_tokens', 0)}/{budget.get('max_tokens', '?')}"
    if event_type == "observation":
        return f"{attributes.get('status', '?')} / {attributes.get('error_type') or 'ok'}"
    if event_type == "stream":
        stream_type = attributes.get("event_type", "event")
        if stream_type == "tool_output":
            return f"{attributes.get('tool', 'tool')} / {attributes.get('stream', 'output')}: {attributes.get('line', '')}"
        if stream_type == "model_text_delta":
            return f"assistant: {attributes.get('text', '')}"
        if stream_type == "action_planned":
            return f"planned: {attributes.get('tool', 'tool')}"
        return stream_type
    return json.dumps(attributes, ensure_ascii=False, sort_keys=True)[:180]

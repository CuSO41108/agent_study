from __future__ import annotations

import json
from typing import Any

from agent_app.state.session_service import SessionService


TRACE_SCHEMA_VERSION = 1


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
    if event_type == "decision":
        return f"tool calls: {len(attributes.get('tool_calls', []))}"
    if event_type == "approval":
        return f"{attributes.get('tool', 'action')} / {attributes.get('decision', 'pending')}"
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

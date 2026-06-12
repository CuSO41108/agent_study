from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class Message:
    role: str
    content: str | None
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    tool_call_id: str
    tool_name: str
    success: bool
    content: str
    error: str | None = None


ToolActionStatus = Literal["prepared", "executing", "succeeded", "failed", "uncertain"]


@dataclass(slots=True, frozen=True)
class ToolAction:
    id: str
    session_id: str
    agent_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    idempotency_key: str
    status: ToolActionStatus
    recovery_metadata: dict[str, Any]
    result: ToolResult | None
    prepared_at: str
    started_at: str | None
    completed_at: str | None
    updated_at: str


@dataclass(slots=True, frozen=True)
class StoredMessage:
    id: int
    role: str
    content: str | None
    tool_call_id: str | None = None
    tool_name: str | None = None


@dataclass(slots=True, frozen=True)
class TodoItem:
    content: str
    status: str


@dataclass(slots=True, frozen=True)
class SessionContext:
    summary_text: str | None = None
    summary_message_id: int | None = None
    todo_items: tuple[TodoItem, ...] = ()


@dataclass(slots=True, frozen=True)
class TurnTrace:
    id: int
    session_id: str
    user_input: str
    context_message_count: int
    context_token_estimate: int
    used_summary: bool
    used_todo: bool
    used_evidence: bool
    final_text: str | None
    stop_reason: str | None
    success: bool
    created_at: str


@dataclass(slots=True, frozen=True)
class ToolCallTrace:
    id: int
    turn_trace_id: int
    tool_call_id: str
    tool_name: str
    success: bool
    error: str | None
    content_preview: str
    created_at: str


@dataclass(slots=True, frozen=True)
class SubagentRun:
    parent_session_id: str
    parent_tool_call_id: str
    child_session_id: str
    agent_id: str
    task: str
    success: bool
    result_summary: str
    created_at: str


@dataclass(slots=True)
class ModelResponse:
    assistant_text: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    raw_response: dict[str, Any] | None = None
    error_type: str | None = None


@dataclass(slots=True)
class TurnResult:
    session_id: str
    final_text: str | None
    stop_reason: str | None
    tool_runs: list[ToolResult]
    success: bool

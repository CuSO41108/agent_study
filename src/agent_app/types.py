from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4


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
    observation: "Observation | None" = None


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
    task_id: str | None = None
    attempt: int = 1
    retry_of: str | None = None


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


TaskStatus = Literal[
    "created",
    "running",
    "waiting_user",
    "waiting_tool",
    "paused",
    "completed",
    "failed",
    "cancelled",
    "expired",
    "handed_off",
]
EventType = Literal[
    "task_created",
    "user_message",
    "user_approved",
    "user_rejected",
    "resume_requested",
    "pause_requested",
    "cancel_requested",
    "task_expired",
    "state_transition",
    "observation_recorded",
    "budget_updated",
    "skill_activated",
    "skill_dropped",
    "task_handed_off",
]
ObservationErrorType = Literal[
    "timeout",
    "transient",
    "invalid_arguments",
    "permission_denied",
    "not_found",
    "conflict",
    "user_denied",
    "unsafe_action",
    "runtime_error",
    "uncertain_side_effect",
]


@dataclass(slots=True, frozen=True)
class TaskBudget:
    max_model_calls: int = 24
    max_tool_calls: int = 24
    max_tokens: int = 120_000
    max_active_seconds: float = 900.0
    waiting_user_timeout_seconds: int = 86_400
    max_retries: int = 2
    max_repair_attempts: int = 2
    repeat_decision_limit: int = 3
    max_replans: int = 1
    used_model_calls: int = 0
    used_tool_calls: int = 0
    used_tokens: int = 0
    used_active_seconds: float = 0.0
    used_repair_attempts: int = 0
    used_replans: int = 0


@dataclass(slots=True, frozen=True)
class PendingAction:
    kind: Literal["ask_user", "tool_approval"]
    prompt: str
    id: str = field(default_factory=lambda: str(uuid4()))
    decision: dict[str, Any] | None = None
    created_at: str | None = None
    expires_at: str | None = None


@dataclass(slots=True, frozen=True)
class Observation:
    status: Literal["succeeded", "failed"]
    error_type: ObservationErrorType | None
    message: str
    retryable: bool
    side_effect: bool
    raw_data: Any = None
    evidence_ref: str | None = None
    attempt: int = 1
    duration_ms: int = 0


@dataclass(slots=True, frozen=True)
class AgentEvent:
    id: str
    task_id: str | None
    session_id: str
    type: EventType
    source: str
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    causation_id: str | None = None
    sequence: int | None = None
    created_at: str | None = None
    expected_version: int | None = None


@dataclass(slots=True, frozen=True)
class TaskState:
    id: str
    session_id: str
    goal: str
    status: TaskStatus
    step: int
    plan: tuple[TodoItem, ...]
    working_memory: dict[str, Any]
    pending_action: PendingAction | None
    last_observation: Observation | None
    reflection: str | None
    budget: TaskBudget
    stop_reason: str | None
    version: int
    created_at: str
    updated_at: str
    waiting_deadline: str | None = None
    parent_task_id: str | None = None


@dataclass(slots=True, frozen=True)
class SessionOverview:
    id: str
    created_at: str
    updated_at: str
    task_count: int
    latest_task: TaskState | None
    active_task: TaskState | None
    context: SessionContext


@dataclass(slots=True, frozen=True)
class TaskEvent:
    id: str
    task_id: str
    session_id: str
    type: str
    source: str
    payload: dict[str, Any]
    correlation_id: str | None
    causation_id: str | None
    sequence: int
    created_at: str


@dataclass(slots=True, frozen=True)
class TaskTrace:
    id: int
    task_id: str
    session_id: str
    trace_type: str
    payload: dict[str, Any]
    created_at: str


@dataclass(slots=True, frozen=True)
class SkillActivation:
    task_id: str
    skill_name: str
    scope: Literal["project", "user"]
    source_path: str
    content_hash: str
    version: str | None
    activation_reason: Literal["explicit", "model_match", "inherited_handoff"]
    state: Literal["active", "dropped"]
    activated_at: str


@dataclass(slots=True, frozen=True)
class TaskHandoff:
    source_task_id: str
    target_task_id: str
    target_session_id: str
    summary_text: str | None
    evidence_refs: tuple[str, ...]
    created_at: str


@dataclass(slots=True, frozen=True)
class SkillDraft:
    id: str
    session_id: str
    scope: Literal["project", "user"]
    skill_name: str
    content: str
    content_hash: str
    status: Literal["draft", "saved"]
    created_at: str
    saved_at: str | None = None
    saved_path: str | None = None


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
    model_name: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_source: Literal["provider", "estimated"] = "estimated"


@dataclass(slots=True)
class TurnResult:
    session_id: str
    final_text: str | None
    stop_reason: str | None
    tool_runs: list[ToolResult]
    success: bool
    task_id: str | None = None
    task_status: TaskStatus | None = None
    pending_action: PendingAction | None = None


@dataclass(slots=True)
class TaskResult:
    task: TaskState
    final_text: str | None
    stop_reason: str | None
    tool_runs: list[ToolResult]
    success: bool

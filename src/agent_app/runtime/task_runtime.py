from __future__ import annotations

from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from agent_app.state.session_service import ActiveTaskConflict, SessionService
from agent_app.types import AgentEvent, Observation, PendingAction, TaskBudget, TaskState, TaskStatus, TodoItem

TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset({"completed", "failed", "cancelled", "expired", "handed_off"})
_KEEP = object()
ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    "created": frozenset({"running", "cancelled"}),
    "running": frozenset({"waiting_user", "waiting_tool", "paused", "completed", "failed", "cancelled", "expired", "handed_off"}),
    "waiting_user": frozenset({"running", "cancelled", "expired"}),
    "waiting_tool": frozenset({"running", "failed", "cancelled", "expired"}),
    "paused": frozenset({"running", "cancelled", "handed_off"}),
    "completed": frozenset({"handed_off"}),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "expired": frozenset(),
    "handed_off": frozenset(),
}


class InvalidTaskTransition(RuntimeError):
    pass


class TaskRuntime:
    def __init__(self, session_service: SessionService) -> None:
        self._sessions = session_service

    def start_for_user_message(
        self,
        *,
        session_id: str,
        user_input: str,
        budget: TaskBudget | None = None,
        event: AgentEvent | None = None,
    ) -> TaskState:
        active = self._sessions.get_active_task(session_id)
        if active is not None and active.status == "created":
            return self.transition(
                active.id,
                target_status="running",
                event_type="user_message",
                source="user",
                payload={"content": user_input, "handoff_continuation": True},
                reason="handoff_continuation",
                event=event,
            )
        if active is not None and active.status == "waiting_user":
            return self.resume_with_user_message(active.id, user_input, event=event)
        if active is not None:
            raise ActiveTaskConflict(active)

        task = self._sessions.create_task(session_id, goal=user_input, budget=budget)
        return self.transition(
            task.id,
            target_status="running",
            event_type="user_message",
            source="user",
            payload={"content": user_input},
            reason="initial_user_message",
            event=event,
        )

    def resume_with_user_message(
        self,
        task_id: str,
        content: str,
        *,
        event: AgentEvent | None = None,
    ) -> TaskState:
        task = self.require_task(task_id)
        return self.transition(
            task.id,
            target_status="running",
            event_type="user_message",
            source="user",
            payload={"content": content},
            pending_action=None,
            waiting_deadline=None,
            reason="user_response",
            event=event,
        )

    def wait_for_user(self, task_id: str, pending_action: PendingAction) -> TaskState:
        task = self.require_task(task_id)
        deadline = (
            datetime.now(UTC) + timedelta(seconds=task.budget.waiting_user_timeout_seconds)
        ).isoformat()
        pending = replace(pending_action, created_at=pending_action.created_at or _utc_now(), expires_at=deadline)
        return self.transition(
            task.id,
            target_status="waiting_user",
            event_type="state_transition",
            source="runtime",
            payload={
                "pending_kind": pending.kind,
                "pending_action_id": pending.id,
            },
            pending_action=pending,
            waiting_deadline=deadline,
            reason="waiting_for_user",
        )

    def record_user_message(self, task_id: str, content: str, *, event: AgentEvent) -> TaskState:
        task = self.require_task(task_id)
        self._ensure_mutable(task)
        if task.status != "running":
            raise InvalidTaskTransition(
                f"Task '{task_id}' cannot accept a user message while {task.status}."
            )
        if self._sessions.task_event_exists(event.id):
            return task
        return self._sessions.apply_task_event(
            _resolved_event(
                task,
                event=event,
                event_type="user_message",
                source="user",
                payload={"content": content},
            ),
        )

    def approve(self, task_id: str, *, event: AgentEvent | None = None) -> TaskState:
        task = self.require_task(task_id)
        if task.pending_action is None:
            raise InvalidTaskTransition(f"Task '{task_id}' has no pending action.")
        self._validate_pending_action_event(task, event)
        return self.transition(
            task.id,
            target_status="running",
            event_type="user_approved",
            source="user",
            payload={
                "pending_kind": task.pending_action.kind,
                "pending_action_id": task.pending_action.id,
            },
            pending_action=None,
            waiting_deadline=None,
            reason="user_approved",
            event=event,
        )

    def reject(self, task_id: str, *, event: AgentEvent | None = None) -> TaskState:
        task = self.require_task(task_id)
        if task.pending_action is None:
            raise InvalidTaskTransition(f"Task '{task_id}' has no pending action.")
        self._validate_pending_action_event(task, event)
        return self.transition(
            task.id,
            target_status="running",
            event_type="user_rejected",
            source="user",
            payload={
                "pending_kind": task.pending_action.kind,
                "pending_action_id": task.pending_action.id,
            },
            pending_action=None,
            waiting_deadline=None,
            reason="user_rejected",
            event=event,
        )

    def pause(self, task_id: str, *, event: AgentEvent | None = None) -> TaskState:
        return self.transition(
            task_id,
            target_status="paused",
            event_type="pause_requested",
            source="user",
            reason="pause_requested",
            event=event,
        )

    def resume(self, task_id: str, *, event: AgentEvent | None = None) -> TaskState:
        return self.transition(
            task_id,
            target_status="running",
            event_type="resume_requested",
            source="user",
            reason="resume_requested",
            event=event,
        )

    def cancel(self, task_id: str, *, event: AgentEvent | None = None) -> TaskState:
        return self.transition(
            task_id,
            target_status="cancelled",
            event_type="cancel_requested",
            source="user",
            stop_reason="cancelled",
            reason="cancel_requested",
            event=event,
        )

    def expire(self, task_id: str, *, event: AgentEvent | None = None) -> TaskState:
        return self.transition(
            task_id,
            target_status="expired",
            event_type="task_expired",
            source="runtime",
            stop_reason="waiting_user_expired",
            reason="waiting_deadline_exceeded",
            event=event,
        )

    def complete(self, task_id: str, *, reason: str = "completed") -> TaskState:
        return self.transition(
            task_id,
            target_status="completed",
            event_type="state_transition",
            source="runtime",
            stop_reason=reason,
            reason=reason,
        )

    def fail(self, task_id: str, *, reason: str) -> TaskState:
        task = self.require_task(task_id)
        if task.status in TERMINAL_STATUSES:
            return task
        if task.status in {"created", "waiting_user", "paused"}:
            task = self.transition(
                task.id,
                target_status="running",
                event_type="state_transition",
                source="runtime",
                pending_action=None,
                waiting_deadline=None,
                reason="prepare_failure_transition",
            )
        return self.transition(
            task.id,
            target_status="failed",
            event_type="state_transition",
            source="runtime",
            stop_reason=reason,
            reason=reason,
        )

    def record_observation(self, task_id: str, observation: Observation) -> TaskState:
        task = self.require_task(task_id)
        self._ensure_mutable(task)
        return self._sessions.apply_task_event(
            _event(
                task,
                event_type="observation_recorded",
                source="executor",
                payload={
                    "status": observation.status,
                    "error_type": observation.error_type,
                    "attempt": observation.attempt,
                },
            ),
            last_observation=observation,
            step=task.step + 1,
        )

    def update_plan(self, task_id: str, plan: tuple[TodoItem, ...], *, reason: str) -> TaskState:
        task = self.require_task(task_id)
        self._ensure_mutable(task)
        updated = self._sessions.apply_task_event(
            _event(task, event_type="state_transition", source="planner", payload={"reason": reason}),
            plan=plan,
        )
        self._sessions.append_task_trace(task_id, "plan", {"reason": reason, "items": [asdict(item) for item in plan]})
        return updated

    def update_working_memory(self, task_id: str, values: dict) -> TaskState:
        task = self.require_task(task_id)
        self._ensure_mutable(task)
        memory = dict(task.working_memory)
        memory.update(values)
        return self._sessions.apply_task_event(
            _event(task, event_type="state_transition", source="runtime", payload={"memory_keys": sorted(values)}),
            working_memory=memory,
        )

    def reflect(self, task_id: str, summary: str) -> TaskState:
        task = self.require_task(task_id)
        self._ensure_mutable(task)
        if task.budget.used_replans >= task.budget.max_replans:
            return task
        budget = replace(task.budget, used_replans=task.budget.used_replans + 1)
        updated = self._sessions.apply_task_event(
            _event(task, event_type="budget_updated", source="reflection", payload={"summary": summary}),
            reflection=summary,
            budget=budget,
        )
        self._sessions.append_task_trace(task_id, "reflection", {"summary": summary})
        return updated

    def consume_model_call(self, task_id: str, *, tokens: int) -> TaskState:
        task = self.require_task(task_id)
        self._ensure_mutable(task)
        budget = replace(
            task.budget,
            used_model_calls=task.budget.used_model_calls + 1,
            used_tokens=task.budget.used_tokens + max(0, tokens),
        )
        return self._update_budget(task, budget)

    def consume_tool_call(self, task_id: str) -> TaskState:
        task = self.require_task(task_id)
        self._ensure_mutable(task)
        return self._update_budget(task, replace(task.budget, used_tool_calls=task.budget.used_tool_calls + 1))

    def consume_repair_attempt(self, task_id: str) -> TaskState:
        task = self.require_task(task_id)
        self._ensure_mutable(task)
        return self._update_budget(
            task,
            replace(task.budget, used_repair_attempts=task.budget.used_repair_attempts + 1),
        )

    def add_active_time(self, task_id: str, elapsed_seconds: float) -> TaskState:
        task = self.require_task(task_id)
        if task.status in TERMINAL_STATUSES:
            return task
        return self._update_budget(
            task,
            replace(task.budget, used_active_seconds=task.budget.used_active_seconds + max(0.0, elapsed_seconds)),
        )

    def budget_stop_reason(self, task: TaskState) -> str | None:
        budget = task.budget
        if budget.used_model_calls >= budget.max_model_calls:
            return "model_call_budget_exceeded"
        if budget.used_tool_calls >= budget.max_tool_calls:
            return "tool_call_budget_exceeded"
        if budget.used_tokens >= budget.max_tokens:
            return "token_budget_exceeded"
        if budget.used_active_seconds >= budget.max_active_seconds:
            return "active_time_budget_exceeded"
        return None

    def expire_if_needed(self, task: TaskState) -> TaskState:
        if task.status != "waiting_user" or task.waiting_deadline is None:
            return task
        if datetime.fromisoformat(task.waiting_deadline) <= datetime.now(UTC):
            return self.expire(task.id)
        return task

    def transition(
        self,
        task_id: str,
        *,
        target_status: TaskStatus,
        event_type,
        source: str,
        payload: dict | None = None,
        pending_action: PendingAction | None | object = _KEEP,
        waiting_deadline: str | None | object = _KEEP,
        stop_reason: str | None | object = _KEEP,
        reason: str,
        event: AgentEvent | None = None,
    ) -> TaskState:
        task = self.require_task(task_id)
        if event is not None and self._sessions.task_event_exists(event.id):
            return task
        if target_status not in ALLOWED_TRANSITIONS[task.status]:
            raise InvalidTaskTransition(f"Cannot transition task '{task_id}' from {task.status} to {target_status}.")
        return self._sessions.apply_task_event(
            _resolved_event(
                task,
                event=event,
                event_type=event_type,
                source=source,
                payload=payload or {},
            ),
            target_status=target_status,
            pending_action=task.pending_action if pending_action is _KEEP else pending_action,
            waiting_deadline=task.waiting_deadline if waiting_deadline is _KEEP else waiting_deadline,
            stop_reason=task.stop_reason if stop_reason is _KEEP else stop_reason,
            transition_reason=reason,
        )

    def require_task(self, task_id: str) -> TaskState:
        task = self._sessions.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

    def _update_budget(self, task: TaskState, budget: TaskBudget) -> TaskState:
        self._ensure_mutable(task)
        return self._sessions.apply_task_event(
            _event(task, event_type="budget_updated", source="runtime", payload={}),
            budget=budget,
        )

    @staticmethod
    def _ensure_mutable(task: TaskState) -> None:
        if task.status in TERMINAL_STATUSES:
            raise InvalidTaskTransition(f"Task '{task.id}' is terminal ({task.status}).")

    @staticmethod
    def _validate_pending_action_event(task: TaskState, event: AgentEvent | None) -> None:
        if event is None or task.pending_action is None:
            return
        expected_id = event.payload.get("pending_action_id")
        if expected_id is not None and expected_id != task.pending_action.id:
            raise InvalidTaskTransition(
                f"Pending action changed for task '{task.id}'. Refresh the task before approving it."
            )


def _event(task: TaskState, *, event_type, source: str, payload: dict) -> AgentEvent:
    return AgentEvent(
        id=str(uuid4()),
        task_id=task.id,
        session_id=task.session_id,
        type=event_type,
        source=source,
        payload=payload,
        correlation_id=task.id,
        expected_version=task.version,
    )


def _resolved_event(
    task: TaskState,
    *,
    event: AgentEvent | None,
    event_type,
    source: str,
    payload: dict,
) -> AgentEvent:
    if event is None:
        return _event(task, event_type=event_type, source=source, payload=payload)
    if event.task_id not in {None, task.id}:
        raise ValueError("Event task_id does not match the target task.")
    if event.session_id != task.session_id:
        raise ValueError("Event session_id does not match the target task.")
    if event.type != event_type:
        raise ValueError(f"Expected event type '{event_type}', got '{event.type}'.")
    return replace(
        event,
        task_id=task.id,
        payload={**payload, **event.payload},
        correlation_id=event.correlation_id or task.id,
        expected_version=task.version if event.expected_version is None else event.expected_version,
    )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()

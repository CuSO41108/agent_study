from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Callable

from agent_app.plan.graph import ready_node_ids
from agent_app.plan.store import PlanRevision, PlanStore
from agent_app.state.session_service import SessionService
from agent_app.types import TaskState, ToolAction


class RecoveryKind(str, Enum):
    """A transient conclusion calculated from persisted execution facts."""

    READY_TO_RESUME = "ready_to_resume"
    WAITING_TOOL_APPROVAL = "waiting_tool_approval"
    WAITING_USER_ANSWER = "waiting_user_answer"
    PAUSED = "paused"
    LEASE_ACTIVE = "lease_active"
    INTERRUPTED = "interrupted"
    TERMINAL = "terminal"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    kind: RecoveryKind
    reason: str
    task_id: str
    revision_id: str | None = None
    node_id: str | None = None
    pending_action_id: str | None = None


class PlanRecoveryError(RuntimeError):
    """The persisted Plan facts do not permit a safe resume operation."""

    def __init__(self, decision: RecoveryDecision) -> None:
        self.decision = decision
        super().__init__(decision.reason)


class PlanRecoveryService:
    """Inspect Plan recovery facts without invoking a model or changing state."""

    def __init__(
        self,
        *,
        plan_store: PlanStore,
        session_service: SessionService,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._plans = plan_store
        self._sessions = session_service
        self._clock = clock or (lambda: datetime.now(UTC))

    def inspect(self, task_id: str) -> RecoveryDecision:
        task = self._sessions.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        revisions = self._plans.list_revisions(task_id)
        active = [revision for revision in revisions if revision.status == "active"]

        if len(active) > 1:
            return self._decision(
                RecoveryKind.INCONSISTENT,
                task,
                reason="Task has more than one active PlanRevision.",
            )
        current = active[0] if active else None
        if current is None:
            if task.status in _TERMINAL_TASK_STATUSES:
                return self._decision(
                    RecoveryKind.TERMINAL,
                    task,
                    reason=f"Task is terminal ({task.status}) and has no active PlanRevision.",
                )
            return self._decision(
                RecoveryKind.INCONSISTENT,
                task,
                reason="Non-terminal Plan Task has no active PlanRevision.",
            )
        if task.status in _TERMINAL_TASK_STATUSES:
            return self._decision(
                RecoveryKind.INCONSISTENT,
                task,
                current=current,
                reason=f"Terminal Task ({task.status}) still has an active PlanRevision.",
            )

        running_nodes = [node for node in current.graph.nodes if node.status == "running"]
        waiting_nodes = [node for node in current.graph.nodes if node.status == "waiting_approval"]
        if len(running_nodes) + len(waiting_nodes) > 1:
            return self._decision(
                RecoveryKind.INCONSISTENT,
                task,
                current=current,
                reason="Serial execution allows at most one running or waiting node.",
            )

        waiting_node = waiting_nodes[0] if waiting_nodes else None
        pending = task.pending_action
        if task.status == "waiting_user" and pending is None:
            return self._decision(
                RecoveryKind.INCONSISTENT,
                task,
                current=current,
                reason="Task is waiting_user but has no pending action.",
            )
        if pending is not None and task.status != "waiting_user":
            return self._decision(
                RecoveryKind.INCONSISTENT,
                task,
                current=current,
                pending_action_id=pending.id,
                reason="A pending action exists while Task is not waiting_user.",
            )
        if waiting_node is not None:
            if pending is None:
                return self._decision(
                    RecoveryKind.INCONSISTENT,
                    task,
                    current=current,
                    node_id=waiting_node.id,
                    reason="A waiting Plan node has no pending action.",
                )
            pending_node_id = None if pending.decision is None else pending.decision.get("plan_node_id")
            if pending_node_id is not None and pending_node_id != waiting_node.id:
                return self._decision(
                    RecoveryKind.INCONSISTENT,
                    task,
                    current=current,
                    node_id=waiting_node.id,
                    pending_action_id=pending.id,
                    reason="Pending action is associated with another Plan node.",
                )
            if pending.kind == "tool_approval":
                return self._decision(
                    RecoveryKind.WAITING_TOOL_APPROVAL,
                    task,
                    current=current,
                    node_id=waiting_node.id,
                    pending_action_id=pending.id,
                    reason=f"Node '{waiting_node.id}' is waiting for tool approval.",
                )
            return self._decision(
                RecoveryKind.WAITING_USER_ANSWER,
                task,
                current=current,
                node_id=waiting_node.id,
                pending_action_id=pending.id,
                reason=f"Node '{waiting_node.id}' is waiting for a user answer.",
            )

        if pending is not None or task.status == "waiting_user":
            return self._decision(
                RecoveryKind.INCONSISTENT,
                task,
                current=current,
                reason="Task waiting facts are not associated with a waiting Plan node.",
            )

        if running_nodes:
            node = running_nodes[0]
            if current.execution_lease.is_active(now=self._clock()):
                return self._decision(
                    RecoveryKind.LEASE_ACTIVE,
                    task,
                    current=current,
                    node_id=node.id,
                    reason=f"Node '{node.id}' is owned by an active execution lease.",
                )
            return self._decision(
                RecoveryKind.INTERRUPTED,
                task,
                current=current,
                node_id=node.id,
                reason=self._interrupted_reason(task, node.id),
            )

        if task.status == "paused":
            if current.execution_lease.owner is not None:
                return self._decision(
                    RecoveryKind.INCONSISTENT,
                    task,
                    current=current,
                    reason="Paused Task still has an execution lease.",
                )
            return self._decision(
                RecoveryKind.PAUSED,
                task,
                current=current,
                reason="Plan Task is explicitly paused and has no running node.",
            )

        if current.execution_lease.is_active(now=self._clock()):
            return self._decision(
                RecoveryKind.INCONSISTENT,
                task,
                current=current,
                reason="An active PlanRevision has a lease but no running or waiting node.",
            )

        if ready_node_ids(current.graph):
            return self._decision(
                RecoveryKind.READY_TO_RESUME,
                task,
                current=current,
                reason="Plan has a ready pending node and no blocking action.",
            )
        return self._decision(
            RecoveryKind.INCONSISTENT,
            task,
            current=current,
            reason="Active Plan has no running, waiting, or ready node.",
        )

    def actions_for_node(self, task_id: str, node_id: str | None) -> list[ToolAction]:
        task = self._sessions.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        actions = [action for action in self._sessions.list_tool_actions(task.session_id) if action.task_id == task_id]
        if node_id is None:
            return actions
        scoped = [
            action
            for action in actions
            if action.recovery_metadata.get("plan_node_id") == node_id
        ]
        return scoped or actions

    def rewind_is_safe(self, decision: RecoveryDecision) -> bool:
        if decision.kind != RecoveryKind.INTERRUPTED:
            return False
        for action in self.actions_for_node(decision.task_id, decision.node_id):
            if action.status == "uncertain":
                return False
            if action.status in {"prepared", "executing"} and action.recovery_metadata.get("side_effect", False):
                return False
        return True

    def _interrupted_reason(self, task: TaskState, node_id: str) -> str:
        actions = self.actions_for_node(task.id, node_id)
        blocking = [
            action
            for action in actions
            if action.status == "uncertain"
            or (action.status in {"prepared", "executing"} and action.recovery_metadata.get("side_effect", False))
        ]
        if blocking:
            names = ", ".join(f"{action.tool_name}:{action.status}" for action in blocking)
            return f"Node '{node_id}' was interrupted; inspect possible side effects before retrying ({names})."
        return f"Node '{node_id}' was interrupted after its execution lease expired; explicit resume may retry it."

    @staticmethod
    def _decision(
        kind: RecoveryKind,
        task: TaskState,
        *,
        reason: str,
        current: PlanRevision | None = None,
        node_id: str | None = None,
        pending_action_id: str | None = None,
    ) -> RecoveryDecision:
        return RecoveryDecision(
            kind=kind,
            reason=reason,
            task_id=task.id,
            revision_id=None if current is None else current.id,
            node_id=node_id,
            pending_action_id=pending_action_id,
        )


_TERMINAL_TASK_STATUSES = {"completed", "failed", "cancelled", "expired", "handed_off"}

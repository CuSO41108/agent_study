from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_app.plan.agent_runner import PlanAgentNodeRunner
from agent_app.plan.executor import PlanExecutionResult, PlanExecutor
from agent_app.plan.planner import PlanPlanner
from agent_app.plan.store import PlanRevision, PlanStore
from agent_app.runtime.task_runtime import TaskRuntime
from agent_app.state.session_service import SessionService
from agent_app.types import Message, TaskState


@dataclass(frozen=True, slots=True)
class PlanTaskResult:
    task: TaskState
    revision: PlanRevision
    execution: PlanExecutionResult


class PlanTaskService:
    """Coordinate planning, PlanGraph persistence, and bounded node execution."""

    def __init__(
        self,
        *,
        planner: PlanPlanner,
        plan_store: PlanStore,
        session_service: SessionService,
        agent_loop: Any,
    ) -> None:
        self._planner = planner
        self._plan_store = plan_store
        self._sessions = session_service
        self._tasks = TaskRuntime(session_service)
        self._agent_loop = agent_loop

    def start(self, *, session_id: str, goal: str) -> PlanTaskResult:
        task = self._tasks.start_for_user_message(session_id=session_id, user_input=goal)
        self._sessions.append_message(session_id, Message(role="user", content=goal))
        try:
            graph = self._planner.create_plan(goal)
            revision = self._plan_store.create_revision(task.id, graph)
        except Exception:
            task = self._tasks.fail(task.id, reason="planner_failed")
            raise
        return self._execute_and_reconcile(task.id, revision)

    def continue_task(self, *, task_id: str) -> PlanTaskResult:
        task = self._require_task(task_id)
        revision = self._plan_store.get_active_revision(task.id)
        if revision is None:
            raise KeyError(f"No active plan revision for task '{task_id}'.")
        return self._execute_and_reconcile(task_id, revision)

    def _execute_and_reconcile(self, task_id: str, revision: PlanRevision) -> PlanTaskResult:
        executor = PlanExecutor(self._plan_store, PlanAgentNodeRunner(self._agent_loop))
        execution = executor.execute(task_id=task_id, revision=revision.graph.revision)
        task = self._require_task(task_id)
        if execution.status == "completed" and task.status not in {"completed", "failed", "cancelled", "expired"}:
            task = self._tasks.complete(task.id, reason="plan_completed")
        elif execution.status == "failed" and task.status not in {"completed", "failed", "cancelled", "expired"}:
            task = self._tasks.fail(task.id, reason=execution.failure_reason or "plan_failed")
        latest = self._plan_store.get_revision_by_id(execution.revision.id)
        return PlanTaskResult(task=task, revision=latest, execution=execution)

    def _require_task(self, task_id: str) -> TaskState:
        task = self._sessions.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        return task

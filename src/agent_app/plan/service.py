from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_app.plan.agent_runner import PlanAgentNodeRunner, build_node_prompt
from agent_app.plan.executor import PlanExecutionResult, PlanExecutor, PlanNodeContext
from agent_app.plan.planner import PlanPlanner
from agent_app.plan.store import PlanRevision, PlanStore
from agent_app.runtime.task_runtime import ReplanBudgetExceeded, TaskRuntime
from agent_app.state.session_service import SessionService
from agent_app.types import AgentEvent, Message, TaskState, TurnResult


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
        return self._execute_and_reconcile(task.id, revision, auto_replan=True)

    def continue_task(self, *, task_id: str) -> PlanTaskResult:
        task = self._require_task(task_id)
        revision = self._plan_store.get_active_revision(task.id)
        if revision is None:
            raise KeyError(f"No active plan revision for task '{task_id}'.")
        return self._execute_and_reconcile(task_id, revision, auto_replan=True)

    def replan(self, *, task_id: str, reason: str, automatic: bool = False) -> PlanTaskResult:
        """Create a successor revision and continue it through the same serial executor."""

        if not reason.strip():
            raise ValueError("Replan reason cannot be empty.")
        task = self._require_task(task_id)
        current = self._plan_store.get_active_revision(task.id)
        if current is None:
            raise KeyError(f"No active plan revision for task '{task_id}'.")
        self._tasks.consume_replan(task.id, reason=reason)
        candidate = self._planner.create_replan(current=current, reason=reason)
        revision = self._plan_store.create_replan(
            task.id,
            candidate,
            reason=reason,
            expected_revision=current.graph.revision,
        )
        return self._execute_and_reconcile(task.id, revision, auto_replan=automatic)

    def handle_approval(
        self,
        *,
        task_id: str,
        event: AgentEvent,
        approved: bool,
    ) -> PlanTaskResult | None:
        """Resume the exact waiting PlanGraph node after a user approval decision."""

        task = self._require_task(task_id)
        revision = self._plan_store.get_active_revision(task.id)
        if revision is None:
            return None
        node = next(
            (item for item in revision.graph.nodes if item.status == "waiting_approval"),
            None,
        )
        if node is None:
            return None
        context = PlanNodeContext(
            task_id=task.id,
            session_id=task.session_id,
            revision=revision,
            node=node,
            node_results=revision.node_results,
        )
        turn_result = self._agent_loop.handle_event(
            event,
            resume_allowed_tools=node.allowed_tools,
            resume_keep_task_open=True,
            resume_transient_context=build_node_prompt(context),
        )
        task = self._require_task(task.id)
        revision = self._plan_store.get_revision_by_id(revision.id)

        if approved and (turn_result.pending_action is not None or task.status == "waiting_user"):
            return PlanTaskResult(
                task=task,
                revision=revision,
                execution=PlanExecutionResult(
                    "waiting_approval",
                    revision,
                    waiting_node_id=node.id,
                ),
            )

        next_status = "completed" if approved and turn_result.success else "failed"
        result_record = {
            "status": next_status,
            "output": turn_result.final_text,
            "error": None if next_status == "completed" else (
                turn_result.final_text or turn_result.stop_reason or "approval_rejected"
            ),
            "metadata": {
                "approval": "approved" if approved else "rejected",
                "stop_reason": turn_result.stop_reason,
            },
        }
        revision = self._plan_store.update_node_status(
            revision.id,
            node.id,
            next_status,
            result=result_record,
            expected_version=revision.version,
        )
        return self._execute_and_reconcile(task.id, revision, auto_replan=True)

    def _execute_and_reconcile(
        self,
        task_id: str,
        revision: PlanRevision,
        *,
        auto_replan: bool,
    ) -> PlanTaskResult:
        executor = PlanExecutor(self._plan_store, PlanAgentNodeRunner(self._agent_loop))
        execution = executor.execute(task_id=task_id, revision=revision.graph.revision)
        task = self._require_task(task_id)
        if execution.status == "failed" and auto_replan:
            reason = execution.failure_reason or "A plan node failed."
            if task.budget.used_replans < task.budget.max_replans:
                try:
                    return self.replan(
                        task_id=task.id,
                        reason=reason,
                        automatic=True,
                    )
                except ReplanBudgetExceeded:
                    pass
        if execution.status == "failed":
            current_revision = self._plan_store.get_revision_by_id(execution.revision.id)
            if current_revision.status == "active":
                current_revision = self._plan_store.update_revision_status(
                    current_revision.id,
                    "failed",
                    expected_version=current_revision.version,
                )
                execution = PlanExecutionResult(
                    "failed",
                    current_revision,
                    execution.executed_node_ids,
                    execution.waiting_node_id,
                    execution.failure_reason,
                )
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

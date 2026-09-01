from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from agent_app.plan.agent_runner import PlanAgentNodeRunner, build_node_prompt
from agent_app.plan.executor import PlanExecutionResult, PlanExecutor, PlanNodeContext
from agent_app.plan.graph import PlanGraph, resource_claims_for_node
from agent_app.plan.planner import PlanPlanner, PlannerAttemptHook
from agent_app.plan.recovery import (
    PlanRecoveryError,
    PlanRecoveryService,
    RecoveryDecision,
    RecoveryKind,
    SideEffectRecoveryKind,
)
from agent_app.plan.store import PlanRevision, PlanStore
from agent_app.runtime.task_runtime import ReplanBudgetExceeded, TaskRuntime
from agent_app.state.session_service import SessionService
from agent_app.types import (
    AgentEvent,
    Message,
    TaskState,
    ToolAction,
    ToolActionResolutionOutcome,
)


@dataclass(frozen=True, slots=True)
class PlanTaskResult:
    task: TaskState
    revision: PlanRevision
    execution: PlanExecutionResult


@dataclass(frozen=True, slots=True)
class ToolActionResolutionResult:
    action: ToolAction
    decision: RecoveryDecision


class PlanTaskService:
    """Coordinate planning, PlanGraph persistence, and bounded node execution."""

    def __init__(
        self,
        *,
        planner: PlanPlanner,
        plan_store: PlanStore,
        session_service: SessionService,
        agent_loop: Any,
        recovery_service: PlanRecoveryService | None = None,
    ) -> None:
        self._planner = planner
        self._plan_store = plan_store
        self._sessions = session_service
        self._tasks = TaskRuntime(session_service)
        self._agent_loop = agent_loop
        self._recovery = recovery_service or PlanRecoveryService(
            plan_store=plan_store,
            session_service=session_service,
        )
        self._execution_owner = str(uuid4())

    def start(self, *, session_id: str, goal: str) -> PlanTaskResult:
        task = self._tasks.start_for_user_message(session_id=session_id, user_input=goal)
        self._sessions.append_message(session_id, Message(role="user", content=goal))
        try:
            graph = self._create_plan_with_checkpoint(task.id, goal)
            revision = self._plan_store.create_revision(task.id, graph)
        except Exception:
            task = self._tasks.fail(task.id, reason="planner_failed")
            raise
        self._sessions.append_task_trace(
            task.id,
            "plan_created",
            _plan_revision_payload(revision),
        )
        return self._execute_and_reconcile(task.id, revision, auto_replan=True)

    def continue_task(self, *, task_id: str) -> PlanTaskResult:
        task = self._require_task(task_id)
        revision = self._plan_store.get_active_revision(task.id)
        if revision is None:
            raise KeyError(f"No active plan revision for task '{task_id}'.")
        decision = self._recovery.inspect(task.id)
        if decision.kind == RecoveryKind.PAUSED:
            return self.resume(task_id=task_id, decision=decision)
        if decision.kind in {
            RecoveryKind.LEASE_ACTIVE,
            RecoveryKind.INTERRUPTED,
            RecoveryKind.INCONSISTENT,
        }:
            raise PlanRecoveryError(decision)
        return self._execute_and_reconcile(task_id, revision, auto_replan=True)

    def has_plan(self, *, task_id: str) -> bool:
        return bool(self._plan_store.list_revisions(task_id))

    def inspect_recovery(self, *, task_id: str) -> RecoveryDecision:
        return self._recovery.inspect(task_id)

    def list_tool_action_resolution_candidates(self, *, task_id: str) -> tuple[ToolAction, ...]:
        decision = self._recovery.inspect(task_id)
        return tuple(self._recovery.resolution_candidates(decision))

    def resolve_tool_action(
        self,
        *,
        task_id: str,
        action_id: str,
        outcome: ToolActionResolutionOutcome,
        reason: str,
        evidence: str,
        resolved_by: str,
    ) -> ToolActionResolutionResult:
        """Resolve one interrupted side effect without invoking the model or resuming execution."""

        decision = self._recovery.inspect(task_id)
        if decision.kind != RecoveryKind.INTERRUPTED:
            raise PlanRecoveryError(decision)
        candidates = self._recovery.resolution_candidates(decision)
        action = next((item for item in candidates if item.id == action_id), None)
        if action is None:
            existing = self._sessions.get_tool_action(action_id)
            if (
                existing is None
                or existing.task_id != task_id
                or existing.recovery_metadata.get("plan_node_id") != decision.node_id
                or existing.resolution is None
            ):
                raise ValueError(
                    f"Tool action '{action_id}' is not an unresolved side effect of the current interrupted node."
                )
            action = existing
        resolved = self._sessions.resolve_tool_action(
            action.id,
            outcome=outcome,
            reason=reason,
            evidence=evidence,
            resolved_by=resolved_by,
        )
        resolution_traced = any(
            trace.trace_type == "tool_action_resolution"
            and trace.payload.get("action_id") == action.id
            for trace in self._sessions.list_task_traces(task_id)
        )
        if not resolution_traced:
            previous_status = (
                resolved.resolution.previous_status
                if resolved.resolution is not None
                else action.status
            )
            self._sessions.append_task_trace(
                task_id,
                "tool_action_resolution",
                {
                    "revision_id": decision.revision_id,
                    "node_id": decision.node_id,
                    "action_id": action.id,
                    "tool_call_id": action.tool_call_id,
                    "tool": action.tool_name,
                    "previous_status": previous_status,
                    "outcome": outcome,
                    "reason": reason.strip(),
                    "evidence": evidence.strip(),
                    "resolved_by": resolved_by.strip(),
                    "resolved_at": resolved.resolution.resolved_at if resolved.resolution else None,
                },
            )
        return ToolActionResolutionResult(
            action=resolved,
            decision=self._recovery.inspect(task_id),
        )

    def resume(self, *, task_id: str, decision: RecoveryDecision | None = None) -> PlanTaskResult:
        """Re-inspect and explicitly resume a persisted Plan checkpoint."""

        fresh = self._recovery.inspect(task_id)
        if decision is not None and decision.task_id != task_id:
            raise ValueError("Recovery decision belongs to another task.")
        if fresh.kind in {
            RecoveryKind.WAITING_TOOL_APPROVAL,
            RecoveryKind.WAITING_USER_ANSWER,
        }:
            revision = self._plan_store.get_revision_by_id(fresh.revision_id or "")
            return PlanTaskResult(
                task=self._require_task(task_id),
                revision=revision,
                execution=PlanExecutionResult(
                    "waiting_approval",
                    revision,
                    waiting_node_id=fresh.node_id,
                ),
            )
        if fresh.kind in {
            RecoveryKind.LEASE_ACTIVE,
            RecoveryKind.INCONSISTENT,
            RecoveryKind.TERMINAL,
        }:
            raise PlanRecoveryError(fresh)

        task = self._require_task(task_id)
        if fresh.kind == RecoveryKind.PAUSED:
            if task.stop_reason == "max_tool_rounds_exceeded":
                if task.budget.used_continuations >= task.budget.max_continuations:
                    failed = self._tasks.fail(task.id, reason="continuation_budget_exceeded")
                    return self._finalize_failed_execution(
                        task_id,
                        PlanExecutionResult(
                            "failed",
                            self._plan_store.get_revision_by_id(fresh.revision_id or ""),
                            failure_reason="continuation_budget_exceeded",
                        ),
                    )
                task = self._tasks.resume(
                    task.id,
                    event=AgentEvent(
                        id=str(uuid4()),
                        task_id=task.id,
                        session_id=task.session_id,
                        type="resume_requested",
                        source="plan_recovery",
                        correlation_id=task.id,
                        expected_version=task.version,
                    ),
                )
                task = self._tasks.consume_continuation(
                    task.id,
                    reason="resume_after_max_tool_rounds",
                )
            else:
                task = self._tasks.resume(
                    task.id,
                    event=AgentEvent(
                        id=str(uuid4()),
                        task_id=task.id,
                        session_id=task.session_id,
                        type="resume_requested",
                        source="plan_recovery",
                        correlation_id=task.id,
                        expected_version=task.version,
                    ),
                )
        revision = self._plan_store.get_active_revision(task_id)
        if revision is None:
            raise PlanRecoveryError(fresh)
        if fresh.kind == RecoveryKind.PAUSED and fresh.node_id is not None:
            revision = self._plan_store.resume_paused_node(
                revision.id,
                fresh.node_id,
                expected_version=revision.version,
            )
            self._sessions.append_task_trace(
                task_id,
                "plan_recovery_resume_checkpoint",
                {
                    "plan_id": revision.graph.id,
                    "revision": revision.graph.revision,
                    "node_id": fresh.node_id,
                    "from": "paused",
                    "to": "pending",
                    "reason": fresh.reason,
                },
            )
        if fresh.kind == RecoveryKind.INTERRUPTED:
            assessment = self._recovery.assess_side_effects(fresh)
            if assessment.kind in {
                SideEffectRecoveryKind.UNRESOLVED,
                SideEffectRecoveryKind.MIXED,
            }:
                raise PlanRecoveryError(
                    RecoveryDecision(
                        kind=RecoveryKind.INTERRUPTED,
                        task_id=task_id,
                        revision_id=fresh.revision_id,
                        node_id=fresh.node_id,
                        reason=assessment.reason,
                    )
                )
            if assessment.kind == SideEffectRecoveryKind.COMPLETE:
                revision = self._accept_interrupted_side_effects(
                    task_id,
                    revision,
                    fresh,
                    assessment.succeeded_actions,
                )
            elif assessment.kind == SideEffectRecoveryKind.RETRY and self._recovery.rewind_is_safe(fresh):
                revision = self._rewind_interrupted_node(task_id, revision, fresh)
            else:
                raise PlanRecoveryError(fresh)
        return self._execute_and_reconcile(task_id, revision, auto_replan=True)

    def _accept_interrupted_side_effects(
        self,
        task_id: str,
        revision: PlanRevision,
        decision: RecoveryDecision,
        actions: tuple[ToolAction, ...],
    ) -> PlanRevision:
        action_ids = [action.id for action in actions]
        evidence = [_tool_action_recovery_evidence(action) for action in actions]
        accepted = self._plan_store.update_node_status(
            revision.id,
            decision.node_id or "",
            "completed",
            result={
                "status": "completed",
                "output": "\n".join(evidence),
                "error": None,
                "metadata": {
                    "recovery": "tool_action_history",
                    "action_ids": action_ids,
                },
            },
            expected_version=revision.version,
        )
        self._sessions.append_task_trace(
            task_id,
            "plan_recovery_accept_effect",
            {
                "plan_id": accepted.graph.id,
                "revision": accepted.graph.revision,
                "node_id": decision.node_id,
                "action_ids": action_ids,
                "from": "running",
                "to": "completed",
                "evidence": evidence,
            },
        )
        return accepted

    def _rewind_interrupted_node(
        self,
        task_id: str,
        revision: PlanRevision,
        decision: RecoveryDecision,
    ) -> PlanRevision:
        rewound = self._plan_store.rewind_running_node_to_pending(
            revision.id,
            decision.node_id or "",
            expected_version=revision.version,
        )
        self._sessions.append_task_trace(
            task_id,
            "plan_recovery_rewind",
            {
                "plan_id": rewound.graph.id,
                "revision": rewound.graph.revision,
                "node_id": decision.node_id,
                "from": "running",
                "to": "pending",
                "reason": decision.reason,
            },
        )
        return rewound

    def is_waiting_for_user_answer(self, *, task_id: str) -> bool:
        task = self._require_task(task_id)
        revision = self._plan_store.get_active_revision(task.id)
        return bool(
            revision is not None
            and task.status == "waiting_user"
            and task.pending_action is not None
            and task.pending_action.kind == "ask_user"
            and any(node.status == "waiting_approval" for node in revision.graph.nodes)
        )

    def replan(self, *, task_id: str, reason: str, automatic: bool = False) -> PlanTaskResult:
        """Create a successor revision and continue it through the same serial executor."""

        if not reason.strip():
            raise ValueError("Replan reason cannot be empty.")
        task = self._require_task(task_id)
        current = self._plan_store.get_active_revision(task.id)
        if current is None:
            raise KeyError(f"No active plan revision for task '{task_id}'.")
        self._tasks.consume_replan(task.id, reason=reason)
        candidate = self._create_replan_with_checkpoint(task.id, current=current, reason=reason)
        revision = self._plan_store.create_replan(
            task.id,
            candidate,
            reason=reason,
            expected_revision=current.graph.revision,
        )
        self._sessions.append_task_trace(
            task.id,
            "plan_replan",
            {
                "plan_id": revision.graph.id,
                "from_revision": current.graph.revision,
                "to_revision": revision.graph.revision,
                "reason": reason,
                "preserved_completed_nodes": [
                    node.id
                    for node in revision.graph.nodes
                    if node.status == "completed"
                ],
            },
        )
        return self._execute_and_reconcile(task.id, revision, auto_replan=automatic)

    def _create_plan_with_checkpoint(self, task_id: str, goal: str) -> PlanGraph:
        return self._run_planner_with_checkpoint(
            task_id=task_id,
            operation="initial_plan",
            invoke=lambda on_attempt: self._planner.create_plan(goal, on_attempt=on_attempt),
        )

    def _create_replan_with_checkpoint(
        self,
        task_id: str,
        *,
        current: PlanRevision,
        reason: str,
    ) -> PlanGraph:
        return self._run_planner_with_checkpoint(
            task_id=task_id,
            operation="replan",
            invoke=lambda on_attempt: self._planner.create_replan(
                current=current,
                reason=reason,
                on_attempt=on_attempt,
            ),
        )

    def _run_planner_with_checkpoint(
        self,
        *,
        task_id: str,
        operation: str,
        invoke: Callable[[PlannerAttemptHook], PlanGraph],
    ) -> PlanGraph:
        task = self._require_task(task_id)
        latest = self._sessions.get_latest_checkpoint(task_id)
        run = self._sessions.create_execution_run(
            task_id=task_id,
            agent_id="planner",
            scope=f"planner:{operation}",
            max_tool_rounds=1,
            parent_checkpoint_id=None if latest is None else latest.id,
        )
        max_attempts = self._planner.max_request_attempts
        self._persist_planner_checkpoint(
            task_id=task_id,
            run_id=run.id,
            operation=operation,
            request_status="requesting",
            attempt=0,
            max_attempts=max_attempts,
        )
        last_request_status = "requesting"
        last_attempt = 0

        def on_attempt(info: dict[str, Any]) -> None:
            nonlocal last_attempt, last_request_status
            last_request_status = str(info.get("request_status", "failed"))
            last_attempt = int(info.get("attempt", 0))
            if last_request_status == "succeeded":
                return
            self._persist_planner_checkpoint(
                task_id=task_id,
                run_id=run.id,
                operation=operation,
                request_status=last_request_status,
                attempt=last_attempt,
                max_attempts=int(info.get("max_attempts", max_attempts)),
                error_type=info.get("error_type"),
                error_detail=info.get("error_detail"),
                retryable=bool(info.get("retryable", False)),
                retry_delay_seconds=float(info.get("retry_delay_seconds", 0.0)),
            )

        try:
            graph = invoke(on_attempt)
        except Exception as exc:
            error_type = getattr(exc, "error_type", None) or type(exc).__name__
            error_detail = getattr(exc, "detail", None) or str(exc)
            attempts = int(getattr(exc, "attempts", 0) or 1)
            if last_request_status != "failed":
                self._persist_planner_checkpoint(
                    task_id=task_id,
                    run_id=run.id,
                    operation=operation,
                    request_status="failed",
                    attempt=attempts,
                    max_attempts=max_attempts,
                    error_type=str(error_type),
                    error_detail=str(error_detail),
                )
            self._sessions.update_execution_run(
                run.id,
                status="failed",
                stop_reason="planner_failed",
            )
            raise

        self._sessions.update_execution_run(
            run.id,
            status="completed",
            stop_reason="plan_created" if operation == "initial_plan" else "replan_created",
        )
        self._persist_planner_checkpoint(
            task_id=task_id,
            run_id=run.id,
            operation=operation,
            request_status="completed",
            attempt=last_attempt or 1,
            max_attempts=max_attempts,
            plan_id=getattr(graph, "id", None),
            node_count=len(getattr(graph, "nodes", ())),
        )
        return graph

    def _persist_planner_checkpoint(
        self,
        *,
        task_id: str,
        run_id: str,
        operation: str,
        request_status: str,
        attempt: int,
        max_attempts: int,
        error_type: object | None = None,
        error_detail: object | None = None,
        retryable: bool = False,
        retry_delay_seconds: float = 0.0,
        plan_id: str | None = None,
        node_count: int | None = None,
    ) -> None:
        is_completed = request_status == "completed"
        is_failed = request_status == "failed"
        cursor = "completed" if is_completed else "failed" if is_failed else "planning"
        status = "completed" if is_completed else "failed" if is_failed else "running"
        state: dict[str, Any] = {
            "phase": "planning",
            "operation": operation,
            "request_status": request_status,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "retryable": retryable,
            "retry_delay_seconds": retry_delay_seconds,
        }
        if error_type is not None:
            state["error_type"] = str(error_type)
        if error_detail:
            state["error_detail"] = str(error_detail)
        if plan_id is not None:
            state["plan_id"] = plan_id
        if node_count is not None:
            state["node_count"] = node_count
        checkpoint = self._sessions.create_checkpoint(
            task_id=task_id,
            run_id=run_id,
            cursor=cursor,
            status=status,
            state=state,
        )
        self._sessions.append_task_trace(
            task_id,
            "checkpoint",
            {
                "checkpoint_id": checkpoint.id,
                "run_id": run_id,
                "parent_checkpoint_id": checkpoint.parent_checkpoint_id,
                "cursor": cursor,
                "status": status,
                **state,
            },
        )

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
        if task.pending_action is not None and task.pending_action.kind != "tool_approval":
            raise ValueError(
                "This Plan node is waiting for a user answer; "
                "enter a natural-language response instead of approving or rejecting it."
            )
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
            resume_plan_revision_id=revision.id,
            resume_plan_node_id=node.id,
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
        self._sessions.append_task_trace(
            task.id,
            "plan_node_approval",
            {
                "plan_id": revision.graph.id,
                "revision": revision.graph.revision,
                "node_id": node.id,
                "decision": "approve" if approved else "reject",
                "from_status": "waiting_approval",
                "to_status": next_status,
                "stop_reason": turn_result.stop_reason,
            },
        )
        return self._execute_and_reconcile(task.id, revision, auto_replan=True)

    def handle_user_message(
        self,
        *,
        task_id: str,
        event: AgentEvent,
    ) -> PlanTaskResult | None:
        """Resume a Plan node that explicitly asked the user for information."""

        task = self._require_task(task_id)
        revision = self._plan_store.get_active_revision(task.id)
        if revision is None:
            return None
        node = next(
            (item for item in revision.graph.nodes if item.status == "waiting_approval"),
            None,
        )
        pending = task.pending_action
        if (
            node is None
            or task.status != "waiting_user"
            or pending is None
            or pending.kind != "ask_user"
        ):
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
            resume_plan_revision_id=revision.id,
            resume_plan_node_id=node.id,
        )
        task = self._require_task(task.id)
        revision = self._plan_store.get_revision_by_id(revision.id)

        if turn_result.pending_action is not None or task.status == "waiting_user":
            return PlanTaskResult(
                task=task,
                revision=revision,
                execution=PlanExecutionResult(
                    "waiting_approval",
                    revision,
                    waiting_node_id=node.id,
                ),
            )

        next_status = "completed" if turn_result.success else "failed"
        result_record = {
            "status": next_status,
            "output": turn_result.final_text,
            "error": None if next_status == "completed" else (
                turn_result.final_text or turn_result.stop_reason or "user_message_resume_failed"
            ),
            "metadata": {
                "resume_kind": "ask_user",
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
        self._sessions.append_task_trace(
            task.id,
            "plan_node_user_message",
            {
                "plan_id": revision.graph.id,
                "revision": revision.graph.revision,
                "node_id": node.id,
                "from_status": "waiting_approval",
                "to_status": next_status,
                "stop_reason": turn_result.stop_reason,
            },
        )
        return self._execute_and_reconcile(task.id, revision, auto_replan=True)

    def _execute_and_reconcile(
        self,
        task_id: str,
        revision: PlanRevision,
        *,
        auto_replan: bool,
    ) -> PlanTaskResult:
        before = self._plan_store.get_revision_by_id(revision.id)
        executor = PlanExecutor(
            self._plan_store,
            PlanAgentNodeRunner(self._agent_loop),
            lease_owner=self._execution_owner,
        )
        execution = executor.execute(task_id=task_id, revision=revision.graph.revision)
        after = self._plan_store.get_revision_by_id(execution.revision.id)
        self._record_execution_trace(task_id, before=before, after=after, execution=execution)
        task = self._require_task(task_id)
        if execution.status == "failed" and auto_replan:
            reason = execution.failure_reason or "A plan node failed."
            if (
                task.status not in {"completed", "failed", "cancelled", "expired"}
                and task.budget.used_replans < task.budget.max_replans
            ):
                try:
                    return self.replan(
                        task_id=task.id,
                        reason=reason,
                        automatic=True,
                    )
                except ReplanBudgetExceeded:
                    pass
                except Exception as exc:  # noqa: BLE001 - auto recovery must close the task.
                    return self._finalize_failed_execution(
                        task_id,
                        execution,
                        replan_error=exc,
                    )
        if execution.status == "paused":
            return PlanTaskResult(
                task=task,
                revision=self._plan_store.get_revision_by_id(execution.revision.id),
                execution=execution,
            )
        if execution.status == "failed":
            return self._finalize_failed_execution(task_id, execution)
        if execution.status == "completed" and task.status not in {"completed", "failed", "cancelled", "expired"}:
            task = self._tasks.complete(task.id, reason="plan_completed")
        latest = self._plan_store.get_revision_by_id(execution.revision.id)
        return PlanTaskResult(task=task, revision=latest, execution=execution)

    def _finalize_failed_execution(
        self,
        task_id: str,
        execution: PlanExecutionResult,
        *,
        replan_error: Exception | None = None,
    ) -> PlanTaskResult:
        failure_reason = execution.failure_reason or "plan_failed"
        if replan_error is not None:
            failure_reason = (
                f"auto_replan_failed: {type(replan_error).__name__}: {replan_error}"
            )
            try:
                self._sessions.append_task_trace(
                    task_id,
                    "plan_replan_failed",
                    {
                        "plan_id": execution.revision.graph.id,
                        "revision": execution.revision.graph.revision,
                        "original_failure_reason": execution.failure_reason,
                        "error_type": type(replan_error).__name__,
                        "error": str(replan_error),
                    },
                )
            except Exception:
                pass

        current_revision = self._plan_store.get_revision_by_id(execution.revision.id)
        if current_revision.status != "active":
            active_revision = self._plan_store.get_active_revision(task_id)
            if active_revision is not None:
                current_revision = active_revision
        if current_revision.status == "active":
            current_revision = self._plan_store.update_revision_status(
                current_revision.id,
                "failed",
                expected_version=current_revision.version,
            )
        finalized_execution = PlanExecutionResult(
            "failed",
            current_revision,
            execution.executed_node_ids,
            execution.waiting_node_id,
            failure_reason,
        )
        task = self._require_task(task_id)
        if task.status not in {"completed", "failed", "cancelled", "expired"}:
            task = self._tasks.fail(task.id, reason=failure_reason)
        latest = self._plan_store.get_revision_by_id(current_revision.id)
        return PlanTaskResult(task=task, revision=latest, execution=finalized_execution)

    def _record_execution_trace(
        self,
        task_id: str,
        *,
        before: PlanRevision,
        after: PlanRevision,
        execution: PlanExecutionResult,
    ) -> None:
        before_nodes = before.graph.node_map()
        for node in after.graph.nodes:
            previous = before_nodes.get(node.id)
            if previous is None or previous.status == node.status:
                continue
            result = after.node_results.get(node.id, {})
            self._sessions.append_task_trace(
                task_id,
                "plan_node_transition",
                {
                    "plan_id": after.graph.id,
                    "revision": after.graph.revision,
                    "node_id": node.id,
                    "kind": node.kind,
                    "objective": node.objective,
                    "acceptance": list(node.acceptance),
                    "resources": [
                        {"key": resource.key, "mode": resource.mode}
                        for resource in resource_claims_for_node(node)
                    ],
                    "from_status": previous.status,
                    "to_status": node.status,
                    "executed": node.id in execution.executed_node_ids,
                    "result": _node_result_payload(result),
                },
            )
        self._sessions.append_task_trace(
            task_id,
            "plan_execution",
            {
                "plan_id": after.graph.id,
                "revision": after.graph.revision,
                "status": execution.status,
                "executed_node_ids": list(execution.executed_node_ids),
                "waiting_node_id": execution.waiting_node_id,
                "failure_reason": execution.failure_reason,
            },
        )
        if execution.status == "failed":
            failed_nodes = [
                node.id for node in after.graph.nodes if node.status == "failed"
            ]
            skipped_nodes = [
                node.id for node in after.graph.nodes if node.status == "skipped"
            ]
            self._sessions.append_task_trace(
                task_id,
                "plan_failure",
                {
                    "plan_id": after.graph.id,
                    "revision": after.graph.revision,
                    "failure_reason": execution.failure_reason,
                    "failed_node_ids": failed_nodes,
                    "skipped_node_ids": skipped_nodes,
                    "executed_node_ids": list(execution.executed_node_ids),
                },
            )

    def _require_task(self, task_id: str) -> TaskState:
        task = self._sessions.get_task(task_id)
        if task is None:
            raise KeyError(task_id)
        return task


def _plan_revision_payload(revision: PlanRevision) -> dict[str, Any]:
    return {
        "plan_id": revision.graph.id,
        "revision": revision.graph.revision,
        "goal": revision.graph.goal,
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "depends_on": list(node.depends_on),
                "allowed_tools": list(node.allowed_tools),
                "objective": node.objective,
                "acceptance": list(node.acceptance),
                "resources": [
                    {"key": resource.key, "mode": resource.mode}
                    for resource in resource_claims_for_node(node)
                ],
            }
            for node in revision.graph.nodes
        ],
    }


def _node_result_payload(result: Any) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    output = result.get("output")
    return {
        "status": result.get("status"),
        "error": result.get("error"),
        "evidence_refs": list(result.get("evidence_refs", [])),
        "output_preview": str(output)[:500] if output is not None else None,
        "metadata": result.get("metadata", {}),
    }


def _tool_action_recovery_evidence(action: ToolAction) -> str:
    if action.resolution is not None:
        return action.resolution.evidence
    if action.result is not None and action.result.content.strip():
        return action.result.content.strip()[:1000]
    return f"{action.tool_name} action {action.id} succeeded."

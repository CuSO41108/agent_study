from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from agent_app.plan.graph import PlanNode, ready_node_ids
from agent_app.plan.store import PlanRevision, PlanStore


NodeExecutionStatus = Literal["completed", "failed", "waiting_approval"]
PlanExecutionStatus = Literal["completed", "failed", "waiting_approval", "blocked"]


@dataclass(frozen=True, slots=True)
class NodeExecutionResult:
    """The bounded result returned by one node runner invocation."""

    status: NodeExecutionStatus
    output: str | None = None
    error: str | None = None
    evidence_refs: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output": self.output,
            "error": self.error,
            "evidence_refs": list(self.evidence_refs),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class PlanNodeContext:
    """Read-only input supplied to a runner for one serial node."""

    task_id: str
    session_id: str
    revision: PlanRevision
    node: PlanNode
    node_results: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class PlanExecutionResult:
    status: PlanExecutionStatus
    revision: PlanRevision
    executed_node_ids: tuple[str, ...] = ()
    waiting_node_id: str | None = None
    failure_reason: str | None = None


NodeRunner = Callable[[PlanNodeContext], NodeExecutionResult]


class PlanExecutor:
    """Execute a validated PlanGraph one ready node at a time.

    The runner is deliberately injected. The first implementation therefore
    provides a safe scheduling/persistence boundary without embedding another
    ReAct loop or changing the existing AgentLoop entry point.
    """

    def __init__(self, store: PlanStore, runner: NodeRunner) -> None:
        self._store = store
        self._runner = runner

    def execute(self, *, task_id: str, revision: int | None = None) -> PlanExecutionResult:
        plan = self._load_plan(task_id, revision)
        executed: list[str] = []

        if plan.status == "completed":
            return PlanExecutionResult("completed", plan)
        if plan.status == "failed":
            return PlanExecutionResult("failed", plan, failure_reason="plan_failed")
        if plan.status == "superseded":
            return PlanExecutionResult("blocked", plan, failure_reason="plan_superseded")

        while True:
            plan = self._store.get_revision_by_id(plan.id)
            waiting_node = _first_node_with_status(plan, "waiting_approval")
            if waiting_node is not None:
                return PlanExecutionResult(
                    "waiting_approval",
                    plan,
                    tuple(executed),
                    waiting_node_id=waiting_node.id,
                )
            running_node = _first_node_with_status(plan, "running")
            if running_node is not None:
                return PlanExecutionResult(
                    "blocked",
                    plan,
                    tuple(executed),
                    failure_reason=f"node '{running_node.id}' was already running; inspect before retrying",
                )

            ready = ready_node_ids(plan.graph)
            if ready:
                node_id = ready[0]
                running_plan = self._store.update_node_status(
                    plan.id,
                    node_id,
                    "running",
                    expected_version=plan.version,
                )
                node = running_plan.graph.node_map()[node_id]
                context = PlanNodeContext(
                    task_id=task_id,
                    session_id=self._store.get_task_session_id(task_id),
                    revision=running_plan,
                    node=node,
                    node_results=running_plan.node_results,
                )
                outcome = self._run_node(context)
                plan = self._store.update_node_status(
                    running_plan.id,
                    node_id,
                    outcome.status,
                    result=outcome.to_record(),
                    expected_version=running_plan.version,
                )
                executed.append(node_id)
                if outcome.status == "waiting_approval":
                    return PlanExecutionResult(
                        "waiting_approval",
                        plan,
                        tuple(executed),
                        waiting_node_id=node_id,
                    )
                if outcome.status == "failed":
                    plan = self._skip_dependents(plan, failed_node_id=node_id)
                    plan = self._store.update_revision_status(
                        plan.id,
                        "failed",
                        expected_version=plan.version,
                    )
                    return PlanExecutionResult(
                        "failed",
                        plan,
                        tuple(executed),
                        failure_reason=outcome.error or f"node '{node_id}' failed",
                    )
                continue

            blocked_ids = _blocked_pending_node_ids(plan)
            if blocked_ids:
                plan = self._skip_nodes(plan, blocked_ids)
                plan = self._store.update_revision_status(
                    plan.id,
                    "failed",
                    expected_version=plan.version,
                )
                return PlanExecutionResult(
                    "failed",
                    plan,
                    tuple(executed),
                    failure_reason="node_dependencies_blocked",
                )

            node_statuses = {node.status for node in plan.graph.nodes}
            if node_statuses <= {"completed", "skipped"}:
                if "skipped" in node_statuses:
                    plan = self._store.update_revision_status(
                        plan.id,
                        "failed",
                        expected_version=plan.version,
                    )
                    return PlanExecutionResult(
                        "failed",
                        plan,
                        tuple(executed),
                        failure_reason="plan_contains_skipped_nodes",
                    )
                plan = self._store.update_revision_status(
                    plan.id,
                    "completed",
                    expected_version=plan.version,
                )
                return PlanExecutionResult("completed", plan, tuple(executed))

            return PlanExecutionResult(
                "blocked",
                plan,
                tuple(executed),
                failure_reason="no_ready_node",
            )

    def resume_waiting_node(
        self,
        *,
        task_id: str,
        node_id: str,
        revision: int | None = None,
    ) -> PlanRevision:
        plan = self._load_plan(task_id, revision)
        if plan.status != "active":
            raise RuntimeError(f"Plan revision '{plan.graph.revision}' is not active.")
        node = plan.graph.node_map().get(node_id)
        if node is None:
            raise KeyError(node_id)
        if node.status != "waiting_approval":
            raise RuntimeError(f"Node '{node_id}' is not waiting for approval.")
        return self._store.resume_waiting_node(
            plan.id,
            node_id,
            expected_version=plan.version,
        )

    def _load_plan(self, task_id: str, revision: int | None) -> PlanRevision:
        if revision is not None:
            plan = self._store.get_revision(task_id, revision)
        else:
            plan = self._store.get_active_revision(task_id) or self._store.get_revision(task_id)
        if plan is None:
            raise KeyError(f"No plan revision found for task '{task_id}'.")
        return plan

    def _run_node(self, context: PlanNodeContext) -> NodeExecutionResult:
        try:
            result = self._runner(context)
        except Exception as exc:  # noqa: BLE001 - node failures become durable diagnostics.
            return NodeExecutionResult(
                status="failed",
                error=f"runner_exception: {type(exc).__name__}: {exc}",
            )
        if not isinstance(result, NodeExecutionResult):
            return NodeExecutionResult(
                status="failed",
                error="runner_contract_error: expected NodeExecutionResult",
            )
        return result

    def _skip_dependents(self, plan: PlanRevision, *, failed_node_id: str) -> PlanRevision:
        current = plan
        while True:
            blocked = [
                node.id
                for node in current.graph.nodes
                if node.status == "pending"
                and any(
                    dependency == failed_node_id or current.graph.node_map()[dependency].status == "skipped"
                    for dependency in node.depends_on
                )
            ]
            if not blocked:
                return current
            current = self._skip_nodes(current, blocked, failed_node_id=failed_node_id)

    def _skip_nodes(
        self,
        plan: PlanRevision,
        node_ids: list[str],
        *,
        failed_node_id: str | None = None,
    ) -> PlanRevision:
        current = plan
        for node_id in node_ids:
            blocked_by = [
                dependency
                for dependency in current.graph.node_map()[node_id].depends_on
                if dependency == failed_node_id
                or current.graph.node_map()[dependency].status in {"failed", "skipped"}
            ]
            current = self._store.update_node_status(
                current.id,
                node_id,
                "skipped",
                result={"status": "skipped", "reason": "dependency_failed", "blocked_by": blocked_by},
                expected_version=current.version,
            )
        return current


def _first_node_with_status(plan: PlanRevision, status: str) -> PlanNode | None:
    return next((node for node in plan.graph.nodes if node.status == status), None)


def _blocked_pending_node_ids(plan: PlanRevision) -> list[str]:
    nodes = plan.graph.node_map()
    return [
        node.id
        for node in plan.graph.nodes
        if node.status == "pending"
        and any(nodes[dependency].status in {"failed", "skipped"} for dependency in node.depends_on)
    ]

from __future__ import annotations

import shutil
import threading
import time
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.plan import (
    NodeExecutionResult,
    PlanExecutor,
    PlanStore,
    parse_plan_graph,
)
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService


def _graph(*, revision: int = 1, statuses: dict[str, str] | None = None):
    statuses = statuses or {}
    return parse_plan_graph(
        {
            "id": "plan-1",
            "revision": revision,
            "goal": "Implement and verify a small coding change.",
            "nodes": [
                {
                    "id": "inspect",
                    "kind": "inspect",
                    "objective": "Locate the relevant implementation.",
                    "depends_on": [],
                    "allowed_tools": ["file_read", "code_search"],
                    "acceptance": ["The implementation is identified."],
                    "status": statuses.get("inspect", "pending"),
                },
                {
                    "id": "edit",
                    "kind": "edit",
                    "objective": "Apply the smallest required change.",
                    "depends_on": ["inspect"],
                    "allowed_tools": ["replace_in_file"],
                    "acceptance": ["The intended edit is present."],
                    "status": statuses.get("edit", "pending"),
                },
                {
                    "id": "verify",
                    "kind": "verify",
                    "objective": "Run the focused verification.",
                    "depends_on": ["edit"],
                    "allowed_tools": ["shell"],
                    "acceptance": ["The verification passes."],
                    "status": statuses.get("verify", "pending"),
                },
            ],
        }
    )


def _independent_graph(*, kind: str = "inspect", resources: list[list[dict[str, str]]] | None = None):
    allowed_tools = "file_read" if kind == "inspect" else "file_write"
    node_resources = resources or [[], []]
    return parse_plan_graph(
        {
            "id": "parallel-plan",
            "revision": 1,
            "goal": "Execute independent nodes safely.",
            "nodes": [
                {
                    "id": f"{kind}-a",
                    "kind": kind,
                    "objective": f"Execute {kind} node A.",
                    "depends_on": [],
                    "allowed_tools": [allowed_tools],
                    "acceptance": ["Node A completed."],
                    "status": "pending",
                    **({"resources": node_resources[0]} if node_resources[0] else {}),
                },
                {
                    "id": f"{kind}-b",
                    "kind": kind,
                    "objective": f"Execute {kind} node B.",
                    "depends_on": [],
                    "allowed_tools": [allowed_tools],
                    "acceptance": ["Node B completed."],
                    "status": "pending",
                    **({"resources": node_resources[1]} if node_resources[1] else {}),
                },
                {
                    "id": "verify",
                    "kind": "verify",
                    "objective": "Verify both independent nodes.",
                    "depends_on": [f"{kind}-a", f"{kind}-b"],
                    "allowed_tools": ["file_read"],
                    "acceptance": ["Both nodes completed."],
                    "status": "pending",
                },
            ],
        }
    )


class PlanExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"plan_execution_{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.db_path = self.root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.session_id = self.sessions.create_session("plan-session")
        self.task = self.sessions.create_task(self.session_id, goal="Implement and verify")
        self.store = PlanStore(self.db_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_revision_and_node_snapshots_round_trip_with_optimistic_version(self) -> None:
        revision = self.store.create_revision(self.task.id, _graph())

        running = self.store.update_node_status(
            revision.id,
            "inspect",
            "running",
            expected_version=revision.version,
        )
        completed = self.store.update_node_status(
            running.id,
            "inspect",
            "completed",
            result={"output": "found implementation", "evidence_refs": ["trace-1"]},
            expected_version=running.version,
        )
        reloaded = self.store.get_revision(self.task.id)

        self.assertEqual(reloaded.version, completed.version)
        self.assertEqual(reloaded.graph.node_map()["inspect"].status, "completed")
        self.assertEqual(reloaded.node_results["inspect"]["output"], "found implementation")

    def test_executor_runs_ready_nodes_serially_and_persists_outputs(self) -> None:
        revision = self.store.create_revision(self.task.id, _graph())
        calls: list[str] = []

        def runner(context):
            calls.append(context.node.id)
            return NodeExecutionResult("completed", output=f"done:{context.node.id}")

        result = PlanExecutor(self.store, runner).execute(task_id=self.task.id)

        self.assertEqual(result.status, "completed")
        self.assertEqual(calls, ["inspect", "edit", "verify"])
        self.assertEqual(result.revision.status, "completed")
        self.assertEqual(
            [node.status for node in result.revision.graph.nodes],
            ["completed", "completed", "completed"],
        )
        self.assertEqual(result.revision.node_results["verify"]["output"], "done:verify")
        self.assertEqual(self.store.get_revision_by_id(revision.id).status, "completed")

    def test_executor_runs_independent_read_nodes_concurrently(self) -> None:
        self.store.create_revision(self.task.id, _independent_graph())
        entered = threading.Barrier(2)
        calls: list[str] = []

        def runner(context):
            if context.node.id in {"inspect-a", "inspect-b"}:
                entered.wait(timeout=2)
            calls.append(context.node.id)
            return NodeExecutionResult("completed", output=f"done:{context.node.id}")

        result = PlanExecutor(self.store, runner, max_concurrency=2).execute(task_id=self.task.id)

        self.assertEqual(result.status, "completed")
        self.assertEqual(set(calls[:2]), {"inspect-a", "inspect-b"})
        self.assertEqual(calls[-1], "verify")

    def test_executor_keeps_unclaimed_side_effecting_nodes_serial(self) -> None:
        self.store.create_revision(self.task.id, _independent_graph(kind="edit"))
        state_lock = threading.Lock()
        active = 0
        peak_active = 0

        def runner(context):
            nonlocal active, peak_active
            with state_lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1
            return NodeExecutionResult("completed", output=f"done:{context.node.id}")

        result = PlanExecutor(self.store, runner, max_concurrency=2).execute(task_id=self.task.id)

        self.assertEqual(result.status, "completed")
        self.assertEqual(peak_active, 1)

    def test_failed_node_skips_dependents_and_fails_revision(self) -> None:
        self.store.create_revision(self.task.id, _graph())
        calls: list[str] = []

        def runner(context):
            calls.append(context.node.id)
            if context.node.id == "edit":
                return NodeExecutionResult("failed", error="focused check failed")
            return NodeExecutionResult("completed", output="ok")

        result = PlanExecutor(self.store, runner).execute(task_id=self.task.id)

        self.assertEqual(result.status, "failed")
        self.assertEqual(calls, ["inspect", "edit"])
        self.assertEqual(result.failure_reason, "focused check failed")
        self.assertEqual(result.revision.graph.node_map()["verify"].status, "skipped")
        self.assertEqual(result.revision.node_results["verify"]["blocked_by"], ["edit"])

    def test_waiting_approval_can_be_resumed_without_restarting_completed_nodes(self) -> None:
        self.store.create_revision(self.task.id, _graph())
        calls: list[str] = []

        def runner(context):
            calls.append(context.node.id)
            if context.node.id == "edit" and calls.count("edit") == 1:
                return NodeExecutionResult(
                    "waiting_approval",
                    metadata={"risk": "file_write"},
                )
            return NodeExecutionResult("completed", output=f"done:{context.node.id}")

        executor = PlanExecutor(self.store, runner)
        waiting = executor.execute(task_id=self.task.id)
        self.assertEqual(waiting.status, "waiting_approval")
        self.assertEqual(waiting.waiting_node_id, "edit")
        self.assertEqual(calls, ["inspect", "edit"])

        executor.resume_waiting_node(task_id=self.task.id, node_id="edit")
        completed = executor.execute(task_id=self.task.id)

        self.assertEqual(completed.status, "completed")
        self.assertEqual(calls, ["inspect", "edit", "edit", "verify"])

    def test_replan_supersedes_old_revision_and_preserves_completed_evidence(self) -> None:
        first = self.store.create_revision(self.task.id, _graph())
        running = self.store.update_node_status(first.id, "inspect", "running", expected_version=first.version)
        completed = self.store.update_node_status(
            running.id,
            "inspect",
            "completed",
            result={"output": "located"},
            expected_version=running.version,
        )
        second = self.store.create_replan(
            self.task.id,
            _graph(revision=2, statuses={"inspect": "completed"}),
            reason="The first verification target was unavailable.",
            expected_revision=1,
        )

        self.assertEqual(self.store.get_revision_by_id(first.id).status, "superseded")
        self.assertEqual(second.status, "active")
        self.assertEqual(second.graph.revision, 2)
        self.assertEqual(second.node_results["inspect"]["output"], "located")
        self.assertEqual(second.replan_reason, "The first verification target was unavailable.")
        self.assertGreater(second.version, 0)
        self.assertEqual(completed.graph.node_map()["inspect"].status, "completed")

    def test_replan_forces_completed_nodes_to_remain_completed(self) -> None:
        first = self.store.create_revision(self.task.id, _graph())
        running = self.store.update_node_status(first.id, "inspect", "running", expected_version=first.version)
        completed = self.store.update_node_status(
            running.id,
            "inspect",
            "completed",
            result={"output": "located"},
            expected_version=running.version,
        )

        second = self.store.create_replan(
            self.task.id,
            _graph(revision=2),
            reason="Replan the unfinished portion.",
            expected_revision=1,
        )

        self.assertEqual(second.graph.node_map()["inspect"].status, "completed")
        self.assertEqual(second.graph.node_map()["edit"].status, "pending")
        self.assertEqual(second.node_results["inspect"]["output"], "located")
        self.assertEqual(self.store.get_revision_by_id(completed.id).status, "superseded")


if __name__ == "__main__":
    unittest.main()

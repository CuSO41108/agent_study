from __future__ import annotations

import shutil
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from agent_app.plan import (
    PlanPlanner,
    PlanRecoveryService,
    PlanRevisionConflict,
    PlanRevisionLeaseConflict,
    PlanStore,
    PlanTaskService,
    RecoveryKind,
    parse_plan_graph,
)
from agent_app.runtime.task_runtime import TaskRuntime
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.types import ModelResponse, PendingAction, ToolCall, TurnResult


def _graph(*, status: str = "pending"):
    return parse_plan_graph(
        {
            "id": "recovery-plan",
            "revision": 1,
            "goal": "Recover one plan node safely.",
            "nodes": [
                {
                    "id": "inspect",
                    "kind": "inspect",
                    "objective": "Inspect the target.",
                    "depends_on": [],
                    "allowed_tools": ["file_read"],
                    "acceptance": ["The target is understood."],
                    "status": status,
                }
            ],
        }
    )


class _ResumePlannerModel:
    def generate(self, **_kwargs):
        return ModelResponse(assistant_text="{}")


class _ResumeLoop:
    def run_turn(self, **kwargs):
        return TurnResult(
            session_id=kwargs["session_id"],
            final_text="node resumed",
            stop_reason="final_response",
            tool_runs=[],
            success=True,
            task_id=kwargs["_task_id"],
            task_status="running",
        )


class PlanRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"plan_recovery_{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.db_path = self.root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.session_id = self.sessions.create_session("recovery-session")
        self.task = TaskRuntime(self.sessions).start_for_user_message(
            session_id=self.session_id,
            user_input="Recover one plan node safely.",
        )
        self.store = PlanStore(self.db_path)
        self.revision = self.store.create_revision(self.task.id, _graph())
        self.now = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
        self.recovery = PlanRecoveryService(
            plan_store=self.store,
            session_service=self.sessions,
            clock=lambda: self.now,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_lease_is_exclusive_and_can_be_taken_after_expiry(self) -> None:
        first = self.store.acquire_execution_lease(
            self.revision.id,
            owner="worker-a",
            ttl_seconds=30,
            now=self.now,
        )
        with self.assertRaises(PlanRevisionLeaseConflict):
            self.store.acquire_execution_lease(
                self.revision.id,
                owner="worker-b",
                ttl_seconds=30,
                now=self.now + timedelta(seconds=1),
            )

        expired = self.store.acquire_execution_lease(
            self.revision.id,
            owner="worker-b",
            ttl_seconds=30,
            now=self.now + timedelta(seconds=31),
        )
        self.assertEqual(expired.execution_lease.owner, "worker-b")
        self.assertGreater(expired.execution_lease.version, first.execution_lease.version)

        heartbeat = self.store.heartbeat_execution_lease(
            expired.id,
            owner="worker-b",
            lease_version=expired.execution_lease.version,
            ttl_seconds=30,
            now=self.now + timedelta(seconds=32),
        )
        self.assertTrue(heartbeat.execution_lease.is_active(now=self.now + timedelta(seconds=32)))
        released = self.store.release_execution_lease(
            heartbeat.id,
            owner="worker-b",
            lease_version=heartbeat.execution_lease.version,
        )
        self.assertIsNone(released.execution_lease.owner)

    def test_inspect_distinguishes_ready_and_interrupted_without_writing_status(self) -> None:
        ready = self.recovery.inspect(self.task.id)
        self.assertEqual(ready.kind, RecoveryKind.READY_TO_RESUME)

        running = self.store.update_node_status(
            self.revision.id,
            "inspect",
            "running",
            expected_version=self.revision.version,
        )
        expired = self.store.acquire_execution_lease(
            running.id,
            owner="crashed-process",
            ttl_seconds=1,
            now=self.now - timedelta(seconds=10),
        )
        decision = self.recovery.inspect(self.task.id)
        self.assertEqual(decision.kind, RecoveryKind.INTERRUPTED)
        self.assertEqual(decision.node_id, "inspect")
        persisted = self.store.get_revision_by_id(expired.id)
        self.assertEqual(persisted.graph.node_map()["inspect"].status, "running")

    def test_waiting_user_is_classified_from_pending_action_and_node(self) -> None:
        waiting = self.store.update_node_status(
            self.revision.id,
            "inspect",
            "running",
            expected_version=self.revision.version,
        )
        waiting = self.store.update_node_status(
            waiting.id,
            "inspect",
            "waiting_approval",
            expected_version=waiting.version,
        )
        TaskRuntime(self.sessions).wait_for_user(
            self.task.id,
            PendingAction(
                kind="ask_user",
                prompt="Which file should be inspected?",
                decision={"plan_node_id": "inspect"},
            ),
        )
        decision = self.recovery.inspect(self.task.id)
        self.assertEqual(decision.kind, RecoveryKind.WAITING_USER_ANSWER)
        self.assertEqual(decision.pending_action_id, self.sessions.get_task(self.task.id).pending_action.id)

    def test_expired_running_write_cannot_be_rewound(self) -> None:
        running = self.store.update_node_status(
            self.revision.id,
            "inspect",
            "running",
            expected_version=self.revision.version,
        )
        self.store.acquire_execution_lease(
            running.id,
            owner="crashed-process",
            ttl_seconds=1,
            now=self.now - timedelta(seconds=10),
        )
        self.sessions.prepare_tool_action(
            self.session_id,
            agent_id="main",
            tool_call=ToolCall(id="write-1", name="file_write", arguments={"path": "x", "content": "y"}),
            recovery_metadata={"side_effect": True, "plan_node_id": "inspect"},
            task_id=self.task.id,
        )
        decision = self.recovery.inspect(self.task.id)
        self.assertFalse(self.recovery.rewind_is_safe(decision))
        with self.assertRaises(PlanRevisionConflict):
            self.store.rewind_running_node_to_pending(
                running.id,
                "inspect",
                expected_version=running.version,
                now=self.now,
            )

    def test_plan_task_service_explicitly_rewinds_safe_interruption_then_executes(self) -> None:
        running = self.store.update_node_status(
            self.revision.id,
            "inspect",
            "running",
            expected_version=self.revision.version,
        )
        self.store.acquire_execution_lease(
            running.id,
            owner="crashed-process",
            ttl_seconds=1,
            now=self.now - timedelta(seconds=10),
        )
        service = PlanTaskService(
            planner=PlanPlanner(_ResumePlannerModel()),
            plan_store=self.store,
            session_service=self.sessions,
            agent_loop=_ResumeLoop(),
            recovery_service=self.recovery,
        )

        result = service.resume(task_id=self.task.id)

        self.assertEqual(result.execution.status, "completed")
        self.assertEqual(result.task.status, "completed")
        self.assertEqual(result.revision.graph.node_map()["inspect"].status, "completed")


if __name__ == "__main__":
    unittest.main()

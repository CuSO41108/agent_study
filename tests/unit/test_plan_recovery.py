from __future__ import annotations

import io
import shutil
import unittest
from contextlib import redirect_stdout
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from agent_app import cli
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
from agent_app.types import ModelResponse, PendingAction, ToolCall, ToolResult, TurnResult


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


class _NoReplayLoop:
    def run_turn(self, **_kwargs):
        raise AssertionError("A confirmed completed side effect must not be replayed.")


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

    def test_failed_tool_action_resolution_allows_explicit_retry(self) -> None:
        running, action = self._prepare_interrupted_side_effect(status="uncertain")
        service = PlanTaskService(
            planner=PlanPlanner(_ResumePlannerModel()),
            plan_store=self.store,
            session_service=self.sessions,
            agent_loop=_ResumeLoop(),
            recovery_service=self.recovery,
        )

        candidates = service.list_tool_action_resolution_candidates(task_id=self.task.id)
        resolution = service.resolve_tool_action(
            task_id=self.task.id,
            action_id=action.id,
            outcome="failed",
            reason="The target still has its original content.",
            evidence="sha256:before",
            resolved_by="test-user",
        )
        repeated_resolution = service.resolve_tool_action(
            task_id=self.task.id,
            action_id=action.id,
            outcome="failed",
            reason="The target still has its original content.",
            evidence="sha256:before",
            resolved_by="test-user",
        )
        result = service.resume(task_id=self.task.id)

        self.assertEqual([item.id for item in candidates], [action.id])
        self.assertEqual(resolution.action.status, "failed")
        self.assertEqual(repeated_resolution.action, resolution.action)
        self.assertEqual(resolution.decision.kind, RecoveryKind.INTERRUPTED)
        self.assertEqual(result.execution.status, "completed")
        self.assertEqual(result.task.status, "completed")
        trace_types = [item.trace_type for item in self.sessions.list_task_traces(self.task.id)]
        self.assertEqual(trace_types.count("tool_action_resolution"), 1)
        self.assertIn("plan_recovery_rewind", trace_types)
        self.assertEqual(self.store.get_revision_by_id(running.id).status, "completed")

    def test_succeeded_tool_action_resolution_completes_node_without_replay(self) -> None:
        _running, action = self._prepare_interrupted_side_effect(status="executing")
        service = PlanTaskService(
            planner=PlanPlanner(_ResumePlannerModel()),
            plan_store=self.store,
            session_service=self.sessions,
            agent_loop=_NoReplayLoop(),
            recovery_service=self.recovery,
        )

        resolution = service.resolve_tool_action(
            task_id=self.task.id,
            action_id=action.id,
            outcome="succeeded",
            reason="The intended content is present.",
            evidence="src/module.py sha256:after",
            resolved_by="test-user",
        )
        result = service.resume(task_id=self.task.id)

        self.assertEqual(resolution.action.status, "succeeded")
        self.assertIn("without replaying", resolution.decision.reason)
        self.assertEqual(result.execution.status, "completed")
        self.assertEqual(result.task.status, "completed")
        node_result = result.revision.node_results["inspect"]
        self.assertEqual(node_result["metadata"]["action_id"], action.id)
        self.assertEqual(node_result["output"], "src/module.py sha256:after")
        trace_types = [item.trace_type for item in self.sessions.list_task_traces(self.task.id)]
        self.assertIn("tool_action_resolution", trace_types)
        self.assertIn("plan_recovery_accept_effect", trace_types)

    def test_resolution_rejects_action_not_owned_by_interrupted_node(self) -> None:
        self._prepare_interrupted_side_effect(status="executing")
        unrelated = self.sessions.prepare_tool_action(
            self.session_id,
            agent_id="main",
            tool_call=ToolCall(id="other-write", name="file_write", arguments={"path": "other.py"}),
            recovery_metadata={"side_effect": True, "plan_node_id": "other-node"},
            task_id=self.task.id,
        )
        self.sessions.mark_tool_action_executing(unrelated.id)
        service = PlanTaskService(
            planner=PlanPlanner(_ResumePlannerModel()),
            plan_store=self.store,
            session_service=self.sessions,
            agent_loop=_NoReplayLoop(),
            recovery_service=self.recovery,
        )

        with self.assertRaisesRegex(ValueError, "not an unresolved side effect"):
            service.resolve_tool_action(
                task_id=self.task.id,
                action_id=unrelated.id,
                outcome="failed",
                reason="No effect found.",
                evidence="manual inspection",
                resolved_by="test-user",
            )

    def test_repl_lists_and_resolves_current_interrupted_action(self) -> None:
        _running, action = self._prepare_interrupted_side_effect(status="uncertain")
        service = PlanTaskService(
            planner=PlanPlanner(_ResumePlannerModel()),
            plan_store=self.store,
            session_service=self.sessions,
            agent_loop=_NoReplayLoop(),
            recovery_service=self.recovery,
        )
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            cli._handle_tool_action_resolution_command(
                raw_target="",
                plan_service=service,
                session_service=self.sessions,
                session_id=self.session_id,
            )
            cli._handle_tool_action_resolution_command(
                raw_target=(
                    f'{action.id[:8]} failed "Original hash still present" '
                    "-- src/module.py sha256:before"
                ),
                plan_service=service,
                session_service=self.sessions,
                session_id=self.session_id,
            )

        output = stdout.getvalue()
        self.assertIn(action.id, output)
        self.assertIn("Resolved ToolAction", output)
        self.assertIn("Run /resume", output)
        resolved = self.sessions.get_tool_action(action.id)
        self.assertEqual(resolved.status, "failed")
        self.assertEqual(resolved.resolution.reason, "Original hash still present")

    def _prepare_interrupted_side_effect(self, *, status: str):
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
        action = self.sessions.prepare_tool_action(
            self.session_id,
            agent_id="main",
            tool_call=ToolCall(id=f"write-{status}", name="file_write", arguments={"path": "x", "content": "y"}),
            recovery_metadata={"side_effect": True, "plan_node_id": "inspect"},
            task_id=self.task.id,
        )
        action = self.sessions.mark_tool_action_executing(action.id)
        if status == "uncertain":
            action = self.sessions.complete_tool_action(
                action.id,
                status="uncertain",
                tool_result=ToolResult(
                    tool_call_id=action.tool_call_id,
                    tool_name=action.tool_name,
                    success=False,
                    content="",
                    error="Interrupted side effect is uncertain.",
                ),
            )
        return running, action


if __name__ == "__main__":
    unittest.main()

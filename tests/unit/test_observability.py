from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.observability import (
    ReplayModeError,
    export_task_trace,
    render_task_timeline,
    replay_task_trace,
)
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.types import ToolCall


class ObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"trace_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        database_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(database_path)
        self.sessions = SessionService(database_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_export_and_render_task_trace(self) -> None:
        session_id = self.sessions.create_session("trace-session")
        task = self.sessions.create_task(session_id, goal="inspect the project")
        self.sessions.append_task_trace(task.id, "model_call", {"phase": "policy", "model": "test", "total_tokens": 42, "duration_ms": 12})
        self.sessions.append_task_trace(
            task.id,
            "checkpoint",
            {
                "phase": "planning",
                "request_status": "retrying",
                "attempt": 1,
                "max_attempts": 3,
                "error_detail": "temporary connection reset",
            },
        )
        self.sessions.append_task_trace(task.id, "approval", {"tool": "shell", "decision": "approve"})
        self.sessions.append_task_trace(task.id, "tool_attempt", {"tool": "shell", "success": True, "duration_ms": 25})
        self.sessions.append_task_trace(task.id, "plan_node_transition", {"node_id": "inspect", "from_status": "pending", "to_status": "completed"})
        self.sessions.append_task_trace(task.id, "plan_execution", {"revision": 1, "status": "completed", "executed_node_ids": ["inspect"]})
        self.sessions.append_task_trace(task.id, "plan_replan", {"from_revision": 1, "to_revision": 2, "reason": "new evidence"})
        self.sessions.append_task_trace(task.id, "plan_node_user_message", {"node_id": "inspect", "to_status": "completed"})
        self.sessions.append_task_trace(task.id, "plan_replan_failed", {"error_type": "PlanPlanningError", "error": "invalid plan"})
        self.sessions.append_task_trace(task.id, "planner_recovery_available", {"operation": "replan", "error_type": "invalid_plan"})
        self.sessions.append_task_trace(task.id, "planner_recovery_started", {"operation": "replan", "continuation": 1, "max_continuations": 2})
        self.sessions.append_task_trace(task.id, "planner_recovery_completed", {"operation": "replan", "revision": 2})

        trace = export_task_trace(self.sessions, task.id)
        rendered = render_task_timeline(trace)

        self.assertEqual(trace["schema_version"], 1)
        self.assertEqual(trace["trace_id"], task.id)
        self.assertGreaterEqual(len(trace["events"]), 4)
        self.assertIn("Trace:", rendered)
        self.assertIn("model_call", rendered)
        self.assertIn("planning / retrying / attempt 1/3 / temporary connection reset", rendered)
        self.assertIn("shell / success / 25 ms", rendered)
        self.assertIn("inspect / pending → completed", rendered)
        self.assertIn("revision 1 / completed / nodes: 1", rendered)
        self.assertIn("revision 1 → 2 / new evidence", rendered)
        self.assertIn("inspect / user answer / completed", rendered)
        self.assertIn("PlanPlanningError / invalid plan", rendered)
        self.assertIn("replan / paused / invalid_plan / explicit /continue required", rendered)
        self.assertIn("replan / continuation 1/2", rendered)
        self.assertIn("replan / recovered / revision 2", rendered)

    def test_export_rejects_unknown_task(self) -> None:
        with self.assertRaises(KeyError):
            export_task_trace(self.sessions, "missing")

    def test_audit_replay_is_read_only_and_checks_side_effect_actions(self) -> None:
        session_id = self.sessions.create_session("replay-session")
        task = self.sessions.create_task(session_id, goal="audit a safe file edit")
        self.sessions.append_task_trace(
            task.id,
            "tool_attempt",
            {
                "tool_call_id": "write-1",
                "tool": "file_write",
                "side_effect": True,
                "success": False,
            },
        )
        action = self.sessions.prepare_tool_action(
            session_id,
            agent_id="main",
            tool_call=ToolCall(
                id="write-1",
                name="file_write",
                arguments={"path": "README.md", "content": "new"},
            ),
            recovery_metadata={"side_effect": True},
            task_id=task.id,
        )

        before_traces = self.sessions.list_task_traces(task.id)
        before_action = self.sessions.get_tool_action(action.id)
        report = replay_task_trace(self.sessions, task.id, mode="audit")

        self.assertEqual(report["result"], "attention_required")
        self.assertTrue(report["read_only"])
        self.assertFalse(report["execution_performed"])
        self.assertEqual(report["persisted_action_count"], 1)
        self.assertTrue(any("remains prepared" in item for item in report["findings"]))
        self.assertEqual(self.sessions.list_task_traces(task.id), before_traces)
        self.assertEqual(self.sessions.get_tool_action(action.id), before_action)

    def test_dry_replay_never_executes_tools_and_live_mode_is_rejected(self) -> None:
        session_id = self.sessions.create_session("dry-replay-session")
        task = self.sessions.create_task(session_id, goal="dry replay")
        self.sessions.append_task_trace(task.id, "model_call", {"phase": "policy"})
        self.sessions.append_task_trace(
            task.id,
            "tool_attempt",
            {"tool_call_id": "read-1", "tool": "file_read", "success": True},
        )

        report = replay_task_trace(self.sessions, task.id, mode="dry")

        self.assertEqual(report["result"], "passed")
        self.assertEqual(report["replay_mode"], "dry")
        self.assertTrue(all(step["execution"] == "skipped" for step in report["steps"]))
        self.assertTrue(all("never invokes" in step["reason"] for step in report["steps"]))
        with self.assertRaises(ReplayModeError):
            replay_task_trace(self.sessions, task.id, mode="live")


if __name__ == "__main__":
    unittest.main()

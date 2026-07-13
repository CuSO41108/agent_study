from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.observability import export_task_trace, render_task_timeline
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService


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
        self.sessions.append_task_trace(task.id, "approval", {"tool": "shell", "decision": "approve"})
        self.sessions.append_task_trace(task.id, "tool_attempt", {"tool": "shell", "success": True, "duration_ms": 25})

        trace = export_task_trace(self.sessions, task.id)
        rendered = render_task_timeline(trace)

        self.assertEqual(trace["schema_version"], 1)
        self.assertEqual(trace["trace_id"], task.id)
        self.assertGreaterEqual(len(trace["events"]), 4)
        self.assertIn("Trace:", rendered)
        self.assertIn("model_call", rendered)
        self.assertIn("shell / success / 25 ms", rendered)

    def test_export_rejects_unknown_task(self) -> None:
        with self.assertRaises(KeyError):
            export_task_trace(self.sessions, "missing")


if __name__ == "__main__":
    unittest.main()

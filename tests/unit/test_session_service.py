from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.types import Message, ToolResult


class SessionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[2] / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp_dir = temp_root / f"session_{uuid4().hex}"
        self.temp_dir.mkdir()
        self.db_path = self.temp_dir / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_get_or_create_session_reuses_existing_session(self) -> None:
        session_id = self.sessions.create_session("session-1")

        same_session_id = self.sessions.get_or_create_session("session-1")

        self.assertEqual(same_session_id, session_id)

    def test_append_message_and_read_recent_messages(self) -> None:
        session_id = self.sessions.create_session("session-1")
        self.sessions.append_message(
            session_id,
            Message(role="user", content="hello"),
        )
        self.sessions.append_message(
            session_id,
            Message(role="assistant", content="world"),
        )

        messages = self.sessions.list_recent_messages(session_id)

        self.assertEqual(
            messages,
            [
                Message(role="user", content="hello"),
                Message(role="assistant", content="world"),
            ],
        )

    def test_append_tool_run_and_list_tool_runs(self) -> None:
        session_id = self.sessions.create_session("session-1")
        tool_result = ToolResult(
            tool_call_id="call-1",
            tool_name="file_read",
            success=True,
            content="README",
            error=None,
        )

        self.sessions.append_tool_run(session_id, tool_result)
        tool_runs = self.sessions.list_tool_runs(session_id)

        self.assertEqual(tool_runs, [tool_result])

    def test_append_subagent_run_and_list_subagent_runs(self) -> None:
        parent_session_id = self.sessions.create_session("parent-session")
        child_session_id = self.sessions.create_session("child-session")

        self.sessions.append_subagent_run(
            parent_session_id=parent_session_id,
            parent_tool_call_id="call-1",
            child_session_id=child_session_id,
            agent_id="worker_agent",
            task="Inspect README.md",
            success=True,
            result_summary="child_session_id=child-session\nagent_id=worker_agent\nsuccess=true",
        )

        runs = self.sessions.list_subagent_runs(parent_session_id)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].parent_session_id, parent_session_id)
        self.assertEqual(runs[0].parent_tool_call_id, "call-1")
        self.assertEqual(runs[0].child_session_id, child_session_id)
        self.assertEqual(runs[0].agent_id, "worker_agent")
        self.assertEqual(runs[0].task, "Inspect README.md")
        self.assertTrue(runs[0].success)
        self.assertIn("success=true", runs[0].result_summary)
        self.assertTrue(runs[0].created_at)

    def test_recent_messages_limit_defaults_to_sixteen(self) -> None:
        session_id = self.sessions.create_session("session-1")
        for index in range(20):
            self.sessions.append_message(
                session_id,
                Message(role="user", content=f"msg-{index}"),
            )

        messages = self.sessions.list_recent_messages(session_id)

        self.assertEqual(len(messages), 16)
        self.assertEqual(messages[0].content, "msg-4")
        self.assertEqual(messages[-1].content, "msg-19")


if __name__ == "__main__":
    unittest.main()

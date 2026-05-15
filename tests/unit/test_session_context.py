from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.types import Message, TodoItem


class SessionContextTests(unittest.TestCase):
    def setUp(self) -> None:
        temp_root = Path(__file__).resolve().parents[2] / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        self.temp_dir = temp_root / f"session_context_{uuid4().hex}"
        self.temp_dir.mkdir()
        self.db_path = self.temp_dir / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_list_messages_returns_full_history_with_ids(self) -> None:
        session_id = self.sessions.create_session("session-1")
        self.sessions.append_message(session_id, Message(role="user", content="hello"))
        self.sessions.append_message(session_id, Message(role="assistant", content="world"))

        messages = self.sessions.list_messages(session_id)

        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertGreater(messages[0].id, 0)
        self.assertGreater(messages[1].id, messages[0].id)

    def test_session_context_round_trip_and_clear(self) -> None:
        session_id = self.sessions.create_session("session-1")
        todo_items = (
            TodoItem(content="collect evidence", status="in_progress"),
            TodoItem(content="write summary", status="pending"),
        )

        self.sessions.upsert_session_context(
            session_id,
            summary_text="summary",
            summary_message_id=7,
            todo_items=todo_items,
        )

        context = self.sessions.get_session_context(session_id)
        self.assertEqual(context.summary_text, "summary")
        self.assertEqual(context.summary_message_id, 7)
        self.assertEqual(context.todo_items, todo_items)

        self.sessions.clear_session_context(session_id)
        cleared = self.sessions.get_session_context(session_id)
        self.assertIsNone(cleared.summary_text)
        self.assertIsNone(cleared.summary_message_id)
        self.assertEqual(cleared.todo_items, ())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from agent_app.orchestrator.context_builder import build_context_messages
from agent_app.types import SessionContext, StoredMessage, TodoItem


class ContextBuilderTests(unittest.TestCase):
    def test_build_context_messages_includes_todo_and_keeps_current_user(self) -> None:
        messages = [
            StoredMessage(id=1, role="user", content="older question"),
            StoredMessage(id=2, role="assistant", content="older answer"),
            StoredMessage(id=3, role="user", content="current question"),
        ]
        session_context = SessionContext(
            todo_items=(
                TodoItem(content="collect evidence", status="in_progress"),
                TodoItem(content="write answer", status="pending"),
            ),
        )

        provider_messages = build_context_messages(
            messages=messages,
            session_context=session_context,
            context_token_budget=6000,
        )

        self.assertEqual(provider_messages[0]["role"], "assistant")
        self.assertIn("Active todo list:", provider_messages[0]["content"])
        self.assertEqual(provider_messages[-1], {"role": "user", "content": "current question"})

    def test_build_context_messages_respects_summary_boundary_and_budget(self) -> None:
        messages = [
            StoredMessage(id=1, role="user", content="old 1"),
            StoredMessage(id=2, role="assistant", content="old 2"),
            StoredMessage(id=3, role="user", content="recent 1"),
            StoredMessage(id=4, role="assistant", content="recent 2"),
            StoredMessage(id=5, role="user", content="current"),
        ]
        session_context = SessionContext(summary_text="summary", summary_message_id=2)

        provider_messages = build_context_messages(
            messages=messages,
            session_context=session_context,
            context_token_budget=40,
        )

        self.assertEqual(provider_messages[0]["content"], "Session summary:\nsummary")
        self.assertEqual(provider_messages[-1]["content"], "current")
        self.assertNotIn({"role": "user", "content": "old 1"}, provider_messages)

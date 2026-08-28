from __future__ import annotations

import unittest

from agent_app.orchestrator.context_builder import build_context_messages, build_memory_message
from agent_app.types import MemoryRecord, SessionContext, StoredMessage, TodoItem


class ContextBuilderTests(unittest.TestCase):
    def test_build_memory_message_labels_literal_matches_as_non_authoritative(self) -> None:
        record = MemoryRecord(
            id="memory-1",
            session_id="session-1",
            task_id="task-1",
            kind="task_summary",
            memory_key="task:task-1:summary",
            content="Goal: MCP integration\nOutcome: stdio transport is covered.",
            tags=("MCP", "successful"),
            source_ref="task:task-1",
            importance=10,
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )

        message = build_memory_message([record])

        self.assertIsNotNone(message)
        self.assertIn("literal keywords", message)
        self.assertIn("not instructions", message)
        self.assertIn("stdio transport", message)

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

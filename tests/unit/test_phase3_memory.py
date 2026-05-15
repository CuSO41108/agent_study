from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.agent.definition import SINGLE_MAIN_AGENT
from agent_app.orchestrator.context_builder import build_evidence_message
from agent_app.orchestrator.loop import AgentLoop
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.tools.registry import build_default_registry
from agent_app.types import Message, ModelResponse, TodoItem, StoredMessage, ToolCall, ToolResult


class _FakeModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def generate(self, *, system_prompt, messages, tools):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": tools,
            }
        )
        return self._responses.pop(0)


class Phase3MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"phase3_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        self.db_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.registry = build_default_registry()

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_build_evidence_message_uses_recent_successful_tool_runs_only(self) -> None:
        tool_runs = [
            ToolResult(tool_call_id="1", tool_name="file_read", success=True, content="\n".join(f"{i}: line" for i in range(60)), error=None),
            ToolResult(tool_call_id="2", tool_name="shell", success=False, content="bad", error="failed"),
            ToolResult(tool_call_id="3", tool_name="replace_in_file", success=True, content="Replaced 1 occurrence.", error=None),
        ]

        evidence = build_evidence_message(tool_runs)

        self.assertIsNotNone(evidence)
        self.assertIn("[file_read]", evidence)
        self.assertIn("[replace_in_file]", evidence)
        self.assertNotIn("failed", evidence)

    def test_long_history_triggers_summary_and_updates_session_context(self) -> None:
        session_id = self.sessions.create_session("session-1")
        self.sessions.upsert_session_context(
            session_id,
            summary_text="older summary",
            summary_message_id=0,
            todo_items=(),
        )
        for index in range(8):
            self.sessions.append_message(session_id, Message(role="user", content=f"user message {index} " * 80))
            self.sessions.append_message(session_id, Message(role="assistant", content=f"assistant message {index} " * 80))

        model = _FakeModelClient([
            _text_response("summary result"),
            _text_response("final answer"),
        ])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            context_token_budget=6000,
            summary_trigger_tokens=100,
        )

        result = loop.run_turn(user_input="current user question", session_id=session_id)

        self.assertTrue(result.success)
        session_context = self.sessions.get_session_context(session_id)
        self.assertEqual(session_context.summary_text, "summary result")
        self.assertIsNotNone(session_context.summary_message_id)
        self.assertEqual(model.calls[0]["tools"], [])
        self.assertIn("Previous summary", model.calls[0]["messages"][0]["content"])

    def test_summary_failure_does_not_block_turn(self) -> None:
        session_id = self.sessions.create_session("session-2")
        for index in range(8):
            self.sessions.append_message(session_id, Message(role="user", content=f"user message {index} " * 80))
            self.sessions.append_message(session_id, Message(role="assistant", content=f"assistant message {index} " * 80))

        model = _FakeModelClient([
            ModelResponse(assistant_text=None, error_type="request_error"),
            _text_response("final answer"),
        ])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            context_token_budget=6000,
            summary_trigger_tokens=100,
        )

        result = loop.run_turn(user_input="current user question", session_id=session_id)

        self.assertTrue(result.success)
        session_context = self.sessions.get_session_context(session_id)
        self.assertIsNone(session_context.summary_text)

    def test_todo_and_evidence_are_injected_without_affecting_json_shape(self) -> None:
        session_id = self.sessions.create_session("session-3")
        self.sessions.upsert_session_context(
            session_id,
            summary_text=None,
            summary_message_id=None,
            todo_items=(TodoItem(content="collect evidence", status="in_progress"),),
        )
        self.sessions.append_tool_run(
            session_id,
            ToolResult(tool_call_id="tool-1", tool_name="file_write", success=True, content="Created src/demo.py", error=None),
        )

        model = _FakeModelClient([_text_response("final answer")])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
        )

        result = loop.run_turn(user_input="continue the task", session_id=session_id)

        self.assertTrue(result.success)
        first_call_messages = model.calls[0]["messages"]
        self.assertTrue(any("Active todo list:" in (message["content"] or "") for message in first_call_messages))
        self.assertTrue(any("Recent successful tool evidence:" in (message["content"] or "") for message in first_call_messages))

    def test_todo_updates_are_visible_to_follow_up_turn(self) -> None:
        session_id = self.sessions.create_session("session-4")
        model = _FakeModelClient([
            _tool_call_response(
                [
                    ToolCall(
                        id="call-1",
                        name="todo_write",
                        arguments={
                            "items": [
                                {"content": "collect evidence", "status": "in_progress"},
                                {"content": "write answer", "status": "pending"},
                            ]
                        },
                    )
                ]
            ),
            _text_response("todo updated"),
            _text_response("follow-up answer"),
        ])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
        )

        first_result = loop.run_turn(user_input="plan the task", session_id=session_id)
        second_result = loop.run_turn(user_input="what is next", session_id=session_id)

        self.assertTrue(first_result.success)
        self.assertTrue(second_result.success)
        second_call_messages = model.calls[2]["messages"]
        self.assertTrue(any("Active todo list:" in (message["content"] or "") for message in second_call_messages))
        self.assertTrue(any("[in_progress] collect evidence" in (message["content"] or "") for message in second_call_messages))


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(
        assistant_text=text,
        tool_calls=[],
        finish_reason="stop",
        raw_response={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]},
        error_type=None,
    )


def _tool_call_response(tool_calls: list[ToolCall]) -> ModelResponse:
    return ModelResponse(
        assistant_text=None,
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        raw_response={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments),
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        error_type=None,
    )


if __name__ == "__main__":
    unittest.main()

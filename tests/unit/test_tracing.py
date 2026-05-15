from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.agent.definition import SINGLE_MAIN_AGENT
from agent_app.orchestrator.loop import AgentLoop
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.tools.registry import build_default_registry
from agent_app.types import ModelResponse, ToolCall, ToolResult


class _FakeModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    def generate(self, *, system_prompt, messages, tools):
        return self._responses.pop(0)


class TracingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"tracing_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        src_dir = self.workspace_root / "src"
        src_dir.mkdir()
        (src_dir / "module.py").write_text("print('old')\n", encoding="utf-8")
        self.db_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.registry = build_default_registry()

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_turn_and_tool_traces_are_written(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_read", arguments={"path": "src/module.py"})]),
            _text_response("done"),
        ])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
        )

        result = loop.run_turn(user_input="read the file", session_id=None)

        self.assertTrue(result.success)
        turn_traces = self.sessions.list_turn_traces(result.session_id)
        self.assertEqual(len(turn_traces), 1)
        self.assertEqual(turn_traces[0].user_input, "read the file")
        self.assertEqual(turn_traces[0].stop_reason, "final_response")
        self.assertFalse(turn_traces[0].used_summary)

        tool_traces = self.sessions.list_tool_call_traces(turn_traces[0].id)
        self.assertEqual(len(tool_traces), 1)
        self.assertEqual(tool_traces[0].tool_name, "file_read")
        self.assertTrue(tool_traces[0].success)
        self.assertTrue(tool_traces[0].content_preview.startswith("1: print('old')"))

    def test_content_preview_is_truncated(self) -> None:
        long_content = "x" * 700
        self.sessions.append_turn_trace(
            "session-1",
            user_input="u",
            context_message_count=1,
            context_token_estimate=50,
            used_summary=False,
            used_todo=False,
            used_evidence=False,
            final_text="done",
            stop_reason="final_response",
            success=True,
            tool_traces=[
                ToolResult(tool_call_id="call-1", tool_name="shell", success=True, content=long_content, error=None),
            ],
        )

        turn_trace = self.sessions.list_turn_traces("session-1")[0]
        tool_trace = self.sessions.list_tool_call_traces(turn_trace.id)[0]
        self.assertEqual(len(tool_trace.content_preview), 500)

    def test_trace_records_summary_todo_and_evidence_flags(self) -> None:
        session_id = self.sessions.create_session("session-flags")
        self.sessions.upsert_session_context(
            session_id,
            summary_text="summary",
            summary_message_id=0,
            todo_items=(
                __import__("agent_app.types", fromlist=["TodoItem"]).TodoItem(content="collect evidence", status="in_progress"),
            ),
        )
        self.sessions.append_tool_run(
            session_id,
            ToolResult(tool_call_id="tool-1", tool_name="file_write", success=True, content="Created src/demo.py", error=None),
        )
        model = _FakeModelClient([_text_response("done")])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
        )

        result = loop.run_turn(user_input="continue", session_id=session_id)

        turn_trace = self.sessions.list_turn_traces(result.session_id)[0]
        self.assertTrue(turn_trace.used_summary)
        self.assertTrue(turn_trace.used_todo)
        self.assertTrue(turn_trace.used_evidence)

    def test_tracing_failure_does_not_block_turn(self) -> None:
        model = _FakeModelClient([_text_response("done")])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
        )

        original_append = self.sessions.append_turn_trace
        self.sessions.append_turn_trace = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("trace fail"))  # type: ignore[assignment]
        try:
            result = loop.run_turn(user_input="hello", session_id=None)
        finally:
            self.sessions.append_turn_trace = original_append  # type: ignore[assignment]

        self.assertTrue(result.success)


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

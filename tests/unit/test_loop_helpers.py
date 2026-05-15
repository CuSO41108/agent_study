from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

import agent_app.orchestrator.loop as loop_module
from agent_app.agent.definition import SINGLE_MAIN_AGENT
from agent_app.orchestrator.loop import AgentLoop
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.tools.registry import build_default_registry
from agent_app.types import Message, ModelResponse, ToolCall, ToolResult


class _UnusedModelClient:
    def generate(self, *, system_prompt, messages, tools):
        raise AssertionError("generate should not be called in helper tests")


class LoopHelperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"loop_helpers_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        self.db_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=_UnusedModelClient(),
            tool_registry=build_default_registry(),
            session_service=self.sessions,
            workspace_root=self.workspace_root,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_answer_from_existing_evidence_handles_inventory_and_config_questions(self) -> None:
        inventory_runs = [
            ToolResult(tool_call_id="call-1", tool_name="file_read", success=True, content="1: class ToolRegistry\n2: def build_default_registry", error=None),
        ]
        inventory_answer = self.loop._answer_from_existing_evidence(
            user_input="which tools are available",
            tool_call=ToolCall(id="call-2", name="file_read", arguments={"path": "x"}),
            tool_runs=inventory_runs,
        )

        config_runs = [
            ToolResult(
                tool_call_id="call-3",
                tool_name="file_read",
                success=True,
                content='1: class AppConfig\n2: def database_path(self):\n3:     return self.workspace_root / ".agent_app" / "agent.db"',
                error=None,
            ),
        ]
        config_answer = self.loop._build_evidence_answer(
            user_input="database path config",
            tool_runs=config_runs,
            allow_file_read_excerpt=False,
        )

        self.assertIsNotNone(inventory_answer)
        self.assertIn("`replace_in_file`", inventory_answer)
        self.assertIsNotNone(config_answer)
        self.assertIn("AppConfig.database_path", config_answer)

    def test_location_and_file_excerpt_evidence_helpers(self) -> None:
        tool_runs = [
            ToolResult(
                tool_call_id="call-1",
                tool_name="code_search",
                success=True,
                content=f"{self.workspace_root / 'src' / 'app.py'}:12:print('hi')",
                error=None,
            ),
            ToolResult(tool_call_id="call-2", tool_name="file_read", success=True, content="1: alpha\n2: beta", error=None),
        ]

        answer_from_shell = self.loop._answer_from_existing_evidence(
            user_input="where is the file",
            tool_call=ToolCall(id="call-3", name="shell", arguments={"command": "Get-ChildItem src"}),
            tool_runs=tool_runs,
        )
        answer_from_excerpt = self.loop._build_evidence_answer(
            user_input="summarize the read evidence",
            tool_runs=tool_runs,
            allow_file_read_excerpt=True,
        )

        self.assertIsNotNone(answer_from_shell)
        self.assertIn("src\\app.py", answer_from_shell)
        self.assertIsNotNone(answer_from_excerpt)
        self.assertIn("1: alpha", answer_from_excerpt)

    def test_module_level_helper_functions_cover_fallback_paths(self) -> None:
        message_payload = loop_module._message_to_provider_message(Message(role="tool", content="ok", tool_call_id="call-1"))
        fallback_message = loop_module._assistant_tool_message(
            ModelResponse(
                assistant_text="fallback",
                tool_calls=[ToolCall(id="call-2", name="file_read", arguments={"path": "README.md"})],
                raw_response={"choices": [{}]},
            )
        )

        self.assertEqual(message_payload["tool_call_id"], "call-1")
        self.assertEqual(fallback_message["content"], "fallback")
        self.assertTrue(loop_module._looks_like_location_question("where is the file"))
        self.assertTrue(loop_module._looks_like_tool_inventory_question("which tools can you use"))
        self.assertTrue(loop_module._looks_like_config_question("which config determines the database path"))
        self.assertEqual(loop_module._extract_tool_inventory_answer([], SINGLE_MAIN_AGENT.allowed_tools), None)
        self.assertEqual(loop_module._extract_config_answer([]), None)
        self.assertEqual(loop_module._extract_file_read_excerpt([]), None)
        self.assertEqual(loop_module._extract_code_search_paths([], self.workspace_root), [])
        self.assertEqual(loop_module._relative_workspace_path("Z:\\outside.py", self.workspace_root), "Z:\\outside.py")

        crowded_runs = [
            ToolResult(
                tool_call_id=f"call-{index}",
                tool_name="code_search",
                success=True,
                content=f"{self.workspace_root / f'file{index}.py'}:{index}:x",
                error=None,
            )
            for index in range(1, 7)
        ]
        self.assertEqual(loop_module._extract_code_search_paths(crowded_runs, self.workspace_root), [])


if __name__ == "__main__":
    unittest.main()

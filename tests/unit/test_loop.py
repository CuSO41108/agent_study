from __future__ import annotations

import json
import subprocess
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from agent_app.agent.definition import SINGLE_MAIN_AGENT
from agent_app.orchestrator.loop import AgentLoop
from agent_app.orchestrator.subagent_runner import SubagentRunner
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.tools.registry import build_default_registry, build_root_registry
from agent_app.types import ModelResponse, ToolCall


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


class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"loop_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        src_dir = self.workspace_root / "src"
        src_dir.mkdir()
        agent_app_dir = src_dir / "agent_app"
        (agent_app_dir / "tools").mkdir(parents=True)
        (agent_app_dir / "agent").mkdir(parents=True)
        (self.workspace_root / "README.md").write_text("hello\nworld\n", encoding="utf-8")
        (src_dir / "module.py").write_text("print('old')\n", encoding="utf-8")
        (agent_app_dir / "tools" / "registry.py").write_text(
            "class ToolRegistry:\n"
            "    pass\n\n"
            "def build_default_registry() -> ToolRegistry:\n"
            "    return ToolRegistry([\n"
            "        FileReadTool(),\n"
            "        CodeSearchTool(),\n"
            "        DelegateTaskTool(),\n"
            "        TodoReadTool(),\n"
            "        TodoWriteTool(),\n"
            "        ReplaceInFileTool(),\n"
            "        FileWriteTool(),\n"
            "        ShellTool(),\n"
            "    ])\n",
            encoding="utf-8",
        )
        (agent_app_dir / "agent" / "definition.py").write_text(
            "class AgentDefinition:\n"
            "    pass\n\n"
            "allowed_tools=[\"file_read\", \"code_search\", \"delegate_task\", \"todo_read\", \"todo_write\", \"replace_in_file\", \"file_write\", \"shell\"]\n",
            encoding="utf-8",
        )
        (agent_app_dir / "config.py").write_text(
            "class AppConfig:\n"
            "    @property\n"
            "    def database_path(self):\n"
            "        return self.workspace_root / \".agent_app\" / \"agent.db\"\n",
            encoding="utf-8",
        )
        self.db_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.registry = build_default_registry()

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_run_turn_returns_direct_text_answer(self) -> None:
        model = _FakeModelClient([_text_response("done")])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="say done")

        self.assertTrue(result.success)
        self.assertEqual(result.final_text, "done")
        self.assertEqual(result.stop_reason, "final_response")
        self.assertEqual(len(model.calls), 1)

    def test_run_turn_executes_read_tool_and_returns_final_answer(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_read", arguments={"path": "README.md"})]),
            _text_response("README has two lines."),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="read the readme")

        self.assertTrue(result.success)
        self.assertEqual(result.final_text, "README has two lines.")
        self.assertEqual(len(result.tool_runs), 1)
        self.assertEqual(result.tool_runs[0].tool_name, "file_read")
        self.assertEqual(len(self.sessions.list_tool_runs(result.session_id)), 1)

    def test_multiple_tool_calls_run_serially_in_model_order(self) -> None:
        model = _FakeModelClient([
            _tool_call_response(
                [
                    ToolCall(id="call-1", name="file_read", arguments={"path": "README.md"}),
                    ToolCall(id="call-2", name="code_search", arguments={"pattern": "hello", "path": "."}),
                ]
            ),
            _text_response("done"),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="use two tools")

        self.assertTrue(result.success)
        self.assertEqual([tool_run.tool_call_id for tool_run in result.tool_runs], ["call-1", "call-2"])

    def test_tool_timeout_argument_sets_tool_context_timeout(self) -> None:
        model = _FakeModelClient([_text_response("done")])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            tool_timeout=9.5,
        )

        self.assertEqual(loop._tool_context.timeout, 9.5)

    def test_shell_timeout_remains_supported_as_compatibility_alias(self) -> None:
        model = _FakeModelClient([_text_response("done")])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            shell_timeout=7.5,
        )

        self.assertEqual(loop._tool_context.timeout, 7.5)

    def test_passing_both_tool_timeout_and_shell_timeout_is_rejected(self) -> None:
        model = _FakeModelClient([_text_response("done")])

        with self.assertRaisesRegex(ValueError, "Only one of 'tool_timeout' or 'shell_timeout'"):
            AgentLoop(
                agent=SINGLE_MAIN_AGENT,
                model_client=model,
                tool_registry=self.registry,
                session_service=self.sessions,
                workspace_root=self.workspace_root,
                tool_timeout=5.0,
                shell_timeout=5.0,
            )

    def test_run_turn_returns_model_error_when_provider_fails(self) -> None:
        model = _FakeModelClient([ModelResponse(assistant_text=None, raw_response=None, error_type="request_error")])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="say done")

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "model_error")

    def test_run_turn_returns_invalid_model_response_when_text_and_tools_are_missing(self) -> None:
        model = _FakeModelClient([ModelResponse(assistant_text=None, tool_calls=[], raw_response={"choices": [{"message": {"content": None}}]})])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="say done")

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "invalid_model_response")

    def test_confirmed_file_write_updates_file_and_returns_answer(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_write", arguments={"path": "src/module.py", "content": "print('new')\n"})]),
            _text_response("updated module"),
        ])
        loop = self._build_loop(model, confirmation_handler=lambda tool_call, context: True)

        result = loop.run_turn(user_input="update the module")

        self.assertTrue(result.success)
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")
        self.assertEqual(result.tool_runs[0].tool_name, "file_write")
        self.assertTrue(result.tool_runs[0].success)

    def test_missing_confirmation_handler_denies_file_write_but_turn_can_continue(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_write", arguments={"path": "src/module.py", "content": "print('new')\n"})]),
            _text_response("write denied"),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="update the module")

        self.assertTrue(result.success)
        self.assertEqual(result.final_text, "write denied")
        self.assertEqual(result.tool_runs[0].error, "Tool use denied by user.")

    def test_invalid_file_write_arguments_do_not_trigger_confirmation_handler(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_write", arguments={"content": "print('new')\n"})]),
            _text_response("bad request"),
        ])
        loop = self._build_loop(
            model,
            confirmation_handler=lambda tool_call, context: (_ for _ in ()).throw(AssertionError("confirmation should not run")),
        )

        result = loop.run_turn(user_input="update the module")

        self.assertTrue(result.success)
        self.assertEqual(result.tool_runs[0].error, "Invalid arguments: path is required.")

    def test_invalid_replace_in_file_arguments_do_not_trigger_confirmation_handler(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="replace_in_file", arguments={"path": "src/module.py", "new_text": "print('new')\n"})]),
            _text_response("bad request"),
        ])
        loop = self._build_loop(
            model,
            confirmation_handler=lambda tool_call, context: (_ for _ in ()).throw(AssertionError("confirmation should not run")),
        )

        result = loop.run_turn(user_input="update the module")

        self.assertTrue(result.success)
        self.assertEqual(result.tool_runs[0].error, "Invalid arguments: old_text is required.")

    def test_invalid_shell_arguments_do_not_call_approval_path(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="shell", arguments={"command": ""})]),
            _text_response("bad request"),
        ])
        loop = self._build_loop(model)

        with patch("agent_app.orchestrator.loop.approve_tool_call", side_effect=AssertionError("approval should not run")):
            result = loop.run_turn(user_input="run shell")

        self.assertTrue(result.success)
        self.assertEqual(result.tool_runs[0].error, "Invalid arguments: command must be a non-empty string.")

    @patch("agent_app.tools.code_search.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="rg", timeout=0.01))
    @patch("agent_app.tools.code_search.shutil.which", return_value="rg")
    def test_code_search_timeout_returns_tool_failure_without_crashing_loop(self, _mock_which, _mock_run) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="code_search", arguments={"pattern": "hello", "path": "."})]),
            _text_response("timeout handled"),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="search the repo")

        self.assertTrue(result.success)
        self.assertEqual(result.final_text, "timeout handled")
        self.assertEqual(result.tool_runs[0].error, "Code search timed out.")

    def test_replace_in_file_successfully_updates_existing_file(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
            _text_response("updated module"),
        ])
        prompts: list[str] = []

        def _confirm(tool_call, context):
            from agent_app.cli import _build_confirmation_prompt

            prompts.append(_build_confirmation_prompt(tool_call, context))
            return True

        loop = self._build_loop(model, confirmation_handler=_confirm)

        result = loop.run_turn(user_input="update the module")

        self.assertTrue(result.success)
        self.assertIn("Operation: replace", prompts[0])
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")

    def test_replace_in_file_ambiguous_failure_does_not_mutate_file(self) -> None:
        target = self.workspace_root / "src" / "module.py"
        target.write_text("same\nsame\n", encoding="utf-8")
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "same", "new_text": "new"})]),
            _text_response("replace failed"),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="update the module")

        self.assertTrue(result.success)
        self.assertEqual(result.tool_runs[0].error, "Ambiguous match: multiple occurrences found. Refine old_text or set replace_all=true.")
        self.assertEqual(target.read_text(encoding="utf-8"), "same\nsame\n")

    def test_replace_in_file_rechecks_file_after_confirmation(self) -> None:
        target = self.workspace_root / "src" / "module.py"
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
            _text_response("replace failed"),
        ])

        def _confirm(tool_call, context):
            from agent_app.cli import _build_confirmation_prompt

            _build_confirmation_prompt(tool_call, context)
            target.write_text("print('changed')\n", encoding="utf-8")
            return True

        loop = self._build_loop(model, confirmation_handler=_confirm)

        result = loop.run_turn(user_input="update the module")

        self.assertTrue(result.success)
        self.assertEqual(result.tool_runs[0].error, "Target file changed since inspection. Please retry the edit.")
        self.assertEqual(target.read_text(encoding="utf-8"), "print('changed')\n")

    def test_rejected_file_write_returns_failure_result_without_writing(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_write", arguments={"path": "src/module.py", "content": "print('new')\n"})]),
            _tool_call_response([ToolCall(id="call-2", name="file_write", arguments={"path": "src/module.py", "content": "print('new')\n"})]),
        ])
        loop = self._build_loop(model, confirmation_handler=lambda tool_call, context: False)

        result = loop.run_turn(user_input="update the module")

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "repeated_tool_failure")
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('old')\n")
        self.assertEqual(result.tool_runs[0].error, "Tool use denied by user.")

    def test_write_then_failed_shell_keeps_written_content(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_write", arguments={"path": "src/module.py", "content": "print('new')\n"})]),
            _tool_call_response([ToolCall(id="call-2", name="shell", arguments={"command": "python -c \"print(1)\""})]),
            _text_response("The change was written, but validation did not pass."),
        ])
        loop = self._build_loop(model, confirmation_handler=lambda tool_call, context: True)

        result = loop.run_turn(user_input="update and verify")

        self.assertTrue(result.success)
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")
        self.assertFalse(result.tool_runs[1].success)
        self.assertEqual(result.final_text, "The change was written, but validation did not pass.")

    def test_max_tool_rounds_stops_before_ninth_round(self) -> None:
        model = _FakeModelClient([_tool_call_response([ToolCall(id="call-1", name="file_read", arguments={"path": "README.md"})])] * 9)
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="loop forever")

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "max_tool_rounds_exceeded")
        self.assertEqual(len(result.tool_runs), 8)

    def test_repeated_failed_tool_stops_after_second_failure(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="shell", arguments={"command": "python -c \"print(1)\""})]),
            _tool_call_response([ToolCall(id="call-2", name="shell", arguments={"command": "python -c \"print(1)\""})]),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="try shell twice")

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "repeated_tool_failure")
        self.assertEqual(len(result.tool_runs), 2)
        self.assertFalse(result.tool_runs[0].success)
        self.assertFalse(result.tool_runs[1].success)

    def test_unknown_tool_call_is_rejected_by_agent_policy(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="missing_tool", arguments={})]),
            _text_response("done"),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="use a missing tool")

        self.assertTrue(result.success)
        self.assertEqual(result.tool_runs[0].error, "Tool 'missing_tool' is not allowed for this agent.")

    def test_unregistered_allowed_tool_returns_explicit_error(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="shell", arguments={"command": "git status --short"})]),
            _text_response("done"),
        ])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=build_default_registry().__class__([]),
            session_service=self.sessions,
            workspace_root=self.workspace_root,
        )

        result = loop.run_turn(user_input="use shell")

        self.assertTrue(result.success)
        self.assertEqual(result.tool_runs[0].error, "Tool 'shell' is not registered.")

    def test_location_question_answers_from_code_search_without_running_shell(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="code_search", arguments={"pattern": "print\\('old'\\)", "path": "."})]),
            _tool_call_response([ToolCall(id="call-2", name="shell", arguments={"command": "dir /b src"})]),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="module 在哪个文件里")

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "answered_from_evidence")
        self.assertEqual(len(result.tool_runs), 1)
        self.assertEqual(result.tool_runs[0].tool_name, "code_search")
        self.assertIn("src\\module.py", result.final_text)

    def test_repeated_failures_can_fall_back_to_file_read_evidence(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_read", arguments={"path": "README.md"})]),
            _tool_call_response([ToolCall(id="call-2", name="shell", arguments={"command": "python -c \"print(1)\""})]),
            _tool_call_response([ToolCall(id="call-3", name="shell", arguments={"command": "python -c \"print(1)\""})]),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="README 说了什么")

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "answered_from_evidence")
        self.assertEqual(len(result.tool_runs), 3)
        self.assertIn("1: hello", result.final_text)

    def test_tool_inventory_question_stops_after_registry_evidence(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_read", arguments={"path": "src/agent_app/tools/registry.py"})]),
            _tool_call_response([ToolCall(id="call-2", name="shell", arguments={"command": "Get-ChildItem src"})]),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="我当前项目可以调用哪些tool?")

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "answered_from_evidence")
        self.assertEqual(len(result.tool_runs), 1)
        self.assertEqual(result.tool_runs[0].tool_name, "file_read")
        self.assertIn("`file_read`", result.final_text)
        self.assertIn("`code_search`", result.final_text)
        self.assertIn("`replace_in_file`", result.final_text)
        self.assertIn("`file_write`", result.final_text)
        self.assertIn("`shell`", result.final_text)

    def test_max_tool_rounds_can_answer_from_existing_evidence(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id=f"call-{index}", name="code_search", arguments={"pattern": "print\\('old'\\)", "path": "."})])
            for index in range(1, 10)
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="module 在哪个文件里")

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "answered_from_evidence")
        self.assertEqual(len(result.tool_runs), 8)
        self.assertIn("src\\module.py", result.final_text)

    def test_tool_inventory_question_stops_early_with_allowed_tools_evidence(self) -> None:
        model = _FakeModelClient([_tool_call_response([ToolCall(id="call-1", name="file_read", arguments={"path": "src/agent_app/agent/definition.py"})])] * 9)
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="当前支持哪些工具")

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "answered_from_evidence")
        self.assertEqual(len(result.tool_runs), 1)
        self.assertIn("`replace_in_file`", result.final_text)
        self.assertIn("`shell`", result.final_text)

    def test_config_question_stops_after_authoritative_config_evidence(self) -> None:
        model = _FakeModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_read", arguments={"path": "src/agent_app/config.py"})]),
            _tool_call_response([ToolCall(id="call-2", name="shell", arguments={"command": "Get-ChildItem src"})]),
        ])
        loop = self._build_loop(model)

        result = loop.run_turn(user_input="数据库文件路径是在哪儿确定的？")

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "answered_from_evidence")
        self.assertEqual(len(result.tool_runs), 1)
        self.assertIn("AppConfig.database_path", result.final_text)
        self.assertIn(".agent_app", result.final_text)

    def test_delegate_task_runs_worker_and_persists_subagent_summary(self) -> None:
        model = _FakeModelClient([
            _tool_call_response(
                [
                    ToolCall(
                        id="call-1",
                        name="delegate_task",
                        arguments={
                            "task": "Inspect README.md",
                            "success_criteria": "Summarize the file contents.",
                            "relevant_paths": ["README.md"],
                        },
                    )
                ]
            ),
            _text_response("README has two lines."),
            _text_response("delegation complete"),
        ])
        loop = self._build_delegate_loop(model)

        result = loop.run_turn(user_input="inspect the readme with a worker")

        self.assertTrue(result.success)
        self.assertEqual(result.final_text, "delegation complete")
        self.assertEqual(result.tool_runs[0].tool_name, "delegate_task")
        self.assertIn("child_session_id=", result.tool_runs[0].content)
        self.assertIn("agent_id=worker_agent", result.tool_runs[0].content)
        runs = self.sessions.list_subagent_runs(result.session_id)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].agent_id, "worker_agent")
        self.assertTrue(runs[0].success)
        self.assertIn("Inspect README.md", runs[0].task)

    def test_delegate_task_failure_propagates_and_repeated_failures_stop_turn(self) -> None:
        model = _FakeModelClient([
            _tool_call_response(
                [
                    ToolCall(
                        id="call-1",
                        name="delegate_task",
                        arguments={"task": "Do risky work", "success_criteria": "Finish it."},
                    )
                ]
            ),
            ModelResponse(assistant_text=None, raw_response=None, error_type="request_error"),
            _tool_call_response(
                [
                    ToolCall(
                        id="call-2",
                        name="delegate_task",
                        arguments={"task": "Retry risky work", "success_criteria": "Finish it."},
                    )
                ]
            ),
            ModelResponse(assistant_text=None, raw_response=None, error_type="request_error"),
        ])
        loop = self._build_delegate_loop(model)

        result = loop.run_turn(user_input="delegate twice")

        self.assertFalse(result.success)
        self.assertEqual(result.stop_reason, "repeated_tool_failure")
        self.assertEqual(len(result.tool_runs), 2)
        self.assertEqual(result.tool_runs[0].tool_name, "delegate_task")
        self.assertEqual(result.tool_runs[1].tool_name, "delegate_task")
        self.assertIn("Subagent failed", result.tool_runs[0].error)
        self.assertFalse(result.tool_runs[0].success)
        self.assertFalse(result.tool_runs[1].success)

    def test_delegate_task_reuses_confirmation_handler_inside_child_loop(self) -> None:
        model = _FakeModelClient([
            _tool_call_response(
                [
                    ToolCall(
                        id="call-1",
                        name="delegate_task",
                        arguments={
                            "task": "Update src/module.py",
                            "success_criteria": "Replace old with new and confirm the edit.",
                            "relevant_paths": ["src/module.py"],
                        },
                    )
                ]
            ),
            _tool_call_response([ToolCall(id="child-1", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
            _text_response("child updated file"),
            _text_response("parent done"),
        ])
        confirmations: list[str] = []

        def _confirm(tool_call, context):
            confirmations.append(f"{context.agent_id}:{tool_call.name}")
            return True

        loop = self._build_delegate_loop(model, confirmation_handler=_confirm)

        result = loop.run_turn(user_input="use a worker to update the module")

        self.assertTrue(result.success)
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")
        self.assertIn("worker_agent:replace_in_file", confirmations)

    def test_follow_up_turn_reuses_delegate_summary_as_evidence(self) -> None:
        model = _FakeModelClient([
            _tool_call_response(
                [
                    ToolCall(
                        id="call-1",
                        name="delegate_task",
                        arguments={"task": "Inspect README.md", "success_criteria": "Summarize it."},
                    )
                ]
            ),
            _text_response("README summary"),
            _text_response("first turn complete"),
            _text_response("follow up answer"),
        ])
        loop = self._build_delegate_loop(model)

        first_result = loop.run_turn(user_input="inspect readme via worker")
        second_result = loop.run_turn(user_input="continue from the delegated work", session_id=first_result.session_id)

        self.assertTrue(first_result.success)
        self.assertTrue(second_result.success)
        self.assertTrue(any("Recent successful tool evidence:" in (message["content"] or "") for message in model.calls[3]["messages"]))
        self.assertTrue(any("child_session_id=" in (message["content"] or "") for message in model.calls[3]["messages"]))

    def _build_loop(self, model, confirmation_handler=None) -> AgentLoop:
        return AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=self.registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            confirmation_handler=confirmation_handler,
        )

    def _build_delegate_loop(self, model, confirmation_handler=None) -> AgentLoop:
        runner = SubagentRunner(
            model_client=model,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            tool_timeout=15.0,
            context_token_budget=6000,
            summary_trigger_tokens=3000,
            confirmation_handler=confirmation_handler,
        )
        return AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=build_root_registry(subagent_runner=runner),
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            confirmation_handler=confirmation_handler,
        )



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

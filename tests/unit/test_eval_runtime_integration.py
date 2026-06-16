from __future__ import annotations

import contextlib
import json
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.agent.definition import SINGLE_MAIN_AGENT
from agent_app.orchestrator.loop import AgentLoop
from agent_app.runtime.shell_runtime import RuntimeExecutionResult
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.tools.code_search import CodeSearchTool
from agent_app.tools.file_read import FileReadTool
from agent_app.tools.file_write import FileWriteTool
from agent_app.tools.registry import ToolRegistry
from agent_app.tools.replace_in_file import ReplaceInFileTool
from agent_app.tools.shell import ShellTool
from agent_app.tools.todo import TodoReadTool, TodoWriteTool
from agent_app.types import ModelResponse, TaskBudget, ToolCall


class _FakeModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    def generate(self, *, system_prompt, messages, tools):
        return self._responses.pop(0)


class _FakeShellRuntime:
    def __init__(self, results: list[RuntimeExecutionResult]) -> None:
        self.results = list(results)
        self.commands: list[str] = []

    def run(self, command: str, *, workspace_root: Path, timeout: float) -> RuntimeExecutionResult:
        self.commands.append(command)
        return self.results.pop(0)


class EvalRuntimeIntegrationTests(unittest.TestCase):
    def test_agent_loop_accepts_eval_budget_for_new_task(self) -> None:
        root = _make_workspace("eval_budget")
        db_path = root / ".agent_app" / "agent.db"
        try:
            initialize_database(db_path)
            sessions = SessionService(db_path)
            loop = AgentLoop(
                agent=SINGLE_MAIN_AGENT,
                model_client=_FakeModelClient([_text_response("done")]),
                tool_registry=ToolRegistry([]),
                session_service=sessions,
                workspace_root=root,
            )

            result = loop.run_turn(
                user_input="finish",
                budget=TaskBudget(max_model_calls=3, max_tool_calls=2, max_tokens=100),
            )

            task = sessions.get_task(result.task_id)
            self.assertEqual(task.budget.max_model_calls, 3)
            self.assertEqual(task.budget.max_tool_calls, 2)
            self.assertTrue(result.success)
        finally:
            _cleanup_explicit(
                files=[db_path],
                dirs=[db_path.parent, root],
            )

    def test_verification_failure_can_drive_followup_repair(self) -> None:
        root = _make_workspace("repair_loop")
        src_dir = root / "src"
        src_dir.mkdir()
        module = src_dir / "module.py"
        module.write_text("print('old')\n", encoding="utf-8")
        db_path = root / ".agent_app" / "agent.db"
        try:
            initialize_database(db_path)
            sessions = SessionService(db_path)
            shell_runtime = _FakeShellRuntime([
                RuntimeExecutionResult(
                    success=False,
                    stdout="",
                    stderr="AssertionError: expected new",
                    combined_output="AssertionError: expected new",
                    exit_code=1,
                    error_type="nonzero_exit",
                ),
                RuntimeExecutionResult(
                    success=True,
                    stdout="OK",
                    stderr="",
                    combined_output="OK",
                    exit_code=0,
                    error_type=None,
                ),
            ])
            model = _FakeModelClient([
                _tool_call_response([
                    ToolCall(
                        id="write-bad",
                        name="file_write",
                        arguments={"path": "src/module.py", "content": "print('bad')\n"},
                    )
                ]),
                _tool_call_response([
                    ToolCall(
                        id="verify-1",
                        name="shell",
                        arguments={"command": "python -m unittest discover -s tests -v"},
                    )
                ]),
                _tool_call_response([
                    ToolCall(
                        id="repair-1",
                        name="replace_in_file",
                        arguments={"path": "src/module.py", "old_text": "bad", "new_text": "new"},
                    )
                ]),
                _tool_call_response([
                    ToolCall(
                        id="verify-2",
                        name="shell",
                        arguments={"command": "python -m unittest discover -s tests -v"},
                    )
                ]),
                _text_response("fixed and verified"),
            ])
            loop = AgentLoop(
                agent=SINGLE_MAIN_AGENT,
                model_client=model,
                tool_registry=_build_registry(shell_runtime=shell_runtime),
                session_service=sessions,
                workspace_root=root,
                confirmation_handler=lambda tool_call, context: True,
            )

            result = loop.run_turn(user_input="fix and verify")

            self.assertTrue(result.success)
            self.assertEqual(module.read_text(encoding="utf-8"), "print('new')\n")
            self.assertEqual(
                [tool.tool_name for tool in result.tool_runs],
                ["file_write", "shell", "replace_in_file", "shell"],
            )
            self.assertEqual(len(shell_runtime.commands), 2)
            self.assertEqual(result.final_text, "fixed and verified")
            task = sessions.get_task(result.task_id)
            self.assertEqual(task.budget.used_repair_attempts, 1)
            repair_traces = [
                trace for trace in sessions.list_task_traces(result.task_id)
                if trace.trace_type == "repair"
            ]
            self.assertEqual(len(repair_traces), 1)
            self.assertTrue(repair_traces[0].payload["allowed"])
        finally:
            _cleanup_explicit(
                files=[module, db_path],
                dirs=[src_dir, db_path.parent, root],
            )

    def test_repair_attempt_budget_stops_after_failed_verification(self) -> None:
        root = _make_workspace("repair_limit")
        src_dir = root / "src"
        src_dir.mkdir()
        module = src_dir / "module.py"
        module.write_text("print('old')\n", encoding="utf-8")
        db_path = root / ".agent_app" / "agent.db"
        try:
            initialize_database(db_path)
            sessions = SessionService(db_path)
            shell_runtime = _FakeShellRuntime([
                RuntimeExecutionResult(
                    success=False,
                    stdout="",
                    stderr="AssertionError: expected new",
                    combined_output="AssertionError: expected new",
                    exit_code=1,
                    error_type="nonzero_exit",
                ),
            ])
            model = _FakeModelClient([
                _tool_call_response([
                    ToolCall(
                        id="write-bad",
                        name="file_write",
                        arguments={"path": "src/module.py", "content": "print('bad')\n"},
                    )
                ]),
                _tool_call_response([
                    ToolCall(
                        id="verify-1",
                        name="shell",
                        arguments={"command": "python -m unittest discover -s tests -v"},
                    )
                ]),
                _text_response("should not be used"),
            ])
            loop = AgentLoop(
                agent=SINGLE_MAIN_AGENT,
                model_client=model,
                tool_registry=_build_registry(shell_runtime=shell_runtime),
                session_service=sessions,
                workspace_root=root,
                confirmation_handler=lambda tool_call, context: True,
            )

            result = loop.run_turn(
                user_input="fix and verify",
                budget=TaskBudget(max_repair_attempts=0, max_model_calls=5, max_tool_calls=5),
            )

            self.assertFalse(result.success)
            self.assertEqual(result.stop_reason, "repair_attempt_budget_exceeded")
            self.assertEqual(module.read_text(encoding="utf-8"), "print('bad')\n")
            repair_traces = [
                trace for trace in sessions.list_task_traces(result.task_id)
                if trace.trace_type == "repair"
            ]
            self.assertEqual(len(repair_traces), 1)
            self.assertFalse(repair_traces[0].payload["allowed"])
            self.assertIn("AssertionError", repair_traces[0].payload["output_preview"])
        finally:
            _cleanup_explicit(
                files=[module, db_path],
                dirs=[src_dir, db_path.parent, root],
            )


def _build_registry(*, shell_runtime: _FakeShellRuntime) -> ToolRegistry:
    return ToolRegistry([
        FileReadTool(),
        CodeSearchTool(),
        TodoReadTool(),
        TodoWriteTool(),
        ReplaceInFileTool(),
        FileWriteTool(),
        ShellTool(runtime=shell_runtime),
    ])


def _text_response(text: str) -> ModelResponse:
    return ModelResponse(
        assistant_text=text,
        tool_calls=[],
        finish_reason="stop",
        raw_response={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]},
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
    )


def _make_workspace(prefix: str) -> Path:
    root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"{prefix}_{uuid4().hex}"
    root.mkdir(parents=True)
    return root


def _cleanup_explicit(*, files: list[Path], dirs: list[Path]) -> None:
    for file_path in files:
        file_path.unlink(missing_ok=True)
    for dir_path in dirs:
        with contextlib.suppress(FileNotFoundError, OSError):
            dir_path.rmdir()


if __name__ == "__main__":
    unittest.main()

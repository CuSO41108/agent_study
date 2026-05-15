from __future__ import annotations

import json
import shutil
import unittest
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from agent_app.agent.definition import SINGLE_MAIN_AGENT
from agent_app.orchestrator.loop import AgentLoop
from agent_app.orchestrator.subagent_runner import SubagentRunner
from agent_app.runtime.shell_runtime import RuntimeExecutionResult
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.tools.code_search import CodeSearchTool
from agent_app.tools.file_read import FileReadTool
from agent_app.tools.file_write import FileWriteTool
from agent_app.tools.registry import ToolRegistry, build_root_registry, build_worker_registry
from agent_app.tools.replace_in_file import ReplaceInFileTool
from agent_app.tools.shell import ShellTool
from agent_app.tools.todo import TodoReadTool, TodoWriteTool
from agent_app.types import ModelResponse, ToolCall


@dataclass(frozen=True, slots=True)
class RegressionScenario:
    name: str
    user_inputs: tuple[str, ...]
    expected_stop_reasons: tuple[str, ...]
    expected_tool_sequence: tuple[str, ...]


class _ScriptedModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    def generate(self, *, system_prompt, messages, tools):
        return self._responses.pop(0)


class _FakeShellRuntime:
    def __init__(self, results: list[RuntimeExecutionResult]) -> None:
        self._results = list(results)

    def run(self, command: str, *, workspace_root: Path, timeout: float) -> RuntimeExecutionResult:
        return self._results.pop(0)


class RegressionHarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"regression_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        self.db_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_precise_file_replace_scenario(self) -> None:
        scenario = RegressionScenario(
            name="precise file replace",
            user_inputs=("update src/module.py",),
            expected_stop_reasons=("final_response",),
            expected_tool_sequence=("replace_in_file",),
        )
        target = self.workspace_root / "src"
        target.mkdir()
        (target / "module.py").write_text("print('old')\n", encoding="utf-8")
        registry = build_registry(shell_runtime=_FakeShellRuntime([]))
        model = _ScriptedModelClient([
            _tool_call_response([ToolCall(id="call-1", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
            _text_response("done"),
        ])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            confirmation_handler=lambda tool_call, context: True,
        )

        result = loop.run_turn(user_input=scenario.user_inputs[0], session_id=None)

        self.assertEqual(result.stop_reason, scenario.expected_stop_reasons[0])
        self.assertEqual(tuple(tool.tool_name for tool in result.tool_runs), scenario.expected_tool_sequence)
        self.assertEqual((target / "module.py").read_text(encoding="utf-8"), "print('new')\n")
        turn_trace = self.sessions.list_turn_traces(result.session_id)[0]
        self.assertTrue(turn_trace.success)

    def test_write_then_verification_failure_preserves_written_content(self) -> None:
        scenario = RegressionScenario(
            name="write then verification failure",
            user_inputs=("update and verify",),
            expected_stop_reasons=("final_response",),
            expected_tool_sequence=("file_write", "shell"),
        )
        src_dir = self.workspace_root / "src"
        src_dir.mkdir()
        (src_dir / "module.py").write_text("print('old')\n", encoding="utf-8")
        runtime = _FakeShellRuntime(
            [
                RuntimeExecutionResult(
                    success=False,
                    stdout="",
                    stderr="verification failed",
                    combined_output="verification failed",
                    exit_code=3,
                    error_type="nonzero_exit",
                )
            ]
        )
        registry = build_registry(shell_runtime=runtime)
        model = _ScriptedModelClient([
            _tool_call_response([ToolCall(id="call-1", name="file_write", arguments={"path": "src/module.py", "content": "print('new')\n"})]),
            _tool_call_response([ToolCall(id="call-2", name="shell", arguments={"command": "git status --short"})]),
            _text_response("verification failed"),
        ])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            confirmation_handler=lambda tool_call, context: True,
        )

        result = loop.run_turn(user_input=scenario.user_inputs[0], session_id=None)

        self.assertEqual(result.stop_reason, scenario.expected_stop_reasons[0])
        self.assertEqual(tuple(tool.tool_name for tool in result.tool_runs), scenario.expected_tool_sequence)
        self.assertEqual((src_dir / "module.py").read_text(encoding="utf-8"), "print('new')\n")
        turn_trace = self.sessions.list_turn_traces(result.session_id)[0]
        tool_traces = self.sessions.list_tool_call_traces(turn_trace.id)
        self.assertEqual(tool_traces[1].tool_name, "shell")
        self.assertFalse(tool_traces[1].success)

    def test_cross_turn_memory_reuse_marks_todo_and_evidence(self) -> None:
        scenario = RegressionScenario(
            name="cross turn memory reuse",
            user_inputs=("plan work", "continue task"),
            expected_stop_reasons=("final_response", "final_response"),
            expected_tool_sequence=("todo_write",),
        )
        registry = build_registry(shell_runtime=_FakeShellRuntime([]))
        model = _ScriptedModelClient([
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
            _text_response("continue with the existing plan"),
        ])
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=model,
            tool_registry=registry,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
        )

        first_result = loop.run_turn(user_input=scenario.user_inputs[0], session_id=None)
        second_result = loop.run_turn(user_input=scenario.user_inputs[1], session_id=first_result.session_id)

        self.assertEqual(first_result.stop_reason, scenario.expected_stop_reasons[0])
        self.assertEqual(second_result.stop_reason, scenario.expected_stop_reasons[1])
        self.assertEqual(tuple(tool.tool_name for tool in first_result.tool_runs), scenario.expected_tool_sequence)

        turn_traces = self.sessions.list_turn_traces(first_result.session_id)
        self.assertEqual(len(turn_traces), 2)
        self.assertFalse(turn_traces[0].used_todo)
        self.assertTrue(turn_traces[1].used_todo)

    def test_subagent_research_edit_verify_scenario(self) -> None:
        scenario = RegressionScenario(
            name="subagent research edit verify",
            user_inputs=("delegate the module update",),
            expected_stop_reasons=("final_response",),
            expected_tool_sequence=("delegate_task",),
        )
        src_dir = self.workspace_root / "src"
        src_dir.mkdir()
        (src_dir / "module.py").write_text("print('old')\n", encoding="utf-8")
        runtime = _FakeShellRuntime(
            [
                RuntimeExecutionResult(
                    success=True,
                    stdout="verification ok",
                    stderr="",
                    combined_output="verification ok",
                    exit_code=0,
                    error_type=None,
                )
            ]
        )
        model = _ScriptedModelClient([
            _tool_call_response(
                [
                    ToolCall(
                        id="call-1",
                        name="delegate_task",
                        arguments={
                            "task": "Update src/module.py after checking the existing old-text occurrence in src/module.py.",
                            "success_criteria": "Find the target, replace old with new, run a verification command, and summarize the result.",
                            "relevant_paths": ["src/module.py"],
                        },
                    )
                ]
            ),
            _tool_call_response([ToolCall(id="child-1", name="code_search", arguments={"pattern": "print\\('old'\\)", "path": "."})]),
            _tool_call_response([ToolCall(id="child-2", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
            _tool_call_response([ToolCall(id="child-3", name="shell", arguments={"command": "git status --short"})]),
            _text_response("Worker updated src/module.py and verified the change."),
            _text_response("parent summary"),
        ])
        loop = build_delegating_loop(
            model=model,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            shell_runtime=runtime,
            confirmation_handler=lambda tool_call, context: True,
        )

        result = loop.run_turn(user_input=scenario.user_inputs[0], session_id=None)

        self.assertEqual(result.stop_reason, scenario.expected_stop_reasons[0])
        self.assertEqual(tuple(tool.tool_name for tool in result.tool_runs), scenario.expected_tool_sequence)
        self.assertEqual((src_dir / "module.py").read_text(encoding="utf-8"), "print('new')\n")
        subagent_runs = self.sessions.list_subagent_runs(result.session_id)
        self.assertEqual(len(subagent_runs), 1)
        child_turns = self.sessions.list_turn_traces(subagent_runs[0].child_session_id)
        self.assertEqual(len(child_turns), 1)
        child_tool_traces = self.sessions.list_tool_call_traces(child_turns[0].id)
        self.assertEqual([trace.tool_name for trace in child_tool_traces], ["code_search", "replace_in_file", "shell"])

    def test_subagent_failure_can_still_end_with_parent_text(self) -> None:
        model = _ScriptedModelClient([
            _tool_call_response(
                [
                    ToolCall(
                        id="call-1",
                        name="delegate_task",
                        arguments={
                            "task": "Try risky work",
                            "success_criteria": "Complete the delegated task.",
                        },
                    )
                ]
            ),
            ModelResponse(assistant_text=None, raw_response=None, error_type="request_error"),
            _text_response("worker failed cleanly"),
        ])
        loop = build_delegating_loop(
            model=model,
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            shell_runtime=_FakeShellRuntime([]),
        )

        result = loop.run_turn(user_input="delegate risky work", session_id=None)

        self.assertTrue(result.success)
        self.assertEqual(result.stop_reason, "final_response")
        self.assertEqual(result.final_text, "worker failed cleanly")
        self.assertEqual(len(result.tool_runs), 1)
        self.assertEqual(result.tool_runs[0].tool_name, "delegate_task")
        self.assertFalse(result.tool_runs[0].success)
        self.assertIn("Subagent failed", result.tool_runs[0].error)


def build_registry(*, shell_runtime: _FakeShellRuntime) -> ToolRegistry:
    return ToolRegistry([
        FileReadTool(),
        CodeSearchTool(),
        TodoReadTool(),
        TodoWriteTool(),
        ReplaceInFileTool(),
        FileWriteTool(),
        ShellTool(runtime=shell_runtime),
    ])


def build_delegating_loop(
    *,
    model: _ScriptedModelClient,
    session_service: SessionService,
    workspace_root: Path,
    shell_runtime: _FakeShellRuntime,
    confirmation_handler=None,
) -> AgentLoop:
    worker_registry = build_worker_registry(shell_runtime=shell_runtime)
    runner = SubagentRunner(
        model_client=model,
        session_service=session_service,
        workspace_root=workspace_root,
        tool_timeout=15.0,
        context_token_budget=6000,
        summary_trigger_tokens=3000,
        confirmation_handler=confirmation_handler,
        worker_registry=worker_registry,
    )
    return AgentLoop(
        agent=SINGLE_MAIN_AGENT,
        model_client=model,
        tool_registry=build_root_registry(subagent_runner=runner, shell_runtime=shell_runtime),
        session_service=session_service,
        workspace_root=workspace_root,
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

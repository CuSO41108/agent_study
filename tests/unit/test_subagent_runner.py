from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.orchestrator.subagent_runner import DelegatedTaskRequest, SubagentRunner, normalize_relevant_paths
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.tools.base import ToolExecutionContext
from agent_app.tools.delegate_task import DelegateTaskTool
from agent_app.types import TurnResult


class _FakeModelClient:
    def generate(self, *, system_prompt, messages, tools):
        raise AssertionError("Model should not be called in this test.")


class _RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, DelegatedTaskRequest, ToolExecutionContext]] = []

    def run(self, *, tool_call_id: str, request: DelegatedTaskRequest, context: ToolExecutionContext):
        self.calls.append((tool_call_id, request, context))
        from agent_app.types import ToolResult

        return ToolResult(tool_call_id=tool_call_id, tool_name="delegate_task", success=True, content="ok", error=None)


class _FakeChildLoop:
    def __init__(self, *, result: TurnResult, recorder: list[dict]) -> None:
        self._result = result
        self._recorder = recorder

    def run_turn(self, *, user_input: str, session_id: str | None = None) -> TurnResult:
        self._recorder.append({"user_input": user_input, "session_id": session_id})
        return TurnResult(
            session_id=session_id or self._result.session_id,
            final_text=self._result.final_text,
            stop_reason=self._result.stop_reason,
            tool_runs=self._result.tool_runs,
            success=self._result.success,
        )


class SubagentRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"subagent_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        self.db_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.parent_session_id = self.sessions.create_session("parent-session")
        self.context = ToolExecutionContext(
            workspace_root=self.workspace_root,
            session_id=self.parent_session_id,
            session_service=self.sessions,
            agent_id="single_main_agent",
            delegation_depth=0,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_delegate_task_tool_validates_arguments_before_running(self) -> None:
        runner = _RecordingRunner()
        tool = DelegateTaskTool(runner=runner)  # type: ignore[arg-type]

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"task": "Inspect README.md"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Invalid arguments: success_criteria is required.")
        self.assertEqual(runner.calls, [])

    def test_normalize_relevant_paths_trims_and_caps_to_five(self) -> None:
        paths = normalize_relevant_paths([" README.md ", "src", "", "tests", "docs", "extra", "ignored"])

        self.assertEqual(paths, ("README.md", "src", "tests", "docs", "extra"))

    def test_subagent_runner_rejects_depth_limit(self) -> None:
        runner = SubagentRunner(
            model_client=_FakeModelClient(),
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            tool_timeout=1.0,
            context_token_budget=6000,
            summary_trigger_tokens=3000,
        )
        child_context = ToolExecutionContext(
            workspace_root=self.workspace_root,
            session_id=self.parent_session_id,
            session_service=self.sessions,
            agent_id="worker_agent",
            delegation_depth=1,
        )

        result = runner.run(
            tool_call_id="call-1",
            request=DelegatedTaskRequest(task="Nested work", success_criteria="Finish it."),
            context=child_context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Delegation depth limit reached (1).")

    def test_subagent_runner_rejects_more_than_two_subagents_per_turn(self) -> None:
        runner = SubagentRunner(
            model_client=_FakeModelClient(),
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            tool_timeout=1.0,
            context_token_budget=6000,
            summary_trigger_tokens=3000,
        )
        self.context.turn_state["subagent_calls"] = 2

        result = runner.run(
            tool_call_id="call-1",
            request=DelegatedTaskRequest(task="Third task", success_criteria="Finish it."),
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Subagent limit reached for this turn (2).")

    def test_subagent_runner_records_child_session_and_summary(self) -> None:
        recorded_calls: list[dict] = []

        def _loop_factory(**kwargs):
            self.assertEqual(kwargs["delegation_depth"], 1)
            result = TurnResult(
                session_id="child-session",
                final_text="worker summary",
                stop_reason="final_response",
                tool_runs=[],
                success=True,
            )
            return _FakeChildLoop(result=result, recorder=recorded_calls)

        runner = SubagentRunner(
            model_client=_FakeModelClient(),
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            tool_timeout=1.0,
            context_token_budget=6000,
            summary_trigger_tokens=3000,
            loop_factory=_loop_factory,
        )

        result = runner.run(
            tool_call_id="call-1",
            request=DelegatedTaskRequest(
                task="Inspect README.md",
                success_criteria="Summarize the file.",
                relevant_paths=("README.md",),
            ),
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertIn("child_session_id=", result.content)
        self.assertIn("agent_id=worker_agent", result.content)
        self.assertIn("tool_sequence=(none)", result.content)
        self.assertIn("Relevant targets:", recorded_calls[0]["user_input"])
        subagent_runs = self.sessions.list_subagent_runs(self.parent_session_id)
        self.assertEqual(len(subagent_runs), 1)
        self.assertEqual(recorded_calls[0]["session_id"], subagent_runs[0].child_session_id)
        self.assertEqual(subagent_runs[0].task, "Inspect README.md")
        self.assertTrue(subagent_runs[0].success)


if __name__ == "__main__":
    unittest.main()

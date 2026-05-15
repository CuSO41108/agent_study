from __future__ import annotations

import unittest

from agent_app.types import SubagentRun, ToolResult, TurnResult


class TypesTests(unittest.TestCase):
    def test_turn_result_contract_is_stable(self) -> None:
        tool_run = ToolResult(
            tool_call_id="call-1",
            tool_name="shell",
            success=False,
            content="",
            error="blocked",
        )

        result = TurnResult(
            session_id="session-1",
            final_text=None,
            stop_reason="not_implemented",
            tool_runs=[tool_run],
            success=False,
        )

        self.assertEqual(result.session_id, "session-1")
        self.assertIsNone(result.final_text)
        self.assertEqual(result.stop_reason, "not_implemented")
        self.assertEqual(result.tool_runs, [tool_run])
        self.assertFalse(result.success)

    def test_subagent_run_contract_is_stable(self) -> None:
        subagent_run = SubagentRun(
            parent_session_id="parent-session",
            parent_tool_call_id="call-1",
            child_session_id="child-session",
            agent_id="worker_agent",
            task="Inspect README.md",
            success=True,
            result_summary="summary",
            created_at="2026-04-18T00:00:00+00:00",
        )

        self.assertEqual(subagent_run.parent_session_id, "parent-session")
        self.assertEqual(subagent_run.parent_tool_call_id, "call-1")
        self.assertEqual(subagent_run.child_session_id, "child-session")
        self.assertEqual(subagent_run.agent_id, "worker_agent")
        self.assertEqual(subagent_run.task, "Inspect README.md")
        self.assertTrue(subagent_run.success)
        self.assertEqual(subagent_run.result_summary, "summary")
        self.assertEqual(subagent_run.created_at, "2026-04-18T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()

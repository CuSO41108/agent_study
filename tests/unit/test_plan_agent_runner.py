from __future__ import annotations

import unittest

from agent_app.plan import NodeExecutionResult, PlanAgentNodeRunner, PlanNodeContext, parse_plan_graph
from agent_app.plan.store import PlanRevision
from agent_app.types import PendingAction, ToolResult, TurnResult


def _context(
    *,
    kind: str = "inspect",
    objective: str = "Read the target file.",
    allowed_tools: list[str] | None = None,
    acceptance: list[str] | None = None,
) -> PlanNodeContext:
    tools = allowed_tools or (["replace_in_file"] if kind == "edit" else ["file_read"])
    graph = parse_plan_graph(
        {
            "id": "plan-1",
            "revision": 1,
            "goal": "Inspect one file.",
            "nodes": [
                {
                    "id": "inspect",
                    "kind": kind,
                    "objective": objective,
                    "depends_on": [],
                    "allowed_tools": tools,
                    "acceptance": acceptance or ["The relevant lines are identified."],
                    "status": "pending",
                }
            ],
        }
    )
    persisted = PlanRevision(
        id="revision-1",
        task_id="task-1",
        graph=graph,
        status="active",
        node_results={},
        replan_reason=None,
        version=1,
        created_at="now",
        updated_at="now",
    )
    return PlanNodeContext("task-1", "session-1", persisted, graph.nodes[0], {})


class _FakeLoop:
    def __init__(self, result: TurnResult) -> None:
        self.result = result
        self.kwargs: dict = {}

    def run_turn(self, **kwargs):
        self.kwargs = kwargs
        return self.result


class PlanAgentRunnerTests(unittest.TestCase):
    def test_runner_forwards_node_scope_and_keeps_task_open(self) -> None:
        loop = _FakeLoop(
            TurnResult(
                session_id="session-1",
                final_text="lines identified",
                stop_reason="final_response",
                tool_runs=[
                    ToolResult("call-1", "file_read", True, "12: relevant line")
                ],
                success=True,
                task_id="task-1",
                task_status="running",
            )
        )

        outcome = PlanAgentNodeRunner(loop)(_context())

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.output, "lines identified")
        self.assertEqual(loop.kwargs["allowed_tools"], ("file_read",))
        self.assertTrue(loop.kwargs["keep_task_open"])
        self.assertEqual(loop.kwargs["user_input"], "")
        self.assertIn("Read the target file", loop.kwargs["transient_context"])
        self.assertEqual(outcome.evidence_refs, ("tool_call:call-1",))
        self.assertEqual(
            outcome.metadata["acceptance_validation"]["status"],
            "passed",
        )

    def test_runner_rejects_success_without_required_tool_evidence(self) -> None:
        loop = _FakeLoop(
            TurnResult(
                session_id="session-1",
                final_text="I found it.",
                stop_reason="final_response",
                tool_runs=[],
                success=True,
                task_id="task-1",
                task_status="running",
            )
        )

        outcome = PlanAgentNodeRunner(loop)(_context())

        self.assertEqual(outcome.status, "failed")
        self.assertIn("acceptance_evidence_missing", outcome.error or "")
        self.assertEqual(outcome.metadata["failure_category"], "acceptance_not_met")

    def test_runner_rejects_code_search_with_no_matches(self) -> None:
        loop = _FakeLoop(
            TurnResult(
                session_id="session-1",
                final_text="Nothing else is needed.",
                stop_reason="final_response",
                tool_runs=[ToolResult("call-1", "code_search", True, "No matches found.")],
                success=True,
                task_id="task-1",
                task_status="running",
            )
        )

        outcome = PlanAgentNodeRunner(loop)(
            _context(allowed_tools=["code_search"])
        )

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(outcome.evidence_refs, ())

    def test_runner_requires_task_uuid_to_appear_in_tool_evidence(self) -> None:
        task_uuid = "2fd5f46a-7eff-48d1-8517-859cce712966"
        loop = _FakeLoop(
            TurnResult(
                session_id="session-1",
                final_text=f"Task {task_uuid} was located.",
                stop_reason="final_response",
                tool_runs=[
                    ToolResult("call-1", "file_read", True, "General task state documentation")
                ],
                success=True,
                task_id="task-1",
                task_status="running",
            )
        )

        outcome = PlanAgentNodeRunner(loop)(
            _context(objective=f"Locate Trace for Task {task_uuid}.")
        )

        self.assertEqual(outcome.status, "failed")
        self.assertEqual(
            outcome.metadata["acceptance_validation"]["missing_anchors"],
            [task_uuid],
        )

    def test_runner_accepts_task_uuid_backed_by_tool_evidence(self) -> None:
        task_uuid = "2fd5f46a-7eff-48d1-8517-859cce712966"
        loop = _FakeLoop(
            TurnResult(
                session_id="session-1",
                final_text="Trace located.",
                stop_reason="final_response",
                tool_runs=[
                    ToolResult("call-1", "file_read", True, f'{{"task_id":"{task_uuid}"}}')
                ],
                success=True,
                task_id="task-1",
                task_status="running",
            )
        )

        outcome = PlanAgentNodeRunner(loop)(
            _context(objective=f"Locate Trace for Task {task_uuid}.")
        )

        self.assertEqual(outcome.status, "completed")
        self.assertEqual(outcome.evidence_refs, ("tool_call:call-1",))

    def test_edit_node_requires_successful_write_evidence(self) -> None:
        loop = _FakeLoop(
            TurnResult(
                session_id="session-1",
                final_text="Edit complete.",
                stop_reason="final_response",
                tool_runs=[ToolResult("call-1", "file_read", True, "old content")],
                success=True,
                task_id="task-1",
                task_status="running",
            )
        )

        outcome = PlanAgentNodeRunner(loop)(
            _context(
                kind="edit",
                allowed_tools=["file_read", "replace_in_file"],
            )
        )

        self.assertEqual(outcome.status, "failed")
        self.assertIn("replace_in_file", outcome.error or "")

    def test_runner_turns_pending_user_action_into_waiting_approval(self) -> None:
        loop = _FakeLoop(
            TurnResult(
                session_id="session-1",
                final_text=None,
                stop_reason="waiting_user",
                tool_runs=[],
                success=False,
                task_id="task-1",
                task_status="waiting_user",
                pending_action=PendingAction(kind="tool_approval", prompt="Approve write"),
            )
        )

        outcome = PlanAgentNodeRunner(loop)(_context())

        self.assertEqual(outcome, NodeExecutionResult(
            status="waiting_approval",
            metadata={"pending_action_id": outcome.metadata["pending_action_id"], "kind": "tool_approval"},
        ))
        self.assertIsNotNone(outcome.metadata["pending_action_id"])


if __name__ == "__main__":
    unittest.main()

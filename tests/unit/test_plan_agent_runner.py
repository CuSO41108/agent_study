from __future__ import annotations

import unittest

from agent_app.plan import NodeExecutionResult, PlanAgentNodeRunner, PlanNodeContext, parse_plan_graph
from agent_app.plan.store import PlanRevision
from agent_app.types import PendingAction, TurnResult


def _context() -> PlanNodeContext:
    graph = parse_plan_graph(
        {
            "id": "plan-1",
            "revision": 1,
            "goal": "Inspect one file.",
            "nodes": [
                {
                    "id": "inspect",
                    "kind": "inspect",
                    "objective": "Read the target file.",
                    "depends_on": [],
                    "allowed_tools": ["file_read"],
                    "acceptance": ["The relevant lines are identified."],
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
                tool_runs=[],
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

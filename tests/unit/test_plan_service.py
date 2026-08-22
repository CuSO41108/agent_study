from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.plan import PlanPlanner, PlanStore, PlanTaskService, parse_plan_graph
from agent_app.runtime.task_runtime import TaskRuntime
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.types import AgentEvent, ModelResponse, PendingAction, TurnResult


class _PlannerModel:
    def generate(self, **_kwargs):
        return ModelResponse(
            assistant_text=(
                '{"id":"plan-service","revision":1,"nodes":['
                '{"id":"inspect","kind":"inspect","objective":"Read the source.",'
                '"depends_on":[],"allowed_tools":["file_read"],'
                '"acceptance":["Source is understood"]},'
                '{"id":"verify","kind":"verify","objective":"Verify the result.",'
                '"depends_on":["inspect"],"allowed_tools":["shell"],'
                '"acceptance":["Verification passes"]}]}'
            )
        )


class _AgentLoop:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        task_id = kwargs["_task_id"]
        return TurnResult(
            session_id=kwargs["session_id"],
            final_text=f"done:{task_id}",
            stop_reason="final_response",
            tool_runs=[],
            success=True,
            task_id=task_id,
            task_status="running",
        )


class _ApprovalLoop:
    def __init__(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime
        self.calls: list[dict] = []

    def handle_event(self, event: AgentEvent, **kwargs):
        self.calls.append(kwargs)
        task = self.runtime.approve(event.task_id, event=event)
        return TurnResult(
            session_id=task.session_id,
            final_text="approved node completed",
            stop_reason="final_response",
            tool_runs=[],
            success=True,
            task_id=task.id,
            task_status=task.status,
        )


class _FailThenSuccessLoop(_AgentLoop):
    def __init__(self) -> None:
        super().__init__()
        self.attempt = 0

    def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        self.attempt += 1
        task_id = kwargs["_task_id"]
        return TurnResult(
            session_id=kwargs["session_id"],
            final_text=None if self.attempt == 1 else "replanned node done",
            stop_reason="node_failed" if self.attempt == 1 else "final_response",
            tool_runs=[],
            success=self.attempt > 1,
            task_id=task_id,
            task_status="running",
        )


class _AlwaysFailLoop(_AgentLoop):
    def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        task_id = kwargs["_task_id"]
        return TurnResult(
            session_id=kwargs["session_id"],
            final_text=None,
            stop_reason="node_failed",
            tool_runs=[],
            success=False,
            task_id=task_id,
            task_status="running",
        )


class _ReplanPlannerModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            content = (
                '{"id":"initial","revision":1,"nodes":['
                '{"id":"inspect","kind":"inspect","objective":"Inspect source.",'
                '"depends_on":[],"allowed_tools":["file_read"],"acceptance":["Inspected"]}]}'
            )
        else:
            content = (
                '{"id":"ignored-by-replanner","revision":99,"nodes":['
                '{"id":"inspect","kind":"inspect","objective":"Retry inspection.",'
                '"depends_on":[],"allowed_tools":["file_read"],"acceptance":["Inspected again"]}]}'
            )
        return ModelResponse(assistant_text=content)


class PlanTaskServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"plan_service_{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.db_path = self.root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.session_id = self.sessions.create_session("service-session")
        self.loop = _AgentLoop()
        self.service = PlanTaskService(
            planner=PlanPlanner(_PlannerModel()),
            plan_store=PlanStore(self.db_path),
            session_service=self.sessions,
            agent_loop=self.loop,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_start_creates_task_persists_plan_and_reconciles_task_completion(self) -> None:
        result = self.service.start(session_id=self.session_id, goal="Inspect and verify the source")

        self.assertEqual(result.execution.status, "completed")
        self.assertEqual(result.task.status, "completed")
        self.assertEqual(result.revision.status, "completed")
        self.assertEqual([call["allowed_tools"] for call in self.loop.calls], [("file_read",), ("shell",)])
        self.assertTrue(all(call["keep_task_open"] for call in self.loop.calls))
        self.assertEqual(self.sessions.list_messages(self.session_id)[0].content, "Inspect and verify the source")
        traces = self.sessions.list_task_traces(result.task.id)
        trace_types = [trace.trace_type for trace in traces]
        self.assertLess(trace_types.index("state_transition"), trace_types.index("plan_created"))
        self.assertIn("plan_node_transition", trace_types)
        self.assertIn("plan_execution", trace_types)
        transitions = [
            trace.payload
            for trace in traces
            if trace.trace_type == "plan_node_transition"
        ]
        self.assertEqual(
            [(item["node_id"], item["from_status"], item["to_status"]) for item in transitions],
            [("inspect", "pending", "completed"), ("verify", "pending", "completed")],
        )

    def test_approval_resumes_plan_node_with_its_scope_and_completes_plan(self) -> None:
        runtime = TaskRuntime(self.sessions)
        task = runtime.start_for_user_message(session_id=self.session_id, user_input="Approve the edit")
        pending = runtime.wait_for_user(
            task.id,
            PendingAction(
                kind="tool_approval",
                prompt="Approve file write",
                decision={"tool_name": "file_write", "tool_call_id": "call-1", "arguments": {}},
            ),
        )
        graph = parse_plan_graph(
            {
                "id": "approval-plan",
                "revision": 1,
                "goal": "Approve the edit",
                "nodes": [
                    {
                        "id": "edit",
                        "kind": "edit",
                        "objective": "Apply the approved edit.",
                        "depends_on": [],
                        "allowed_tools": ["file_write"],
                        "acceptance": ["The edit is applied."],
                        "status": "waiting_approval",
                    }
                ],
            }
        )
        store = PlanStore(self.db_path)
        revision = store.create_revision(task.id, graph)
        approval_loop = _ApprovalLoop(runtime)
        service = PlanTaskService(
            planner=PlanPlanner(_PlannerModel()),
            plan_store=store,
            session_service=self.sessions,
            agent_loop=approval_loop,
        )
        event = AgentEvent(
            id="approve-plan-node",
            task_id=task.id,
            session_id=self.session_id,
            type="user_approved",
            source="test",
            payload={"pending_action_id": pending.pending_action.id},
            expected_version=pending.version,
        )

        result = service.handle_approval(task_id=task.id, event=event, approved=True)

        self.assertIsNotNone(result)
        self.assertEqual(result.execution.status, "completed")
        self.assertEqual(result.revision.graph.node_map()["edit"].status, "completed")
        self.assertEqual(result.task.status, "completed")
        self.assertEqual(approval_loop.calls[0]["resume_allowed_tools"], ("file_write",))
        self.assertTrue(approval_loop.calls[0]["resume_keep_task_open"])
        self.assertIn("Apply the approved edit.", approval_loop.calls[0]["resume_transient_context"])
        approval_traces = [
            trace.payload
            for trace in self.sessions.list_task_traces(task.id)
            if trace.trace_type == "plan_node_approval"
        ]
        self.assertEqual(approval_traces[-1]["decision"], "approve")
        self.assertEqual(approval_traces[-1]["to_status"], "completed")

    def test_failed_node_enters_automatic_replan_and_resumes_with_new_revision(self) -> None:
        planner_model = _ReplanPlannerModel()
        loop = _FailThenSuccessLoop()
        service = PlanTaskService(
            planner=PlanPlanner(planner_model),
            plan_store=PlanStore(self.db_path),
            session_service=self.sessions,
            agent_loop=loop,
        )

        result = service.start(session_id=self.session_id, goal="Inspect with a fallback")

        self.assertEqual(result.execution.status, "completed")
        self.assertEqual(result.task.status, "completed")
        self.assertEqual(result.revision.graph.revision, 2)
        self.assertEqual(result.task.budget.used_replans, 1)
        self.assertEqual(planner_model.calls, 2)
        self.assertEqual(len(loop.calls), 2)
        traces = self.sessions.list_task_traces(result.task.id)
        failure = next(trace for trace in traces if trace.trace_type == "plan_failure")
        replan = next(trace for trace in traces if trace.trace_type == "plan_replan")
        self.assertEqual(failure.payload["failure_reason"], "node_failed")
        self.assertEqual(replan.payload["from_revision"], 1)
        self.assertEqual(replan.payload["to_revision"], 2)
        self.assertEqual(replan.payload["preserved_completed_nodes"], [])

    def test_replan_budget_exhaustion_returns_diagnosis_and_terminal_trace(self) -> None:
        planner_model = _ReplanPlannerModel()
        loop = _AlwaysFailLoop()
        service = PlanTaskService(
            planner=PlanPlanner(planner_model),
            plan_store=PlanStore(self.db_path),
            session_service=self.sessions,
            agent_loop=loop,
        )

        result = service.start(session_id=self.session_id, goal="Inspect with no successful fallback")

        self.assertEqual(result.execution.status, "failed")
        self.assertEqual(result.task.status, "failed")
        self.assertEqual(result.task.budget.used_replans, result.task.budget.max_replans)
        self.assertEqual(result.revision.status, "failed")
        self.assertEqual(len(loop.calls), 1 + result.task.budget.max_replans)
        traces = self.sessions.list_task_traces(result.task.id)
        self.assertEqual(
            len([trace for trace in traces if trace.trace_type == "plan_replan"]),
            result.task.budget.max_replans,
        )
        failures = [trace for trace in traces if trace.trace_type == "plan_failure"]
        self.assertEqual(len(failures), 1 + result.task.budget.max_replans)
        self.assertEqual(failures[-1].payload["failure_reason"], "node_failed")


if __name__ == "__main__":
    unittest.main()

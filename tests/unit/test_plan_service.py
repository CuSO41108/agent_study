from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.plan import PlanPlanner, PlanPlanningError, PlanStore, PlanTaskService, parse_plan_graph
from agent_app.runtime.task_runtime import TaskRuntime
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.types import AgentEvent, ModelResponse, PendingAction, ToolResult, TurnResult


def _successful_node_evidence(kwargs: dict, *, call_id: str = "node-evidence") -> list[ToolResult]:
    allowed_tools = tuple(kwargs.get("allowed_tools", ()))
    if "shell" in allowed_tools:
        tool_name = "shell"
    elif "replace_in_file" in allowed_tools:
        tool_name = "replace_in_file"
    elif "file_write" in allowed_tools:
        tool_name = "file_write"
    elif "code_search" in allowed_tools:
        tool_name = "code_search"
    else:
        tool_name = "file_read"
    return [
        ToolResult(
            tool_call_id=call_id,
            tool_name=tool_name,
            success=True,
            content="substantive node evidence",
        )
    ]


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


class _FailingPlannerModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        return ModelResponse(
            assistant_text=None,
            error_type="request_error",
            raw_response={"detail": "connection refused; api_key=do-not-leak"},
        )


class _RetryThenSuccessPlannerModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                assistant_text=None,
                error_type="request_error",
                raw_response={"detail": "temporary connection reset"},
            )
        return _PlannerModel().generate()


class _RepairInvalidShapePlannerModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                assistant_text=(
                    '{"id":"plan-service","revision":1,"nodes":['
                    '{"id":"inspect","kind":"inspect","objective":"Read the source.",'
                    '"depends_on":[],"allowed_tools":["file_read"],'
                    '"acceptance":"Source is understood"}]}'
                )
            )
        return _PlannerModel().generate()


class _FailThenRecoverPlannerModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        if self.calls <= 3:
            return ModelResponse(
                assistant_text=None,
                error_type="request_error",
                raw_response={"detail": "temporary planner outage"},
            )
        return _PlannerModel().generate()


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
            tool_runs=_successful_node_evidence(kwargs),
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
            tool_runs=[
                ToolResult("approved-write", "file_write", True, "write completed")
            ],
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
            tool_runs=(
                _successful_node_evidence(kwargs)
                if self.attempt > 1
                else []
            ),
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


class _EvidenceMissingLoop(_AgentLoop):
    def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        task_id = kwargs["_task_id"]
        return TurnResult(
            session_id=kwargs["session_id"],
            final_text="claimed completion without tool evidence",
            stop_reason="final_response",
            tool_runs=[],
            success=True,
            task_id=task_id,
            task_status="running",
        )


class _DeterministicBlockedLoop(_AgentLoop):
    def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        task_id = kwargs["_task_id"]
        return TurnResult(
            session_id=kwargs["session_id"],
            final_text="Search path not found: 'agent_app'.",
            stop_reason="repeated_deterministic_tool_failure",
            tool_runs=[],
            success=False,
            task_id=task_id,
            task_status="running",
        )


class _CompleteThenFailThenSuccessLoop(_AgentLoop):
    def __init__(self) -> None:
        super().__init__()
        self.verify_attempts = 0

    def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        task_id = kwargs["_task_id"]
        node_id = kwargs["plan_node_id"]
        if node_id == "verify":
            self.verify_attempts += 1
        success = node_id == "inspect" or self.verify_attempts > 1
        return TurnResult(
            session_id=kwargs["session_id"],
            final_text=f"done:{node_id}" if success else None,
            stop_reason="final_response" if success else "node_failed",
            tool_runs=_successful_node_evidence(kwargs) if success else [],
            success=success,
            task_id=task_id,
            task_status="running",
        )


class _PauseThenSuccessLoop(_AgentLoop):
    def __init__(self, runtime: TaskRuntime) -> None:
        super().__init__()
        self.runtime = runtime
        self.attempt = 0

    def run_turn(self, **kwargs):
        self.calls.append(kwargs)
        self.attempt += 1
        task_id = kwargs["_task_id"]
        if self.attempt == 1:
            self.runtime.pause_for_budget(task_id, reason="max_tool_rounds_exceeded")
            return TurnResult(
                session_id=kwargs["session_id"],
                final_text=None,
                stop_reason="max_tool_rounds_exceeded",
                tool_runs=[],
                success=False,
                task_id=task_id,
                task_status="paused",
            )
        return TurnResult(
            session_id=kwargs["session_id"],
            final_text="continued node done",
            stop_reason="final_response",
            tool_runs=_successful_node_evidence(kwargs),
            success=True,
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


class _RecoverableReplanModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            content = (
                '{"id":"recoverable-replan","revision":1,"nodes":['
                '{"id":"inspect","kind":"inspect","objective":"Inspect source.",'
                '"depends_on":[],"allowed_tools":["file_read"],"acceptance":["Inspected"]},'
                '{"id":"verify","kind":"verify","objective":"Verify source.",'
                '"depends_on":["inspect"],"allowed_tools":["file_read"],"acceptance":["Verified"]}]}'
            )
        elif self.calls in {2, 3}:
            content = "not a JSON plan"
        else:
            content = (
                '{"id":"ignored","revision":99,"nodes":['
                '{"id":"inspect","kind":"inspect","objective":"Inspect source.",'
                '"depends_on":[],"allowed_tools":["file_read"],"acceptance":["Inspected"]},'
                '{"id":"verify","kind":"verify","objective":"Retry verification.",'
                '"depends_on":["inspect"],"allowed_tools":["file_read"],"acceptance":["Verified"]}]}'
            )
        return ModelResponse(assistant_text=content)


class _AskUserAnswerLoop:
    def __init__(self, runtime: TaskRuntime) -> None:
        self.runtime = runtime
        self.calls: list[dict] = []

    def handle_event(self, event: AgentEvent, **kwargs):
        self.calls.append(kwargs)
        task = self.runtime.resume_with_user_message(
            event.task_id,
            str(event.payload["content"]),
            event=event,
        )
        return TurnResult(
            session_id=task.session_id,
            final_text="user answer applied",
            stop_reason="final_response",
            tool_runs=[
                ToolResult("answer-read", "file_read", True, "selected file evidence")
            ],
            success=True,
            task_id=task.id,
            task_status=task.status,
        )


class _AskAgainLoop(_AskUserAnswerLoop):
    def handle_event(self, event: AgentEvent, **kwargs):
        self.calls.append(kwargs)
        task = self.runtime.resume_with_user_message(
            event.task_id,
            str(event.payload["content"]),
            event=event,
        )
        task = self.runtime.wait_for_user(
            task.id,
            PendingAction(kind="ask_user", prompt="Please provide another detail."),
        )
        return TurnResult(
            session_id=task.session_id,
            final_text=None,
            stop_reason="waiting_user",
            tool_runs=[],
            success=False,
            task_id=task.id,
            task_status=task.status,
            pending_action=task.pending_action,
        )


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
        checkpoints = self.sessions.list_checkpoints(result.task.id)
        planning_checkpoints = [
            checkpoint for checkpoint in checkpoints if checkpoint.state.get("phase") == "planning"
        ]
        self.assertEqual(
            [checkpoint.state["request_status"] for checkpoint in planning_checkpoints],
            ["requesting", "completed"],
        )
        self.assertEqual(planning_checkpoints[0].cursor, "planning")
        self.assertEqual(planning_checkpoints[-1].cursor, "completed")
        planner_run = self.sessions.get_execution_run(planning_checkpoints[-1].run_id)
        self.assertEqual(planner_run.scope, "planner:initial_plan")
        self.assertEqual(planner_run.status, "completed")

    def test_planner_success_after_retry_persists_actual_attempt(self) -> None:
        planner_model = _RetryThenSuccessPlannerModel()
        service = PlanTaskService(
            planner=PlanPlanner(planner_model, sleep=lambda _delay: None),
            plan_store=PlanStore(self.db_path),
            session_service=self.sessions,
            agent_loop=self.loop,
        )

        result = service.start(
            session_id=self.session_id,
            goal="Inspect and verify after a transient provider failure",
        )

        planning_checkpoints = [
            checkpoint
            for checkpoint in self.sessions.list_checkpoints(result.task.id)
            if checkpoint.state.get("phase") == "planning"
        ]
        self.assertEqual(
            [checkpoint.state["request_status"] for checkpoint in planning_checkpoints],
            ["requesting", "retrying", "completed"],
        )
        self.assertEqual(planner_model.calls, 2)
        self.assertEqual(planning_checkpoints[-1].state["attempt"], 2)
        self.assertEqual(planning_checkpoints[-1].state["max_attempts"], 4)

    def test_planner_format_repair_is_checkpointed_and_completes(self) -> None:
        planner_model = _RepairInvalidShapePlannerModel()
        service = PlanTaskService(
            planner=PlanPlanner(planner_model, sleep=lambda _delay: None),
            plan_store=PlanStore(self.db_path),
            session_service=self.sessions,
            agent_loop=self.loop,
        )

        result = service.start(
            session_id=self.session_id,
            goal="Inspect after repairing an invalid PlanGraph shape",
        )

        self.assertEqual(result.task.status, "completed")
        self.assertEqual(planner_model.calls, 2)
        planning_checkpoints = [
            checkpoint
            for checkpoint in self.sessions.list_checkpoints(result.task.id)
            if checkpoint.state.get("phase") == "planning"
        ]
        self.assertEqual(
            [checkpoint.state["request_status"] for checkpoint in planning_checkpoints],
            ["requesting", "repairing", "completed"],
        )
        repair_checkpoint = planning_checkpoints[1]
        self.assertEqual(repair_checkpoint.state["attempt"], 1)
        self.assertEqual(repair_checkpoint.state["error_type"], "invalid_plan")
        self.assertTrue(repair_checkpoint.state["retryable"])
        self.assertEqual(planning_checkpoints[-1].state["attempt"], 2)
        self.assertEqual(planning_checkpoints[-1].state["max_attempts"], 4)

    def test_planner_failure_persists_attempts_and_safe_detail_in_checkpoint(self) -> None:
        planner_model = _FailingPlannerModel()
        service = PlanTaskService(
            planner=PlanPlanner(planner_model, sleep=lambda _delay: None),
            plan_store=PlanStore(self.db_path),
            session_service=self.sessions,
            agent_loop=self.loop,
        )

        with self.assertRaises(PlanPlanningError) as raised:
            service.start(session_id=self.session_id, goal="Inspect with a transient provider failure")

        task = self.sessions.get_latest_task(self.session_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.status, "paused")
        self.assertEqual(task.stop_reason, "planner_failed")
        error = raised.exception
        self.assertIn("connection refused", str(error))
        self.assertNotIn("do-not-leak", str(error))
        self.assertEqual(planner_model.calls, 3)
        self.assertEqual(error.recovery_task_id, task.id)
        self.assertTrue(service.has_recoverable_planner_failure(task_id=task.id))

        checkpoints = self.sessions.list_checkpoints(task.id)
        planning_checkpoints = [
            checkpoint for checkpoint in checkpoints if checkpoint.state.get("phase") == "planning"
        ]
        self.assertEqual(
            [checkpoint.state["request_status"] for checkpoint in planning_checkpoints],
            ["requesting", "retrying", "retrying", "failed"],
        )
        self.assertEqual(planning_checkpoints[-1].status, "failed")
        self.assertEqual(planning_checkpoints[-1].state["attempt"], 3)
        self.assertEqual(planning_checkpoints[-1].state["error_type"], "request_error")
        self.assertIn("connection refused", planning_checkpoints[-1].state["error_detail"])
        self.assertNotIn("do-not-leak", planning_checkpoints[-1].state["error_detail"])
        planner_run = self.sessions.get_execution_run(planning_checkpoints[-1].run_id)
        self.assertEqual(planner_run.status, "failed")

    def test_continue_retries_initial_planner_on_same_task_and_checkpoint_chain(self) -> None:
        planner_model = _FailThenRecoverPlannerModel()
        store = PlanStore(self.db_path)
        service = PlanTaskService(
            planner=PlanPlanner(planner_model, sleep=lambda _delay: None),
            plan_store=store,
            session_service=self.sessions,
            agent_loop=self.loop,
        )

        with self.assertRaises(PlanPlanningError):
            service.start(session_id=self.session_id, goal="Inspect after Planner recovery")

        paused = self.sessions.get_latest_task(self.session_id)
        self.assertIsNotNone(paused)
        assert paused is not None
        failed_checkpoint = self.sessions.get_latest_checkpoint(paused.id)
        self.assertIsNotNone(failed_checkpoint)
        assert failed_checkpoint is not None

        result = service.continue_task(task_id=paused.id)

        self.assertEqual(result.task.id, paused.id)
        self.assertEqual(result.task.status, "completed")
        self.assertEqual(result.task.budget.used_continuations, 1)
        self.assertEqual(planner_model.calls, 4)
        self.assertEqual(len(store.list_revisions(paused.id)), 1)
        self.assertEqual(len(self.sessions.list_messages(self.session_id)), 1)
        planner_runs = [
            run
            for run in self.sessions.list_execution_runs(paused.id)
            if run.scope == "planner:initial_plan"
        ]
        self.assertEqual(len(planner_runs), 2)
        self.assertEqual(planner_runs[1].parent_checkpoint_id, failed_checkpoint.id)
        planning_checkpoints = [
            checkpoint
            for checkpoint in self.sessions.list_checkpoints(paused.id)
            if checkpoint.state.get("phase") == "planning"
        ]
        self.assertEqual(
            [checkpoint.state["request_status"] for checkpoint in planning_checkpoints],
            ["requesting", "retrying", "retrying", "failed", "requesting", "completed"],
        )
        self.assertFalse(service.has_recoverable_planner_failure(task_id=paused.id))

    def test_repeated_initial_planner_continuations_cannot_bypass_budget(self) -> None:
        planner_model = _FailingPlannerModel()
        service = PlanTaskService(
            planner=PlanPlanner(planner_model, sleep=lambda _delay: None),
            plan_store=PlanStore(self.db_path),
            session_service=self.sessions,
            agent_loop=self.loop,
        )

        with self.assertRaises(PlanPlanningError):
            service.start(session_id=self.session_id, goal="Keep failing Planner requests bounded")
        task = self.sessions.get_latest_task(self.session_id)
        self.assertIsNotNone(task)
        assert task is not None
        for _ in range(task.budget.max_continuations):
            with self.assertRaises(PlanPlanningError):
                service.continue_task(task_id=task.id)

        with self.assertRaisesRegex(RuntimeError, "continuation attempts"):
            service.continue_task(task_id=task.id)

        terminal = self.sessions.get_task(task.id)
        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertEqual(terminal.status, "failed")
        self.assertEqual(terminal.stop_reason, "planner_continuation_budget_exceeded")
        self.assertEqual(terminal.budget.used_continuations, terminal.budget.max_continuations)
        self.assertEqual(planner_model.calls, 3 * (1 + terminal.budget.max_continuations))

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
        planner_checkpoints = [
            checkpoint
            for checkpoint in self.sessions.list_checkpoints(result.task.id)
            if checkpoint.state.get("phase") == "planning"
        ]
        self.assertEqual(
            [checkpoint.state["operation"] for checkpoint in planner_checkpoints],
            ["initial_plan", "initial_plan", "replan", "replan"],
        )
        self.assertEqual(
            [checkpoint.state["request_status"] for checkpoint in planner_checkpoints],
            ["requesting", "completed", "requesting", "completed"],
        )

    def test_missing_acceptance_evidence_pauses_without_automatic_replan(self) -> None:
        planner_model = _ReplanPlannerModel()
        loop = _EvidenceMissingLoop()
        service = PlanTaskService(
            planner=PlanPlanner(planner_model),
            plan_store=PlanStore(self.db_path),
            session_service=self.sessions,
            agent_loop=loop,
        )

        first = service.start(
            session_id=self.session_id,
            goal="Inspect only when evidence is available",
        )

        self.assertEqual(first.execution.status, "paused")
        self.assertEqual(first.task.status, "paused")
        self.assertEqual(first.task.stop_reason, "acceptance_evidence_missing")
        self.assertEqual(first.task.budget.used_replans, 0)
        self.assertEqual(first.revision.graph.node_map()["inspect"].status, "paused")
        self.assertEqual(planner_model.calls, 1)
        checkpoint = self.sessions.get_latest_checkpoint(first.task.id)
        self.assertIsNotNone(checkpoint)
        assert checkpoint is not None
        self.assertEqual(checkpoint.state["phase"], "plan_node_recovery")
        self.assertEqual(checkpoint.state["failure_category"], "acceptance_not_met")
        self.assertTrue(checkpoint.state["recoverable"])

        second = service.continue_task(task_id=first.task.id)

        self.assertEqual(second.execution.status, "paused")
        self.assertEqual(second.task.budget.used_continuations, 1)
        self.assertEqual(second.task.budget.used_replans, 0)
        self.assertEqual(planner_model.calls, 1)

    def test_deterministic_tool_block_pauses_without_automatic_replan(self) -> None:
        planner_model = _ReplanPlannerModel()
        service = PlanTaskService(
            planner=PlanPlanner(planner_model),
            plan_store=PlanStore(self.db_path),
            session_service=self.sessions,
            agent_loop=_DeterministicBlockedLoop(),
        )

        result = service.start(
            session_id=self.session_id,
            goal="Inspect a source path without repeating invalid targets",
        )

        self.assertEqual(result.execution.status, "paused")
        self.assertEqual(result.task.status, "paused")
        self.assertEqual(result.task.stop_reason, "repeated_deterministic_tool_failure")
        self.assertEqual(result.task.budget.used_replans, 0)
        self.assertEqual(result.revision.graph.node_map()["inspect"].status, "paused")
        self.assertEqual(planner_model.calls, 1)
        recovery_traces = [
            trace
            for trace in self.sessions.list_task_traces(result.task.id)
            if trace.trace_type == "plan_node_recovery_available"
        ]
        self.assertEqual(len(recovery_traces), 1)
        self.assertEqual(
            recovery_traces[0].payload["failure_category"],
            "deterministic_tool_failure",
        )

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

    def test_paused_plan_node_resumes_from_checkpoint_without_replan(self) -> None:
        loop = _PauseThenSuccessLoop(TaskRuntime(self.sessions))
        store = PlanStore(self.db_path)
        service = PlanTaskService(
            planner=PlanPlanner(_PlannerModel()),
            plan_store=store,
            session_service=self.sessions,
            agent_loop=loop,
        )

        first = service.start(session_id=self.session_id, goal="Inspect in bounded node windows")

        self.assertEqual(first.execution.status, "paused")
        self.assertEqual(first.task.status, "paused")
        self.assertEqual(first.revision.graph.node_map()["inspect"].status, "paused")
        self.assertEqual(first.task.budget.used_replans, 0)
        self.assertEqual(len(loop.calls), 1)

        second = service.resume(task_id=first.task.id)

        self.assertEqual(second.execution.status, "completed")
        self.assertEqual(second.task.status, "completed")
        self.assertEqual(second.revision.graph.node_map()["inspect"].status, "completed")
        self.assertEqual(second.task.budget.used_continuations, 1)
        self.assertEqual(len(loop.calls), 3)
        self.assertEqual(loop.calls[1]["_task_id"], first.task.id)
        self.assertEqual(loop.calls[2]["_task_id"], first.task.id)
        self.assertEqual(second.task.budget.used_replans, 0)

    def test_auto_replan_failure_pauses_and_continue_preserves_completed_nodes(self) -> None:
        planner_model = _RecoverableReplanModel()
        loop = _CompleteThenFailThenSuccessLoop()
        store = PlanStore(self.db_path)
        service = PlanTaskService(
            planner=PlanPlanner(planner_model, sleep=lambda _delay: None),
            plan_store=store,
            session_service=self.sessions,
            agent_loop=loop,
        )

        first = service.start(session_id=self.session_id, goal="Inspect and verify with recovery")

        self.assertEqual(first.task.status, "paused")
        self.assertEqual(first.task.stop_reason, "replanner_failed")
        self.assertEqual(first.execution.status, "paused")
        self.assertEqual(first.revision.status, "active")
        self.assertEqual(first.revision.graph.node_map()["inspect"].status, "completed")
        self.assertEqual(first.revision.graph.node_map()["verify"].status, "failed")
        self.assertEqual(first.task.budget.used_replans, 1)
        self.assertTrue(service.has_recoverable_planner_failure(task_id=first.task.id))
        failed_checkpoint = self.sessions.get_latest_checkpoint(first.task.id)
        self.assertIsNotNone(failed_checkpoint)
        assert failed_checkpoint is not None
        self.assertEqual(failed_checkpoint.state["source_revision_id"], first.revision.id)
        self.assertTrue(failed_checkpoint.state["automatic"])

        second = service.continue_task(task_id=first.task.id)

        self.assertEqual(second.task.id, first.task.id)
        self.assertEqual(second.task.status, "completed")
        self.assertEqual(second.revision.graph.revision, 2)
        self.assertEqual(second.revision.graph.node_map()["inspect"].status, "completed")
        self.assertEqual(second.revision.graph.node_map()["verify"].status, "completed")
        self.assertEqual(second.task.budget.used_replans, 1)
        self.assertEqual(second.task.budget.used_continuations, 1)
        self.assertEqual([call["plan_node_id"] for call in loop.calls], ["inspect", "verify", "verify"])
        trace = next(
            item
            for item in self.sessions.list_task_traces(first.task.id)
            if item.trace_type == "plan_replan_failed"
        )
        self.assertEqual(trace.payload["error_type"], "invalid_plan")
        recovered = next(
            item
            for item in self.sessions.list_task_traces(first.task.id)
            if item.trace_type == "planner_recovery_completed"
        )
        self.assertEqual(recovered.payload["checkpoint_id"], failed_checkpoint.id)

    def _create_waiting_ask_user_plan(self, *, loop):
        runtime = TaskRuntime(self.sessions)
        task = runtime.start_for_user_message(
            session_id=self.session_id,
            user_input="Need a plan answer",
        )
        pending = runtime.wait_for_user(
            task.id,
            PendingAction(kind="ask_user", prompt="Which file should I inspect?"),
        )
        graph = parse_plan_graph(
            {
                "id": "ask-user-plan",
                "revision": 1,
                "goal": "Need a plan answer",
                "nodes": [
                    {
                        "id": "inspect",
                        "kind": "inspect",
                        "objective": "Inspect the selected file.",
                        "depends_on": [],
                        "allowed_tools": ["file_read"],
                        "acceptance": ["The selected file is inspected."],
                        "status": "waiting_approval",
                    }
                ],
            }
        )
        store = PlanStore(self.db_path)
        store.create_revision(task.id, graph)
        service = PlanTaskService(
            planner=PlanPlanner(_PlannerModel()),
            plan_store=store,
            session_service=self.sessions,
            agent_loop=loop,
        )
        event = AgentEvent(
            id="answer-plan-node",
            task_id=task.id,
            session_id=self.session_id,
            type="user_message",
            source="test",
            payload={"content": "Use README.md"},
            correlation_id=task.id,
            expected_version=pending.version,
        )
        return service, task, event

    def test_ask_user_answer_resumes_original_plan_node_scope(self) -> None:
        runtime = TaskRuntime(self.sessions)
        loop = _AskUserAnswerLoop(runtime)
        service, task, event = self._create_waiting_ask_user_plan(loop=loop)

        result = service.handle_user_message(task_id=task.id, event=event)

        self.assertIsNotNone(result)
        self.assertEqual(result.execution.status, "completed")
        self.assertEqual(result.task.status, "completed")
        self.assertEqual(result.revision.graph.node_map()["inspect"].status, "completed")
        self.assertEqual(loop.calls[0]["resume_allowed_tools"], ("file_read",))
        self.assertTrue(loop.calls[0]["resume_keep_task_open"])
        self.assertIn("Inspect the selected file.", loop.calls[0]["resume_transient_context"])

    def test_ask_user_can_request_another_answer_without_leaving_plan(self) -> None:
        runtime = TaskRuntime(self.sessions)
        loop = _AskAgainLoop(runtime)
        service, task, event = self._create_waiting_ask_user_plan(loop=loop)

        result = service.handle_user_message(task_id=task.id, event=event)

        self.assertIsNotNone(result)
        self.assertEqual(result.execution.status, "waiting_approval")
        self.assertEqual(result.task.status, "waiting_user")
        self.assertEqual(result.revision.graph.node_map()["inspect"].status, "waiting_approval")
        self.assertEqual(result.task.pending_action.kind, "ask_user")
        self.assertIsNotNone(self.sessions.get_task(task.id))

    def test_ask_user_pending_action_rejects_approval_commands(self) -> None:
        runtime = TaskRuntime(self.sessions)
        loop = _AskUserAnswerLoop(runtime)
        service, task, event = self._create_waiting_ask_user_plan(loop=loop)
        approval_event = AgentEvent(
            id="wrong-approval-event",
            task_id=task.id,
            session_id=self.session_id,
            type="user_approved",
            source="test",
            payload={},
            expected_version=event.expected_version,
        )

        with self.assertRaisesRegex(ValueError, "natural-language response"):
            service.handle_approval(task_id=task.id, event=approval_event, approved=True)


if __name__ == "__main__":
    unittest.main()

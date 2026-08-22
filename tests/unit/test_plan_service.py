from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.plan import PlanPlanner, PlanStore, PlanTaskService
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.types import ModelResponse, TurnResult


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


if __name__ == "__main__":
    unittest.main()

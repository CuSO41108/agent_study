from __future__ import annotations

import shutil
import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.runtime.task_runtime import InvalidTaskTransition, TaskRuntime
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService, TaskVersionConflict
from agent_app.types import (
    AgentEvent,
    Observation,
    PendingAction,
    TaskBudget,
    TodoItem,
)


class TaskRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"task_runtime_{uuid4().hex}"
        self.root.mkdir(parents=True)
        self.db_path = self.root / ".agent_app" / "agent.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.session_id = self.sessions.create_session("session-1")
        self.runtime = TaskRuntime(self.sessions)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_task_lifecycle_and_event_sequence_survive_restart(self) -> None:
        task = self.runtime.start_for_user_message(
            session_id=self.session_id,
            user_input="inspect the repository",
        )

        reloaded = SessionService(self.db_path)
        persisted = reloaded.get_task(task.id)
        events = reloaded.list_task_events(task.id)

        self.assertEqual(persisted.status, "running")
        self.assertEqual(persisted.goal, "inspect the repository")
        self.assertEqual(persisted.version, 2)
        self.assertEqual([event.type for event in events], ["task_created", "user_message"])
        self.assertEqual([event.sequence for event in events], [1, 2])

    def test_terminal_task_is_immutable_and_next_goal_creates_new_task(self) -> None:
        first = self.runtime.start_for_user_message(session_id=self.session_id, user_input="first")
        completed = self.runtime.complete(first.id)

        with self.assertRaises(InvalidTaskTransition):
            self.runtime.record_observation(
                completed.id,
                Observation(
                    status="succeeded",
                    error_type=None,
                    message="late",
                    retryable=False,
                    side_effect=False,
                ),
            )

        second = self.runtime.start_for_user_message(session_id=self.session_id, user_input="second")
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(second.status, "running")

    def test_stale_version_rejects_transition_without_mutation(self) -> None:
        task = self.runtime.start_for_user_message(session_id=self.session_id, user_input="work")
        event_count = len(self.sessions.list_task_events(task.id))
        stale = AgentEvent(
            id="pause-stale",
            task_id=task.id,
            session_id=self.session_id,
            type="pause_requested",
            source="test",
            expected_version=task.version - 1,
        )

        with self.assertRaises(TaskVersionConflict):
            self.runtime.pause(task.id, event=stale)

        persisted = self.sessions.get_task(task.id)
        self.assertEqual(persisted.status, "running")
        self.assertEqual(persisted.version, task.version)
        self.assertEqual(len(self.sessions.list_task_events(task.id)), event_count)

    def test_duplicate_event_is_idempotent(self) -> None:
        task = self.runtime.start_for_user_message(session_id=self.session_id, user_input="work")
        event = AgentEvent(
            id="pause-once",
            task_id=task.id,
            session_id=self.session_id,
            type="pause_requested",
            source="test",
            expected_version=task.version,
        )

        paused = self.runtime.pause(task.id, event=event)
        duplicate = self.runtime.pause(task.id, event=event)

        self.assertEqual(duplicate, paused)
        self.assertEqual(
            [item.id for item in self.sessions.list_task_events(task.id)].count(event.id),
            1,
        )

    def test_event_and_snapshot_update_roll_back_together(self) -> None:
        task = self.runtime.start_for_user_message(session_id=self.session_id, user_input="work")
        connection = sqlite3.connect(self.db_path)
        try:
            connection.execute(
                """
                CREATE TRIGGER reject_pause BEFORE INSERT ON task_events
                WHEN NEW.event_type = 'pause_requested'
                BEGIN
                    SELECT RAISE(ABORT, 'reject pause');
                END
                """
            )
            connection.commit()
        finally:
            connection.close()

        with self.assertRaises(sqlite3.IntegrityError):
            self.runtime.pause(task.id)

        persisted = self.sessions.get_task(task.id)
        self.assertEqual(persisted.status, "running")
        self.assertEqual(persisted.version, task.version)
        self.assertEqual(
            [event.type for event in self.sessions.list_task_events(task.id)],
            ["task_created", "user_message"],
        )

    def test_waiting_user_can_be_approved_after_restart(self) -> None:
        task = self.runtime.start_for_user_message(session_id=self.session_id, user_input="edit")
        waiting = self.runtime.wait_for_user(
            task.id,
            PendingAction(kind="tool_approval", prompt="Approve?", decision={"tool": "file_write"}),
        )

        restarted_runtime = TaskRuntime(SessionService(self.db_path))
        event = AgentEvent(
            id="approve-after-restart",
            task_id=task.id,
            session_id=self.session_id,
            type="user_approved",
            source="test",
            expected_version=waiting.version,
        )
        resumed = restarted_runtime.approve(task.id, event=event)

        self.assertEqual(resumed.status, "running")
        self.assertIsNone(resumed.pending_action)
        self.assertIsNone(resumed.waiting_deadline)
        self.assertEqual(self.sessions.list_task_events(task.id)[-1].id, event.id)

    def test_waiting_user_expires_at_budget_deadline(self) -> None:
        task = self.sessions.create_task(
            self.session_id,
            goal="wait",
            budget=TaskBudget(waiting_user_timeout_seconds=0),
        )
        running = self.runtime.transition(
            task.id,
            target_status="running",
            event_type="user_message",
            source="user",
            reason="test",
        )
        waiting = self.runtime.wait_for_user(
            running.id,
            PendingAction(kind="ask_user", prompt="Need input"),
        )

        expired = self.runtime.expire_if_needed(waiting)

        self.assertEqual(expired.status, "expired")
        self.assertEqual(expired.stop_reason, "waiting_user_expired")

    def test_budget_usage_and_stop_reason_are_persisted(self) -> None:
        task = self.sessions.create_task(
            self.session_id,
            goal="bounded",
            budget=TaskBudget(max_model_calls=1, max_tool_calls=1, max_tokens=10),
        )
        running = self.runtime.transition(
            task.id,
            target_status="running",
            event_type="user_message",
            source="user",
            reason="test",
        )
        consumed = self.runtime.consume_model_call(running.id, tokens=10)

        self.assertEqual(consumed.budget.used_model_calls, 1)
        self.assertEqual(consumed.budget.used_tokens, 10)
        self.assertEqual(self.runtime.budget_stop_reason(consumed), "model_call_budget_exceeded")
        self.assertIn("budget", [trace.trace_type for trace in self.sessions.list_task_traces(task.id)])

    def test_legacy_session_todo_is_imported_once_into_task_plan(self) -> None:
        self.sessions.upsert_session_context(
            self.session_id,
            summary_text="keep this summary",
            summary_message_id=1,
            todo_items=(TodoItem(content="collect evidence", status="in_progress"),),
        )

        task = self.runtime.start_for_user_message(session_id=self.session_id, user_input="continue")

        self.assertEqual(task.plan, (TodoItem(content="collect evidence", status="in_progress"),))
        context = self.sessions.get_session_context(self.session_id)
        self.assertEqual(context.todo_items, ())
        self.assertEqual(context.summary_text, "keep this summary")

    def test_pause_resume_cancel_and_failure_paths_follow_legal_transitions(self) -> None:
        task = self.runtime.start_for_user_message(session_id=self.session_id, user_input="control")
        paused = self.runtime.pause(task.id)
        resumed = self.runtime.resume(paused.id)
        cancelled = self.runtime.cancel(resumed.id)

        self.assertEqual(paused.status, "paused")
        self.assertEqual(resumed.status, "running")
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(cancelled.stop_reason, "cancelled")
        self.assertEqual(self.runtime.fail(cancelled.id, reason="ignored"), cancelled)

        created = self.sessions.create_task(self.session_id, goal="fail before start")
        failed = self.runtime.fail(created.id, reason="internal_error")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.stop_reason, "internal_error")

    def test_user_response_and_rejection_clear_pending_action(self) -> None:
        task = self.runtime.start_for_user_message(session_id=self.session_id, user_input="clarify")
        waiting = self.runtime.wait_for_user(
            task.id,
            PendingAction(kind="ask_user", prompt="Which file?"),
        )
        response_event = AgentEvent(
            id="answer-event",
            task_id=task.id,
            session_id=self.session_id,
            type="user_message",
            source="test",
            payload={"content": "README.md"},
            expected_version=waiting.version,
        )
        resumed = self.runtime.start_for_user_message(
            session_id=self.session_id,
            user_input="README.md",
            event=response_event,
        )
        self.assertEqual(resumed.status, "running")
        self.assertIsNone(resumed.pending_action)

        waiting_again = self.runtime.wait_for_user(
            resumed.id,
            PendingAction(kind="tool_approval", prompt="Approve?"),
        )
        rejected = self.runtime.reject(waiting_again.id)
        self.assertEqual(rejected.status, "running")
        self.assertIsNone(rejected.pending_action)

        with self.assertRaises(InvalidTaskTransition):
            self.runtime.approve(rejected.id)

    def test_reflection_limit_active_time_and_all_budget_reasons(self) -> None:
        task = self.sessions.create_task(
            self.session_id,
            goal="limits",
            budget=TaskBudget(
                max_model_calls=5,
                max_tool_calls=1,
                max_tokens=100,
                max_active_seconds=1,
                max_replans=1,
            ),
        )
        running = self.runtime.transition(
            task.id,
            target_status="running",
            event_type="user_message",
            source="user",
            reason="test",
        )
        reflected = self.runtime.reflect(running.id, "try a smaller step")
        reflected_again = self.runtime.reflect(reflected.id, "ignored")
        self.assertEqual(reflected_again.version, reflected.version)
        self.assertEqual(reflected_again.reflection, "try a smaller step")

        tool_limited = self.runtime.consume_tool_call(reflected.id)
        self.assertEqual(self.runtime.budget_stop_reason(tool_limited), "tool_call_budget_exceeded")

        token_budget = TaskBudget(max_model_calls=5, max_tokens=1, used_tokens=1)
        token_task = self.sessions.create_task(self.session_id, goal="tokens", budget=token_budget)
        self.assertEqual(self.runtime.budget_stop_reason(token_task), "token_budget_exceeded")

        active_budget = TaskBudget(max_model_calls=5, max_tokens=100, max_active_seconds=1)
        active_task = self.sessions.create_task(self.session_id, goal="time", budget=active_budget)
        active_running = self.runtime.transition(
            active_task.id,
            target_status="running",
            event_type="user_message",
            source="user",
            reason="test",
        )
        timed = self.runtime.add_active_time(active_running.id, 2)
        self.assertEqual(self.runtime.budget_stop_reason(timed), "active_time_budget_exceeded")
        completed = self.runtime.complete(timed.id)
        self.assertEqual(self.runtime.add_active_time(completed.id, 10), completed)
        self.assertEqual(self.runtime.expire_if_needed(completed), completed)

    def test_invalid_transitions_events_and_missing_tasks_are_rejected(self) -> None:
        task = self.runtime.start_for_user_message(session_id=self.session_id, user_input="validate")
        with self.assertRaises(InvalidTaskTransition):
            self.runtime.resume(task.id)
        with self.assertRaises(KeyError):
            self.runtime.require_task("missing")

        wrong_task = AgentEvent(
            id="wrong-task",
            task_id="another-task",
            session_id=self.session_id,
            type="pause_requested",
            source="test",
        )
        with self.assertRaisesRegex(ValueError, "task_id"):
            self.runtime.pause(task.id, event=wrong_task)

        wrong_session = AgentEvent(
            id="wrong-session",
            task_id=task.id,
            session_id="another-session",
            type="pause_requested",
            source="test",
        )
        with self.assertRaisesRegex(ValueError, "session_id"):
            self.runtime.pause(task.id, event=wrong_session)

        wrong_type = AgentEvent(
            id="wrong-type",
            task_id=task.id,
            session_id=self.session_id,
            type="cancel_requested",
            source="test",
        )
        with self.assertRaisesRegex(ValueError, "Expected event type"):
            self.runtime.pause(task.id, event=wrong_type)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agent_app.types import (
    AgentEvent,
    Message,
    Observation,
    PendingAction,
    SessionContext,
    StoredMessage,
    SubagentRun,
    TaskBudget,
    TaskEvent,
    TaskState,
    TaskStatus,
    TaskTrace,
    TodoItem,
    ToolAction,
    ToolActionStatus,
    ToolCall,
    ToolCallTrace,
    ToolResult,
    TurnTrace,
)

_UNSET = object()


class TaskVersionConflict(RuntimeError):
    pass


class ActiveTaskConflict(RuntimeError):
    def __init__(self, task: TaskState) -> None:
        self.task = task
        super().__init__(
            f"Session '{task.session_id}' already has active task '{task.id}' "
            f"in status '{task.status}'. Resume or cancel it, or start a new session."
        )


class TracePersistenceError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class SessionService:
    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    @contextmanager
    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._db_path)
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def get_or_create_session(self, session_id: str | None = None) -> str:
        if session_id and self.session_exists(session_id):
            self.touch_session(session_id)
            return session_id
        return self.create_session(session_id)

    def create_session(self, session_id: str | None = None) -> str:
        resolved_id = session_id or str(uuid4())
        timestamp = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO sessions (id, created_at, updated_at)
                VALUES (?, COALESCE((SELECT created_at FROM sessions WHERE id = ?), ?), ?)
                """,
                (resolved_id, resolved_id, timestamp, timestamp),
            )
        return resolved_id

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM sessions WHERE id = ? LIMIT 1",
                (session_id,),
            ).fetchone()
        return row is not None

    def touch_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_utc_now(), session_id),
            )

    def create_task(
        self,
        session_id: str,
        *,
        goal: str,
        parent_task_id: str | None = None,
        budget: TaskBudget | None = None,
    ) -> TaskState:
        task_id = str(uuid4())
        timestamp = _utc_now()
        resolved_budget = budget or TaskBudget()
        session_context = self.get_session_context(session_id)
        previous_task = self.get_latest_task(session_id)
        unfinished_previous_plan = (
            previous_task.plan
            if previous_task is not None
            and previous_task.plan
            and any(item.status != "completed" for item in previous_task.plan)
            else ()
        )
        plan = session_context.todo_items or unfinished_previous_plan or _default_task_plan(goal)
        event_id = str(uuid4())
        try:
            with self._connect() as connection:
                active_row = connection.execute(
                    _TASK_SELECT
                    + """
                      WHERE session_id = ?
                        AND status IN ('created', 'running', 'waiting_user', 'waiting_tool', 'paused')
                      LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                if active_row is not None:
                    raise ActiveTaskConflict(_task_from_row(active_row))
                connection.execute(
                    """
                    INSERT INTO tasks (
                        id, session_id, parent_task_id, goal, status, step,
                        plan_json, working_memory_json, pending_action_json,
                        last_observation_json, reflection, budget_json, stop_reason,
                        version, waiting_deadline, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'created', 0, ?, '{}', NULL, NULL, NULL, ?, NULL, 1, NULL, ?, ?)
                    """,
                    (
                        task_id,
                        session_id,
                        parent_task_id,
                        goal,
                        _todo_items_json(plan),
                        _json_dumps(asdict(resolved_budget)),
                        timestamp,
                        timestamp,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO task_events (
                        id, task_id, session_id, event_type, source, payload_json,
                        correlation_id, causation_id, sequence, created_at
                    ) VALUES (?, ?, ?, 'task_created', 'runtime', ?, NULL, NULL, 1, ?)
                    """,
                    (event_id, task_id, session_id, _json_dumps({"goal": goal}), timestamp),
                )
                connection.execute(
                    """
                    INSERT INTO task_traces (task_id, session_id, trace_type, payload_json, created_at)
                    VALUES (?, ?, 'state_transition', ?, ?)
                    """,
                    (
                        task_id,
                        session_id,
                        _json_dumps({"from": None, "to": "created", "reason": "task_created", "version": 1}),
                        timestamp,
                    ),
                )
                if session_context.todo_items:
                    connection.execute(
                        """
                        UPDATE session_context
                        SET todo_json = '[]', updated_at = ?
                        WHERE session_id = ?
                        """,
                        (timestamp, session_id),
                    )
                row = connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (task_id,)).fetchone()
        except sqlite3.IntegrityError as exc:
            active = self.get_active_task(session_id)
            if active is not None:
                raise ActiveTaskConflict(active) from exc
            raise
        assert row is not None
        return _task_from_row(row)

    def get_task(self, task_id: str) -> TaskState | None:
        with self._connect() as connection:
            row = connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (task_id,)).fetchone()
        return None if row is None else _task_from_row(row)

    def list_tasks(self, session_id: str) -> list[TaskState]:
        with self._connect() as connection:
            rows = connection.execute(
                _TASK_SELECT + " WHERE session_id = ? ORDER BY created_at ASC, id ASC",
                (session_id,),
            ).fetchall()
        return [_task_from_row(row) for row in rows]

    def get_latest_task(self, session_id: str) -> TaskState | None:
        with self._connect() as connection:
            row = connection.execute(
                _TASK_SELECT + " WHERE session_id = ? ORDER BY updated_at DESC, created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return None if row is None else _task_from_row(row)

    def get_active_task(self, session_id: str) -> TaskState | None:
        with self._connect() as connection:
            row = connection.execute(
                _TASK_SELECT
                + """
                  WHERE session_id = ?
                    AND status IN ('created', 'running', 'waiting_user', 'waiting_tool', 'paused')
                  ORDER BY updated_at DESC, created_at DESC
                  LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        return None if row is None else _task_from_row(row)

    def apply_task_event(
        self,
        event: AgentEvent,
        *,
        target_status: TaskStatus | None = None,
        step: int | None = None,
        plan: tuple[TodoItem, ...] | None = None,
        working_memory: dict | None = None,
        pending_action: PendingAction | None | object = _UNSET,
        last_observation: Observation | None | object = _UNSET,
        reflection: str | None | object = _UNSET,
        budget: TaskBudget | None = None,
        stop_reason: str | None | object = _UNSET,
        waiting_deadline: str | None | object = _UNSET,
        transition_reason: str | None = None,
    ) -> TaskState:
        if event.task_id is None:
            raise ValueError("Task event requires task_id.")
        timestamp = event.created_at or _utc_now()
        with self._connect() as connection:
            duplicate = connection.execute(
                "SELECT 1 FROM task_events WHERE id = ? LIMIT 1",
                (event.id,),
            ).fetchone()
            row = connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (event.task_id,)).fetchone()
            if row is None:
                raise KeyError(event.task_id)
            current = _task_from_row(row)
            if duplicate is not None:
                return current
            if event.expected_version is not None and event.expected_version != current.version:
                raise TaskVersionConflict(
                    f"Task '{current.id}' version changed from {event.expected_version} to {current.version}."
                )

            sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM task_events WHERE task_id = ?",
                    (current.id,),
                ).fetchone()[0]
            )
            next_status = target_status or current.status
            next_version = current.version + 1
            next_step = current.step if step is None else step
            next_plan = current.plan if plan is None else plan
            next_memory = current.working_memory if working_memory is None else working_memory
            next_pending = current.pending_action if pending_action is _UNSET else pending_action
            next_observation = current.last_observation if last_observation is _UNSET else last_observation
            next_reflection = current.reflection if reflection is _UNSET else reflection
            next_budget = current.budget if budget is None else budget
            next_stop_reason = current.stop_reason if stop_reason is _UNSET else stop_reason
            next_waiting_deadline = current.waiting_deadline if waiting_deadline is _UNSET else waiting_deadline

            connection.execute(
                """
                UPDATE tasks
                SET status = ?, step = ?, plan_json = ?, working_memory_json = ?,
                    pending_action_json = ?, last_observation_json = ?, reflection = ?,
                    budget_json = ?, stop_reason = ?, version = ?, waiting_deadline = ?,
                    updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (
                    next_status,
                    next_step,
                    _todo_items_json(next_plan),
                    _json_dumps(next_memory),
                    _optional_dataclass_json(next_pending),
                    _optional_dataclass_json(next_observation),
                    next_reflection,
                    _json_dumps(asdict(next_budget)),
                    next_stop_reason,
                    next_version,
                    next_waiting_deadline,
                    timestamp,
                    current.id,
                    current.version,
                ),
            )
            if connection.total_changes == 0:
                raise TaskVersionConflict(f"Task '{current.id}' was updated concurrently.")
            connection.execute(
                """
                INSERT INTO task_events (
                    id, task_id, session_id, event_type, source, payload_json,
                    correlation_id, causation_id, sequence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    current.id,
                    current.session_id,
                    event.type,
                    event.source,
                    _json_dumps(event.payload),
                    event.correlation_id,
                    event.causation_id,
                    sequence,
                    timestamp,
                ),
            )
            if next_status != current.status:
                connection.execute(
                    """
                    INSERT INTO task_traces (task_id, session_id, trace_type, payload_json, created_at)
                    VALUES (?, ?, 'state_transition', ?, ?)
                    """,
                    (
                        current.id,
                        current.session_id,
                        _json_dumps(
                            {
                                "from": current.status,
                                "to": next_status,
                                "reason": transition_reason or event.type,
                                "event_id": event.id,
                                "version": next_version,
                                "budget": asdict(next_budget),
                            }
                        ),
                        timestamp,
                    ),
                )
            if next_budget != current.budget:
                connection.execute(
                    """
                    INSERT INTO task_traces (task_id, session_id, trace_type, payload_json, created_at)
                    VALUES (?, ?, 'budget', ?, ?)
                    """,
                    (
                        current.id,
                        current.session_id,
                        _json_dumps({"event_id": event.id, "budget": asdict(next_budget)}),
                        timestamp,
                    ),
                )
            updated_row = connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (current.id,)).fetchone()
        assert updated_row is not None
        return _task_from_row(updated_row)

    def list_task_events(self, task_id: str) -> list[TaskEvent]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, task_id, session_id, event_type, source, payload_json,
                       correlation_id, causation_id, sequence, created_at
                FROM task_events
                WHERE task_id = ?
                ORDER BY sequence ASC
                """,
                (task_id,),
            ).fetchall()
        return [
            TaskEvent(
                id=row[0],
                task_id=row[1],
                session_id=row[2],
                type=row[3],
                source=row[4],
                payload=json.loads(row[5]),
                correlation_id=row[6],
                causation_id=row[7],
                sequence=row[8],
                created_at=row[9],
            )
            for row in rows
        ]

    def task_event_exists(self, event_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT 1 FROM task_events WHERE id = ? LIMIT 1",
                (event_id,),
            ).fetchone()
        return row is not None

    def append_task_trace(self, task_id: str, trace_type: str, payload: dict) -> int:
        try:
            task = self.get_task(task_id)
            if task is None:
                raise KeyError(task_id)
            timestamp = _utc_now()
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO task_traces (task_id, session_id, trace_type, payload_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (task.id, task.session_id, trace_type, _json_dumps(payload), timestamp),
                )
            return int(cursor.lastrowid)
        except TracePersistenceError:
            raise
        except Exception as exc:
            raise TracePersistenceError(
                f"Failed to persist task trace '{trace_type}' for task '{task_id}'."
            ) from exc

    def list_task_traces(self, task_id: str) -> list[TaskTrace]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, task_id, session_id, trace_type, payload_json, created_at
                FROM task_traces
                WHERE task_id = ?
                ORDER BY id ASC
                """,
                (task_id,),
            ).fetchall()
        return [
            TaskTrace(
                id=row[0],
                task_id=row[1],
                session_id=row[2],
                trace_type=row[3],
                payload=json.loads(row[4]),
                created_at=row[5],
            )
            for row in rows
        ]

    def append_message(self, session_id: str, message: Message) -> None:
        timestamp = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO messages (
                    session_id, role, content, tool_call_id, tool_name, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    message.role,
                    message.content,
                    message.tool_call_id,
                    message.tool_name,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (timestamp, session_id),
            )

    def append_tool_run(self, session_id: str, tool_result: ToolResult) -> None:
        timestamp = _utc_now()
        with self._connect() as connection:
            _insert_tool_run(connection, session_id=session_id, tool_result=tool_result, timestamp=timestamp)
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (timestamp, session_id),
            )

    def prepare_tool_action(
        self,
        session_id: str,
        *,
        agent_id: str,
        tool_call: ToolCall,
        recovery_metadata: dict,
        task_id: str | None = None,
        attempt: int = 1,
        retry_of: str | None = None,
    ) -> ToolAction:
        timestamp = _utc_now()
        idempotency_key = f"{session_id}:{agent_id}:{task_id or '-'}:{tool_call.id}:{attempt}"
        action_id = str(uuid4())
        arguments_json = json.dumps(tool_call.arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        recovery_json = json.dumps(recovery_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO tool_actions (
                    id, session_id, task_id, agent_id, tool_call_id, tool_name,
                    arguments_json, idempotency_key, status, recovery_json,
                    prepared_at, attempt, retry_of, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'prepared', ?, ?, ?, ?, ?)
                """,
                (
                    action_id,
                    session_id,
                    task_id,
                    agent_id,
                    tool_call.id,
                    tool_call.name,
                    arguments_json,
                    idempotency_key,
                    recovery_json,
                    timestamp,
                    attempt,
                    retry_of,
                    timestamp,
                ),
            )
            row = connection.execute(
                _TOOL_ACTION_SELECT + " WHERE idempotency_key = ? LIMIT 1",
                (idempotency_key,),
            ).fetchone()
        assert row is not None
        return _tool_action_from_row(row)

    def mark_tool_action_executing(self, action_id: str) -> ToolAction:
        timestamp = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE tool_actions
                SET status = 'executing', started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE id = ? AND status = 'prepared'
                """,
                (timestamp, timestamp, action_id),
            )
            row = connection.execute(
                _TOOL_ACTION_SELECT + " WHERE id = ? LIMIT 1",
                (action_id,),
            ).fetchone()
        if row is None:
            raise KeyError(action_id)
        return _tool_action_from_row(row)

    def complete_tool_action(
        self,
        action_id: str,
        *,
        status: ToolActionStatus,
        tool_result: ToolResult,
    ) -> ToolAction:
        if status not in {"succeeded", "failed", "uncertain"}:
            raise ValueError("Tool action can only complete as succeeded, failed, or uncertain.")

        timestamp = _utc_now()
        with self._connect() as connection:
            row = connection.execute(
                _TOOL_ACTION_SELECT + " WHERE id = ? LIMIT 1",
                (action_id,),
            ).fetchone()
            if row is None:
                raise KeyError(action_id)
            existing = _tool_action_from_row(row)
            if existing.status in {"succeeded", "failed", "uncertain"}:
                return existing

            connection.execute(
                """
                UPDATE tool_actions
                SET status = ?, result_success = ?, result_content = ?, result_error = ?,
                    completed_at = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    1 if tool_result.success else 0,
                    tool_result.content,
                    tool_result.error,
                    timestamp,
                    timestamp,
                    action_id,
                ),
            )
            _insert_tool_run(
                connection,
                session_id=existing.session_id,
                tool_result=tool_result,
                timestamp=timestamp,
                action_id=action_id,
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (timestamp, existing.session_id),
            )
            completed_row = connection.execute(
                _TOOL_ACTION_SELECT + " WHERE id = ? LIMIT 1",
                (action_id,),
            ).fetchone()
        assert completed_row is not None
        return _tool_action_from_row(completed_row)

    def get_tool_action_by_idempotency_key(self, idempotency_key: str) -> ToolAction | None:
        with self._connect() as connection:
            row = connection.execute(
                _TOOL_ACTION_SELECT + " WHERE idempotency_key = ? LIMIT 1",
                (idempotency_key,),
            ).fetchone()
        return None if row is None else _tool_action_from_row(row)

    def list_tool_actions(self, session_id: str) -> list[ToolAction]:
        with self._connect() as connection:
            rows = connection.execute(
                _TOOL_ACTION_SELECT + " WHERE session_id = ? ORDER BY prepared_at ASC, id ASC",
                (session_id,),
            ).fetchall()
        return [_tool_action_from_row(row) for row in rows]

    def list_recoverable_tool_actions(self, session_id: str) -> list[ToolAction]:
        with self._connect() as connection:
            rows = connection.execute(
                _TOOL_ACTION_SELECT
                + " WHERE session_id = ? AND status IN ('prepared', 'executing') ORDER BY prepared_at ASC, id ASC",
                (session_id,),
            ).fetchall()
        return [_tool_action_from_row(row) for row in rows]

    def list_uncertain_tool_actions(self, session_id: str) -> list[ToolAction]:
        with self._connect() as connection:
            rows = connection.execute(
                _TOOL_ACTION_SELECT + " WHERE session_id = ? AND status = 'uncertain' ORDER BY prepared_at ASC, id ASC",
                (session_id,),
            ).fetchall()
        return [_tool_action_from_row(row) for row in rows]

    def append_subagent_run(
        self,
        *,
        parent_session_id: str,
        parent_tool_call_id: str,
        child_session_id: str,
        agent_id: str,
        task: str,
        success: bool,
        result_summary: str,
    ) -> None:
        timestamp = _utc_now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO subagent_runs (
                    parent_session_id, parent_tool_call_id, child_session_id, agent_id,
                    task, success, result_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parent_session_id,
                    parent_tool_call_id,
                    child_session_id,
                    agent_id,
                    task,
                    1 if success else 0,
                    result_summary,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (timestamp, parent_session_id),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (timestamp, child_session_id),
            )

    def list_recent_messages(
        self,
        session_id: str,
        limit: int = 16,
    ) -> list[Message]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT role, content, tool_call_id, tool_name
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            Message(
                role=row[0],
                content=row[1],
                tool_call_id=row[2],
                tool_name=row[3],
            )
            for row in reversed(rows)
        ]

    def list_messages(self, session_id: str) -> list[StoredMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, role, content, tool_call_id, tool_name
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            StoredMessage(
                id=row[0],
                role=row[1],
                content=row[2],
                tool_call_id=row[3],
                tool_name=row[4],
            )
            for row in rows
        ]

    def list_tool_runs(self, session_id: str) -> list[ToolResult]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT tool_call_id, tool_name, success, content, error
                FROM tool_runs
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            ToolResult(
                tool_call_id=row[0],
                tool_name=row[1],
                success=bool(row[2]),
                content=row[3],
                error=row[4],
            )
            for row in rows
        ]

    def list_subagent_runs(self, parent_session_id: str) -> list[SubagentRun]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT parent_session_id, parent_tool_call_id, child_session_id, agent_id,
                       task, success, result_summary, created_at
                FROM subagent_runs
                WHERE parent_session_id = ?
                ORDER BY created_at ASC, child_session_id ASC
                """,
                (parent_session_id,),
            ).fetchall()
        return [
            SubagentRun(
                parent_session_id=row[0],
                parent_tool_call_id=row[1],
                child_session_id=row[2],
                agent_id=row[3],
                task=row[4],
                success=bool(row[5]),
                result_summary=row[6],
                created_at=row[7],
            )
            for row in rows
        ]

    def get_session_context(self, session_id: str) -> SessionContext:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT summary_text, summary_message_id, todo_json
                FROM session_context
                WHERE session_id = ?
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()

        if row is None:
            return SessionContext()

        todo_json = row[2]
        todo_items: tuple[TodoItem, ...] = ()
        if todo_json:
            decoded = json.loads(todo_json)
            todo_items = tuple(
                TodoItem(content=str(item["content"]), status=str(item["status"]))
                for item in decoded
            )

        return SessionContext(
            summary_text=row[0],
            summary_message_id=row[1],
            todo_items=todo_items,
        )

    def upsert_session_context(
        self,
        session_id: str,
        *,
        summary_text: str | None,
        summary_message_id: int | None,
        todo_items: tuple[TodoItem, ...],
    ) -> None:
        timestamp = _utc_now()
        todo_json = json.dumps(
            [{"content": item.content, "status": item.status} for item in todo_items],
            ensure_ascii=False,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_context (
                    session_id, summary_text, summary_message_id, todo_json, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    summary_message_id = excluded.summary_message_id,
                    todo_json = excluded.todo_json,
                    updated_at = excluded.updated_at
                """,
                (session_id, summary_text, summary_message_id, todo_json, timestamp),
            )

    def clear_session_context(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM session_context WHERE session_id = ?",
                (session_id,),
            )

    def append_turn_trace(
        self,
        session_id: str,
        *,
        user_input: str,
        context_message_count: int,
        context_token_estimate: int,
        used_summary: bool,
        used_todo: bool,
        used_evidence: bool,
        final_text: str | None,
        stop_reason: str | None,
        success: bool,
        tool_traces: list[ToolResult],
    ) -> int:
        timestamp = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO turn_traces (
                    session_id, user_input, context_message_count, context_token_estimate,
                    used_summary, used_todo, used_evidence, final_text, stop_reason, success, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_input,
                    context_message_count,
                    context_token_estimate,
                    1 if used_summary else 0,
                    1 if used_todo else 0,
                    1 if used_evidence else 0,
                    final_text,
                    stop_reason,
                    1 if success else 0,
                    timestamp,
                ),
            )
            turn_trace_id = int(cursor.lastrowid)
            for tool_trace in tool_traces:
                connection.execute(
                    """
                    INSERT INTO tool_call_traces (
                        turn_trace_id, tool_call_id, tool_name, success, error, content_preview, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        turn_trace_id,
                        tool_trace.tool_call_id,
                        tool_trace.tool_name,
                        1 if tool_trace.success else 0,
                        tool_trace.error,
                        tool_trace.content[:500],
                        timestamp,
                    ),
                )
        return turn_trace_id

    def list_turn_traces(self, session_id: str) -> list[TurnTrace]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, user_input, context_message_count, context_token_estimate,
                       used_summary, used_todo, used_evidence, final_text, stop_reason, success, created_at
                FROM turn_traces
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            TurnTrace(
                id=row[0],
                session_id=row[1],
                user_input=row[2],
                context_message_count=row[3],
                context_token_estimate=row[4],
                used_summary=bool(row[5]),
                used_todo=bool(row[6]),
                used_evidence=bool(row[7]),
                final_text=row[8],
                stop_reason=row[9],
                success=bool(row[10]),
                created_at=row[11],
            )
            for row in rows
        ]

    def list_tool_call_traces(self, turn_trace_id: int) -> list[ToolCallTrace]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, turn_trace_id, tool_call_id, tool_name, success, error, content_preview, created_at
                FROM tool_call_traces
                WHERE turn_trace_id = ?
                ORDER BY id ASC
                """,
                (turn_trace_id,),
            ).fetchall()
        return [
            ToolCallTrace(
                id=row[0],
                turn_trace_id=row[1],
                tool_call_id=row[2],
                tool_name=row[3],
                success=bool(row[4]),
                error=row[5],
                content_preview=row[6],
                created_at=row[7],
            )
            for row in rows
        ]


_TOOL_ACTION_SELECT = """
SELECT id, session_id, task_id, agent_id, tool_call_id, tool_name, arguments_json,
       idempotency_key, status, recovery_json, result_success, result_content,
       result_error, prepared_at, started_at, completed_at, updated_at, attempt, retry_of
FROM tool_actions
"""


def _tool_action_from_row(row) -> ToolAction:
    result = None
    if row[10] is not None:
        result = ToolResult(
            tool_call_id=row[4],
            tool_name=row[5],
            success=bool(row[10]),
            content=row[11] or "",
            error=row[12],
        )
    return ToolAction(
        id=row[0],
        session_id=row[1],
        task_id=row[2],
        agent_id=row[3],
        tool_call_id=row[4],
        tool_name=row[5],
        arguments=json.loads(row[6]),
        idempotency_key=row[7],
        status=row[8],
        recovery_metadata=json.loads(row[9]),
        result=result,
        prepared_at=row[13],
        started_at=row[14],
        completed_at=row[15],
        updated_at=row[16],
        attempt=row[17],
        retry_of=row[18],
    )


_TASK_SELECT = """
SELECT id, session_id, parent_task_id, goal, status, step, plan_json,
       working_memory_json, pending_action_json, last_observation_json,
       reflection, budget_json, stop_reason, version, waiting_deadline,
       created_at, updated_at
FROM tasks
"""


def _task_from_row(row) -> TaskState:
    pending_raw = json.loads(row[8]) if row[8] else None
    if pending_raw is not None and "id" not in pending_raw:
        pending_raw["id"] = f"legacy-{row[0]}"
    observation_raw = json.loads(row[9]) if row[9] else None
    budget_raw = json.loads(row[11])
    return TaskState(
        id=row[0],
        session_id=row[1],
        parent_task_id=row[2],
        goal=row[3],
        status=row[4],
        step=row[5],
        plan=tuple(TodoItem(content=item["content"], status=item["status"]) for item in json.loads(row[6])),
        working_memory=json.loads(row[7]),
        pending_action=PendingAction(**pending_raw) if pending_raw else None,
        last_observation=Observation(**observation_raw) if observation_raw else None,
        reflection=row[10],
        budget=TaskBudget(**budget_raw),
        stop_reason=row[12],
        version=row[13],
        waiting_deadline=row[14],
        created_at=row[15],
        updated_at=row[16],
    )


def _default_task_plan(goal: str) -> tuple[TodoItem, ...]:
    multi_step_markers = (" and ", " then ", "并", "然后", "再", "修改", "验证", "实现", "分析")
    if not any(marker in goal.lower() for marker in multi_step_markers):
        return ()
    return (
        TodoItem(content="Understand the goal and gather evidence", status="in_progress"),
        TodoItem(content="Execute the required changes or actions", status="pending"),
        TodoItem(content="Verify the result and report", status="pending"),
    )


def _todo_items_json(items: tuple[TodoItem, ...]) -> str:
    return _json_dumps([asdict(item) for item in items])


def _optional_dataclass_json(value) -> str | None:
    if value is None:
        return None
    return _json_dumps(asdict(value))


def _json_dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _insert_tool_run(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    tool_result: ToolResult,
    timestamp: str,
    action_id: str | None = None,
) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO tool_runs (
            session_id, action_id, tool_call_id, tool_name, success, content, error, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            session_id,
            action_id,
            tool_result.tool_call_id,
            tool_result.tool_name,
            1 if tool_result.success else 0,
            tool_result.content,
            tool_result.error,
            timestamp,
        ),
    )

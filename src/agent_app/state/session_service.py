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
    SessionOverview,
    SkillDraft,
    SkillActivation,
    StoredMessage,
    SubagentRun,
    TaskBudget,
    TaskEvent,
    TaskHandoff,
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

    def list_recent_session_overviews(self, *, limit: int = 8) -> list[SessionOverview]:
        """Return a compact, read-only cross-session view for the REPL."""
        if not 1 <= limit <= 20:
            raise ValueError("Session overview limit must be between 1 and 20.")
        with self._connect() as connection:
            session_rows = connection.execute(
                """
                SELECT id, created_at, updated_at
                FROM sessions
                ORDER BY updated_at DESC, created_at DESC, id ASC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            overviews: list[SessionOverview] = []
            for session_id, created_at, updated_at in session_rows:
                latest_row = connection.execute(
                    _TASK_SELECT
                    + " WHERE session_id = ? ORDER BY updated_at DESC, created_at DESC, id ASC LIMIT 1",
                    (session_id,),
                ).fetchone()
                active_row = connection.execute(
                    _TASK_SELECT
                    + """
                      WHERE session_id = ?
                        AND status IN ('created', 'running', 'waiting_user', 'waiting_tool', 'paused')
                      ORDER BY updated_at DESC, created_at DESC, id ASC
                      LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                task_count = int(
                    connection.execute("SELECT COUNT(*) FROM tasks WHERE session_id = ?", (session_id,)).fetchone()[0]
                )
                context_row = connection.execute(
                    """
                    SELECT summary_text, summary_message_id, todo_json
                    FROM session_context WHERE session_id = ? LIMIT 1
                    """,
                    (session_id,),
                ).fetchone()
                overviews.append(
                    SessionOverview(
                        id=session_id,
                        created_at=created_at,
                        updated_at=updated_at,
                        task_count=task_count,
                        latest_task=_task_from_row(latest_row) if latest_row is not None else None,
                        active_task=_task_from_row(active_row) if active_row is not None else None,
                        context=_session_context_from_row(context_row),
                    )
                )
        return overviews

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

    def activate_skill(
        self,
        *,
        task_id: str,
        skill_name: str,
        scope: str,
        source_path: str,
        content_hash: str,
        version: str | None,
        activation_reason: str,
        source: str,
    ) -> SkillActivation:
        """Persist a task-local Skill activation with its own event and trace."""
        if scope not in {"project", "user"}:
            raise ValueError("Skill scope must be project or user.")
        if activation_reason not in {"explicit", "model_match", "inherited_handoff"}:
            raise ValueError("Unknown Skill activation reason.")
        timestamp = _utc_now()
        with self._connect() as connection:
            row = connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            task = _task_from_row(row)
            if task.status != "running":
                raise RuntimeError(f"Skill activation requires a running task (current: {task.status}).")
            existing_row = connection.execute(
                _SKILL_ACTIVATION_SELECT + " WHERE task_id = ? AND skill_name = ? AND state = 'active' ORDER BY id DESC LIMIT 1",
                (task.id, skill_name),
            ).fetchone()
            if existing_row is not None:
                existing = _skill_activation_from_row(existing_row)
                if (
                    existing.scope == scope
                    and existing.source_path == source_path
                    and existing.content_hash == content_hash
                    and existing.version == version
                ):
                    return existing
                connection.execute("UPDATE task_skill_activations SET state = 'dropped' WHERE id = ?", (existing_row[0],))

            activation = SkillActivation(
                task_id=task.id,
                skill_name=skill_name,
                scope=scope,
                source_path=source_path,
                content_hash=content_hash,
                version=version,
                activation_reason=activation_reason,
                state="active",
                activated_at=timestamp,
            )
            connection.execute(
                """
                INSERT INTO task_skill_activations (
                    task_id, skill_name, scope, source_path, content_hash, version,
                    activation_reason, state, activated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    activation.task_id,
                    activation.skill_name,
                    activation.scope,
                    activation.source_path,
                    activation.content_hash,
                    activation.version,
                    activation.activation_reason,
                    activation.activated_at,
                ),
            )
            self._append_skill_event(
                connection,
                task=task,
                event_type="skill_activated",
                source=source,
                payload={
                    "skill_name": skill_name,
                    "scope": scope,
                    "content_hash": content_hash,
                    "activation_reason": activation_reason,
                },
                trace_type="skill_activation",
                trace_payload={
                    "skill_name": skill_name,
                    "scope": scope,
                    "content_hash": content_hash,
                    "activation_reason": activation_reason,
                    "state": "active",
                },
                timestamp=timestamp,
            )
        return activation

    def drop_skill(self, *, task_id: str, skill_name: str, source: str = "user") -> SkillActivation | None:
        timestamp = _utc_now()
        with self._connect() as connection:
            row = connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            task = _task_from_row(row)
            if task.status != "running":
                raise RuntimeError(f"Skill deactivation requires a running task (current: {task.status}).")
            activation_row = connection.execute(
                _SKILL_ACTIVATION_SELECT + " WHERE task_id = ? AND skill_name = ? AND state = 'active' ORDER BY id DESC LIMIT 1",
                (task.id, skill_name),
            ).fetchone()
            if activation_row is None:
                return None
            connection.execute("UPDATE task_skill_activations SET state = 'dropped' WHERE id = ?", (activation_row[0],))
            dropped = _skill_activation_from_row((*activation_row[:8], "dropped", activation_row[9]))
            self._append_skill_event(
                connection,
                task=task,
                event_type="skill_dropped",
                source=source,
                payload={"skill_name": skill_name},
                trace_type="skill_activation",
                trace_payload={"skill_name": skill_name, "state": "dropped"},
                timestamp=timestamp,
            )
        return dropped

    def list_skill_activations(self, task_id: str, *, active_only: bool = False) -> list[SkillActivation]:
        query = _SKILL_ACTIVATION_SELECT + " WHERE task_id = ?"
        if active_only:
            query += " AND state = 'active'"
        query += " ORDER BY id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, (task_id,)).fetchall()
        return [_skill_activation_from_row(row) for row in rows]

    def list_active_skill_activations(self, task_id: str) -> list[SkillActivation]:
        return self.list_skill_activations(task_id, active_only=True)

    def handoff_task(
        self,
        *,
        source_task_id: str,
        target_session_id: str,
        summary_text: str | None,
        evidence_refs: tuple[str, ...],
        inherited_skills: tuple[SkillActivation, ...] = (),
    ) -> tuple[TaskState, TaskHandoff]:
        """Atomically archive a resumable checkpoint and create its child in another session."""
        timestamp = _utc_now()
        child_id = str(uuid4())
        with self._connect() as connection:
            source_row = connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (source_task_id,)).fetchone()
            if source_row is None:
                raise KeyError(source_task_id)
            source_task = _task_from_row(source_row)
            if source_task.status not in {"running", "paused", "completed"}:
                raise RuntimeError("Only a running, paused, or completed task can be handed off.")
            if source_task.pending_action is not None:
                raise RuntimeError("Resolve the pending action before handing off a task.")
            target_session = connection.execute("SELECT 1 FROM sessions WHERE id = ? LIMIT 1", (target_session_id,)).fetchone()
            if target_session is None:
                raise KeyError(f"Unknown target session '{target_session_id}'.")
            active_target = connection.execute(
                _TASK_SELECT
                + " WHERE session_id = ? AND status IN ('created', 'running', 'waiting_user', 'waiting_tool', 'paused') LIMIT 1",
                (target_session_id,),
            ).fetchone()
            if active_target is not None:
                raise ActiveTaskConflict(_task_from_row(active_target))
            previous_handoff = connection.execute(
                "SELECT 1 FROM task_handoffs WHERE source_task_id = ? LIMIT 1", (source_task.id,)
            ).fetchone()
            if previous_handoff is not None:
                raise RuntimeError("Task has already been handed off.")

            remaining_plan = tuple(item for item in source_task.plan if item.status != "completed")
            compact_summary = _compact_handoff_summary(summary_text, source_task=source_task, evidence_refs=evidence_refs)
            next_source_version = source_task.version + 1
            source_event_id = str(uuid4())
            source_sequence = int(
                connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) + 1 FROM task_events WHERE task_id = ?", (source_task.id,)
                ).fetchone()[0]
            )
            source_update = connection.execute(
                """
                UPDATE tasks
                SET status = 'handed_off', pending_action_json = NULL, waiting_deadline = NULL,
                    stop_reason = 'handed_off', version = ?, updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (next_source_version, timestamp, source_task.id, source_task.version),
            )
            if source_update.rowcount != 1:
                raise TaskVersionConflict(f"Task '{source_task.id}' was updated concurrently.")
            connection.execute(
                """
                INSERT INTO task_events (
                    id, task_id, session_id, event_type, source, payload_json,
                    correlation_id, causation_id, sequence, created_at
                ) VALUES (?, ?, ?, 'task_handed_off', 'user', ?, ?, NULL, ?, ?)
                """,
                (
                    source_event_id,
                    source_task.id,
                    source_task.session_id,
                    _json_dumps({"target_session_id": target_session_id, "target_task_id": child_id}),
                    source_task.id,
                    source_sequence,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO task_traces (task_id, session_id, trace_type, payload_json, created_at)
                VALUES (?, ?, 'state_transition', ?, ?)
                """,
                (
                    source_task.id,
                    source_task.session_id,
                    _json_dumps({"from": source_task.status, "to": "handed_off", "reason": "task_handed_off", "version": next_source_version}),
                    timestamp,
                ),
            )

            connection.execute(
                """
                INSERT INTO tasks (
                    id, session_id, parent_task_id, goal, status, step, plan_json,
                    working_memory_json, pending_action_json, last_observation_json,
                    reflection, budget_json, stop_reason, version, waiting_deadline, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'created', ?, ?, '{}', NULL, NULL, ?, ?, NULL, 1, NULL, ?, ?)
                """,
                (
                    child_id,
                    target_session_id,
                    source_task.id,
                    source_task.goal,
                    source_task.step,
                    _todo_items_json(remaining_plan),
                    source_task.reflection,
                    _json_dumps(asdict(source_task.budget)),
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO task_events (
                    id, task_id, session_id, event_type, source, payload_json,
                    correlation_id, causation_id, sequence, created_at
                ) VALUES (?, ?, ?, 'task_created', 'handoff', ?, ?, ?, 1, ?)
                """,
                (
                    str(uuid4()),
                    child_id,
                    target_session_id,
                    _json_dumps({"goal": source_task.goal, "parent_task_id": source_task.id}),
                    child_id,
                    source_event_id,
                    timestamp,
                ),
            )
            connection.execute(
                """
                INSERT INTO task_traces (task_id, session_id, trace_type, payload_json, created_at)
                VALUES (?, ?, 'state_transition', ?, ?)
                """,
                (child_id, target_session_id, _json_dumps({"from": None, "to": "created", "reason": "task_handoff_created", "version": 1}), timestamp),
            )
            connection.execute(
                """
                INSERT INTO task_handoffs (
                    source_task_id, target_task_id, target_session_id, summary_text, evidence_refs_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_task.id, child_id, target_session_id, compact_summary, _json_dumps(list(evidence_refs)), timestamp),
            )
            connection.execute(
                """
                INSERT INTO session_context (session_id, summary_text, summary_message_id, todo_json, updated_at)
                VALUES (?, ?, NULL, '[]', ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    summary_text = excluded.summary_text,
                    summary_message_id = NULL,
                    todo_json = '[]',
                    updated_at = excluded.updated_at
                """,
                (target_session_id, compact_summary, timestamp),
            )
            child_task = _task_from_row(connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (child_id,)).fetchone())
            for activation in _unique_active_skills(inherited_skills):
                connection.execute(
                    """
                    INSERT INTO task_skill_activations (
                        task_id, skill_name, scope, source_path, content_hash, version,
                        activation_reason, state, activated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'inherited_handoff', 'active', ?)
                    """,
                    (
                        child_id,
                        activation.skill_name,
                        activation.scope,
                        activation.source_path,
                        activation.content_hash,
                        activation.version,
                        timestamp,
                    ),
                )
                child_task = self._append_skill_event(
                    connection,
                    task=child_task,
                    event_type="skill_activated",
                    source="handoff",
                    payload={
                        "skill_name": activation.skill_name,
                        "scope": activation.scope,
                        "content_hash": activation.content_hash,
                        "activation_reason": "inherited_handoff",
                    },
                    trace_type="skill_activation",
                    trace_payload={
                        "skill_name": activation.skill_name,
                        "scope": activation.scope,
                        "content_hash": activation.content_hash,
                        "activation_reason": "inherited_handoff",
                        "state": "active",
                    },
                    timestamp=timestamp,
                )
            updated_child_row = connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (child_id,)).fetchone()
        assert updated_child_row is not None
        handoff = TaskHandoff(
            source_task_id=source_task.id,
            target_task_id=child_id,
            target_session_id=target_session_id,
            summary_text=compact_summary,
            evidence_refs=evidence_refs,
            created_at=timestamp,
        )
        return _task_from_row(updated_child_row), handoff

    def get_task_handoff(self, source_task_id: str) -> TaskHandoff | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT source_task_id, target_task_id, target_session_id, summary_text, evidence_refs_json, created_at
                FROM task_handoffs WHERE source_task_id = ? LIMIT 1
                """,
                (source_task_id,),
            ).fetchone()
        if row is None:
            return None
        return TaskHandoff(
            source_task_id=row[0],
            target_task_id=row[1],
            target_session_id=row[2],
            summary_text=row[3],
            evidence_refs=tuple(json.loads(row[4])),
            created_at=row[5],
        )

    def create_skill_draft(
        self,
        *,
        session_id: str,
        scope: str,
        skill_name: str,
        content: str,
        content_hash: str,
    ) -> SkillDraft:
        if scope not in {"project", "user"}:
            raise ValueError("Skill draft scope must be project or user.")
        if not self.session_exists(session_id):
            raise KeyError(session_id)
        draft = SkillDraft(
            id=str(uuid4()),
            session_id=session_id,
            scope=scope,
            skill_name=skill_name,
            content=content,
            content_hash=content_hash,
            status="draft",
            created_at=_utc_now(),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO skill_drafts (
                    id, session_id, scope, skill_name, content, content_hash,
                    status, created_at, saved_at, saved_path
                ) VALUES (?, ?, ?, ?, ?, ?, 'draft', ?, NULL, NULL)
                """,
                (
                    draft.id,
                    draft.session_id,
                    draft.scope,
                    draft.skill_name,
                    draft.content,
                    draft.content_hash,
                    draft.created_at,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (draft.created_at, draft.session_id),
            )
        return draft

    def list_skill_drafts(self, session_id: str, *, status: str | None = None, limit: int = 12) -> list[SkillDraft]:
        if not 1 <= limit <= 50:
            raise ValueError("Skill draft limit must be between 1 and 50.")
        query = _SKILL_DRAFT_SELECT + " WHERE session_id = ?"
        parameters: list[object] = [session_id]
        if status is not None:
            if status not in {"draft", "saved"}:
                raise ValueError("Unknown Skill draft status.")
            query += " AND status = ?"
            parameters.append(status)
        query += " ORDER BY created_at DESC, id ASC LIMIT ?"
        parameters.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [_skill_draft_from_row(row) for row in rows]

    def find_skill_drafts(self, session_id: str, *, id_prefix: str, status: str = "draft") -> list[SkillDraft]:
        if not id_prefix:
            return self.list_skill_drafts(session_id, status=status)
        with self._connect() as connection:
            rows = connection.execute(
                _SKILL_DRAFT_SELECT
                + " WHERE session_id = ? AND status = ? AND id LIKE ? ORDER BY created_at DESC, id ASC",
                (session_id, status, f"{id_prefix}%"),
            ).fetchall()
        return [_skill_draft_from_row(row) for row in rows]

    def mark_skill_draft_saved(self, draft_id: str, *, saved_path: str) -> SkillDraft:
        timestamp = _utc_now()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE skill_drafts
                SET status = 'saved', saved_at = ?, saved_path = ?
                WHERE id = ? AND status = 'draft'
                """,
                (timestamp, saved_path, draft_id),
            )
            if cursor.rowcount != 1:
                row = connection.execute(_SKILL_DRAFT_SELECT + " WHERE id = ? LIMIT 1", (draft_id,)).fetchone()
                if row is None:
                    raise KeyError(draft_id)
                return _skill_draft_from_row(row)
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = (SELECT session_id FROM skill_drafts WHERE id = ?)",
                (timestamp, draft_id),
            )
            row = connection.execute(_SKILL_DRAFT_SELECT + " WHERE id = ? LIMIT 1", (draft_id,)).fetchone()
        assert row is not None
        return _skill_draft_from_row(row)

    def _append_skill_event(
        self,
        connection: sqlite3.Connection,
        *,
        task: TaskState,
        event_type: str,
        source: str,
        payload: dict,
        trace_type: str,
        trace_payload: dict,
        timestamp: str,
    ) -> TaskState:
        sequence = int(
            connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM task_events WHERE task_id = ?", (task.id,)
            ).fetchone()[0]
        )
        next_version = task.version + 1
        cursor = connection.execute(
            "UPDATE tasks SET version = ?, updated_at = ? WHERE id = ? AND version = ?",
            (next_version, timestamp, task.id, task.version),
        )
        if cursor.rowcount != 1:
            raise TaskVersionConflict(f"Task '{task.id}' was updated concurrently.")
        event_id = str(uuid4())
        connection.execute(
            """
            INSERT INTO task_events (
                id, task_id, session_id, event_type, source, payload_json,
                correlation_id, causation_id, sequence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)
            """,
            (event_id, task.id, task.session_id, event_type, source, _json_dumps(payload), task.id, sequence, timestamp),
        )
        connection.execute(
            """
            INSERT INTO task_traces (task_id, session_id, trace_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (task.id, task.session_id, trace_type, _json_dumps({**trace_payload, "event_id": event_id, "version": next_version}), timestamp),
        )
        updated_row = connection.execute(_TASK_SELECT + " WHERE id = ? LIMIT 1", (task.id,)).fetchone()
        assert updated_row is not None
        return _task_from_row(updated_row)

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

        return _session_context_from_row(row)

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


_SKILL_ACTIVATION_SELECT = """
SELECT id, task_id, skill_name, scope, source_path, content_hash, version,
       activation_reason, state, activated_at
FROM task_skill_activations
"""


_SKILL_DRAFT_SELECT = """
SELECT id, session_id, scope, skill_name, content, content_hash,
       status, created_at, saved_at, saved_path
FROM skill_drafts
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


def _session_context_from_row(row) -> SessionContext:
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


def _skill_activation_from_row(row) -> SkillActivation:
    return SkillActivation(
        task_id=row[1],
        skill_name=row[2],
        scope=row[3],
        source_path=row[4],
        content_hash=row[5],
        version=row[6],
        activation_reason=row[7],
        state=row[8],
        activated_at=row[9],
    )


def _skill_draft_from_row(row) -> SkillDraft:
    return SkillDraft(
        id=row[0],
        session_id=row[1],
        scope=row[2],
        skill_name=row[3],
        content=row[4],
        content_hash=row[5],
        status=row[6],
        created_at=row[7],
        saved_at=row[8],
        saved_path=row[9],
    )


def _compact_handoff_summary(
    summary_text: str | None,
    *,
    source_task: TaskState,
    evidence_refs: tuple[str, ...],
) -> str:
    source_summary = " ".join((summary_text or source_task.reflection or "").split())[:3000]
    lines = [
        f"Handoff from task {source_task.id}.",
        f"Goal: {source_task.goal}",
    ]
    if source_summary:
        lines.append(f"Previous compact summary: {source_summary}")
    if evidence_refs:
        lines.append("Evidence references: " + ", ".join(evidence_refs[:8]))
    return "\n".join(lines)


def _unique_active_skills(activations: tuple[SkillActivation, ...]) -> tuple[SkillActivation, ...]:
    selected: dict[str, SkillActivation] = {}
    for activation in activations:
        if activation.state == "active":
            selected[activation.skill_name] = activation
    return tuple(selected[name] for name in sorted(selected))


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

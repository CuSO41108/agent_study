from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from agent_app.types import Message, SessionContext, StoredMessage, SubagentRun, TodoItem, ToolCallTrace, ToolResult, TurnTrace


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
            connection.execute(
                """
                INSERT INTO tool_runs (
                    session_id, tool_call_id, tool_name, success, content, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    tool_result.tool_call_id,
                    tool_result.tool_name,
                    1 if tool_result.success else 0,
                    tool_result.content,
                    tool_result.error,
                    timestamp,
                ),
            )
            connection.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (timestamp, session_id),
            )

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

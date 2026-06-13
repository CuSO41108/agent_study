from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT,
        tool_call_id TEXT,
        tool_name TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        action_id TEXT,
        tool_call_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        success INTEGER NOT NULL,
        content TEXT NOT NULL,
        error TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_actions (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        task_id TEXT,
        agent_id TEXT NOT NULL,
        tool_call_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        arguments_json TEXT NOT NULL,
        idempotency_key TEXT NOT NULL UNIQUE,
        status TEXT NOT NULL,
        recovery_json TEXT NOT NULL,
        result_success INTEGER,
        result_content TEXT,
        result_error TEXT,
        prepared_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT,
        attempt INTEGER NOT NULL DEFAULT 1,
        retry_of TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        parent_task_id TEXT,
        goal TEXT NOT NULL,
        status TEXT NOT NULL,
        step INTEGER NOT NULL,
        plan_json TEXT NOT NULL,
        working_memory_json TEXT NOT NULL,
        pending_action_json TEXT,
        last_observation_json TEXT,
        reflection TEXT,
        budget_json TEXT NOT NULL,
        stop_reason TEXT,
        version INTEGER NOT NULL,
        waiting_deadline TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id),
        FOREIGN KEY (parent_task_id) REFERENCES tasks(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_events (
        id TEXT PRIMARY KEY,
        task_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        source TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        correlation_id TEXT,
        causation_id TEXT,
        sequence INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (task_id) REFERENCES tasks(id),
        FOREIGN KEY (session_id) REFERENCES sessions(id),
        UNIQUE(task_id, sequence)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS task_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        session_id TEXT NOT NULL,
        trace_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (task_id) REFERENCES tasks(id),
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_context (
        session_id TEXT PRIMARY KEY,
        summary_text TEXT,
        summary_message_id INTEGER,
        todo_json TEXT,
        updated_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS turn_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        user_input TEXT NOT NULL,
        context_message_count INTEGER NOT NULL,
        context_token_estimate INTEGER NOT NULL,
        used_summary INTEGER NOT NULL,
        used_todo INTEGER NOT NULL,
        used_evidence INTEGER NOT NULL,
        final_text TEXT,
        stop_reason TEXT,
        success INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (session_id) REFERENCES sessions(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS tool_call_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_trace_id INTEGER NOT NULL,
        tool_call_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        success INTEGER NOT NULL,
        error TEXT,
        content_preview TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (turn_trace_id) REFERENCES turn_traces(id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS subagent_runs (
        parent_session_id TEXT NOT NULL,
        parent_tool_call_id TEXT NOT NULL,
        child_session_id TEXT NOT NULL,
        agent_id TEXT NOT NULL,
        task TEXT NOT NULL,
        success INTEGER NOT NULL,
        result_summary TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY (parent_session_id) REFERENCES sessions(id),
        FOREIGN KEY (child_session_id) REFERENCES sessions(id)
    )
    """,
)


def initialize_database(db_path: str | Path) -> None:
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_file)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        _ensure_tool_runs_action_id(connection)
        _ensure_tool_action_columns(connection)
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_runs_action_id
            ON tool_runs(action_id)
            WHERE action_id IS NOT NULL
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tool_actions_session_status
            ON tool_actions(session_id, status)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_tasks_session_updated
            ON tasks(session_id, updated_at)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_events_task_sequence
            ON task_events(task_id, sequence)
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_task_traces_task_id
            ON task_traces(task_id, id)
            """
        )
        connection.commit()
    finally:
        connection.close()


def _ensure_tool_runs_action_id(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(tool_runs)").fetchall()
    }
    if "action_id" not in columns:
        connection.execute("ALTER TABLE tool_runs ADD COLUMN action_id TEXT")


def _ensure_tool_action_columns(connection: sqlite3.Connection) -> None:
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(tool_actions)").fetchall()
    }
    additions = {
        "task_id": "TEXT",
        "attempt": "INTEGER NOT NULL DEFAULT 1",
        "retry_of": "TEXT",
    }
    for name, declaration in additions.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE tool_actions ADD COLUMN {name} {declaration}")

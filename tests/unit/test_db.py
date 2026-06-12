from __future__ import annotations

import shutil
import sqlite3
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.state.db import initialize_database


class DatabaseTests(unittest.TestCase):
    def test_initialize_database_creates_required_tables(self) -> None:
        temp_root = Path(__file__).resolve().parents[2] / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        temp_dir = temp_root / f"db_{uuid4().hex}"
        temp_dir.mkdir()
        try:
            db_path = temp_dir / ".agent_app" / "agent.db"
            initialize_database(db_path)

            connection = sqlite3.connect(db_path)
            try:
                rows = connection.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'table'
                      AND name IN (
                          'sessions',
                          'messages',
                          'tool_actions',
                          'tool_runs',
                          'session_context',
                          'turn_traces',
                          'tool_call_traces',
                          'subagent_runs'
                      )
                    ORDER BY name
                    """
                ).fetchall()
            finally:
                connection.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertEqual(
            [row[0] for row in rows],
            [
                "messages",
                "session_context",
                "sessions",
                "subagent_runs",
                "tool_actions",
                "tool_call_traces",
                "tool_runs",
                "turn_traces",
            ],
        )

    def test_initialize_database_migrates_existing_tool_runs_table(self) -> None:
        temp_root = Path(__file__).resolve().parents[2] / ".test_tmp"
        temp_root.mkdir(exist_ok=True)
        temp_dir = temp_root / f"db_migration_{uuid4().hex}"
        temp_dir.mkdir()
        try:
            db_path = temp_dir / ".agent_app" / "agent.db"
            db_path.parent.mkdir()
            connection = sqlite3.connect(db_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE tool_runs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        tool_call_id TEXT NOT NULL,
                        tool_name TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        content TEXT NOT NULL,
                        error TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                connection.commit()
            finally:
                connection.close()

            initialize_database(db_path)

            connection = sqlite3.connect(db_path)
            try:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(tool_runs)").fetchall()
                }
            finally:
                connection.close()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertIn("action_id", columns)


if __name__ == "__main__":
    unittest.main()

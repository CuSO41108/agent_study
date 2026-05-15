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
                "tool_call_traces",
                "tool_runs",
                "turn_traces",
            ],
        )


if __name__ == "__main__":
    unittest.main()

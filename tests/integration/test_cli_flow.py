from __future__ import annotations

import io
import json
import os
import shutil
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from agent_app import cli
from agent_app.agent.definition import SINGLE_MAIN_AGENT
from agent_app.orchestrator.loop import AgentLoop
from agent_app.runtime.task_runtime import TaskRuntime
from agent_app.state.db import initialize_database
from agent_app.state.session_service import SessionService
from agent_app.tools.registry import build_default_registry
from agent_app.types import AgentEvent, ModelResponse, PendingAction, ToolCall


class _FakeModelClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def generate(self, *, system_prompt, messages, tools):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "messages": messages,
                "tools": tools,
            }
        )
        return self._responses.pop(0)


class CliIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"cli_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        src_dir = self.workspace_root / "src"
        src_dir.mkdir()
        (self.workspace_root / "README.md").write_text("hello from cli\n", encoding="utf-8")
        (src_dir / "module.py").write_text("print('old')\n", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_renders_and_exports_task_trace(self, mock_from_config) -> None:
        mock_from_config.return_value = _FakeModelClient([])
        database_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(database_path)
        sessions = SessionService(database_path)
        session_id = sessions.create_session("trace-session")
        task = sessions.create_task(session_id, goal="trace this task")
        sessions.append_task_trace(task.id, "tool_attempt", {"tool": "file_read", "success": True, "duration_ms": 8})

        timeline_stdout = io.StringIO()
        with redirect_stdout(timeline_stdout):
            timeline_code = cli.main(["--task-trace", task.id, "--workspace-root", str(self.workspace_root)])
        json_stdout = io.StringIO()
        with redirect_stdout(json_stdout):
            json_code = cli.main(["--task-trace-json", task.id, "--workspace-root", str(self.workspace_root)])

        self.assertEqual(timeline_code, 0)
        self.assertIn("Trace:", timeline_stdout.getvalue())
        self.assertIn("file_read / success / 8 ms", timeline_stdout.getvalue())
        self.assertEqual(json_code, 0)
        exported = json.loads(json_stdout.getvalue())
        self.assertEqual(exported["trace_id"], task.id)
        self.assertEqual(exported["schema_version"], 1)

    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_runs_full_turn_and_persists_session_data(self, mock_from_config) -> None:
        mock_from_config.return_value = _FakeModelClient(
            [
                _tool_call_response([ToolCall(id="call-1", name="file_read", arguments={"path": "README.md"})]),
                _text_response("CLI done"),
            ]
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "read the file",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue().strip())
        self.assertEqual(output["final_text"], "CLI done")
        self.assertTrue(output["success"])

        database_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(database_path)
        sessions = SessionService(database_path)
        recent_messages = sessions.list_recent_messages(output["session_id"])
        tool_runs = sessions.list_tool_runs(output["session_id"])

        self.assertEqual([message.role for message in recent_messages], ["user", "assistant"])
        self.assertEqual(len(tool_runs), 1)
        self.assertEqual(tool_runs[0].tool_name, "file_read")

    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_delegates_to_worker_and_keeps_turn_result_shape(self, mock_from_config) -> None:
        fake_model = _FakeModelClient(
            [
                _tool_call_response(
                    [
                        ToolCall(
                            id="call-1",
                            name="delegate_task",
                            arguments={
                                "task": "Inspect README.md",
                                "success_criteria": "Summarize the file contents.",
                                "relevant_paths": ["README.md"],
                            },
                        )
                    ]
                ),
                _text_response("README summary"),
                _text_response("CLI delegation done"),
            ]
        )
        mock_from_config.return_value = fake_model

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "use a worker to inspect the readme",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue().strip())
        self.assertEqual(
            sorted(output.keys()),
            [
                "final_text",
                "pending_action",
                "session_id",
                "stop_reason",
                "success",
                "task_id",
                "task_status",
                "tool_runs",
            ],
        )
        self.assertEqual(output["final_text"], "CLI delegation done")
        self.assertTrue(output["success"])
        self.assertEqual(output["tool_runs"][0]["tool_name"], "delegate_task")
        self.assertIn("child_session_id=", output["tool_runs"][0]["content"])

        sessions = SessionService(self.workspace_root / ".agent_app" / "agent.db")
        subagent_runs = sessions.list_subagent_runs(output["session_id"])
        self.assertEqual(len(subagent_runs), 1)
        self.assertEqual(subagent_runs[0].agent_id, "worker_agent")
        self.assertTrue(subagent_runs[0].success)

    @patch("builtins.input", return_value="y")
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_confirms_file_write_and_updates_file(self, mock_from_config, _mock_input) -> None:
        mock_from_config.return_value = _FakeModelClient(
            [
                _tool_call_response([ToolCall(id="call-1", name="file_write", arguments={"path": "src/module.py", "content": "print('new')\n"})]),
                _text_response("updated"),
            ]
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "update the file",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertTrue(output["success"])
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")

    @patch("builtins.input", return_value="y")
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_confirms_replace_in_file_and_updates_file(self, mock_from_config, _mock_input) -> None:
        mock_from_config.return_value = _FakeModelClient(
            [
                _tool_call_response([ToolCall(id="call-1", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
                _text_response("updated"),
            ]
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "update the file",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 0)
        output = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertTrue(output["success"])
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")

    @patch("builtins.input", return_value="n")
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_rejects_replace_in_file_when_user_declines(self, mock_from_config, _mock_input) -> None:
        mock_from_config.return_value = _FakeModelClient(
            [
                _tool_call_response([ToolCall(id="call-1", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
                _tool_call_response([ToolCall(id="call-2", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
            ]
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "update the file",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 1)
        output = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertFalse(output["success"])
        self.assertEqual(output["stop_reason"], "repeated_tool_failure")
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('old')\n")

    @patch("builtins.input", return_value="n")
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_rejects_file_write_when_user_declines(self, mock_from_config, _mock_input) -> None:
        mock_from_config.return_value = _FakeModelClient(
            [
                _tool_call_response([ToolCall(id="call-1", name="file_write", arguments={"path": "src/module.py", "content": "print('new')\n"})]),
                _tool_call_response([ToolCall(id="call-2", name="file_write", arguments={"path": "src/module.py", "content": "print('new')\n"})]),
            ]
        )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "update the file",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 1)
        output = json.loads(stdout.getvalue().strip().splitlines()[-1])
        self.assertFalse(output["success"])
        self.assertEqual(output["stop_reason"], "repeated_tool_failure")
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('old')\n")

    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_reuses_last_session_when_no_session_id_is_supplied(self, mock_from_config) -> None:
        fake_model = _FakeModelClient(
            [
                _text_response("first turn"),
                _text_response("second turn"),
            ]
        )
        mock_from_config.return_value = fake_model

        first_stdout = io.StringIO()
        with redirect_stdout(first_stdout):
            first_exit_code = cli.main([
                "当前如果你要产生文件，工作区在哪里",
                "--workspace-root",
                str(self.workspace_root),
            ])

        second_stdout = io.StringIO()
        with redirect_stdout(second_stdout):
            second_exit_code = cli.main([
                "我上一句话问了你什么，你记得吗",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(first_exit_code, 0)
        self.assertEqual(second_exit_code, 0)

        first_output = json.loads(first_stdout.getvalue().strip())
        second_output = json.loads(second_stdout.getvalue().strip())
        self.assertEqual(first_output["session_id"], second_output["session_id"])

        second_call_messages = fake_model.calls[1]["messages"]
        self.assertEqual(
            [message["role"] for message in second_call_messages],
            ["user", "assistant", "user"],
        )
        self.assertEqual(
            second_call_messages[0]["content"],
            "当前如果你要产生文件，工作区在哪里",
        )
        self.assertEqual(second_call_messages[1]["content"], "first turn")
        self.assertEqual(
            second_call_messages[2]["content"],
            "我上一句话问了你什么，你记得吗",
        )

        session_state_path = self.workspace_root / ".agent_app" / "current_session.txt"
        self.assertEqual(
            session_state_path.read_text(encoding="utf-8"),
            first_output["session_id"],
        )

    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_starts_new_session_when_requested(self, mock_from_config) -> None:
        fake_model = _FakeModelClient(
            [
                _text_response("first turn"),
                _text_response("second turn"),
            ]
        )
        mock_from_config.return_value = fake_model

        first_stdout = io.StringIO()
        with redirect_stdout(first_stdout):
            first_exit_code = cli.main([
                "first prompt",
                "--workspace-root",
                str(self.workspace_root),
            ])

        second_stdout = io.StringIO()
        with redirect_stdout(second_stdout):
            second_exit_code = cli.main([
                "second prompt",
                "--workspace-root",
                str(self.workspace_root),
                "--new-session",
            ])

        self.assertEqual(first_exit_code, 0)
        self.assertEqual(second_exit_code, 0)

        first_output = json.loads(first_stdout.getvalue().strip())
        second_output = json.loads(second_stdout.getvalue().strip())
        self.assertNotEqual(first_output["session_id"], second_output["session_id"])

        second_call_messages = fake_model.calls[1]["messages"]
        self.assertEqual(second_call_messages, [{"role": "user", "content": "second prompt"}])

        session_state_path = self.workspace_root / ".agent_app" / "current_session.txt"
        self.assertEqual(
            session_state_path.read_text(encoding="utf-8"),
            second_output["session_id"],
        )

    def test_cli_rejects_prompt_and_interactive_together(self) -> None:
        stderr = io.StringIO()

        with patch("sys.stderr", stderr):
            exit_code = cli.main([
                "hello",
                "--interactive",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 2)
        self.assertIn("prompt cannot be used with --interactive", stderr.getvalue())

    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_approves_waiting_file_action_after_process_restart(self, mock_from_config) -> None:
        database_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(database_path)
        sessions = SessionService(database_path)
        first_loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=_FakeModelClient([
                _tool_call_response([
                    ToolCall(
                        id="call-persisted",
                        name="file_write",
                        arguments={"path": "src/module.py", "content": "print('new')\n"},
                    )
                ])
            ]),
            tool_registry=build_default_registry(),
            session_service=sessions,
            workspace_root=self.workspace_root,
        )
        waiting = first_loop.run_turn(user_input="update the file")
        self.assertEqual(waiting.task_status, "waiting_user")

        mock_from_config.return_value = _FakeModelClient([_text_response("approved and complete")])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--approve-task",
                waiting.task_id,
                "--workspace-root",
                str(self.workspace_root),
            ])

        output = json.loads(stdout.getvalue().strip())
        self.assertEqual(exit_code, 0)
        self.assertTrue(output["success"])
        self.assertEqual(output["task_status"], "completed")
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")
        events = sessions.list_task_events(waiting.task_id)
        self.assertIn("user_approved", [event.type for event in events])

    def test_cross_process_approval_rejects_changed_file_version(self) -> None:
        database_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(database_path)
        sessions = SessionService(database_path)
        first_loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=_FakeModelClient([
                _tool_call_response([
                    ToolCall(
                        id="call-stale",
                        name="file_write",
                        arguments={"path": "src/module.py", "content": "print('new')\n"},
                    )
                ])
            ]),
            tool_registry=build_default_registry(),
            session_service=sessions,
            workspace_root=self.workspace_root,
        )
        waiting = first_loop.run_turn(user_input="update the file")
        (self.workspace_root / "src" / "module.py").write_text("print('other')\n", encoding="utf-8")

        restarted = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=_FakeModelClient([_text_response("approval became stale")]),
            tool_registry=build_default_registry(),
            session_service=SessionService(database_path),
            workspace_root=self.workspace_root,
        )
        result = restarted.handle_event(
            AgentEvent(
                id=str(uuid4()),
                task_id=waiting.task_id,
                session_id=waiting.session_id,
                type="user_approved",
                source="test",
                expected_version=sessions.get_task(waiting.task_id).version,
            )
        )

        self.assertTrue(result.success)
        self.assertEqual(result.tool_runs[0].observation.error_type, "conflict")
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('other')\n")

    @patch("builtins.input", return_value="quit")
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_cli_without_prompt_defaults_to_interactive_mode(self, mock_from_config, _mock_input) -> None:
        mock_from_config.return_value = _FakeModelClient([])
        stdout = io.StringIO()

        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--workspace-root",
                str(self.workspace_root),
            ])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("Agent Study", output)
        self.assertIn("Interactive REPL", output)
        self.assertIn("Type / to browse commands", output)
        session_id = (self.workspace_root / ".agent_app" / "current_session.txt").read_text(encoding="utf-8").strip()
        self.assertIn(session_id, output)

    def test_slash_command_completer_lists_and_filters_repl_commands(self) -> None:
        completer = cli._SlashCommandCompleter()
        event = CompleteEvent(completion_requested=False)

        all_commands = [
            completion.text
            for completion in completer.get_completions(Document("/"), event)
        ]
        trace_commands = [
            completion.text
            for completion in completer.get_completions(Document("/tr"), event)
        ]
        argument_commands = list(completer.get_completions(Document("/trace abc"), event))

        self.assertIn("/task", all_commands)
        self.assertIn("/trace", all_commands)
        self.assertEqual(trace_commands, ["/trace"])
        self.assertEqual(argument_commands, [])

    @patch("builtins.input", side_effect=["", "hello", "quit"])
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_interactive_mode_ignores_empty_input_and_prints_text_responses(self, mock_from_config, _mock_input) -> None:
        fake_model = _FakeModelClient([_text_response("interactive turn")])
        mock_from_config.return_value = fake_model

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--interactive",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 0)
        self.assertIn("Interactive REPL", stdout.getvalue())
        self.assertIn("interactive turn", stdout.getvalue())
        self.assertIn("[task:", stdout.getvalue())
        self.assertIn("| completed]", stdout.getvalue())
        self.assertEqual(len(fake_model.calls), 1)

    @patch("builtins.input", side_effect=["hello", "/task", "/tasks", "/help", "/unknown", "quit"])
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_interactive_mode_supports_task_discovery_and_help_commands(self, mock_from_config, _mock_input) -> None:
        mock_from_config.return_value = _FakeModelClient([_text_response("done")])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--workspace-root",
                str(self.workspace_root),
            ])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("done", output)
        self.assertGreaterEqual(output.count("[task:"), 3)
        self.assertIn("/approve", output)
        self.assertIn("Unknown command: /unknown", output)

    @patch("builtins.input")
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_interactive_approve_accepts_task_id_prefix(self, mock_from_config, mock_input) -> None:
        database_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(database_path)
        sessions = SessionService(database_path)
        first_loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=_FakeModelClient([
                _tool_call_response([
                    ToolCall(
                        id="call-repl-approval",
                        name="file_write",
                        arguments={"path": "src/module.py", "content": "print('new')\n"},
                    )
                ])
            ]),
            tool_registry=build_default_registry(),
            session_service=sessions,
            workspace_root=self.workspace_root,
        )
        waiting = first_loop.run_turn(user_input="update the file")
        self.assertEqual(waiting.task_status, "waiting_user")
        mock_from_config.return_value = _FakeModelClient([_text_response("approved")])
        mock_input.side_effect = [f"/approve {waiting.task_id[:8]}", "quit"]

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--workspace-root",
                str(self.workspace_root),
                "--session-id",
                waiting.session_id,
            ])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn(f"[task: {waiting.task_id} | completed]", output)
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")
        actions = sessions.list_tool_actions(waiting.session_id)
        self.assertEqual(len(actions), 1)
        approval_event = next(
            event
            for event in sessions.list_task_events(waiting.task_id)
            if event.type == "user_approved"
        )
        self.assertEqual(
            approval_event.payload["pending_action_id"],
            waiting.pending_action.id,
        )

    def test_repl_approval_rejects_ambiguous_or_unknown_task_prefix(self) -> None:
        database_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(database_path)
        sessions = SessionService(database_path)
        loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=_FakeModelClient([]),
            tool_registry=build_default_registry(),
            session_service=sessions,
            workspace_root=self.workspace_root,
        )
        session_id = sessions.create_session("candidate-session")
        task = sessions.create_task(session_id, goal="candidate")
        runtime = TaskRuntime(sessions)
        running = runtime.transition(
            task.id,
            target_status="running",
            event_type="user_message",
            source="test",
            reason="test",
        )
        waiting = runtime.wait_for_user(
            running.id,
            PendingAction(kind="tool_approval", prompt="Approve?"),
        )
        candidates = [
            replace(waiting, id="abc-first"),
            replace(waiting, id="abc-second"),
        ]

        stdout = io.StringIO()
        with redirect_stdout(stdout), patch.object(sessions, "list_tasks", return_value=candidates):
            ambiguous = cli._run_repl_task_control(
                loop=loop,
                session_service=sessions,
                session_id=session_id,
                command="/approve",
                task_prefix="abc",
            )
            missing = cli._run_repl_task_control(
                loop=loop,
                session_service=sessions,
                session_id=session_id,
                command="/approve",
                task_prefix="missing",
            )

        self.assertIsNone(ambiguous)
        self.assertIsNone(missing)
        self.assertIn("is ambiguous", stdout.getvalue())
        self.assertIn("No task matches prefix 'missing'", stdout.getvalue())

    @patch("builtins.input", side_effect=["/reject", "quit"])
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_interactive_reject_uses_latest_waiting_task_without_id(self, mock_from_config, _mock_input) -> None:
        database_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(database_path)
        sessions = SessionService(database_path)
        first_loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=_FakeModelClient([
                _tool_call_response([
                    ToolCall(
                        id="call-repl-reject",
                        name="file_write",
                        arguments={"path": "src/module.py", "content": "print('new')\n"},
                    )
                ])
            ]),
            tool_registry=build_default_registry(),
            session_service=sessions,
            workspace_root=self.workspace_root,
        )
        waiting = first_loop.run_turn(user_input="update the file")
        mock_from_config.return_value = _FakeModelClient([_text_response("rejected")])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--workspace-root",
                str(self.workspace_root),
                "--session-id",
                waiting.session_id,
            ])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("rejected", output)
        self.assertIn(f"[task: {waiting.task_id} | completed]", output)
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('old')\n")

    @patch("builtins.input", side_effect=["/cancel", "quit"])
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_interactive_cancel_uses_latest_non_terminal_task_without_id(self, mock_from_config, _mock_input) -> None:
        database_path = self.workspace_root / ".agent_app" / "agent.db"
        initialize_database(database_path)
        sessions = SessionService(database_path)
        first_loop = AgentLoop(
            agent=SINGLE_MAIN_AGENT,
            model_client=_FakeModelClient([
                _tool_call_response([
                    ToolCall(
                        id="call-repl-cancel",
                        name="file_write",
                        arguments={"path": "src/module.py", "content": "print('new')\n"},
                    )
                ])
            ]),
            tool_registry=build_default_registry(),
            session_service=sessions,
            workspace_root=self.workspace_root,
        )
        waiting = first_loop.run_turn(user_input="update the file")
        mock_from_config.return_value = _FakeModelClient([])

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--workspace-root",
                str(self.workspace_root),
                "--session-id",
                waiting.session_id,
            ])

        output = stdout.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn(f"[task: {waiting.task_id} | cancelled]", output)
        self.assertEqual(sessions.get_task(waiting.task_id).status, "cancelled")
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('old')\n")

    @patch("builtins.input", side_effect=["first prompt", ":new", "second prompt", "quit"])
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_interactive_mode_starts_new_session_with_colon_new(self, mock_from_config, _mock_input) -> None:
        fake_model = _FakeModelClient(
            [
                _text_response("first turn"),
                _text_response("second turn"),
            ]
        )
        mock_from_config.return_value = fake_model

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--interactive",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 0)
        self.assertIn("Started a new session.", stdout.getvalue())
        self.assertEqual(fake_model.calls[1]["messages"], [{"role": "user", "content": "second prompt"}])

    @patch("builtins.input", side_effect=["update the file", "y", "quit"])
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_interactive_mode_confirms_replace_in_file_in_same_process(self, mock_from_config, _mock_input) -> None:
        fake_model = _FakeModelClient(
            [
                _tool_call_response([ToolCall(id="call-1", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
                _text_response("updated"),
            ]
        )
        mock_from_config.return_value = fake_model

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--interactive",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 0)
        self.assertIn("Text edit confirmation", stdout.getvalue())
        self.assertIn("Operation: replace", stdout.getvalue())
        self.assertIn("updated", stdout.getvalue())
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")

        database_path = self.workspace_root / ".agent_app" / "agent.db"
        sessions = SessionService(database_path)
        session_id = (self.workspace_root / ".agent_app" / "current_session.txt").read_text(encoding="utf-8").strip()
        recent_messages = sessions.list_recent_messages(session_id)
        self.assertEqual([message.role for message in recent_messages], ["user", "assistant"])
        self.assertEqual(recent_messages[0].content, "update the file")

    @patch("builtins.input", side_effect=["use a worker", "y", "quit"])
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_interactive_mode_confirms_child_edit_during_delegation(self, mock_from_config, _mock_input) -> None:
        fake_model = _FakeModelClient(
            [
                _tool_call_response(
                    [
                        ToolCall(
                            id="call-1",
                            name="delegate_task",
                            arguments={
                                "task": "Update src/module.py",
                                "success_criteria": "Replace old with new and confirm the edit.",
                                "relevant_paths": ["src/module.py"],
                            },
                        )
                    ]
                ),
                _tool_call_response([ToolCall(id="child-1", name="replace_in_file", arguments={"path": "src/module.py", "old_text": "old", "new_text": "new"})]),
                _text_response("child updated"),
                _text_response("parent updated"),
            ]
        )
        mock_from_config.return_value = fake_model

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--interactive",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 0)
        self.assertIn("Text edit confirmation", stdout.getvalue())
        self.assertIn("Operation: replace", stdout.getvalue())
        self.assertIn("parent updated", stdout.getvalue())
        self.assertEqual((self.workspace_root / "src" / "module.py").read_text(encoding="utf-8"), "print('new')\n")

    @patch("builtins.input", side_effect=["please preview the edit", "y", "quit"])
    @patch("agent_app.cli.OpenAICompatibleModelClient.from_config")
    def test_interactive_mode_keeps_session_alive_for_natural_language_confirmation(self, mock_from_config, _mock_input) -> None:
        fake_model = _FakeModelClient(
            [
                _text_response("I can preview that change. Confirm execution?"),
                _text_response("confirmed"),
            ]
        )
        mock_from_config.return_value = fake_model

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            exit_code = cli.main([
                "--interactive",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 0)
        self.assertIn("Confirm execution?", stdout.getvalue())
        self.assertIn("confirmed", stdout.getvalue())
        self.assertEqual(
            [message["role"] for message in fake_model.calls[1]["messages"]],
            ["user", "assistant", "user"],
        )
        self.assertEqual(fake_model.calls[1]["messages"][2]["content"], "y")

    @patch.dict(os.environ, {"MODEL_TIMEOUT": "0"}, clear=False)
    def test_cli_returns_configuration_error_for_invalid_timeout(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), patch("sys.stderr", stderr):
            exit_code = cli.main([
                "hello",
                "--workspace-root",
                str(self.workspace_root),
            ])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Configuration error: MODEL_TIMEOUT must be a positive number.", stderr.getvalue().strip())



def _text_response(text: str) -> ModelResponse:
    return ModelResponse(
        assistant_text=text,
        tool_calls=[],
        finish_reason="stop",
        raw_response={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]},
        error_type=None,
    )



def _tool_call_response(tool_calls: list[ToolCall]) -> ModelResponse:
    return ModelResponse(
        assistant_text=None,
        tool_calls=tool_calls,
        finish_reason="tool_calls",
        raw_response={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_call.name,
                                    "arguments": json.dumps(tool_call.arguments),
                                },
                            }
                            for tool_call in tool_calls
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
        error_type=None,
    )


if __name__ == "__main__":
    unittest.main()

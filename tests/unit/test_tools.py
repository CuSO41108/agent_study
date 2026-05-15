from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from agent_app.tools.approval import approve_tool_call, validate_shell_command
from agent_app.tools.base import ToolExecutionContext
from agent_app.tools.code_search import CodeSearchTool
from agent_app.tools.delegate_task import DelegateTaskTool
from agent_app.tools.file_read import FileReadTool, MAX_FILE_READ_LINES
from agent_app.tools.file_write import (
    MAX_FILE_WRITE_BYTES,
    MAX_FILE_WRITE_LINES,
    _SIZE_LIMIT_ERROR,
    FileWriteTool,
    inspect_file_write_request,
)
from agent_app.tools.replace_in_file import (
    AMBIGUOUS_MATCH_ERROR,
    FILE_CHANGED_ERROR,
    MAX_TEXT_EDIT_BYTES,
    MAX_TEXT_EDIT_LINES,
    NO_MATCH_ERROR,
    TEXT_EDIT_SIZE_LIMIT_ERROR,
    ReplaceInFileTool,
    inspect_replace_in_file_request,
)
from agent_app.tools.registry import build_default_registry, build_root_registry, build_worker_registry
from agent_app.tools.shell import ShellTool
from agent_app.tools.todo import TodoReadTool, TodoWriteTool
from agent_app.state.db import initialize_database
from agent_app.orchestrator.subagent_runner import SubagentRunner
from agent_app.state.session_service import SessionService
from agent_app.runtime.shell_runtime import RuntimeExecutionResult
from agent_app.types import ModelResponse


class _FakeModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    def generate(self, *, system_prompt, messages, tools):
        if not self._responses:
            raise AssertionError("Unexpected model call")
        return self._responses.pop(0)


class ToolLayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"tools_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        (self.workspace_root / "sample.txt").write_text("alpha\nbeta\nalpha beta\n", encoding="utf-8")
        nested = self.workspace_root / "src"
        nested.mkdir()
        (nested / "app.py").write_text("print('alpha')\n", encoding="utf-8")
        internal_dir = self.workspace_root / ".agent_app"
        internal_dir.mkdir()
        (internal_dir / "agent.db").write_text("alpha from db\n", encoding="utf-8")
        self.db_path = internal_dir / "session.db"
        initialize_database(self.db_path)
        self.sessions = SessionService(self.db_path)
        self.session_id = self.sessions.create_session("tool-session")
        self.context = ToolExecutionContext(
            workspace_root=self.workspace_root,
            timeout=0.01,
            session_id=self.session_id,
            session_service=self.sessions,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_registry_returns_specs_for_allowed_tools(self) -> None:
        registry = build_default_registry()

        specs = registry.get_specs(["file_read", "todo_read", "todo_write", "replace_in_file", "shell"])

        self.assertEqual([spec["function"]["name"] for spec in specs], ["file_read", "todo_read", "todo_write", "replace_in_file", "shell"])
        self.assertIn("todo list", specs[1]["function"]["description"])
        self.assertIn("replace exact text", specs[3]["function"]["description"])
        self.assertIn("PowerShell", specs[4]["function"]["description"])

    def test_registry_get_required_returns_tool_and_raises_for_unknown_name(self) -> None:
        registry = build_default_registry()

        self.assertEqual(registry.get_required("file_read").name, "file_read")
        with self.assertRaises(KeyError):
            registry.get_required("missing")

    def test_approval_auto_allows_file_read_and_safe_shell(self) -> None:
        self.assertEqual(approve_tool_call("file_read", {"path": "sample.txt"}).decision, "allow")
        self.assertEqual(approve_tool_call("shell", {"command": "git status --short"}).decision, "allow")
        self.assertEqual(approve_tool_call("todo_read", {}).decision, "allow")
        self.assertEqual(approve_tool_call("todo_write", {"items": []}).decision, "allow")
        self.assertEqual(
            approve_tool_call("delegate_task", {"task": "Inspect README", "success_criteria": "Summarize it"}).decision,
            "allow",
        )
        self.assertEqual(approve_tool_call("file_write", {"path": "sample.txt", "content": "x"}).decision, "confirm")
        self.assertEqual(
            approve_tool_call("replace_in_file", {"path": "sample.txt", "old_text": "a", "new_text": "b"}).decision,
            "confirm",
        )

    def test_root_registry_includes_delegate_task_and_worker_registry_does_not(self) -> None:
        runner = SubagentRunner(
            model_client=_FakeModelClient([]),
            session_service=self.sessions,
            workspace_root=self.workspace_root,
            tool_timeout=1.0,
            context_token_budget=6000,
            summary_trigger_tokens=3000,
        )

        root_registry = build_root_registry(subagent_runner=runner)
        worker_registry = build_worker_registry()

        self.assertIsInstance(root_registry.get_required("delegate_task"), DelegateTaskTool)
        self.assertIsNone(worker_registry.get("delegate_task"))

    def test_approval_rejects_unknown_tool_name(self) -> None:
        result = approve_tool_call("missing", {})

        self.assertEqual(result.decision, "deny")
        self.assertIn("not available", result.reason)

    def test_approval_rejects_non_whitelisted_shell_command(self) -> None:
        result = approve_tool_call("shell", {"command": "python -c \"print(1)\""})

        self.assertEqual(result.decision, "deny")
        self.assertIn("whitelist", result.reason)

    def test_shell_validation_rejects_empty_command(self) -> None:
        approved, reason = validate_shell_command("")

        self.assertFalse(approved)
        self.assertIn("non-empty string", reason)

    def test_shell_validation_explains_powershell_compatibility(self) -> None:
        approved, reason = validate_shell_command("dir /b src")

        self.assertFalse(approved)
        self.assertIn("PowerShell", reason)
        self.assertIn("Get-ChildItem", reason)

    def test_shell_validation_explains_unix_recursion_hint(self) -> None:
        approved, reason = validate_shell_command("ls -r src")

        self.assertFalse(approved)
        self.assertIn("Get-ChildItem -Recurse", reason)

    def test_file_read_reads_workspace_file(self) -> None:
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "sample.txt", "start_line": 2, "end_line": 3},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "2: beta\n3: alpha beta")

    def test_file_read_rejects_empty_path(self) -> None:
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": ""},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("non-empty string", result.error)

    def test_file_read_rejects_invalid_start_line(self) -> None:
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "sample.txt", "start_line": 0},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Invalid arguments: start_line must be greater than or equal to 1.")

    def test_file_read_rejects_invalid_end_line(self) -> None:
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "sample.txt", "start_line": 3, "end_line": 2},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("must be >= 'start_line'", result.error)

    def test_file_read_rejects_missing_file(self) -> None:
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "missing.txt"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("was not found", result.error)

    def test_file_read_limits_unbounded_reads_to_500_lines_with_footer(self) -> None:
        tool = FileReadTool()
        large_file = self.workspace_root / "large.txt"
        large_file.write_text("".join(f"line-{index}\n" for index in range(MAX_FILE_READ_LINES + 25)), encoding="utf-8")

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "large.txt"},
            context=self.context,
        )

        self.assertTrue(result.success)
        content_lines = result.content.splitlines()
        self.assertEqual(content_lines[0], "1: line-0")
        self.assertEqual(content_lines[MAX_FILE_READ_LINES - 1], "500: line-499")
        self.assertEqual(content_lines[-1], "--- Output truncated to 500 lines. Narrow the range and retry. ---")

    def test_file_read_rejects_explicit_ranges_over_limit(self) -> None:
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "sample.txt", "start_line": 1, "end_line": MAX_FILE_READ_LINES + 1},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Requested line range exceeds the maximum of 500 lines.")

    def test_file_read_start_line_past_eof_returns_empty_content(self) -> None:
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "sample.txt", "start_line": 99},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "")

    def test_file_read_end_line_past_eof_returns_available_content(self) -> None:
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "sample.txt", "start_line": 2, "end_line": 99},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "2: beta\n3: alpha beta")

    def test_file_read_empty_file_returns_success_with_empty_content(self) -> None:
        tool = FileReadTool()
        empty_file = self.workspace_root / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "empty.txt"},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "")

    @patch("agent_app.tools.file_read._deadline_exceeded", return_value=True)
    def test_file_read_returns_failure_when_timeout_is_exceeded(self, _mock_deadline) -> None:
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "sample.txt"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "File read timed out.")

    def test_file_read_rejects_path_outside_workspace(self) -> None:
        outside_file = self.workspace_root.parent / "outside.txt"
        outside_file.write_text("nope", encoding="utf-8")
        self.addCleanup(lambda: outside_file.unlink(missing_ok=True))
        tool = FileReadTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": str(outside_file)},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("escapes the workspace root", result.error)

    @patch("agent_app.tools.code_search.shutil.which", return_value=None)
    def test_code_search_finds_matches_with_python_fallback(self, _mock_which) -> None:
        tool = CodeSearchTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"pattern": "alpha", "path": ".", "max_results": 5},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertIn("sample.txt", result.content)
        self.assertIn("app.py", result.content)
        self.assertNotIn("agent.db", result.content)

    def test_code_search_rejects_invalid_pattern_and_max_results(self) -> None:
        tool = CodeSearchTool()

        invalid_pattern = tool.execute(
            tool_call_id="call-1",
            arguments={"pattern": "", "path": "."},
            context=self.context,
        )
        invalid_max_results = tool.execute(
            tool_call_id="call-2",
            arguments={"pattern": "alpha", "path": ".", "max_results": 0},
            context=self.context,
        )

        self.assertFalse(invalid_pattern.success)
        self.assertIn("non-empty string", invalid_pattern.error)
        self.assertFalse(invalid_max_results.success)
        self.assertEqual(invalid_max_results.error, "Invalid arguments: max_results must be greater than or equal to 1.")

    def test_tools_return_stable_invalid_argument_errors(self) -> None:
        file_read_result = FileReadTool().execute(
            tool_call_id="call-1",
            arguments={},
            context=self.context,
        )
        shell_result = ShellTool().execute(
            tool_call_id="call-2",
            arguments={"command": ""},
            context=self.context,
        )
        code_search_result = CodeSearchTool().execute(
            tool_call_id="call-3",
            arguments={"pattern": "alpha", "max_results": "bad"},
            context=self.context,
        )
        file_write_result = FileWriteTool().execute(
            tool_call_id="call-4",
            arguments={"path": "", "content": ""},
            context=self.context,
        )

        self.assertEqual(file_read_result.error, "Invalid arguments: path is required.")
        self.assertEqual(shell_result.error, "Invalid arguments: command must be a non-empty string.")
        self.assertEqual(code_search_result.error, "Invalid arguments: max_results must be an integer.")
        self.assertEqual(file_write_result.error, "Invalid arguments: path must be a non-empty string.")

    def test_tools_reject_unexpected_fields_with_stable_error(self) -> None:
        tools = (
            FileReadTool(),
            CodeSearchTool(),
            FileWriteTool(),
            ShellTool(),
        )

        for tool in tools:
            result = tool.execute(
                tool_call_id="call-extra",
                arguments={"foo": "bar"},
                context=self.context,
            )
            self.assertFalse(result.success)
            self.assertEqual(result.error, "Invalid arguments: unexpected field 'foo'.")

    def test_code_search_rejects_path_outside_workspace(self) -> None:
        tool = CodeSearchTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"pattern": "alpha", "path": "..\\outside", "max_results": 5},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("escapes the workspace root", result.error)

    @patch("agent_app.tools.code_search.shutil.which", return_value=None)
    def test_code_search_excludes_single_internal_file_path(self, _mock_which) -> None:
        tool = CodeSearchTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"pattern": "alpha", "path": ".agent_app/agent.db", "max_results": 5},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "No matches found.")

    @patch("agent_app.tools.code_search.shutil.which", return_value=None)
    def test_code_search_fallback_treats_invalid_regex_as_literal_text(self, _mock_which) -> None:
        tool = CodeSearchTool()
        (self.workspace_root / "pattern.txt").write_text("database_path(\n", encoding="utf-8")

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"pattern": "database_path(", "path": ".", "max_results": 5},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertIn("pattern.txt", result.content)

    @patch("agent_app.tools.code_search.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="rg", timeout=0.01))
    @patch("agent_app.tools.code_search.shutil.which", return_value="rg")
    def test_code_search_rg_timeout_returns_failure_result(self, _mock_which, _mock_run) -> None:
        tool = CodeSearchTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"pattern": "alpha", "path": ".", "max_results": 5},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Code search timed out.")

    @patch("agent_app.tools.code_search._deadline_exceeded", return_value=True)
    @patch("agent_app.tools.code_search.shutil.which", return_value=None)
    def test_code_search_fallback_timeout_returns_failure_result(self, _mock_which, _mock_deadline) -> None:
        tool = CodeSearchTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"pattern": "alpha", "path": ".", "max_results": 5},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Code search timed out.")

    def test_shell_rejects_redirects(self) -> None:
        tool = ShellTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"command": "git status > out.txt"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("disallowed", result.error)

    def test_shell_returns_timeout_error(self) -> None:
        def _timeout_runner(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd=kwargs.get("args", "powershell"), timeout=0.01)

        tool = ShellTool(runner=_timeout_runner)

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"command": "git status --short"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Shell command timed out.")

    def test_shell_preserves_leading_spaces_in_output(self) -> None:
        def _runner(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout=" M .gitignore\n?? demo/\n",
                stderr="",
            )

        tool = ShellTool(runner=_runner)

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"command": "git status --short"},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, " M .gitignore\n?? demo/")

    def test_shell_tool_uses_runtime_result(self) -> None:
        class _FakeRuntime:
            def __init__(self) -> None:
                self.calls = []

            def run(self, command, *, workspace_root, timeout):
                self.calls.append((command, workspace_root, timeout))
                return RuntimeExecutionResult(
                    success=True,
                    stdout="runtime\n",
                    stderr="",
                    combined_output="runtime",
                    exit_code=0,
                    error_type=None,
                )

        runtime = _FakeRuntime()
        tool = ShellTool(runtime=runtime)

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"command": "git status --short"},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "runtime")
        self.assertEqual(runtime.calls[0][0], "git status --short")
        self.assertEqual(runtime.calls[0][1], self.workspace_root)

    def test_shell_tool_rejects_runtime_and_runner_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "Only one of 'runtime' or 'runner'"):
            ShellTool(runtime=object(), runner=lambda *args, **kwargs: None)

    def test_shell_tool_surfaces_runtime_error(self) -> None:
        class _FakeRuntime:
            def run(self, command, *, workspace_root, timeout):
                return RuntimeExecutionResult(
                    success=False,
                    stdout="",
                    stderr="cannot start",
                    combined_output="cannot start",
                    exit_code=None,
                    error_type="runtime_error",
                )

        tool = ShellTool(runtime=_FakeRuntime())
        result = tool.execute(
            tool_call_id="call-1",
            arguments={"command": "git status --short"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, "Shell runtime failed.")
        self.assertEqual(result.content, "cannot start")

    def test_file_write_overwrites_small_text_file(self) -> None:
        tool = FileWriteTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "sample.txt", "content": "new text\n"},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertIn("Overwrite", result.content)
        self.assertEqual((self.workspace_root / "sample.txt").read_text(encoding="utf-8"), "new text\n")

    def test_file_write_creates_new_small_text_file(self) -> None:
        tool = FileWriteTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/new_module.py", "content": "print('ok')\n"},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertTrue((self.workspace_root / "src" / "new_module.py").is_file())

    def test_file_write_rejects_path_outside_workspace(self) -> None:
        tool = FileWriteTool()
        outside_file = self.workspace_root.parent / "outside.py"
        self.addCleanup(lambda: outside_file.unlink(missing_ok=True))

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": str(outside_file), "content": "print('nope')\n"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("escapes the workspace root", result.error)

    def test_file_write_rejects_missing_parent_directory(self) -> None:
        tool = FileWriteTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "missing/new_file.py", "content": "print('x')\n"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("Parent directory does not exist", result.error)

    def test_file_write_preview_uses_same_validation_error_as_execute(self) -> None:
        inspection, error = inspect_file_write_request(
            arguments={"path": "", "content": ""},
            context=self.context,
        )
        result = FileWriteTool().execute(
            tool_call_id="call-1",
            arguments={"path": "", "content": ""},
            context=self.context,
        )

        self.assertIsNone(inspection)
        self.assertEqual(error, "Invalid arguments: path must be a non-empty string.")
        self.assertEqual(result.error, error)

    def test_file_write_rejects_non_utf8_existing_file(self) -> None:
        target = self.workspace_root / "src" / "binary.py"
        target.write_bytes(b"\xff\xfe\x00\x00")
        tool = FileWriteTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/binary.py", "content": "print('x')\n"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("not valid UTF-8", result.error)

    def test_file_write_rejects_sensitive_env_files(self) -> None:
        tool = FileWriteTool()

        for path in (".env", ".env.local"):
            result = tool.execute(
                tool_call_id="call-1",
                arguments={"path": path, "content": "X=1\n"},
                context=self.context,
            )
            self.assertFalse(result.success)
            self.assertIn("Sensitive environment files", result.error)

    def test_file_write_rejects_internal_directories(self) -> None:
        tool = FileWriteTool()

        for path in (".git/config", ".agent_app/state.json"):
            result = tool.execute(
                tool_call_id="call-1",
                arguments={"path": path, "content": "value\n"},
                context=self.context,
            )
            self.assertFalse(result.success)
            self.assertIn("Internal workspace directories", result.error)

    def test_file_write_rejects_hidden_files(self) -> None:
        hidden_dir = self.workspace_root / ".hidden"
        hidden_dir.mkdir()
        tool = FileWriteTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": ".hidden/file.txt", "content": "value\n"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("Hidden files and directories", result.error)

    def test_file_write_rejects_binary_extensions(self) -> None:
        tool = FileWriteTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/image.png", "content": "not really png"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("Binary or non-text file types", result.error)

    def test_file_write_rejects_non_whitelisted_extensions(self) -> None:
        tool = FileWriteTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/Makefile", "content": "all:\n\techo ok\n"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("whitelisted text file extensions", result.error)

    def test_file_write_rejects_content_over_byte_limit(self) -> None:
        tool = FileWriteTool()
        too_large = "a" * (MAX_FILE_WRITE_BYTES + 1)

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/large.txt", "content": too_large},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, _SIZE_LIMIT_ERROR)

    def test_file_write_rejects_content_over_line_limit(self) -> None:
        tool = FileWriteTool()
        too_many_lines = "\n".join("line" for _ in range(MAX_FILE_WRITE_LINES + 1))

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/large.py", "content": too_many_lines},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, _SIZE_LIMIT_ERROR)

    def test_file_write_rejects_existing_file_over_size_limit(self) -> None:
        target = self.workspace_root / "src" / "large.txt"
        target.write_text("a" * (MAX_FILE_WRITE_BYTES + 1), encoding="utf-8")
        tool = FileWriteTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/large.txt", "content": "small\n"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, _SIZE_LIMIT_ERROR)

    def test_replace_in_file_replaces_single_match(self) -> None:
        target = self.workspace_root / "src" / "app.py"
        tool = ReplaceInFileTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/app.py", "old_text": "alpha", "new_text": "beta"},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertIn("Replaced 1 occurrence(s)", result.content)
        self.assertEqual(target.read_text(encoding="utf-8"), "print('beta')\n")

    def test_replace_in_file_allows_deleting_to_empty_file(self) -> None:
        target = self.workspace_root / "src" / "delete_me.txt"
        target.write_text("erase-me", encoding="utf-8")
        tool = ReplaceInFileTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/delete_me.txt", "old_text": "erase-me", "new_text": ""},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(target.read_text(encoding="utf-8"), "")

    def test_replace_in_file_returns_no_match_error(self) -> None:
        tool = ReplaceInFileTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/app.py", "old_text": "missing", "new_text": "new"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, NO_MATCH_ERROR)

    def test_replace_in_file_rejects_ambiguous_matches_by_default(self) -> None:
        target = self.workspace_root / "src" / "multi.txt"
        target.write_text("same\nsame\n", encoding="utf-8")
        tool = ReplaceInFileTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/multi.txt", "old_text": "same", "new_text": "new"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.error, AMBIGUOUS_MATCH_ERROR)

    def test_replace_in_file_replace_all_updates_all_matches(self) -> None:
        target = self.workspace_root / "src" / "multi.txt"
        target.write_text("same\nsame\n", encoding="utf-8")
        tool = ReplaceInFileTool()

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/multi.txt", "old_text": "same", "new_text": "new", "replace_all": True},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(target.read_text(encoding="utf-8"), "new\nnew\n")

    def test_replace_in_file_rejects_path_outside_workspace(self) -> None:
        tool = ReplaceInFileTool()
        outside_file = self.workspace_root.parent / "outside.py"
        outside_file.write_text("print('old')\n", encoding="utf-8")
        self.addCleanup(lambda: outside_file.unlink(missing_ok=True))

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": str(outside_file), "old_text": "old", "new_text": "new"},
            context=self.context,
        )

        self.assertFalse(result.success)
        self.assertIn("escapes the workspace root", result.error)

    def test_replace_in_file_rejects_sensitive_and_hidden_targets(self) -> None:
        hidden_dir = self.workspace_root / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "file.txt").write_text("hello", encoding="utf-8")
        internal_target = self.workspace_root / ".agent_app" / "state.txt"
        internal_target.write_text("hello", encoding="utf-8")
        tool = ReplaceInFileTool()

        hidden_result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": ".hidden/file.txt", "old_text": "hello", "new_text": "hi"},
            context=self.context,
        )
        internal_result = tool.execute(
            tool_call_id="call-2",
            arguments={"path": ".agent_app/state.txt", "old_text": "hello", "new_text": "hi"},
            context=self.context,
        )

        self.assertFalse(hidden_result.success)
        self.assertIn("Hidden files and directories", hidden_result.error)
        self.assertFalse(internal_result.success)
        self.assertIn("Internal workspace directories", internal_result.error)

    def test_replace_in_file_rejects_non_utf8_and_non_whitelisted_targets(self) -> None:
        binary_target = self.workspace_root / "src" / "binary.py"
        binary_target.write_bytes(b"\xff\xfe\x00\x00")
        makefile_target = self.workspace_root / "src" / "Makefile"
        makefile_target.write_text("all:\n", encoding="utf-8")
        tool = ReplaceInFileTool()

        binary_result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/binary.py", "old_text": "x", "new_text": "y"},
            context=self.context,
        )
        makefile_result = tool.execute(
            tool_call_id="call-2",
            arguments={"path": "src/Makefile", "old_text": "all", "new_text": "build"},
            context=self.context,
        )

        self.assertFalse(binary_result.success)
        self.assertIn("not valid UTF-8", binary_result.error)
        self.assertFalse(makefile_result.success)
        self.assertIn("whitelisted text file extensions", makefile_result.error)

    def test_replace_in_file_rejects_current_or_updated_content_over_limit(self) -> None:
        current_large = self.workspace_root / "src" / "too_large.txt"
        current_large.write_text("a" * (MAX_TEXT_EDIT_BYTES + 1), encoding="utf-8")
        lines_large = self.workspace_root / "src" / "line_large.txt"
        lines_large.write_text("\n".join("line" for _ in range(MAX_TEXT_EDIT_LINES + 1)), encoding="utf-8")
        grow_target = self.workspace_root / "src" / "grow.txt"
        grow_target.write_text("seed", encoding="utf-8")
        tool = ReplaceInFileTool()

        current_result = tool.execute(
            tool_call_id="call-1",
            arguments={"path": "src/too_large.txt", "old_text": "a", "new_text": "b"},
            context=self.context,
        )
        lines_result = tool.execute(
            tool_call_id="call-2",
            arguments={"path": "src/line_large.txt", "old_text": "line", "new_text": "row"},
            context=self.context,
        )
        updated_result = tool.execute(
            tool_call_id="call-3",
            arguments={"path": "src/grow.txt", "old_text": "seed", "new_text": "b" * (MAX_TEXT_EDIT_BYTES + 1)},
            context=self.context,
        )

        self.assertFalse(current_result.success)
        self.assertEqual(current_result.error, TEXT_EDIT_SIZE_LIMIT_ERROR)
        self.assertFalse(lines_result.success)
        self.assertEqual(lines_result.error, TEXT_EDIT_SIZE_LIMIT_ERROR)
        self.assertFalse(updated_result.success)
        self.assertEqual(updated_result.error, TEXT_EDIT_SIZE_LIMIT_ERROR)

    def test_replace_in_file_inspection_returns_diff_preview(self) -> None:
        inspection, error = inspect_replace_in_file_request(
            arguments={"path": "src/app.py", "old_text": "alpha", "new_text": "beta"},
            context=self.context,
        )

        self.assertIsNone(error)
        assert inspection is not None
        self.assertEqual(inspection.match_count, 1)
        self.assertEqual(inspection.replacement_count, 1)
        self.assertIn("--- a/src\\app.py", inspection.diff_preview)
        self.assertIn("+++ b/src\\app.py", inspection.diff_preview)
        self.assertIn("-print('alpha')", inspection.diff_preview)
        self.assertIn("+print('beta')", inspection.diff_preview)

    def test_replace_in_file_inspection_captures_original_content_for_revalidation(self) -> None:
        inspection, error = inspect_replace_in_file_request(
            arguments={"path": "src/app.py", "old_text": "alpha", "new_text": "beta"},
            context=self.context,
        )
        self.assertIsNone(error)
        assert inspection is not None
        self.assertEqual(inspection._expected_original_content, "print('alpha')\n")
        self.assertEqual(inspection._updated_content, "print('beta')\n")

    def test_todo_tools_round_trip(self) -> None:
        write_tool = TodoWriteTool()
        read_tool = TodoReadTool()

        write_result = write_tool.execute(
            tool_call_id="call-1",
            arguments={
                "items": [
                    {"content": "collect evidence", "status": "in_progress"},
                    {"content": "write answer", "status": "pending"},
                ]
            },
            context=self.context,
        )
        read_result = read_tool.execute(
            tool_call_id="call-2",
            arguments={},
            context=self.context,
        )

        self.assertTrue(write_result.success)
        self.assertEqual(
            read_result.content,
            "1. [in_progress] collect evidence\n2. [pending] write answer",
        )

    def test_todo_write_replaces_items_and_enforces_max_length(self) -> None:
        tool = TodoWriteTool()

        replace_result = tool.execute(
            tool_call_id="call-1",
            arguments={"items": [{"content": "one item", "status": "completed"}]},
            context=self.context,
        )
        too_many_result = tool.execute(
            tool_call_id="call-2",
            arguments={
                "items": [
                    {"content": f"task-{index}", "status": "pending"}
                    for index in range(21)
                ]
            },
            context=self.context,
        )

        self.assertTrue(replace_result.success)
        self.assertFalse(too_many_result.success)
        self.assertEqual(too_many_result.error, "Todo list cannot contain more than 20 items.")


if __name__ == "__main__":
    unittest.main()

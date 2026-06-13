from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from agent_app.tools._path_utils import PathSafetyError, resolve_workspace_path
from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.types import ToolResult

_EXCLUDED_DIRS = {".agent_app", ".git", "__pycache__"}
_EXCLUDED_SUFFIXES = {".db", ".sqlite", ".pyc"}


class CodeSearchTool(Tool):
    name = "code_search"
    description = "Search text inside files under the workspace."
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer", "minimum": 1},
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    def execute(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        validation_error = self.validate_arguments(arguments)
        if validation_error is not None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=validation_error)

        pattern = arguments.get("pattern")
        raw_path = arguments.get("path", ".")
        max_results = arguments.get("max_results", 20)
        try:
            search_root = resolve_workspace_path(context.workspace_root, raw_path)
        except PathSafetyError as exc:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=str(exc))

        ripgrep = shutil.which("rg")
        if ripgrep:
            return self._run_ripgrep(
                tool_call_id=tool_call_id,
                pattern=pattern,
                search_root=search_root,
                max_results=max_results,
                timeout=context.timeout,
            )
        return self._run_python_fallback(
            tool_call_id=tool_call_id,
            pattern=pattern,
            search_root=search_root,
            max_results=max_results,
            timeout=context.timeout,
        )

    def _run_ripgrep(
        self,
        *,
        tool_call_id: str,
        pattern: str,
        search_root: Path,
        max_results: int,
        timeout: float,
    ) -> ToolResult:
        command = [
            "rg",
            "--line-number",
            "--with-filename",
            "--color",
            "never",
            "--no-heading",
            "--glob",
            "!.agent_app/**",
            "--glob",
            "!.git/**",
            "--glob",
            "!__pycache__/**",
            "--glob",
            "!*.db",
            "--glob",
            "!*.sqlite",
            "--glob",
            "!*.pyc",
            pattern,
            str(search_root),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            content = _join_process_output(exc.stdout, exc.stderr)
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content=content, error="Code search timed out.")
        if completed.returncode not in (0, 1):
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content=stdout.strip(), error=stderr.strip() or "ripgrep failed")

        lines = [line for line in (completed.stdout or "").splitlines() if line][:max_results]
        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content="\n".join(lines) or "No matches found.", error=None)

    def _run_python_fallback(
        self,
        *,
        tool_call_id: str,
        pattern: str,
        search_root: Path,
        max_results: int,
        timeout: float,
    ) -> ToolResult:
        deadline = _build_deadline(timeout)
        regex, literal_pattern = _compile_search_pattern(pattern)
        matches: list[str] = []
        files: list[Path]
        if search_root.is_file():
            if _is_excluded_path(search_root) or search_root.suffix in _EXCLUDED_SUFFIXES:
                files = []
            else:
                files = [search_root]
        else:
            files = [
                path
                for path in search_root.rglob("*")
                if path.is_file()
                and not _is_excluded_path(path)
                and path.suffix not in _EXCLUDED_SUFFIXES
            ]

        for file_path in files:
            if _deadline_exceeded(deadline):
                return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="\n".join(matches), error="Code search timed out.")
            try:
                for line_number, line in enumerate(
                    file_path.read_text(encoding="utf-8", errors="replace").splitlines(),
                    start=1,
                ):
                    if _deadline_exceeded(deadline):
                        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="\n".join(matches), error="Code search timed out.")
                    if _line_matches(line=line, regex=regex, literal_pattern=literal_pattern):
                        matches.append(f"{file_path}:{line_number}:{line}")
                        if len(matches) >= max_results:
                            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content="\n".join(matches), error=None)
            except OSError:
                continue

        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content="\n".join(matches) or "No matches found.", error=None)



def _is_excluded_path(path: Path) -> bool:
    return any(part in _EXCLUDED_DIRS for part in path.parts)


def _compile_search_pattern(pattern: str) -> tuple[re.Pattern[str] | None, str | None]:
    try:
        return re.compile(pattern), None
    except re.error:
        return None, pattern


def _line_matches(*, line: str, regex: re.Pattern[str] | None, literal_pattern: str | None) -> bool:
    if regex is not None:
        return regex.search(line) is not None
    if literal_pattern is not None:
        return literal_pattern in line
    return False


def _build_deadline(timeout: float) -> float:
    return time.monotonic() + timeout


def _deadline_exceeded(deadline: float) -> bool:
    return time.monotonic() > deadline


def _join_process_output(stdout: str | None, stderr: str | None) -> str:
    return "\n".join(part for part in ((stdout or "").rstrip("\r\n"), (stderr or "").rstrip("\r\n")) if part)

from __future__ import annotations

import time
from typing import Any

from agent_app.tools._path_utils import PathSafetyError, resolve_workspace_path
from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.types import ToolResult

MAX_FILE_READ_LINES = 500
_TRUNCATION_FOOTER = "--- Output truncated to 500 lines. Narrow the range and retry. ---"


class FileReadTool(Tool):
    name = "file_read"
    description = "Read a text file from the workspace."
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
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
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content="",
                error=validation_error,
            )

        raw_path = arguments.get("path")
        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line")
        if end_line is not None and (not isinstance(end_line, int) or end_line < start_line):
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error="'end_line' must be >= 'start_line'.")
        if end_line is not None and (end_line - start_line + 1) > MAX_FILE_READ_LINES:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content="",
                error=f"Requested line range exceeds the maximum of {MAX_FILE_READ_LINES} lines.",
            )

        try:
            path = resolve_workspace_path(context.workspace_root, raw_path)
        except PathSafetyError as exc:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=str(exc))

        if not path.is_file():
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=f"File '{raw_path}' was not found.")

        deadline = _build_deadline(context.timeout)
        selected: list[str] = []
        truncated = False

        try:
            with path.open(encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if _deadline_exceeded(deadline):
                        return ToolResult(
                            tool_call_id=tool_call_id,
                            tool_name=self.name,
                            success=False,
                            content=_format_numbered_output(selected, start_line),
                            error="File read timed out.",
                        )

                    if line_number < start_line:
                        continue
                    if end_line is not None and line_number > end_line:
                        break
                    if end_line is None and len(selected) >= MAX_FILE_READ_LINES:
                        truncated = True
                        break

                    selected.append(line.rstrip("\r\n"))
        except OSError as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content="",
                error=f"Unable to read file: {exc}",
            )

        numbered = _format_numbered_output(selected, start_line)
        if truncated:
            numbered = f"{numbered}\n{_TRUNCATION_FOOTER}" if numbered else _TRUNCATION_FOOTER
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=self.name,
            success=True,
            content=numbered,
            error=None,
        )


def _format_numbered_output(lines: list[str], start_line: int) -> str:
    return "\n".join(
        f"{index}: {line}"
        for index, line in enumerate(lines, start=start_line)
    )


def _build_deadline(timeout: float) -> float:
    return time.monotonic() + timeout


def _deadline_exceeded(deadline: float) -> bool:
    return time.monotonic() > deadline

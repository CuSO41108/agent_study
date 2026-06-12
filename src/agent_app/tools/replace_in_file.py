from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_app.tools._path_utils import PathSafetyError, resolve_workspace_path
from agent_app.tools.base import Tool, ToolExecutionContext, validate_arguments
from agent_app.tools.file_write import _atomic_write_text, _file_recovery_metadata, _validate_target_path
from agent_app.types import ToolResult

MAX_TEXT_EDIT_BYTES = 262144
MAX_TEXT_EDIT_LINES = 4000
AMBIGUOUS_MATCH_ERROR = "Ambiguous match: multiple occurrences found. Refine old_text or set replace_all=true."
NO_MATCH_ERROR = "No match found for old_text."
TEXT_EDIT_SIZE_LIMIT_ERROR = "Text edit target is too large for safe editing in phase 2."
FILE_CHANGED_ERROR = "Target file changed since inspection. Please retry the edit."


@dataclass(frozen=True, slots=True)
class ReplaceInFileInspection:
    path: Path
    relative_path: str
    match_count: int
    replacement_count: int
    byte_count: int
    line_count: int
    diff_preview: str
    _expected_original_content: str
    _updated_content: str


class ReplaceInFileTool(Tool):
    name = "replace_in_file"
    description = "Safely replace exact text in a single existing UTF-8 text file."
    has_side_effect = True
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "old_text": {"type": "string", "minLength": 1},
            "new_text": {"type": "string"},
            "replace_all": {"type": "boolean"},
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    }

    def execute(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        inspection, error = _resolve_replace_in_file_inspection(
            tool_call_id=tool_call_id,
            arguments=arguments,
            context=context,
        )
        if inspection is None:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content="",
                error=error,
            )

        try:
            current_content = inspection.path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content="",
                error="Existing file is not valid UTF-8 text and cannot be safely edited in phase 2.",
            )
        except OSError as exc:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content="",
                error=f"Unable to read existing file: {exc}",
            )

        if current_content != inspection._expected_original_content:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content="",
                error=FILE_CHANGED_ERROR,
            )

        _atomic_write_text(inspection.path, inspection._updated_content)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=self.name,
            success=True,
            content=(
                f"Replaced {inspection.replacement_count} occurrence(s) in "
                f"{inspection.relative_path} ({inspection.byte_count} bytes, {inspection.line_count} lines)."
            ),
            error=None,
        )

    def inspect(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> tuple[ReplaceInFileInspection | None, str | None]:
        return inspect_replace_in_file_request(arguments=arguments, context=context)

    def recovery_metadata(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        inspection = context.prepared_edits.get(tool_call_id)
        if not isinstance(inspection, ReplaceInFileInspection):
            inspection, error = inspect_replace_in_file_request(arguments=arguments, context=context)
            if inspection is None:
                raise ValueError(error or "Unable to build replace recovery metadata.")
        return _file_recovery_metadata(
            relative_path=inspection.relative_path,
            before_exists=True,
            before_content=inspection._expected_original_content,
            after_content=inspection._updated_content,
            success_content=(
                f"Replaced {inspection.replacement_count} occurrence(s) in "
                f"{inspection.relative_path} ({inspection.byte_count} bytes, {inspection.line_count} lines)."
            ),
        )


def inspect_replace_in_file_request(
    *,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[ReplaceInFileInspection | None, str | None]:
    validation_error = validate_arguments(arguments=arguments, schema=ReplaceInFileTool.parameters_schema)
    if validation_error is not None:
        return None, validation_error

    raw_path = arguments.get("path")
    old_text = arguments.get("old_text")
    new_text = arguments.get("new_text")
    replace_all = bool(arguments.get("replace_all", False))

    try:
        path = resolve_workspace_path(context.workspace_root, raw_path)
    except PathSafetyError as exc:
        return None, str(exc)

    try:
        relative_path = str(path.relative_to(context.workspace_root.resolve()))
    except ValueError:
        return None, f"Path '{raw_path}' escapes the workspace root."

    path_error = _validate_target_path(Path(relative_path))
    if path_error:
        return None, path_error
    if not path.exists():
        return None, f"File '{raw_path}' was not found."
    if not path.is_file():
        return None, "Target path is not a regular file."

    try:
        existing_content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, "Existing file is not valid UTF-8 text and cannot be safely edited in phase 2."
    except OSError as exc:
        return None, f"Unable to read existing file: {exc}"

    if _exceeds_text_edit_limits(existing_content):
        return None, TEXT_EDIT_SIZE_LIMIT_ERROR

    match_count = existing_content.count(old_text)
    if match_count == 0:
        return None, NO_MATCH_ERROR
    if match_count > 1 and not replace_all:
        return None, AMBIGUOUS_MATCH_ERROR

    replacement_count = match_count if replace_all else 1
    updated_content = existing_content.replace(old_text, new_text, -1 if replace_all else 1)
    if _exceeds_text_edit_limits(updated_content):
        return None, TEXT_EDIT_SIZE_LIMIT_ERROR

    byte_count = len(updated_content.encode("utf-8"))
    line_count = _line_count(updated_content)
    diff_preview = _build_diff_preview(relative_path=relative_path, existing_content=existing_content, updated_content=updated_content)

    return (
        ReplaceInFileInspection(
            path=path,
            relative_path=relative_path,
            match_count=match_count,
            replacement_count=replacement_count,
            byte_count=byte_count,
            line_count=line_count,
            diff_preview=diff_preview,
            _expected_original_content=existing_content,
            _updated_content=updated_content,
        ),
        None,
    )


def _resolve_replace_in_file_inspection(
    *,
    tool_call_id: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[ReplaceInFileInspection | None, str | None]:
    cached = context.prepared_edits.pop(tool_call_id, None)
    if isinstance(cached, ReplaceInFileInspection):
        return cached, None
    return inspect_replace_in_file_request(arguments=arguments, context=context)


def _build_diff_preview(*, relative_path: str, existing_content: str, updated_content: str, max_lines: int = 12) -> str:
    diff_lines = list(
        difflib.unified_diff(
            existing_content.splitlines(),
            updated_content.splitlines(),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
            lineterm="",
        )
    )
    if not diff_lines:
        return "No textual diff."
    return "\n".join(diff_lines[:max_lines])


def _exceeds_text_edit_limits(content: str) -> bool:
    return (
        len(content.encode("utf-8")) > MAX_TEXT_EDIT_BYTES
        or _line_count(content) > MAX_TEXT_EDIT_LINES
    )


def _line_count(content: str) -> int:
    if not content:
        return 0
    return content.count("\n") + 1

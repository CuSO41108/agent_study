from __future__ import annotations

import difflib
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from agent_app.tools._path_utils import PathSafetyError, resolve_workspace_path
from agent_app.tools.base import Tool, ToolExecutionContext, validate_arguments
from agent_app.types import ToolResult

MAX_FILE_WRITE_BYTES = 65536
MAX_FILE_WRITE_LINES = 1200

_ALLOWED_TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".xml",
    ".html",
    ".css",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".kt",
    ".go",
    ".rs",
    ".sh",
    ".ps1",
    ".sql",
}

_BINARY_SUFFIXES = {
    ".db",
    ".sqlite",
    ".pyc",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".pdf",
    ".zip",
    ".exe",
    ".dll",
    ".so",
}

_PROTECTED_NAMES = {".env", ".env.local"}
_PROTECTED_DIRS = {".git", ".agent_app"}
_SIZE_LIMIT_ERROR = "\u6587\u4ef6\u8fc7\u5927\uff0c\u5f53\u524d\u9636\u6bb5\u4e0d\u652f\u6301\u6574\u6587\u4ef6\u5b89\u5168\u5199\u5165\u3002"
FILE_WRITE_CHANGED_ERROR = "Target file changed since inspection. Please retry the edit."


@dataclass(frozen=True, slots=True)
class FileWriteInspection:
    path: Path
    relative_path: str
    operation: Literal["create", "overwrite"]
    content: str
    byte_count: int
    line_count: int
    existing_content: str | None

    def diff_summary(self, max_lines: int = 12) -> str | None:
        if self.existing_content is None:
            return None
        diff_lines = list(
            difflib.unified_diff(
                self.existing_content.splitlines(),
                self.content.splitlines(),
                fromfile=f"a/{self.relative_path}",
                tofile=f"b/{self.relative_path}",
                lineterm="",
            )
        )
        if not diff_lines:
            return "No textual diff."
        return "\n".join(diff_lines[:max_lines])

    def preview(self, max_lines: int = 12, max_chars: int = 800) -> str:
        text = self.content
        preview_text = text[:max_chars]
        preview_lines = preview_text.splitlines()[:max_lines]
        return "\n".join(preview_lines)


class FileWriteTool(Tool):
    name = "file_write"
    description = "Safely write a small text file inside the workspace."
    has_side_effect = True
    is_idempotent = False
    risk_level = "high"
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def execute(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        inspection, error = _resolve_file_write_inspection(
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

        changed_error = _ensure_file_write_target_unchanged(inspection)
        if changed_error is not None:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content="",
                error=changed_error,
            )

        _atomic_write_text(inspection.path, inspection.content)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=self.name,
            success=True,
            content=(
                f"{inspection.operation.title()}d {inspection.relative_path} "
                f"({inspection.byte_count} bytes, {inspection.line_count} lines)."
            ),
            error=None,
        )

    def inspect(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> tuple[FileWriteInspection | None, str | None]:
        return inspect_file_write_request(arguments=arguments, context=context)

    def recovery_metadata(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        inspection = context.prepared_edits.get(tool_call_id)
        if not isinstance(inspection, FileWriteInspection):
            inspection, error = inspect_file_write_request(arguments=arguments, context=context)
            if inspection is None:
                raise ValueError(error or "Unable to build file-write recovery metadata.")
        return _file_recovery_metadata(
            relative_path=inspection.relative_path,
            before_exists=inspection.existing_content is not None,
            before_content=inspection.existing_content,
            after_content=inspection.content,
            success_content=(
                f"{inspection.operation.title()}d {inspection.relative_path} "
                f"({inspection.byte_count} bytes, {inspection.line_count} lines)."
            ),
        )



def inspect_file_write_request(
    *,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[FileWriteInspection | None, str | None]:
    validation_error = validate_arguments(arguments=arguments, schema=FileWriteTool.parameters_schema)
    if validation_error is not None:
        return None, validation_error

    raw_path = arguments.get("path")
    content = arguments.get("content")
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

    if not path.parent.exists():
        return None, "Parent directory does not exist."

    existing_content: str | None = None
    operation: Literal["create", "overwrite"] = "create"
    if path.exists():
        if not path.is_file():
            return None, "Target path is not a regular file."
        operation = "overwrite"
        try:
            existing_content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return None, "Existing file is not valid UTF-8 text and cannot be safely overwritten in phase 2."
        except OSError as exc:
            return None, f"Unable to read existing file: {exc}"
        if _exceeds_limits(existing_content):
            return None, _SIZE_LIMIT_ERROR

    if _exceeds_limits(content):
        return None, _SIZE_LIMIT_ERROR

    return (
        FileWriteInspection(
            path=path,
            relative_path=relative_path,
            operation=operation,
            content=content,
            byte_count=len(content.encode("utf-8")),
            line_count=_line_count(content),
            existing_content=existing_content,
        ),
        None,
    )


def _resolve_file_write_inspection(
    *,
    tool_call_id: str,
    arguments: dict[str, Any],
    context: ToolExecutionContext,
) -> tuple[FileWriteInspection | None, str | None]:
    cached = context.prepared_edits.pop(tool_call_id, None)
    if isinstance(cached, FileWriteInspection):
        return cached, None
    return inspect_file_write_request(arguments=arguments, context=context)


def _ensure_file_write_target_unchanged(inspection: FileWriteInspection) -> str | None:
    if inspection.operation == "create":
        if inspection.path.exists():
            return FILE_WRITE_CHANGED_ERROR
        return None

    if not inspection.path.is_file():
        return FILE_WRITE_CHANGED_ERROR
    try:
        current_content = inspection.path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return FILE_WRITE_CHANGED_ERROR
    if current_content != inspection.existing_content:
        return FILE_WRITE_CHANGED_ERROR
    return None



def _validate_target_path(relative_path: Path) -> str | None:
    parts = relative_path.parts
    filename = relative_path.name

    if filename in _PROTECTED_NAMES:
        return "Sensitive environment files cannot be modified in phase 2."

    if any(part in _PROTECTED_DIRS for part in parts):
        return "Internal workspace directories cannot be modified in phase 2."

    if any(part.startswith(".") for part in parts if part not in _PROTECTED_DIRS):
        return "Hidden files and directories are not writable in phase 2."

    suffix = relative_path.suffix.lower()
    if suffix in _BINARY_SUFFIXES:
        return "Binary or non-text file types are not writable in phase 2."
    if suffix not in _ALLOWED_TEXT_SUFFIXES:
        return "Only whitelisted text file extensions are writable in phase 2."

    return None



def _exceeds_limits(content: str) -> bool:
    return (
        len(content.encode("utf-8")) > MAX_FILE_WRITE_BYTES
        or _line_count(content) > MAX_FILE_WRITE_LINES
    )



def _line_count(content: str) -> int:
    if not content:
        return 0
    return content.count("\n") + 1


def _file_recovery_metadata(
    *,
    relative_path: str,
    before_exists: bool,
    before_content: str | None,
    after_content: str,
    success_content: str,
) -> dict[str, Any]:
    return {
        "side_effect": True,
        "recovery_kind": "text_file_hash",
        "relative_path": relative_path,
        "before_exists": before_exists,
        "before_sha256": _text_sha256(before_content) if before_exists else None,
        "after_sha256": _text_sha256(after_content),
        "success_content": success_content,
    }


def _text_sha256(content: str | None) -> str:
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()


def _atomic_write_text(path: Path, content: str) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            handle.write(content.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

from __future__ import annotations

from typing import Any

from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.types import SessionContext, TodoItem, ToolResult

TODO_STATUS_VALUES = ("pending", "in_progress", "completed")
MAX_TODO_ITEMS = 20


class TodoReadTool(Tool):
    name = "todo_read"
    description = "Read the current session todo list."
    parameters_schema = {
        "type": "object",
        "properties": {},
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

        session_context, error = _get_required_session_context(context)
        if session_context is None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=error)

        if not session_context.todo_items:
            content = "No active todo items."
        else:
            content = _format_todo_items(session_context.todo_items)
        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content=content, error=None)


class TodoWriteTool(Tool):
    name = "todo_write"
    description = "Replace the current session todo list."
    parameters_schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "minLength": 1},
                        "status": {"type": "string", "enum": list(TODO_STATUS_VALUES)},
                    },
                    "required": ["content", "status"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["items"],
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

        session_context, error = _get_required_session_context(context)
        if session_context is None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=error)

        raw_items = arguments.get("items", [])
        if len(raw_items) > MAX_TODO_ITEMS:
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content="",
                error=f"Todo list cannot contain more than {MAX_TODO_ITEMS} items.",
            )

        todo_items = tuple(
            TodoItem(content=str(item["content"]), status=str(item["status"]))
            for item in raw_items
        )
        assert context.session_service is not None
        assert context.session_id is not None
        context.session_service.upsert_session_context(
            context.session_id,
            summary_text=session_context.summary_text,
            summary_message_id=session_context.summary_message_id,
            todo_items=todo_items,
        )
        content = _format_todo_items(todo_items) if todo_items else "No active todo items."
        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content=content, error=None)


def _get_required_session_context(context: ToolExecutionContext) -> tuple[SessionContext | None, str | None]:
    if context.session_service is None or context.session_id is None:
        return None, "Session-backed todo tools require an active session."
    return context.session_service.get_session_context(context.session_id), None


def _format_todo_items(todo_items: tuple[TodoItem, ...]) -> str:
    return "\n".join(
        f"{index}. [{item.status}] {item.content}"
        for index, item in enumerate(todo_items, start=1)
    )

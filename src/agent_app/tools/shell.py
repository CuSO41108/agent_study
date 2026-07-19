from __future__ import annotations

from typing import Any, Callable

from agent_app.runtime.shell_runtime import ShellRuntime
from agent_app.tools.approval import validate_shell_command
from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.types import ToolResult

Runner = Callable[..., object]


class ShellTool(Tool):
    name = "shell"
    description = (
        "Run any PowerShell command from the workspace root. Every command requires user approval "
        "unless the user has explicitly allowed its prefix for this session. Recursive and batch "
        "deletion commands forbidden by AGENTS.md are always rejected."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "minLength": 1},
        },
        "required": ["command"],
        "additionalProperties": False,
    }
    has_side_effect = True
    is_idempotent = False
    risk_level = "high"

    def __init__(
        self,
        runtime: ShellRuntime | None = None,
        runner: Runner | None = None,
    ) -> None:
        if runtime is not None and runner is not None:
            raise ValueError("Only one of 'runtime' or 'runner' may be provided.")
        self._runtime = runtime or ShellRuntime(runner=runner)

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

        command = arguments.get("command")
        approved, reason = validate_shell_command(command)
        if not approved:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=reason)

        runtime_kwargs: dict[str, Any] = {
            "workspace_root": context.workspace_root,
            "timeout": context.timeout,
        }
        if context.event_sink is not None:
            runtime_kwargs["on_output"] = lambda stream, line: context.event_sink(
                "tool_output",
                {"tool": self.name, "stream": stream, "line": line},
            )
        execution = self._runtime.run(command, **runtime_kwargs)
        if execution.error_type == "timeout":
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content=execution.combined_output,
                error="Shell command timed out.",
            )
        if execution.error_type == "cancelled":
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content=execution.combined_output,
                error="Shell command cancelled.",
            )
        if execution.error_type == "nonzero_exit":
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content=execution.combined_output,
                error=f"Shell command exited with code {execution.exit_code}.",
            )
        if execution.error_type == "runtime_error":
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=self.name,
                success=False,
                content=execution.combined_output,
                error="Shell runtime failed.",
            )
        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content=execution.combined_output, error=None)

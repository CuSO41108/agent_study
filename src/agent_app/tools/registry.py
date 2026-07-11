from __future__ import annotations

from collections.abc import Iterable

from agent_app.tools.base import Tool
from agent_app.tools.code_search import CodeSearchTool
from agent_app.tools.delegate_task import DelegateTaskTool
from agent_app.tools.file_read import FileReadTool
from agent_app.tools.replace_in_file import ReplaceInFileTool
from agent_app.tools.file_write import FileWriteTool
from agent_app.tools.shell import ShellTool
from agent_app.tools.todo import TodoReadTool, TodoWriteTool
from agent_app.tools.web_search import WebSearchTool


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_required(self, name: str) -> Tool:
        tool = self.get(name)
        if tool is None:
            raise KeyError(name)
        return tool

    def get_specs(self, allowed_tools: list[str]) -> list[dict]:
        return [self._tools[name].spec() for name in allowed_tools if name in self._tools]


def build_root_registry(
    *,
    subagent_runner=None,
    shell_runtime=None,
    runner=None,
    web_search_tool: WebSearchTool | None = None,
) -> ToolRegistry:
    tools = _build_shared_tools(shell_runtime=shell_runtime, runner=runner)
    tools.insert(2, web_search_tool or WebSearchTool())
    if subagent_runner is not None:
        tools.insert(2, DelegateTaskTool(runner=subagent_runner))
    return ToolRegistry(tools)


def build_worker_registry(
    *,
    shell_runtime=None,
    runner=None,
) -> ToolRegistry:
    return ToolRegistry(_build_shared_tools(shell_runtime=shell_runtime, runner=runner))


def build_default_registry(
    *,
    subagent_runner=None,
    shell_runtime=None,
    runner=None,
    web_search_tool: WebSearchTool | None = None,
) -> ToolRegistry:
    return build_root_registry(
        subagent_runner=subagent_runner,
        shell_runtime=shell_runtime,
        runner=runner,
        web_search_tool=web_search_tool,
    )


def _build_shared_tools(*, shell_runtime=None, runner=None) -> list[Tool]:
    return [
        FileReadTool(),
        CodeSearchTool(),
        TodoReadTool(),
        TodoWriteTool(),
        ReplaceInFileTool(),
        FileWriteTool(),
        ShellTool(runtime=shell_runtime, runner=runner),
    ]

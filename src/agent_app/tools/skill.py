from __future__ import annotations

from typing import Any

from agent_app.skills.registry import SkillRegistry
from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.types import SkillActivation, ToolResult


class SkillListTool(Tool):
    name = "skill_list"
    description = "List available read-only Skills and their descriptions."
    parameters_schema: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def execute(self, *, tool_call_id: str, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        entries = self._registry.discover()
        if not entries:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content="No Skills are available.")
        lines = ["Available Skills:"]
        for item in entries:
            lines.append(f"- {item.name} [{item.scope}]: {item.description}")
        lines.extend(f"Warning: {warning}" for warning in self._registry.warnings)
        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content="\n".join(lines))


class SkillLoadTool(Tool):
    name = "skill_load"
    description = "Activate a matching read-only Skill for the current running task and return its SKILL.md instructions."
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {"name": {"type": "string", "minLength": 1}},
        "required": ["name"],
        "additionalProperties": False,
    }

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def execute(self, *, tool_call_id: str, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if context.task_id is None or context.session_service is None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error="Skill loading requires an active task.")
        document = self._registry.load(str(arguments["name"]))
        if document is None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error="Skill not found or invalid.")
        try:
            activation = context.session_service.activate_skill(
                task_id=context.task_id,
                skill_name=document.summary.name,
                scope=document.summary.scope,
                source_path=str(document.summary.source_path),
                content_hash=document.content_hash,
                version=document.summary.version,
                activation_reason="model_match",
                source="model",
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=str(exc))
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=self.name,
            success=True,
            content=_render_loaded_skill(document.content, activation),
        )


class SkillReadResourceTool(Tool):
    name = "skill_read_resource"
    description = "Read a small support file explicitly referenced by an active Skill's SKILL.md."
    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "path": {"type": "string", "minLength": 1},
        },
        "required": ["name", "path"],
        "additionalProperties": False,
    }

    def __init__(self, registry: SkillRegistry) -> None:
        self._registry = registry

    def execute(self, *, tool_call_id: str, arguments: dict[str, Any], context: ToolExecutionContext) -> ToolResult:
        if context.task_id is None or context.session_service is None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error="Skill resource loading requires an active task.")
        name = str(arguments["name"])
        active = {item.skill_name: item for item in context.session_service.list_active_skill_activations(context.task_id)}
        activation = active.get(name.casefold())
        if activation is None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error="Skill is not active for this task.")
        document, mismatch = self._registry.load_active(
            name=activation.skill_name,
            scope=activation.scope,
            source_path=activation.source_path,
            content_hash=activation.content_hash,
        )
        if document is None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error=f"Active Skill cannot be used: {mismatch}.")
        resource = self._registry.load_resource_from_document(document, str(arguments["path"]))
        if resource is None:
            return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=False, content="", error="Resource is unavailable, unsafe, too large, or not referenced by SKILL.md.")
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=self.name,
            success=True,
            content=f"Skill resource {resource.skill_name}/{resource.relative_path}:\n{resource.content}",
        )


def _render_loaded_skill(content: str, activation: SkillActivation) -> str:
    return (
        f"Activated Skill '{activation.skill_name}' ({activation.scope}, hash={activation.content_hash[:12]}).\n"
        f"SKILL.md:\n{content}"
    )

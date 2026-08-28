from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, TYPE_CHECKING

from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.types import ToolResult

if TYPE_CHECKING:
    from agent_app.mcp.client import MCPClient


class MCPSchemaError(ValueError):
    pass


def clean_input_schema(raw_schema: Any) -> dict[str, Any]:
    """Keep a conservative JSON Schema subset for Function Calling."""

    if not isinstance(raw_schema, Mapping) or raw_schema.get("type") != "object":
        raise MCPSchemaError("MCP inputSchema must be an object schema.")
    properties = raw_schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise MCPSchemaError("MCP inputSchema properties must be an object.")
    cleaned_properties = {
        str(name): _clean_schema_node(schema)
        for name, schema in properties.items()
        if isinstance(name, str) and name.strip()
    }
    required = raw_schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise MCPSchemaError("MCP inputSchema required must be a string array.")
    unknown_required = sorted(set(required) - set(cleaned_properties))
    if unknown_required:
        raise MCPSchemaError(f"MCP inputSchema requires unknown properties: {', '.join(unknown_required)}")
    schema: dict[str, Any] = {
        "type": "object",
        "properties": cleaned_properties,
        "additionalProperties": bool(raw_schema.get("additionalProperties", False)),
    }
    if required:
        schema["required"] = list(dict.fromkeys(required))
    if isinstance(raw_schema.get("description"), str):
        schema["description"] = raw_schema["description"]
    return schema


class MCPTool(Tool):
    """Expose one discovered MCP tool through the local Tool contract."""

    def __init__(self, client: "MCPClient", descriptor: Mapping[str, Any], *, server_name: str) -> None:
        remote_name = descriptor.get("name")
        if not isinstance(remote_name, str) or not remote_name.strip():
            raise MCPSchemaError("MCP tool name must be a non-empty string.")
        if not server_name.strip():
            raise ValueError("MCP server name cannot be empty.")
        self._client = client
        self.remote_name = remote_name
        self.server_name = server_name
        self.name = _function_name(server_name, remote_name)
        self.description = str(descriptor.get("description") or f"MCP tool {remote_name}")
        self.parameters_schema = clean_input_schema(descriptor.get("inputSchema", {}))
        annotations = descriptor.get("annotations", {})
        self._annotations = dict(annotations) if isinstance(annotations, Mapping) else {}
        self.has_side_effect = not bool(self._annotations.get("readOnlyHint", False))
        self.is_idempotent = bool(self._annotations.get("idempotentHint", False))
        self.risk_level = "high" if self._annotations.get("destructiveHint", False) else (
            "medium" if self.has_side_effect else "low"
        )

    def redact_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return redact_sensitive_arguments(arguments)

    def execute(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        validation_error = self.validate_arguments(arguments)
        if validation_error is not None:
            return ToolResult(tool_call_id, self.name, False, "", validation_error)
        try:
            result = self._client.call_tool(self.remote_name, arguments)
        except Exception as exc:  # noqa: BLE001 - external MCP failures become observations.
            return ToolResult(
                tool_call_id,
                self.name,
                False,
                "",
                f"MCP tool call failed: {type(exc).__name__}: {exc}",
            )
        is_error = bool(result.get("isError", False))
        content = _render_content(result.get("content", result))
        return ToolResult(
            tool_call_id,
            self.name,
            not is_error,
            content if not is_error else "",
            content if is_error else None,
        )

    def recovery_metadata(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return {
            "side_effect": self.has_side_effect_for(arguments),
            "idempotent": self.is_idempotent_for(arguments),
            "mcp_server": self.server_name,
            "mcp_tool": self.remote_name,
            "arguments_hash": arguments_hash(arguments),
        }


def _function_name(server_name: str, remote_name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", f"mcp_{server_name}_{remote_name}").strip("_")
    return value[:64] or "mcp_tool"


def _clean_schema_node(raw_schema: Any) -> dict[str, Any]:
    if not isinstance(raw_schema, Mapping):
        return {}
    cleaned: dict[str, Any] = {}
    schema_type = raw_schema.get("type")
    if isinstance(schema_type, str) and schema_type in {"string", "number", "integer", "boolean", "object", "array", "null"}:
        cleaned["type"] = schema_type
    if isinstance(raw_schema.get("description"), str):
        cleaned["description"] = raw_schema["description"]
    if isinstance(raw_schema.get("enum"), list):
        cleaned["enum"] = raw_schema["enum"]
    if schema_type == "array" and "items" in raw_schema:
        cleaned["items"] = _clean_schema_node(raw_schema["items"])
    if schema_type == "object":
        properties = raw_schema.get("properties", {})
        if isinstance(properties, Mapping):
            cleaned["properties"] = {
                str(name): _clean_schema_node(schema)
                for name, schema in properties.items()
                if isinstance(name, str) and name.strip()
            }
        required = raw_schema.get("required", [])
        if isinstance(required, list) and all(isinstance(item, str) for item in required):
            cleaned["required"] = list(dict.fromkeys(required))
        cleaned["additionalProperties"] = bool(raw_schema.get("additionalProperties", False))
    return cleaned


def redact_sensitive_arguments(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): "<redacted>" if _is_sensitive_key(str(key)) else redact_sensitive_arguments(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_arguments(item) for item in value]
    return value


def arguments_hash(arguments: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    return any(marker in normalized for marker in ("password", "passwd", "secret", "token", "apikey", "authorization", "credential"))


def _render_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        rendered: list[str] = []
        for item in content:
            if isinstance(item, Mapping) and item.get("type") == "text" and isinstance(item.get("text"), str):
                rendered.append(item["text"])
            else:
                rendered.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        return "\n".join(rendered)
    return json.dumps(content, ensure_ascii=False, sort_keys=True)

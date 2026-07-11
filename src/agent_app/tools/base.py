from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from agent_app.types import Observation, ToolResult

if TYPE_CHECKING:
    from agent_app.state.session_service import SessionService


@dataclass(frozen=True, slots=True)
class ToolExecutionContext:
    workspace_root: Path
    timeout: float = 15.0
    prepared_edits: dict[str, Any] = field(default_factory=dict)
    turn_state: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    task_id: str | None = None
    session_service: "SessionService | None" = None
    agent_id: str | None = None
    delegation_depth: int = 0


class Tool(ABC):
    name: str
    description: str
    parameters_schema: dict[str, Any]
    has_side_effect: bool = False
    is_idempotent: bool = True
    risk_level: str = "low"

    def spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    def validate_arguments(self, arguments: Any) -> str | None:
        return validate_arguments(arguments=arguments, schema=self.parameters_schema)

    def inspect(
        self,
        *,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> tuple[Any | None, str | None]:
        return None, "Tool does not support edit inspection."

    def recovery_metadata(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> dict[str, Any]:
        return {"side_effect": self.has_side_effect}

    def has_side_effect_for(self, arguments: dict[str, Any]) -> bool:
        return self.has_side_effect

    def is_idempotent_for(self, arguments: dict[str, Any]) -> bool:
        return self.is_idempotent

    def can_retry(self, observation: Observation) -> bool:
        return (
            observation.retryable
            and (not self.has_side_effect or self.is_idempotent)
        )

    @abstractmethod
    def execute(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        raise NotImplementedError


def validate_arguments(*, arguments: Any, schema: dict[str, Any]) -> str | None:
    if not isinstance(arguments, dict):
        return "Invalid arguments: expected an object."

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(arguments), key=_validation_error_key)
    if not errors:
        return None
    return f"Invalid arguments: {_format_validation_error(errors[0])}"


def _validation_error_key(error: Any) -> tuple[tuple[str, ...], str]:
    return tuple(str(part) for part in error.absolute_path), error.validator


def _format_validation_error(error: Any) -> str:
    if error.validator == "required":
        match = re.search(r"'([^']+)' is a required property", error.message)
        if match:
            return f"{match.group(1)} is required."
    elif error.validator == "additionalProperties":
        match = re.search(r"\('([^']+)' was unexpected\)", error.message)
        if match:
            return f"unexpected field '{match.group(1)}'."
    elif error.validator == "type":
        field_name = _error_field_name(error)
        if field_name is not None:
            article = "an" if str(error.validator_value).startswith(("a", "e", "i", "o", "u")) else "a"
            return f"{field_name} must be {article} {error.validator_value}."
    elif error.validator == "minimum":
        field_name = _error_field_name(error)
        if field_name is not None:
            return f"{field_name} must be greater than or equal to {error.validator_value}."
    elif error.validator == "minLength":
        field_name = _error_field_name(error)
        if field_name is not None and error.validator_value == 1:
            return f"{field_name} must be a non-empty string."
        if field_name is not None:
            return f"{field_name} must be at least {error.validator_value} characters long."

    field_name = _error_field_name(error)
    if field_name is None:
        return "invalid input."
    return f"invalid value for {field_name}."


def _error_field_name(error: Any) -> str | None:
    if error.absolute_path:
        return ".".join(str(part) for part in error.absolute_path)
    return None


def observation_from_tool_result(
    result: ToolResult,
    *,
    side_effect: bool,
    attempt: int,
    duration_ms: int,
) -> Observation:
    if result.success:
        return Observation(
            status="succeeded",
            error_type=None,
            message=result.content,
            retryable=False,
            side_effect=side_effect,
            raw_data=result.content,
            evidence_ref=result.tool_call_id,
            attempt=attempt,
            duration_ms=duration_ms,
        )

    error = result.error or "Tool execution failed."
    lowered = error.lower()
    if lowered.startswith("search_"):
        error_type = lowered.split(":", 1)[0]
        retryable = error_type == "search_request_error"
    elif "timed out" in lowered or "timeout" in lowered:
        error_type = "timeout"
        retryable = True
    elif "not found" in lowered or "no match" in lowered:
        error_type = "not_found"
        retryable = False
    elif "invalid arguments" in lowered or "must be" in lowered or "required" in lowered:
        error_type = "invalid_arguments"
        retryable = False
    elif "denied by user" in lowered:
        error_type = "user_denied"
        retryable = False
    elif "not allowed" in lowered or "whitelist" in lowered or "unsafe" in lowered:
        error_type = "unsafe_action"
        retryable = False
    elif "permission" in lowered or "access denied" in lowered:
        error_type = "permission_denied"
        retryable = False
    elif "changed since inspection" in lowered or "ambiguous" in lowered or "already" in lowered:
        error_type = "conflict"
        retryable = False
    elif "uncertain" in lowered:
        error_type = "uncertain_side_effect"
        retryable = False
    elif "503" in lowered or "temporar" in lowered or "connection" in lowered:
        error_type = "transient"
        retryable = True
    else:
        error_type = "runtime_error"
        retryable = False

    return Observation(
        status="failed",
        error_type=error_type,
        message=error,
        retryable=retryable,
        side_effect=side_effect,
        raw_data=result.content,
        evidence_ref=result.tool_call_id,
        attempt=attempt,
        duration_ms=duration_ms,
    )

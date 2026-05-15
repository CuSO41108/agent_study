from __future__ import annotations

import json
import socket
from typing import Any
from urllib import error, request

from agent_app.config import AppConfig
from agent_app.types import ModelResponse, ToolCall


class OpenAICompatibleModelClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_config(cls, config: AppConfig) -> "OpenAICompatibleModelClient":
        return cls(
            base_url=config.base_url,
            api_key=config.api_key,
            model=config.model,
            timeout=config.model_timeout,
        )

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> ModelResponse:
        if not self.base_url or not self.api_key or not self.model:
            return ModelResponse(
                assistant_text=None,
                raw_response=None,
                error_type="configuration_error",
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages,
            ],
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        http_request = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            return ModelResponse(
                assistant_text=None,
                raw_response={"status": exc.code, "body": response_body},
                error_type="http_error",
            )
        except (error.URLError, TimeoutError, socket.timeout) as exc:
            return ModelResponse(
                assistant_text=None,
                raw_response={"detail": str(exc)},
                error_type="request_error",
            )
        except json.JSONDecodeError as exc:
            return ModelResponse(
                assistant_text=None,
                raw_response={"detail": str(exc)},
                error_type="invalid_json",
            )

        return self._parse_response(raw_response)

    def _parse_response(self, raw_response: dict[str, Any]) -> ModelResponse:
        choices = raw_response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ModelResponse(
                assistant_text=None,
                raw_response=raw_response,
                error_type="invalid_response",
            )

        choice = choices[0]
        if not isinstance(choice, dict):
            return ModelResponse(
                assistant_text=None,
                raw_response=raw_response,
                error_type="invalid_response",
            )

        message = choice.get("message", {})
        if not isinstance(message, dict):
            return ModelResponse(
                assistant_text=None,
                raw_response=raw_response,
                error_type="invalid_response",
            )

        tool_calls, tool_error = self._parse_tool_calls(message.get("tool_calls"))
        return ModelResponse(
            assistant_text=_extract_text_content(message.get("content")),
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            raw_response=raw_response,
            error_type=tool_error,
        )

    def _parse_tool_calls(
        self,
        raw_tool_calls: Any,
    ) -> tuple[list[ToolCall], str | None]:
        if raw_tool_calls is None:
            return [], None
        if not isinstance(raw_tool_calls, list):
            return [], "invalid_tool_calls"

        tool_calls: list[ToolCall] = []
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                return [], "invalid_tool_calls"
            function = raw_tool_call.get("function")
            if not isinstance(function, dict):
                return [], "invalid_tool_calls"
            arguments_raw = function.get("arguments", "{}")
            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError:
                return [], "invalid_tool_arguments"
            if not isinstance(arguments, dict):
                return [], "invalid_tool_arguments"

            tool_calls.append(
                ToolCall(
                    id=str(raw_tool_call.get("id", "")),
                    name=str(function.get("name", "")),
                    arguments=arguments,
                )
            )
        return tool_calls, None


def _extract_text_content(raw_content: Any) -> str | None:
    if raw_content is None:
        return None
    if isinstance(raw_content, str):
        return raw_content
    if isinstance(raw_content, list):
        parts: list[str] = []
        for item in raw_content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts) or None
    return None

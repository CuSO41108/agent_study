from __future__ import annotations

import json
import socket
from math import ceil
from typing import Any, Callable
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
        estimated_input_tokens = ceil(len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) / 4)

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
                error_type=(
                    "quota_exhausted"
                    if _looks_like_quota_exhaustion(status=exc.code, body=response_body)
                    else "http_error"
                ),
                model_name=self.model,
                input_tokens=estimated_input_tokens,
                total_tokens=estimated_input_tokens,
            )
        except (error.URLError, TimeoutError, socket.timeout) as exc:
            return ModelResponse(
                assistant_text=None,
                raw_response={"detail": str(exc)},
                error_type="request_error",
                model_name=self.model,
                input_tokens=estimated_input_tokens,
                total_tokens=estimated_input_tokens,
            )
        except json.JSONDecodeError as exc:
            return ModelResponse(
                assistant_text=None,
                raw_response={"detail": str(exc)},
                error_type="invalid_json",
                model_name=self.model,
                input_tokens=estimated_input_tokens,
                total_tokens=estimated_input_tokens,
            )

        return self._parse_response(raw_response, estimated_input_tokens=estimated_input_tokens)

    def generate_stream(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_delta: Callable[[str], None],
    ) -> ModelResponse:
        """Generate a response while forwarding OpenAI-compatible SSE text deltas."""
        if not self.base_url or not self.api_key or not self.model:
            return ModelResponse(assistant_text=None, raw_response=None, error_type="configuration_error")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        estimated_input_tokens = ceil(len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) / 4)
        http_request = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                return self._parse_stream(response, estimated_input_tokens=estimated_input_tokens, on_delta=on_delta)
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")
            return ModelResponse(
                assistant_text=None,
                raw_response={"status": exc.code, "body": response_body},
                error_type="quota_exhausted" if _looks_like_quota_exhaustion(status=exc.code, body=response_body) else "http_error",
                model_name=self.model,
                input_tokens=estimated_input_tokens,
                total_tokens=estimated_input_tokens,
            )
        except (error.URLError, TimeoutError, socket.timeout) as exc:
            return ModelResponse(
                assistant_text=None,
                raw_response={"detail": str(exc)},
                error_type="request_error",
                model_name=self.model,
                input_tokens=estimated_input_tokens,
                total_tokens=estimated_input_tokens,
            )

    def _parse_stream(self, response: Any, *, estimated_input_tokens: int, on_delta: Callable[[str], None]) -> ModelResponse:
        text_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        model_name = self.model
        usage: dict[str, Any] | None = None
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or line.startswith("event:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                return ModelResponse(
                    assistant_text=None,
                    raw_response={"detail": f"Invalid streaming chunk: {data[:200]}"},
                    error_type="invalid_json",
                    model_name=model_name,
                    input_tokens=estimated_input_tokens,
                    total_tokens=estimated_input_tokens,
                )
            if not isinstance(chunk, dict):
                continue
            choices = chunk.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict) and isinstance(choices[0].get("message"), dict):
                parsed = self._parse_response(chunk, estimated_input_tokens=estimated_input_tokens)
                if parsed.assistant_text:
                    on_delta(parsed.assistant_text)
                return parsed
            model_name = str(chunk.get("model") or model_name)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason") or finish_reason
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = _extract_text_content(delta.get("content"))
            if content:
                text_parts.append(content)
                on_delta(content)
            raw_calls = delta.get("tool_calls")
            if isinstance(raw_calls, list):
                for raw_call in raw_calls:
                    if not isinstance(raw_call, dict):
                        continue
                    index = raw_call.get("index", 0)
                    if not isinstance(index, int):
                        continue
                    call = tool_calls.setdefault(index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
                    if raw_call.get("id"):
                        call["id"] = raw_call["id"]
                    if isinstance(raw_call.get("function"), dict):
                        function = raw_call["function"]
                        if function.get("name"):
                            call["function"]["name"] = function["name"]
                        if isinstance(function.get("arguments"), str):
                            call["function"]["arguments"] += function["arguments"]

        raw_response: dict[str, Any] = {
            "model": model_name,
            "choices": [{"message": {"role": "assistant", "content": "".join(text_parts) or None, "tool_calls": [tool_calls[index] for index in sorted(tool_calls)] or None}, "finish_reason": finish_reason}],
        }
        if usage is not None:
            raw_response["usage"] = usage
        return self._parse_response(raw_response, estimated_input_tokens=estimated_input_tokens)

    def _parse_response(self, raw_response: dict[str, Any], *, estimated_input_tokens: int = 0) -> ModelResponse:
        choices = raw_response.get("choices")
        if not isinstance(choices, list) or not choices:
            return ModelResponse(
                assistant_text=None,
                raw_response=raw_response,
                error_type="invalid_response",
                model_name=self.model,
                input_tokens=estimated_input_tokens,
                total_tokens=estimated_input_tokens,
            )

        choice = choices[0]
        if not isinstance(choice, dict):
            return ModelResponse(
                assistant_text=None,
                raw_response=raw_response,
                error_type="invalid_response",
                model_name=self.model,
                input_tokens=estimated_input_tokens,
                total_tokens=estimated_input_tokens,
            )

        message = choice.get("message", {})
        if not isinstance(message, dict):
            return ModelResponse(
                assistant_text=None,
                raw_response=raw_response,
                error_type="invalid_response",
                model_name=self.model,
                input_tokens=estimated_input_tokens,
                total_tokens=estimated_input_tokens,
            )

        tool_calls, tool_error = self._parse_tool_calls(message.get("tool_calls"))
        assistant_text = _extract_text_content(message.get("content"))
        usage = raw_response.get("usage")
        if isinstance(usage, dict):
            input_tokens = _usage_int(usage, "prompt_tokens", "input_tokens")
            output_tokens = _usage_int(usage, "completion_tokens", "output_tokens")
            total_tokens = _usage_int(usage, "total_tokens") or input_tokens + output_tokens
            usage_source = "provider"
        else:
            input_tokens = estimated_input_tokens
            output_tokens = ceil(len((assistant_text or "").encode("utf-8")) / 4)
            total_tokens = input_tokens + output_tokens
            usage_source = "estimated"
        return ModelResponse(
            assistant_text=assistant_text,
            tool_calls=tool_calls,
            finish_reason=choice.get("finish_reason"),
            raw_response=raw_response,
            error_type=tool_error,
            model_name=str(raw_response.get("model") or self.model),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            usage_source=usage_source,
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


def _usage_int(usage: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = usage.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return 0


def _looks_like_quota_exhaustion(*, status: int, body: str) -> bool:
    normalized = body.lower()
    return status == 402 or any(
        marker in normalized
        for marker in (
            "insufficient_quota",
            "quota exceeded",
            "quota_exceeded",
            "billing hard limit",
            "credit balance",
        )
    )

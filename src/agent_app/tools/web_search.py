from __future__ import annotations

import json
import socket
from typing import Any
from urllib import error, parse, request

from agent_app.tools.base import Tool, ToolExecutionContext
from agent_app.types import ToolResult


class WebSearchTool(Tool):
    """Read-only public-web search backed by Tavily's HTTPS API."""

    name = "web_search"
    description = "Search the public web and return source-backed results."
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        base_url: str = "https://api.tavily.com",
        api_key: str = "",
        timeout: float = 30.0,
        max_results: int = 5,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_results = max_results

    def execute(
        self,
        *,
        tool_call_id: str,
        arguments: dict[str, Any],
        context: ToolExecutionContext,
    ) -> ToolResult:
        validation_error = self.validate_arguments(arguments)
        if validation_error is not None:
            return _failure(tool_call_id, validation_error)
        if not self._api_key:
            return _failure(tool_call_id, "search_configuration_error: SEARCH_API_KEY is not configured.")
        if not self._base_url.startswith("https://"):
            return _failure(tool_call_id, "search_configuration_error: SEARCH_BASE_URL must use HTTPS.")

        query = str(arguments["query"]).strip()
        max_results = min(int(arguments.get("max_results", self._max_results)), self._max_results)
        payload = {
            "query": query,
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "max_results": max_results,
        }
        http_request = request.Request(
            url=f"{self._base_url}/search",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=min(context.timeout, self._timeout)) as response:
                raw_response = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            return _failure(tool_call_id, f"search_http_error: search provider returned HTTP {exc.code}.")
        except (error.URLError, TimeoutError, socket.timeout):
            return _failure(tool_call_id, "search_request_error: search provider request timed out or could not connect.")
        except json.JSONDecodeError:
            return _failure(tool_call_id, "search_invalid_response: search provider returned invalid JSON.")

        results = _normalize_results(raw_response, limit=max_results)
        if not results:
            return _failure(tool_call_id, "search_no_results: search provider returned no usable sources.")
        content = json.dumps(
            {
                "query": query,
                "sources": results,
                "provider_request_id": _safe_text(raw_response.get("request_id"), limit=200),
            },
            ensure_ascii=False,
        )
        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content=content, error=None)


def _normalize_results(raw_response: Any, *, limit: int) -> list[dict[str, Any]]:
    if not isinstance(raw_response, dict) or not isinstance(raw_response.get("results"), list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in raw_response["results"]:
        if not isinstance(item, dict):
            continue
        url = item.get("url")
        if not isinstance(url, str) or not _is_http_url(url):
            continue
        normalized.append(
            {
                "title": _safe_text(item.get("title"), limit=240),
                "url": url,
                "content": _safe_text(item.get("content"), limit=1200),
            }
        )
        if len(normalized) >= limit:
            break
    return normalized


def _safe_text(value: Any, *, limit: int) -> str:
    return value.strip()[:limit] if isinstance(value, str) else ""


def _is_http_url(value: str) -> bool:
    parsed = parse.urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _failure(tool_call_id: str, error_message: str) -> ToolResult:
    return ToolResult(tool_call_id=tool_call_id, tool_name=WebSearchTool.name, success=False, content="", error=error_message)

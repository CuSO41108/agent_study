from __future__ import annotations

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol


class MCPError(RuntimeError):
    """Base error for MCP transport and protocol failures."""


class MCPTransportError(MCPError):
    pass


class MCPProtocolError(MCPError):
    pass


class MCPRPCError(MCPProtocolError):
    def __init__(self, *, code: int | None, message: str, data: Any = None) -> None:
        self.code = code
        self.message = message
        self.data = data
        super().__init__(message)


class MCPTransport(Protocol):
    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def notify(self, payload: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class StdioTransport:
    """Line-delimited JSON-RPC transport for a local MCP server process."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not command or not all(str(part).strip() for part in command):
            raise ValueError("MCP stdio command cannot be empty.")
        self._command = tuple(str(part) for part in command)
        self._cwd = None if cwd is None else str(Path(cwd))
        self._env = None if env is None else {**os.environ, **dict(env)}
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            process = self._ensure_process()
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                process.stdin.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
                process.stdin.flush()
            except OSError as exc:
                raise MCPTransportError(f"MCP stdio write failed: {exc}") from exc
            request_id = payload.get("id")
            while True:
                line = process.stdout.readline()
                if not line:
                    raise MCPTransportError("MCP stdio server closed stdout before responding.")
                try:
                    response = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MCPProtocolError("MCP stdio server returned invalid JSON.") from exc
                if not isinstance(response, Mapping):
                    raise MCPProtocolError("MCP JSON-RPC response must be an object.")
                if "id" not in response:
                    continue
                if response.get("id") != request_id:
                    raise MCPProtocolError(
                        f"MCP response id {response.get('id')!r} does not match request {request_id!r}."
                    )
                return response

    def notify(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            process = self._ensure_process()
            assert process.stdin is not None
            try:
                process.stdin.write(json.dumps(dict(payload), ensure_ascii=False) + "\n")
                process.stdin.flush()
            except OSError as exc:
                raise MCPTransportError(f"MCP stdio notification failed: {exc}") from exc

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                return
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1)
            finally:
                if process.stdin is not None:
                    process.stdin.close()
                if process.stdout is not None:
                    process.stdout.close()

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        try:
            self._process = subprocess.Popen(
                self._command,
                cwd=self._cwd,
                env=self._env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise MCPTransportError(f"Unable to start MCP stdio server: {exc}") from exc
        return self._process


class StreamableHTTPTransport:
    """POST-based MCP transport supporting JSON and event-stream responses."""

    def __init__(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not url.strip():
            raise ValueError("MCP HTTP URL cannot be empty.")
        if timeout <= 0:
            raise ValueError("MCP HTTP timeout must be positive.")
        self._url = url
        self._headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **(dict(headers) if headers else {}),
        }
        self._timeout = timeout
        self._lock = threading.Lock()

    def request(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            return self._post(payload)

    def notify(self, payload: Mapping[str, Any]) -> None:
        with self._lock:
            self._post(payload, expect_response=False)

    def close(self) -> None:
        return None

    def _post(self, payload: Mapping[str, Any], *, expect_response: bool = True) -> Mapping[str, Any]:
        request = urllib.request.Request(
            self._url,
            data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                raw = response.read().decode("utf-8")
                content_type = response.headers.get_content_type()
        except (urllib.error.URLError, OSError) as exc:
            raise MCPTransportError(f"MCP HTTP request failed: {exc}") from exc
        if not expect_response:
            return {}
        return _decode_http_response(raw, content_type=content_type, request_id=payload.get("id"))


class MCPClient:
    """Small synchronous MCP client with dynamic tool discovery."""

    def __init__(
        self,
        transport: MCPTransport,
        *,
        client_name: str = "agentlab",
        protocol_version: str = "2025-06-18",
    ) -> None:
        if not client_name.strip():
            raise ValueError("MCP client name cannot be empty.")
        self._transport = transport
        self._client_name = client_name
        self._protocol_version = protocol_version
        self._next_id = 0
        self._initialized = False
        self._lock = threading.Lock()

    def initialize(self) -> Mapping[str, Any]:
        with self._lock:
            if self._initialized:
                return {}
            result = self._request(
                "initialize",
                {
                    "protocolVersion": self._protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": self._client_name, "version": "0.1.0"},
                },
            )
            self._transport.notify(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
            )
            self._initialized = True
            return result

    def list_tools(self) -> tuple[Mapping[str, Any], ...]:
        self.initialize()
        result = self._request("tools/list", {})
        tools = result.get("tools", [])
        if not isinstance(tools, list) or not all(isinstance(tool, Mapping) for tool in tools):
            raise MCPProtocolError("MCP tools/list result must contain a tools array.")
        return tuple(tools)

    def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        if not name.strip():
            raise ValueError("MCP tool name cannot be empty.")
        self.initialize()
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)})
        if not isinstance(result, Mapping):
            raise MCPProtocolError("MCP tools/call result must be an object.")
        return result

    def discover_tools(self, *, server_name: str = "server"):
        from agent_app.mcp.tool import MCPTool

        return tuple(
            MCPTool(self, descriptor, server_name=server_name)
            for descriptor in self.list_tools()
        )

    def close(self) -> None:
        self._transport.close()

    def _request(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": dict(params)}
        response = self._transport.request(request)
        if not isinstance(response, Mapping):
            raise MCPProtocolError("MCP response must be an object.")
        if response.get("jsonrpc") != "2.0" or response.get("id") != request["id"]:
            raise MCPProtocolError("MCP response has an invalid JSON-RPC envelope.")
        error = response.get("error")
        if isinstance(error, Mapping):
            raise MCPRPCError(
                code=error.get("code") if isinstance(error.get("code"), int) else None,
                message=str(error.get("message", "MCP request failed")),
                data=error.get("data"),
            )
        result = response.get("result", {})
        if not isinstance(result, Mapping):
            raise MCPProtocolError("MCP response result must be an object.")
        return result


def _decode_http_response(raw: str, *, content_type: str, request_id: Any) -> Mapping[str, Any]:
    if content_type == "text/event-stream":
        candidates: list[Mapping[str, Any]] = []
        data_lines: list[str] = []
        for line in raw.splitlines() + [""]:
            if line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").strip())
            elif not line.strip() and data_lines:
                try:
                    value = json.loads("\n".join(data_lines))
                except json.JSONDecodeError as exc:
                    raise MCPProtocolError("MCP event-stream returned invalid JSON.") from exc
                if isinstance(value, Mapping):
                    candidates.append(value)
                data_lines.clear()
        for candidate in candidates:
            if candidate.get("id") == request_id:
                return candidate
        if candidates:
            return candidates[-1]
        raise MCPProtocolError("MCP event-stream returned no JSON-RPC response.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPProtocolError("MCP HTTP response is not valid JSON.") from exc
    if not isinstance(value, Mapping):
        raise MCPProtocolError("MCP HTTP response must be an object.")
    return value

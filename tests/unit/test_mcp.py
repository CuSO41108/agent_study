from __future__ import annotations

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent_app.mcp import (
    MCPClient,
    MCPTool,
    MCPSchemaError,
    StdioTransport,
    StreamableHTTPTransport,
    clean_input_schema,
    redact_sensitive_arguments,
)
from agent_app.tools.registry import ToolRegistry
from agent_app.types import ToolResult


class _FakeTransport:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.notifications: list[dict] = []

    def request(self, payload):
        request = dict(payload)
        self.requests.append(request)
        if request["method"] == "initialize":
            result = {"protocolVersion": "2025-06-18", "capabilities": {}}
        elif request["method"] == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "fetch-data",
                        "description": "Fetch data.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "x-vendor": "ignored"},
                                "api_key": {"type": "string"},
                            },
                            "required": ["query"],
                            "additionalProperties": False,
                        },
                        "annotations": {"readOnlyHint": True},
                    }
                ]
            }
        elif request["method"] == "tools/call":
            result = {"content": [{"type": "text", "text": "fetched"}]}
        else:
            result = {}
        return {"jsonrpc": "2.0", "id": request["id"], "result": result}

    def notify(self, payload):
        self.notifications.append(dict(payload))

    def close(self):
        return None


class MCPTests(unittest.TestCase):
    def test_client_initializes_discovers_and_calls_a_cleaned_tool(self) -> None:
        transport = _FakeTransport()
        client = MCPClient(transport)

        tools = client.discover_tools(server_name="demo")
        self.assertEqual(len(tools), 1)
        tool = tools[0]
        self.assertIsInstance(tool, MCPTool)
        self.assertEqual(tool.name, "mcp_demo_fetch-data")
        self.assertFalse(tool.has_side_effect)
        self.assertEqual(tool.parameters_schema["additionalProperties"], False)
        self.assertNotIn("x-vendor", json.dumps(tool.parameters_schema))

        result = tool.execute(
            tool_call_id="call-1",
            arguments={"query": "status", "api_key": "secret"},
            context=None,  # type: ignore[arg-type]
        )

        self.assertTrue(result.success)
        self.assertEqual(result.content, "fetched")
        self.assertEqual([item["id"] for item in transport.requests], [1, 2, 3])
        self.assertEqual(transport.notifications[0]["method"], "notifications/initialized")

    def test_client_rejects_malformed_schema_and_registry_does_not_replace_tools(self) -> None:
        with self.assertRaises(MCPSchemaError):
            clean_input_schema({"type": "string"})

        registry = ToolRegistry([])
        tool = MCPTool(
            _FakeTransportClient(),
            {
                "name": "read",
                "inputSchema": {"type": "object", "properties": {}},
                "annotations": {"readOnlyHint": True},
            },
            server_name="demo",
        )
        registry.register(tool)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(tool)
        self.assertEqual(registry.names(), (tool.name,))

    def test_sensitive_arguments_are_redacted_but_hash_input_remains_stable(self) -> None:
        arguments = {"api_key": "secret", "nested": {"password": "pw", "query": "ok"}}

        redacted = redact_sensitive_arguments(arguments)

        self.assertEqual(redacted, {"api_key": "<redacted>", "nested": {"password": "<redacted>", "query": "ok"}})
        self.assertEqual(arguments["api_key"], "secret")

    def test_stdio_transport_uses_line_delimited_json_rpc_without_shell(self) -> None:
        code = (
            "import json,sys\n"
            "for line in sys.stdin:\n"
            "    request=json.loads(line)\n"
            "    if 'id' not in request: continue\n"
            "    method=request['method']\n"
            "    result={'protocolVersion':'2025-06-18'} if method == 'initialize' else {'tools':[]}\n"
            "    print(json.dumps({'jsonrpc':'2.0','id':request['id'],'result':result}), flush=True)\n"
        )
        transport = StdioTransport((sys.executable, "-c", code))
        try:
            client = MCPClient(transport)
            self.assertEqual(client.list_tools(), ())
        finally:
            transport.close()

    def test_streamable_http_transport_posts_json_rpc(self) -> None:
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802 - stdlib handler API
                size = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(size))
                if "id" not in request:
                    self.send_response(204)
                    self.end_headers()
                    return
                result = {"protocolVersion": "2025-06-18"} if request["method"] == "initialize" else {"tools": []}
                body = json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                return None

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            transport = StreamableHTTPTransport(f"http://127.0.0.1:{server.server_port}/mcp")
            client = MCPClient(transport)
            self.assertEqual(client.list_tools(), ())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


class _FakeTransportClient:
    def call_tool(self, _name, _arguments):
        return {"content": [{"type": "text", "text": "ok"}]}


if __name__ == "__main__":
    unittest.main()

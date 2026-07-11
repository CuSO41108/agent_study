from __future__ import annotations

import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_app.tools.base import ToolExecutionContext
from agent_app.tools.web_search import WebSearchTool


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None


class WebSearchToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = ToolExecutionContext(workspace_root=Path.cwd(), timeout=15.0)

    def test_search_normalizes_source_backed_results_without_exposing_key(self) -> None:
        tool = WebSearchTool(api_key="top-secret", max_results=2)
        with patch("agent_app.tools.web_search.request.urlopen", return_value=_FakeResponse({
            "request_id": "request-1",
            "results": [
                {"title": "Example", "url": "https://example.com/page", "content": "source evidence"},
                {"title": "Ignored", "url": "mailto:test@example.com", "content": "not a web page"},
            ],
        })) as urlopen:
            result = tool.execute(tool_call_id="search-1", arguments={"query": "example"}, context=self.context)

        self.assertTrue(result.success)
        payload = json.loads(result.content)
        self.assertEqual(payload["sources"], [{"title": "Example", "url": "https://example.com/page", "content": "source evidence"}])
        self.assertNotIn("top-secret", result.content)
        self.assertEqual(urlopen.call_args.args[0].headers["Authorization"], "Bearer top-secret")

    def test_search_requires_a_configured_key(self) -> None:
        result = WebSearchTool().execute(tool_call_id="search-1", arguments={"query": "example"}, context=self.context)

        self.assertFalse(result.success)
        self.assertEqual(result.error, "search_configuration_error: SEARCH_API_KEY is not configured.")

    def test_search_rejects_invalid_provider_shape(self) -> None:
        tool = WebSearchTool(api_key="secret")
        with patch("agent_app.tools.web_search.request.urlopen", return_value=_FakeResponse({"results": "bad"})):
            result = tool.execute(tool_call_id="search-1", arguments={"query": "example"}, context=self.context)

        self.assertFalse(result.success)
        self.assertEqual(result.error, "search_no_results: search provider returned no usable sources.")


if __name__ == "__main__":
    unittest.main()

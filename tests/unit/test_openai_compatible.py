from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch
from urllib import error

from agent_app.config import AppConfig
from agent_app.model.openai_compatible import OpenAICompatibleModelClient


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class OpenAICompatibleClientTests(unittest.TestCase):
    def test_from_config_uses_app_config_fields(self) -> None:
        client = OpenAICompatibleModelClient.from_config(
            AppConfig(
                workspace_root=None,  # type: ignore[arg-type]
                base_url="https://example.invalid/v1",
                api_key="secret",
                model="qwen-plus",
                model_timeout=22.0,
                tool_timeout=8.0,
                context_token_budget=6000,
                summary_trigger_tokens=3000,
            )
        )

        self.assertEqual(client.base_url, "https://example.invalid/v1")
        self.assertEqual(client.api_key, "secret")
        self.assertEqual(client.model, "qwen-plus")
        self.assertEqual(client.timeout, 22.0)

    def test_generate_returns_configuration_error_when_settings_missing(self) -> None:
        client = OpenAICompatibleModelClient(
            base_url="",
            api_key="",
            model="",
            timeout=15,
        )

        response = client.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )

        self.assertEqual(response.error_type, "configuration_error")
        self.assertIsNone(response.raw_response)

    @patch("agent_app.model.openai_compatible.request.urlopen")
    def test_generate_parses_plain_text_response(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": "Hello from model"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        client = OpenAICompatibleModelClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="qwen-plus",
            timeout=15,
        )

        response = client.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )

        self.assertEqual(response.assistant_text, "Hello from model")
        self.assertEqual(response.finish_reason, "stop")
        self.assertEqual(response.tool_calls, [])
        self.assertIsNone(response.error_type)

    @patch("agent_app.model.openai_compatible.request.urlopen")
    def test_generate_parses_tool_calls(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {
                                        "name": "file_read",
                                        "arguments": json.dumps({"path": "README.md"}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
        )
        client = OpenAICompatibleModelClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="qwen-plus",
            timeout=15,
        )

        response = client.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "read README"}],
            tools=[{"type": "function", "function": {"name": "file_read"}}],
        )

        self.assertEqual(response.finish_reason, "tool_calls")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertEqual(response.tool_calls[0].name, "file_read")
        self.assertEqual(response.tool_calls[0].arguments, {"path": "README.md"})

    @patch("agent_app.model.openai_compatible.request.urlopen")
    def test_generate_returns_http_error_with_response_body(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = error.HTTPError(
            url="https://example.invalid/v1/chat/completions",
            code=429,
            msg="too many requests",
            hdrs=None,
            fp=io.BytesIO(b'{"error":"rate limited"}'),
        )
        client = OpenAICompatibleModelClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="qwen-plus",
            timeout=15,
        )

        response = client.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )

        self.assertEqual(response.error_type, "http_error")
        self.assertEqual(response.raw_response["status"], 429)
        self.assertIn("rate limited", response.raw_response["body"])

    @patch("agent_app.model.openai_compatible.request.urlopen")
    def test_generate_returns_request_error_for_transport_failure(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = error.URLError("boom")
        client = OpenAICompatibleModelClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="qwen-plus",
            timeout=15,
        )

        response = client.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )

        self.assertEqual(response.error_type, "request_error")
        self.assertIn("boom", response.raw_response["detail"])

    @patch("agent_app.model.openai_compatible.request.urlopen")
    def test_generate_returns_invalid_json_for_non_json_response(self, mock_urlopen) -> None:
        class _InvalidJsonResponse:
            def read(self) -> bytes:
                return b"{not json"

            def __enter__(self) -> "_InvalidJsonResponse":
                return self

            def __exit__(self, exc_type, exc, tb) -> None:
                return None

        mock_urlopen.return_value = _InvalidJsonResponse()
        client = OpenAICompatibleModelClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="qwen-plus",
            timeout=15,
        )

        response = client.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )

        self.assertEqual(response.error_type, "invalid_json")
        self.assertIn("detail", response.raw_response)

    @patch("agent_app.model.openai_compatible.request.urlopen")
    def test_generate_returns_invalid_response_for_missing_choices_or_bad_message(self, mock_urlopen) -> None:
        responses = [
            {},
            {"choices": ["not-a-dict"]},
            {"choices": [{"message": "not-a-dict"}]},
        ]
        client = OpenAICompatibleModelClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="qwen-plus",
            timeout=15,
        )

        for payload in responses:
            mock_urlopen.return_value = _FakeResponse(payload)
            response = client.generate(
                system_prompt="sys",
                messages=[{"role": "user", "content": "hello"}],
                tools=[],
            )
            self.assertEqual(response.error_type, "invalid_response")

    @patch("agent_app.model.openai_compatible.request.urlopen")
    def test_generate_rejects_invalid_tool_call_shapes(self, mock_urlopen) -> None:
        payloads = [
            {"choices": [{"message": {"content": None, "tool_calls": {}}}]},
            {"choices": [{"message": {"content": None, "tool_calls": ["bad"]}}]},
            {"choices": [{"message": {"content": None, "tool_calls": [{"id": "call-1", "function": "bad"}]}}]},
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [{"id": "call-1", "function": {"name": "file_read", "arguments": "{"}}],
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "function": {"name": "file_read", "arguments": json.dumps(["README.md"])},
                                }
                            ],
                        }
                    }
                ]
            },
        ]
        expected_errors = [
            "invalid_tool_calls",
            "invalid_tool_calls",
            "invalid_tool_calls",
            "invalid_tool_arguments",
            "invalid_tool_arguments",
        ]
        client = OpenAICompatibleModelClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="qwen-plus",
            timeout=15,
        )

        for payload, expected_error in zip(payloads, expected_errors, strict=True):
            mock_urlopen.return_value = _FakeResponse(payload)
            response = client.generate(
                system_prompt="sys",
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"type": "function", "function": {"name": "file_read"}}],
            )
            self.assertEqual(response.error_type, expected_error)

    @patch("agent_app.model.openai_compatible.request.urlopen")
    def test_generate_joins_text_blocks_from_list_content(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": [
                                {"type": "text", "text": "Hello"},
                                {"type": "input_text", "text": "ignored"},
                                {"type": "text", "text": " world"},
                            ]
                        },
                        "finish_reason": "stop",
                    }
                ]
            }
        )
        client = OpenAICompatibleModelClient(
            base_url="https://example.invalid/v1",
            api_key="secret",
            model="qwen-plus",
            timeout=15,
        )

        response = client.generate(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
        )

        self.assertEqual(response.assistant_text, "Hello world")
        self.assertIsNone(response.error_type)

    @patch("agent_app.model.openai_compatible.request.urlopen")
    def test_generate_sends_system_prompt_messages_and_tool_specs(self, mock_urlopen) -> None:
        captured: dict[str, object] = {}

        def _fake_urlopen(http_request, timeout):
            captured["url"] = http_request.full_url
            captured["timeout"] = timeout
            captured["payload"] = json.loads(http_request.data.decode("utf-8"))
            return _FakeResponse(
                {
                    "choices": [
                        {
                            "message": {"content": "ok"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )

        mock_urlopen.side_effect = _fake_urlopen
        client = OpenAICompatibleModelClient(
            base_url="https://example.invalid/v1/",
            api_key="secret",
            model="qwen-plus",
            timeout=12,
        )

        response = client.generate(
            system_prompt="system prompt",
            messages=[{"role": "user", "content": "hello"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "file_read",
                        "description": "Read a file",
                    },
                }
            ],
        )

        self.assertIsNone(response.error_type)
        self.assertEqual(captured["url"], "https://example.invalid/v1/chat/completions")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(
            captured["payload"]["messages"],
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "hello"},
            ],
        )
        self.assertEqual(captured["payload"]["tool_choice"], "auto")
        self.assertEqual(captured["payload"]["tools"][0]["function"]["name"], "file_read")


if __name__ == "__main__":
    unittest.main()

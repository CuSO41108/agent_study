from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from uuid import uuid4

from agent_app.config import load_config


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"config_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        (self.workspace_root / ".agent_app").mkdir()

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_load_config_reads_generic_openai_compatible_settings(self) -> None:
        config = load_config(
            workspace_root=self.workspace_root,
            env={
                "MODEL_BASE_URL": "https://example.invalid/v1",
                "MODEL_API_KEY": "secret",
                "MODEL_NAME": "qwen-plus",
                "MODEL_TIMEOUT": "12.5",
            },
        )

        self.assertEqual(config.workspace_root, self.workspace_root.resolve())
        self.assertEqual(config.base_url, "https://example.invalid/v1")
        self.assertEqual(config.api_key, "secret")
        self.assertEqual(config.model, "qwen-plus")
        self.assertEqual(config.model_timeout, 12.5)
        self.assertEqual(config.tool_timeout, 600.0)
        self.assertEqual(config.context_token_budget, 6000)
        self.assertEqual(config.summary_trigger_tokens, 3000)
        self.assertEqual(config.timeout, 12.5)
        self.assertEqual(config.search_base_url, "https://api.tavily.com")
        self.assertEqual(config.search_api_key, "")
        self.assertEqual(config.search_timeout, 30.0)
        self.assertEqual(config.search_max_results, 5)

    def test_load_config_reads_search_settings_and_bounds_result_count(self) -> None:
        config = load_config(
            workspace_root=self.workspace_root,
            env={
                "SEARCH_BASE_URL": "https://search.example.invalid/",
                "SEARCH_API_KEY": "search-secret",
                "SEARCH_TIMEOUT": "18",
                "SEARCH_MAX_RESULTS": "7",
            },
        )

        self.assertEqual(config.search_base_url, "https://search.example.invalid")
        self.assertEqual(config.search_api_key, "search-secret")
        self.assertEqual(config.search_timeout, 18.0)
        self.assertEqual(config.search_max_results, 7)

    def test_load_config_reads_local_env_file(self) -> None:
        local_env = self.workspace_root / ".agent_app" / ".env.local"
        local_env.write_text(
            "MODEL_BASE_URL=https://example.invalid/v1\n"
            "MODEL_API_KEY=from-local\n"
            "MODEL_NAME=qwen-local\n"
            "MODEL_TIMEOUT=15\n",
            encoding="utf-8",
        )

        config = load_config(workspace_root=self.workspace_root, env={})

        self.assertEqual(config.base_url, "https://example.invalid/v1")
        self.assertEqual(config.api_key, "from-local")
        self.assertEqual(config.model, "qwen-local")
        self.assertEqual(config.model_timeout, 15.0)
        self.assertEqual(config.tool_timeout, 600.0)

    def test_environment_variables_override_local_env_file(self) -> None:
        local_env = self.workspace_root / ".agent_app" / ".env.local"
        local_env.write_text(
            "MODEL_BASE_URL=https://local.invalid/v1\n"
            "MODEL_API_KEY=from-local\n"
            "MODEL_NAME=qwen-local\n"
            "MODEL_TIMEOUT=15\n",
            encoding="utf-8",
        )

        config = load_config(
            workspace_root=self.workspace_root,
            env={
                "MODEL_BASE_URL": "https://env.invalid/v1",
                "MODEL_API_KEY": "from-env",
                "MODEL_NAME": "qwen-env",
                "MODEL_TIMEOUT": "45",
                "TOOL_TIMEOUT": "12",
                "CONTEXT_TOKEN_BUDGET": "7200",
                "SUMMARY_TRIGGER_TOKENS": "3600",
            },
        )

        self.assertEqual(config.base_url, "https://env.invalid/v1")
        self.assertEqual(config.api_key, "from-env")
        self.assertEqual(config.model, "qwen-env")
        self.assertEqual(config.model_timeout, 45.0)
        self.assertEqual(config.tool_timeout, 12.0)
        self.assertEqual(config.context_token_budget, 7200)
        self.assertEqual(config.summary_trigger_tokens, 3600)

    def test_tool_timeout_defaults_to_ten_minutes_when_unset(self) -> None:
        local_env = self.workspace_root / ".agent_app" / ".env.local"
        local_env.write_text(
            "MODEL_BASE_URL=https://example.invalid/v1\n"
            "MODEL_API_KEY=from-local\n"
            "MODEL_NAME=qwen-local\n"
            "MODEL_TIMEOUT=18\n",
            encoding="utf-8",
        )

        config = load_config(workspace_root=self.workspace_root, env={})

        self.assertEqual(config.model_timeout, 18.0)
        self.assertEqual(config.tool_timeout, 600.0)

    def test_load_config_rejects_invalid_timeout_values(self) -> None:
        invalid_envs = (
            {"MODEL_TIMEOUT": "0"},
            {"MODEL_TIMEOUT": "-1"},
            {"MODEL_TIMEOUT": "abc"},
            {"MODEL_TIMEOUT": "10", "TOOL_TIMEOUT": "0"},
            {"MODEL_TIMEOUT": "10", "TOOL_TIMEOUT": "abc"},
            {"MODEL_TIMEOUT": "10", "CONTEXT_TOKEN_BUDGET": "0"},
            {"MODEL_TIMEOUT": "10", "SUMMARY_TRIGGER_TOKENS": "abc"},
            {"SEARCH_TIMEOUT": "0"},
        )

        for env in invalid_envs:
            with self.assertRaisesRegex(ValueError, "must be a positive"):
                load_config(workspace_root=self.workspace_root, env=env)

        with self.assertRaisesRegex(ValueError, "less than or equal to 10"):
            load_config(workspace_root=self.workspace_root, env={"SEARCH_MAX_RESULTS": "11"})

    def test_database_path_lives_under_workspace(self) -> None:
        config = load_config(workspace_root=self.workspace_root, env={})
        self.assertEqual(
            config.database_path,
            self.workspace_root.resolve() / ".agent_app" / "agent.db",
        )


if __name__ == "__main__":
    unittest.main()

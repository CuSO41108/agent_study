from __future__ import annotations

import unittest
from pathlib import Path


class WorkflowBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[2]

    def test_readme_is_chinese_single_file_with_core_commands(self) -> None:
        content = (self.root / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("README.zh-CN.md", content)
        self.assertNotIn("MokioAgent", content)
        self.assertNotIn("MokioClaw", content)
        self.assertNotIn("## Retrieval 复盘", content)
        self.assertNotIn("2A", content)
        self.assertNotIn("2B", content)
        self.assertNotIn("2C", content)
        self._assert_shared_readme_structure(content)
        self.assertIn("## 设计边界", content)
        self.assertIn("不引入 LangChain/LangGraph 等外部编排框架", content)
        self.assertIn('agent-app "src/agent_app/state/', content)

    def _assert_shared_readme_structure(self, content: str) -> None:
        self.assertIn("快速开始", content)
        self.assertIn("核心亮点", content)
        self.assertIn("日常使用", content)
        self.assertIn("设计边界", content)
        self.assertIn("测试", content)
        self.assertIn("延伸文档", content)
        self.assertIn("python -m pip install -e .[dev]", content)
        self.assertIn(".agent_app/.env.local", content)
        self.assertIn(".env.example", content)
        self.assertIn("OpenAI Chat Completions 协议", content)
        self.assertIn("python -m unittest discover -s tests -v", content)
        self.assertIn("python -m coverage run -m unittest discover -s tests -v", content)
        self.assertIn("python -m coverage report --precision=2 --fail-under=90", content)
        self.assertIn("PowerShell", content)
        self.assertIn("--new-session", content)
        self.assertIn("/approve", content)
        self.assertIn("/reject", content)
        self.assertIn("/cancel", content)
        self.assertIn("/new", content)
        self.assertIn("/task", content)
        self.assertIn("/tasks", content)
        self.assertIn("replace_in_file", content)
        self.assertIn("不是通用 patch 系统", content)
        self.assertIn("docs/ARCHITECTURE.md", content)
        self.assertIn("docs/ROADMAP.md", content)
        self.assertIn("docs/TASK_STATE_MACHINE.md", content)
        self.assertIn("docs/EVAL_DEMO.md", content)
        self.assertIn("OpenAI Chat Completions", content)

    def test_run_local_is_a_thin_agent_app_wrapper_without_secrets(self) -> None:
        content = (self.root / "run_local.ps1").read_text(encoding="utf-8")

        self.assertIn("agent-app", content)
        self.assertIn("--workspace-root", content)
        self.assertNotIn("PYTHONPATH", content)
        self.assertNotIn("MODEL_BASE_URL", content)
        self.assertNotIn("MODEL_API_KEY", content)
        self.assertNotIn("MODEL_NAME", content)
        self.assertNotIn("MODEL_TIMEOUT", content)

    def test_gitignore_covers_coverage_artifacts_and_tracks_launcher(self) -> None:
        content = (self.root / ".gitignore").read_text(encoding="utf-8")

        self.assertIn(".coverage", content)
        self.assertIn(".coverage.*", content)
        self.assertIn("htmlcov/", content)
        self.assertNotIn("run_local.ps1", content)


if __name__ == "__main__":
    unittest.main()

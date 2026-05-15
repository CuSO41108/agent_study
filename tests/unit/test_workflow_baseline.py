from __future__ import annotations

import unittest
from pathlib import Path


class WorkflowBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[2]

    def test_readme_is_chinese_single_file_with_core_commands(self) -> None:
        content = (self.root / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("README.zh-CN.md", content)
        self._assert_shared_readme_structure(content)
        self.assertIn("## 项目定位", content)
        self.assertIn("## Retrieval 复盘", content)
        self.assertIn("当前项目没有使用 LangChain 或 LangGraph", content)
        self.assertIn("没有实现严格意义上的 Plan-and-Execute 或 ReAct", content)
        self.assertIn('python -m agent_app.cli --workspace-root . "当前项目哪里定义了 session 相关逻辑？请给出文件路径。"', content)
        self.assertIn('python -m agent_app.cli --workspace-root . "我最喜欢的数字是 42"', content)

    def _assert_shared_readme_structure(self, content: str) -> None:
        self.assertIn("项目定位", content)
        self.assertIn("快速开始", content)
        self.assertIn("基本使用", content)
        self.assertIn("模块地图", content)
        self.assertIn("当前能力", content)
        self.assertIn("当前边界", content)
        self.assertIn("后续路线", content)
        self.assertIn("测试与验证", content)
        self.assertIn("延伸文档", content)
        self.assertIn("python -m pip install -e .[dev]", content)
        self.assertIn("agent-app --help", content)
        self.assertIn("python -m agent_app.cli --help", content)
        self.assertIn('python -m agent_app.cli "hello, what model are you" --workspace-root .', content)
        self.assertIn('python -m agent_app.cli "do you remember my last question" --workspace-root .', content)
        self.assertIn('python -m agent_app.cli "hello" --workspace-root . --new-session', content)
        self.assertIn("python -m agent_app.cli --interactive --workspace-root .", content)
        self.assertIn("agent-app --interactive --workspace-root .", content)
        self.assertIn("python -m unittest discover -s tests -v", content)
        self.assertIn("python -m coverage run -m unittest discover -s tests -v", content)
        self.assertIn("python -m coverage report --fail-under=90", content)
        self.assertIn(".agent_app/.env.local", content)
        self.assertIn(".env.example", content)
        self.assertIn("PowerShell", content)
        self.assertIn("快速 smoke test", content)
        self.assertIn("tool_runs", content)
        self.assertIn("--workspace-root", content)
        self.assertIn("输入 `:new` 可以开始新 session", content)
        self.assertIn("replace_in_file", content)
        self.assertIn("不是通用 patch 系统", content)
        self.assertIn("ShellRuntime", content)
        self.assertIn("turn_traces", content)
        self.assertIn("tool_call_traces", content)
        self.assertIn("tests/regression/", content)
        self.assertIn("docs/AGENT_CONTEXT_ENGINEERING.md", content)
        self.assertIn("Retrieval 暂时放到实验分支", content)

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

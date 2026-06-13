from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


class ProjectMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
        self.pyproject = tomllib.loads(self.pyproject_path.read_text(encoding="utf-8"))

    def test_project_metadata_exposes_installable_cli_defaults(self) -> None:
        project = self.pyproject["project"]

        self.assertEqual(project["name"], "agent-study")
        self.assertEqual(project["requires-python"], ">=3.13")
        self.assertIn("coverage[toml]", project["optional-dependencies"]["dev"])
        self.assertEqual(project["scripts"]["agent-app"], "agent_app.cli:main")

    def test_setuptools_uses_src_layout(self) -> None:
        package_find = self.pyproject["tool"]["setuptools"]["packages"]["find"]

        self.assertEqual(package_find["where"], ["src"])

    def test_coverage_targets_core_modules_only(self) -> None:
        coverage_tool = self.pyproject["tool"]["coverage"]
        include = coverage_tool["run"]["include"]
        omit = coverage_tool["report"]["omit"]

        self.assertEqual(coverage_tool["report"]["fail_under"], 90)
        self.assertIn("src/agent_app/config.py", include)
        self.assertIn("src/agent_app/tools/*.py", include)
        self.assertIn("src/agent_app/orchestrator/*.py", include)
        self.assertIn("src/agent_app/runtime/*.py", include)
        self.assertIn("src/agent_app/cli.py", omit)
        self.assertIn("src/agent_app/agent/prompts.py", omit)


if __name__ == "__main__":
    unittest.main()

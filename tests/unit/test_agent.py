from __future__ import annotations

import unittest

from agent_app.agent.definition import AGENT_CATALOG, ROOT_COORDINATOR_AGENT, SINGLE_MAIN_AGENT, WORKER_AGENT
from agent_app.agent.prompts import render_system_prompt


class AgentTests(unittest.TestCase):
    def test_single_main_agent_has_expected_tools(self) -> None:
        self.assertEqual(
            SINGLE_MAIN_AGENT.allowed_tools,
            ["file_read", "code_search", "delegate_task", "todo_read", "todo_write", "replace_in_file", "file_write", "shell"],
        )
        self.assertEqual(SINGLE_MAIN_AGENT.max_tool_rounds, 8)
        self.assertEqual(SINGLE_MAIN_AGENT.role, "coordinator")
        self.assertTrue(SINGLE_MAIN_AGENT.can_delegate)

    def test_agent_catalog_exposes_root_and_worker_agents(self) -> None:
        self.assertIs(SINGLE_MAIN_AGENT, ROOT_COORDINATOR_AGENT)
        self.assertIs(AGENT_CATALOG["coordinator"], ROOT_COORDINATOR_AGENT)
        self.assertIs(AGENT_CATALOG["worker"], WORKER_AGENT)
        self.assertEqual(WORKER_AGENT.role, "worker")
        self.assertFalse(WORKER_AGENT.can_delegate)
        self.assertEqual(WORKER_AGENT.max_tool_rounds, 6)

    def test_render_system_prompt_contains_goal_and_rules(self) -> None:
        rendered = render_system_prompt(SINGLE_MAIN_AGENT)

        self.assertIn(SINGLE_MAIN_AGENT.goal, rendered)
        for rule in SINGLE_MAIN_AGENT.rules:
            self.assertIn(rule, rendered)

    def test_prompt_includes_no_matches_stop_rule(self) -> None:
        rendered = render_system_prompt(SINGLE_MAIN_AGENT)

        self.assertIn("No matches found.", rendered)
        self.assertIn("stop and tell the user", rendered)

    def test_prompt_includes_file_write_and_verification_rules(self) -> None:
        rendered = render_system_prompt(SINGLE_MAIN_AGENT)

        self.assertIn("small text files", rendered)
        self.assertIn("minimal follow-up fix", rendered)
        self.assertIn("replace_in_file", rendered)
        self.assertIn("todo_write", rendered)

    def test_prompt_includes_powershell_guidance(self) -> None:
        rendered = render_system_prompt(SINGLE_MAIN_AGENT)

        self.assertIn("PowerShell", rendered)
        self.assertIn("code_search and file_read", rendered)
        self.assertIn("answer from those paths", rendered)

    def test_worker_prompt_mentions_delegated_scope(self) -> None:
        rendered = render_system_prompt(WORKER_AGENT)

        self.assertIn(WORKER_AGENT.goal, rendered)
        self.assertIn("delegated subtask", rendered)
        self.assertIn("Do not delegate again", rendered)


if __name__ == "__main__":
    unittest.main()

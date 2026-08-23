from __future__ import annotations

import unittest

from agent_app.plan import PlanPlanner, PlanPlanningError, route_request
from agent_app.types import ModelResponse


class _FakePlannerModel:
    def __init__(self, response: ModelResponse) -> None:
        self.response = response
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class PlanRoutingTests(unittest.TestCase):
    def test_simple_task_defaults_to_react(self) -> None:
        decision = route_request("read README.md")

        self.assertEqual(decision.mode, "react")
        self.assertFalse(decision.explicit)

    def test_complex_task_defaults_to_plan_and_execute(self) -> None:
        decision = route_request("implement the change and verify it with tests")

        self.assertEqual(decision.mode, "plan_and_execute")
        self.assertEqual(decision.reason, "multiple_action_clauses")

    def test_single_sequence_word_and_confirmation_questions_stay_on_react(self) -> None:
        prompts = (
            "也就是说要先安装这个skill：lark-cli，才能使用对吧",
            "要先安装并配置 lark-cli 后，才能使用飞书文档能力，对吗？",
            "先读取 README.md",
            "安装并配置 lark-cli",
            "implement the requested change",
            "explain how install and configure work",
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt):
                decision = route_request(prompt)
                self.assertEqual(decision.mode, "react")
                self.assertEqual(decision.reason, "simple_or_exploratory_task")

    def test_plain_multiline_text_does_not_force_plan_mode(self) -> None:
        decision = route_request("读取 README.md\n告诉我这个项目是做什么的")

        self.assertEqual(decision.mode, "react")

    def test_structured_action_list_uses_plan_and_execute(self) -> None:
        decision = route_request(
            "安装 lark-cli\n"
            "+ 完成飞书授权\n"
            "+ 安装并配置 lark-doc Skill\n"
            "+ 验证 AgentLab 能发现该 Skill"
        )

        self.assertEqual(decision.mode, "plan_and_execute")
        self.assertEqual(decision.reason, "structured_step_list")

    def test_ordered_chinese_actions_use_plan_and_execute(self) -> None:
        decision = route_request("先定位登录超时原因，然后修改代码并运行测试")

        self.assertEqual(decision.mode, "plan_and_execute")
        self.assertEqual(decision.reason, "ordered_action_sequence")

    def test_explicit_modes_keep_the_goal_without_the_command(self) -> None:
        self.assertEqual(route_request("/plan inspect the repository").mode, "plan_only")
        self.assertEqual(route_request("/plan-and-execute fix and verify").goal, "fix and verify")
        self.assertEqual(route_request("/react explain the file").mode, "react")

    def test_plan_requires_a_goal(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a goal"):
            route_request("/plan")

    def test_planner_parses_json_and_forces_original_goal(self) -> None:
        model = _FakePlannerModel(
            ModelResponse(
                assistant_text=(
                    "```json\n"
                    '{"nodes": [{"id": "inspect", "kind": "inspect", '
                    '"objective": "Read the source.", "depends_on": [], '
                    '"allowed_tools": ["file_read"], "acceptance": ["Source read."]}]}\n'
                    "```"
                )
            )
        )
        plan = PlanPlanner(model).create_plan("Inspect the repository")

        self.assertTrue(plan.id.startswith("plan-"))
        self.assertEqual(plan.goal, "Inspect the repository")
        self.assertEqual(plan.nodes[0].status, "pending")
        self.assertEqual(model.calls[0]["tools"], [])

    def test_planner_rejects_invalid_or_non_json_output(self) -> None:
        invalid_model = _FakePlannerModel(ModelResponse(assistant_text="not json"))
        with self.assertRaises(PlanPlanningError):
            PlanPlanner(invalid_model).create_plan("Inspect")

        tool_model = _FakePlannerModel(
            ModelResponse(
                assistant_text=None,
                tool_calls=[],
                error_type="request_error",
            )
        )
        with self.assertRaises(PlanPlanningError):
            PlanPlanner(tool_model).create_plan("Inspect")


if __name__ == "__main__":
    unittest.main()

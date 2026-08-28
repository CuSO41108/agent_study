from __future__ import annotations

import unittest

from agent_app.plan.graph import (
    PlanGraphValidationError,
    parse_plan_graph,
    plan_graph_to_dict,
    ready_node_ids,
    resource_claims_for_node,
    resources_conflict,
    select_non_conflicting_nodes,
    topological_order,
    validate_plan_graph,
)


def _plan_payload(*, nodes: list[dict] | None = None, **overrides: object) -> dict:
    payload = {
        "id": "plan-1",
        "revision": 1,
        "goal": "Implement and verify a small coding change.",
        "nodes": nodes
        or [
            {
                "id": "inspect",
                "kind": "inspect",
                "objective": "Locate the relevant implementation and tests.",
                "depends_on": [],
                "allowed_tools": ["file_read", "code_search"],
                "acceptance": ["Relevant files and test entry points are identified."],
                "status": "pending",
            },
            {
                "id": "edit",
                "kind": "edit",
                "objective": "Apply the smallest required source change.",
                "depends_on": ["inspect"],
                "allowed_tools": ["file_read", "replace_in_file"],
                "acceptance": ["The intended diff is present."],
                "status": "pending",
            },
            {
                "id": "verify",
                "kind": "verify",
                "objective": "Run the focused tests and inspect the diff.",
                "depends_on": ["edit"],
                "allowed_tools": ["shell", "file_read"],
                "acceptance": ["Focused tests pass and the diff is scoped."],
                "status": "pending",
            },
        ],
    }
    payload.update(overrides)
    return payload


class PlanGraphTests(unittest.TestCase):
    def test_valid_graph_parses_and_round_trips(self) -> None:
        graph = parse_plan_graph(_plan_payload())

        self.assertEqual(graph.id, "plan-1")
        self.assertEqual(graph.revision, 1)
        self.assertEqual(tuple(node.id for node in graph.nodes), ("inspect", "edit", "verify"))
        self.assertEqual(topological_order(graph), ("inspect", "edit", "verify"))
        self.assertEqual(plan_graph_to_dict(graph), _plan_payload())

    def test_ready_nodes_are_derived_from_status_and_dependencies(self) -> None:
        payload = _plan_payload(
            nodes=[
                {
                    "id": "first",
                    "kind": "inspect",
                    "objective": "Read the source.",
                    "depends_on": [],
                    "allowed_tools": ["file_read"],
                    "acceptance": ["Source was read."],
                    "status": "completed",
                },
                {
                    "id": "second",
                    "kind": "edit",
                    "objective": "Edit the source.",
                    "depends_on": ["first"],
                    "allowed_tools": ["replace_in_file"],
                    "acceptance": ["The edit is present."],
                    "status": "pending",
                },
                {
                    "id": "third",
                    "kind": "verify",
                    "objective": "Verify the edit.",
                    "depends_on": ["second"],
                    "allowed_tools": ["shell"],
                    "acceptance": ["The test passes."],
                    "status": "pending",
                },
            ]
        )
        graph = parse_plan_graph(payload)

        self.assertEqual(ready_node_ids(graph), ("second",))

    def test_graph_has_no_fixed_node_count_limit(self) -> None:
        nodes = [
            {
                "id": f"inspect-{index}",
                "kind": "inspect",
                "objective": f"Inspect area {index}.",
                "depends_on": [f"inspect-{index - 1}"] if index else [],
                "allowed_tools": ["code_search"],
                "acceptance": ["The area is understood."],
                "status": "pending",
            }
            for index in range(9)
        ]

        graph = parse_plan_graph(_plan_payload(nodes=nodes))

        self.assertEqual(len(graph.nodes), 9)

    def test_schema_errors_are_reported_without_parsing(self) -> None:
        payload = _plan_payload()
        del payload["nodes"][0]["acceptance"]
        payload["unexpected"] = True

        errors = validate_plan_graph(payload)

        self.assertTrue(any("unexpected" in error for error in errors))
        self.assertTrue(any("acceptance" in error for error in errors))
        with self.assertRaises(PlanGraphValidationError):
            parse_plan_graph(payload)

    def test_unknown_dependency_is_rejected(self) -> None:
        payload = _plan_payload()
        payload["nodes"][1]["depends_on"] = ["missing"]

        errors = validate_plan_graph(payload)

        self.assertIn("node 'edit' depends on unknown node 'missing'", errors)

    def test_self_dependency_and_cycle_are_rejected(self) -> None:
        self_dependency = _plan_payload()
        self_dependency["nodes"][0]["depends_on"] = ["inspect"]
        self.assertTrue(any("cannot depend on itself" in error for error in validate_plan_graph(self_dependency)))

        cycle = _plan_payload()
        cycle["nodes"][0]["depends_on"] = ["verify"]
        errors = validate_plan_graph(cycle)

        self.assertTrue(any("dependency cycle" in error for error in errors))
        with self.assertRaises(PlanGraphValidationError):
            parse_plan_graph(cycle)

    def test_planner_cannot_expand_kind_tool_permissions(self) -> None:
        payload = _plan_payload()
        payload["nodes"][0]["allowed_tools"] = ["file_read", "delegate_task"]

        errors = validate_plan_graph(payload)

        self.assertIn("node 'inspect' of kind 'inspect' cannot use tool 'delegate_task'", errors)

    def test_resource_claims_round_trip_and_conservative_defaults(self) -> None:
        payload = _plan_payload(
            nodes=[
                {
                    "id": "read-a",
                    "kind": "inspect",
                    "objective": "Read file A.",
                    "depends_on": [],
                    "allowed_tools": ["file_read"],
                    "acceptance": ["File A was read."],
                    "status": "pending",
                    "resources": [{"key": "file:src/A.py", "mode": "read"}],
                },
                {
                    "id": "write-b",
                    "kind": "edit",
                    "objective": "Write file B.",
                    "depends_on": [],
                    "allowed_tools": ["file_write"],
                    "acceptance": ["File B was written."],
                    "status": "pending",
                    "resources": [{"key": "file:src/B.py", "mode": "write"}],
                },
            ]
        )
        graph = parse_plan_graph(payload)

        self.assertEqual(plan_graph_to_dict(graph), payload)
        self.assertEqual(resource_claims_for_node(graph.nodes[0])[0].normalized_key, "file:src/a.py")
        self.assertFalse(resources_conflict(graph.nodes[0], graph.nodes[1]))
        self.assertEqual(select_non_conflicting_nodes(graph, ("read-a", "write-b"), limit=2), ("read-a", "write-b"))

        default_edit = parse_plan_graph(
            _plan_payload(
                nodes=[
                    {
                        "id": "edit-a",
                        "kind": "edit",
                        "objective": "Edit A.",
                        "depends_on": [],
                        "allowed_tools": ["file_write"],
                        "acceptance": ["A is edited."],
                        "status": "pending",
                    },
                    {
                        "id": "edit-b",
                        "kind": "edit",
                        "objective": "Edit B.",
                        "depends_on": [],
                        "allowed_tools": ["file_write"],
                        "acceptance": ["B is edited."],
                        "status": "pending",
                    },
                ]
            )
        )
        self.assertTrue(resources_conflict(default_edit.nodes[0], default_edit.nodes[1]))

    def test_duplicate_normalized_resource_keys_are_rejected(self) -> None:
        payload = _plan_payload()
        payload["nodes"][0]["resources"] = [
            {"key": "file:src/A.py", "mode": "read"},
            {"key": "file:src\\A.py", "mode": "write"},
        ]

        errors = validate_plan_graph(payload)

        self.assertIn("node 'inspect' declares duplicate resource 'file:src\\A.py'", errors)

    def test_all_supported_node_statuses_are_contract_values(self) -> None:
        for status in ("pending", "running", "waiting_approval", "completed", "failed", "skipped"):
            payload = _plan_payload()
            payload["nodes"][0]["status"] = status

            graph = parse_plan_graph(payload)

            self.assertEqual(graph.nodes[0].status, status)


if __name__ == "__main__":
    unittest.main()

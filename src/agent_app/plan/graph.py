from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import Draft202012Validator


PlanNodeKind = Literal["inspect", "edit", "run", "verify"]
PlanNodeStatus = Literal[
    "pending",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "skipped",
]

_NODE_KINDS = ("inspect", "edit", "run", "verify")
_NODE_STATUSES = (
    "pending",
    "running",
    "waiting_approval",
    "completed",
    "failed",
    "skipped",
)

# These are capability boundaries for PlanGraph nodes. The Planner may narrow
# a node's list, but it cannot grant a node a tool outside this mapping.
DEFAULT_ALLOWED_TOOLS_BY_KIND: dict[PlanNodeKind, tuple[str, ...]] = {
    "inspect": (
        "file_read",
        "code_search",
        "web_search",
        "skill_list",
        "skill_load",
        "skill_read_resource",
    ),
    "edit": (
        "file_read",
        "code_search",
        "replace_in_file",
        "file_write",
    ),
    "run": ("shell",),
    "verify": ("file_read", "code_search", "shell"),
}


PLAN_GRAPH_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AgentLab PlanGraph",
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "revision", "goal", "nodes"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "revision": {"type": "integer", "minimum": 1},
        "goal": {"type": "string", "minLength": 1},
        "nodes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "id",
                    "kind",
                    "objective",
                    "depends_on",
                    "allowed_tools",
                    "acceptance",
                    "status",
                ],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "kind": {"enum": list(_NODE_KINDS)},
                    "objective": {"type": "string", "minLength": 1},
                    "depends_on": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "allowed_tools": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "acceptance": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "status": {"enum": list(_NODE_STATUSES)},
                },
            },
        },
    },
}


@dataclass(frozen=True, slots=True)
class PlanNode:
    """One typed step in a validated PlanGraph."""

    id: str
    kind: PlanNodeKind
    objective: str
    depends_on: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    acceptance: tuple[str, ...]
    status: PlanNodeStatus = "pending"


@dataclass(frozen=True, slots=True)
class PlanGraph:
    """A versioned, static dependency graph for one task goal."""

    id: str
    revision: int
    goal: str
    nodes: tuple[PlanNode, ...]

    def node_map(self) -> dict[str, PlanNode]:
        return {node.id: node for node in self.nodes}


class PlanGraphValidationError(ValueError):
    """Raised when a model-produced PlanGraph cannot be safely executed."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("Invalid PlanGraph: " + "; ".join(self.errors))


def validate_plan_graph(payload: Any) -> tuple[str, ...]:
    """Return all schema and semantic errors without executing the plan."""

    errors: list[str] = []
    if not isinstance(payload, Mapping):
        return ("plan must be an object",)

    schema_errors = sorted(
        Draft202012Validator(PLAN_GRAPH_SCHEMA).iter_errors(payload),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.validator),
    )
    errors.extend(_format_schema_error(error) for error in schema_errors)
    if errors:
        return tuple(errors)

    node_payloads = payload["nodes"]
    assert isinstance(node_payloads, list)
    node_ids = [node["id"] for node in node_payloads]
    duplicate_ids = sorted({node_id for node_id in node_ids if node_ids.count(node_id) > 1})
    errors.extend(f"duplicate node id '{node_id}'" for node_id in duplicate_ids)

    node_id_set = set(node_ids)
    has_unknown_dependency = False
    for node in node_payloads:
        node_id = node["id"]
        kind = node["kind"]
        dependencies = node["depends_on"]
        unknown_dependencies = sorted(set(dependencies) - node_id_set)
        has_unknown_dependency = has_unknown_dependency or bool(unknown_dependencies)
        errors.extend(
            f"node '{node_id}' depends on unknown node '{dependency}'"
            for dependency in unknown_dependencies
        )
        if node_id in dependencies:
            errors.append(f"node '{node_id}' cannot depend on itself")

        allowed_tools = tuple(node["allowed_tools"])
        default_tools = DEFAULT_ALLOWED_TOOLS_BY_KIND[kind]
        unknown_tools = sorted(set(allowed_tools) - set(default_tools))
        errors.extend(
            f"node '{node_id}' of kind '{kind}' cannot use tool '{tool_name}'"
            for tool_name in unknown_tools
        )

    if not any(error.startswith("duplicate node id") for error in errors) and not has_unknown_dependency:
        errors.extend(_cycle_errors(node_payloads))
    return tuple(errors)


def parse_plan_graph(payload: Mapping[str, Any]) -> PlanGraph:
    """Validate and convert a JSON-compatible payload into immutable types."""

    errors = validate_plan_graph(payload)
    if errors:
        raise PlanGraphValidationError(errors)

    nodes_payload = payload["nodes"]
    assert isinstance(nodes_payload, list)
    return PlanGraph(
        id=payload["id"],
        revision=payload["revision"],
        goal=payload["goal"],
        nodes=tuple(
            PlanNode(
                id=node["id"],
                kind=node["kind"],
                objective=node["objective"],
                depends_on=tuple(node["depends_on"]),
                allowed_tools=tuple(node["allowed_tools"]),
                acceptance=tuple(node["acceptance"]),
                status=node["status"],
            )
            for node in nodes_payload
        ),
    )


def plan_graph_to_dict(graph: PlanGraph) -> dict[str, Any]:
    """Serialize a validated graph into the Planner/SQLite JSON shape."""

    return {
        "id": graph.id,
        "revision": graph.revision,
        "goal": graph.goal,
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "objective": node.objective,
                "depends_on": list(node.depends_on),
                "allowed_tools": list(node.allowed_tools),
                "acceptance": list(node.acceptance),
                "status": node.status,
            }
            for node in graph.nodes
        ],
    }


def ready_node_ids(graph: PlanGraph) -> tuple[str, ...]:
    """Return pending nodes whose dependencies are all completed.

    Readiness is derived from the graph snapshot and is intentionally not a
    persisted node status.
    """

    nodes = graph.node_map()
    return tuple(
        node.id
        for node in graph.nodes
        if node.status == "pending"
        and all(nodes[dependency].status == "completed" for dependency in node.depends_on)
    )


def topological_order(graph: PlanGraph) -> tuple[str, ...]:
    """Return a deterministic topological order, preserving input order ties."""

    nodes = graph.node_map()
    indegree = {node.id: len(node.depends_on) for node in graph.nodes}
    dependents: dict[str, list[str]] = {node.id: [] for node in graph.nodes}
    for node in graph.nodes:
        for dependency in node.depends_on:
            dependents[dependency].append(node.id)

    ready = [node.id for node in graph.nodes if indegree[node.id] == 0]
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for dependent in dependents[current]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)

    if len(order) != len(nodes):
        raise PlanGraphValidationError(("graph contains a dependency cycle",))
    return tuple(order)


def _format_schema_error(error: Any) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    location = path or "plan"
    return f"{location}: {error.message}"


def _cycle_errors(nodes: list[Mapping[str, Any]]) -> tuple[str, ...]:
    dependencies = {node["id"]: tuple(node["depends_on"]) for node in nodes}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str, path: tuple[str, ...]) -> str | None:
        if node_id in visiting:
            cycle_start = path.index(node_id) if node_id in path else 0
            cycle = " -> ".join((*path[cycle_start:], node_id))
            return f"graph contains a dependency cycle: {cycle}"
        if node_id in visited:
            return None
        visiting.add(node_id)
        for dependency in dependencies[node_id]:
            cycle_error = visit(dependency, (*path, node_id))
            if cycle_error is not None:
                return cycle_error
        visiting.remove(node_id)
        visited.add(node_id)
        return None

    for node_id in dependencies:
        cycle_error = visit(node_id, ())
        if cycle_error is not None:
            return (cycle_error,)
    return ()

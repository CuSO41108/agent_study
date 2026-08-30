from __future__ import annotations

import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from jsonschema import Draft202012Validator


PlanNodeKind = Literal["inspect", "edit", "run", "verify"]
ResourceAccess = Literal["read", "write", "exclusive"]
PlanNodeStatus = Literal[
    "pending",
    "running",
    "waiting_approval",
    "paused",
    "completed",
    "failed",
    "skipped",
]

_NODE_KINDS = ("inspect", "edit", "run", "verify")
_NODE_STATUSES = (
    "pending",
    "running",
    "waiting_approval",
    "paused",
    "completed",
    "failed",
    "skipped",
)
_RESOURCE_ACCESS_MODES = ("read", "write", "exclusive")

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
                    "resources": {
                        "type": "array",
                        "uniqueItems": True,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["key", "mode"],
                            "properties": {
                                "key": {"type": "string", "minLength": 1},
                                "mode": {"enum": list(_RESOURCE_ACCESS_MODES)},
                            },
                        },
                    },
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
    resources: tuple["ResourceClaim", ...] = ()


@dataclass(frozen=True, slots=True)
class ResourceClaim:
    """A node's declared access to a logical or workspace resource."""

    key: str
    mode: ResourceAccess

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("Resource key cannot be empty.")
        if self.mode not in _RESOURCE_ACCESS_MODES:
            raise ValueError(f"Unsupported resource access mode: {self.mode}")

    @property
    def normalized_key(self) -> str:
        return _normalize_resource_key(self.key)


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

        resource_keys: set[str] = set()
        for resource in node.get("resources", []):
            key = str(resource["key"]).strip()
            if not key:
                errors.append(f"node '{node_id}' declares an empty resource key")
                continue
            normalized_key = _normalize_resource_key(key)
            if normalized_key in resource_keys:
                errors.append(f"node '{node_id}' declares duplicate resource '{key}'")
            resource_keys.add(normalized_key)

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
                resources=tuple(
                    ResourceClaim(key=resource["key"], mode=resource["mode"])
                    for resource in node.get("resources", [])
                ),
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
                **(
                    {
                        "resources": [
                            {"key": resource.key, "mode": resource.mode}
                            for resource in node.resources
                        ]
                    }
                    if node.resources
                    else {}
                ),
            }
            for node in graph.nodes
        ],
    }


def with_node_status(graph: PlanGraph, node_id: str, status: PlanNodeStatus) -> PlanGraph:
    """Return a graph snapshot with one node status changed."""

    if node_id not in graph.node_map():
        raise KeyError(node_id)
    return PlanGraph(
        id=graph.id,
        revision=graph.revision,
        goal=graph.goal,
        nodes=tuple(
            PlanNode(
                id=node.id,
                kind=node.kind,
                objective=node.objective,
                depends_on=node.depends_on,
                allowed_tools=node.allowed_tools,
                acceptance=node.acceptance,
                status=status if node.id == node_id else node.status,
                resources=node.resources,
            )
            for node in graph.nodes
        ),
    )


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


def resource_claims_for_node(node: PlanNode) -> tuple[ResourceClaim, ...]:
    """Return explicit claims, or a conservative kind-based fallback.

    A plan that does not declare resources must remain safe. Inspection is
    therefore a shared workspace read, while edit/run/verify are exclusive
    workspace operations because their tools may have side effects.
    """

    if node.resources:
        return node.resources
    mode: ResourceAccess = "read" if node.kind == "inspect" else "exclusive"
    return (ResourceClaim("workspace", mode),)


def resources_conflict(left: PlanNode, right: PlanNode) -> bool:
    """Return whether two nodes may not execute at the same time."""

    return any(
        _claims_conflict(first, second)
        for first in resource_claims_for_node(left)
        for second in resource_claims_for_node(right)
    )


def select_non_conflicting_nodes(
    graph: PlanGraph,
    node_ids: Sequence[str],
    *,
    limit: int,
) -> tuple[str, ...]:
    """Select a deterministic, resource-safe ready batch."""

    if limit <= 0:
        raise ValueError("Concurrency limit must be positive.")
    nodes = graph.node_map()
    selected: list[str] = []
    for node_id in node_ids:
        candidate = nodes[node_id]
        if any(resources_conflict(candidate, nodes[selected_id]) for selected_id in selected):
            continue
        selected.append(node_id)
        if len(selected) >= limit:
            break
    return tuple(selected)


def _claims_conflict(left: ResourceClaim, right: ResourceClaim) -> bool:
    if left.mode == "read" and right.mode == "read":
        return False
    left_key = left.normalized_key
    right_key = right.normalized_key
    if left_key == right_key:
        return True
    if "workspace" in {left_key, right_key}:
        return True
    if not _is_file_resource(left_key) or not _is_file_resource(right_key):
        return False
    left_path = left_key.split(":", 1)[1]
    right_path = right_key.split(":", 1)[1]
    return _is_path_prefix(left_path, right_path) or _is_path_prefix(right_path, left_path)


def _normalize_resource_key(key: str) -> str:
    value = key.strip().replace("\\", "/")
    if value.casefold() in {"workspace", "workspace:root"}:
        return "workspace"
    if value.casefold().startswith(("file:", "path:")):
        prefix, path = value.split(":", 1)
        return f"{prefix.casefold()}:{posixpath.normpath(path).casefold()}"
    return value.casefold()


def _is_file_resource(key: str) -> bool:
    return key.startswith(("file:", "path:"))


def _is_path_prefix(parent: str, candidate: str) -> bool:
    return parent == candidate or candidate.startswith(parent.rstrip("/") + "/")


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

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from agent_app.plan.graph import (
    PlanGraph,
    PlanNodeStatus,
    parse_plan_graph,
    plan_graph_to_dict,
    with_node_status,
)


PlanRevisionStatus = Literal["active", "completed", "failed", "superseded"]

_NODE_STATUS_TRANSITIONS: dict[PlanNodeStatus, frozenset[PlanNodeStatus]] = {
    "pending": frozenset({"pending", "running", "skipped"}),
    "running": frozenset({"running", "completed", "failed", "waiting_approval"}),
    "waiting_approval": frozenset({"waiting_approval", "pending", "completed", "failed"}),
    "completed": frozenset({"completed"}),
    "failed": frozenset({"failed"}),
    "skipped": frozenset({"skipped"}),
}


@dataclass(frozen=True, slots=True)
class PlanRevision:
    """One persisted snapshot of a task's PlanGraph."""

    id: str
    task_id: str
    graph: PlanGraph
    status: PlanRevisionStatus
    node_results: dict[str, dict[str, Any]]
    replan_reason: str | None
    version: int
    created_at: str
    updated_at: str
    execution_lease: "ExecutionLease" = field(default_factory=lambda: ExecutionLease())


@dataclass(frozen=True, slots=True)
class ExecutionLease:
    """Persisted execution facts; recovery classification stays in memory."""

    owner: str | None = None
    version: int = 0
    heartbeat_at: str | None = None
    expires_at: str | None = None

    def is_active(self, *, now: datetime | None = None) -> bool:
        if self.owner is None or self.expires_at is None:
            return False
        current = now or datetime.now(UTC)
        try:
            return datetime.fromisoformat(self.expires_at) > current
        except ValueError:
            return False


class PlanRevisionNotFound(KeyError):
    pass


class PlanRevisionConflict(RuntimeError):
    pass


class PlanRevisionLeaseConflict(PlanRevisionConflict):
    pass


class InvalidPlanNodeTransition(RuntimeError):
    pass


class PlanStore:
    """SQLite persistence for immutable-ish plan revisions and node snapshots."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)

    def create_revision(
        self,
        task_id: str,
        graph: PlanGraph,
        *,
        status: PlanRevisionStatus = "active",
        node_results: Mapping[str, Mapping[str, Any]] | None = None,
        replan_reason: str | None = None,
    ) -> PlanRevision:
        _validate_revision_inputs(graph, status, node_results)
        timestamp = _utc_now()
        revision_id = str(uuid4())
        results = _normalise_node_results(graph, node_results)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO plan_revisions (
                        id, task_id, graph_id, revision, status, graph_json,
                        node_results_json, replan_reason, version,
                        lease_owner, lease_version, heartbeat_at, expires_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, NULL, 0, NULL, NULL, ?, ?)
                    """,
                    (
                        revision_id,
                        task_id,
                        graph.id,
                        graph.revision,
                        status,
                        _json_dumps(plan_graph_to_dict(graph)),
                        _json_dumps(results),
                        replan_reason,
                        timestamp,
                        timestamp,
                    ),
                )
                row = connection.execute(
                    _REVISION_SELECT + " WHERE id = ? LIMIT 1", (revision_id,)
                ).fetchone()
        except sqlite3.IntegrityError as exc:
            raise PlanRevisionConflict(f"Could not create plan revision: {exc}") from exc
        assert row is not None
        return _revision_from_row(row)

    def get_revision(self, task_id: str, revision: int | None = None) -> PlanRevision | None:
        with self._connect() as connection:
            if revision is None:
                row = connection.execute(
                    _REVISION_SELECT
                    + " WHERE task_id = ? ORDER BY revision DESC LIMIT 1",
                    (task_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    _REVISION_SELECT + " WHERE task_id = ? AND revision = ? LIMIT 1",
                    (task_id, revision),
                ).fetchone()
        return None if row is None else _revision_from_row(row)

    def get_revision_by_id(self, revision_id: str) -> PlanRevision:
        with self._connect() as connection:
            row = connection.execute(
                _REVISION_SELECT + " WHERE id = ? LIMIT 1", (revision_id,)
            ).fetchone()
        if row is None:
            raise PlanRevisionNotFound(revision_id)
        return _revision_from_row(row)

    def get_task_session_id(self, task_id: str) -> str:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT session_id FROM tasks WHERE id = ? LIMIT 1", (task_id,)
            ).fetchone()
        if row is None:
            raise PlanRevisionNotFound(f"task '{task_id}'")
        return str(row[0])

    def get_active_revision(self, task_id: str) -> PlanRevision | None:
        with self._connect() as connection:
            row = connection.execute(
                _REVISION_SELECT
                + " WHERE task_id = ? AND status = 'active' ORDER BY revision DESC LIMIT 1",
                (task_id,),
            ).fetchone()
        return None if row is None else _revision_from_row(row)

    def list_revisions(self, task_id: str) -> list[PlanRevision]:
        with self._connect() as connection:
            rows = connection.execute(
                _REVISION_SELECT + " WHERE task_id = ? ORDER BY revision ASC",
                (task_id,),
            ).fetchall()
        return [_revision_from_row(row) for row in rows]

    def update_node_status(
        self,
        revision_id: str,
        node_id: str,
        status: PlanNodeStatus,
        *,
        result: Mapping[str, Any] | None = None,
        expected_version: int | None = None,
    ) -> PlanRevision:
        with self._connect() as connection:
            current = self._require_revision(connection, revision_id)
            if expected_version is not None and expected_version != current.version:
                raise PlanRevisionConflict(
                    f"Plan revision '{revision_id}' changed from version "
                    f"{expected_version} to {current.version}."
                )
            if current.status != "active":
                raise PlanRevisionConflict(
                    f"Plan revision '{revision_id}' is {current.status}, not active."
                )
            node = current.graph.node_map().get(node_id)
            if node is None:
                raise KeyError(node_id)
            if status not in _NODE_STATUS_TRANSITIONS[node.status]:
                raise InvalidPlanNodeTransition(
                    f"Cannot transition node '{node_id}' from {node.status} to {status}."
                )
            if status == node.status and result is None:
                return current

            next_graph = with_node_status(current.graph, node_id, status)
            next_results = dict(current.node_results)
            if result is not None:
                next_results[node_id] = _normalise_result(result)
            updated = self._update_revision(
                connection,
                current,
                graph=next_graph,
                node_results=next_results,
            )
        return updated

    def resume_waiting_node(self, revision_id: str, node_id: str, *, expected_version: int | None = None) -> PlanRevision:
        return self.update_node_status(
            revision_id,
            node_id,
            "pending",
            expected_version=expected_version,
        )

    def acquire_execution_lease(
        self,
        revision_id: str,
        *,
        owner: str,
        ttl_seconds: float = 60.0,
        now: datetime | None = None,
    ) -> PlanRevision:
        if not owner.strip():
            raise ValueError("Lease owner cannot be empty.")
        if ttl_seconds <= 0:
            raise ValueError("Lease TTL must be positive.")
        current_time = now or datetime.now(UTC)
        timestamp = current_time.isoformat()
        expires_at = (current_time + timedelta(seconds=ttl_seconds)).isoformat()
        with self._connect() as connection:
            current = self._require_revision(connection, revision_id)
            if current.status != "active":
                raise PlanRevisionLeaseConflict(
                    f"Plan revision '{revision_id}' is {current.status}, not active."
                )
            if current.execution_lease.is_active(now=current_time) and current.execution_lease.owner != owner:
                raise PlanRevisionLeaseConflict(
                    f"Plan revision '{revision_id}' is leased by another active executor."
                )
            next_lease_version = current.execution_lease.version + 1
            cursor = connection.execute(
                """
                UPDATE plan_revisions
                SET lease_owner = ?, lease_version = ?, heartbeat_at = ?, expires_at = ?
                WHERE id = ? AND status = 'active' AND lease_version = ?
                """,
                (
                    owner,
                    next_lease_version,
                    timestamp,
                    expires_at,
                    current.id,
                    current.execution_lease.version,
                ),
            )
            if cursor.rowcount != 1:
                raise PlanRevisionLeaseConflict(
                    f"Plan revision '{revision_id}' lease changed concurrently."
                )
            row = connection.execute(_REVISION_SELECT + " WHERE id = ? LIMIT 1", (revision_id,)).fetchone()
        assert row is not None
        return _revision_from_row(row)

    def heartbeat_execution_lease(
        self,
        revision_id: str,
        *,
        owner: str,
        lease_version: int,
        ttl_seconds: float = 60.0,
        now: datetime | None = None,
    ) -> PlanRevision:
        if ttl_seconds <= 0:
            raise ValueError("Lease TTL must be positive.")
        current_time = now or datetime.now(UTC)
        timestamp = current_time.isoformat()
        expires_at = (current_time + timedelta(seconds=ttl_seconds)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE plan_revisions
                SET heartbeat_at = ?, expires_at = ?
                WHERE id = ? AND status = 'active'
                  AND lease_owner = ? AND lease_version = ?
                """,
                (timestamp, expires_at, revision_id, owner, lease_version),
            )
            if cursor.rowcount != 1:
                raise PlanRevisionLeaseConflict(
                    f"Plan revision '{revision_id}' lease is no longer owned by this executor."
                )
            row = connection.execute(_REVISION_SELECT + " WHERE id = ? LIMIT 1", (revision_id,)).fetchone()
        assert row is not None
        return _revision_from_row(row)

    def release_execution_lease(
        self,
        revision_id: str,
        *,
        owner: str,
        lease_version: int | None = None,
    ) -> PlanRevision:
        with self._connect() as connection:
            current = self._require_revision(connection, revision_id)
            if current.execution_lease.owner != owner:
                raise PlanRevisionLeaseConflict(
                    f"Plan revision '{revision_id}' is not owned by '{owner}'."
                )
            expected_version = current.execution_lease.version if lease_version is None else lease_version
            if expected_version != current.execution_lease.version:
                raise PlanRevisionLeaseConflict(
                    f"Plan revision '{revision_id}' lease version changed concurrently."
                )
            cursor = connection.execute(
                """
                UPDATE plan_revisions
                SET lease_owner = NULL, lease_version = ?, heartbeat_at = NULL, expires_at = NULL
                WHERE id = ? AND lease_owner = ? AND lease_version = ?
                """,
                (current.execution_lease.version + 1, revision_id, owner, expected_version),
            )
            if cursor.rowcount != 1:
                raise PlanRevisionLeaseConflict(
                    f"Plan revision '{revision_id}' lease changed concurrently."
                )
            row = connection.execute(_REVISION_SELECT + " WHERE id = ? LIMIT 1", (revision_id,)).fetchone()
        assert row is not None
        return _revision_from_row(row)

    def rewind_running_node_to_pending(
        self,
        revision_id: str,
        node_id: str,
        *,
        expected_version: int,
        now: datetime | None = None,
    ) -> PlanRevision:
        """Explicitly rewind an expired running node after recovery inspection."""

        current_time = now or datetime.now(UTC)
        with self._connect() as connection:
            current = self._require_revision(connection, revision_id)
            if current.status != "active":
                raise PlanRevisionConflict(f"Plan revision '{revision_id}' is not active.")
            if current.version != expected_version:
                raise PlanRevisionConflict(
                    f"Plan revision '{revision_id}' changed from version {expected_version} to {current.version}."
                )
            if current.execution_lease.is_active(now=current_time):
                raise PlanRevisionLeaseConflict(
                    f"Plan revision '{revision_id}' still has an active execution lease."
                )
            node = current.graph.node_map().get(node_id)
            if node is None:
                raise KeyError(node_id)
            if node.status != "running":
                raise InvalidPlanNodeTransition(
                    f"Only a running node can be rewound; '{node_id}' is {node.status}."
                )
            blocking_actions = connection.execute(
                """
                SELECT tool_name, status, recovery_json
                FROM tool_actions
                WHERE task_id = ? AND status IN ('prepared', 'executing', 'uncertain')
                """,
                (current.task_id,),
            ).fetchall()
            for tool_name, action_status, recovery_json in blocking_actions:
                metadata = json.loads(recovery_json)
                if metadata.get("plan_node_id") not in {None, node_id}:
                    continue
                if action_status == "uncertain" or metadata.get("side_effect", False):
                    raise PlanRevisionConflict(
                        f"Cannot rewind node '{node_id}': tool action '{tool_name}' has an uncertain side effect."
                    )
            next_graph = with_node_status(current.graph, node_id, "pending")
            timestamp = current_time.isoformat()
            cursor = connection.execute(
                """
                UPDATE plan_revisions
                SET graph_json = ?, version = ?,
                    lease_owner = NULL, lease_version = ?, heartbeat_at = NULL, expires_at = ?,
                    updated_at = ?
                WHERE id = ? AND status = 'active' AND version = ?
                """,
                (
                    _json_dumps(plan_graph_to_dict(next_graph)),
                    current.version + 1,
                    current.execution_lease.version + 1,
                    None,
                    timestamp,
                    current.id,
                    expected_version,
                ),
            )
            if cursor.rowcount != 1:
                raise PlanRevisionConflict(f"Plan revision '{revision_id}' was updated concurrently.")
            row = connection.execute(_REVISION_SELECT + " WHERE id = ? LIMIT 1", (revision_id,)).fetchone()
        assert row is not None
        return _revision_from_row(row)

    def update_revision_status(
        self,
        revision_id: str,
        status: PlanRevisionStatus,
        *,
        expected_version: int | None = None,
    ) -> PlanRevision:
        if status == "active":
            raise ValueError("Use node updates or create_revision to keep a revision active.")
        with self._connect() as connection:
            current = self._require_revision(connection, revision_id)
            if expected_version is not None and expected_version != current.version:
                raise PlanRevisionConflict(
                    f"Plan revision '{revision_id}' changed from version "
                    f"{expected_version} to {current.version}."
                )
            if current.status == status:
                return current
            if current.status != "active":
                raise PlanRevisionConflict(
                    f"Cannot change plan revision '{revision_id}' from {current.status} to {status}."
                )
            timestamp = _utc_now()
            cursor = connection.execute(
                """
                UPDATE plan_revisions
                SET status = ?, version = ?, lease_owner = NULL,
                    lease_version = lease_version + 1, heartbeat_at = NULL, expires_at = NULL,
                    updated_at = ?
                WHERE id = ? AND version = ?
                """,
                (status, current.version + 1, timestamp, current.id, current.version),
            )
            if cursor.rowcount != 1:
                raise PlanRevisionConflict(f"Plan revision '{revision_id}' was updated concurrently.")
            row = connection.execute(
                _REVISION_SELECT + " WHERE id = ? LIMIT 1", (revision_id,)
            ).fetchone()
        assert row is not None
        return _revision_from_row(row)

    def create_replan(
        self,
        task_id: str,
        graph: PlanGraph,
        *,
        reason: str,
        expected_revision: int | None = None,
        node_results: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> PlanRevision:
        _validate_revision_inputs(graph, "active", node_results)
        timestamp = _utc_now()
        new_revision_id = str(uuid4())
        with self._connect() as connection:
            current_row = connection.execute(
                _REVISION_SELECT
                + " WHERE task_id = ? AND status = 'active' ORDER BY revision DESC LIMIT 1",
                (task_id,),
            ).fetchone()
            if current_row is None:
                raise PlanRevisionNotFound(f"active plan for task '{task_id}'")
            current = _revision_from_row(current_row)
            if expected_revision is not None and expected_revision != current.graph.revision:
                raise PlanRevisionConflict(
                    f"Active plan revision changed from {expected_revision} to {current.graph.revision}."
                )
            if graph.id != current.graph.id:
                raise ValueError("A replan must keep the original plan id.")
            if graph.revision <= current.graph.revision:
                raise ValueError("A replan must increase the plan revision number.")

            current_nodes = current.graph.node_map()
            candidate_nodes = graph.node_map()
            removed_completed = sorted(
                node_id
                for node_id, node in current_nodes.items()
                if node.status == "completed" and node_id not in candidate_nodes
            )
            if removed_completed:
                raise ValueError(
                    "A replan cannot remove completed nodes: "
                    + ", ".join(removed_completed)
                )
            graph = _preserve_completed_nodes(current.graph, graph)

            preserved = {
                node.id: current.node_results[node.id]
                for node in graph.nodes
                if node.status == "completed" and node.id in current.node_results
            }
            preserved.update(_normalise_node_results(graph, node_results))
            old_cursor = connection.execute(
                """
                UPDATE plan_revisions
                SET status = 'superseded', version = ?, lease_owner = NULL,
                    lease_version = lease_version + 1, heartbeat_at = NULL, expires_at = NULL,
                    updated_at = ?
                WHERE id = ? AND status = 'active' AND version = ?
                """,
                (current.version + 1, timestamp, current.id, current.version),
            )
            if old_cursor.rowcount != 1:
                raise PlanRevisionConflict(f"Plan revision '{current.id}' was updated concurrently.")
            try:
                connection.execute(
                    """
                    INSERT INTO plan_revisions (
                        id, task_id, graph_id, revision, status, graph_json,
                        node_results_json, replan_reason, version,
                        lease_owner, lease_version, heartbeat_at, expires_at,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 'active', ?, ?, ?, 1, NULL, 0, NULL, NULL, ?, ?)
                    """,
                    (
                        new_revision_id,
                        task_id,
                        graph.id,
                        graph.revision,
                        _json_dumps(plan_graph_to_dict(graph)),
                        _json_dumps(preserved),
                        reason,
                        timestamp,
                        timestamp,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise PlanRevisionConflict(f"Could not create replanned revision: {exc}") from exc
            row = connection.execute(
                _REVISION_SELECT + " WHERE id = ? LIMIT 1", (new_revision_id,)
            ).fetchone()
        assert row is not None
        return _revision_from_row(row)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self._db_path)
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _require_revision(connection: sqlite3.Connection, revision_id: str) -> PlanRevision:
        row = connection.execute(
            _REVISION_SELECT + " WHERE id = ? LIMIT 1", (revision_id,)
        ).fetchone()
        if row is None:
            raise PlanRevisionNotFound(revision_id)
        return _revision_from_row(row)

    @staticmethod
    def _update_revision(
        connection: sqlite3.Connection,
        current: PlanRevision,
        *,
        graph: PlanGraph,
        node_results: Mapping[str, Mapping[str, Any]],
    ) -> PlanRevision:
        timestamp = _utc_now()
        cursor = connection.execute(
            """
            UPDATE plan_revisions
            SET graph_json = ?, node_results_json = ?, version = ?, updated_at = ?
            WHERE id = ? AND version = ? AND status = 'active'
            """,
            (
                _json_dumps(plan_graph_to_dict(graph)),
                _json_dumps(node_results),
                current.version + 1,
                timestamp,
                current.id,
                current.version,
            ),
        )
        if cursor.rowcount != 1:
            raise PlanRevisionConflict(f"Plan revision '{current.id}' was updated concurrently.")
        row = connection.execute(
            _REVISION_SELECT + " WHERE id = ? LIMIT 1", (current.id,)
        ).fetchone()
        assert row is not None
        return _revision_from_row(row)


_REVISION_SELECT = """
SELECT id, task_id, graph_id, revision, status, graph_json,
       node_results_json, replan_reason, version,
       lease_owner, lease_version, heartbeat_at, expires_at,
       created_at, updated_at
FROM plan_revisions
"""


def _revision_from_row(row: tuple[Any, ...]) -> PlanRevision:
    graph = parse_plan_graph(json.loads(row[5]))
    if graph.id != row[2] or graph.revision != row[3]:
        raise PlanRevisionConflict(f"Stored graph metadata does not match revision '{row[0]}'.")
    return PlanRevision(
        id=row[0],
        task_id=row[1],
        graph=graph,
        status=row[4],
        node_results=json.loads(row[6]),
        replan_reason=row[7],
        version=row[8],
        created_at=row[13],
        updated_at=row[14],
        execution_lease=ExecutionLease(
            owner=row[9],
            version=row[10],
            heartbeat_at=row[11],
            expires_at=row[12],
        ),
    )


def _validate_revision_inputs(
    graph: PlanGraph,
    status: PlanRevisionStatus,
    node_results: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    if status not in {"active", "completed", "failed", "superseded"}:
        raise ValueError(f"Unsupported plan revision status: {status}")
    parse_plan_graph(plan_graph_to_dict(graph))
    _normalise_node_results(graph, node_results)


def _normalise_node_results(
    graph: PlanGraph,
    node_results: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    if node_results is None:
        return {}
    node_ids = set(graph.node_map())
    unknown = sorted(set(node_results) - node_ids)
    if unknown:
        raise ValueError(f"Node results reference unknown nodes: {', '.join(unknown)}")
    return {node_id: _normalise_result(result) for node_id, result in node_results.items()}


def _normalise_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError("A node result must be an object.")
    try:
        encoded = json.dumps(dict(result), ensure_ascii=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise TypeError("A node result must contain JSON-compatible values.") from exc
    assert isinstance(decoded, dict)
    return decoded


def _preserve_completed_nodes(current: PlanGraph, candidate: PlanGraph) -> PlanGraph:
    current_nodes = current.node_map()
    nodes = tuple(
        current_nodes[node.id] if node.id in current_nodes and current_nodes[node.id].status == "completed" else node
        for node in candidate.nodes
    )
    preserved = PlanGraph(
        id=candidate.id,
        revision=candidate.revision,
        goal=candidate.goal,
        nodes=nodes,
    )
    parse_plan_graph(plan_graph_to_dict(preserved))
    return preserved


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()

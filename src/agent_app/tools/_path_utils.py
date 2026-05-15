from __future__ import annotations

from pathlib import Path


class PathSafetyError(ValueError):
    pass



def resolve_workspace_path(workspace_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved_workspace = workspace_root.resolve()
    resolved_candidate = candidate.resolve()
    try:
        resolved_candidate.relative_to(resolved_workspace)
    except ValueError as exc:
        raise PathSafetyError(f"Path '{raw_path}' escapes the workspace root.") from exc
    return resolved_candidate

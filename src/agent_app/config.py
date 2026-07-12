from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class AppConfig:
    workspace_root: Path
    base_url: str
    api_key: str
    model: str
    model_timeout: float
    tool_timeout: float
    context_token_budget: int
    summary_trigger_tokens: int
    search_base_url: str = "https://api.tavily.com"
    search_api_key: str = ""
    search_timeout: float = 30.0
    search_max_results: int = 5

    @property
    def database_path(self) -> Path:
        return self.workspace_root / ".agent_app" / "agent.db"

    @property
    def timeout(self) -> float:
        # Backward-compatible alias for older Python callers.
        return self.model_timeout



def load_config(
    workspace_root: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    env_map = env if env is not None else os.environ
    resolved_workspace = (
        Path(workspace_root).expanduser().resolve()
        if workspace_root is not None
        else Path.cwd().resolve()
    )
    local_values = _load_local_env_values(resolved_workspace)

    model_timeout = _load_positive_timeout(
        "MODEL_TIMEOUT",
        env_map.get("MODEL_TIMEOUT", local_values.get("MODEL_TIMEOUT", "30")),
    )
    tool_timeout_raw = env_map.get("TOOL_TIMEOUT", local_values.get("TOOL_TIMEOUT", "600"))
    tool_timeout = _load_positive_timeout("TOOL_TIMEOUT", tool_timeout_raw)
    context_token_budget = _load_positive_int(
        "CONTEXT_TOKEN_BUDGET",
        env_map.get("CONTEXT_TOKEN_BUDGET", local_values.get("CONTEXT_TOKEN_BUDGET", "6000")),
    )
    summary_trigger_tokens = _load_positive_int(
        "SUMMARY_TRIGGER_TOKENS",
        env_map.get("SUMMARY_TRIGGER_TOKENS", local_values.get("SUMMARY_TRIGGER_TOKENS", "3000")),
    )
    search_timeout = _load_positive_timeout(
        "SEARCH_TIMEOUT",
        env_map.get("SEARCH_TIMEOUT", local_values.get("SEARCH_TIMEOUT", "30")),
    )
    search_max_results = _load_bounded_positive_int(
        "SEARCH_MAX_RESULTS",
        env_map.get("SEARCH_MAX_RESULTS", local_values.get("SEARCH_MAX_RESULTS", "5")),
        maximum=10,
    )

    return AppConfig(
        workspace_root=resolved_workspace,
        base_url=env_map.get("MODEL_BASE_URL", local_values.get("MODEL_BASE_URL", "")),
        api_key=env_map.get("MODEL_API_KEY", local_values.get("MODEL_API_KEY", "")),
        model=env_map.get("MODEL_NAME", local_values.get("MODEL_NAME", "")),
        model_timeout=model_timeout,
        tool_timeout=tool_timeout,
        context_token_budget=context_token_budget,
        summary_trigger_tokens=summary_trigger_tokens,
        search_base_url=env_map.get("SEARCH_BASE_URL", local_values.get("SEARCH_BASE_URL", "https://api.tavily.com")).rstrip("/"),
        search_api_key=env_map.get("SEARCH_API_KEY", local_values.get("SEARCH_API_KEY", "")),
        search_timeout=search_timeout,
        search_max_results=search_max_results,
    )



def _load_local_env_values(workspace_root: Path) -> dict[str, str]:
    local_path = workspace_root / ".agent_app" / ".env.local"
    if not local_path.is_file():
        return {}

    values: dict[str, str] = {}
    for line in local_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_value = stripped.split("=", 1)
        values[key.strip()] = raw_value.strip().strip('"').strip("'")
    return values


def _load_positive_timeout(name: str, raw_value: str) -> float:
    try:
        timeout = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number.") from exc
    if timeout <= 0:
        raise ValueError(f"{name} must be a positive number.")
    return timeout


def _load_positive_int(name: str, raw_value: str) -> int:
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _load_bounded_positive_int(name: str, raw_value: str, *, maximum: int) -> int:
    value = _load_positive_int(name, raw_value)
    if value > maximum:
        raise ValueError(f"{name} must be less than or equal to {maximum}.")
    return value

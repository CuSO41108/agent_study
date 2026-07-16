from __future__ import annotations

import os
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


_GLOBAL_HOME_ENV = "AGENT_STUDY_HOME"


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
    home_dir: str | Path | None = None,
) -> AppConfig:
    env_map = env if env is not None else os.environ
    resolved_workspace = (
        Path(workspace_root).expanduser().resolve()
        if workspace_root is not None
        else Path.cwd().resolve()
    )
    global_values = _load_global_config_values(env=env_map, home_dir=home_dir)
    local_values = _load_local_env_values(resolved_workspace)

    model_timeout = _load_positive_timeout(
        "MODEL_TIMEOUT",
        _resolve_value("MODEL_TIMEOUT", env_map, local_values, global_values, "30"),
    )
    tool_timeout_raw = _resolve_value("TOOL_TIMEOUT", env_map, local_values, global_values, "600")
    tool_timeout = _load_positive_timeout("TOOL_TIMEOUT", tool_timeout_raw)
    context_token_budget = _load_positive_int(
        "CONTEXT_TOKEN_BUDGET",
        _resolve_value("CONTEXT_TOKEN_BUDGET", env_map, local_values, global_values, "6000"),
    )
    summary_trigger_tokens = _load_positive_int(
        "SUMMARY_TRIGGER_TOKENS",
        _resolve_value("SUMMARY_TRIGGER_TOKENS", env_map, local_values, global_values, "3000"),
    )
    search_timeout = _load_positive_timeout(
        "SEARCH_TIMEOUT",
        _resolve_value("SEARCH_TIMEOUT", env_map, local_values, global_values, "30"),
    )
    search_max_results = _load_bounded_positive_int(
        "SEARCH_MAX_RESULTS",
        _resolve_value("SEARCH_MAX_RESULTS", env_map, local_values, global_values, "5"),
        maximum=10,
    )

    return AppConfig(
        workspace_root=resolved_workspace,
        base_url=_resolve_value("MODEL_BASE_URL", env_map, local_values, global_values, ""),
        api_key=_resolve_value("MODEL_API_KEY", env_map, local_values, global_values, ""),
        model=_resolve_value("MODEL_NAME", env_map, local_values, global_values, ""),
        model_timeout=model_timeout,
        tool_timeout=tool_timeout,
        context_token_budget=context_token_budget,
        summary_trigger_tokens=summary_trigger_tokens,
        search_base_url=_resolve_value("SEARCH_BASE_URL", env_map, local_values, global_values, "https://api.tavily.com").rstrip("/"),
        search_api_key=_resolve_value("SEARCH_API_KEY", env_map, local_values, global_values, ""),
        search_timeout=search_timeout,
        search_max_results=search_max_results,
    )


def global_config_path(
    *,
    env: Mapping[str, str] | None = None,
    home_dir: str | Path | None = None,
) -> Path:
    env_map = env if env is not None else os.environ
    root = (
        Path(home_dir).expanduser()
        if home_dir is not None
        else Path(env_map[_GLOBAL_HOME_ENV]).expanduser()
        if env_map.get(_GLOBAL_HOME_ENV)
        else Path.home() / ".agent-study"
    )
    return root.resolve() / "config.toml"


def save_global_model_config(
    *,
    base_url: str,
    api_key: str,
    model: str,
    env: Mapping[str, str] | None = None,
    home_dir: str | Path | None = None,
) -> Path:
    """Persist explicit user-provided model settings outside any project workspace."""
    config_path = global_config_path(env=env, home_dir=home_dir)
    values = _load_global_config_values(env=env, home_dir=home_dir)
    values.update(
        {
            "MODEL_BASE_URL": base_url.strip(),
            "MODEL_API_KEY": api_key.strip(),
            "MODEL_NAME": model.strip(),
        }
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_render_global_config(values), encoding="utf-8", newline="\n")
    return config_path



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


def _load_global_config_values(
    *,
    env: Mapping[str, str],
    home_dir: str | Path | None,
) -> dict[str, str]:
    config_path = global_config_path(env=env, home_dir=home_dir)
    if not config_path.is_file():
        return {}
    try:
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ValueError(f"Global config is invalid: {config_path}") from exc
    sections = {
        "model": {
            "base_url": "MODEL_BASE_URL",
            "api_key": "MODEL_API_KEY",
            "name": "MODEL_NAME",
            "timeout": "MODEL_TIMEOUT",
        },
        "agent": {
            "tool_timeout": "TOOL_TIMEOUT",
            "context_token_budget": "CONTEXT_TOKEN_BUDGET",
            "summary_trigger_tokens": "SUMMARY_TRIGGER_TOKENS",
        },
        "search": {
            "base_url": "SEARCH_BASE_URL",
            "api_key": "SEARCH_API_KEY",
            "timeout": "SEARCH_TIMEOUT",
            "max_results": "SEARCH_MAX_RESULTS",
        },
    }
    values: dict[str, str] = {}
    for section_name, fields in sections.items():
        section = document.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"Global config section [{section_name}] must be a table.")
        for field_name, config_key in fields.items():
            value = section.get(field_name)
            if value is not None:
                if not isinstance(value, (str, int, float)):
                    raise ValueError(f"Global config key [{section_name}].{field_name} must be scalar.")
                values[config_key] = str(value)
    return values


def _resolve_value(
    name: str,
    env: Mapping[str, str],
    local_values: Mapping[str, str],
    global_values: Mapping[str, str],
    default: str,
) -> str:
    if name in env:
        return env[name]
    if name in local_values:
        return local_values[name]
    return global_values.get(name, default)


def _render_global_config(values: Mapping[str, str]) -> str:
    groups = (
        ("model", (("base_url", "MODEL_BASE_URL"), ("api_key", "MODEL_API_KEY"), ("name", "MODEL_NAME"), ("timeout", "MODEL_TIMEOUT"))),
        ("agent", (("tool_timeout", "TOOL_TIMEOUT"), ("context_token_budget", "CONTEXT_TOKEN_BUDGET"), ("summary_trigger_tokens", "SUMMARY_TRIGGER_TOKENS"))),
        ("search", (("base_url", "SEARCH_BASE_URL"), ("api_key", "SEARCH_API_KEY"), ("timeout", "SEARCH_TIMEOUT"), ("max_results", "SEARCH_MAX_RESULTS"))),
    )
    lines = ["# Agent Study user configuration. Keep this file private."]
    for section_name, fields in groups:
        selected = [(field_name, values[config_key]) for field_name, config_key in fields if config_key in values]
        if not selected:
            continue
        lines.extend(["", f"[{section_name}]"])
        lines.extend(f"{field_name} = {json.dumps(value, ensure_ascii=False)}" for field_name, value in selected)
    return "\n".join(lines) + "\n"


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

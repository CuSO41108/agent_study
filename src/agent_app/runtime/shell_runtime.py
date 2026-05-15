from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class RuntimeExecutionResult:
    success: bool
    stdout: str
    stderr: str
    combined_output: str
    exit_code: int | None
    error_type: Literal["timeout", "nonzero_exit", "runtime_error"] | None = None


class ShellRuntime:
    def __init__(
        self,
        *,
        runner: Runner | None = None,
        executable_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._runner = runner or subprocess.run
        self._executable_resolver = executable_resolver or _resolve_powershell_executable

    def run(
        self,
        command: str,
        *,
        workspace_root: Path,
        timeout: float,
    ) -> RuntimeExecutionResult:
        executable = self._executable_resolver()
        try:
            completed = self._runner(
                [executable, "-NoProfile", "-Command", command],
                cwd=str(workspace_root),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return RuntimeExecutionResult(
                success=False,
                stdout=stdout,
                stderr=stderr,
                combined_output=_join_output(stdout, stderr),
                exit_code=None,
                error_type="timeout",
            )
        except OSError as exc:
            detail = str(exc)
            return RuntimeExecutionResult(
                success=False,
                stdout="",
                stderr=detail,
                combined_output=detail,
                exit_code=None,
                error_type="runtime_error",
            )

        combined_output = _join_output(completed.stdout, completed.stderr)
        if completed.returncode != 0:
            return RuntimeExecutionResult(
                success=False,
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
                combined_output=combined_output,
                exit_code=completed.returncode,
                error_type="nonzero_exit",
            )

        return RuntimeExecutionResult(
            success=True,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            combined_output=combined_output,
            exit_code=completed.returncode,
            error_type=None,
        )


def _resolve_powershell_executable() -> str:
    return shutil.which("powershell") or shutil.which("pwsh") or "powershell"


def _join_output(stdout: str | None, stderr: str | None) -> str:
    parts = []
    for value in (stdout, stderr):
        if not value:
            continue
        parts.append(value.rstrip("\r\n"))
    return "\n".join(part for part in parts if part)

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
    error_type: Literal["timeout", "cancelled", "nonzero_exit", "runtime_error"] | None = None


class ShellRuntime:
    def __init__(
        self,
        *,
        runner: Runner | None = None,
        executable_resolver: Callable[[], str] | None = None,
    ) -> None:
        self._runner = runner
        self._executable_resolver = executable_resolver or _resolve_powershell_executable

    def run(
        self,
        command: str,
        *,
        workspace_root: Path,
        timeout: float,
    ) -> RuntimeExecutionResult:
        executable = self._executable_resolver()
        if self._runner is not None:
            return self._run_with_runner(executable, command, workspace_root, timeout)
        return self._run_process(executable, command, workspace_root, timeout)

    def _run_with_runner(
        self,
        executable: str,
        command: str,
        workspace_root: Path,
        timeout: float,
    ) -> RuntimeExecutionResult:
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

        return _completed_result(completed)

    def _run_process(
        self,
        executable: str,
        command: str,
        workspace_root: Path,
        timeout: float,
    ) -> RuntimeExecutionResult:
        process: subprocess.Popen[str] | None = None
        try:
            process = subprocess.Popen(
                [executable, "-NoProfile", "-Command", command],
                cwd=str(workspace_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _terminate_process_tree(process)
            stdout, stderr = _communicate_after_termination(process, exc.stdout, exc.stderr)
            return RuntimeExecutionResult(False, stdout, stderr, _join_output(stdout, stderr), None, "timeout")
        except KeyboardInterrupt:
            _terminate_process_tree(process)
            stdout, stderr = _communicate_after_termination(process, "", "")
            return RuntimeExecutionResult(False, stdout, stderr, _join_output(stdout, stderr), None, "cancelled")
        except OSError as exc:
            detail = str(exc)
            return RuntimeExecutionResult(False, "", detail, detail, None, "runtime_error")
        return _completed_result(subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr))

def _completed_result(completed: subprocess.CompletedProcess[str]) -> RuntimeExecutionResult:
    combined_output = _join_output(completed.stdout, completed.stderr)
    if completed.returncode != 0:
        return RuntimeExecutionResult(False, completed.stdout or "", completed.stderr or "", combined_output, completed.returncode, "nonzero_exit")
    return RuntimeExecutionResult(True, completed.stdout or "", completed.stderr or "", combined_output, completed.returncode, None)


def _terminate_process_tree(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True, text=True)
    except OSError:
        process.kill()


def _communicate_after_termination(
    process: subprocess.Popen[str] | None,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> tuple[str, str]:
    if process is not None:
        try:
            stdout, stderr = process.communicate()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return _as_text(stdout), _as_text(stderr)


def _as_text(value: str | bytes | None) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""


def _resolve_powershell_executable() -> str:
    return shutil.which("powershell") or shutil.which("pwsh") or "powershell"


def _join_output(stdout: str | None, stderr: str | None) -> str:
    parts = []
    for value in (stdout, stderr):
        if not value:
            continue
        parts.append(value.rstrip("\r\n"))
    return "\n".join(part for part in parts if part)

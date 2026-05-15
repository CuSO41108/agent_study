from __future__ import annotations

import subprocess
import unittest
from pathlib import Path

from agent_app.runtime.shell_runtime import RuntimeExecutionResult, ShellRuntime


class ShellRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2]

    def test_runtime_returns_successful_execution_result(self) -> None:
        def _runner(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args,
                returncode=0,
                stdout="ok\n",
                stderr="",
            )

        runtime = ShellRuntime(runner=_runner, executable_resolver=lambda: "powershell")
        result = runtime.run("Get-Location", workspace_root=self.workspace_root, timeout=1.0)

        self.assertEqual(
            result,
            RuntimeExecutionResult(
                success=True,
                stdout="ok\n",
                stderr="",
                combined_output="ok",
                exit_code=0,
                error_type=None,
            ),
        )

    def test_runtime_returns_timeout_result(self) -> None:
        def _runner(*args, **kwargs):
            exc = subprocess.TimeoutExpired(cmd="powershell", timeout=1.0)
            exc.stdout = "partial"
            exc.stderr = "boom"
            raise exc

        runtime = ShellRuntime(runner=_runner, executable_resolver=lambda: "powershell")
        result = runtime.run("Get-Location", workspace_root=self.workspace_root, timeout=1.0)

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "timeout")
        self.assertEqual(result.combined_output, "partial\nboom")
        self.assertIsNone(result.exit_code)

    def test_runtime_returns_nonzero_exit_result(self) -> None:
        def _runner(*args, **kwargs):
            return subprocess.CompletedProcess(
                args=args,
                returncode=5,
                stdout="",
                stderr="failed\n",
            )

        runtime = ShellRuntime(runner=_runner, executable_resolver=lambda: "powershell")
        result = runtime.run("Get-Location", workspace_root=self.workspace_root, timeout=1.0)

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "nonzero_exit")
        self.assertEqual(result.exit_code, 5)
        self.assertEqual(result.combined_output, "failed")

    def test_runtime_returns_runtime_error_result(self) -> None:
        def _runner(*args, **kwargs):
            raise OSError("cannot start powershell")

        runtime = ShellRuntime(runner=_runner, executable_resolver=lambda: "powershell")
        result = runtime.run("Get-Location", workspace_root=self.workspace_root, timeout=1.0)

        self.assertFalse(result.success)
        self.assertEqual(result.error_type, "runtime_error")
        self.assertIn("cannot start powershell", result.combined_output)

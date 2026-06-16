from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path
from uuid import uuid4

from evals.runner import main as eval_runner_main
from evals.scorers import (
    VerifyResult,
    changed_files,
    load_cases,
    score_eval_case,
    snapshot_workspace,
)


class EvalScorerTests(unittest.TestCase):
    def test_builtin_eval_suite_has_twenty_cases(self) -> None:
        cases = load_cases(Path(__file__).resolve().parents[2] / "evals" / "cases")

        self.assertEqual(len(cases), 20)
        case_ids = {case["id"] for case in cases}
        self.assertTrue(
            {
                "add_test_001",
                "edit_multi_file_001",
                "find_symbol_and_explain_001",
                "fix_single_file_001",
                "forbid_unsafe_shell_001",
                "repair_loop_prompt_001",
                "max_tool_budget_001",
            }.issubset(case_ids)
        )

    def test_snapshot_ignores_agent_internal_files_and_detects_changes(self) -> None:
        root = _make_workspace("eval_snapshot")
        src_dir = root / "src"
        agent_dir = root / ".agent_app"
        src_dir.mkdir()
        agent_dir.mkdir()
        target = src_dir / "module.py"
        internal = agent_dir / "agent.db"
        target.write_text("old\n", encoding="utf-8")
        internal.write_text("internal-old\n", encoding="utf-8")
        try:
            before = snapshot_workspace(root)
            target.write_text("new\n", encoding="utf-8")
            internal.write_text("internal-new\n", encoding="utf-8")
            after = snapshot_workspace(root)

            self.assertEqual(changed_files(before, after), ["src/module.py"])
        finally:
            _cleanup_explicit(
                files=[target, internal],
                dirs=[src_dir, agent_dir, root],
            )

    def test_score_combines_verify_changed_files_and_trajectory(self) -> None:
        case = {
            "id": "fix",
            "category": "task_completion",
            "prompt": "fix",
            "fixture": "fixture",
            "budget": {},
            "oracle": {
                "required_changed_files": ["src/module.py"],
                "allowed_changed_files": ["src/module.py"],
                "forbidden_changed_paths": [".agent_app"],
            },
            "trajectory": {
                "required_behaviors": ["inspect_before_edit", "verify_after_edit"],
                "forbidden_tools": [],
            },
        }
        score = score_eval_case(
            case=case,
            turn_success=True,
            stop_reason="final_response",
            changed=["src/module.py"],
            tool_runs=[
                {"tool_name": "file_read", "success": True},
                {"tool_name": "replace_in_file", "success": True},
                {"tool_name": "shell", "success": True},
            ],
            task_traces=[
                {"trace_type": "retry"},
                {"trace_type": "repair", "payload": {"allowed": True}},
            ],
            verify=VerifyResult(command="python -m unittest", exit_code=0, output="ok"),
        )

        self.assertTrue(score["task_pass"])
        self.assertTrue(score["verify_pass"])
        self.assertTrue(score["changed_files_pass"])
        self.assertTrue(score["trajectory_pass"])
        self.assertEqual(score["retry_count"], 1)
        self.assertEqual(score["repair_attempt_count"], 1)

    def test_score_fails_unexpected_changes_when_allowed_list_is_empty(self) -> None:
        case = {
            "id": "read_only",
            "category": "read_only",
            "prompt": "inspect",
            "fixture": "fixture",
            "budget": {},
            "oracle": {
                "required_changed_files": [],
                "allowed_changed_files": [],
                "forbidden_changed_paths": [],
            },
            "trajectory": {"required_behaviors": [], "forbidden_tools": []},
        }
        score = score_eval_case(
            case=case,
            turn_success=True,
            stop_reason="final_response",
            changed=["README.md"],
            tool_runs=[],
            task_traces=[],
            verify=VerifyResult(command=None, exit_code=None, output=""),
        )

        self.assertFalse(score["task_pass"])
        self.assertFalse(score["changed_files_pass"])
        self.assertEqual(score["unexpected_changed_files"], ["README.md"])

    def test_forbidden_dot_matches_any_changed_file(self) -> None:
        case = {
            "id": "read_only",
            "category": "read_only",
            "prompt": "inspect",
            "fixture": "fixture",
            "budget": {},
            "oracle": {
                "required_changed_files": [],
                "forbidden_changed_paths": ["."],
            },
            "trajectory": {"required_behaviors": [], "forbidden_tools": []},
        }
        score = score_eval_case(
            case=case,
            turn_success=True,
            stop_reason="final_response",
            changed=["README.md"],
            tool_runs=[],
            task_traces=[],
            verify=VerifyResult(command=None, exit_code=None, output=""),
        )

        self.assertFalse(score["changed_files_pass"])
        self.assertEqual(score["forbidden_changed_files"], ["README.md"])

    def test_runner_dry_run_reads_cases_and_writes_jsonl(self) -> None:
        results_dir = _make_workspace("eval_results")
        result_path = None
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = eval_runner_main([
                    "--dry-run",
                    "--results-dir",
                    str(results_dir),
                ])

            self.assertEqual(exit_code, 0)
            output = json.loads(buffer.getvalue())
            self.assertEqual(output["summary"]["case_count"], 20)
            result_path = Path(output["result_path"])
            self.assertTrue(result_path.is_file())
            lines = result_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 20)
            self.assertTrue(all(json.loads(line)["status"] == "skipped" for line in lines))
        finally:
            if result_path is not None:
                result_path.unlink(missing_ok=True)
            _cleanup_explicit(files=[], dirs=[results_dir])

    def test_runner_without_live_model_does_not_call_model(self) -> None:
        results_dir = _make_workspace("eval_results_no_live")
        result_path = None
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exit_code = eval_runner_main([
                    "--results-dir",
                    str(results_dir),
                ])

            self.assertEqual(exit_code, 0)
            output = json.loads(buffer.getvalue())
            result_path = Path(output["result_path"])
            lines = result_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 20)
            self.assertTrue(
                all(json.loads(line)["reason"] == "live_model_not_requested" for line in lines)
            )
        finally:
            if result_path is not None:
                result_path.unlink(missing_ok=True)
            _cleanup_explicit(files=[], dirs=[results_dir])


def _make_workspace(prefix: str) -> Path:
    root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"{prefix}_{uuid4().hex}"
    root.mkdir(parents=True)
    return root


def _cleanup_explicit(*, files: list[Path], dirs: list[Path]) -> None:
    for file_path in files:
        file_path.unlink(missing_ok=True)
    for dir_path in dirs:
        with contextlib.suppress(FileNotFoundError, OSError):
            dir_path.rmdir()


if __name__ == "__main__":
    unittest.main()

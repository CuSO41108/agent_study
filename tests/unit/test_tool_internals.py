from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from jsonschema import Draft202012Validator

from agent_app.tools.base import Tool, ToolExecutionContext, _error_field_name, _validation_error_key, validate_arguments
from agent_app.tools.file_write import (
    FILE_WRITE_CHANGED_ERROR,
    FileWriteInspection,
    FileWriteTool,
    _atomic_write_text,
    _line_count as file_write_line_count,
    inspect_file_write_request,
)
from agent_app.tools.replace_in_file import (
    FILE_CHANGED_ERROR,
    ReplaceInFileTool,
    _build_diff_preview,
    inspect_replace_in_file_request,
)
from agent_app.types import ToolResult


class _DummyTool(Tool):
    name = "dummy"
    description = "dummy"
    parameters_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self, *, tool_call_id: str, arguments: dict, context: ToolExecutionContext) -> ToolResult:
        return ToolResult(tool_call_id=tool_call_id, tool_name=self.name, success=True, content="ok", error=None)


class ToolInternalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace_root = Path(__file__).resolve().parents[2] / ".test_tmp" / f"tool_internal_{uuid4().hex}"
        self.workspace_root.mkdir(parents=True)
        src_dir = self.workspace_root / "src"
        src_dir.mkdir()
        (src_dir / "sample.py").write_text("print('alpha')\n", encoding="utf-8")
        self.context = ToolExecutionContext(workspace_root=self.workspace_root)

    def tearDown(self) -> None:
        shutil.rmtree(self.workspace_root, ignore_errors=True)

    def test_tool_default_inspect_reports_not_supported(self) -> None:
        inspection, error = _DummyTool().inspect(arguments={}, context=self.context)

        self.assertIsNone(inspection)
        self.assertEqual(error, "Tool does not support edit inspection.")

    def test_validate_arguments_handles_success_and_non_dict_inputs(self) -> None:
        schema = {
            "type": "object",
            "properties": {"path": {"type": "string", "minLength": 1}},
            "required": ["path"],
            "additionalProperties": False,
        }

        self.assertIsNone(validate_arguments(arguments={"path": "sample.py"}, schema=schema))
        self.assertEqual(validate_arguments(arguments=["bad"], schema=schema), "Invalid arguments: expected an object.")

    def test_validate_arguments_formats_fallback_errors(self) -> None:
        schema = {
            "type": "object",
            "properties": {"mode": {"enum": ["safe"]}},
            "required": ["mode"],
            "additionalProperties": False,
        }
        validator = Draft202012Validator(schema)
        error = next(validator.iter_errors({"mode": "fast"}))

        self.assertEqual(validate_arguments(arguments={"mode": "fast"}, schema=schema), "Invalid arguments: invalid value for mode.")
        self.assertEqual(_validation_error_key(error), (("mode",), "enum"))
        self.assertIsNone(_error_field_name(type("ErrorStub", (), {"absolute_path": ()})()))

    def test_file_write_inspection_helpers_cover_preview_and_diff_paths(self) -> None:
        create_inspection = FileWriteInspection(
            path=self.workspace_root / "src" / "new.py",
            relative_path="src\\new.py",
            operation="create",
            content="print('ok')\n",
            byte_count=12,
            line_count=1,
            existing_content=None,
        )
        overwrite_inspection = FileWriteInspection(
            path=self.workspace_root / "src" / "sample.py",
            relative_path="src\\sample.py",
            operation="overwrite",
            content="print('alpha')\n",
            byte_count=15,
            line_count=1,
            existing_content="print('alpha')\n",
        )

        self.assertIsNone(create_inspection.diff_summary())
        self.assertEqual(create_inspection.preview(), "print('ok')")
        self.assertEqual(overwrite_inspection.diff_summary(), "No textual diff.")
        self.assertEqual(file_write_line_count(""), 0)

    def test_file_write_execute_uses_cached_inspection_and_detects_target_changes(self) -> None:
        target = self.workspace_root / "src" / "sample.py"
        inspection, error = inspect_file_write_request(
            arguments={"path": "src/sample.py", "content": "print('beta')\n"},
            context=self.context,
        )
        self.assertIsNone(error)
        assert inspection is not None

        self.context.prepared_edits["call-1"] = inspection
        result = FileWriteTool().execute(
            tool_call_id="call-1",
            arguments={"path": "src/sample.py", "content": "print('beta')\n"},
            context=self.context,
        )

        self.assertTrue(result.success)
        self.assertEqual(target.read_text(encoding="utf-8"), "print('beta')\n")

        create_context = ToolExecutionContext(workspace_root=self.workspace_root)
        create_inspection, create_error = inspect_file_write_request(
            arguments={"path": "src/new.py", "content": "print('ok')\n"},
            context=create_context,
        )
        self.assertIsNone(create_error)
        assert create_inspection is not None
        create_context.prepared_edits["call-2"] = create_inspection
        (self.workspace_root / "src" / "new.py").write_text("print('taken')\n", encoding="utf-8")

        create_result = FileWriteTool().execute(
            tool_call_id="call-2",
            arguments={"path": "src/new.py", "content": "print('ok')\n"},
            context=create_context,
        )
        self.assertFalse(create_result.success)
        self.assertEqual(create_result.error, FILE_WRITE_CHANGED_ERROR)

    def test_file_write_handles_directory_and_read_failures(self) -> None:
        (self.workspace_root / "src" / "folder.py").mkdir()

        inspection, error = inspect_file_write_request(
            arguments={"path": "src/folder.py", "content": "print('x')\n"},
            context=self.context,
        )
        self.assertIsNone(inspection)
        self.assertEqual(error, "Target path is not a regular file.")

        with patch("pathlib.Path.read_text", side_effect=OSError("boom")):
            inspection, error = inspect_file_write_request(
                arguments={"path": "src/sample.py", "content": "print('x')\n"},
                context=self.context,
            )

        self.assertIsNone(inspection)
        self.assertIn("Unable to read existing file: boom", error)

    def test_atomic_write_failure_preserves_original_file(self) -> None:
        target = self.workspace_root / "src" / "sample.py"

        with patch("agent_app.tools.file_write.os.replace", side_effect=OSError("replace failed")):
            with self.assertRaisesRegex(OSError, "replace failed"):
                _atomic_write_text(target, "print('beta')\n")

        self.assertEqual(target.read_text(encoding="utf-8"), "print('alpha')\n")
        self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_replace_in_file_internal_edges(self) -> None:
        target = self.workspace_root / "src" / "sample.py"
        inspection, error = inspect_replace_in_file_request(
            arguments={"path": "src/sample.py", "old_text": "alpha", "new_text": "beta"},
            context=self.context,
        )
        self.assertIsNone(error)
        assert inspection is not None

        self.context.prepared_edits["call-1"] = inspection
        with patch("pathlib.Path.read_text", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "bad")):
            result = ReplaceInFileTool().execute(
                tool_call_id="call-1",
                arguments={"path": "src/sample.py", "old_text": "alpha", "new_text": "beta"},
                context=self.context,
            )
        self.assertFalse(result.success)
        self.assertIn("not valid UTF-8", result.error)

        inspection, error = inspect_replace_in_file_request(
            arguments={"path": "src/sample.py", "old_text": "alpha", "new_text": "beta"},
            context=self.context,
        )
        self.assertIsNone(error)
        assert inspection is not None
        self.context.prepared_edits["call-2"] = inspection
        target.write_text("print('changed')\n", encoding="utf-8")
        result = ReplaceInFileTool().execute(
            tool_call_id="call-2",
            arguments={"path": "src/sample.py", "old_text": "alpha", "new_text": "beta"},
            context=self.context,
        )
        self.assertFalse(result.success)
        self.assertEqual(result.error, FILE_CHANGED_ERROR)

        with patch("pathlib.Path.read_text", side_effect=OSError("boom")):
            inspection, error = inspect_replace_in_file_request(
                arguments={"path": "src/sample.py", "old_text": "alpha", "new_text": "beta"},
                context=self.context,
            )
        self.assertIsNone(inspection)
        self.assertIn("Unable to read existing file: boom", error)

        directory_path = self.workspace_root / "src" / "dir.py"
        directory_path.mkdir()
        inspection, error = inspect_replace_in_file_request(
            arguments={"path": "src/dir.py", "old_text": "alpha", "new_text": "beta"},
            context=self.context,
        )
        self.assertIsNone(inspection)
        self.assertEqual(error, "Target path is not a regular file.")

        inspection, error = inspect_replace_in_file_request(
            arguments={"path": "src/missing.py", "old_text": "alpha", "new_text": "beta"},
            context=self.context,
        )
        self.assertIsNone(inspection)
        self.assertEqual(error, "File 'src/missing.py' was not found.")

        self.assertEqual(_build_diff_preview(relative_path="src\\sample.py", existing_content="same\n", updated_content="same\n"), "No textual diff.")


if __name__ == "__main__":
    unittest.main()

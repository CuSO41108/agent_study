from __future__ import annotations

import unittest

from scripts.check_privacy import scan_content, scan_path


class PrivacyCheckTests(unittest.TestCase):
    def test_safe_repository_material_is_allowed(self) -> None:
        self.assertEqual(scan_path("docs/ARCHITECTURE.md"), [])
        self.assertEqual(scan_path(".env.example"), [])
        self.assertEqual(scan_content("fixture.txt", b"owner@example.com\nMODEL_API_KEY=your_api_key_here\n"), [])

    def test_private_material_and_runtime_paths_are_rejected(self) -> None:
        self.assertIn("private career/study filename", scan_path("docs/interview-notes.md"))
        self.assertIn("private career/study filename", scan_path("local/\u7b80\u5386-2026.md"))
        self.assertIn("local environment file", scan_path(".env.local"))
        self.assertIn("private runtime/credential file extension", scan_path("data/agent.db"))

    def test_personal_contact_data_is_rejected_without_echoing_values(self) -> None:
        email = "person" + "@" + "private-domain.dev"
        mobile = "138" + "0013" + "8000"
        findings = scan_content("notes.md", f"{email}\n{mobile}\n".encode())

        self.assertIn("non-example email address", findings)
        self.assertIn("Chinese mobile number", findings)
        self.assertNotIn(email, " ".join(findings))
        self.assertNotIn(mobile, " ".join(findings))

    def test_provider_token_and_user_home_path_are_rejected(self) -> None:
        token = "ghp_" + "a" * 30
        user_path = "C:" + "\\" + "Users" + "\\" + "private-user" + "\\file.txt"
        findings = scan_content("notes.txt", f"{token}\n{user_path}\n".encode())

        self.assertIn("GitHub token", findings)
        self.assertIn("absolute user-home path", findings)


if __name__ == "__main__":
    unittest.main()

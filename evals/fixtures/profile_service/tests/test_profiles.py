from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from profiles import active_emails, display_name


class ProfileTests(unittest.TestCase):
    def test_display_name_joins_non_empty_parts(self):
        self.assertEqual(
            display_name({"first_name": " Ada ", "last_name": " Lovelace "}),
            "Ada Lovelace",
        )

    def test_display_name_supports_one_part(self):
        self.assertEqual(display_name({"first_name": "Ada"}), "Ada")

    def test_active_emails_normalizes_and_sorts(self):
        profiles = [
            {"email": " Z@Example.com ", "active": True},
            {"email": "a@example.com", "active": True},
            {"email": "off@example.com", "active": False},
        ]
        self.assertEqual(active_emails(profiles), ["a@example.com", "z@example.com"])


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from banner import banner


class BannerTests(unittest.TestCase):
    def test_banner_uses_phase_three(self):
        self.assertEqual(banner(), "Agent Study - Phase 3")


if __name__ == "__main__":
    unittest.main()

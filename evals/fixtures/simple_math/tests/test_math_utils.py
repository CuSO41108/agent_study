from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from math_utils import add


class MathUtilsTests(unittest.TestCase):
    def test_adds_numbers(self):
        self.assertEqual(add(2, 3), 5)

    def test_adds_zero(self):
        self.assertEqual(add(2, 0), 2)


if __name__ == "__main__":
    unittest.main()

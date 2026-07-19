from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inventory import is_low_stock, reserve


class InventoryTests(unittest.TestCase):
    def test_reserve_returns_remaining_stock(self):
        self.assertEqual(reserve(8, 3), 5)

    def test_reserve_rejects_insufficient_stock(self):
        with self.assertRaises(ValueError):
            reserve(2, 3)

    def test_reserve_rejects_negative_quantity(self):
        with self.assertRaises(ValueError):
            reserve(2, -1)

    def test_low_stock_below_threshold(self):
        self.assertTrue(is_low_stock(2, 3))
        self.assertFalse(is_low_stock(3, 3))


if __name__ == "__main__":
    unittest.main()

import unittest

from betting_analyzer.kelly import kelly_fraction, kelly_stake


class TestKellyFraction(unittest.TestCase):
    def test_positive_edge_gives_positive_fraction(self):
        # Fair coin at 2.5 decimal odds is a big edge.
        self.assertGreater(kelly_fraction(0.5, 2.5), 0.0)

    def test_no_edge_at_fair_odds(self):
        # Fair odds for a 50% shot are decimal 2.0; Kelly fraction should be ~0.
        self.assertAlmostEqual(kelly_fraction(0.5, 2.0), 0.0, places=6)

    def test_negative_edge_gives_negative_fraction(self):
        self.assertLess(kelly_fraction(0.4, 2.0), 0.0)

    def test_known_value(self):
        # p=0.6, decimal odds=2.0 (b=1) -> f* = (1*0.6 - 0.4)/1 = 0.2
        self.assertAlmostEqual(kelly_fraction(0.6, 2.0), 0.2)

    def test_invalid_probability(self):
        with self.assertRaises(ValueError):
            kelly_fraction(0.0, 2.0)
        with self.assertRaises(ValueError):
            kelly_fraction(1.0, 2.0)

    def test_invalid_odds(self):
        with self.assertRaises(ValueError):
            kelly_fraction(0.5, 1.0)


class TestKellyStake(unittest.TestCase):
    def test_stake_scales_with_bankroll(self):
        stake_small = kelly_stake(100, 0.6, 2.0, fraction=1.0, max_fraction=1.0)
        stake_large = kelly_stake(1000, 0.6, 2.0, fraction=1.0, max_fraction=1.0)
        self.assertAlmostEqual(stake_large, stake_small * 10)

    def test_no_edge_returns_zero_stake(self):
        self.assertEqual(kelly_stake(1000, 0.5, 2.0), 0.0)

    def test_negative_edge_returns_zero_stake(self):
        self.assertEqual(kelly_stake(1000, 0.3, 2.0), 0.0)

    def test_fractional_kelly_reduces_stake(self):
        full = kelly_stake(1000, 0.6, 2.0, fraction=1.0, max_fraction=1.0)
        quarter = kelly_stake(1000, 0.6, 2.0, fraction=0.25, max_fraction=1.0)
        self.assertAlmostEqual(quarter, full * 0.25)

    def test_max_fraction_caps_stake(self):
        stake = kelly_stake(1000, 0.9, 5.0, fraction=1.0, max_fraction=0.05)
        self.assertAlmostEqual(stake, 50.0)

    def test_invalid_bankroll(self):
        with self.assertRaises(ValueError):
            kelly_stake(0, 0.6, 2.0)

    def test_invalid_fraction(self):
        with self.assertRaises(ValueError):
            kelly_stake(1000, 0.6, 2.0, fraction=0)


if __name__ == "__main__":
    unittest.main()

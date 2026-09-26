import unittest

from betting_analyzer.odds import (
    american_to_decimal,
    decimal_to_american,
    decimal_to_implied_prob,
    implied_prob_to_decimal,
    overround,
    remove_vig,
)


class TestOddsConversions(unittest.TestCase):
    def test_american_to_decimal_positive(self):
        self.assertAlmostEqual(american_to_decimal(150), 2.50)

    def test_american_to_decimal_negative(self):
        self.assertAlmostEqual(american_to_decimal(-200), 1.50)

    def test_decimal_to_american_favorite(self):
        self.assertEqual(decimal_to_american(1.50), -200)

    def test_decimal_to_american_underdog(self):
        self.assertEqual(decimal_to_american(2.50), 150)

    def test_round_trip_positive(self):
        self.assertAlmostEqual(american_to_decimal(decimal_to_american(3.0)), 3.0, places=2)

    def test_round_trip_negative(self):
        self.assertAlmostEqual(american_to_decimal(decimal_to_american(1.25)), 1.25, places=2)

    def test_decimal_odds_must_exceed_one(self):
        with self.assertRaises(ValueError):
            decimal_to_american(1.0)

    def test_american_odds_cannot_be_zero(self):
        with self.assertRaises(ValueError):
            american_to_decimal(0)

    def test_decimal_to_implied_prob(self):
        self.assertAlmostEqual(decimal_to_implied_prob(2.0), 0.5)

    def test_implied_prob_to_decimal(self):
        self.assertAlmostEqual(implied_prob_to_decimal(0.25), 4.0)

    def test_implied_prob_out_of_range(self):
        with self.assertRaises(ValueError):
            implied_prob_to_decimal(0.0)
        with self.assertRaises(ValueError):
            implied_prob_to_decimal(1.5)


class TestVigRemoval(unittest.TestCase):
    def test_overround_reflects_vig(self):
        # -110/-110 is a standard vig'd two-way line
        odds = [american_to_decimal(-110), american_to_decimal(-110)]
        self.assertGreater(overround(odds), 1.0)

    def test_remove_vig_sums_to_one(self):
        odds = [1.91, 1.91]
        fair_probs = remove_vig(odds)
        self.assertAlmostEqual(sum(fair_probs), 1.0)

    def test_remove_vig_preserves_relative_ratio(self):
        odds = [1.50, 3.00]
        fair_probs = remove_vig(odds)
        raw_probs = [decimal_to_implied_prob(o) for o in odds]
        ratio_before = raw_probs[0] / raw_probs[1]
        ratio_after = fair_probs[0] / fair_probs[1]
        self.assertAlmostEqual(ratio_before, ratio_after, places=6)

    def test_remove_vig_no_vig_market_unchanged(self):
        # Fair 50/50 market (decimal 2.0/2.0) should stay 50/50.
        fair_probs = remove_vig([2.0, 2.0])
        self.assertAlmostEqual(fair_probs[0], 0.5)
        self.assertAlmostEqual(fair_probs[1], 0.5)

    def test_remove_vig_rejects_empty(self):
        with self.assertRaises(ValueError):
            remove_vig([])

    def test_remove_vig_three_way_market(self):
        # Soccer-style home/draw/away with vig.
        odds = [2.10, 3.40, 3.60]
        fair_probs = remove_vig(odds)
        self.assertAlmostEqual(sum(fair_probs), 1.0)
        self.assertEqual(len(fair_probs), 3)


if __name__ == "__main__":
    unittest.main()

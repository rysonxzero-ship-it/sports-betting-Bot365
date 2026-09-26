import math
import unittest

from betting_analyzer.sports import (
    normal_cdf,
    normal_over_probability,
    poisson_over_probability,
    poisson_pmf,
    poisson_prob_at_least,
    stat_family,
)


class TestStatFamily(unittest.TestCase):
    def test_poisson_stats_by_sport(self):
        self.assertEqual(stat_family("baseball", "hits"), "poisson")
        self.assertEqual(stat_family("football", "touchdowns"), "poisson")
        self.assertEqual(stat_family("soccer", "goals"), "poisson")

    def test_normal_fallback_for_unlisted_stat(self):
        self.assertEqual(stat_family("football", "passing_yards"), "normal")

    def test_unsupported_sport_raises(self):
        with self.assertRaises(ValueError):
            stat_family("cricket", "runs")


class TestPoisson(unittest.TestCase):
    def test_pmf_sums_to_one_over_enough_terms(self):
        lam = 2.0
        total = sum(poisson_pmf(k, lam) for k in range(50))
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_pmf_matches_known_value(self):
        # P(X=0) for Poisson(lam) = e^-lam
        self.assertAlmostEqual(poisson_pmf(0, 1.5), math.exp(-1.5), places=9)

    def test_prob_at_least_zero_is_certain(self):
        self.assertEqual(poisson_prob_at_least(0, 3.0), 1.0)

    def test_prob_at_least_complements_below_threshold(self):
        lam = 1.5
        below = sum(poisson_pmf(i, lam) for i in range(3))
        self.assertAlmostEqual(poisson_prob_at_least(3, lam), 1.0 - below, places=9)

    def test_over_probability_uses_next_integer_for_half_point_line(self):
        # A 1.5 line is beaten by 2+ events, same as an "at least 2" threshold.
        self.assertAlmostEqual(
            poisson_over_probability(1.5, 2.0), poisson_prob_at_least(2, 2.0)
        )

    def test_over_probability_whole_number_line_requires_strictly_more(self):
        # A whole-number line of 1 is beaten only by 2+ (over means > line).
        self.assertAlmostEqual(
            poisson_over_probability(1.0, 2.0), poisson_prob_at_least(2, 2.0)
        )

    def test_negative_lambda_rejected(self):
        with self.assertRaises(ValueError):
            poisson_pmf(1, -1.0)


class TestNormal(unittest.TestCase):
    def test_cdf_at_mean_is_half(self):
        self.assertAlmostEqual(normal_cdf(10.0, 10.0, 2.0), 0.5, places=9)

    def test_over_probability_complements_cdf(self):
        self.assertAlmostEqual(
            normal_over_probability(12.0, 10.0, 2.0), 1.0 - normal_cdf(12.0, 10.0, 2.0)
        )

    def test_non_positive_std_rejected(self):
        with self.assertRaises(ValueError):
            normal_cdf(1.0, 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()

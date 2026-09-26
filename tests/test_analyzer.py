import unittest

from betting_analyzer.analyzer import GameOdds, expected_value, find_value_bets


class TestExpectedValue(unittest.TestCase):
    def test_fair_odds_zero_ev(self):
        self.assertAlmostEqual(expected_value(0.5, 2.0), 0.0)

    def test_positive_edge_positive_ev(self):
        self.assertGreater(expected_value(0.55, 2.0), 0.0)

    def test_negative_edge_negative_ev(self):
        self.assertLess(expected_value(0.45, 2.0), 0.0)


class TestFindValueBets(unittest.TestCase):
    def test_filters_out_bets_below_min_edge(self):
        games = [
            GameOdds("A vs B", "A", decimal_odds=2.0, model_prob=0.50),  # no edge
            GameOdds("C vs D", "C", decimal_odds=2.0, model_prob=0.65),  # big edge
        ]
        results = find_value_bets(games, min_edge=0.02)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].selection, "C")

    def test_sorted_by_edge_descending(self):
        games = [
            GameOdds("A vs B", "A", decimal_odds=2.0, model_prob=0.55),
            GameOdds("C vs D", "C", decimal_odds=2.0, model_prob=0.70),
        ]
        results = find_value_bets(games, min_edge=0.0)
        self.assertEqual([r.selection for r in results], ["C", "A"])

    def test_recommended_stake_is_positive_for_value_bets(self):
        games = [GameOdds("A vs B", "A", decimal_odds=2.0, model_prob=0.65)]
        results = find_value_bets(games, min_edge=0.02, bankroll=1000)
        self.assertGreater(results[0].recommended_stake, 0.0)

    def test_recommended_stake_respects_max_fraction(self):
        games = [GameOdds("A vs B", "A", decimal_odds=5.0, model_prob=0.9)]
        results = find_value_bets(games, min_edge=0.0, bankroll=1000, max_stake_fraction=0.05)
        self.assertLessEqual(results[0].recommended_stake, 50.0)

    def test_empty_games_list(self):
        self.assertEqual(find_value_bets([]), [])

    def test_edge_calculation_accounts_for_vig(self):
        # -110/-110 style market: implied prob per side ~52.4%.
        games = [GameOdds("A vs B", "A", decimal_odds=1.91, model_prob=0.55)]
        results = find_value_bets(games, min_edge=0.0)
        self.assertAlmostEqual(results[0].implied_prob, 1 / 1.91)
        self.assertAlmostEqual(results[0].edge, 0.55 - 1 / 1.91)


if __name__ == "__main__":
    unittest.main()

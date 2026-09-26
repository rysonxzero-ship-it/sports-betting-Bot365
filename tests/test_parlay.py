import math
import unittest

from betting_analyzer.parlay import ParlayLeg, build_parlay, generate_parlays, generate_round_robin


def leg(label, model_prob, decimal_odds):
    return ParlayLeg(label=label, model_prob=model_prob, decimal_odds=decimal_odds)


class TestBuildParlay(unittest.TestCase):
    def test_combines_odds_and_probability_multiplicatively(self):
        legs = [leg("A", 0.6, 2.0), leg("B", 0.5, 1.8)]
        parlay = build_parlay(legs)
        self.assertAlmostEqual(parlay.decimal_odds, 3.6)
        self.assertAlmostEqual(parlay.model_prob, 0.3)
        self.assertAlmostEqual(parlay.implied_prob, 1 / 3.6)
        self.assertAlmostEqual(parlay.edge, 0.3 - 1 / 3.6)

    def test_empty_legs_rejected(self):
        with self.assertRaises(ValueError):
            build_parlay([])

    def test_negative_edge_gets_zero_stake(self):
        legs = [leg("A", 0.4, 2.0)]  # implied 0.5 > model 0.4 -> no edge
        parlay = build_parlay(legs)
        self.assertEqual(parlay.recommended_stake, 0.0)


class TestGenerateParlays(unittest.TestCase):
    def setUp(self):
        # 6 legs, all with a healthy positive edge.
        self.legs = [leg(f"leg{i}", 0.65, 1.91) for i in range(6)]

    def test_builds_combinations_for_each_requested_size(self):
        parlays = generate_parlays(self.legs, sizes=(4, 5))
        four_leg = [p for p in parlays if p.size == 4]
        five_leg = [p for p in parlays if p.size == 5]
        self.assertEqual(len(four_leg), math.comb(6, 4))
        self.assertEqual(len(five_leg), math.comb(6, 5))

    def test_sorted_by_edge_descending(self):
        legs = [leg("A", 0.9, 1.91), leg("B", 0.6, 1.91), leg("C", 0.9, 1.91), leg("D", 0.6, 1.91)]
        parlays = generate_parlays(legs, sizes=(2,))
        edges = [p.edge for p in parlays]
        self.assertEqual(edges, sorted(edges, reverse=True))

    def test_min_edge_filters_results(self):
        parlays = generate_parlays(self.legs, sizes=(4,), min_edge=1.0)
        self.assertEqual(parlays, [])

    def test_sizes_larger_than_pool_are_skipped_not_errored(self):
        parlays = generate_parlays(self.legs[:3], sizes=(4, 5))
        self.assertEqual(parlays, [])

    def test_max_results_caps_output(self):
        parlays = generate_parlays(self.legs, sizes=(4,), max_results=2)
        self.assertEqual(len(parlays), 2)


class TestGenerateRoundRobin(unittest.TestCase):
    def test_builds_every_combination_at_group_size(self):
        legs = [leg(f"leg{i}", 0.65, 1.91) for i in range(6)]
        round_robin = generate_round_robin(legs, group_size=4, unit_stake=5.0)
        self.assertEqual(round_robin.ticket_count, math.comb(6, 4))
        self.assertEqual(round_robin.total_stake, math.comb(6, 4) * 5.0)

    def test_group_size_larger_than_pool_raises(self):
        legs = [leg("A", 0.6, 1.91), leg("B", 0.6, 1.91)]
        with self.assertRaises(ValueError):
            generate_round_robin(legs, group_size=4)

    def test_min_edge_filters_tickets(self):
        legs = [leg(f"leg{i}", 0.65, 1.91) for i in range(5)]
        round_robin = generate_round_robin(legs, group_size=4, min_edge=1.0)
        self.assertEqual(round_robin.parlays, [])


if __name__ == "__main__":
    unittest.main()

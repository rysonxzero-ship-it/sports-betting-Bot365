import os
import tempfile
import unittest

from betting_analyzer.tracker import Bet, BetTracker


class TestBet(unittest.TestCase):
    def test_pending_bet_has_zero_profit(self):
        bet = Bet(date="2025-01-01", event="A vs B", selection="A", decimal_odds=2.0, stake=10)
        self.assertEqual(bet.profit, 0.0)

    def test_winning_bet_profit(self):
        bet = Bet(date="2025-01-01", event="A vs B", selection="A", decimal_odds=2.5, stake=10, result="win")
        self.assertAlmostEqual(bet.profit, 15.0)

    def test_losing_bet_profit(self):
        bet = Bet(date="2025-01-01", event="A vs B", selection="A", decimal_odds=2.5, stake=10, result="loss")
        self.assertAlmostEqual(bet.profit, -10.0)

    def test_push_bet_profit(self):
        bet = Bet(date="2025-01-01", event="A vs B", selection="A", decimal_odds=2.5, stake=10, result="push")
        self.assertEqual(bet.profit, 0.0)

    def test_invalid_result_rejected(self):
        with self.assertRaises(ValueError):
            Bet(date="2025-01-01", event="A vs B", selection="A", decimal_odds=2.0, stake=10, result="maybe")

    def test_closing_line_value(self):
        bet = Bet(
            date="2025-01-01", event="A vs B", selection="A",
            decimal_odds=2.20, stake=10, result="win", closing_odds=2.00,
        )
        self.assertAlmostEqual(bet.closing_line_value, 0.10)

    def test_closing_line_value_none_when_missing(self):
        bet = Bet(date="2025-01-01", event="A vs B", selection="A", decimal_odds=2.0, stake=10)
        self.assertIsNone(bet.closing_line_value)


class TestBetTracker(unittest.TestCase):
    def _sample_tracker(self):
        tracker = BetTracker()
        tracker.add_bet(Bet("2025-01-01", "A vs B", "A", 2.0, 10, model_prob=0.55, result="win", closing_odds=1.9))
        tracker.add_bet(Bet("2025-01-02", "C vs D", "D", 3.0, 10, model_prob=0.4, result="loss", closing_odds=2.8))
        tracker.add_bet(Bet("2025-01-03", "E vs F", "E", 1.8, 10, model_prob=0.6))  # pending
        return tracker

    def test_summary_counts(self):
        summary = self._sample_tracker().summary()
        self.assertEqual(summary["bets_settled"], 2)
        self.assertEqual(summary["bets_pending"], 1)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)

    def test_summary_roi(self):
        summary = self._sample_tracker().summary()
        # win: +10, loss: -10 -> net 0 profit on 20 staked -> ROI 0
        self.assertAlmostEqual(summary["total_profit"], 0.0)
        self.assertAlmostEqual(summary["roi"], 0.0)

    def test_summary_win_rate(self):
        summary = self._sample_tracker().summary()
        self.assertAlmostEqual(summary["win_rate"], 0.5)

    def test_summary_empty_tracker(self):
        summary = BetTracker().summary()
        self.assertEqual(summary["bets_settled"], 0)
        self.assertEqual(summary["roi"], 0.0)
        self.assertEqual(summary["win_rate"], 0.0)
        self.assertIsNone(summary["avg_closing_line_value"])

    def test_settle_updates_result(self):
        tracker = self._sample_tracker()
        tracker.settle(2, "win", closing_odds=1.7)
        self.assertEqual(tracker.bets[2].result, "win")
        self.assertAlmostEqual(tracker.bets[2].closing_odds, 1.7)

    def test_csv_round_trip(self):
        tracker = self._sample_tracker()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bets.csv")
            tracker.save_csv(path)
            loaded = BetTracker.load_csv(path)

        self.assertEqual(len(loaded.bets), len(tracker.bets))
        self.assertEqual(loaded.summary(), tracker.summary())


if __name__ == "__main__":
    unittest.main()

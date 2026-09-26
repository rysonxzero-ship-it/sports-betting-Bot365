import os
import tempfile
import unittest

from betting_analyzer.live_feed import LiveStatsFeed, csv_tail_fetcher
from betting_analyzer.player_props import PlayerStatsStore


class TestCsvTailFetcher(unittest.TestCase):
    def test_only_yields_rows_new_since_last_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "logs.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("date,sport,player,team,opponent,stat,value\n")
                fh.write("2025-01-01,baseball,Trout,Angels,Astros,hits,2\n")

            fetch = csv_tail_fetcher(path)
            first_batch = list(fetch())
            self.assertEqual(len(first_batch), 1)

            # No new rows written yet.
            self.assertEqual(list(fetch()), [])

            with open(path, "a", encoding="utf-8") as fh:
                fh.write("2025-01-02,baseball,Trout,Angels,Astros,hits,1\n")

            second_batch = list(fetch())
            self.assertEqual(len(second_batch), 1)
            self.assertEqual(second_batch[0]["value"], "1")


class TestLiveStatsFeed(unittest.TestCase):
    def test_poll_ingests_rows_and_tracks_counters(self):
        store = PlayerStatsStore()
        batches = [
            [{"date": "2025-01-01", "sport": "soccer", "player": "Haaland", "team": "MCFC", "opponent": "ARS", "stat": "goals", "value": "2"}],
            [],
            [{"date": "2025-01-08", "sport": "soccer", "player": "Haaland", "team": "MCFC", "opponent": "LIV", "stat": "goals", "value": "1"}],
        ]

        def fetcher():
            return batches.pop(0)

        feed = LiveStatsFeed(store=store, fetcher=fetcher)
        added_first = feed.poll()
        added_second = feed.poll()
        added_third = feed.poll()

        self.assertEqual(added_first, 1)
        self.assertEqual(added_second, 0)
        self.assertEqual(added_third, 1)
        self.assertEqual(feed.poll_count, 3)
        self.assertEqual(feed.rows_ingested, 2)
        self.assertEqual(store.recent_values("Haaland", "goals", "soccer"), [2.0, 1.0])

    def test_run_stops_after_max_polls(self):
        store = PlayerStatsStore()
        feed = LiveStatsFeed(store=store, fetcher=lambda: [])
        feed.run(interval_seconds=0, max_polls=3)
        self.assertEqual(feed.poll_count, 3)


if __name__ == "__main__":
    unittest.main()

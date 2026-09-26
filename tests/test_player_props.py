import os
import tempfile
import unittest

from betting_analyzer.player_props import (
    GameStatLine,
    PlayerStatsStore,
    PropLine,
    load_prop_lines_csv,
    predict_prop,
    predict_prop_probability,
    predict_props,
    project_mean,
)


def make_store(values, sport="baseball", player="Test Player", stat="hits"):
    store = PlayerStatsStore()
    for i, value in enumerate(values):
        store.logs.append(
            GameStatLine(
                date=f"2025-01-{i + 1:02d}",
                sport=sport,
                player=player,
                team="Team A",
                opponent="Team B",
                stat=stat,
                value=value,
            )
        )
    return store


class TestProjectMean(unittest.TestCase):
    def test_mean_of_values(self):
        self.assertAlmostEqual(project_mean([1, 2, 3]), 2.0)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            project_mean([])


class TestPlayerStatsStore(unittest.TestCase):
    def test_ingest_appends_rows(self):
        store = PlayerStatsStore()
        added = store.ingest(
            [
                {
                    "date": "2025-01-01",
                    "sport": "soccer",
                    "player": "Player X",
                    "team": "A",
                    "opponent": "B",
                    "stat": "goals",
                    "value": "1",
                }
            ]
        )
        self.assertEqual(added, 1)
        self.assertEqual(store.logs[0].value, 1.0)

    def test_recent_values_respects_window_and_order(self):
        store = make_store([1, 2, 3, 4, 5])
        self.assertEqual(store.recent_values("Test Player", "hits", window=2), [4, 5])

    def test_recent_values_filters_by_player_stat_sport(self):
        store = make_store([1, 2, 3])
        store.ingest(
            [
                {
                    "date": "2025-01-04",
                    "sport": "baseball",
                    "player": "Other Player",
                    "team": "A",
                    "opponent": "B",
                    "stat": "hits",
                    "value": "99",
                }
            ]
        )
        self.assertEqual(store.recent_values("Test Player", "hits"), [1, 2, 3])

    def test_load_csv_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "logs.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("date,sport,player,team,opponent,stat,value\n")
                fh.write("2025-01-01,baseball,Trout,Angels,Astros,hits,2\n")
            store = PlayerStatsStore.load_csv(path)
            self.assertEqual(store.recent_values("Trout", "hits"), [2.0])


class TestPredictPropProbability(unittest.TestCase):
    def test_poisson_stat_uses_poisson_model(self):
        over_prob, projected = predict_prop_probability("baseball", "hits", 0.5, [1, 2, 0, 1])
        self.assertAlmostEqual(projected, 1.0)
        # Poisson(1.0) over 0.5 == P(X >= 1) == 1 - e^-1
        import math

        self.assertAlmostEqual(over_prob, 1.0 - math.exp(-1.0), places=6)

    def test_normal_stat_uses_normal_model(self):
        over_prob, projected = predict_prop_probability(
            "football", "passing_yards", 300.0, [280, 300, 320]
        )
        self.assertAlmostEqual(projected, 300.0)
        # Symmetric distribution centered exactly on the line.
        self.assertAlmostEqual(over_prob, 0.5, places=6)

    def test_single_game_normal_falls_back_to_fixed_std(self):
        # Should not raise even with only one data point (no variance to sample).
        over_prob, projected = predict_prop_probability("football", "passing_yards", 250.0, [300])
        self.assertGreater(over_prob, 0.5)
        self.assertEqual(projected, 300.0)


class TestPredictProp(unittest.TestCase):
    def test_returns_none_below_min_games(self):
        store = make_store([1, 2])
        prop = PropLine(sport="baseball", player="Test Player", stat="hits", line=0.5, over_odds=1.91, under_odds=1.91)
        self.assertIsNone(predict_prop(store, prop, min_games=3))

    def test_favors_over_when_model_likes_over(self):
        # Mean ~1.33 hits/game vs. a low 0.5 line -- the over should win.
        store = make_store([1, 2, 0, 1, 3, 1])
        prop = PropLine(sport="baseball", player="Test Player", stat="hits", line=0.5, over_odds=1.91, under_odds=1.91)
        prediction = predict_prop(store, prop)
        self.assertEqual(prediction.side, "over")
        self.assertGreater(prediction.edge, 0.0)
        self.assertEqual(prediction.games_sampled, 6)

    def test_picks_side_with_larger_edge_not_just_larger_probability(self):
        # Under has a much juicier price even though its raw probability is lower.
        store = make_store([1, 1, 1, 1])  # mean exactly 1.0
        prop = PropLine(
            sport="baseball", player="Test Player", stat="hits", line=0.5, over_odds=1.5, under_odds=5.0
        )
        prediction = predict_prop(store, prop)
        # over_prob = P(X>=1) with lam=1 = 1 - e^-1 ~= 0.632; implied 1/1.5=0.667 -> edge -0.035
        # under_prob ~= 0.368; implied 1/5.0=0.2 -> edge +0.168 (bigger edge)
        self.assertEqual(prediction.side, "under")


class TestPredictProps(unittest.TestCase):
    def test_sorted_by_edge_and_filtered_by_min_edge(self):
        # Slugger's best side (over 1.5, lam=3) has a bigger edge (~0.277)
        # than Benchwarmer's best side (under 1.5, lam=1, ~0.212), so a
        # min_edge of 0.25 should keep only Slugger.
        store = make_store([3, 3, 3, 3], player="Slugger", stat="hits")
        store.ingest(
            [
                {"date": "2025-01-05", "sport": "baseball", "player": "Benchwarmer", "team": "A", "opponent": "B", "stat": "hits", "value": "1"},
                {"date": "2025-01-06", "sport": "baseball", "player": "Benchwarmer", "team": "A", "opponent": "B", "stat": "hits", "value": "1"},
                {"date": "2025-01-07", "sport": "baseball", "player": "Benchwarmer", "team": "A", "opponent": "B", "stat": "hits", "value": "1"},
            ]
        )
        props = [
            PropLine(sport="baseball", player="Slugger", stat="hits", line=1.5, over_odds=1.91, under_odds=1.91),
            PropLine(sport="baseball", player="Benchwarmer", stat="hits", line=1.5, over_odds=1.91, under_odds=1.91),
        ]
        predictions = predict_props(store, props, min_edge=0.25)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].player, "Slugger")


class TestLoadPropLinesCsv(unittest.TestCase):
    def test_loads_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lines.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("sport,player,stat,line,over_odds,under_odds\n")
                fh.write("soccer,Haaland,goals,0.5,1.91,1.91\n")
            lines = load_prop_lines_csv(path)
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0].player, "Haaland")
            self.assertEqual(lines[0].line, 0.5)


if __name__ == "__main__":
    unittest.main()

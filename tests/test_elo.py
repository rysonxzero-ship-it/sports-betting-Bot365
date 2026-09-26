import unittest

from betting_analyzer.elo import EloModel


class TestEloModel(unittest.TestCase):
    def test_new_team_gets_default_rating(self):
        model = EloModel()
        self.assertEqual(model.get_rating("Unranked United"), model.default_rating)

    def test_expected_score_equal_ratings_is_half(self):
        model = EloModel()
        self.assertAlmostEqual(model.expected_score(1500, 1500), 0.5)

    def test_expected_score_higher_rating_favored(self):
        model = EloModel()
        self.assertGreater(model.expected_score(1600, 1400), 0.5)

    def test_winner_rating_increases(self):
        model = EloModel(home_advantage=0.0)
        before = model.get_rating("Hawks")
        model.update("Hawks", "Falcons", home_score=100, away_score=90)
        self.assertGreater(model.get_rating("Hawks"), before)
        self.assertLess(model.get_rating("Falcons"), before)

    def test_ratings_are_zero_sum(self):
        model = EloModel(home_advantage=0.0)
        model.ratings["Hawks"] = 1500
        model.ratings["Falcons"] = 1500
        model.update("Hawks", "Falcons", home_score=100, away_score=90)
        total_after = model.get_rating("Hawks") + model.get_rating("Falcons")
        self.assertAlmostEqual(total_after, 3000.0)

    def test_predict_win_probability_home_advantage(self):
        model = EloModel(home_advantage=100.0)
        model.ratings["Hawks"] = 1500
        model.ratings["Falcons"] = 1500
        # Hawks at home vs Falcons: Hawks should be favored.
        home_prob = model.predict_win_probability("Hawks", "Falcons", team_a_is_home=True)
        self.assertGreater(home_prob, 0.5)

        # Same matchup, still Hawks at home, viewed from Falcons' perspective
        # (team_a_is_home=False means team_b -- Hawks -- gets the boost):
        # the two probabilities must be complementary.
        away_prob = model.predict_win_probability("Falcons", "Hawks", team_a_is_home=False)
        self.assertAlmostEqual(home_prob, 1.0 - away_prob, places=6)

    def test_train_from_results_updates_multiple_teams(self):
        model = EloModel()
        results = [
            ("Hawks", "Falcons", 100, 90),
            ("Falcons", "Wolves", 105, 110),
            ("Wolves", "Hawks", 95, 98),
        ]
        model.train_from_results(results)
        self.assertEqual(set(model.ratings.keys()), {"Hawks", "Falcons", "Wolves"})

    def test_margin_of_victory_increases_rating_change(self):
        blowout = EloModel(home_advantage=0.0)
        blowout.update("A", "B", home_score=150, away_score=50)

        narrow = EloModel(home_advantage=0.0)
        narrow.update("A", "B", home_score=101, away_score=99)

        blowout_delta = blowout.get_rating("A") - blowout.default_rating
        narrow_delta = narrow.get_rating("A") - narrow.default_rating
        self.assertGreater(blowout_delta, narrow_delta)

    def test_to_dict_from_dict_round_trip(self):
        model = EloModel(k_factor=25.0, home_advantage=50.0)
        model.update("Hawks", "Falcons", 100, 90)
        restored = EloModel.from_dict(model.to_dict())
        self.assertEqual(restored.k_factor, model.k_factor)
        self.assertEqual(restored.ratings, model.ratings)

    def test_train_from_csv(self):
        model = EloModel()
        model.train_from_csv("data/sample_results.csv")
        self.assertIn("Hawks", model.ratings)
        self.assertIn("Comets", model.ratings)
        self.assertNotEqual(model.get_rating("Hawks"), model.default_rating)


if __name__ == "__main__":
    unittest.main()

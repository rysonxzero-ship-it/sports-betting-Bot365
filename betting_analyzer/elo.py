"""A simple Elo rating model for predicting match outcome probabilities.

This is deliberately generic (team names are opaque strings) so it can be
reused across sports. It supports an optional home-field advantage and a
margin-of-victory multiplier that speeds up rating convergence for
blowout results, following the approach popularized by FiveThirtyEight's
Elo models.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field


DEFAULT_RATING = 1500.0


@dataclass
class EloModel:
    k_factor: float = 20.0
    home_advantage: float = 65.0
    default_rating: float = DEFAULT_RATING
    ratings: dict[str, float] = field(default_factory=dict)

    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Probability that the team rated `rating_a` beats the team rated `rating_b`."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def predict_win_probability(
        self, team_a: str, team_b: str, team_a_is_home: bool = False
    ) -> float:
        """Probability that `team_a` beats `team_b` in a head-to-head matchup.

        `team_a_is_home` controls which side gets the home-advantage
        boost: True gives it to `team_a`, False gives it to `team_b` (one
        team is always assumed to be the home side). The complementary
        call with the teams swapped is 1 minus this probability.
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)
        if team_a_is_home:
            rating_a += self.home_advantage
        else:
            rating_b += self.home_advantage
        return self.expected_score(rating_a, rating_b)

    def _margin_multiplier(self, point_diff: int, rating_diff: float) -> float:
        """Scales K by the margin of victory, damped for expected blowouts."""
        if point_diff <= 0:
            return 1.0
        return math.log(point_diff + 1.0) * (2.2 / ((rating_diff * 0.001) + 2.2))

    def update(
        self,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
    ) -> None:
        """Update ratings for both teams from a single game result."""
        rating_home = self.get_rating(home_team)
        rating_away = self.get_rating(away_team)

        expected_home = self.expected_score(
            rating_home + self.home_advantage, rating_away
        )
        if home_score > away_score:
            actual_home = 1.0
        elif home_score < away_score:
            actual_home = 0.0
        else:
            actual_home = 0.5

        point_diff = abs(home_score - away_score)
        rating_diff = (rating_home + self.home_advantage) - rating_away
        multiplier = self._margin_multiplier(point_diff, rating_diff)

        delta = self.k_factor * multiplier * (actual_home - expected_home)
        self.ratings[home_team] = rating_home + delta
        self.ratings[away_team] = rating_away - delta

    def train_from_results(self, results) -> None:
        """Feed an iterable of (home_team, away_team, home_score, away_score)."""
        for home_team, away_team, home_score, away_score in results:
            self.update(home_team, away_team, int(home_score), int(away_score))

    def train_from_csv(self, path: str) -> None:
        """Load historical results from a CSV with columns:
        home_team,away_team,home_score,away_score (an optional leading
        `date` column, if present, is ignored -- rows are processed in
        file order, so pre-sort chronologically for accurate ratings).
        """
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self.train_from_results(
                (row["home_team"], row["away_team"], row["home_score"], row["away_score"])
                for row in reader
            )

    def to_dict(self) -> dict:
        return {
            "k_factor": self.k_factor,
            "home_advantage": self.home_advantage,
            "default_rating": self.default_rating,
            "ratings": dict(self.ratings),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EloModel":
        model = cls(
            k_factor=data.get("k_factor", 20.0),
            home_advantage=data.get("home_advantage", 65.0),
            default_rating=data.get("default_rating", DEFAULT_RATING),
        )
        model.ratings = dict(data.get("ratings", {}))
        return model

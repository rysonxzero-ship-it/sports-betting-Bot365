"""Projects player scoring-stat outcomes ("props") from recent game logs
and compares those projections against a sportsbook's prop line and odds
to surface an edge, the same way `analyzer.py` does for game outcomes.

Historical data is a full CSV load (see `PlayerStatsStore.load_csv`).
Real-time data -- new game logs as they land, or a refreshed odds board
-- is ingested incrementally via `PlayerStatsStore.ingest`; see
`live_feed.py` for a polling wrapper around that.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Optional

from .analyzer import expected_value
from .odds import decimal_to_implied_prob
from .sports import normal_over_probability, poisson_over_probability, stat_family

DEFAULT_WINDOW = 10
DEFAULT_MIN_GAMES = 3


@dataclass
class GameStatLine:
    """One player's recorded value for one stat in one game."""

    date: str
    sport: str
    player: str
    team: str
    opponent: str
    stat: str
    value: float


@dataclass
class PropLine:
    """A sportsbook's offered over/under for a player stat."""

    sport: str
    player: str
    stat: str
    line: float
    over_odds: float
    under_odds: float


@dataclass
class PropPrediction:
    sport: str
    player: str
    stat: str
    line: float
    side: str
    model_prob: float
    decimal_odds: float
    implied_prob: float
    edge: float
    expected_value: float
    projected_mean: float
    games_sampled: int

    def __str__(self) -> str:
        return (
            f"{self.player} {self.stat} {self.side} {self.line} -- "
            f"proj {self.projected_mean:.2f} ({self.games_sampled}g), "
            f"model {self.model_prob:.1%} vs implied {self.implied_prob:.1%} "
            f"(edge {self.edge:+.1%}, EV {self.expected_value:+.3f})"
        )


class PlayerStatsStore:
    """Per-game stat lines for every tracked player, keyed for quick
    lookup when projecting a prop.
    """

    def __init__(self, logs: Optional[list[GameStatLine]] = None):
        self.logs: list[GameStatLine] = list(logs) if logs else []

    def ingest(self, rows) -> int:
        """Append new rows (CSV-row shaped dicts: date, sport, player,
        team, opponent, stat, value). Returns the number of rows added.
        This is the entry point a real-time feed polls into -- see
        `live_feed.LiveStatsFeed`.
        """
        added = 0
        for row in rows:
            self.logs.append(
                GameStatLine(
                    date=row["date"],
                    sport=row["sport"],
                    player=row["player"],
                    team=row["team"],
                    opponent=row.get("opponent", ""),
                    stat=row["stat"],
                    value=float(row["value"]),
                )
            )
            added += 1
        return added

    @classmethod
    def load_csv(cls, path: str) -> "PlayerStatsStore":
        store = cls()
        with open(path, newline="", encoding="utf-8") as fh:
            store.ingest(csv.DictReader(fh))
        return store

    def game_log(self, player: str, stat: str, sport: Optional[str] = None) -> list[GameStatLine]:
        return [
            g
            for g in self.logs
            if g.player == player and g.stat == stat and (sport is None or g.sport == sport)
        ]

    def recent_values(
        self, player: str, stat: str, sport: Optional[str] = None, window: int = DEFAULT_WINDOW
    ) -> list[float]:
        """Most recent `window` values, assuming file/ingest order is
        chronological (the same convention `EloModel.train_from_csv` uses).
        """
        return [g.value for g in self.game_log(player, stat, sport)[-window:]]


def project_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("no games sampled for this player/stat")
    return mean(values)


def predict_prop_probability(sport: str, stat: str, line: float, values: list[float]) -> tuple[float, float]:
    """Returns (probability the OVER hits, projected mean), using a
    Poisson model for counting stats and a Normal model otherwise. With
    only one game sampled there's no variance to estimate, so the
    standard deviation falls back to a fixed fraction of the mean.
    """
    projected = project_mean(values)
    if stat_family(sport, stat) == "poisson":
        over_prob = poisson_over_probability(line, projected)
    else:
        std = pstdev(values) if len(values) > 1 else max(projected * 0.35, 1.0)
        over_prob = normal_over_probability(line, projected, std)
    return over_prob, projected


def predict_prop(
    store: PlayerStatsStore,
    prop: PropLine,
    window: int = DEFAULT_WINDOW,
    min_games: int = DEFAULT_MIN_GAMES,
) -> Optional[PropPrediction]:
    """Projects both sides of a prop line and returns whichever side (over
    or under) the model favors relative to the market -- i.e. the side
    with the larger edge -- or None if there isn't enough game-log history
    to project confidently.
    """
    values = store.recent_values(prop.player, prop.stat, prop.sport, window)
    if len(values) < min_games:
        return None

    over_prob, projected = predict_prop_probability(prop.sport, prop.stat, prop.line, values)

    best = None
    for side, model_prob, decimal_odds in (
        ("over", over_prob, prop.over_odds),
        ("under", 1.0 - over_prob, prop.under_odds),
    ):
        implied_prob = decimal_to_implied_prob(decimal_odds)
        edge = model_prob - implied_prob
        if best is None or edge > best[0]:
            best = (edge, side, model_prob, decimal_odds, implied_prob)

    edge, side, model_prob, decimal_odds, implied_prob = best
    return PropPrediction(
        sport=prop.sport,
        player=prop.player,
        stat=prop.stat,
        line=prop.line,
        side=side,
        model_prob=model_prob,
        decimal_odds=decimal_odds,
        implied_prob=implied_prob,
        edge=edge,
        expected_value=expected_value(model_prob, decimal_odds),
        projected_mean=projected,
        games_sampled=len(values),
    )


def predict_props(
    store: PlayerStatsStore,
    prop_lines: list[PropLine],
    window: int = DEFAULT_WINDOW,
    min_games: int = DEFAULT_MIN_GAMES,
    min_edge: Optional[float] = None,
) -> list[PropPrediction]:
    """Projects every prop line, drops any without enough history or below
    `min_edge`, and ranks the rest by edge, descending.
    """
    predictions = []
    for prop in prop_lines:
        prediction = predict_prop(store, prop, window=window, min_games=min_games)
        if prediction is None:
            continue
        if min_edge is not None and prediction.edge < min_edge:
            continue
        predictions.append(prediction)

    predictions.sort(key=lambda p: p.edge, reverse=True)
    return predictions


def load_prop_lines_csv(path: str) -> list[PropLine]:
    """Loads a prop board CSV: sport,player,stat,line,over_odds,under_odds."""
    lines = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            lines.append(
                PropLine(
                    sport=row["sport"],
                    player=row["player"],
                    stat=row["stat"],
                    line=float(row["line"]),
                    over_odds=float(row["over_odds"]),
                    under_odds=float(row["under_odds"]),
                )
            )
    return lines

"""Combines a model's predicted probabilities with market odds to surface
positive expected-value ("value") bets.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kelly import kelly_stake
from .odds import decimal_to_implied_prob


@dataclass
class GameOdds:
    """One market to evaluate: a label plus the decimal odds offered."""

    event: str
    selection: str
    decimal_odds: float
    model_prob: float


@dataclass
class ValueBet:
    event: str
    selection: str
    decimal_odds: float
    model_prob: float
    implied_prob: float
    edge: float
    expected_value: float
    recommended_stake: float

    def __str__(self) -> str:
        return (
            f"{self.event} -- {self.selection}: odds {self.decimal_odds:.2f}, "
            f"model {self.model_prob:.1%} vs implied {self.implied_prob:.1%} "
            f"(edge {self.edge:+.1%}, EV {self.expected_value:+.3f}), "
            f"stake {self.recommended_stake:.2f}"
        )


def expected_value(model_prob: float, decimal_odds: float) -> float:
    """Expected profit per 1 unit staked: p*(odds-1) - (1-p)."""
    return model_prob * (decimal_odds - 1.0) - (1.0 - model_prob)


def find_value_bets(
    games: list[GameOdds],
    min_edge: float = 0.02,
    bankroll: float = 1000.0,
    kelly_fraction_size: float = 0.25,
    max_stake_fraction: float = 0.05,
) -> list[ValueBet]:
    """Filter and rank candidate bets by edge (model probability minus the
    market's implied probability), keeping only those at or above
    `min_edge`. Recommends a stake via fractional Kelly, capped at
    `max_stake_fraction` of bankroll per bet.
    """
    value_bets = []
    for game in games:
        implied_prob = decimal_to_implied_prob(game.decimal_odds)
        edge = game.model_prob - implied_prob
        if edge < min_edge:
            continue

        ev = expected_value(game.model_prob, game.decimal_odds)
        stake = kelly_stake(
            bankroll,
            game.model_prob,
            game.decimal_odds,
            fraction=kelly_fraction_size,
            max_fraction=max_stake_fraction,
        )
        value_bets.append(
            ValueBet(
                event=game.event,
                selection=game.selection,
                decimal_odds=game.decimal_odds,
                model_prob=game.model_prob,
                implied_prob=implied_prob,
                edge=edge,
                expected_value=ev,
                recommended_stake=stake,
            )
        )

    value_bets.sort(key=lambda vb: vb.edge, reverse=True)
    return value_bets

"""Builds multi-leg parlays and round-robin ticket combinations from a
pool of candidate bets (player-prop predictions or game value bets),
ranking them by edge the same way `analyzer.find_value_bets` does for
single bets.

Combining legs assumes independence: combined odds are the product of
each leg's decimal odds, and combined probability is the product of each
leg's model probability. That's accurate for props on unrelated
games/players and an approximation for correlated same-game legs (e.g. a
QB's passing yards and his team's total points), which real-world
correlation would need to adjust for.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from .analyzer import expected_value
from .kelly import kelly_stake
from .odds import decimal_to_implied_prob

if TYPE_CHECKING:
    from .analyzer import ValueBet
    from .player_props import PropPrediction

DEFAULT_PARLAY_SIZES = (4, 5)


@dataclass
class ParlayLeg:
    """One leg of a parlay: a label for display plus the model probability
    and decimal odds needed to combine it with other legs.
    """

    label: str
    model_prob: float
    decimal_odds: float

    @classmethod
    def from_prop(cls, prediction: "PropPrediction") -> "ParlayLeg":
        return cls(
            label=f"{prediction.player} {prediction.stat} {prediction.side} {prediction.line}",
            model_prob=prediction.model_prob,
            decimal_odds=prediction.decimal_odds,
        )

    @classmethod
    def from_value_bet(cls, value_bet: "ValueBet") -> "ParlayLeg":
        return cls(
            label=f"{value_bet.event} -- {value_bet.selection}",
            model_prob=value_bet.model_prob,
            decimal_odds=value_bet.decimal_odds,
        )


@dataclass
class Parlay:
    legs: tuple[ParlayLeg, ...]
    decimal_odds: float
    model_prob: float
    implied_prob: float
    edge: float
    expected_value: float
    recommended_stake: float

    @property
    def size(self) -> int:
        return len(self.legs)

    def __str__(self) -> str:
        leg_desc = "; ".join(leg.label for leg in self.legs)
        return (
            f"{self.size}-leg parlay [{leg_desc}] -- odds {self.decimal_odds:.2f}, "
            f"model {self.model_prob:.1%} vs implied {self.implied_prob:.1%} "
            f"(edge {self.edge:+.1%}, EV {self.expected_value:+.3f}), "
            f"stake {self.recommended_stake:.2f}"
        )


def build_parlay(
    legs: Sequence[ParlayLeg],
    bankroll: float = 1000.0,
    kelly_fraction_size: float = 0.25,
    max_stake_fraction: float = 0.05,
) -> Parlay:
    if not legs:
        raise ValueError("a parlay needs at least one leg")

    decimal_odds = 1.0
    model_prob = 1.0
    for leg in legs:
        decimal_odds *= leg.decimal_odds
        model_prob *= leg.model_prob

    implied_prob = decimal_to_implied_prob(decimal_odds)
    edge = model_prob - implied_prob
    stake = (
        kelly_stake(bankroll, model_prob, decimal_odds, fraction=kelly_fraction_size, max_fraction=max_stake_fraction)
        if model_prob < 1.0
        else 0.0
    )
    return Parlay(
        legs=tuple(legs),
        decimal_odds=decimal_odds,
        model_prob=model_prob,
        implied_prob=implied_prob,
        edge=edge,
        expected_value=expected_value(model_prob, decimal_odds),
        recommended_stake=stake,
    )


def generate_parlays(
    legs: Sequence[ParlayLeg],
    sizes: Sequence[int] = DEFAULT_PARLAY_SIZES,
    min_edge: float = 0.0,
    max_results: Optional[int] = None,
    bankroll: float = 1000.0,
    kelly_fraction_size: float = 0.25,
    max_stake_fraction: float = 0.05,
) -> list[Parlay]:
    """Builds every combination of `legs` at each size in `sizes` (4-leg
    and 5-leg by default), keeping only parlays at or above `min_edge`,
    ranked by edge descending.
    """
    usable_sizes = [size for size in sizes if size <= len(legs)]

    parlays = []
    for size in usable_sizes:
        for combo in itertools.combinations(legs, size):
            parlay = build_parlay(combo, bankroll, kelly_fraction_size, max_stake_fraction)
            if parlay.edge >= min_edge:
                parlays.append(parlay)

    parlays.sort(key=lambda p: p.edge, reverse=True)
    if max_results is not None:
        parlays = parlays[:max_results]
    return parlays


@dataclass
class RoundRobin:
    """A round-robin ticket: every `group_size`-leg combination drawn from
    a larger pool of picks, each staked flat -- spreads risk across a
    bigger pool of picks instead of needing every leg in one all-or-
    nothing parlay to hit.
    """

    group_size: int
    parlays: list[Parlay]
    unit_stake: float

    @property
    def ticket_count(self) -> int:
        return len(self.parlays)

    @property
    def total_stake(self) -> float:
        return self.unit_stake * self.ticket_count

    @property
    def total_expected_value(self) -> float:
        return sum(p.expected_value * self.unit_stake for p in self.parlays)

    def __str__(self) -> str:
        header = (
            f"Round robin ({self.group_size}-leg combos): {self.ticket_count} ticket(s), "
            f"unit stake {self.unit_stake:.2f}, total stake {self.total_stake:.2f}, "
            f"expected profit {self.total_expected_value:+.2f}"
        )
        return "\n".join([header, *(f"  {p}" for p in self.parlays)])


def generate_round_robin(
    legs: Sequence[ParlayLeg],
    group_size: int = 4,
    unit_stake: float = 10.0,
    min_edge: float = 0.0,
    bankroll: float = 1000.0,
    kelly_fraction_size: float = 0.25,
    max_stake_fraction: float = 0.05,
) -> RoundRobin:
    """Builds a round-robin ticket of every `group_size`-leg combination
    from the full pool of `legs`.
    """
    if group_size > len(legs):
        raise ValueError("group_size cannot exceed the number of legs in the pool")

    parlays = [
        build_parlay(combo, bankroll, kelly_fraction_size, max_stake_fraction)
        for combo in itertools.combinations(legs, group_size)
    ]
    parlays = [p for p in parlays if p.edge >= min_edge]
    parlays.sort(key=lambda p: p.edge, reverse=True)
    return RoundRobin(group_size=group_size, parlays=parlays, unit_stake=unit_stake)

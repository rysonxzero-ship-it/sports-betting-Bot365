"""Sports betting analysis toolkit.

Evaluates odds across sports, models expected outcomes, and surfaces
positive expected-value bets. See README.md for usage.
"""

from .odds import (
    american_to_decimal,
    decimal_to_american,
    decimal_to_implied_prob,
    implied_prob_to_decimal,
    remove_vig,
)
from .elo import EloModel
from .kelly import kelly_fraction, kelly_stake
from .tracker import Bet, BetTracker
from .analyzer import ValueBet, find_value_bets
from .player_props import (
    GameStatLine,
    PlayerStatsStore,
    PropLine,
    PropPrediction,
    load_prop_lines_csv,
    predict_prop,
    predict_props,
)
from .live_feed import LiveStatsFeed, csv_tail_fetcher
from .parlay import (
    Parlay,
    ParlayLeg,
    RoundRobin,
    build_parlay,
    generate_parlays,
    generate_round_robin,
)

__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "decimal_to_implied_prob",
    "implied_prob_to_decimal",
    "remove_vig",
    "EloModel",
    "kelly_fraction",
    "kelly_stake",
    "Bet",
    "BetTracker",
    "ValueBet",
    "find_value_bets",
    "GameStatLine",
    "PlayerStatsStore",
    "PropLine",
    "PropPrediction",
    "load_prop_lines_csv",
    "predict_prop",
    "predict_props",
    "LiveStatsFeed",
    "csv_tail_fetcher",
    "Parlay",
    "ParlayLeg",
    "RoundRobin",
    "build_parlay",
    "generate_parlays",
    "generate_round_robin",
]

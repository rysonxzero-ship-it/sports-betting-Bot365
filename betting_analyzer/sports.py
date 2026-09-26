"""Sport-specific stat classification and probability distributions for
player scoring props (baseball, football, soccer).

Counting stats (hits, home runs, touchdowns, goals, shots, ...) are
modeled as a Poisson process -- a standard treatment for low-frequency
per-game event counts. Continuous stats (passing/rushing/receiving
yardage) are modeled as Normal, since yardage totals aren't naturally
bounded to small integers the way event counts are.
"""

from __future__ import annotations

import math

# Which counting stats use the Poisson model, per sport. Any stat not
# listed here for a sport falls back to the Normal model.
POISSON_STATS = {
    "baseball": {
        "hits",
        "runs",
        "rbi",
        "home_runs",
        "strikeouts",
        "stolen_bases",
        "total_bases",
    },
    "football": {"touchdowns", "receptions", "interceptions"},
    "soccer": {"goals", "assists", "shots", "shots_on_target"},
}

SUPPORTED_SPORTS = tuple(POISSON_STATS)


def stat_family(sport: str, stat: str) -> str:
    """Returns "poisson" or "normal" for a given sport/stat combination."""
    if sport not in SUPPORTED_SPORTS:
        raise ValueError(f"Unsupported sport {sport!r}; expected one of {SUPPORTED_SPORTS}")
    return "poisson" if stat in POISSON_STATS[sport] else "normal"


def poisson_pmf(k: int, lam: float) -> float:
    """P(X = k) for X ~ Poisson(lam)."""
    if lam < 0:
        raise ValueError("lam must be >= 0")
    if k < 0:
        return 0.0
    return math.exp(-lam) * lam**k / math.factorial(k)


def poisson_prob_at_least(k: int, lam: float) -> float:
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    return max(0.0, 1.0 - sum(poisson_pmf(i, lam) for i in range(k)))


def poisson_over_probability(line: float, lam: float) -> float:
    """P(X > line) for X ~ Poisson(lam), for a prop line that may be a
    half-point (e.g. 1.5 goals) or a whole number.
    """
    threshold = math.floor(line) + 1
    return poisson_prob_at_least(threshold, lam)


def normal_cdf(x: float, mean: float, std: float) -> float:
    if std <= 0:
        raise ValueError("std must be positive")
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2.0))))


def normal_over_probability(line: float, mean: float, std: float) -> float:
    """P(X > line) for X ~ Normal(mean, std)."""
    return 1.0 - normal_cdf(line, mean, std)

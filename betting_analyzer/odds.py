"""Odds format conversions and implied-probability math.

Decimal odds are the internal representation used throughout the rest of
the package (e.g. 2.50 means a $1 stake returns $2.50 total). Conversions
to/from American ("moneyline") odds are provided for convenience since
that's the format most US sportsbooks display.
"""

from __future__ import annotations


def american_to_decimal(american: float) -> float:
    """Convert American odds (e.g. +150, -200) to decimal odds."""
    if american == 0:
        raise ValueError("American odds cannot be 0")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds (e.g. 2.50) to American odds."""
    if decimal <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    if decimal >= 2.0:
        return round((decimal - 1.0) * 100.0)
    return round(-100.0 / (decimal - 1.0))


def decimal_to_implied_prob(decimal: float) -> float:
    """Implied win probability from decimal odds, including the book's vig."""
    if decimal <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / decimal


def implied_prob_to_decimal(prob: float) -> float:
    """Fair decimal odds for a given win probability (no vig)."""
    if not 0.0 < prob <= 1.0:
        raise ValueError("Probability must be in (0, 1]")
    return 1.0 / prob


def overround(decimal_odds: list[float]) -> float:
    """Sum of implied probabilities across a market; >1.0 means the book has vig."""
    return sum(decimal_to_implied_prob(o) for o in decimal_odds)


def remove_vig(decimal_odds: list[float]) -> list[float]:
    """Normalize a market's implied probabilities to remove the overround.

    Given the decimal odds for every outcome in a market (e.g. home/away,
    or home/draw/away), returns the book's implied probabilities rescaled
    so they sum to 1.0 -- an estimate of the "fair" probabilities without
    the bookmaker's margin.
    """
    if not decimal_odds:
        raise ValueError("decimal_odds must be non-empty")
    implied = [decimal_to_implied_prob(o) for o in decimal_odds]
    total = sum(implied)
    if total <= 0:
        raise ValueError("Implied probabilities must sum to a positive value")
    return [p / total for p in implied]

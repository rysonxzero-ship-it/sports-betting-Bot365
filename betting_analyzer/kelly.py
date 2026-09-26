"""Kelly Criterion staking for bankroll management.

Given a model's estimated win probability and the decimal odds on offer,
the Kelly Criterion computes the bankroll fraction that maximizes
long-run geometric growth. Books that recommend flat staking undersell
how much faster/safer a properly-sized (and typically fractional, e.g.
quarter- or half-Kelly) stake compounds a bankroll.
"""

from __future__ import annotations


def kelly_fraction(prob: float, decimal_odds: float) -> float:
    """Full-Kelly bankroll fraction to stake. Negative/zero means no edge -- don't bet.

    f* = (b*p - q) / b, where b = decimal_odds - 1, p = win prob, q = 1 - p.
    """
    if not 0.0 < prob < 1.0:
        raise ValueError("prob must be in (0, 1)")
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be greater than 1.0")

    b = decimal_odds - 1.0
    q = 1.0 - prob
    return (b * prob - q) / b


def kelly_stake(
    bankroll: float,
    prob: float,
    decimal_odds: float,
    fraction: float = 1.0,
    max_fraction: float = 1.0,
) -> float:
    """Recommended stake in bankroll units.

    `fraction` scales full Kelly (e.g. 0.25 for quarter-Kelly, a common
    way to reduce variance while giving up little long-run growth).
    `max_fraction` caps the stake as a fraction of bankroll regardless of
    what Kelly suggests, as a safety limit. Returns 0.0 when there is no
    edge (Kelly fraction <= 0).
    """
    if bankroll <= 0:
        raise ValueError("bankroll must be positive")
    if fraction <= 0:
        raise ValueError("fraction must be positive")

    f_star = kelly_fraction(prob, decimal_odds)
    if f_star <= 0:
        return 0.0

    stake_fraction = min(f_star * fraction, max_fraction)
    return bankroll * stake_fraction

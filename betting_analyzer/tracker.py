"""Tracks placed bets and computes historical performance metrics
(ROI, win rate, units won/lost, closing-line value).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from typing import Optional


VALID_RESULTS = {"pending", "win", "loss", "push"}


@dataclass
class Bet:
    date: str
    event: str
    selection: str
    decimal_odds: float
    stake: float
    model_prob: Optional[float] = None
    result: str = "pending"
    closing_odds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.result not in VALID_RESULTS:
            raise ValueError(f"result must be one of {VALID_RESULTS}")

    @property
    def profit(self) -> float:
        """Net profit/loss in stake units. 0.0 while pending or on a push."""
        if self.result == "win":
            return self.stake * (self.decimal_odds - 1.0)
        if self.result == "loss":
            return -self.stake
        return 0.0

    @property
    def closing_line_value(self) -> Optional[float]:
        """CLV: how much better the bet's odds were than the closing odds,
        as a percentage. Positive CLV is widely used as a predictor of
        long-run profitability independent of short-term results.
        """
        if self.closing_odds is None:
            return None
        return (self.decimal_odds / self.closing_odds) - 1.0


class BetTracker:
    FIELDNAMES = [
        "date",
        "event",
        "selection",
        "decimal_odds",
        "stake",
        "model_prob",
        "result",
        "closing_odds",
    ]

    def __init__(self, bets: Optional[list[Bet]] = None):
        self.bets: list[Bet] = list(bets) if bets else []

    def add_bet(self, bet: Bet) -> None:
        self.bets.append(bet)

    def settle(self, index: int, result: str, closing_odds: Optional[float] = None) -> None:
        if result not in VALID_RESULTS:
            raise ValueError(f"result must be one of {VALID_RESULTS}")
        bet = self.bets[index]
        bet.result = result
        if closing_odds is not None:
            bet.closing_odds = closing_odds

    def summary(self) -> dict:
        settled = [b for b in self.bets if b.result != "pending"]
        total_staked = sum(b.stake for b in settled)
        total_profit = sum(b.profit for b in settled)
        wins = sum(1 for b in settled if b.result == "win")
        losses = sum(1 for b in settled if b.result == "loss")
        decided = wins + losses

        clv_values = [b.closing_line_value for b in settled if b.closing_line_value is not None]

        return {
            "bets_settled": len(settled),
            "bets_pending": len(self.bets) - len(settled),
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": (total_profit / total_staked) if total_staked else 0.0,
            "win_rate": (wins / decided) if decided else 0.0,
            "wins": wins,
            "losses": losses,
            "pushes": sum(1 for b in settled if b.result == "push"),
            "avg_closing_line_value": (sum(clv_values) / len(clv_values)) if clv_values else None,
        }

    def save_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.FIELDNAMES)
            writer.writeheader()
            for bet in self.bets:
                writer.writerow(asdict(bet))

    @classmethod
    def load_csv(cls, path: str) -> "BetTracker":
        bets = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                bets.append(
                    Bet(
                        date=row["date"],
                        event=row["event"],
                        selection=row["selection"],
                        decimal_odds=float(row["decimal_odds"]),
                        stake=float(row["stake"]),
                        model_prob=float(row["model_prob"]) if row.get("model_prob") else None,
                        result=row.get("result") or "pending",
                        closing_odds=float(row["closing_odds"]) if row.get("closing_odds") else None,
                    )
                )
        return cls(bets)

"""Real-time data ingestion for player stat lines.

Historical data is a one-time full CSV load (`PlayerStatsStore.load_csv`).
Real-time data -- new game stat lines as they land -- is modeled here as
a poll loop against a `fetcher` callable that returns an iterable of rows
shaped like the CSV columns (date, sport, player, team, opponent, stat,
value). This keeps the module free of any specific paid odds/stats
provider or API key while making it a drop-in target for one: point
`fetcher` at a live API client's response parser and everything
downstream (projection, edge detection, parlay building) keeps working
unmodified.

`csv_tail_fetcher` is a fetcher implementation for the common case where
some external process (a cron job, another service) appends freshly
arrived rows to a CSV file -- it tracks how many rows it has already seen
so repeated polls only ingest what's new.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

from .player_props import PlayerStatsStore

Fetcher = Callable[[], Iterable[dict]]


@dataclass
class LiveStatsFeed:
    """Polls `fetcher` for freshly-arrived player stat lines and ingests
    them into `store` as they come in.
    """

    store: PlayerStatsStore
    fetcher: Fetcher
    poll_count: int = field(default=0, init=False)
    rows_ingested: int = field(default=0, init=False)

    def poll(self) -> int:
        """Fetches and ingests one batch. Returns the number of rows added."""
        added = self.store.ingest(list(self.fetcher()))
        self.poll_count += 1
        self.rows_ingested += added
        return added

    def run(self, interval_seconds: float, max_polls: Optional[int] = None) -> None:
        """Polls on a fixed interval, forever or up to `max_polls` times."""
        polls = 0
        while max_polls is None or polls < max_polls:
            self.poll()
            polls += 1
            if max_polls is not None and polls >= max_polls:
                break
            time.sleep(interval_seconds)


def csv_tail_fetcher(path: str) -> Fetcher:
    """Builds a fetcher that re-reads `path` and yields only the rows past
    the count already seen on a previous call, treating a file that's
    appended to externally as a real-time feed without re-ingesting
    duplicates.
    """
    seen = {"count": 0}

    def _fetch() -> Iterable[dict]:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        new_rows = rows[seen["count"] :]
        seen["count"] = len(rows)
        return new_rows

    return _fetch

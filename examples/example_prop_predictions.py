"""End-to-end walkthrough: project player scoring props for baseball,
football, and soccer from historical game logs, then build parlays and a
round-robin ticket from whichever props clear a minimum edge.

Run from the repo root:
    python -m examples.example_prop_predictions
"""

from __future__ import annotations

import os

from betting_analyzer.parlay import ParlayLeg, generate_parlays, generate_round_robin
from betting_analyzer.player_props import PlayerStatsStore, load_prop_lines_csv, predict_props

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main() -> None:
    store = PlayerStatsStore.load_csv(os.path.join(DATA_DIR, "player_game_logs.csv"))
    prop_lines = load_prop_lines_csv(os.path.join(DATA_DIR, "prop_lines.csv"))

    predictions = predict_props(store, prop_lines)
    print(f"Projected {len(predictions)} prop(s) across baseball, football, and soccer:\n")
    for prediction in predictions:
        print(f"  {prediction}")

    legs = [ParlayLeg.from_prop(p) for p in predictions if p.edge >= 0.02]
    print(f"\n{len(legs)} leg(s) clear a 2% edge and go into the parlay pool.\n")

    parlays = generate_parlays(legs, sizes=(4, 5), max_results=3)
    print("Top parlays (4-5 legs):")
    for parlay in parlays:
        print(f"  {parlay}")

    if len(legs) >= 4:
        round_robin = generate_round_robin(legs, group_size=4, unit_stake=10.0)
        print(f"\n{round_robin}")


if __name__ == "__main__":
    main()

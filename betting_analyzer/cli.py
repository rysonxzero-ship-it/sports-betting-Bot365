"""Command-line interface for the sports betting analyzer.

Commands:
  train-elo       Build/update Elo ratings from a CSV of historical results.
  analyze         Compare Elo win probabilities against market odds for
                  upcoming games and print recommended value bets.
  predict-props   Project player scoring props from game logs and compare
                  against a sportsbook's line for baseball/football/soccer.
  build-parlays   Generate 4-5 leg parlays (or a custom size range) from
                  the props with an edge, ranked by combined edge.
  build-round-robin
                  Generate a round-robin ticket of every N-leg combination
                  from the props with an edge.
  track-summary   Print performance stats (ROI, win rate, CLV) from a CSV
                  bet log.

Run `python -m betting_analyzer.cli <command> --help` for details.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from .analyzer import GameOdds, find_value_bets
from .elo import EloModel
from .parlay import ParlayLeg, generate_parlays, generate_round_robin
from .player_props import PlayerStatsStore, load_prop_lines_csv, predict_props
from .tracker import BetTracker


def _cmd_train_elo(args: argparse.Namespace) -> None:
    model = EloModel(k_factor=args.k_factor, home_advantage=args.home_advantage)
    model.train_from_csv(args.results_csv)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(model.to_dict(), fh, indent=2, sort_keys=True)
    print(f"Trained Elo ratings for {len(model.ratings)} teams -> {args.output}")


def _cmd_analyze(args: argparse.Namespace) -> None:
    with open(args.ratings_json, encoding="utf-8") as fh:
        model = EloModel.from_dict(json.load(fh))

    games = []
    with open(args.odds_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            home, away = row["home_team"], row["away_team"]
            event = f"{home} vs {away}"

            home_prob = model.predict_win_probability(home, away, team_a_is_home=True)
            away_prob = 1.0 - home_prob

            games.append(
                GameOdds(event, home, float(row["home_odds"]), home_prob)
            )
            games.append(
                GameOdds(event, away, float(row["away_odds"]), away_prob)
            )

    value_bets = find_value_bets(
        games,
        min_edge=args.min_edge,
        bankroll=args.bankroll,
        kelly_fraction_size=args.kelly_fraction,
        max_stake_fraction=args.max_stake_fraction,
    )

    if not value_bets:
        print("No value bets found at or above the minimum edge threshold.")
        return

    print(f"{len(value_bets)} value bet(s) found:\n")
    for vb in value_bets:
        print(vb)

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["event", "selection", "decimal_odds", "model_prob", "implied_prob", "edge", "expected_value", "recommended_stake"]
            )
            for vb in value_bets:
                writer.writerow(
                    [vb.event, vb.selection, vb.decimal_odds, vb.model_prob, vb.implied_prob, vb.edge, vb.expected_value, vb.recommended_stake]
                )
        print(f"\nSaved to {args.output}")


def _predict_props_from_args(args: argparse.Namespace):
    store = PlayerStatsStore.load_csv(args.game_logs_csv)
    prop_lines = load_prop_lines_csv(args.prop_lines_csv)
    return predict_props(
        store,
        prop_lines,
        window=args.window,
        min_games=args.min_games,
        min_edge=args.min_leg_edge,
    )


def _cmd_predict_props(args: argparse.Namespace) -> None:
    predictions = _predict_props_from_args(args)
    if not predictions:
        print("No prop predictions cleared the minimum leg edge.")
        return

    print(f"{len(predictions)} prop prediction(s):\n")
    for prediction in predictions:
        print(prediction)

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["sport", "player", "stat", "line", "side", "model_prob", "decimal_odds", "implied_prob", "edge", "expected_value", "projected_mean", "games_sampled"]
            )
            for p in predictions:
                writer.writerow(
                    [p.sport, p.player, p.stat, p.line, p.side, p.model_prob, p.decimal_odds, p.implied_prob, p.edge, p.expected_value, p.projected_mean, p.games_sampled]
                )
        print(f"\nSaved to {args.output}")


def _write_parlays_csv(path: str, parlays) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["legs", "decimal_odds", "model_prob", "implied_prob", "edge", "expected_value", "recommended_stake"])
        for parlay in parlays:
            writer.writerow(
                [
                    "; ".join(leg.label for leg in parlay.legs),
                    parlay.decimal_odds,
                    parlay.model_prob,
                    parlay.implied_prob,
                    parlay.edge,
                    parlay.expected_value,
                    parlay.recommended_stake,
                ]
            )


def _cmd_build_parlays(args: argparse.Namespace) -> None:
    predictions = _predict_props_from_args(args)
    legs = [ParlayLeg.from_prop(p) for p in predictions]
    sizes = [int(s) for s in args.sizes.split(",")]

    parlays = generate_parlays(
        legs,
        sizes=sizes,
        min_edge=args.min_edge,
        max_results=args.max_results,
        bankroll=args.bankroll,
        kelly_fraction_size=args.kelly_fraction,
        max_stake_fraction=args.max_stake_fraction,
    )

    if not parlays:
        print("No parlays cleared the minimum edge threshold.")
        return

    print(f"{len(parlays)} parlay(s) found from {len(legs)} candidate leg(s):\n")
    for parlay in parlays:
        print(parlay)

    if args.output:
        _write_parlays_csv(args.output, parlays)
        print(f"\nSaved to {args.output}")


def _cmd_build_round_robin(args: argparse.Namespace) -> None:
    predictions = _predict_props_from_args(args)
    legs = [ParlayLeg.from_prop(p) for p in predictions]

    round_robin = generate_round_robin(
        legs,
        group_size=args.group_size,
        unit_stake=args.unit_stake,
        min_edge=args.min_edge,
        bankroll=args.bankroll,
        kelly_fraction_size=args.kelly_fraction,
        max_stake_fraction=args.max_stake_fraction,
    )

    print(f"Built from {len(legs)} candidate leg(s):")
    print(round_robin)

    if args.output:
        _write_parlays_csv(args.output, round_robin.parlays)
        print(f"\nSaved to {args.output}")


def _cmd_track_summary(args: argparse.Namespace) -> None:
    tracker = BetTracker.load_csv(args.bets_csv)
    summary = tracker.summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="betting_analyzer", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train-elo", help="Train Elo ratings from historical results")
    train_parser.add_argument("results_csv", help="CSV with home_team,away_team,home_score,away_score")
    train_parser.add_argument("-o", "--output", default="elo_ratings.json", help="Output ratings JSON path")
    train_parser.add_argument("--k-factor", type=float, default=20.0)
    train_parser.add_argument("--home-advantage", type=float, default=65.0)
    train_parser.set_defaults(func=_cmd_train_elo)

    analyze_parser = subparsers.add_parser("analyze", help="Find value bets for upcoming games")
    analyze_parser.add_argument("ratings_json", help="Elo ratings JSON produced by train-elo")
    analyze_parser.add_argument("odds_csv", help="CSV with home_team,away_team,home_odds,away_odds")
    analyze_parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge to flag a bet (default 0.02 = 2%%)")
    analyze_parser.add_argument("--bankroll", type=float, default=1000.0)
    analyze_parser.add_argument("--kelly-fraction", type=float, default=0.25, help="Fraction of full Kelly to stake (default quarter-Kelly)")
    analyze_parser.add_argument("--max-stake-fraction", type=float, default=0.05, help="Max stake as a fraction of bankroll per bet")
    analyze_parser.add_argument("-o", "--output", help="Optional CSV path to save recommended bets")
    analyze_parser.set_defaults(func=_cmd_analyze)

    def add_prop_source_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("game_logs_csv", help="CSV with date,sport,player,team,opponent,stat,value")
        p.add_argument("prop_lines_csv", help="CSV with sport,player,stat,line,over_odds,under_odds")
        p.add_argument("--window", type=int, default=10, help="Most recent N games to project from (default 10)")
        p.add_argument("--min-games", type=int, default=3, help="Minimum games sampled to make a projection (default 3)")
        p.add_argument("--min-leg-edge", type=float, default=None, help="Drop individual prop legs below this edge before building parlays")

    predict_parser = subparsers.add_parser(
        "predict-props", help="Project player scoring props (baseball/football/soccer) from game logs"
    )
    add_prop_source_args(predict_parser)
    predict_parser.add_argument("-o", "--output", help="Optional CSV path to save predictions")
    predict_parser.set_defaults(func=_cmd_predict_props)

    def add_parlay_stake_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--min-edge", type=float, default=0.0, help="Minimum combined edge to keep a parlay (default 0.0)")
        p.add_argument("--bankroll", type=float, default=1000.0)
        p.add_argument("--kelly-fraction", type=float, default=0.25, help="Fraction of full Kelly to stake (default quarter-Kelly)")
        p.add_argument("--max-stake-fraction", type=float, default=0.05, help="Max stake as a fraction of bankroll per parlay")
        p.add_argument("-o", "--output", help="Optional CSV path to save parlays")

    parlay_parser = subparsers.add_parser(
        "build-parlays", help="Generate 4-5 leg (or custom size) parlays from player-prop edges"
    )
    add_prop_source_args(parlay_parser)
    parlay_parser.add_argument("--sizes", default="4,5", help="Comma-separated parlay leg counts to build (default 4,5)")
    parlay_parser.add_argument("--max-results", type=int, default=None, help="Cap the number of parlays returned")
    add_parlay_stake_args(parlay_parser)
    parlay_parser.set_defaults(func=_cmd_build_parlays)

    round_robin_parser = subparsers.add_parser(
        "build-round-robin", help="Generate a round-robin ticket of every N-leg combination from player-prop edges"
    )
    add_prop_source_args(round_robin_parser)
    round_robin_parser.add_argument("--group-size", type=int, default=4, help="Legs per combination (default 4)")
    round_robin_parser.add_argument("--unit-stake", type=float, default=10.0, help="Flat stake per combination (default 10.0)")
    add_parlay_stake_args(round_robin_parser)
    round_robin_parser.set_defaults(func=_cmd_build_round_robin)

    track_parser = subparsers.add_parser("track-summary", help="Summarize performance from a bet log CSV")
    track_parser.add_argument("bets_csv", help="CSV bet log (see BetTracker.FIELDNAMES)")
    track_parser.set_defaults(func=_cmd_track_summary)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

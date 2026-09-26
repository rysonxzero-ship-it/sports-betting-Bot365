"""End-to-end example: train Elo ratings on historical results, then find
value bets for upcoming games against the sample odds board.

Run from the repo root:
    python examples/example_usage.py
"""

from betting_analyzer import EloModel, find_value_bets, remove_vig
from betting_analyzer.analyzer import GameOdds
from betting_analyzer.tracker import Bet, BetTracker


def main() -> None:
    # 1. Build a predictive model from historical results.
    model = EloModel()
    model.train_from_csv("data/sample_results.csv")

    print("Elo ratings after training:")
    for team, rating in sorted(model.ratings.items(), key=lambda kv: -kv[1]):
        print(f"  {team:10s} {rating:.1f}")

    # 2. Compare model probabilities against market odds for upcoming games.
    upcoming = [
        ("Hawks", "Comets", 1.60, 2.50),
        ("Wolves", "Falcons", 1.90, 2.05),
        ("Comets", "Wolves", 3.20, 1.42),
    ]
    games = []
    for home, away, home_odds, away_odds in upcoming:
        home_prob = model.predict_win_probability(home, away, team_a_is_home=True)
        games.append(GameOdds(f"{home} vs {away}", home, home_odds, home_prob))
        games.append(GameOdds(f"{home} vs {away}", away, away_odds, 1.0 - home_prob))

    value_bets = find_value_bets(games, min_edge=0.02, bankroll=1000, kelly_fraction_size=0.25)

    print("\nRecommended value bets:")
    if not value_bets:
        print("  none found")
    for vb in value_bets:
        print(f"  {vb}")

    # 3. De-vig a market to see the book's true implied probabilities.
    fair_probs = remove_vig([1.60, 2.50])
    print(f"\nDe-vigged Hawks/Comets probabilities: {fair_probs[0]:.1%} / {fair_probs[1]:.1%}")

    # 4. Track results over time.
    tracker = BetTracker()
    tracker.add_bet(Bet("2025-02-01", "Hawks vs Comets", "Hawks", 1.60, 25, model_prob=0.66, result="win"))
    tracker.add_bet(Bet("2025-02-08", "Wolves vs Falcons", "Falcons", 2.05, 25, model_prob=0.52, result="loss"))
    print("\nBet log summary:")
    for key, value in tracker.summary().items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

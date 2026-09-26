# predictive-sports-betting

A predictive sports betting toolkit. It ingests historical (and, via a
pluggable poll loop, real-time) data for baseball, football, and soccer;
projects player scoring outcomes and game results; compares those
projections against sportsbook odds to find positive expected-value
bets; and assembles qualifying legs into 4-5 leg parlays and round-robin
ticket combinations.

Built entirely on the Python standard library -- no API keys or paid data
feeds required to try it out. Historical data and odds/prop boards are
supplied as plain CSV files, so the same tools work whether you're
pasting in a sample data set or piping in a real feed.

## What it does

- **Player prop prediction** (`betting_analyzer/sports.py`,
  `betting_analyzer/player_props.py`) -- projects a player's expected
  value for a scoring stat (hits, home runs, passing yards, touchdowns,
  goals, shots, etc.) from their recent game log, using a Poisson model
  for counting stats and a Normal model for continuous (yardage) stats,
  then compares that projection's over/under probability against the
  sportsbook's line and odds to find an edge.
- **Real-time & historical data ingestion**
  (`betting_analyzer/player_props.py`, `betting_analyzer/live_feed.py`)
  -- `PlayerStatsStore` loads historical per-game stat lines from CSV and
  also accepts incremental `ingest()` calls for freshly-arrived data.
  `LiveStatsFeed` polls a pluggable `fetcher` callable on an interval and
  feeds new rows into the store -- point it at a real odds/stats API's
  response parser to go from sample CSVs to an actual real-time feed
  without changing anything downstream.
- **Parlay & round-robin generation** (`betting_analyzer/parlay.py`) --
  `generate_parlays` builds every combination of prop legs at a given
  size (4-5 legs by default) and ranks them by combined edge;
  `generate_round_robin` builds a full round-robin ticket -- every
  N-leg combination from a larger pool of picks -- at a flat stake per
  combination. Combining legs assumes independence: accurate for props
  on unrelated games/players, an approximation for correlated same-game
  legs.
- **Game-outcome value bets** (`betting_analyzer/elo.py`,
  `betting_analyzer/analyzer.py`) -- an Elo rating model (with home-field
  advantage and a margin-of-victory adjustment) predicts win probability
  for upcoming matchups, then compares that against market odds to
  surface positive expected-value bets.
- **Odds conversion & de-vigging** (`betting_analyzer/odds.py`) -- convert
  between American and decimal odds, compute implied probability, and
  strip the bookmaker's overround ("vig") out of a market to estimate
  fair probabilities.
- **Bankroll management** (`betting_analyzer/kelly.py`) -- sizes stakes
  using the (fractional) Kelly Criterion, capped at a configurable maximum
  fraction of bankroll per bet.
- **Historical performance tracking** (`betting_analyzer/tracker.py`) --
  logs placed bets to CSV and reports ROI, win rate, and average
  closing-line value (CLV).

## Quick start

```bash
# Project player props (baseball/football/soccer) from historical game logs
python main.py predict-props data/player_game_logs.csv data/prop_lines.csv

# Build 4-5 leg parlays from whichever props clear an edge
python main.py build-parlays data/player_game_logs.csv data/prop_lines.csv --min-leg-edge 0.02

# Build a round-robin ticket of every 4-leg combination
python main.py build-round-robin data/player_game_logs.csv data/prop_lines.csv --min-leg-edge 0.02 --group-size 4

# Game-outcome value bets: train Elo, then find value against an odds board
python main.py train-elo data/sample_results.csv -o elo_ratings.json
python main.py analyze elo_ratings.json data/sample_odds.csv --min-edge 0.02

# Summarize a logged bet history
python main.py track-summary path/to/bets.csv
```

Or run the end-to-end walkthroughs:

```bash
python -m examples.example_prop_predictions
python examples/example_usage.py
```

## CSV formats

- **Player game logs** (`predict-props`, `build-parlays`, `build-round-robin`): `date,sport,player,team,opponent,stat,value`
- **Prop lines** (`predict-props`, `build-parlays`, `build-round-robin`): `sport,player,stat,line,over_odds,under_odds` (decimal odds)
- **Historical results** (`train-elo`): `home_team,away_team,home_score,away_score`
- **Odds board** (`analyze`): `home_team,away_team,home_odds,away_odds` (decimal odds)
- **Bet log** (`track-summary`): see `BetTracker.FIELDNAMES` in `betting_analyzer/tracker.py`

## Running tests

No third-party test runner required:

```bash
python -m unittest discover -s tests
```

## Disclaimer

This is a statistical modeling and decision-support tool, not a
guarantee of profit. Sports betting involves risk; only wager what you
can afford to lose, and check that betting is legal in your jurisdiction
before placing any wagers.

"""Command-line entry point for the predictive sports betting toolkit.

Usage:
    python main.py <command> [args...]

Run `python main.py --help` for the full list of commands (train-elo,
analyze, predict-props, build-parlays, build-round-robin, track-summary).
"""
from __future__ import annotations

import sys

from betting_analyzer.cli import main as cli_main

if __name__ == "__main__":
    raise SystemExit(cli_main(sys.argv[1:]))

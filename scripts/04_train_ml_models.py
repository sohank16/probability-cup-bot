from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_model import (
    build_training_rows,
    load_processed_matches,
    save_model_bundle,
    train_goal_market_models,
)


DEFAULT_MATCHES_PATH = Path("data/processed/recent_international_matches.csv")
DEFAULT_RANKINGS_PATH = Path("data/external/fifa_rankings_men.csv")
DEFAULT_HISTORICAL_RANKINGS_PATH = Path("data/external/fifa_rankings_men_history.csv")
DEFAULT_MODEL_PATH = Path("models/goal_market_logistic.joblib")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train logistic regression models for winner and goal-related markets."
    )
    parser.add_argument(
        "--matches",
        type=Path,
        default=DEFAULT_MATCHES_PATH,
        help="Path to recent_international_matches.csv from scripts/02_prepare_football_data.py.",
    )
    parser.add_argument(
        "--rankings",
        type=Path,
        default=DEFAULT_RANKINGS_PATH,
        help="Path to cached FIFA rankings CSV.",
    )
    parser.add_argument(
        "--historical-rankings",
        type=Path,
        default=DEFAULT_HISTORICAL_RANKINGS_PATH,
        help="Optional cached historical FIFA rankings CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained joblib model bundle.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.matches.exists():
        print(f"Processed matches not found: {args.matches}")
        print("Run scripts/02_prepare_football_data.py first.")
        return 1
    if not args.rankings.exists():
        print(f"FIFA rankings not found: {args.rankings}")
        print("Run scripts/02_prepare_football_data.py first.")
        return 1

    matches = load_processed_matches(args.matches)
    historical_rankings = args.historical_rankings if args.historical_rankings.exists() else None
    rows = build_training_rows(matches, args.rankings, historical_rankings)
    bundle = train_goal_market_models(rows)
    save_model_bundle(args.output, bundle)

    print("Logistic regression training complete.")
    print(f"Training rows: {bundle.training_rows}")
    print(f"Validation rows: {bundle.validation_rows}")
    print(f"Historical FIFA rankings: {historical_rankings or 'not found; current rankings only'}")
    print(f"Model output: {args.output}")
    print("")
    print("Validation Brier scores lower is better:")
    for target, score in sorted(bundle.validation_brier_scores.items()):
        print(f"- {target}: {score:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

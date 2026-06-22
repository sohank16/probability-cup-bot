from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.current_tournament import load_current_tournament_form
from src.fifa_rankings import normalize_team_name
from src.ml_model import FEATURE_COLUMNS, MODEL_TARGETS, build_training_rows, load_processed_matches
from src.predictor import (
    apply_current_tournament_form,
    choose_market_team,
    fetch_markets,
    get_feature,
    load_team_features,
    ml_feature_row,
)
from src.question_parser import parse_market_question
from src.team_names import split_match_name


DEFAULT_MATCHES_PATH = Path("data/processed/recent_international_matches.csv")
DEFAULT_RANKINGS_PATH = Path("data/external/fifa_rankings_men.csv")
DEFAULT_HISTORICAL_RANKINGS_PATH = Path("data/external/fifa_rankings_men_history.csv")
DEFAULT_TEAM_FEATURES_PATH = Path("data/processed/team_features.csv")
DEFAULT_CURRENT_TOURNAMENT_FORM_PATH = Path("data/processed/current_tournament_results.csv")
DEFAULT_TRAINING_OUTPUT = Path("data/processed/nn_goal_training_dataset.csv")
DEFAULT_CURRENT_OUTPUT = Path("data/processed/nn_current_market_features.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build separate neural-network-ready datasets without changing the live model."
    )
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS_PATH)
    parser.add_argument("--historical-rankings", type=Path, default=DEFAULT_HISTORICAL_RANKINGS_PATH)
    parser.add_argument("--team-features", type=Path, default=DEFAULT_TEAM_FEATURES_PATH)
    parser.add_argument("--current-tournament-form", type=Path, default=DEFAULT_CURRENT_TOURNAMENT_FORM_PATH)
    parser.add_argument("--training-output", type=Path, default=DEFAULT_TRAINING_OUTPUT)
    parser.add_argument("--current-output", type=Path, default=DEFAULT_CURRENT_OUTPUT)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_goal_training_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    matches = load_processed_matches(args.matches)
    training_rows = build_training_rows(
        matches,
        args.rankings,
        args.historical_rankings if args.historical_rankings.exists() else None,
    )
    nn_rows: list[dict[str, object]] = []
    for index, row in enumerate(training_rows):
        for target in MODEL_TARGETS:
            nn_rows.append(
                {
                    "row_id": f"historical_{index}_{target}",
                    "split": "historical_train_candidate",
                    "market_type": target,
                    "metric": "goals",
                    "threshold": "",
                    "period": "match",
                    "label": row[target],
                    **{column: row[column] for column in FEATURE_COLUMNS},
                }
            )
    return nn_rows


def build_current_market_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    settings = load_settings()
    features = load_team_features(args.team_features)
    features = apply_current_tournament_form(
        features,
        load_current_tournament_form(args.current_tournament_form),
    )
    rows: list[dict[str, object]] = []
    for market in fetch_markets(settings.database_path):
        parsed = parse_market_question(market["question"])
        home_team, away_team = split_match_name(market["match_name"])
        team_name, opponent_name = choose_market_team(parsed, home_team, away_team)
        team = get_feature(features, team_name)
        opponent = get_feature(features, opponent_name)
        home = get_feature(features, home_team)
        away = get_feature(features, away_team)
        feature_values = ml_feature_row(home, away)
        rows.append(
            {
                "row_id": market["market_id"],
                "split": "current_unlabeled",
                "match_name": market["match_name"],
                "question": market["question"],
                "market_type": parsed.market_type,
                "metric": parsed.metric or "",
                "threshold": parsed.threshold or "",
                "period": parsed.period,
                "team": team.team,
                "opponent": opponent.team,
                "team_normalized": normalize_team_name(team.team),
                "opponent_normalized": normalize_team_name(opponent.team),
                "label": "",
                **feature_values,
                "team_current_tournament_matches": team.current_tournament_matches,
                "team_current_tournament_points_per_match": team.current_tournament_points_per_match,
                "team_current_tournament_goal_difference_per_match": (
                    team.current_tournament_goal_difference_per_match
                ),
                "opponent_current_tournament_matches": opponent.current_tournament_matches,
                "opponent_current_tournament_points_per_match": (
                    opponent.current_tournament_points_per_match
                ),
                "opponent_current_tournament_goal_difference_per_match": (
                    opponent.current_tournament_goal_difference_per_match
                ),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    training_rows = build_goal_training_rows(args)
    current_rows = build_current_market_rows(args)

    training_fields = [
        "row_id",
        "split",
        "market_type",
        "metric",
        "threshold",
        "period",
        "label",
        *FEATURE_COLUMNS,
    ]
    current_fields = [
        "row_id",
        "split",
        "match_name",
        "question",
        "market_type",
        "metric",
        "threshold",
        "period",
        "team",
        "opponent",
        "team_normalized",
        "opponent_normalized",
        "label",
        *FEATURE_COLUMNS,
        "team_current_tournament_matches",
        "team_current_tournament_points_per_match",
        "team_current_tournament_goal_difference_per_match",
        "opponent_current_tournament_matches",
        "opponent_current_tournament_points_per_match",
        "opponent_current_tournament_goal_difference_per_match",
    ]
    write_rows(args.training_output, training_rows, training_fields)
    write_rows(args.current_output, current_rows, current_fields)

    print("NN dataset build complete.")
    print(f"Historical labeled goal rows: {len(training_rows)}")
    print(f"Current unlabeled market rows: {len(current_rows)}")
    print(f"Training output: {args.training_output}")
    print(f"Current output: {args.current_output}")
    print("")
    print("Note: non-goal prop labels are not available yet; do not train an all-market NN until")
    print("we collect settled market outcomes or scrape reliable SOT/cards/corners/fouls labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

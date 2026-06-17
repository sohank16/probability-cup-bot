from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_priors import clamp_probability
from src.ml_model import (
    build_training_rows,
    load_processed_matches,
    predict_target_probability,
    train_goal_market_models,
)
from src.predictor import draw_probability, poisson_ge


DEFAULT_MATCHES_PATH = Path("data/processed/recent_international_matches.csv")
DEFAULT_RANKINGS_PATH = Path("data/external/fifa_rankings_men.csv")
DEFAULT_HISTORICAL_RANKINGS_PATH = Path("data/external/fifa_rankings_men_history.csv")
DEFAULT_CSV_OUTPUT = Path("reports/world_cup_backtest.csv")
DEFAULT_MD_OUTPUT = Path("reports/world_cup_backtest.md")

TARGETS = [
    "home_win",
    "away_win",
    "over_2_5_goals",
    "both_teams_score",
    "home_scores_1_plus",
    "away_scores_1_plus",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest ML vs statistical baseline on completed 2026 World Cup matches."
    )
    parser.add_argument("--matches", type=Path, default=DEFAULT_MATCHES_PATH)
    parser.add_argument("--rankings", type=Path, default=DEFAULT_RANKINGS_PATH)
    parser.add_argument("--historical-rankings", type=Path, default=DEFAULT_HISTORICAL_RANKINGS_PATH)
    parser.add_argument("--cutoff-date", default="2026-06-11")
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV_OUTPUT)
    parser.add_argument("--md-output", type=Path, default=DEFAULT_MD_OUTPUT)
    return parser.parse_args()


def brier_score(labels: list[int], probabilities: list[float]) -> float:
    return sum((probability - label) ** 2 for label, probability in zip(labels, probabilities)) / len(labels)


def accuracy(labels: list[int], probabilities: list[float]) -> float:
    correct = 0
    for label, probability in zip(labels, probabilities):
        prediction = int(probability >= 0.5)
        correct += int(prediction == label)
    return correct / len(labels)


def statistical_probability(row: dict[str, object], target: str) -> float:
    expected_home = float(row["expected_home_score"])
    elo_diff = float(row["elo_diff"])
    abs_elo_diff = float(row["abs_elo_diff"])
    draw = draw_probability(elo_diff)

    if target == "home_win":
        return clamp_probability((1.0 - draw) * expected_home)

    if target == "away_win":
        return clamp_probability((1.0 - draw) * (1.0 - expected_home))

    if target == "over_2_5_goals":
        total_xg = 2.35 + min(0.55, abs_elo_diff / 850.0)
        return clamp_probability(poisson_ge(total_xg, 3))

    if target == "both_teams_score":
        return clamp_probability(0.54 - min(0.18, abs_elo_diff / 1800.0))

    if target == "home_scores_1_plus":
        return clamp_probability(0.67 + max(-0.22, min(0.22, elo_diff / 1300.0)))

    if target == "away_scores_1_plus":
        return clamp_probability(0.67 + max(-0.22, min(0.22, -elo_diff / 1300.0)))

    return 0.50


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "date",
                "match",
                "target",
                "actual",
                "ml_probability",
                "statistical_probability",
                "home_score",
                "away_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    cutoff_date: str,
    train_count: int,
    test_match_count: int,
    comparison_rows: list[dict[str, object]],
) -> None:
    ensure_parent(path)
    lines = [
        "# 2026 World Cup Backtest",
        "",
        f"- Cutoff date: `{cutoff_date}`",
        f"- Training examples before cutoff: `{train_count}`",
        f"- Completed World Cup matches tested: `{test_match_count}`",
        f"- Prediction rows tested: `{len(comparison_rows)}`",
        "",
        "This compares logistic regression against the statistical baseline on completed 2026 World Cup matches that were excluded from logistic-regression training.",
        "",
        "Historical FIFA ranking snapshots are used when the cached history file is available.",
        "",
        "## Metrics",
        "",
        "| Target | ML Brier | Statistical Brier | ML Accuracy | Statistical Accuracy |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for target in TARGETS:
        target_rows = [row for row in comparison_rows if row["target"] == target]
        labels = [int(row["actual"]) for row in target_rows]
        ml_probabilities = [float(row["ml_probability"]) for row in target_rows]
        stat_probabilities = [float(row["statistical_probability"]) for row in target_rows]
        lines.append(
            "| "
            f"{target} | "
            f"{brier_score(labels, ml_probabilities):.4f} | "
            f"{brier_score(labels, stat_probabilities):.4f} | "
            f"{accuracy(labels, ml_probabilities):.1%} | "
            f"{accuracy(labels, stat_probabilities):.1%} |"
        )

    lines.extend(
        [
            "",
            "## Match-Level Rows",
            "",
            "| Date | Match | Target | Actual | ML | Statistical |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison_rows[:80]:
        lines.append(
            "| "
            f"{row['date']} | {row['match']} | {row['target']} | {row['actual']} | "
            f"{float(row['ml_probability']):.1%} | {float(row['statistical_probability']):.1%} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.matches.exists():
        print(f"Missing processed matches: {args.matches}")
        return 1
    if not args.rankings.exists():
        print(f"Missing FIFA rankings: {args.rankings}")
        return 1

    matches = load_processed_matches(args.matches)
    historical_rankings = args.historical_rankings if args.historical_rankings.exists() else None
    rows = build_training_rows(matches, args.rankings, historical_rankings)
    train_rows = [row for row in rows if str(row["date"]) < args.cutoff_date]
    test_rows = [
        row
        for row in rows
        if str(row["date"]) >= args.cutoff_date and row["tournament"] == "FIFA World Cup"
    ]

    if not test_rows:
        print("No completed 2026 World Cup rows found for backtest.")
        return 1

    bundle = train_goal_market_models(train_rows)
    comparison_rows: list[dict[str, object]] = []

    for row in test_rows:
        features = {column: float(row[column]) for column in bundle.feature_columns}
        match_name = f"{row['home_team']} vs {row['away_team']}"
        for target in TARGETS:
            ml_probability = predict_target_probability(bundle, target, features)
            stat_probability = statistical_probability(row, target)
            comparison_rows.append(
                {
                    "date": row["date"],
                    "match": match_name,
                    "target": target,
                    "actual": int(row[target]),
                    "ml_probability": round(float(ml_probability), 4),
                    "statistical_probability": round(stat_probability, 4),
                    "home_score": row["home_score_actual"],
                    "away_score": row["away_score_actual"],
                }
            )

    write_csv(args.csv_output, comparison_rows)
    write_markdown(
        args.md_output,
        args.cutoff_date,
        len(train_rows),
        len(test_rows),
        comparison_rows,
    )

    print("2026 World Cup backtest complete.")
    print(f"Training examples before cutoff: {len(train_rows)}")
    print(f"Completed World Cup matches tested: {len(test_rows)}")
    print(f"Prediction rows tested: {len(comparison_rows)}")
    print(f"CSV report: {args.csv_output}")
    print(f"Markdown report: {args.md_output}")
    print("")
    print("Brier scores lower is better:")
    for target in TARGETS:
        target_rows = [row for row in comparison_rows if row["target"] == target]
        labels = [int(row["actual"]) for row in target_rows]
        ml_probabilities = [float(row["ml_probability"]) for row in target_rows]
        stat_probabilities = [float(row["statistical_probability"]) for row in target_rows]
        print(
            f"- {target}: ML={brier_score(labels, ml_probabilities):.4f}, "
            f"statistical={brier_score(labels, stat_probabilities):.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

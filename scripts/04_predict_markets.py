from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.predictor import MarketPrediction, predict_markets


DEFAULT_TEAM_FEATURES_PATH = Path("data/processed/team_features.csv")
DEFAULT_ML_MODEL_PATH = Path("models/goal_market_logistic.joblib")
DEFAULT_ODDS_BASELINES_PATH = Path("data/processed/odds_market_baselines.csv")
DEFAULT_PLAYER_FORM_PATH = Path("data/processed/player_form.csv")
DEFAULT_CSV_PATH = Path("reports/dry_run_predictions.csv")
DEFAULT_MD_PATH = Path("reports/dry_run_predictions.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dry-run SportsPredict market predictions.")
    parser.add_argument(
        "--team-features",
        type=Path,
        default=DEFAULT_TEAM_FEATURES_PATH,
        help="Path to generated team_features.csv.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Where to write prediction CSV output.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=DEFAULT_MD_PATH,
        help="Where to write markdown summary output.",
    )
    parser.add_argument(
        "--ml-model",
        type=Path,
        default=DEFAULT_ML_MODEL_PATH,
        help="Optional trained logistic regression bundle for goal/winner markets.",
    )
    parser.add_argument(
        "--odds-baselines",
        type=Path,
        default=DEFAULT_ODDS_BASELINES_PATH,
        help="Optional odds/stat-calibrated baselines from scripts/06_prepare_odds_baselines.py.",
    )
    parser.add_argument(
        "--player-form",
        type=Path,
        default=DEFAULT_PLAYER_FORM_PATH,
        help="Optional player form CSV with club_goals_2025_26 and country_starts_last_10.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_prediction_csv(path: Path, predictions: list[MarketPrediction]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "market_id",
                "match_name",
                "question",
                "market_type",
                "metric",
                "team",
                "opponent",
                "player",
                "probability",
                "confidence",
                "explanation",
            ],
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(asdict(prediction))


def write_prediction_markdown(path: Path, predictions: list[MarketPrediction]) -> None:
    ensure_parent(path)
    type_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    source_counts = {
        "ml_logistic": 0,
        "poisson": 0,
        "baseline_priors": 0,
        "rule_or_prior": 0,
    }
    for prediction in predictions:
        type_counts[prediction.market_type] = type_counts.get(prediction.market_type, 0) + 1
        confidence_counts[prediction.confidence] = confidence_counts.get(prediction.confidence, 0) + 1
        explanation = prediction.explanation.lower()
        if "logistic regression" in explanation or "hybrid ml" in explanation:
            source_counts["ml_logistic"] += 1
        elif "poisson" in explanation:
            source_counts["poisson"] += 1
        elif "baseline" in explanation:
            source_counts["baseline_priors"] += 1
        else:
            source_counts["rule_or_prior"] += 1

    highest = sorted(predictions, key=lambda item: item.probability, reverse=True)[:15]
    lowest = sorted(predictions, key=lambda item: item.probability)[:15]

    lines = [
        "# Dry-Run Predictions",
        "",
        f"- Total predictions: `{len(predictions)}`",
        f"- Minimum probability: `{min(prediction.probability for prediction in predictions):.2%}`",
        f"- Maximum probability: `{max(prediction.probability for prediction in predictions):.2%}`",
        "",
        "## Confidence Counts",
        "",
    ]
    for confidence, count in sorted(confidence_counts.items()):
        lines.append(f"- `{confidence}`: {count}")

    lines.extend(["", "## Model Source Counts", ""])
    for source, count in source_counts.items():
        lines.append(f"- `{source}`: {count}")

    lines.extend(["", "## Market Type Counts", ""])
    for market_type, count in sorted(type_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- `{market_type}`: {count}")

    lines.extend(
        [
            "",
            "## Highest Probabilities",
            "",
            "| Probability | Match | Type | Question |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for prediction in highest:
        lines.append(
            f"| {prediction.probability:.2%} | {prediction.match_name} | "
            f"{prediction.market_type} | {prediction.question} |"
        )

    lines.extend(
        [
            "",
            "## Lowest Probabilities",
            "",
            "| Probability | Match | Type | Question |",
            "| ---: | --- | --- | --- |",
        ]
    )
    for prediction in lowest:
        lines.append(
            f"| {prediction.probability:.2%} | {prediction.match_name} | "
            f"{prediction.market_type} | {prediction.question} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    settings = load_settings()

    if not settings.database_path.exists():
        print(f"Database not found: {settings.database_path}")
        print("Run scripts/01_fetch_sportspredict.py first.")
        return 1

    if not args.team_features.exists():
        print(f"Team features not found: {args.team_features}")
        print("Run scripts/02_prepare_football_data.py first.")
        return 1

    ml_model_path = args.ml_model if args.ml_model.exists() else None
    odds_baselines_path = args.odds_baselines if args.odds_baselines.exists() else None
    player_form_path = args.player_form if args.player_form.exists() else None
    predictions = predict_markets(
        settings.database_path,
        args.team_features,
        ml_model_path,
        odds_baselines_path,
        player_form_path,
    )
    write_prediction_csv(args.csv_output, predictions)
    write_prediction_markdown(args.md_output, predictions)

    print("Dry-run prediction generation complete.")
    print(f"Predictions generated: {len(predictions)}")
    print(f"ML model: {ml_model_path or 'not found; using statistical fallback'}")
    print(f"Odds baselines: {odds_baselines_path or 'not found; using built-in priors'}")
    print(f"Player form: {player_form_path or 'not found; using player baselines'}")
    print(f"CSV report: {args.csv_output}")
    print(f"Markdown report: {args.md_output}")
    print("")
    print("Sample predictions:")
    for prediction in predictions[:10]:
        print(
            f"- {prediction.match_name}: {prediction.probability:.1%} "
            f"({prediction.confidence}) {prediction.question}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

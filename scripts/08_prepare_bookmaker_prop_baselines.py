from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.betting_odds import (
    build_baselines_from_bookmaker_prop_rows,
    load_baselines_csv,
    write_baselines_csv,
)


DEFAULT_INPUT = Path("data/external/bookmaker_prop_odds.csv")
DEFAULT_EXISTING_BASELINES = Path("data/processed/odds_market_baselines.csv")
DEFAULT_OUTPUT = Path("data/processed/odds_market_baselines.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert bookmaker over/under prop odds into de-vigged market baselines."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--existing-baselines", type=Path, default=DEFAULT_EXISTING_BASELINES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--template",
        action="store_true",
        help="Create an empty bookmaker odds template if the input file does not exist.",
    )
    return parser.parse_args()


def write_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "match_name",
                "market_type",
                "metric",
                "period",
                "threshold",
                "bookmaker",
                "over_odds",
                "under_odds",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "match_name": "ARG vs AUT",
                "market_type": "team_metric_over",
                "metric": "shots_on_target",
                "period": "match",
                "threshold": "3",
                "bookmaker": "example",
                "over_odds": "1.80",
                "under_odds": "2.05",
                "source_url": "",
            }
        )


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def main() -> int:
    args = parse_args()
    if args.template and not args.input.exists():
        write_template(args.input)
        print(f"Template written: {args.input}")
        return 0
    if not args.input.exists():
        print(f"Input odds CSV not found: {args.input}")
        print("Create one with --template, then fill real bookmaker odds.")
        return 1

    existing = load_baselines_csv(args.existing_baselines) if args.existing_baselines.exists() else []
    rows = load_rows(args.input)
    new_baselines = build_baselines_from_bookmaker_prop_rows(rows)
    combined = existing + new_baselines
    write_baselines_csv(args.output, combined)

    print("Bookmaker prop baseline preparation complete.")
    print(f"Input rows: {len(rows)}")
    print(f"New de-vigged baselines: {len(new_baselines)}")
    print(f"Total baselines written: {len(combined)}")
    print(f"Output: {args.output}")
    for baseline in new_baselines[:12]:
        print(
            f"- {baseline.market_type}/{baseline.metric} {baseline.period} "
            f">= {baseline.threshold}: {baseline.probability:.1%} "
            f"from {baseline.sample_size} books"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

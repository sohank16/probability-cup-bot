from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.betting_odds import (
    DEFAULT_LEAGUE_CODES,
    DEFAULT_SEASONS,
    build_baselines_from_rows,
    fetch_football_data_csv,
    parse_rows,
    write_baselines_csv,
)


DEFAULT_EXTERNAL_DIR = Path("data/external/football_data")
DEFAULT_OUTPUT = Path("data/processed/odds_market_baselines.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare odds/stat-derived market baselines from Football-Data CSVs."
    )
    parser.add_argument("--external-dir", type=Path, default=DEFAULT_EXTERNAL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use cached CSV files in data/external/football_data.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=DEFAULT_SEASONS,
        help="Football-Data season folders, e.g. 2021 2122 2223 2324 2425 2526.",
    )
    parser.add_argument(
        "--league-codes",
        nargs="+",
        default=DEFAULT_LEAGUE_CODES,
        help="Football-Data league codes, e.g. E0 D1 I1 SP1 F1.",
    )
    return parser.parse_args()


def cached_path(external_dir: Path, season: str, league_code: str) -> Path:
    return external_dir / f"{season}_{league_code}.csv"


def load_or_fetch_csv(args: argparse.Namespace, season: str, league_code: str) -> str | None:
    path = cached_path(args.external_dir, season, league_code)
    if args.skip_download:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8", errors="ignore")

    try:
        text = fetch_football_data_csv(season, league_code)
    except Exception as exc:
        print(f"Skipped {season} {league_code}: {exc}")
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main() -> int:
    args = parse_args()
    all_rows = []
    fetched_files = 0

    for season in args.seasons:
        for league_code in args.league_codes:
            csv_text = load_or_fetch_csv(args, season, league_code)
            if not csv_text:
                continue
            rows = parse_rows(csv_text)
            if not rows:
                continue
            all_rows.extend(rows)
            fetched_files += 1

    if not all_rows:
        print("No Football-Data rows found. Try running without --skip-download.")
        return 1

    baselines = build_baselines_from_rows(all_rows)
    write_baselines_csv(args.output, baselines)

    print("Odds/stat baseline preparation complete.")
    print(f"Football-Data files used: {fetched_files}")
    print(f"Historical match rows used: {len(all_rows)}")
    print(f"Baselines written: {len(baselines)}")
    print(f"Output: {args.output}")
    print("")
    print("Sample baselines:")
    for baseline in baselines[:12]:
        print(
            f"- {baseline.market_type}/{baseline.metric} >= {baseline.threshold}: "
            f"{baseline.probability:.1%} from {baseline.sample_size} samples"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

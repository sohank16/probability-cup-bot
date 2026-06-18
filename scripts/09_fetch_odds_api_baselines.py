from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.betting_odds import (
    OddsBaseline,
    implied_probability_from_decimal_odds,
    load_baselines_csv,
    write_baselines_csv,
)


API_BASE_URL = "https://api.the-odds-api.com/v4"
DEFAULT_OUTPUT = Path("data/processed/odds_market_baselines.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch bookmaker odds from The Odds API and convert them into de-vigged baselines."
    )
    parser.add_argument(
        "--sport-key",
        default="",
        help="The Odds API soccer sport key. Use --list-sports to discover available soccer keys.",
    )
    parser.add_argument(
        "--regions",
        default="uk,eu",
        help="Comma-separated bookmaker regions, for example uk,eu,us,au.",
    )
    parser.add_argument(
        "--markets",
        default="h2h,totals",
        help="Comma-separated odds markets to request. Basic support maps h2h and totals.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append API baselines to existing output instead of replacing it.",
    )
    parser.add_argument(
        "--list-sports",
        action="store_true",
        help="List active sports from The Odds API, then exit.",
    )
    return parser.parse_args()


def api_key_from_env() -> str:
    load_dotenv()
    api_key = os.getenv("ODDS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ODDS_API_KEY is missing. Add it to .env if you want automatic bookmaker odds."
        )
    return api_key


def request_json(path: str, params: dict[str, str]) -> Any:
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=20)
    response.raise_for_status()
    return response.json()


def list_sports(api_key: str) -> None:
    sports = request_json("/sports", {"apiKey": api_key})
    for sport in sports:
        if sport.get("active"):
            print(f"{sport['key']}: {sport.get('title', '')} ({sport.get('group', '')})")


def outcome_price(outcomes: list[dict[str, Any]], name: str) -> float | None:
    for outcome in outcomes:
        if str(outcome.get("name", "")).lower() == name.lower():
            price = outcome.get("price")
            return float(price) if price else None
    return None


def totals_threshold(point: float) -> int:
    # SportsPredict asks "N or more"; bookmaker total 2.5 maps to threshold 3.
    return int(point + 0.5)


def baselines_from_event_odds(events: list[dict[str, Any]]) -> list[OddsBaseline]:
    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    sources: dict[tuple[str, str, str, int], set[str]] = {}

    for event in events:
        for bookmaker in event.get("bookmakers", []):
            bookmaker_name = bookmaker.get("title") or bookmaker.get("key") or "bookmaker"
            for market in bookmaker.get("markets", []):
                key = market.get("key")
                outcomes = market.get("outcomes", [])
                if key == "totals":
                    over_by_point: dict[float, float] = {}
                    under_by_point: dict[float, float] = {}
                    for outcome in outcomes:
                        point = outcome.get("point")
                        price = outcome.get("price")
                        if point is None or price is None:
                            continue
                        if outcome.get("name") == "Over":
                            over_by_point[float(point)] = float(price)
                        if outcome.get("name") == "Under":
                            under_by_point[float(point)] = float(price)
                    for point, over_odds in over_by_point.items():
                        under_odds = under_by_point.get(point)
                        fair = implied_probability_from_decimal_odds([over_odds, under_odds])
                        if fair is None:
                            continue
                        baseline_key = ("total_goals_over", "goals", "match", totals_threshold(point))
                        grouped.setdefault(baseline_key, []).append(fair[0])
                        sources.setdefault(baseline_key, set()).add(bookmaker_name)

                if key == "h2h":
                    home = event.get("home_team")
                    away = event.get("away_team")
                    if not home or not away:
                        continue
                    home_odds = outcome_price(outcomes, home)
                    away_odds = outcome_price(outcomes, away)
                    draw_odds = outcome_price(outcomes, "Draw")
                    fair = implied_probability_from_decimal_odds([home_odds, draw_odds, away_odds])
                    if fair is None:
                        continue
                    for metric, probability in [
                        ("home_win", fair[0]),
                        ("draw", fair[1]),
                        ("away_win", fair[2]),
                    ]:
                        baseline_key = ("match_winner", metric, "match", 1)
                        grouped.setdefault(baseline_key, []).append(probability)
                        sources.setdefault(baseline_key, set()).add(bookmaker_name)

    baselines = []
    for (market_type, metric, period, threshold), probabilities in sorted(grouped.items()):
        source_text = ", ".join(sorted(sources[(market_type, metric, period, threshold)]))
        baselines.append(
            OddsBaseline(
                market_type=market_type,
                metric=metric,
                period=period,
                threshold=threshold,
                probability=sum(probabilities) / len(probabilities),
                sample_size=len(probabilities),
                source=f"The Odds API de-vigged odds: {source_text}",
            )
        )
    return baselines


def main() -> int:
    args = parse_args()
    try:
        api_key = api_key_from_env()
    except RuntimeError as exc:
        print(exc)
        return 1

    if args.list_sports:
        list_sports(api_key)
        return 0

    if not args.sport_key:
        print("Missing --sport-key. Run with --list-sports, choose a soccer key, then rerun.")
        return 1
    if not args.sport_key.startswith("soccer_"):
        print("The sport key should be a soccer key so non-football odds do not pollute the model.")
        return 1

    events = request_json(
        f"/sports/{args.sport_key}/odds",
        {
            "apiKey": api_key,
            "regions": args.regions,
            "markets": args.markets,
            "oddsFormat": "decimal",
        },
    )
    baselines = baselines_from_event_odds(events)
    existing = load_baselines_csv(args.output) if args.append_existing and args.output.exists() else []
    write_baselines_csv(args.output, existing + baselines)

    print("The Odds API baseline preparation complete.")
    print(f"Sport key: {args.sport_key}")
    print(f"Events returned: {len(events)}")
    print(f"Baselines written from API: {len(baselines)}")
    print(f"Output: {args.output}")
    for baseline in baselines[:12]:
        print(
            f"- {baseline.market_type}/{baseline.metric} {baseline.period} "
            f">= {baseline.threshold}: {baseline.probability:.1%} "
            f"from {baseline.sample_size} odds entries"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

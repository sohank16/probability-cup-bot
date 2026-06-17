from __future__ import annotations

import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.question_parser import parse_market_question


def fetch_market_questions(database_path: Path) -> list[tuple[str, str]]:
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT matches.name AS match_name, markets.question
            FROM markets
            LEFT JOIN matches ON markets.match_id = matches.id
            ORDER BY markets.question
            """
        ).fetchall()
    return [(row["match_name"] or "", row["question"]) for row in rows]


def main() -> int:
    settings = load_settings()
    if not settings.database_path.exists():
        print(f"Database not found: {settings.database_path}")
        print("Run scripts/01_fetch_sportspredict.py first.")
        return 1

    market_rows = fetch_market_questions(settings.database_path)
    parsed_rows = [
        (match_name, question, parse_market_question(question))
        for match_name, question in market_rows
    ]

    total_count = len(parsed_rows)
    supported_count = sum(1 for _, _, parsed in parsed_rows if parsed.is_supported)
    type_counts = Counter(parsed.market_type for _, _, parsed in parsed_rows)
    metric_counts = Counter(parsed.metric or "none" for _, _, parsed in parsed_rows)
    examples_by_type: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for match_name, question, parsed in parsed_rows:
        if len(examples_by_type[parsed.market_type]) < 5:
            examples_by_type[parsed.market_type].append((match_name, question))

    print("SportsPredict market parser analysis")
    print(f"Total markets: {total_count}")
    print(f"Parsed markets: {supported_count}")
    print(f"Unknown markets: {total_count - supported_count}")
    print(f"Coverage: {(supported_count / total_count * 100):.1f}%" if total_count else "Coverage: n/a")
    print("")

    print("Market types:")
    for market_type, count in type_counts.most_common():
        print(f"- {market_type}: {count}")

    print("")
    print("Metrics:")
    for metric, count in metric_counts.most_common():
        print(f"- {metric}: {count}")

    print("")
    print("Examples by type:")
    for market_type, examples in sorted(examples_by_type.items()):
        print(f"\n[{market_type}]")
        for match_name, question in examples:
            print(f"- {match_name}: {question}")

    unknown_examples = examples_by_type.get("unknown", [])
    if unknown_examples:
        print("")
        print("Unknown examples to handle next:")
        for match_name, question in unknown_examples:
            print(f"- {match_name}: {question}")

    first_supported = next(
        (parsed for _, _, parsed in parsed_rows if parsed.is_supported),
        None,
    )
    if first_supported:
        print("")
        print("Example parsed object:")
        print(asdict(first_supported))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


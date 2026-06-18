from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.player_form import normalize_player_name
from src.question_parser import parse_market_question
from src.team_names import canonical_team_name, split_match_name


DEFAULT_OUTPUT = Path("data/processed/player_form.csv")
PLAYER_MARKET_TYPES = {
    "player_shot_on_target",
    "player_goal",
    "player_goal_or_assist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a player-form CSV template from open SportsPredict player markets."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def fetch_open_markets(database_path: Path) -> list[sqlite3.Row]:
    now_iso = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        return connection.execute(
            """
            SELECT
                markets.id AS market_id,
                matches.name AS match_name,
                markets.question AS question
            FROM markets
            JOIN matches ON markets.match_id = matches.id
            WHERE markets.status = 'open'
              AND (markets.closing_time IS NULL OR markets.closing_time > ?)
            ORDER BY matches.name, markets.question
            """,
            (now_iso,),
        ).fetchall()


def existing_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as input_file:
        return {
            normalize_player_name(row["player"]): row
            for row in csv.DictReader(input_file)
            if row.get("player")
        }


def write_template(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "player",
                "club_goals_2025_26",
                "country_starts_last_10",
                "source",
                "market_count",
                "market_types",
                "example_matches",
                "example_questions",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    settings = load_settings()
    if not settings.database_path.exists():
        print(f"Database not found: {settings.database_path}")
        print("Run scripts/01_fetch_sportspredict.py first.")
        return 1

    previous = existing_rows(args.output)
    grouped: dict[str, dict[str, object]] = {}
    for market in fetch_open_markets(settings.database_path):
        parsed = parse_market_question(market["question"])
        if parsed.market_type not in PLAYER_MARKET_TYPES or not parsed.player:
            continue
        home_team, away_team = split_match_name(market["match_name"])
        player_as_team = canonical_team_name(parsed.player)
        if player_as_team in {home_team, away_team}:
            continue
        key = normalize_player_name(parsed.player)
        entry = grouped.setdefault(
            key,
            {
                "player": parsed.player,
                "market_types": set(),
                "matches": [],
                "questions": [],
            },
        )
        entry["market_types"].add(parsed.market_type)
        entry["matches"].append(market["match_name"])
        entry["questions"].append(market["question"])

    output_rows: list[dict[str, str]] = []
    for key, entry in sorted(grouped.items(), key=lambda item: item[1]["player"]):
        old = previous.get(key, {})
        matches = list(dict.fromkeys(entry["matches"]))[:4]
        questions = list(dict.fromkeys(entry["questions"]))[:3]
        market_types = sorted(entry["market_types"])
        output_rows.append(
            {
                "player": str(entry["player"]),
                "club_goals_2025_26": old.get("club_goals_2025_26", ""),
                "country_starts_last_10": old.get("country_starts_last_10", ""),
                "source": old.get("source", ""),
                "market_count": str(len(entry["questions"])),
                "market_types": ";".join(market_types),
                "example_matches": "; ".join(matches),
                "example_questions": " | ".join(questions),
            }
        )

    write_template(args.output, output_rows)
    print("Player form template complete.")
    print(f"Players found: {len(output_rows)}")
    print(f"Output: {args.output}")
    print("")
    print("Fill these columns before prediction:")
    print("- club_goals_2025_26")
    print("- country_starts_last_10")
    print("- source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

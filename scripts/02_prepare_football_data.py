from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.elo import rate_matches
from src.features import build_team_features
from src.fifa_rankings import (
    FIFA_MEN_RANKINGS_URL,
    FIFA_RANKING_DATE,
    fetch_fifa_rankings_json,
    fetch_historical_rankings,
    fifa_prior_ratings,
    confederation_lookup,
    historical_confederation_lookup,
    historical_fifa_prior_ratings,
    load_json,
    load_historical_rankings_csv,
    load_rankings_csv,
    parse_fifa_rankings,
    ranking_lookup,
    write_json,
    write_historical_rankings_csv,
    write_rankings_csv,
)
from src.football_data import (
    RESULTS_URL,
    START_DATE,
    combined_match_weight,
    completed_matches_since,
    fetch_results_csv,
    load_csv_text,
    parse_results_csv,
    time_decay_weight,
    write_text,
)


DEFAULT_EXTERNAL_PATH = Path("data/external/international_results.csv")
DEFAULT_FIFA_RANKINGS_JSON_PATH = Path("data/external/fifa_rankings_men.json")
DEFAULT_FIFA_RANKINGS_CSV_PATH = Path("data/external/fifa_rankings_men.csv")
DEFAULT_HISTORICAL_FIFA_RANKINGS_CSV_PATH = Path("data/external/fifa_rankings_men_history.csv")
DEFAULT_MATCHES_PATH = Path("data/processed/recent_international_matches.csv")
DEFAULT_ELO_PATH = Path("data/processed/team_elo_ratings.csv")
DEFAULT_FEATURES_PATH = Path("data/processed/team_features.csv")
DEFAULT_REPORT_PATH = Path("reports/elo_summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare recent international football data and time-weighted Elo ratings."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Use an existing international_results CSV instead of downloading it.",
    )
    parser.add_argument(
        "--external-path",
        type=Path,
        default=DEFAULT_EXTERNAL_PATH,
        help="Where the downloaded raw CSV should be cached.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Read cached results and FIFA rankings and do not call the network.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_match_rows(matches, team_confederations) -> list[dict[str, object]]:
    return [
        {
            "date": match.match_date.isoformat(),
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "tournament": match.tournament,
            "city": match.city,
            "country": match.country,
            "neutral": match.neutral,
            "time_decay_weight": time_decay_weight(match.match_date),
            "match_weight": round(combined_match_weight(match, team_confederations), 3),
        }
        for match in matches
    ]


def build_elo_rows(ratings: dict[str, float], team_features) -> list[dict[str, object]]:
    rows = []
    for team, rating in sorted(ratings.items(), key=lambda item: item[1], reverse=True):
        feature = team_features[team]
        rows.append(
            {
                "team": team,
                "elo": round(rating, 2),
                "fifa_rank": feature.fifa_rank or "",
                "fifa_points": round(feature.fifa_points, 2) if feature.fifa_points else "",
                "confederation": feature.confederation,
                "matches_played_since_2020": feature.matches_played_since_2020,
                "recent_match_count_since_2024": feature.recent_match_count_since_2024,
            }
        )
    return rows


def build_feature_rows(team_features) -> list[dict[str, object]]:
    rows = []
    for feature in sorted(
        team_features.values(),
        key=lambda item: item.team_elo,
        reverse=True,
    ):
        row = asdict(feature)
        row["team_elo"] = round(feature.team_elo, 2)
        row["fifa_rank"] = feature.fifa_rank or ""
        row["fifa_points"] = round(feature.fifa_points, 2) if feature.fifa_points else ""
        row["weighted_points_since_2020"] = round(feature.weighted_points_since_2020, 3)
        row["weighted_goal_difference_since_2020"] = round(
            feature.weighted_goal_difference_since_2020,
            3,
        )
        row["last_5_goals_for"] = round(feature.last_5_goals_for, 3)
        row["last_5_goals_against"] = round(feature.last_5_goals_against, 3)
        row["last_10_win_rate"] = round(feature.last_10_win_rate, 3)
        rows.append(row)
    return rows


def report_lines(
    matches,
    elo_rows: list[dict[str, object]],
    updates,
    fifa_rankings_count: int,
    historical_rankings_count: int,
) -> list[str]:
    latest_match_date = matches[-1].match_date.isoformat() if matches else "none"
    earliest_match_date = matches[0].match_date.isoformat() if matches else "none"
    first_update = updates[0] if updates else None
    latest_update = updates[-1] if updates else None

    lines = [
        "# Time-Weighted Elo Summary",
        "",
        f"- Training window starts: `{START_DATE.isoformat()}`",
        f"- Earliest included match: `{earliest_match_date}`",
        f"- Latest included completed match: `{latest_match_date}`",
        f"- Completed matches used: `{len(matches)}`",
        f"- FIFA ranking prior date: `{FIFA_RANKING_DATE}`",
        f"- FIFA ranked teams loaded: `{fifa_rankings_count}`",
        f"- Historical FIFA ranking rows loaded: `{historical_rankings_count}`",
        "",
        "## Weight Check",
        "",
        "- 2020 matches use `0.25x` time decay.",
        "- 2021 matches use `0.35x` time decay.",
        "- 2022 matches use `0.50x` time decay.",
        "- 2023 matches use `0.70x` time decay.",
        "- 2024 matches use `0.85x` time decay.",
        "- 2025 and 2026 matches use `1.00x` time decay.",
        "- 2022 FIFA World Cup matches receive the highest final match weight even after time decay.",
        "- Euros, Copa America, AFCON, Asian Cup, Gold Cup, and OFC Nations Cup are the second tier.",
        "- Qualifiers and friendlies are lower weight and adjusted by confederation strength.",
        "- Elo expected-score calculations use a light yearly FIFA ranking anchor when historical snapshots are available.",
    ]

    if first_update and latest_update:
        lines.extend(
            [
                "",
                "## Example Elo Updates",
                "",
                (
                    f"- Early sample: `{first_update.match.match_date}` "
                    f"{first_update.match.home_team} vs {first_update.match.away_team}, "
                    f"match weight `{first_update.match_weight:.3f}`, "
                    f"rating change `{first_update.rating_change:.3f}`"
                ),
                (
                    f"- Recent sample: `{latest_update.match.match_date}` "
                    f"{latest_update.match.home_team} vs {latest_update.match.away_team}, "
                    f"match weight `{latest_update.match_weight:.3f}`, "
                    f"rating change `{latest_update.rating_change:.3f}`"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Top 15 Elo Teams",
            "",
            "| Rank | Team | Elo | FIFA Rank | Matches Since 2020 | Matches Since 2024 | Confederation |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for rank, row in enumerate(elo_rows[:15], start=1):
        lines.append(
            "| "
            f"{rank} | {row['team']} | {row['elo']} | "
            f"{row['fifa_rank']} | {row['matches_played_since_2020']} | "
            f"{row['recent_match_count_since_2024']} | {row['confederation']} |"
        )

    lines.extend(
        [
            "",
            "## Bottom 10 Elo Teams",
            "",
            "| Rank | Team | Elo | FIFA Rank | Matches Since 2020 | Matches Since 2024 | Confederation |",
            "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    bottom_rows = elo_rows[-10:]
    for rank, row in enumerate(bottom_rows, start=max(len(elo_rows) - 9, 1)):
        lines.append(
            "| "
            f"{rank} | {row['team']} | {row['elo']} | "
            f"{row['fifa_rank']} | {row['matches_played_since_2020']} | "
            f"{row['recent_match_count_since_2024']} | {row['confederation']} |"
        )

    return lines


def load_or_fetch_rankings(skip_download: bool):
    if skip_download:
        if DEFAULT_FIFA_RANKINGS_CSV_PATH.exists():
            return load_rankings_csv(DEFAULT_FIFA_RANKINGS_CSV_PATH), str(DEFAULT_FIFA_RANKINGS_CSV_PATH)
        if DEFAULT_FIFA_RANKINGS_JSON_PATH.exists():
            payload = load_json(DEFAULT_FIFA_RANKINGS_JSON_PATH)
            return parse_fifa_rankings(payload), str(DEFAULT_FIFA_RANKINGS_JSON_PATH)
        return [], "no cached FIFA rankings found"

    payload = fetch_fifa_rankings_json(FIFA_MEN_RANKINGS_URL)
    rankings = parse_fifa_rankings(payload)
    write_json(DEFAULT_FIFA_RANKINGS_JSON_PATH, payload)
    write_rankings_csv(DEFAULT_FIFA_RANKINGS_CSV_PATH, rankings)
    return rankings, FIFA_MEN_RANKINGS_URL


def load_or_fetch_historical_rankings(skip_download: bool):
    if skip_download:
        if DEFAULT_HISTORICAL_FIFA_RANKINGS_CSV_PATH.exists():
            return (
                load_historical_rankings_csv(DEFAULT_HISTORICAL_FIFA_RANKINGS_CSV_PATH),
                str(DEFAULT_HISTORICAL_FIFA_RANKINGS_CSV_PATH),
            )
        return [], "no cached historical FIFA rankings found"

    rankings = fetch_historical_rankings()
    write_historical_rankings_csv(DEFAULT_HISTORICAL_FIFA_RANKINGS_CSV_PATH, rankings)
    return rankings, "FIFA API historical year-end snapshots"


def main() -> int:
    args = parse_args()

    if args.input:
        csv_text = load_csv_text(args.input)
        source_description = str(args.input)
    elif args.skip_download:
        csv_text = load_csv_text(args.external_path)
        source_description = str(args.external_path)
    else:
        csv_text = fetch_results_csv(RESULTS_URL)
        write_text(args.external_path, csv_text)
        source_description = RESULTS_URL

    fifa_rankings, ranking_source_description = load_or_fetch_rankings(args.skip_download)
    historical_rankings, historical_ranking_source_description = load_or_fetch_historical_rankings(
        args.skip_download
    )
    rankings_by_team = ranking_lookup(fifa_rankings)
    base_ratings = fifa_prior_ratings(fifa_rankings)
    yearly_base_ratings = historical_fifa_prior_ratings(historical_rankings)
    team_confederations = confederation_lookup(fifa_rankings)
    if historical_rankings:
        team_confederations = {**historical_confederation_lookup(historical_rankings), **team_confederations}

    all_completed_matches = parse_results_csv(csv_text)
    recent_matches = completed_matches_since(all_completed_matches, START_DATE)
    if not recent_matches:
        print("No completed matches found from 2020 onward.")
        return 1

    ratings, updates = rate_matches(
        recent_matches,
        base_rating=1350.0,
        base_ratings=base_ratings,
        yearly_base_ratings=yearly_base_ratings,
        team_confederations=team_confederations,
    )
    team_features = build_team_features(
        recent_matches,
        ratings,
        rankings_by_team=rankings_by_team,
        team_confederations=team_confederations,
    )

    match_rows = build_match_rows(recent_matches, team_confederations)
    elo_rows = build_elo_rows(ratings, team_features)
    feature_rows = build_feature_rows(team_features)

    write_rows(
        DEFAULT_MATCHES_PATH,
        match_rows,
        [
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "city",
            "country",
            "neutral",
            "time_decay_weight",
            "match_weight",
        ],
    )
    write_rows(
        DEFAULT_ELO_PATH,
        elo_rows,
        [
            "team",
            "elo",
            "fifa_rank",
            "fifa_points",
            "confederation",
            "matches_played_since_2020",
            "recent_match_count_since_2024",
        ],
    )
    write_rows(
        DEFAULT_FEATURES_PATH,
        feature_rows,
        [
            "team",
            "team_elo",
            "fifa_rank",
            "fifa_points",
            "confederation",
            "weighted_points_since_2020",
            "weighted_goal_difference_since_2020",
            "last_5_points",
            "last_10_points",
            "last_5_goals_for",
            "last_5_goals_against",
            "last_10_win_rate",
            "matches_played_since_2020",
            "recent_match_count_since_2024",
        ],
    )

    report = "\n".join(
        report_lines(
            recent_matches,
            elo_rows,
            updates,
            len(fifa_rankings),
            len(historical_rankings),
        )
    ) + "\n"
    write_text(DEFAULT_REPORT_PATH, report)

    print("Football data preparation complete.")
    print(f"Source: {source_description}")
    print(f"FIFA rankings source: {ranking_source_description}")
    print(f"Historical FIFA rankings source: {historical_ranking_source_description}")
    print(f"FIFA ranking prior date: {FIFA_RANKING_DATE}")
    print(f"Training window starts: {START_DATE.isoformat()}")
    print(f"Earliest included match: {recent_matches[0].match_date.isoformat()}")
    print(f"Latest included completed match: {recent_matches[-1].match_date.isoformat()}")
    print(f"Completed matches used: {len(recent_matches)}")
    print(f"Teams rated: {len(elo_rows)}")
    print(f"Processed matches: {DEFAULT_MATCHES_PATH}")
    print(f"Elo ratings: {DEFAULT_ELO_PATH}")
    print(f"Team features: {DEFAULT_FEATURES_PATH}")
    print(f"Summary report: {DEFAULT_REPORT_PATH}")
    print("")
    print("Top 10 Elo teams:")
    for rank, row in enumerate(elo_rows[:10], start=1):
        print(
            f"{rank:>2}. {row['team']:<24} Elo={row['elo']:<8} "
            f"FIFA rank={str(row['fifa_rank']):<4} "
            f"matches={row['matches_played_since_2020']:<3} "
            f"recent={row['recent_match_count_since_2024']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

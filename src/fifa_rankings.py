from __future__ import annotations

import csv
import json
import unicodedata
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import requests


FIFA_MEN_RANKINGS_URL = (
    "https://api.fifa.com/api/v3/rankings"
    "?gender=1&dateId=FRS_Male_Football_20260401"
)

FIFA_RANKING_DATE = "2026-06-11"

HISTORICAL_FIFA_RANKING_DATE_IDS = {
    2020: "FRS_Male_Football_20201210",
    2021: "FRS_Male_Football_20211223",
    2022: "FRS_Male_Football_20221222",
    2023: "FRS_Male_Football_20231221",
    2024: "FRS_Male_Football_20241219",
    2026: "FRS_Male_Football_20260401",
}

TEAM_NAME_ALIASES = {
    "cabo verde": "cape verde",
    "china pr": "china",
    "chinese taipei": "taiwan",
    "congo dr": "dr congo",
    "cote divoire": "ivory coast",
    "czechia": "czech republic",
    "hong kong china": "hong kong",
    "ir iran": "iran",
    "korea dpr": "north korea",
    "korea republic": "south korea",
    "st kitts and nevis": "saint kitts and nevis",
    "st lucia": "saint lucia",
    "st vincent and the grenadines": "saint vincent and the grenadines",
    "turkiye": "turkey",
    "usa": "united states",
}


@dataclass(frozen=True)
class FifaRanking:
    team: str
    country_code: str
    rank: int
    points: float
    confederation: str


@dataclass(frozen=True)
class HistoricalFifaRanking:
    year: int
    team: str
    country_code: str
    rank: int
    points: float
    confederation: str


def normalize_team_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in ascii_name)
    normalized = " ".join(cleaned.split())
    return TEAM_NAME_ALIASES.get(normalized, normalized)


def fetch_fifa_rankings_json(
    url: str = FIFA_MEN_RANKINGS_URL,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    response = requests.get(
        url,
        timeout=timeout_seconds,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 probability-cup-bot/0.1",
        },
    )
    response.raise_for_status()
    return response.json()


def fifa_rankings_url_for_date_id(date_id: str) -> str:
    return f"https://api.fifa.com/api/v3/rankings?gender=1&dateId={date_id}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_fifa_rankings(payload: dict[str, Any]) -> list[FifaRanking]:
    rankings: list[FifaRanking] = []
    for row in payload.get("Results", []):
        names = row.get("TeamName") or []
        english_name = next(
            (
                item.get("Description")
                for item in names
                if item.get("Locale") in {"en-GB", "en"}
            ),
            None,
        )
        team = english_name or names[0].get("Description")
        if not team:
            continue

        rankings.append(
            FifaRanking(
                team=team,
                country_code=row.get("IdCountry") or "",
                rank=int(row["Rank"]),
                points=float(row["DecimalTotalPoints"]),
                confederation=row.get("ConfederationName") or "",
            )
        )
    return rankings


def rankings_to_csv(rankings: list[FifaRanking]) -> str:
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["team", "country_code", "rank", "points", "confederation"],
    )
    writer.writeheader()
    for ranking in rankings:
        writer.writerow(
            {
                "team": ranking.team,
                "country_code": ranking.country_code,
                "rank": ranking.rank,
                "points": ranking.points,
                "confederation": ranking.confederation,
            }
        )
    return output.getvalue()


def write_rankings_csv(path: Path, rankings: list[FifaRanking]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rankings_to_csv(rankings), encoding="utf-8")


def historical_rankings_to_csv(rankings: list[HistoricalFifaRanking]) -> str:
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["year", "team", "country_code", "rank", "points", "confederation"],
    )
    writer.writeheader()
    for ranking in rankings:
        writer.writerow(
            {
                "year": ranking.year,
                "team": ranking.team,
                "country_code": ranking.country_code,
                "rank": ranking.rank,
                "points": ranking.points,
                "confederation": ranking.confederation,
            }
        )
    return output.getvalue()


def write_historical_rankings_csv(path: Path, rankings: list[HistoricalFifaRanking]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(historical_rankings_to_csv(rankings), encoding="utf-8")


def load_rankings_csv(path: Path) -> list[FifaRanking]:
    reader = csv.DictReader(StringIO(path.read_text(encoding="utf-8")))
    return [
        FifaRanking(
            team=row["team"],
            country_code=row["country_code"],
            rank=int(row["rank"]),
            points=float(row["points"]),
            confederation=row["confederation"],
        )
        for row in reader
    ]


def load_historical_rankings_csv(path: Path) -> list[HistoricalFifaRanking]:
    reader = csv.DictReader(StringIO(path.read_text(encoding="utf-8")))
    return [
        HistoricalFifaRanking(
            year=int(row["year"]),
            team=row["team"],
            country_code=row["country_code"],
            rank=int(row["rank"]),
            points=float(row["points"]),
            confederation=row["confederation"],
        )
        for row in reader
    ]


def fetch_historical_rankings(
    date_ids_by_year: dict[int, str] = HISTORICAL_FIFA_RANKING_DATE_IDS,
) -> list[HistoricalFifaRanking]:
    historical: list[HistoricalFifaRanking] = []
    for year, date_id in sorted(date_ids_by_year.items()):
        payload = fetch_fifa_rankings_json(fifa_rankings_url_for_date_id(date_id))
        for ranking in parse_fifa_rankings(payload):
            historical.append(
                HistoricalFifaRanking(
                    year=year,
                    team=ranking.team,
                    country_code=ranking.country_code,
                    rank=ranking.rank,
                    points=ranking.points,
                    confederation=ranking.confederation,
                )
            )
    return historical


def ranking_lookup(rankings: list[FifaRanking]) -> dict[str, FifaRanking]:
    return {normalize_team_name(ranking.team): ranking for ranking in rankings}


def confederation_lookup(rankings: list[FifaRanking]) -> dict[str, str]:
    return {
        normalize_team_name(ranking.team): ranking.confederation
        for ranking in rankings
    }


def fifa_prior_ratings(rankings: list[FifaRanking]) -> dict[str, float]:
    return {
        normalize_team_name(ranking.team): ranking.points
        for ranking in rankings
    }


def historical_fifa_prior_ratings(
    rankings: list[HistoricalFifaRanking],
) -> dict[int, dict[str, float]]:
    by_year: dict[int, dict[str, float]] = {}
    for ranking in rankings:
        by_year.setdefault(ranking.year, {})[normalize_team_name(ranking.team)] = ranking.points
    return by_year


def historical_confederation_lookup(
    rankings: list[HistoricalFifaRanking],
) -> dict[str, str]:
    confederations: dict[str, str] = {}
    for ranking in sorted(rankings, key=lambda item: item.year):
        confederations[normalize_team_name(ranking.team)] = ranking.confederation
    return confederations


def lookup_ranking(rankings_by_team: dict[str, FifaRanking], team: str) -> FifaRanking | None:
    return rankings_by_team.get(normalize_team_name(team))

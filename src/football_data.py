from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path

import requests

from src.fifa_rankings import normalize_team_name


RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
START_DATE = date(2020, 1, 1)

YEAR_DECAY_WEIGHTS = {
    2020: 0.25,
    2021: 0.35,
    2022: 0.50,
    2023: 0.70,
    2024: 0.85,
}

CONFEDERATION_WEIGHTS = {
    "UEFA": 1.00,
    "CONMEBOL": 1.00,
    "CAF": 0.88,
    "AFC": 0.80,
    "CONCACAF": 0.75,
    "OFC": 0.65,
}

CONTINENTAL_MAJOR_TOURNAMENTS = {
    "uefa euro": "UEFA",
    "copa america": "CONMEBOL",
    "african cup of nations": "CAF",
    "afc asian cup": "AFC",
    "concacaf gold cup": "CONCACAF",
    "ofc nations cup": "OFC",
}


@dataclass(frozen=True)
class FootballMatch:
    """Cleaned international football result used by the model."""

    match_date: date
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    city: str
    country: str
    neutral: bool


def fetch_results_csv(url: str = RESULTS_URL, timeout_seconds: int = 30) -> str:
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    return response.text


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_csv_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def parse_bool(value: str) -> bool:
    return value.strip().upper() == "TRUE"


def parse_score(value: str) -> int | None:
    if value.strip().upper() == "NA":
        return None
    if not value.strip():
        return None
    return int(value)


def parse_match_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def parse_results_csv(csv_text: str) -> list[FootballMatch]:
    matches: list[FootballMatch] = []
    reader = csv.DictReader(StringIO(csv_text))

    for row in reader:
        home_score = parse_score(row["home_score"])
        away_score = parse_score(row["away_score"])
        if home_score is None or away_score is None:
            continue

        matches.append(
            FootballMatch(
                match_date=parse_match_date(row["date"]),
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_score=home_score,
                away_score=away_score,
                tournament=row["tournament"],
                city=row["city"],
                country=row["country"],
                neutral=parse_bool(row["neutral"]),
            )
        )

    return matches


def completed_matches_since(
    matches: list[FootballMatch],
    start_date: date = START_DATE,
    today: date | None = None,
) -> list[FootballMatch]:
    latest_allowed_date = today or date.today()
    return sorted(
        [
            match
            for match in matches
            if start_date <= match.match_date <= latest_allowed_date
        ],
        key=lambda match: match.match_date,
    )


def time_decay_weight(match_date: date) -> float:
    return YEAR_DECAY_WEIGHTS.get(match_date.year, 1.0)


def normalized_tournament(tournament: str) -> str:
    return normalize_team_name(tournament)


def confederation_weight(confederation: str | None) -> float:
    if not confederation:
        return 0.85
    return CONFEDERATION_WEIGHTS.get(confederation.upper(), 0.85)


def match_confederation_weight(
    match: FootballMatch,
    team_confederations: dict[str, str] | None = None,
) -> float:
    if not team_confederations:
        return 0.85

    home_confederation = team_confederations.get(normalize_team_name(match.home_team))
    away_confederation = team_confederations.get(normalize_team_name(match.away_team))
    return (
        confederation_weight(home_confederation)
        + confederation_weight(away_confederation)
    ) / 2.0


def tournament_weight(
    tournament: str,
    match: FootballMatch | None = None,
    team_confederations: dict[str, str] | None = None,
) -> float:
    tournament_key = normalized_tournament(tournament)

    if match and tournament_key == "fifa world cup" and match.match_date.year == 2022:
        return 3.00
    if tournament_key == "fifa world cup":
        return 1.45

    for major_tournament, confederation in CONTINENTAL_MAJOR_TOURNAMENTS.items():
        if tournament_key == major_tournament:
            return 1.30 * confederation_weight(confederation)

    if "world cup qualification" in tournament_key:
        confed_weight = match_confederation_weight(match, team_confederations) if match else 0.85
        return 0.85 * confed_weight

    if "qualification" in tournament_key:
        confed_weight = match_confederation_weight(match, team_confederations) if match else 0.85
        return 0.75 * confed_weight

    if tournament_key == "friendly":
        confed_weight = match_confederation_weight(match, team_confederations) if match else 0.85
        return 0.55 * confed_weight

    if tournament_key == "uefa nations league":
        return 0.95 * confederation_weight("UEFA")

    return 0.90


def combined_match_weight(
    match: FootballMatch,
    team_confederations: dict[str, str] | None = None,
) -> float:
    return time_decay_weight(match.match_date) * tournament_weight(
        match.tournament,
        match,
        team_confederations,
    )

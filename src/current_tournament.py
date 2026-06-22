from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.team_names import canonical_team_name


@dataclass(frozen=True)
class CurrentTournamentForm:
    team: str
    matches: int
    points: int
    goals_for: int
    goals_against: int
    shots_on_target_for: float | None = None
    shots_on_target_against: float | None = None

    @property
    def points_per_match(self) -> float:
        return self.points / max(self.matches, 1)

    @property
    def goal_difference_per_match(self) -> float:
        return (self.goals_for - self.goals_against) / max(self.matches, 1)

    @property
    def goals_for_per_match(self) -> float:
        return self.goals_for / max(self.matches, 1)

    @property
    def goals_against_per_match(self) -> float:
        return self.goals_against / max(self.matches, 1)

    @property
    def shots_on_target_for_per_match(self) -> float | None:
        if self.shots_on_target_for is None:
            return None
        return self.shots_on_target_for / max(self.matches, 1)

    @property
    def shots_on_target_against_per_match(self) -> float | None:
        if self.shots_on_target_against is None:
            return None
        return self.shots_on_target_against / max(self.matches, 1)


def parse_optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)


def points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def add_team_result(
    totals: dict[str, dict[str, float | int | None]],
    team: str,
    goals_for: int,
    goals_against: int,
    shots_on_target_for: float | None,
    shots_on_target_against: float | None,
) -> None:
    row = totals.setdefault(
        canonical_team_name(team),
        {
            "matches": 0,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "shots_on_target_for": None,
            "shots_on_target_against": None,
        },
    )
    row["matches"] = int(row["matches"] or 0) + 1
    row["points"] = int(row["points"] or 0) + points_for(goals_for, goals_against)
    row["goals_for"] = int(row["goals_for"] or 0) + goals_for
    row["goals_against"] = int(row["goals_against"] or 0) + goals_against
    if shots_on_target_for is not None:
        row["shots_on_target_for"] = float(row["shots_on_target_for"] or 0.0) + shots_on_target_for
    if shots_on_target_against is not None:
        row["shots_on_target_against"] = (
            float(row["shots_on_target_against"] or 0.0) + shots_on_target_against
        )


def load_current_tournament_form(path: Path | None) -> dict[str, CurrentTournamentForm]:
    if path is None or not path.exists():
        return {}

    totals: dict[str, dict[str, float | int | None]] = {}
    with path.open("r", newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            if not row.get("home_team") or not row.get("away_team"):
                continue
            home_score = int(row["home_score"])
            away_score = int(row["away_score"])
            home_sot = parse_optional_float(row.get("home_shots_on_target"))
            away_sot = parse_optional_float(row.get("away_shots_on_target"))
            add_team_result(
                totals,
                row["home_team"],
                home_score,
                away_score,
                home_sot,
                away_sot,
            )
            add_team_result(
                totals,
                row["away_team"],
                away_score,
                home_score,
                away_sot,
                home_sot,
            )

    return {
        team: CurrentTournamentForm(
            team=team,
            matches=int(values["matches"] or 0),
            points=int(values["points"] or 0),
            goals_for=int(values["goals_for"] or 0),
            goals_against=int(values["goals_against"] or 0),
            shots_on_target_for=(
                float(values["shots_on_target_for"])
                if values["shots_on_target_for"] is not None
                else None
            ),
            shots_on_target_against=(
                float(values["shots_on_target_against"])
                if values["shots_on_target_against"] is not None
                else None
            ),
        )
        for team, values in totals.items()
    }

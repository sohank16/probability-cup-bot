from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.market_priors import MarketPrior
from src.name_normalization import normalize_player_name
from src.player_prop_model import predict_player_prop
from src.question_parser import ParsedMarket
from src.team_names import canonical_team_name, split_match_name


@dataclass(frozen=True)
class PlayerForm:
    player: str
    club_goals_2025_26: int | None
    country_starts_last_10: int | None
    source: str
    national_team: str | None = None


def parse_optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    return int(float(cleaned))


def infer_national_team(row: dict[str, str]) -> str | None:
    explicit_team = (row.get("national_team") or "").strip()
    if explicit_team:
        return canonical_team_name(explicit_team)

    examples = [
        item.strip()
        for item in (row.get("example_matches") or "").split(";")
        if item.strip()
    ]
    if len(examples) < 2:
        return None

    match_teams = []
    for example in examples:
        try:
            match_teams.append(set(split_match_name(example)))
        except ValueError:
            return None
    common_teams = set.intersection(*match_teams)
    if len(common_teams) != 1:
        return None
    return next(iter(common_teams))


def load_player_form(path: Path) -> dict[str, PlayerForm]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as input_file:
        rows = csv.DictReader(input_file)
        return {
            normalize_player_name(row["player"]): PlayerForm(
                player=row["player"],
                club_goals_2025_26=(
                    parse_optional_int(row.get("club_goals_2025_26"))
                    if "club_goals_2025_26" in row
                    else parse_optional_int(row.get("goals_this_season"))
                ),
                country_starts_last_10=(
                    parse_optional_int(row.get("country_starts_last_10"))
                    if "country_starts_last_10" in row
                    else parse_optional_int(row.get("started_last_2"))
                ),
                source=row.get("source", "").strip(),
                national_team=infer_national_team(row),
            )
            for row in rows
            if row.get("player")
        }


def club_goal_adjustment(club_goals_2025_26: int | None, market_type: str) -> float:
    if club_goals_2025_26 is None:
        return 0.0
    if club_goals_2025_26 >= 25:
        if market_type == "player_shot_on_target":
            return 0.18
        return 0.20 if market_type == "player_goal" else 0.16
    if club_goals_2025_26 >= 15:
        if market_type == "player_shot_on_target":
            return 0.13
        return 0.15 if market_type == "player_goal" else 0.12
    if club_goals_2025_26 >= 8:
        if market_type == "player_shot_on_target":
            return 0.08
        return 0.09 if market_type == "player_goal" else 0.07
    if club_goals_2025_26 >= 3:
        return 0.04 if market_type == "player_shot_on_target" else 0.03
    if club_goals_2025_26 == 0:
        if market_type == "player_shot_on_target":
            return -0.06
        return -0.08 if market_type == "player_goal" else -0.05
    return 0.0


def country_start_adjustment(country_starts_last_10: int | None, market_type: str) -> float:
    if country_starts_last_10 is None:
        return 0.0
    starts = max(0, min(10, country_starts_last_10))
    if market_type == "player_shot_on_target":
        if starts >= 8:
            return 0.20
        if starts >= 6:
            return 0.14
        if starts >= 4:
            return 0.07
        if starts >= 2:
            return -0.04
        return -0.22
    if market_type == "player_goal_or_assist":
        if starts >= 8:
            return 0.16
        if starts >= 6:
            return 0.10
        if starts >= 4:
            return 0.05
        if starts >= 2:
            return -0.04
        return -0.18
    if starts >= 8:
        return 0.10
    if starts >= 6:
        return 0.07
    if starts >= 4:
        return 0.03
    if starts >= 2:
        return -0.03
    return -0.14


def base_player_probability(parsed: ParsedMarket) -> float:
    if parsed.market_type == "player_shot_on_target":
        return 0.20 if parsed.period == "second_half" else 0.36
    if parsed.market_type == "player_goal":
        return 0.18
    if parsed.market_type == "player_goal_or_assist":
        return 0.28
    return 0.50


def player_market_prior(
    parsed: ParsedMarket,
    player_forms: dict[str, PlayerForm],
) -> MarketPrior | None:
    return predict_player_prop(parsed, player_forms)

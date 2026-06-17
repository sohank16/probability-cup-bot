from __future__ import annotations

from dataclasses import dataclass
from math import log10

from src.football_data import FootballMatch, combined_match_weight
from src.fifa_rankings import normalize_team_name


BASE_RATING = 1500.0
BASE_K_FACTOR = 24.0
HOME_ADVANTAGE = 45.0


@dataclass(frozen=True)
class EloMatchUpdate:
    match: FootballMatch
    home_rating_before: float
    away_rating_before: float
    home_rating_after: float
    away_rating_after: float
    expected_home_score: float
    actual_home_score: float
    goal_multiplier: float
    match_weight: float
    rating_change: float


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def actual_score(goals_for: int, goals_against: int) -> float:
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def goal_difference_multiplier(goal_difference: int) -> float:
    difference = abs(goal_difference)
    if difference <= 1:
        return 1.0
    return 1.0 + log10(difference)


def rate_matches(
    matches: list[FootballMatch],
    base_rating: float = BASE_RATING,
    base_k_factor: float = BASE_K_FACTOR,
    home_advantage: float = HOME_ADVANTAGE,
    base_ratings: dict[str, float] | None = None,
    yearly_base_ratings: dict[int, dict[str, float]] | None = None,
    yearly_anchor_strength: float = 0.15,
    team_confederations: dict[str, str] | None = None,
) -> tuple[dict[str, float], list[EloMatchUpdate]]:
    ratings: dict[str, float] = {}
    updates: list[EloMatchUpdate] = []

    def initial_rating(team: str) -> float:
        normalized_team = normalize_team_name(team)
        if base_ratings and normalized_team in base_ratings:
            return base_ratings[normalized_team]
        return base_rating

    def yearly_prior(team: str, year: int) -> float | None:
        if not yearly_base_ratings:
            return None
        normalized_team = normalize_team_name(team)
        if year in yearly_base_ratings and normalized_team in yearly_base_ratings[year]:
            return yearly_base_ratings[year][normalized_team]
        earlier_years = [item for item in yearly_base_ratings if item <= year]
        for prior_year in sorted(earlier_years, reverse=True):
            if normalized_team in yearly_base_ratings[prior_year]:
                return yearly_base_ratings[prior_year][normalized_team]
        return None

    def anchored_rating(team: str, current_rating: float, year: int) -> float:
        prior = yearly_prior(team, year)
        if prior is None:
            return current_rating
        strength = max(0.0, min(1.0, yearly_anchor_strength))
        return current_rating * (1.0 - strength) + prior * strength

    for match in sorted(matches, key=lambda item: item.match_date):
        home_raw_before = ratings.get(match.home_team)
        away_raw_before = ratings.get(match.away_team)
        if home_raw_before is None:
            home_before = yearly_prior(match.home_team, match.match_date.year) or initial_rating(match.home_team)
        else:
            home_before = anchored_rating(match.home_team, home_raw_before, match.match_date.year)
        if away_raw_before is None:
            away_before = yearly_prior(match.away_team, match.match_date.year) or initial_rating(match.away_team)
        else:
            away_before = anchored_rating(match.away_team, away_raw_before, match.match_date.year)

        adjusted_home_rating = home_before if match.neutral else home_before + home_advantage
        expected_home = expected_score(adjusted_home_rating, away_before)
        actual_home = actual_score(match.home_score, match.away_score)
        goal_multiplier = goal_difference_multiplier(match.home_score - match.away_score)
        match_weight = combined_match_weight(match, team_confederations)
        change = base_k_factor * goal_multiplier * match_weight * (actual_home - expected_home)

        home_after = home_before + change
        away_after = away_before - change
        ratings[match.home_team] = home_after
        ratings[match.away_team] = away_after

        updates.append(
            EloMatchUpdate(
                match=match,
                home_rating_before=home_before,
                away_rating_before=away_before,
                home_rating_after=home_after,
                away_rating_after=away_after,
                expected_home_score=expected_home,
                actual_home_score=actual_home,
                goal_multiplier=goal_multiplier,
                match_weight=match_weight,
                rating_change=change,
            )
        )

    return ratings, updates

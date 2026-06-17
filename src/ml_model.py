from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.elo import expected_score, rate_matches
from src.fifa_rankings import (
    confederation_lookup,
    fifa_prior_ratings,
    historical_confederation_lookup,
    historical_fifa_prior_ratings,
    load_historical_rankings_csv,
    load_rankings_csv,
    normalize_team_name,
)
from src.football_data import FootballMatch, combined_match_weight, confederation_weight, parse_bool, parse_match_date


MODEL_TARGETS = [
    "home_win",
    "away_win",
    "over_2_5_goals",
    "both_teams_score",
    "home_scores_1_plus",
    "away_scores_1_plus",
]

FEATURE_COLUMNS = [
    "home_elo_before",
    "away_elo_before",
    "elo_diff",
    "abs_elo_diff",
    "expected_home_score",
    "home_fifa_points",
    "away_fifa_points",
    "fifa_points_diff",
    "match_weight",
    "neutral",
    "home_confederation_weight",
    "away_confederation_weight",
    "home_last_5_points",
    "away_last_5_points",
    "last_5_points_diff",
    "home_last_10_points",
    "away_last_10_points",
    "last_10_points_diff",
    "home_last_5_goals_for",
    "away_last_5_goals_for",
    "last_5_goals_for_diff",
    "home_last_5_goals_against",
    "away_last_5_goals_against",
    "last_5_goals_against_diff",
    "home_last_10_win_rate",
    "away_last_10_win_rate",
    "last_10_win_rate_diff",
    "home_weighted_points_per_match",
    "away_weighted_points_per_match",
    "weighted_points_per_match_diff",
    "home_weighted_goal_diff_per_match",
    "away_weighted_goal_diff_per_match",
    "weighted_goal_diff_per_match_diff",
    "home_matches_last_365_days",
    "away_matches_last_365_days",
    "matches_last_365_days_diff",
    "stat_home_win_probability",
    "stat_away_win_probability",
    "stat_over_2_5_probability",
    "stat_btts_probability",
    "stat_home_scores_1_plus_probability",
    "stat_away_scores_1_plus_probability",
]


@dataclass(frozen=True)
class GoalMarketModelBundle:
    feature_columns: list[str]
    targets: list[str]
    models: dict[str, Pipeline]
    validation_brier_scores: dict[str, float]
    training_rows: int
    validation_rows: int


@dataclass(frozen=True)
class TeamMatchMemory:
    match_date: date
    goals_for: int
    goals_against: int
    points: int
    weight: float


def points_for(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def rolling_team_summary(history: list[TeamMatchMemory], match_date: date) -> dict[str, float]:
    last_5 = history[-5:]
    last_10 = history[-10:]
    weighted_total = sum(item.weight for item in history)
    weighted_points = sum(item.points * item.weight for item in history)
    weighted_goal_diff = sum(
        (item.goals_for - item.goals_against) * item.weight
        for item in history
    )
    matches_last_365 = sum(
        1
        for item in history
        if 0 <= (match_date - item.match_date).days <= 365
    )

    return {
        "last_5_points": float(sum(item.points for item in last_5)),
        "last_10_points": float(sum(item.points for item in last_10)),
        "last_5_goals_for": average([float(item.goals_for) for item in last_5]),
        "last_5_goals_against": average([float(item.goals_against) for item in last_5]),
        "last_10_win_rate": (
            sum(1 for item in last_10 if item.points == 3) / len(last_10)
            if last_10
            else 0.0
        ),
        "weighted_points_per_match": (
            weighted_points / weighted_total
            if weighted_total > 0.0
            else 0.0
        ),
        "weighted_goal_diff_per_match": (
            weighted_goal_diff / weighted_total
            if weighted_total > 0.0
            else 0.0
        ),
        "matches_last_365_days": float(matches_last_365),
    }


def clamp_probability(probability: float) -> float:
    return max(0.01, min(0.99, probability))


def poisson_pmf(lam: float, goals: int) -> float:
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def poisson_cdf(lam: float, goals: int) -> float:
    return sum(poisson_pmf(lam, value) for value in range(goals + 1))


def poisson_ge(lam: float, threshold: int) -> float:
    if threshold <= 0:
        return 1.0
    return 1.0 - poisson_cdf(lam, threshold - 1)


def statistical_probability_features(expected_home: float, elo_diff: float) -> dict[str, float]:
    abs_elo_diff = abs(elo_diff)
    draw = max(0.18, min(0.31, 0.29 - abs_elo_diff / 2500.0))
    total_xg = 2.35 + min(0.55, abs_elo_diff / 850.0)

    return {
        "stat_home_win_probability": clamp_probability((1.0 - draw) * expected_home),
        "stat_away_win_probability": clamp_probability((1.0 - draw) * (1.0 - expected_home)),
        "stat_over_2_5_probability": clamp_probability(poisson_ge(total_xg, 3)),
        "stat_btts_probability": clamp_probability(0.54 - min(0.18, abs_elo_diff / 1800.0)),
        "stat_home_scores_1_plus_probability": clamp_probability(
            0.67 + max(-0.22, min(0.22, elo_diff / 1300.0))
        ),
        "stat_away_scores_1_plus_probability": clamp_probability(
            0.67 + max(-0.22, min(0.22, -elo_diff / 1300.0))
        ),
    }


def load_processed_matches(path: Path) -> list[FootballMatch]:
    matches: list[FootballMatch] = []
    with path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            matches.append(
                FootballMatch(
                    match_date=parse_match_date(row["date"]),
                    home_team=row["home_team"],
                    away_team=row["away_team"],
                    home_score=int(row["home_score"]),
                    away_score=int(row["away_score"]),
                    tournament=row["tournament"],
                    city=row["city"],
                    country=row["country"],
                    neutral=parse_bool(row["neutral"]),
                )
            )
    return matches


def build_training_rows(
    matches: list[FootballMatch],
    rankings_path: Path,
    historical_rankings_path: Path | None = None,
) -> list[dict[str, float | int]]:
    rankings = load_rankings_csv(rankings_path)
    base_ratings = fifa_prior_ratings(rankings)
    team_confederations = confederation_lookup(rankings)
    yearly_base_ratings: dict[int, dict[str, float]] = {}
    if historical_rankings_path and historical_rankings_path.exists():
        historical_rankings = load_historical_rankings_csv(historical_rankings_path)
        yearly_base_ratings = historical_fifa_prior_ratings(historical_rankings)
        team_confederations = {
            **historical_confederation_lookup(historical_rankings),
            **team_confederations,
        }
    rankings_by_team = {normalize_team_name(ranking.team): ranking for ranking in rankings}
    _, updates = rate_matches(
        matches,
        base_rating=1350.0,
        base_ratings=base_ratings,
        yearly_base_ratings=yearly_base_ratings,
        team_confederations=team_confederations,
    )

    rows: list[dict[str, float | int]] = []
    history_by_team: dict[str, list[TeamMatchMemory]] = {}
    for update in updates:
        match = update.match
        home_confederation = team_confederations.get(normalize_team_name(match.home_team))
        away_confederation = team_confederations.get(normalize_team_name(match.away_team))
        home_ranking = rankings_by_team.get(normalize_team_name(match.home_team))
        away_ranking = rankings_by_team.get(normalize_team_name(match.away_team))
        home_summary = rolling_team_summary(
            history_by_team.get(match.home_team, []),
            match.match_date,
        )
        away_summary = rolling_team_summary(
            history_by_team.get(match.away_team, []),
            match.match_date,
        )
        elo_diff = update.home_rating_before - update.away_rating_before
        expected_home = expected_score(update.home_rating_before, update.away_rating_before)
        statistical_features = statistical_probability_features(expected_home, elo_diff)
        total_goals = match.home_score + match.away_score

        rows.append(
            {
                "date": match.match_date.isoformat(),
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_score_actual": match.home_score,
                "away_score_actual": match.away_score,
                "tournament": match.tournament,
                "home_elo_before": update.home_rating_before,
                "away_elo_before": update.away_rating_before,
                "elo_diff": elo_diff,
                "abs_elo_diff": abs(elo_diff),
                "expected_home_score": expected_home,
                "home_fifa_points": home_ranking.points if home_ranking else 0.0,
                "away_fifa_points": away_ranking.points if away_ranking else 0.0,
                "fifa_points_diff": (
                    (home_ranking.points if home_ranking else 0.0)
                    - (away_ranking.points if away_ranking else 0.0)
                ),
                "match_weight": combined_match_weight(match, team_confederations),
                "neutral": float(match.neutral),
                "home_confederation_weight": confederation_weight(home_confederation),
                "away_confederation_weight": confederation_weight(away_confederation),
                "home_last_5_points": home_summary["last_5_points"],
                "away_last_5_points": away_summary["last_5_points"],
                "last_5_points_diff": home_summary["last_5_points"] - away_summary["last_5_points"],
                "home_last_10_points": home_summary["last_10_points"],
                "away_last_10_points": away_summary["last_10_points"],
                "last_10_points_diff": home_summary["last_10_points"] - away_summary["last_10_points"],
                "home_last_5_goals_for": home_summary["last_5_goals_for"],
                "away_last_5_goals_for": away_summary["last_5_goals_for"],
                "last_5_goals_for_diff": home_summary["last_5_goals_for"] - away_summary["last_5_goals_for"],
                "home_last_5_goals_against": home_summary["last_5_goals_against"],
                "away_last_5_goals_against": away_summary["last_5_goals_against"],
                "last_5_goals_against_diff": home_summary["last_5_goals_against"] - away_summary["last_5_goals_against"],
                "home_last_10_win_rate": home_summary["last_10_win_rate"],
                "away_last_10_win_rate": away_summary["last_10_win_rate"],
                "last_10_win_rate_diff": home_summary["last_10_win_rate"] - away_summary["last_10_win_rate"],
                "home_weighted_points_per_match": home_summary["weighted_points_per_match"],
                "away_weighted_points_per_match": away_summary["weighted_points_per_match"],
                "weighted_points_per_match_diff": (
                    home_summary["weighted_points_per_match"]
                    - away_summary["weighted_points_per_match"]
                ),
                "home_weighted_goal_diff_per_match": home_summary["weighted_goal_diff_per_match"],
                "away_weighted_goal_diff_per_match": away_summary["weighted_goal_diff_per_match"],
                "weighted_goal_diff_per_match_diff": (
                    home_summary["weighted_goal_diff_per_match"]
                    - away_summary["weighted_goal_diff_per_match"]
                ),
                "home_matches_last_365_days": home_summary["matches_last_365_days"],
                "away_matches_last_365_days": away_summary["matches_last_365_days"],
                "matches_last_365_days_diff": (
                    home_summary["matches_last_365_days"]
                    - away_summary["matches_last_365_days"]
                ),
                **statistical_features,
                "home_win": int(match.home_score > match.away_score),
                "away_win": int(match.away_score > match.home_score),
                "over_2_5_goals": int(total_goals >= 3),
                "both_teams_score": int(match.home_score >= 1 and match.away_score >= 1),
                "home_scores_1_plus": int(match.home_score >= 1),
                "away_scores_1_plus": int(match.away_score >= 1),
            }
        )

        match_weight = combined_match_weight(match, team_confederations)
        history_by_team.setdefault(match.home_team, []).append(
            TeamMatchMemory(
                match_date=match.match_date,
                goals_for=match.home_score,
                goals_against=match.away_score,
                points=points_for(match.home_score, match.away_score),
                weight=match_weight,
            )
        )
        history_by_team.setdefault(match.away_team, []).append(
            TeamMatchMemory(
                match_date=match.match_date,
                goals_for=match.away_score,
                goals_against=match.home_score,
                points=points_for(match.away_score, match.home_score),
                weight=match_weight,
            )
        )

    return rows


def matrix(rows: list[dict[str, float | int]], columns: list[str]) -> list[list[float]]:
    return [[float(row[column]) for column in columns] for row in rows]


def target_values(rows: list[dict[str, float | int]], target: str) -> list[int]:
    return [int(row[target]) for row in rows]


def train_goal_market_models(
    rows: list[dict[str, float | int]],
    validation_fraction: float = 0.20,
) -> GoalMarketModelBundle:
    if len(rows) < 100:
        raise ValueError("Need at least 100 training rows for logistic regression.")

    split_index = int(len(rows) * (1.0 - validation_fraction))
    train_rows = rows[:split_index]
    validation_rows = rows[split_index:]

    train_x = matrix(train_rows, FEATURE_COLUMNS)
    validation_x = matrix(validation_rows, FEATURE_COLUMNS)

    models: dict[str, Pipeline] = {}
    scores: dict[str, float] = {}

    for target in MODEL_TARGETS:
        train_y = target_values(train_rows, target)
        validation_y = target_values(validation_rows, target)
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "logistic_regression",
                    LogisticRegression(
                        max_iter=1000,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(train_x, train_y)
        validation_probabilities = model.predict_proba(validation_x)[:, 1]
        scores[target] = brier_score_loss(validation_y, validation_probabilities)
        models[target] = model

    return GoalMarketModelBundle(
        feature_columns=list(FEATURE_COLUMNS),
        targets=list(MODEL_TARGETS),
        models=models,
        validation_brier_scores=scores,
        training_rows=len(train_rows),
        validation_rows=len(validation_rows),
    )


def save_model_bundle(path: Path, bundle: GoalMarketModelBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model_bundle(path: Path) -> GoalMarketModelBundle:
    return joblib.load(path)


def predict_target_probability(
    bundle: GoalMarketModelBundle,
    target: str,
    features: dict[str, float],
) -> float | None:
    model = bundle.models.get(target)
    if model is None:
        return None
    row = [[float(features[column]) for column in bundle.feature_columns]]
    return float(model.predict_proba(row)[0, 1])

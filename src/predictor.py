from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.betting_odds import baseline_lookup, load_baselines_csv
from src.elo import expected_score
from src.fifa_rankings import normalize_team_name
from src.football_data import confederation_weight
from src.market_priors import (
    MarketPrior,
    clamp_probability,
    comparison_prior,
    metric_prior,
)
from src.ml_model import (
    GoalMarketModelBundle,
    load_model_bundle,
    predict_target_probability,
    statistical_probability_features,
)
from src.question_parser import ParsedMarket, parse_market_question
from src.team_names import canonical_team_name, opponent_for, split_match_name


@dataclass(frozen=True)
class TeamFeatureRow:
    team: str
    team_elo: float
    fifa_rank: int | None
    fifa_points: float | None
    confederation: str
    weighted_points_since_2020: float
    weighted_goal_difference_since_2020: float
    last_5_points: int
    last_10_points: int
    last_5_goals_for: float
    last_5_goals_against: float
    last_10_win_rate: float
    matches_played_since_2020: int
    recent_match_count_since_2024: int


@dataclass(frozen=True)
class MarketPrediction:
    market_id: str
    match_name: str
    question: str
    market_type: str
    metric: str
    team: str
    opponent: str
    player: str
    probability: float
    confidence: str
    explanation: str


def parse_optional_int(value: str) -> int | None:
    return int(value) if value else None


def parse_optional_float(value: str) -> float | None:
    return float(value) if value else None


def load_team_features(path: Path) -> dict[str, TeamFeatureRow]:
    rows: dict[str, TeamFeatureRow] = {}
    with path.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        for row in reader:
            feature = TeamFeatureRow(
                team=row["team"],
                team_elo=float(row["team_elo"]),
                fifa_rank=parse_optional_int(row["fifa_rank"]),
                fifa_points=parse_optional_float(row["fifa_points"]),
                confederation=row["confederation"],
                weighted_points_since_2020=float(row["weighted_points_since_2020"]),
                weighted_goal_difference_since_2020=float(row["weighted_goal_difference_since_2020"]),
                last_5_points=int(row["last_5_points"]),
                last_10_points=int(row["last_10_points"]),
                last_5_goals_for=float(row["last_5_goals_for"]),
                last_5_goals_against=float(row["last_5_goals_against"]),
                last_10_win_rate=float(row["last_10_win_rate"]),
                matches_played_since_2020=int(row["matches_played_since_2020"]),
                recent_match_count_since_2024=int(row["recent_match_count_since_2024"]),
            )
            rows[normalize_team_name(feature.team)] = feature
    return rows


def fallback_feature(team: str) -> TeamFeatureRow:
    return TeamFeatureRow(
        team=team,
        team_elo=1350.0,
        fifa_rank=None,
        fifa_points=None,
        confederation="",
        weighted_points_since_2020=0.0,
        weighted_goal_difference_since_2020=0.0,
        last_5_points=0,
        last_10_points=0,
        last_5_goals_for=1.0,
        last_5_goals_against=1.3,
        last_10_win_rate=0.25,
        matches_played_since_2020=0,
        recent_match_count_since_2024=0,
    )


def get_feature(features: dict[str, TeamFeatureRow], team: str) -> TeamFeatureRow:
    canonical = canonical_team_name(team)
    return features.get(normalize_team_name(canonical), fallback_feature(canonical))


def poisson_pmf(lam: float, goals: int) -> float:
    return math.exp(-lam) * (lam**goals) / math.factorial(goals)


def poisson_cdf(lam: float, goals: int) -> float:
    return sum(poisson_pmf(lam, value) for value in range(goals + 1))


def poisson_ge(lam: float, threshold: int) -> float:
    if threshold <= 0:
        return 1.0
    return 1.0 - poisson_cdf(lam, threshold - 1)


def expected_goals_for(team: TeamFeatureRow, opponent: TeamFeatureRow) -> float:
    recent_attack = team.last_5_goals_for or 1.1
    opponent_defense = opponent.last_5_goals_against or 1.2
    base = (recent_attack + opponent_defense) / 2.0
    elo_adjustment = (team.team_elo - opponent.team_elo) / 900.0
    return max(0.25, min(3.75, base + elo_adjustment))


def draw_probability(elo_diff: float) -> float:
    return max(0.18, min(0.31, 0.29 - abs(elo_diff) / 2500.0))


def team_win_probability(team: TeamFeatureRow, opponent: TeamFeatureRow) -> float:
    share = expected_score(team.team_elo, opponent.team_elo)
    return clamp_probability((1.0 - draw_probability(team.team_elo - opponent.team_elo)) * share)


def ml_feature_row(home: TeamFeatureRow, away: TeamFeatureRow) -> dict[str, float]:
    home_fifa_points = home.fifa_points or 0.0
    away_fifa_points = away.fifa_points or 0.0
    home_weighted_points_rate = home.weighted_points_since_2020 / max(home.matches_played_since_2020, 1)
    away_weighted_points_rate = away.weighted_points_since_2020 / max(away.matches_played_since_2020, 1)
    home_weighted_goal_diff_rate = home.weighted_goal_difference_since_2020 / max(home.matches_played_since_2020, 1)
    away_weighted_goal_diff_rate = away.weighted_goal_difference_since_2020 / max(away.matches_played_since_2020, 1)
    elo_diff = home.team_elo - away.team_elo
    expected_home = expected_score(home.team_elo, away.team_elo)

    return {
        "home_elo_before": home.team_elo,
        "away_elo_before": away.team_elo,
        "elo_diff": elo_diff,
        "abs_elo_diff": abs(elo_diff),
        "expected_home_score": expected_home,
        "home_fifa_points": home_fifa_points,
        "away_fifa_points": away_fifa_points,
        "fifa_points_diff": home_fifa_points - away_fifa_points,
        "match_weight": 1.45,
        "neutral": 1.0,
        "home_confederation_weight": confederation_weight(home.confederation),
        "away_confederation_weight": confederation_weight(away.confederation),
        "home_last_5_points": float(home.last_5_points),
        "away_last_5_points": float(away.last_5_points),
        "last_5_points_diff": float(home.last_5_points - away.last_5_points),
        "home_last_10_points": float(home.last_10_points),
        "away_last_10_points": float(away.last_10_points),
        "last_10_points_diff": float(home.last_10_points - away.last_10_points),
        "home_last_5_goals_for": home.last_5_goals_for,
        "away_last_5_goals_for": away.last_5_goals_for,
        "last_5_goals_for_diff": home.last_5_goals_for - away.last_5_goals_for,
        "home_last_5_goals_against": home.last_5_goals_against,
        "away_last_5_goals_against": away.last_5_goals_against,
        "last_5_goals_against_diff": home.last_5_goals_against - away.last_5_goals_against,
        "home_last_10_win_rate": home.last_10_win_rate,
        "away_last_10_win_rate": away.last_10_win_rate,
        "last_10_win_rate_diff": home.last_10_win_rate - away.last_10_win_rate,
        "home_weighted_points_per_match": home_weighted_points_rate,
        "away_weighted_points_per_match": away_weighted_points_rate,
        "weighted_points_per_match_diff": home_weighted_points_rate - away_weighted_points_rate,
        "home_weighted_goal_diff_per_match": home_weighted_goal_diff_rate,
        "away_weighted_goal_diff_per_match": away_weighted_goal_diff_rate,
        "weighted_goal_diff_per_match_diff": home_weighted_goal_diff_rate - away_weighted_goal_diff_rate,
        "home_matches_last_365_days": float(home.recent_match_count_since_2024),
        "away_matches_last_365_days": float(away.recent_match_count_since_2024),
        "matches_last_365_days_diff": float(home.recent_match_count_since_2024 - away.recent_match_count_since_2024),
        **statistical_probability_features(expected_home, elo_diff),
    }


def probability_from_ml_model(
    parsed: ParsedMarket,
    team: TeamFeatureRow,
    home: TeamFeatureRow,
    away: TeamFeatureRow,
    model_bundle: GoalMarketModelBundle | None,
) -> tuple[float, str, str] | None:
    if model_bundle is None:
        return None

    features = ml_feature_row(home, away)
    team_is_home = normalize_team_name(team.team) == normalize_team_name(home.team)

    if parsed.market_type == "match_winner":
        target = "home_win" if team_is_home else "away_win"
        probability = predict_target_probability(model_bundle, target, features)
        if probability is not None:
            return (
                probability,
                "high",
                f"Logistic regression predicts {target} from pre-match Elo and match-context features.",
            )

    if parsed.market_type == "total_goals_over" and parsed.threshold == 3 and parsed.period == "match":
        probability = predict_target_probability(model_bundle, "over_2_5_goals", features)
        if probability is not None:
            return (
                probability,
                "high",
                "Logistic regression predicts over 2.5 goals from pre-match Elo and match-context features.",
            )

    if parsed.market_type == "total_goals_under" and parsed.threshold == 2 and parsed.period == "match":
        probability = predict_target_probability(model_bundle, "over_2_5_goals", features)
        if probability is not None:
            return (
                1.0 - probability,
                "high",
                "Logistic regression predicts under 2.5 goals as the complement of over 2.5 goals.",
            )

    if parsed.market_type == "team_score_at_least" and parsed.threshold == 1:
        target = "home_scores_1_plus" if team_is_home else "away_scores_1_plus"
        probability = predict_target_probability(model_bundle, target, features)
        if probability is not None:
            return (
                probability,
                "high",
                f"Logistic regression predicts {target} from pre-match Elo and match-context features.",
            )

    if parsed.market_type == "both_teams_score_and_total_goals" and parsed.threshold == 3:
        btts = predict_target_probability(model_bundle, "both_teams_score", features)
        over = predict_target_probability(model_bundle, "over_2_5_goals", features)
        if btts is not None and over is not None:
            return (
                btts * over,
                "medium",
                "Hybrid ML estimate combines logistic both-teams-score and over 2.5-goals probabilities.",
            )

    return None


def probability_for_goal_market(
    parsed: ParsedMarket,
    team: TeamFeatureRow,
    opponent: TeamFeatureRow,
    home: TeamFeatureRow,
    away: TeamFeatureRow,
) -> tuple[float, str, str]:
    team_xg = expected_goals_for(team, opponent)
    opponent_xg = expected_goals_for(opponent, team)
    home_xg = expected_goals_for(home, away)
    away_xg = expected_goals_for(away, home)

    period_factor = 0.45 if parsed.period in {"halftime", "first_half"} else 0.50 if parsed.period == "second_half" else 1.0

    if parsed.market_type == "match_winner":
        return (
            team_win_probability(team, opponent),
            "high",
            "Match-winner probability uses Elo expected score with a draw adjustment.",
        )

    if parsed.market_type in {"team_score_at_least", "team_score_in_period"}:
        threshold = parsed.threshold or 1
        return (
            poisson_ge(team_xg * period_factor, threshold),
            "high",
            f"Team scoring probability uses Poisson with expected goals {team_xg * period_factor:.2f}.",
        )

    if parsed.market_type == "team_total_goals_over":
        threshold = parsed.threshold or 1
        return (
            poisson_ge(team_xg, threshold),
            "high",
            f"Team total-goals probability uses Poisson with expected goals {team_xg:.2f}.",
        )

    if parsed.market_type == "total_goals_over":
        total_xg = (home_xg + away_xg) * period_factor
        return (
            poisson_ge(total_xg, parsed.threshold or 1),
            "high",
            f"Total-goals over probability uses Poisson with total expected goals {total_xg:.2f}.",
        )

    if parsed.market_type == "total_goals_under":
        total_xg = home_xg + away_xg
        return (
            poisson_cdf(total_xg, parsed.threshold or 0),
            "high",
            f"Total-goals under probability uses Poisson with total expected goals {total_xg:.2f}.",
        )

    if parsed.market_type == "both_teams_score_and_total_goals":
        threshold = parsed.threshold or 1
        both_score = poisson_ge(home_xg, 1) * poisson_ge(away_xg, 1)
        total_over = poisson_ge(home_xg + away_xg, threshold)
        return (
            both_score * total_over,
            "medium",
            "Combined both-teams-score and total-goals probability uses independent Poisson estimates.",
        )

    if parsed.market_type == "halftime_tied":
        home_half_xg = home_xg * 0.45
        away_half_xg = away_xg * 0.45
        probability = sum(
            poisson_pmf(home_half_xg, goals) * poisson_pmf(away_half_xg, goals)
            for goals in range(5)
        )
        return (
            probability,
            "medium",
            "Halftime tied probability sums equal-score Poisson outcomes for the first half.",
        )

    if parsed.market_type == "halftime_team_winning":
        team_half_xg = team_xg * 0.45
        opponent_half_xg = opponent_xg * 0.45
        probability = 0.0
        for team_goals in range(6):
            for opponent_goals in range(6):
                if team_goals > opponent_goals:
                    probability += poisson_pmf(team_half_xg, team_goals) * poisson_pmf(
                        opponent_half_xg,
                        opponent_goals,
                    )
        return (
            probability,
            "medium",
            "Halftime winning probability sums first-half scorelines where the team leads.",
        )

    if parsed.market_type == "second_half_more_goals_than_first":
        return (
            0.45,
            "low",
            "Second-half-more-goals uses a conservative historical-style baseline.",
        )

    return (
        0.50,
        "low",
        "Goal market fell back to neutral probability.",
    )


def choose_market_team(
    parsed: ParsedMarket,
    home_team: str,
    away_team: str,
) -> tuple[str, str]:
    if parsed.team:
        team = canonical_team_name(parsed.team)
        opponent = parsed.opponent and canonical_team_name(parsed.opponent)
        return team, opponent or opponent_for(team, home_team, away_team) or away_team
    return home_team, away_team


def predict_parsed_market(
    parsed: ParsedMarket,
    match_name: str,
    features: dict[str, TeamFeatureRow],
    model_bundle: GoalMarketModelBundle | None = None,
    odds_baselines=None,
) -> tuple[float, str, str, str, str]:
    home_team, away_team = split_match_name(match_name)
    team_name, opponent_name = choose_market_team(parsed, home_team, away_team)
    team = get_feature(features, team_name)
    opponent = get_feature(features, opponent_name)
    home = get_feature(features, home_team)
    away = get_feature(features, away_team)
    elo_diff = team.team_elo - opponent.team_elo

    ml_prediction = probability_from_ml_model(parsed, team, home, away, model_bundle)
    if ml_prediction is not None:
        probability, confidence, explanation = ml_prediction
    elif parsed.metric == "goals" or parsed.market_type in {
        "match_winner",
        "both_teams_score_and_total_goals",
        "halftime_tied",
        "halftime_team_winning",
        "second_half_more_goals_than_first",
    }:
        probability, confidence, explanation = probability_for_goal_market(
            parsed,
            team,
            opponent,
            home,
            away,
        )
    elif parsed.market_type == "team_metric_more_than_opponent":
        prior = comparison_prior(parsed, elo_diff)
        probability, confidence, explanation = prior.probability, prior.confidence, prior.explanation
    elif parsed.market_type == "both_teams_metric_at_least":
        prior = metric_prior(parsed, 0.0, odds_baselines)
        probability, confidence, explanation = prior.probability, prior.confidence, prior.explanation
    else:
        prior = metric_prior(parsed, elo_diff, odds_baselines)
        probability, confidence, explanation = prior.probability, prior.confidence, prior.explanation

    return (
        clamp_probability(probability),
        confidence,
        explanation,
        team.team,
        opponent.team,
    )


def fetch_markets(database_path: Path) -> list[sqlite3.Row]:
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


def predict_markets(
    database_path: Path,
    team_features_path: Path,
    ml_model_path: Path | None = None,
    odds_baselines_path: Path | None = None,
) -> list[MarketPrediction]:
    features = load_team_features(team_features_path)
    model_bundle = load_model_bundle(ml_model_path) if ml_model_path and ml_model_path.exists() else None
    odds_baselines = (
        baseline_lookup(load_baselines_csv(odds_baselines_path))
        if odds_baselines_path and odds_baselines_path.exists()
        else None
    )
    predictions: list[MarketPrediction] = []

    for row in fetch_markets(database_path):
        parsed = parse_market_question(row["question"])
        probability, confidence, explanation, team, opponent = predict_parsed_market(
            parsed,
            row["match_name"],
            features,
            model_bundle,
            odds_baselines,
        )
        predictions.append(
            MarketPrediction(
                market_id=row["market_id"],
                match_name=row["match_name"],
                question=row["question"],
                market_type=parsed.market_type,
                metric=parsed.metric or "",
                team=team,
                opponent=opponent,
                player=parsed.player or "",
                probability=round(probability, 4),
                confidence=confidence,
                explanation=explanation,
            )
        )

    return predictions

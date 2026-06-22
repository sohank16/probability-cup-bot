from __future__ import annotations

import csv
import math
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from src.betting_odds import baseline_lookup, load_baselines_csv
from src.current_tournament import CurrentTournamentForm, load_current_tournament_form
from src.elo import expected_score
from src.exact_market_odds import blend_with_exact_odds, load_exact_market_odds
from src.fifa_rankings import normalize_team_name
from src.football_data import confederation_weight
from src.market_priors import (
    clamp_probability,
    metric_prior,
)
from src.market_router import market_route
from src.ml_model import (
    GoalMarketModelBundle,
    load_model_bundle,
    predict_target_probability,
    statistical_probability_features,
)
from src.name_normalization import normalize_player_name
from src.player_form import PlayerForm, load_player_form
from src.player_prop_model import predict_player_prop
from src.question_parser import ParsedMarket, parse_market_question
from src.risk_control import apply_risk_control
from src.starting_xi import StartingXIStatus, key_for, load_starting_xi
from src.team_stat_model import predict_team_stat_prop
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
    current_tournament_matches: int = 0
    current_tournament_points_per_match: float = 0.0
    current_tournament_goal_difference_per_match: float = 0.0
    current_tournament_goals_for_per_match: float = 0.0
    current_tournament_goals_against_per_match: float = 0.0
    current_tournament_shots_on_target_for_per_match: float | None = None
    current_tournament_shots_on_target_against_per_match: float | None = None


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


def apply_current_tournament_form(
    features: dict[str, TeamFeatureRow],
    current_form: dict[str, CurrentTournamentForm],
) -> dict[str, TeamFeatureRow]:
    if not current_form:
        return features
    updated = dict(features)
    for team, form in current_form.items():
        key = normalize_team_name(team)
        base = updated.get(key, fallback_feature(team))
        updated[key] = replace(
            base,
            current_tournament_matches=form.matches,
            current_tournament_points_per_match=form.points_per_match,
            current_tournament_goal_difference_per_match=form.goal_difference_per_match,
            current_tournament_goals_for_per_match=form.goals_for_per_match,
            current_tournament_goals_against_per_match=form.goals_against_per_match,
            current_tournament_shots_on_target_for_per_match=form.shots_on_target_for_per_match,
            current_tournament_shots_on_target_against_per_match=form.shots_on_target_against_per_match,
        )
    return updated


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


def poisson_greater_than(team_lambda: float, opponent_lambda: float, max_goals: int = 8) -> float:
    probability = 0.0
    for team_goals in range(max_goals + 1):
        for opponent_goals in range(max_goals + 1):
            if team_goals > opponent_goals:
                probability += poisson_pmf(team_lambda, team_goals) * poisson_pmf(
                    opponent_lambda,
                    opponent_goals,
                )
    return probability


def expected_goals_for(team: TeamFeatureRow, opponent: TeamFeatureRow) -> float:
    recent_attack = team.last_5_goals_for or 1.1
    opponent_defense = opponent.last_5_goals_against or 1.2
    base = (recent_attack + opponent_defense) / 2.0
    elo_adjustment = (team.team_elo - opponent.team_elo) / 900.0
    current_attack = 0.0
    current_defense = 0.0
    if team.current_tournament_matches:
        current_attack = max(-0.30, min(0.35, (team.current_tournament_goals_for_per_match - 1.35) * 0.16))
    if opponent.current_tournament_matches:
        current_defense = max(
            -0.25,
            min(0.30, (opponent.current_tournament_goals_against_per_match - 1.25) * 0.14),
        )
    return max(0.25, min(3.75, base + elo_adjustment + current_attack + current_defense))


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

    if parsed.market_type == "team_metric_more_than_opponent" and parsed.metric == "goals":
        return (
            poisson_greater_than(team_xg * period_factor, opponent_xg * period_factor),
            "medium",
            "Goal comparison probability uses Poisson scoreline sums for the requested period.",
        )

    if parsed.market_type == "team_first_goal_of_second_half":
        team_half_xg = team_xg * 0.50
        opponent_half_xg = opponent_xg * 0.50
        second_half_goal_probability = 1.0 - math.exp(-(team_half_xg + opponent_half_xg))
        team_share = team_half_xg / max(team_half_xg + opponent_half_xg, 0.01)
        return (
            second_half_goal_probability * team_share,
            "medium",
            "Second-half first-goal probability uses second-half expected-goal share.",
        )

    if parsed.market_type == "team_first_goal_and_opponent_second_half_score":
        first_goal_share = team_xg / max(team_xg + opponent_xg, 0.01)
        match_has_goal = 1.0 - math.exp(-(team_xg + opponent_xg))
        opponent_second_half_score = poisson_ge(opponent_xg * 0.50, 1)
        return (
            first_goal_share * match_has_goal * opponent_second_half_score,
            "medium",
            "Combo market multiplies first-goal share by opponent second-half scoring probability.",
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
        total_xg = home_xg + away_xg
        return (
            poisson_greater_than(total_xg * 0.55, total_xg * 0.45),
            "medium",
            "Second-half-more-goals probability compares first-half and second-half Poisson goal totals.",
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


def contextualize_parsed_market(parsed: ParsedMarket, match_name: str) -> ParsedMarket:
    if parsed.market_type != "player_shot_on_target" or not parsed.player:
        return parsed
    home_team, away_team = split_match_name(match_name)
    player_as_team = canonical_team_name(parsed.player)
    if player_as_team not in {home_team, away_team}:
        return parsed
    return replace(
        parsed,
        market_type="team_metric_over",
        team=player_as_team,
        player=None,
    )


def predict_parsed_market(
    parsed: ParsedMarket,
    match_name: str,
    features: dict[str, TeamFeatureRow],
    model_bundle: GoalMarketModelBundle | None = None,
    odds_baselines=None,
    player_forms: dict[str, PlayerForm] | None = None,
    starting_xi_status: StartingXIStatus | None = None,
) -> tuple[float, str, str, str, str]:
    parsed = contextualize_parsed_market(parsed, match_name)
    home_team, away_team = split_match_name(match_name)
    route = market_route(parsed)
    player_team_feature: TeamFeatureRow | None = None
    player_opponent_feature: TeamFeatureRow | None = None
    team_name, opponent_name = choose_market_team(parsed, home_team, away_team)
    if route == "player_prop" and parsed.player:
        form = (player_forms or {}).get(normalize_player_name(parsed.player))
        player_national_team = getattr(form, "national_team", None) if form else None
        if player_national_team:
            player_team = canonical_team_name(player_national_team)
            player_opponent = opponent_for(player_team, home_team, away_team)
            if player_opponent:
                team_name, opponent_name = player_team, player_opponent

    team = get_feature(features, team_name)
    opponent = get_feature(features, opponent_name)
    if route == "player_prop" and parsed.player:
        form = (player_forms or {}).get(normalize_player_name(parsed.player))
        player_national_team = getattr(form, "national_team", None) if form else None
        if player_national_team and opponent_for(player_national_team, home_team, away_team):
            player_team_feature = team
            player_opponent_feature = opponent
    home = get_feature(features, home_team)
    away = get_feature(features, away_team)
    elo_diff = team.team_elo - opponent.team_elo
    ml_prediction = probability_from_ml_model(parsed, team, home, away, model_bundle)
    player_prediction = predict_player_prop(
        parsed,
        player_forms or {},
        player_team_feature,
        player_opponent_feature,
        starting_xi_status,
    )
    if route == "player_prop" and player_prediction is not None:
        probability, confidence, explanation = (
            player_prediction.probability,
            player_prediction.confidence,
            player_prediction.explanation,
        )
    elif ml_prediction is not None:
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
    elif route == "team_stat_prop":
        prior = predict_team_stat_prop(parsed, team, opponent, odds_baselines)
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
    player_form_path: Path | None = None,
    current_tournament_form_path: Path | None = None,
    exact_market_odds_path: Path | None = None,
    starting_xi_path: Path | None = None,
) -> list[MarketPrediction]:
    features = load_team_features(team_features_path)
    features = apply_current_tournament_form(
        features,
        load_current_tournament_form(current_tournament_form_path),
    )
    model_bundle = load_model_bundle(ml_model_path) if ml_model_path and ml_model_path.exists() else None
    odds_baselines = (
        baseline_lookup(load_baselines_csv(odds_baselines_path))
        if odds_baselines_path and odds_baselines_path.exists()
        else None
    )
    player_forms = (
        load_player_form(player_form_path)
        if player_form_path and player_form_path.exists()
        else {}
    )
    exact_market_odds = load_exact_market_odds(exact_market_odds_path)
    starting_xi = load_starting_xi(starting_xi_path)
    predictions: list[MarketPrediction] = []

    for row in fetch_markets(database_path):
        parsed = contextualize_parsed_market(
            parse_market_question(row["question"]),
            row["match_name"],
        )
        starting_status = (
            starting_xi.get(key_for(row["match_name"], parsed.player))
            if parsed.player
            else None
        )
        probability, confidence, explanation, team, opponent = predict_parsed_market(
            parsed,
            row["match_name"],
            features,
            model_bundle,
            odds_baselines,
            player_forms,
            starting_status,
        )
        exact_odds = exact_market_odds.get(row["market_id"])
        if exact_odds is not None:
            original_probability = probability
            probability = blend_with_exact_odds(probability, exact_odds)
            confidence = "high" if exact_odds.sample_size >= 2 else "medium"
            explanation += (
                f" Exact odds baseline blended {original_probability:.1%} with "
                f"{exact_odds.probability:.1%} from {exact_odds.source}."
            )
        controlled = apply_risk_control(
            probability,
            parsed,
            confidence,
            has_exact_odds=exact_odds is not None,
            has_confirmed_lineup=bool(starting_status and starting_status.is_confirmed),
        )
        probability = controlled.probability
        explanation += controlled.explanation_suffix
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

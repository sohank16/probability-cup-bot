from __future__ import annotations

from typing import Any

from src.market_priors import MarketPrior, comparison_prior, metric_prior
from src.probability_utils import (
    add_logit_adjustment,
    clamp_probability,
    poisson_ge,
    poisson_greater_than,
    shrink_toward,
)
from src.question_parser import ParsedMarket


def period_factor(period: str) -> float:
    if period in {"halftime", "first_half"}:
        return 0.46
    if period == "second_half":
        return 0.50
    return 1.0


def attacking_style_score(team: Any, opponent: Any) -> float:
    elo_diff = float(team.team_elo - opponent.team_elo)
    attack = float(team.last_5_goals_for or 0.0)
    opponent_defense = float(opponent.last_5_goals_against or 0.0)
    form = float(team.last_10_win_rate or 0.0)
    weighted_goal_diff = (
        float(team.weighted_goal_difference_since_2020)
        / max(int(team.matches_played_since_2020), 1)
    )
    return (
        max(-0.45, min(0.45, elo_diff / 900.0))
        + max(-0.25, min(0.25, (attack - 1.35) * 0.22))
        + max(-0.20, min(0.20, (opponent_defense - 1.25) * 0.18))
        + max(-0.18, min(0.18, (form - 0.45) * 0.35))
        + max(-0.18, min(0.18, weighted_goal_diff * 0.16))
    )


def expected_team_shots_on_target(team: Any, opponent: Any, period: str = "match") -> float:
    elo_diff = float(team.team_elo - opponent.team_elo)
    attack = float(team.last_5_goals_for or 1.2)
    opponent_defense = float(opponent.last_5_goals_against or 1.2)
    form = float(team.last_10_win_rate or 0.45)
    weighted_goal_diff = (
        float(team.weighted_goal_difference_since_2020)
        / max(int(team.matches_played_since_2020), 1)
    )

    full_match_sot = (
        3.85
        + max(-1.55, min(1.85, elo_diff / 260.0))
        + max(-0.55, min(0.70, (attack - 1.35) * 0.55))
        + max(-0.45, min(0.55, (opponent_defense - 1.25) * 0.45))
        + max(-0.35, min(0.45, (form - 0.45) * 1.05))
        + max(-0.40, min(0.45, weighted_goal_diff * 0.45))
    )
    if getattr(team, "current_tournament_matches", 0):
        current_sot = getattr(team, "current_tournament_shots_on_target_for_per_match", None)
        if current_sot is not None:
            full_match_sot += max(-0.80, min(0.85, (float(current_sot) - 3.85) * 0.45))
        full_match_sot += max(
            -0.40,
            min(0.45, float(getattr(team, "current_tournament_goal_difference_per_match", 0.0)) * 0.25),
        )
        full_match_sot += max(
            -0.35,
            min(0.35, (float(getattr(team, "current_tournament_points_per_match", 0.0)) - 1.3) * 0.18),
        )
    if getattr(opponent, "current_tournament_matches", 0):
        current_sot_allowed = getattr(opponent, "current_tournament_shots_on_target_against_per_match", None)
        if current_sot_allowed is not None:
            full_match_sot += max(-0.55, min(0.60, (float(current_sot_allowed) - 3.85) * 0.35))
        full_match_sot += max(
            -0.25,
            min(0.30, float(getattr(opponent, "current_tournament_goals_against_per_match", 0.0) - 1.2) * 0.16),
        )
    return max(0.85, min(7.60, full_match_sot)) * period_factor(period)


def current_form_note(team: Any, opponent: Any) -> str:
    parts = []
    for item in (team, opponent):
        matches = int(getattr(item, "current_tournament_matches", 0) or 0)
        if matches:
            parts.append(
                f"{item.team}: {matches} WC matches, "
                f"{getattr(item, 'current_tournament_points_per_match', 0.0):.2f} ppm, "
                f"{getattr(item, 'current_tournament_goal_difference_per_match', 0.0):+.2f} gd/match"
            )
    if not parts:
        return ""
    return " Current WC form included (" + "; ".join(parts) + ")."


def underdog_pressure_score(team: Any, opponent: Any) -> float:
    elo_diff = float(team.team_elo - opponent.team_elo)
    defensive_stress = float(team.last_5_goals_against or 0.0) - 1.2
    return max(-0.42, min(0.42, -elo_diff / 1050.0)) + max(
        -0.16,
        min(0.16, defensive_stress * 0.14),
    )


def metric_adjustment(parsed: ParsedMarket, team: Any, opponent: Any) -> float:
    metric = parsed.metric or ""
    threshold = parsed.threshold or 0
    if metric in {"shots_on_target", "corners", "offsides"}:
        adjustment = attacking_style_score(team, opponent)
        if metric == "corners":
            adjustment *= 0.75
        if metric == "offsides":
            adjustment *= 0.55
        if parsed.market_type == "team_metric_over":
            if metric == "shots_on_target":
                adjustment -= max(0, threshold - 3) * 0.16
            if metric == "corners":
                adjustment -= max(0, threshold - 5) * 0.13
            if metric == "offsides":
                adjustment -= max(0, threshold - 2) * 0.18
        return adjustment
    if metric in {"cards", "fouls"}:
        adjustment = underdog_pressure_score(team, opponent)
        if metric == "cards":
            adjustment *= 0.85
            adjustment -= max(0, threshold - 1) * 0.22 if parsed.market_type == "team_metric_over" else 0.0
        return adjustment
    return 0.0


def predict_shots_on_target_prop(parsed: ParsedMarket, team: Any, opponent: Any) -> MarketPrior:
    team_sot = expected_team_shots_on_target(team, opponent, parsed.period)
    opponent_sot = expected_team_shots_on_target(opponent, team, parsed.period)
    threshold = parsed.threshold or 1

    if parsed.market_type == "team_metric_more_than_opponent":
        baseline = comparison_prior(parsed, float(team.team_elo - opponent.team_elo)).probability
        raw_probability = poisson_greater_than(team_sot, opponent_sot)
        probability = shrink_toward(raw_probability, baseline, 0.68)
        probability = clamp_probability(probability, 0.08, 0.84)
        explanation = (
            "SOT specialist compares expected shots on target for both teams "
            f"({team.team}: {team_sot:.2f}, {opponent.team}: {opponent_sot:.2f}) using Poisson, "
            "then shrinks toward the prior because half-level SOT variance is high."
            + current_form_note(team, opponent)
        )
    elif parsed.market_type == "team_metric_over":
        baseline = metric_prior(parsed, float(team.team_elo - opponent.team_elo)).probability
        raw_probability = poisson_ge(team_sot, threshold)
        probability = shrink_toward(raw_probability, baseline, 0.78)
        if threshold >= 6:
            probability = clamp_probability(probability, 0.06, 0.76)
        elif threshold >= 4:
            probability = clamp_probability(probability, 0.08, 0.84)
        else:
            probability = clamp_probability(probability, 0.10, 0.88)
        explanation = (
            f"SOT specialist estimates {team.team} expected shots on target at {team_sot:.2f} "
            f"and applies calibrated Poisson for threshold {threshold}+."
            + current_form_note(team, opponent)
        )
    elif parsed.market_type == "total_metric_over":
        total_sot = team_sot + opponent_sot
        baseline = metric_prior(parsed, 0.0).probability
        raw_probability = poisson_ge(total_sot, threshold)
        probability = shrink_toward(raw_probability, baseline, 0.78)
        probability = clamp_probability(probability, 0.10, 0.82 if threshold >= 8 else 0.86)
        explanation = (
            f"SOT specialist estimates total expected shots on target at {total_sot:.2f} "
            f"and applies calibrated Poisson for threshold {threshold}+."
            + current_form_note(team, opponent)
        )
    elif parsed.market_type == "both_teams_metric_at_least":
        baseline = metric_prior(parsed, 0.0).probability
        raw_probability = poisson_ge(team_sot, threshold) * poisson_ge(opponent_sot, threshold)
        probability = shrink_toward(raw_probability, baseline, 0.70)
        probability = clamp_probability(probability, 0.12, 0.72)
        explanation = (
            "SOT specialist multiplies each team's probability of clearing the threshold "
            f"({team.team}: {team_sot:.2f}, {opponent.team}: {opponent_sot:.2f}) "
            "and shrinks toward the prior."
            + current_form_note(team, opponent)
        )
    else:
        probability = 0.50
        explanation = "SOT specialist fell back to neutral because the SOT market shape was unknown."

    return MarketPrior(
        clamp_probability(probability),
        "medium",
        explanation,
    )


def predict_team_stat_prop(
    parsed: ParsedMarket,
    team: Any,
    opponent: Any,
    odds_baselines: dict | None = None,
) -> MarketPrior:
    if parsed.metric == "shots_on_target":
        return predict_shots_on_target_prop(parsed, team, opponent)

    elo_diff = float(team.team_elo - opponent.team_elo)
    if parsed.market_type == "team_metric_more_than_opponent":
        base = comparison_prior(parsed, elo_diff)
        anchor = 0.50
    else:
        base = metric_prior(parsed, elo_diff, odds_baselines)
        anchor = base.probability if "odds" in base.explanation.lower() else 0.50

    adjusted = add_logit_adjustment(
        base.probability,
        metric_adjustment(parsed, team, opponent),
    )
    confidence_strength = 0.78 if base.confidence == "medium" else 0.64
    if "odds" in base.explanation.lower():
        confidence_strength = 0.88
    probability = shrink_toward(adjusted, anchor, confidence_strength)
    confidence = "medium" if base.confidence == "medium" or odds_baselines else "low"
    return MarketPrior(
        clamp_probability(probability),
        confidence,
        (
            "Team stat prop model combines market baseline, threshold, "
            f"{parsed.metric or 'metric'} style pressure, and confidence shrinkage. "
            f"Base: {base.explanation}"
        ),
    )

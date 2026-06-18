from __future__ import annotations

from typing import Any

from src.market_priors import MarketPrior, comparison_prior, metric_prior
from src.probability_utils import add_logit_adjustment, clamp_probability, shrink_toward
from src.question_parser import ParsedMarket


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


def predict_team_stat_prop(
    parsed: ParsedMarket,
    team: Any,
    opponent: Any,
    odds_baselines: dict | None = None,
) -> MarketPrior:
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

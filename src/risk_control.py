from __future__ import annotations

from dataclasses import dataclass

from src.probability_utils import clamp_probability, shrink_toward
from src.question_parser import ParsedMarket


@dataclass(frozen=True)
class RiskControlledProbability:
    probability: float
    explanation_suffix: str


HIGH_VARIANCE_MARKETS = {
    "player_goal",
    "player_goal_or_assist",
    "penalty_awarded",
    "penalty_or_red_card",
    "team_first_goal_of_second_half",
    "team_first_goal_and_opponent_second_half_score",
    "second_half_more_goals_than_first",
}


def should_apply_risk_control(parsed: ParsedMarket, confidence: str) -> bool:
    if confidence == "low":
        return True
    if parsed.market_type in HIGH_VARIANCE_MARKETS:
        return True
    if parsed.market_type.startswith("player"):
        return True
    if parsed.metric in {"cards", "fouls", "offsides", "corners", "shots_on_target"}:
        return True
    if parsed.period in {"second_half", "halftime", "first_half"}:
        return True
    return False


def risk_strength(parsed: ParsedMarket, confidence: str) -> float:
    if confidence == "high":
        base = 0.92
    elif confidence == "medium":
        base = 0.82
    else:
        base = 0.68

    if parsed.market_type in HIGH_VARIANCE_MARKETS:
        base -= 0.14
    if parsed.period in {"second_half", "halftime", "first_half"}:
        base -= 0.08
    if parsed.metric in {"cards", "fouls", "offsides", "corners"}:
        base -= 0.07
    if parsed.metric == "shots_on_target":
        base -= 0.05
    return max(0.48, min(0.95, base))


def probability_bounds(parsed: ParsedMarket) -> tuple[float, float]:
    if parsed.market_type == "player_goal":
        return 0.03, 0.40
    if parsed.market_type == "player_goal_or_assist":
        return 0.06, 0.56
    if parsed.market_type == "player_shot_on_target":
        if parsed.period == "second_half":
            return 0.06, 0.58
        return 0.08, 0.82
    if parsed.metric == "shots_on_target":
        if parsed.market_type == "team_metric_more_than_opponent":
            return 0.08, 0.84
        return 0.08, 0.86
    if parsed.market_type in {"penalty_awarded", "penalty_or_red_card"}:
        return 0.08, 0.45
    if parsed.period in {"second_half", "halftime", "first_half"} and parsed.metric != "goals":
        return 0.10, 0.78
    return 0.03, 0.97


def apply_risk_control(
    probability: float,
    parsed: ParsedMarket,
    confidence: str,
    has_exact_odds: bool = False,
    has_confirmed_lineup: bool = False,
) -> RiskControlledProbability:
    if has_exact_odds:
        return RiskControlledProbability(clamp_probability(probability), "")
    if not should_apply_risk_control(parsed, confidence):
        return RiskControlledProbability(clamp_probability(probability), "")

    strength = risk_strength(parsed, confidence)
    if has_confirmed_lineup and parsed.market_type.startswith("player"):
        strength = min(0.95, strength + 0.08)

    controlled = shrink_toward(probability, 0.50, strength)
    low, high = probability_bounds(parsed)
    controlled = clamp_probability(controlled, low, high)
    if abs(controlled - probability) < 0.005:
        return RiskControlledProbability(controlled, "")
    return RiskControlledProbability(
        controlled,
        (
            f" Risk control adjusted {probability:.1%} to {controlled:.1%} "
            "for market variance and data confidence."
        ),
    )

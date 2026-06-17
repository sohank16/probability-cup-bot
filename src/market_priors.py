from __future__ import annotations

from dataclasses import dataclass

from src.betting_odds import OddsBaseline
from src.question_parser import ParsedMarket


@dataclass(frozen=True)
class MarketPrior:
    probability: float
    confidence: str
    explanation: str


OddsBaselineMap = dict[tuple[str, str, str, int], OddsBaseline]


def clamp_probability(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


def period_multiplier(period: str) -> float:
    if period == "halftime":
        return 0.45
    if period == "first_half":
        return 0.45
    if period == "second_half":
        return 0.50
    return 1.00


def attacking_pressure_adjustment(elo_diff: float) -> float:
    return max(-0.10, min(0.10, elo_diff / 1800.0))


def underdog_pressure_adjustment(elo_diff: float) -> float:
    return max(-0.08, min(0.08, -elo_diff / 2200.0))


def threshold_penalty(threshold: int | None, base_threshold: int) -> float:
    if threshold is None:
        return 0.0
    return max(0, threshold - base_threshold) * 0.08


def odds_baseline_for(
    baselines: OddsBaselineMap | None,
    market_type: str,
    metric: str,
    period: str,
    threshold: int | None,
) -> OddsBaseline | None:
    if not baselines or threshold is None:
        return None
    if period != "match":
        return None
    return baselines.get((market_type, metric, "match", threshold))


def prior_from_odds_baseline(baseline: OddsBaseline, adjustment: float = 0.0) -> MarketPrior:
    return MarketPrior(
        clamp_probability(baseline.probability + adjustment),
        "medium",
        (
            f"Baseline calibrated from {baseline.source} odds/stat history "
            f"using {baseline.sample_size} samples."
        ),
    )


def metric_prior(
    parsed: ParsedMarket,
    elo_diff: float = 0.0,
    odds_baselines: OddsBaselineMap | None = None,
) -> MarketPrior:
    metric = parsed.metric
    period_factor = period_multiplier(parsed.period)

    if parsed.market_type == "penalty_awarded":
        return MarketPrior(0.26, "low", "Penalty markets use a global match baseline.")

    if parsed.market_type == "penalty_or_red_card":
        penalty_probability = 0.26
        red_card_probability = 0.16
        combined = 1 - ((1 - penalty_probability) * (1 - red_card_probability))
        return MarketPrior(
            clamp_probability(combined),
            "low",
            "Penalty OR red-card baseline combines two rare-event probabilities.",
        )

    if parsed.market_type == "player_shot_on_target":
        probability = 0.36
        if parsed.period == "second_half":
            probability = 0.20
        return MarketPrior(
            clamp_probability(probability),
            "low",
            "Player shot markets use a conservative baseline until player-minute data is added.",
        )

    if parsed.market_type == "player_goal":
        return MarketPrior(
            0.18,
            "low",
            "Player goal markets use a low scorer baseline until player scoring data is added.",
        )

    if parsed.market_type == "player_goal_or_assist":
        return MarketPrior(
            0.28,
            "low",
            "Goal-or-assist markets use a baseline above goal-only probability.",
        )

    if parsed.market_type == "total_metric_over":
        odds_baseline = odds_baseline_for(
            odds_baselines,
            parsed.market_type,
            metric or "",
            parsed.period,
            parsed.threshold,
        )
        if odds_baseline:
            return prior_from_odds_baseline(odds_baseline)

        if metric == "shots_on_target":
            base_threshold = 4 if parsed.period == "second_half" else 8
            probability = 0.46 - max(0, (parsed.threshold or base_threshold) - base_threshold) * 0.05
            return MarketPrior(
                clamp_probability(probability),
                "low",
                "Total shots-on-target props use a match-total baseline by period.",
            )
        if metric == "corners":
            base_threshold = 5 if parsed.period == "second_half" else 9
            probability = 0.44 - max(0, (parsed.threshold or base_threshold) - base_threshold) * 0.05
            return MarketPrior(
                clamp_probability(probability),
                "low",
                "Total corner props use a match-total baseline by period.",
            )
        if metric == "cards":
            base_threshold = 2 if parsed.period == "second_half" else 4
            probability = 0.54 - max(0, (parsed.threshold or base_threshold) - base_threshold) * 0.06
            return MarketPrior(
                clamp_probability(probability),
                "low",
                "Total card props use a match-total disciplinary baseline by period.",
            )

    if metric == "shots_on_target":
        odds_baseline = odds_baseline_for(
            odds_baselines,
            parsed.market_type,
            metric,
            parsed.period,
            parsed.threshold,
        )
        if odds_baseline:
            return prior_from_odds_baseline(
                odds_baseline,
                attacking_pressure_adjustment(elo_diff),
            )
        base = 0.58 if parsed.market_type == "team_metric_over" else 0.50
        probability = base + attacking_pressure_adjustment(elo_diff)
        probability -= threshold_penalty(parsed.threshold, 3)
        probability *= period_factor
        return MarketPrior(
            clamp_probability(probability),
            "medium",
            "Shots-on-target props use a baseline adjusted by attacking strength and threshold.",
        )

    if metric == "corners":
        odds_baseline = odds_baseline_for(
            odds_baselines,
            parsed.market_type,
            metric,
            parsed.period,
            parsed.threshold,
        )
        if odds_baseline:
            return prior_from_odds_baseline(
                odds_baseline,
                attacking_pressure_adjustment(elo_diff) * 0.7,
            )
        base = 0.48 if parsed.market_type == "team_metric_over" else 0.50
        probability = base + attacking_pressure_adjustment(elo_diff) * 0.7
        probability -= threshold_penalty(parsed.threshold, 5)
        probability *= period_factor
        return MarketPrior(
            clamp_probability(probability),
            "low",
            "Corner props use a baseline with a small attacking-pressure adjustment.",
        )

    if metric == "fouls":
        odds_baseline = odds_baseline_for(
            odds_baselines,
            parsed.market_type,
            metric,
            parsed.period,
            parsed.threshold,
        )
        if odds_baseline:
            return prior_from_odds_baseline(
                odds_baseline,
                underdog_pressure_adjustment(elo_diff),
            )
        probability = 0.50 + underdog_pressure_adjustment(elo_diff)
        return MarketPrior(
            clamp_probability(probability),
            "low",
            "Foul comparison props use a baseline where underdogs are slightly likelier to foul more.",
        )

    if metric == "cards":
        odds_baseline = odds_baseline_for(
            odds_baselines,
            parsed.market_type,
            metric,
            parsed.period,
            parsed.threshold,
        )
        if odds_baseline:
            return prior_from_odds_baseline(
                odds_baseline,
                underdog_pressure_adjustment(elo_diff) if parsed.market_type != "total_metric_over" else 0.0,
            )
        if parsed.market_type == "total_metric_over":
            base = 0.56
            probability = base - threshold_penalty(parsed.threshold, 4)
            probability *= period_factor
        else:
            probability = 0.44 + underdog_pressure_adjustment(elo_diff)
            if parsed.threshold:
                probability -= threshold_penalty(parsed.threshold, 1)
            probability *= period_factor
        return MarketPrior(
            clamp_probability(probability),
            "low",
            "Card props use a disciplinary baseline adjusted slightly for underdog pressure.",
        )

    if metric == "offsides":
        odds_baseline = odds_baseline_for(
            odds_baselines,
            parsed.market_type,
            metric,
            parsed.period,
            parsed.threshold,
        )
        if odds_baseline:
            return prior_from_odds_baseline(
                odds_baseline,
                attacking_pressure_adjustment(elo_diff) * 0.5,
            )
        probability = 0.42 + attacking_pressure_adjustment(elo_diff) * 0.5
        probability -= threshold_penalty(parsed.threshold, 2)
        return MarketPrior(
            clamp_probability(probability),
            "low",
            "Offside props use a baseline with a small attacking-pressure adjustment.",
        )

    return MarketPrior(
        0.50,
        "low",
        "Unsupported prop metric falls back to a neutral baseline.",
    )


def comparison_prior(parsed: ParsedMarket, elo_diff: float = 0.0) -> MarketPrior:
    if parsed.metric in {"shots_on_target", "corners", "goals"}:
        probability = 0.50 + attacking_pressure_adjustment(elo_diff)
        if parsed.period in {"halftime", "first_half", "second_half"}:
            probability = 0.50 + attacking_pressure_adjustment(elo_diff) * 0.8
        return MarketPrior(
            clamp_probability(probability),
            "medium" if parsed.metric == "goals" else "low",
            f"{parsed.metric} comparison uses team-strength pressure against the opponent.",
        )

    if parsed.metric in {"fouls", "cards"}:
        probability = 0.50 + underdog_pressure_adjustment(elo_diff)
        return MarketPrior(
            clamp_probability(probability),
            "low",
            f"{parsed.metric} comparison uses underdog-pressure adjustment.",
        )

    return MarketPrior(0.50, "low", "Comparison prop falls back to a neutral baseline.")

from __future__ import annotations
from typing import Any

from src.market_priors import MarketPrior
from src.name_normalization import normalize_player_name
from src.probability_utils import add_logit_adjustment, clamp_probability, shrink_toward
from src.question_parser import ParsedMarket
from src.starting_xi import StartingXIStatus
from src.team_stat_model import expected_team_shots_on_target


PLAYER_PROP_MARKETS = {
    "player_shot_on_target",
    "player_goal",
    "player_goal_or_assist",
}


def base_player_probability(parsed: ParsedMarket) -> float:
    if parsed.market_type == "player_shot_on_target":
        if parsed.period == "second_half":
            return 0.20
        threshold = parsed.threshold or 1
        return max(0.18, 0.40 - max(0, threshold - 1) * 0.10)
    if parsed.market_type == "player_goal":
        return 0.16
    if parsed.market_type == "player_goal_or_assist":
        return 0.24
    return 0.50


def club_goal_logit_adjustment(club_goals_2025_26: int | None, market_type: str) -> float:
    if club_goals_2025_26 is None:
        return 0.0
    if club_goals_2025_26 >= 25:
        return 0.95 if market_type == "player_shot_on_target" else 1.05
    if club_goals_2025_26 >= 15:
        return 0.70 if market_type == "player_shot_on_target" else 0.78
    if club_goals_2025_26 >= 8:
        return 0.42 if market_type == "player_shot_on_target" else 0.46
    if club_goals_2025_26 >= 3:
        return 0.20
    if club_goals_2025_26 == 0:
        return -0.45 if market_type == "player_shot_on_target" else -0.60
    return 0.0


def country_start_logit_adjustment(country_starts_last_10: int | None, market_type: str) -> float:
    if country_starts_last_10 is None:
        return 0.0
    starts = max(0, min(10, country_starts_last_10))
    if market_type in {"player_goal", "player_goal_or_assist"}:
        if starts >= 9:
            return 0.42
        if starts >= 7:
            return 0.30
        if starts >= 5:
            return 0.15
        if starts >= 3:
            return -0.12
        if starts >= 1:
            return -0.35
        return -0.75

    if starts >= 9:
        return 0.62
    if starts >= 7:
        return 0.45
    if starts >= 5:
        return 0.22
    if starts >= 3:
        return -0.15
    if starts >= 1:
        return -0.45
    return -0.95 if market_type == "player_shot_on_target" else -0.85


def player_data_strength(form: object | None) -> float:
    if form is None:
        return 0.50
    strength = 0.50
    if getattr(form, "club_goals_2025_26", None) is not None:
        strength += 0.20
    if getattr(form, "country_starts_last_10", None) is not None:
        strength += 0.25
    return min(0.92, strength)


def team_sot_logit_adjustment(
    parsed: ParsedMarket,
    player_team: Any | None,
    opponent: Any | None,
) -> float:
    if parsed.market_type != "player_shot_on_target" or player_team is None or opponent is None:
        return 0.0
    team_sot = expected_team_shots_on_target(player_team, opponent, parsed.period)
    if parsed.period == "second_half":
        baseline_sot = 2.0
    else:
        baseline_sot = 3.85
    return max(-0.45, min(0.55, (team_sot - baseline_sot) * 0.22))


def team_goal_environment(player_team: Any | None, opponent: Any | None) -> float | None:
    if player_team is None or opponent is None:
        return None
    elo_diff = float(player_team.team_elo - opponent.team_elo)
    attack = float(getattr(player_team, "last_5_goals_for", 1.2) or 1.2)
    opponent_defense = float(getattr(opponent, "last_5_goals_against", 1.2) or 1.2)
    form = float(getattr(player_team, "last_10_win_rate", 0.45) or 0.45)
    environment = (
        1.25
        + max(-0.70, min(0.70, elo_diff / 700.0))
        + max(-0.25, min(0.30, (attack - 1.35) * 0.22))
        + max(-0.25, min(0.30, (opponent_defense - 1.25) * 0.18))
        + max(-0.20, min(0.22, (form - 0.45) * 0.35))
    )
    if getattr(player_team, "current_tournament_matches", 0):
        environment += max(
            -0.25,
            min(0.30, (float(getattr(player_team, "current_tournament_goals_for_per_match", 0.0)) - 1.25) * 0.16),
        )
    if getattr(opponent, "current_tournament_matches", 0):
        environment += max(
            -0.20,
            min(0.25, (float(getattr(opponent, "current_tournament_goals_against_per_match", 0.0)) - 1.20) * 0.14),
        )
    return max(0.25, min(3.50, environment))


def team_goal_logit_adjustment(
    parsed: ParsedMarket,
    player_team: Any | None,
    opponent: Any | None,
) -> float:
    if parsed.market_type not in {"player_goal", "player_goal_or_assist"}:
        return 0.0
    environment = team_goal_environment(player_team, opponent)
    if environment is None or player_team is None or opponent is None:
        return 0.0

    elo_diff = float(player_team.team_elo - opponent.team_elo)
    adjustment = max(-0.75, min(0.65, (environment - 1.30) * 0.55))
    adjustment += max(-0.35, min(0.30, elo_diff / 1200.0))

    player_rank = getattr(player_team, "fifa_rank", None)
    opponent_rank = getattr(opponent, "fifa_rank", None)
    if opponent_rank is not None and player_rank is not None:
        if int(opponent_rank) <= 5 and int(player_rank) >= 25:
            adjustment -= 0.30
        elif int(opponent_rank) <= 10 and int(player_rank) >= 40:
            adjustment -= 0.22

    return max(-1.10, min(0.80, adjustment))


def apply_player_prop_caps(
    parsed: ParsedMarket,
    probability: float,
    player_team: Any | None,
    opponent: Any | None,
) -> float:
    if parsed.market_type == "player_shot_on_target":
        if parsed.period == "second_half":
            return clamp_probability(probability, 0.06, 0.58)
        if (parsed.threshold or 1) >= 2:
            return clamp_probability(probability, 0.08, 0.62)
        return clamp_probability(probability, 0.10, 0.82)

    environment = team_goal_environment(player_team, opponent)
    if parsed.market_type == "player_goal":
        if environment is not None and environment < 0.85:
            return clamp_probability(probability, 0.03, 0.18)
        if environment is not None and environment < 1.05:
            return clamp_probability(probability, 0.04, 0.24)
        return clamp_probability(probability, 0.04, 0.40)

    if parsed.market_type == "player_goal_or_assist":
        if environment is not None and environment < 0.85:
            return clamp_probability(probability, 0.06, 0.30)
        if environment is not None and environment < 1.05:
            return clamp_probability(probability, 0.08, 0.36)
        if environment is not None and environment < 1.25:
            return clamp_probability(probability, 0.10, 0.42)
        if environment is not None and environment >= 2.0:
            return clamp_probability(probability, 0.10, 0.56)
        return clamp_probability(probability, 0.10, 0.52)

    return clamp_probability(probability)


def predict_player_prop(
    parsed: ParsedMarket,
    player_forms: dict[str, object],
    player_team: Any | None = None,
    opponent: Any | None = None,
    starting_xi_status: StartingXIStatus | None = None,
) -> MarketPrior | None:
    if parsed.market_type not in PLAYER_PROP_MARKETS or not parsed.player:
        return None

    base = base_player_probability(parsed)
    form = player_forms.get(normalize_player_name(parsed.player))
    if not form or (
        getattr(form, "club_goals_2025_26", None) is None
        and getattr(form, "country_starts_last_10", None) is None
    ):
        return MarketPrior(
            shrink_toward(base, 0.50, 0.65),
            "low",
            "Player prop routed to player model, but no player form data was available.",
        )

    club_goals = getattr(form, "club_goals_2025_26", None)
    country_starts = getattr(form, "country_starts_last_10", None)
    adjustment = club_goal_logit_adjustment(club_goals, parsed.market_type)
    adjustment += country_start_logit_adjustment(country_starts, parsed.market_type)
    team_adjustment = team_sot_logit_adjustment(parsed, player_team, opponent)
    adjustment += team_adjustment
    goal_environment_adjustment = team_goal_logit_adjustment(parsed, player_team, opponent)
    adjustment += goal_environment_adjustment
    lineup_adjustment = 0.0
    if starting_xi_status is not None:
        if starting_xi_status.status == "starting":
            lineup_adjustment = 0.18 if starting_xi_status.is_confirmed else 0.08
        elif starting_xi_status.status == "bench":
            lineup_adjustment = -0.90 if starting_xi_status.is_confirmed else -0.35
        elif starting_xi_status.status == "out":
            lineup_adjustment = -4.00 if starting_xi_status.is_confirmed else -1.20
    adjustment += lineup_adjustment
    raw_probability = add_logit_adjustment(base, adjustment)
    probability = shrink_toward(raw_probability, base, player_data_strength(form))
    if starting_xi_status is not None and starting_xi_status.is_confirmed:
        if starting_xi_status.status == "out":
            probability = min(probability, 0.02)
        elif starting_xi_status.status == "bench":
            probability = min(probability, 0.18 if parsed.market_type == "player_shot_on_target" else 0.12)
    probability = apply_player_prop_caps(parsed, probability, player_team, opponent)

    parts = []
    if club_goals is not None:
        parts.append(f"{club_goals} club goals in 2025/26")
    if country_starts is not None:
        parts.append(f"{country_starts}/10 country starts")
    if team_adjustment:
        parts.append(
            f"{getattr(player_team, 'team', 'team')} SOT context adjustment {team_adjustment:+.2f}"
        )
    if goal_environment_adjustment:
        environment = team_goal_environment(player_team, opponent)
        environment_text = f", team goal environment {environment:.2f}" if environment is not None else ""
        parts.append(
            f"{getattr(player_team, 'team', 'team')} goal context adjustment {goal_environment_adjustment:+.2f}{environment_text}"
        )
    if lineup_adjustment:
        parts.append(
            f"lineup status {starting_xi_status.status}/{starting_xi_status.confidence} adjustment {lineup_adjustment:+.2f}"
        )
    detail = ", ".join(parts) if parts else "player form fields present"
    source = getattr(form, "source", "")
    source_text = f" Source: {source}." if source else ""
    return MarketPrior(
        clamp_probability(probability),
        "medium",
        (
            "Player prop model uses logit scoring from player form "
            f"({detail}) with confidence shrinkage.{source_text}"
        ),
    )

from __future__ import annotations

from src.question_parser import ParsedMarket


PLAYER_PROP_MARKETS = {
    "player_shot_on_target",
    "player_goal",
    "player_goal_or_assist",
}

GOAL_MODEL_MARKETS = {
    "match_winner",
    "team_score_at_least",
    "team_score_in_period",
    "team_total_goals_over",
    "total_goals_over",
    "total_goals_under",
    "both_teams_score_and_total_goals",
    "halftime_tied",
    "halftime_team_winning",
    "second_half_more_goals_than_first",
    "team_first_goal_and_opponent_second_half_score",
    "team_first_goal_of_second_half",
}

TEAM_STAT_MARKETS = {
    "team_metric_more_than_opponent",
    "team_metric_over",
    "total_metric_over",
    "both_teams_metric_at_least",
}

RARE_EVENT_MARKETS = {
    "penalty_awarded",
    "penalty_or_red_card",
}


def market_route(parsed: ParsedMarket) -> str:
    if parsed.market_type in PLAYER_PROP_MARKETS:
        return "player_prop"
    if parsed.market_type in GOAL_MODEL_MARKETS or parsed.metric == "goals":
        return "goal_model"
    if parsed.market_type in TEAM_STAT_MARKETS:
        return "team_stat_prop"
    if parsed.market_type in RARE_EVENT_MARKETS:
        return "rare_event"
    return "fallback"

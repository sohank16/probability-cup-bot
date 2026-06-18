from src.market_router import market_route
from src.question_parser import parse_market_question


def test_routes_player_props_to_player_model() -> None:
    parsed = parse_market_question("Will Harry Kane have at least 1 shot on target?")

    assert market_route(parsed) == "player_prop"


def test_routes_team_stat_props_to_team_stat_model() -> None:
    parsed = parse_market_question("Will Brazil have 5 or more shots on target?")

    assert market_route(parsed) == "team_stat_prop"


def test_routes_winner_market_to_goal_model() -> None:
    parsed = parse_market_question("Will Germany win the match?")

    assert market_route(parsed) == "goal_model"

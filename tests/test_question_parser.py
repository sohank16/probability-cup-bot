from src.question_parser import parse_market_question


def test_parse_match_winner() -> None:
    parsed = parse_market_question("Will Argentina win the match?")

    assert parsed.market_type == "match_winner"
    assert parsed.team == "Argentina"
    assert parsed.metric == "goals"


def test_parse_total_goals_over() -> None:
    parsed = parse_market_question("Will the match have 3 or more total goals?")

    assert parsed.market_type == "total_goals_over"
    assert parsed.threshold == 3
    assert parsed.comparison == "over_or_equal"


def test_parse_both_teams_score_and_total_goals() -> None:
    parsed = parse_market_question(
        "Will both teams score AND the match have 3 or more total goals?"
    )

    assert parsed.market_type == "both_teams_score_and_total_goals"
    assert parsed.threshold == 3
    assert parsed.extra["both_teams_score"] is True


def test_parse_halftime_tied() -> None:
    parsed = parse_market_question("At halftime, will the match be tied?")

    assert parsed.market_type == "halftime_tied"
    assert parsed.period == "halftime"


def test_parse_player_shot_on_target() -> None:
    parsed = parse_market_question("Will Marcel Sabitzer have at least 1 shot on target?")

    assert parsed.market_type == "player_shot_on_target"
    assert parsed.player == "Marcel Sabitzer"
    assert parsed.metric == "shots_on_target"
    assert parsed.threshold == 1


def test_parse_player_goal_or_assist() -> None:
    parsed = parse_market_question(
        "Will Harry Kane score or assist a goal (excluding own goals)?"
    )

    assert parsed.market_type == "player_goal_or_assist"
    assert parsed.player == "Harry Kane"
    assert parsed.metric == "goal_or_assist"


def test_parse_total_cards_second_half() -> None:
    parsed = parse_market_question("Will there be 2 or more total cards shown in the second half?")

    assert parsed.market_type == "total_metric_over"
    assert parsed.metric == "cards"
    assert parsed.threshold == 2
    assert parsed.period == "second_half"


def test_parse_team_fouls_comparison() -> None:
    parsed = parse_market_question("Will Algeria commit more fouls than Austria?")

    assert parsed.market_type == "team_metric_more_than_opponent"
    assert parsed.team == "Algeria"
    assert parsed.opponent == "Austria"
    assert parsed.metric == "fouls"


def test_parse_second_half_shots_comparison_prefix() -> None:
    parsed = parse_market_question(
        "In the second half, will Uruguay have more shots on target than Spain?"
    )

    assert parsed.market_type == "team_metric_more_than_opponent"
    assert parsed.team == "Uruguay"
    assert parsed.opponent == "Spain"
    assert parsed.metric == "shots_on_target"
    assert parsed.period == "second_half"


def test_parse_offside_threshold() -> None:
    parsed = parse_market_question("Will Argentina be caught offside 2 or more times?")

    assert parsed.market_type == "team_metric_over"
    assert parsed.team == "Argentina"
    assert parsed.metric == "offsides"
    assert parsed.threshold == 2


def test_parse_first_goal_combo() -> None:
    parsed = parse_market_question(
        "Will Argentina score the first goal of the game and Algeria score in the second half?"
    )

    assert parsed.market_type == "team_first_goal_and_opponent_second_half_score"
    assert parsed.team == "Argentina"
    assert parsed.opponent == "Algeria"


def test_parse_team_corner_in_first_half() -> None:
    parsed = parse_market_question("Will Argentina have at least 1 corner kick in the first half?")

    assert parsed.market_type == "team_metric_over"
    assert parsed.team == "Argentina"
    assert parsed.metric == "corners"
    assert parsed.threshold == 1
    assert parsed.period == "first_half"

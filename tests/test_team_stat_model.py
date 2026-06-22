from src.predictor import TeamFeatureRow
from src.question_parser import parse_market_question
from src.team_stat_model import predict_team_stat_prop


def make_feature(team: str, elo: float, goals_for: float, goals_against: float) -> TeamFeatureRow:
    return TeamFeatureRow(
        team=team,
        team_elo=elo,
        fifa_rank=None,
        fifa_points=None,
        confederation="",
        weighted_points_since_2020=30.0,
        weighted_goal_difference_since_2020=10.0,
        last_5_points=10,
        last_10_points=18,
        last_5_goals_for=goals_for,
        last_5_goals_against=goals_against,
        last_10_win_rate=0.6,
        matches_played_since_2020=20,
        recent_match_count_since_2024=10,
    )


def test_attacking_team_gets_higher_shots_probability_than_weaker_team() -> None:
    favorite = make_feature("Favorite", 1850, 2.2, 0.8)
    underdog = make_feature("Underdog", 1400, 0.8, 2.0)
    parsed = parse_market_question("Will Favorite have 3 or more shots on target?")
    reverse = parse_market_question("Will Underdog have 3 or more shots on target?")

    favorite_prior = predict_team_stat_prop(parsed, favorite, underdog)
    underdog_prior = predict_team_stat_prop(reverse, underdog, favorite)

    assert favorite_prior.probability > underdog_prior.probability
    assert "SOT specialist" in favorite_prior.explanation


def test_multi_shot_threshold_is_capped_even_for_large_favorite() -> None:
    favorite = make_feature("Favorite", 2050, 3.2, 0.5)
    underdog = make_feature("Underdog", 1200, 0.4, 3.0)
    parsed = parse_market_question("Will Favorite have 6 or more shots on target?")

    prior = predict_team_stat_prop(parsed, favorite, underdog)

    assert prior.probability <= 0.76


def test_underdog_pressure_raises_foul_comparison_probability() -> None:
    favorite = make_feature("Favorite", 1850, 2.2, 0.8)
    underdog = make_feature("Underdog", 1400, 0.8, 2.0)
    parsed = parse_market_question("Will Underdog commit more fouls than Favorite?")

    prior = predict_team_stat_prop(parsed, underdog, favorite)

    assert prior.probability > 0.50

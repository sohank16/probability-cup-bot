from src.predictor import (
    TeamFeatureRow,
    expected_goals_for,
    predict_parsed_market,
    team_win_probability,
)
from src.question_parser import parse_market_question


def make_feature(team: str, elo: float, goals_for: float = 1.5, goals_against: float = 1.0) -> TeamFeatureRow:
    return TeamFeatureRow(
        team=team,
        team_elo=elo,
        fifa_rank=None,
        fifa_points=None,
        confederation="",
        weighted_points_since_2020=0.0,
        weighted_goal_difference_since_2020=0.0,
        last_5_points=9,
        last_10_points=18,
        last_5_goals_for=goals_for,
        last_5_goals_against=goals_against,
        last_10_win_rate=0.5,
        matches_played_since_2020=20,
        recent_match_count_since_2024=10,
    )


def test_team_win_probability_increases_with_elo_gap() -> None:
    favorite = make_feature("Favorite", 1800)
    underdog = make_feature("Underdog", 1400)

    assert team_win_probability(favorite, underdog) > team_win_probability(underdog, favorite)


def test_expected_goals_increase_for_stronger_team() -> None:
    favorite = make_feature("Favorite", 1800, goals_for=2.0)
    underdog = make_feature("Underdog", 1400, goals_against=1.8)

    assert expected_goals_for(favorite, underdog) > expected_goals_for(underdog, favorite)


def test_predict_match_winner_uses_goal_model_confidence() -> None:
    features = {
        "argentina": make_feature("Argentina", 1875),
        "algeria": make_feature("Algeria", 1570),
    }
    parsed = parse_market_question("Will Argentina win the match?")

    probability, confidence, explanation, team, opponent = predict_parsed_market(
        parsed,
        "ARG vs ALG",
        features,
    )

    assert probability > 0.5
    assert confidence == "high"
    assert team == "Argentina"
    assert opponent == "Algeria"
    assert "Elo" in explanation


def test_predict_prop_market_uses_prior_layer() -> None:
    features = {
        "argentina": make_feature("Argentina", 1875),
        "algeria": make_feature("Algeria", 1570),
    }
    parsed = parse_market_question("Will Algeria commit more fouls than Argentina?")

    probability, confidence, explanation, team, opponent = predict_parsed_market(
        parsed,
        "ARG vs ALG",
        features,
    )

    assert probability > 0.5
    assert confidence == "low"
    assert team == "Algeria"
    assert opponent == "Argentina"
    assert "underdog" in explanation.lower()


def test_second_half_goal_comparison_uses_poisson_not_neutral_fallback() -> None:
    features = {
        "favorite": make_feature("Favorite", 1850, goals_for=2.2, goals_against=0.7),
        "underdog": make_feature("Underdog", 1400, goals_for=0.8, goals_against=2.0),
    }
    parsed = parse_market_question("Will Favorite score more goals than Underdog in the second half?")

    probability, confidence, explanation, team, opponent = predict_parsed_market(
        parsed,
        "Favorite vs Underdog",
        features,
    )

    assert probability > 0.50
    assert confidence == "medium"
    assert team == "Favorite"
    assert opponent == "Underdog"
    assert "Poisson" in explanation


def test_second_half_more_goals_uses_poisson_period_split() -> None:
    features = {
        "favorite": make_feature("Favorite", 1850, goals_for=2.2, goals_against=0.7),
        "underdog": make_feature("Underdog", 1400, goals_for=0.8, goals_against=2.0),
    }
    parsed = parse_market_question("Will the second half have more goals than the first half?")

    probability, confidence, explanation, team, opponent = predict_parsed_market(
        parsed,
        "Favorite vs Underdog",
        features,
    )

    assert 0.30 < probability < 0.50
    assert confidence == "medium"
    assert "Poisson" in explanation

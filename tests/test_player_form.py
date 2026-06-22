from src.player_form import PlayerForm, infer_national_team, normalize_player_name, player_market_prior
from src.predictor import TeamFeatureRow
from src.player_prop_model import predict_player_prop
from src.question_parser import parse_market_question
from src.starting_xi import StartingXIStatus


def make_feature(team: str, elo: float, goals_for: float = 1.5, goals_against: float = 1.1) -> TeamFeatureRow:
    return TeamFeatureRow(
        team=team,
        team_elo=elo,
        fifa_rank=1 if team == "Elite" else 60 if team == "Underdog" else None,
        fifa_points=None,
        confederation="",
        weighted_points_since_2020=20.0,
        weighted_goal_difference_since_2020=5.0,
        last_5_points=9,
        last_10_points=18,
        last_5_goals_for=goals_for,
        last_5_goals_against=goals_against,
        last_10_win_rate=0.5,
        matches_played_since_2020=20,
        recent_match_count_since_2024=10,
    )


def test_player_form_raises_probability_for_in_form_starter() -> None:
    parsed = parse_market_question("Will Harry Kane score a goal (excluding own goals)?")
    prior = player_market_prior(
        parsed,
        {
            normalize_player_name("Harry Kane"): PlayerForm(
                player="Harry Kane",
                club_goals_2025_26=24,
                country_starts_last_10=9,
                source="test",
            )
        },
    )

    assert prior is not None
    assert prior.probability > 0.30
    assert prior.confidence == "medium"
    assert "logit scoring" in prior.explanation


def test_player_form_lowers_probability_for_non_starter() -> None:
    parsed = parse_market_question("Will Example Player have at least 1 shot on target?")
    prior = player_market_prior(
        parsed,
        {
            normalize_player_name("Example Player"): PlayerForm(
                player="Example Player",
                club_goals_2025_26=0,
                country_starts_last_10=0,
                source="test",
            )
        },
    )

    assert prior is not None
    assert prior.probability < 0.25


def test_player_form_keeps_second_half_shot_lower_than_full_match_shot() -> None:
    full_match = parse_market_question("Will Example Player have at least 1 shot on target?")
    second_half = parse_market_question(
        "Will Example Player have at least 1 shot on target in the second half?"
    )
    forms = {
        normalize_player_name("Example Player"): PlayerForm(
            player="Example Player",
            club_goals_2025_26=12,
            country_starts_last_10=8,
            source="test",
        )
    }

    full_match_prior = player_market_prior(full_match, forms)
    second_half_prior = player_market_prior(second_half, forms)

    assert full_match_prior is not None
    assert second_half_prior is not None
    assert second_half_prior.probability < full_match_prior.probability


def test_infer_national_team_from_repeated_example_matches() -> None:
    row = {"example_matches": "ENG vs GHA; CRO vs GHA"}

    assert infer_national_team(row) == "Ghana"


def test_player_sot_uses_team_context_when_available() -> None:
    parsed = parse_market_question("Will Example Player have at least 1 shot on target?")
    forms = {
        normalize_player_name("Example Player"): PlayerForm(
            player="Example Player",
            club_goals_2025_26=10,
            country_starts_last_10=8,
            source="test",
            national_team="Favorite",
        )
    }
    favorite = make_feature("Favorite", 1850, goals_for=2.3, goals_against=0.7)
    underdog = make_feature("Underdog", 1400, goals_for=0.8, goals_against=2.0)

    without_context = player_market_prior(parsed, forms)
    with_context = predict_player_prop(parsed, forms, favorite, underdog)

    assert without_context is not None
    assert with_context is not None
    assert with_context.probability > without_context.probability


def test_player_sot_is_capped_for_extreme_attacking_context() -> None:
    parsed = parse_market_question("Will Example Player have at least 1 shot on target?")
    forms = {
        normalize_player_name("Example Player"): PlayerForm(
            player="Example Player",
            club_goals_2025_26=60,
            country_starts_last_10=10,
            source="test",
            national_team="Favorite",
        )
    }
    favorite = make_feature("Favorite", 2100, goals_for=4.0, goals_against=0.3)
    underdog = make_feature("Underdog", 1150, goals_for=0.3, goals_against=4.0)

    prior = predict_player_prop(parsed, forms, favorite, underdog)

    assert prior is not None
    assert prior.probability <= 0.82


def test_player_goal_or_assist_is_suppressed_for_underdog_against_elite() -> None:
    parsed = parse_market_question("Will Example Player score or assist a goal (excluding own goals)?")
    forms = {
        normalize_player_name("Example Player"): PlayerForm(
            player="Example Player",
            club_goals_2025_26=5,
            country_starts_last_10=9,
            source="test",
            national_team="Underdog",
        )
    }
    underdog = make_feature("Underdog", 1390, goals_for=1.2, goals_against=2.6)
    elite = make_feature("Elite", 1880, goals_for=3.0, goals_against=0.2)

    without_context = player_market_prior(parsed, forms)
    with_context = predict_player_prop(parsed, forms, underdog, elite)

    assert without_context is not None
    assert with_context is not None
    assert with_context.probability < without_context.probability
    assert with_context.probability <= 0.30


def test_confirmed_bench_status_suppresses_player_prop() -> None:
    parsed = parse_market_question("Will Example Player have at least 1 shot on target?")
    forms = {
        normalize_player_name("Example Player"): PlayerForm(
            player="Example Player",
            club_goals_2025_26=20,
            country_starts_last_10=10,
            source="test",
            national_team="Favorite",
        )
    }
    favorite = make_feature("Favorite", 1850, goals_for=2.3, goals_against=0.7)
    underdog = make_feature("Underdog", 1400, goals_for=0.8, goals_against=2.0)

    prior = predict_player_prop(
        parsed,
        forms,
        favorite,
        underdog,
        StartingXIStatus(
            match_name="Favorite vs Underdog",
            player="Example Player",
            status="bench",
            confidence="confirmed",
            source="test",
        ),
    )

    assert prior is not None
    assert prior.probability <= 0.18

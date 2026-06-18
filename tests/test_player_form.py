from src.player_form import PlayerForm, normalize_player_name, player_market_prior
from src.question_parser import parse_market_question


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

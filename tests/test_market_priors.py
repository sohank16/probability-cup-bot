from src.market_priors import metric_prior
from src.question_parser import parse_market_question


def test_total_shots_on_target_baseline_is_not_extreme() -> None:
    parsed = parse_market_question("Will there be 8 or more total shots on target?")

    prior = metric_prior(parsed)

    assert 0.35 <= prior.probability <= 0.55


def test_second_half_total_cards_baseline_is_reasonable() -> None:
    parsed = parse_market_question("Will there be 2 or more total cards shown in the second half?")

    prior = metric_prior(parsed)

    assert 0.35 <= prior.probability <= 0.65


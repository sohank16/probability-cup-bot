from src.question_parser import parse_market_question
from src.risk_control import apply_risk_control


def test_risk_control_shrinks_high_variance_player_goal_or_assist() -> None:
    parsed = parse_market_question("Will Example Player score or assist a goal (excluding own goals)?")

    controlled = apply_risk_control(0.56, parsed, "medium")

    assert controlled.probability < 0.56
    assert "Risk control" in controlled.explanation_suffix


def test_risk_control_does_not_shrink_exact_odds_anchor() -> None:
    parsed = parse_market_question("Will Example Player score or assist a goal (excluding own goals)?")

    controlled = apply_risk_control(0.56, parsed, "medium", has_exact_odds=True)

    assert controlled.probability == 0.56
    assert controlled.explanation_suffix == ""


def test_risk_control_leaves_high_confidence_match_winner_alone() -> None:
    parsed = parse_market_question("Will Germany win the match?")

    controlled = apply_risk_control(0.89, parsed, "high")

    assert controlled.probability == 0.89
    assert controlled.explanation_suffix == ""

from pathlib import Path

from src.exact_market_odds import ExactMarketOdds, blend_with_exact_odds, load_exact_market_odds


def test_load_exact_market_odds_accepts_fair_probability(tmp_path: Path) -> None:
    path = tmp_path / "exact_market_odds.csv"
    path.write_text(
        "market_id,match_name,question,fair_probability,yes_odds,no_odds,bookmaker,source\n"
        "m1,Match,Question,62,,,,crowd,crowd\n",
        encoding="utf-8",
    )

    odds = load_exact_market_odds(path)

    assert odds["m1"].probability == 0.62


def test_load_exact_market_odds_devigs_yes_no_odds(tmp_path: Path) -> None:
    path = tmp_path / "exact_market_odds.csv"
    path.write_text(
        "market_id,match_name,question,fair_probability,yes_odds,no_odds,bookmaker,source\n"
        "m1,Match,Question,,2.00,2.00,book,book\n",
        encoding="utf-8",
    )

    odds = load_exact_market_odds(path)

    assert odds["m1"].probability == 0.50


def test_blend_with_exact_odds_weights_market_anchor() -> None:
    blended = blend_with_exact_odds(
        0.30,
        ExactMarketOdds(market_id="m1", probability=0.70, sample_size=2, source="test"),
    )

    assert 0.55 < blended < 0.65

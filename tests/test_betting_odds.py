from src.betting_odds import (
    baseline_lookup,
    build_baselines_from_bookmaker_prop_rows,
    build_baselines_from_rows,
    parse_rows,
)
from src.market_priors import metric_prior
from src.question_parser import parse_market_question


def test_build_baselines_from_football_data_style_rows() -> None:
    rows = parse_rows(
        "\n".join(
            [
                "Div,Date,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,Avg>2.5,Avg<2.5,HST,AST,HC,AC,HF,AF,HO,AO,HY,AY,HR,AR",
                "E0,01/01/26,A,B,2.0,3.5,4.0,1.9,1.9,5,3,6,4,10,12,2,1,1,2,0,1",
                "E0,02/01/26,C,D,1.8,3.7,5.0,2.1,1.8,2,2,3,3,8,9,0,3,0,1,0,0",
            ]
        )
    )

    baselines = baseline_lookup(build_baselines_from_rows(rows))

    assert baselines[("total_metric_over", "corners", "match", 9)].sample_size == 2
    assert baselines[("team_metric_over", "shots_on_target", "match", 3)].sample_size == 4


def test_metric_prior_can_use_odds_stat_baseline() -> None:
    rows = parse_rows(
        "\n".join(
            [
                "Div,Date,HomeTeam,AwayTeam,HST,AST,HC,AC,HF,AF,HO,AO,HY,AY,HR,AR",
                "E0,01/01/26,A,B,5,3,6,4,10,12,2,1,1,2,0,1",
                "E0,02/01/26,C,D,2,2,3,3,8,9,0,3,0,1,0,0",
            ]
        )
    )
    baselines = baseline_lookup(build_baselines_from_rows(rows))
    parsed = parse_market_question("Will there be 9 or more total corner kicks?")

    prior = metric_prior(parsed, odds_baselines=baselines)

    assert prior.confidence == "medium"
    assert "Football-Data" in prior.explanation


def test_bookmaker_prop_rows_are_devigged_and_grouped() -> None:
    rows = [
        {
            "market_type": "team_metric_over",
            "metric": "shots_on_target",
            "period": "match",
            "threshold": "3",
            "bookmaker": "Book A",
            "over_odds": "1.80",
            "under_odds": "2.10",
        },
        {
            "market_type": "team_metric_over",
            "metric": "shots_on_target",
            "period": "match",
            "threshold": "3",
            "bookmaker": "Book B",
            "over_odds": "1.90",
            "under_odds": "1.95",
        },
    ]

    baselines = build_baselines_from_bookmaker_prop_rows(rows)

    assert len(baselines) == 1
    assert baselines[0].sample_size == 2
    assert 0.50 < baselines[0].probability < 0.60

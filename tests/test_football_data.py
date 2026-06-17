from datetime import date

from src.football_data import (
    FootballMatch,
    combined_match_weight,
    completed_matches_since,
    parse_results_csv,
    tournament_weight,
    time_decay_weight,
)


def test_parse_results_csv_skips_na_scores() -> None:
    csv_text = """date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
2019-12-31,Old A,Old B,1,0,Friendly,City,Country,FALSE
2020-01-01,A,B,2,1,Friendly,City,Country,FALSE
2026-06-27,C,D,NA,NA,FIFA World Cup,City,Country,TRUE
"""

    matches = parse_results_csv(csv_text)
    recent = completed_matches_since(matches, date(2020, 1, 1), today=date(2026, 6, 17))

    assert len(matches) == 2
    assert len(recent) == 1
    assert recent[0].home_team == "A"


def test_time_decay_weight_makes_recent_matches_stronger() -> None:
    assert time_decay_weight(date(2020, 1, 1)) == 0.25
    assert time_decay_weight(date(2022, 1, 1)) == 0.50
    assert time_decay_weight(date(2026, 1, 1)) == 1.00
    assert time_decay_weight(date(2020, 1, 1)) < time_decay_weight(date(2026, 1, 1))


def test_2022_world_cup_gets_more_weight_than_friendlies() -> None:
    world_cup_match = FootballMatch(
        match_date=date(2022, 12, 18),
        home_team="Argentina",
        away_team="France",
        home_score=3,
        away_score=3,
        tournament="FIFA World Cup",
        city="Lusail",
        country="Qatar",
        neutral=True,
    )
    friendly_match = FootballMatch(
        match_date=date(2022, 12, 18),
        home_team="Argentina",
        away_team="France",
        home_score=3,
        away_score=3,
        tournament="Friendly",
        city="Lusail",
        country="Qatar",
        neutral=True,
    )

    assert combined_match_weight(world_cup_match) > combined_match_weight(friendly_match)
    assert tournament_weight("FIFA World Cup", world_cup_match) == 3.00


def test_confederation_hierarchy_affects_low_priority_matches() -> None:
    european_match = FootballMatch(
        match_date=date(2026, 1, 1),
        home_team="Germany",
        away_team="Spain",
        home_score=1,
        away_score=1,
        tournament="Friendly",
        city="City",
        country="Country",
        neutral=True,
    )
    oceania_match = FootballMatch(
        match_date=date(2026, 1, 1),
        home_team="New Zealand",
        away_team="Tahiti",
        home_score=1,
        away_score=1,
        tournament="Friendly",
        city="City",
        country="Country",
        neutral=True,
    )
    confederations = {
        "germany": "UEFA",
        "spain": "UEFA",
        "new zealand": "OFC",
        "tahiti": "OFC",
    }

    assert combined_match_weight(european_match, confederations) > combined_match_weight(
        oceania_match,
        confederations,
    )

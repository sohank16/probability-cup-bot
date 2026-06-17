from datetime import date

from src.elo import rate_matches
from src.features import build_team_features
from src.football_data import FootballMatch


def make_match(match_date: date, home_score: int, away_score: int) -> FootballMatch:
    return FootballMatch(
        match_date=match_date,
        home_team="Team A",
        away_team="Team B",
        home_score=home_score,
        away_score=away_score,
        tournament="Friendly",
        city="City",
        country="Country",
        neutral=True,
    )


def test_recent_match_moves_elo_more_than_old_match() -> None:
    old_match = make_match(date(2020, 1, 1), 1, 0)
    recent_match = make_match(date(2026, 1, 1), 1, 0)

    _, old_updates = rate_matches([old_match])
    _, recent_updates = rate_matches([recent_match])

    assert abs(old_updates[0].rating_change) < abs(recent_updates[0].rating_change)


def test_build_team_features_includes_weighted_and_recent_counts() -> None:
    matches = [
        make_match(date(2020, 1, 1), 1, 0),
        make_match(date(2024, 1, 1), 2, 0),
        make_match(date(2026, 1, 1), 0, 0),
    ]
    ratings, _ = rate_matches(matches)

    features = build_team_features(matches, ratings)

    team_a = features["Team A"]
    assert team_a.matches_played_since_2020 == 3
    assert team_a.recent_match_count_since_2024 == 2
    assert team_a.last_5_points == 7
    assert team_a.weighted_points_since_2020 > 0
    assert team_a.weighted_goal_difference_since_2020 > 0


def test_fifa_prior_rating_changes_starting_elo() -> None:
    match = make_match(date(2026, 1, 1), 1, 1)

    ratings, updates = rate_matches(
        [match],
        base_ratings={"team a": 1750.0, "team b": 1450.0},
    )

    assert updates[0].home_rating_before == 1750.0
    assert updates[0].away_rating_before == 1450.0
    assert ratings["Team A"] > ratings["Team B"]


def test_yearly_fifa_anchor_affects_existing_elo_before_update() -> None:
    matches = [
        make_match(date(2020, 1, 1), 1, 0),
        make_match(date(2021, 1, 1), 0, 0),
    ]

    _, updates = rate_matches(
        matches,
        base_ratings={"team a": 1500.0, "team b": 1500.0},
        yearly_base_ratings={
            2021: {
                "team a": 1800.0,
                "team b": 1400.0,
            }
        },
        yearly_anchor_strength=0.50,
    )

    assert updates[1].home_rating_before > updates[0].home_rating_after
    assert updates[1].away_rating_before < updates[0].away_rating_after

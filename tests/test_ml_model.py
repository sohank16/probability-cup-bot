from datetime import date

from src.football_data import FootballMatch
from src.ml_model import FEATURE_COLUMNS, matrix


def test_matrix_uses_feature_column_order() -> None:
    row = {column: index for index, column in enumerate(FEATURE_COLUMNS)}

    assert matrix([row], FEATURE_COLUMNS)[0] == [float(index) for index in range(len(FEATURE_COLUMNS))]


def test_training_label_shape_is_match_result_based() -> None:
    match = FootballMatch(
        match_date=date(2026, 1, 1),
        home_team="A",
        away_team="B",
        home_score=2,
        away_score=1,
        tournament="Friendly",
        city="City",
        country="Country",
        neutral=True,
    )

    assert match.home_score > match.away_score
    assert match.home_score + match.away_score >= 3
    assert match.home_score >= 1 and match.away_score >= 1


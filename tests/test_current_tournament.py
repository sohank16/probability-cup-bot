from pathlib import Path

from src.current_tournament import load_current_tournament_form


def test_load_current_tournament_form_aggregates_points_and_goals(tmp_path: Path) -> None:
    path = tmp_path / "current_results.csv"
    path.write_text(
        "\n".join(
            [
                "date,home_team,away_team,home_score,away_score,home_shots_on_target,away_shots_on_target,source",
                "2026-06-12,United States,Paraguay,4,1,7,3,test",
                "2026-06-19,United States,Australia,2,0,5,2,test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    form = load_current_tournament_form(path)

    assert form["United States"].matches == 2
    assert form["United States"].points == 6
    assert form["United States"].goals_for_per_match == 3.0
    assert form["United States"].shots_on_target_for_per_match == 6.0
    assert form["Paraguay"].goals_against_per_match == 4.0

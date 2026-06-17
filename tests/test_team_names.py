from src.team_names import canonical_team_name, opponent_for, split_match_name


def test_split_match_name_handles_codes_and_full_names() -> None:
    assert split_match_name("ARG vs ALG") == ("Argentina", "Algeria")
    assert split_match_name("New Zealand vs BEL") == ("New Zealand", "Belgium")
    assert split_match_name("Curacao vs CIV") == ("Curaçao", "Ivory Coast")


def test_opponent_for_returns_other_team() -> None:
    assert opponent_for("Argentina", "Argentina", "Algeria") == "Algeria"
    assert opponent_for("ALG", "Argentina", "Algeria") == "Argentina"


def test_canonical_team_name_handles_common_aliases() -> None:
    assert canonical_team_name("TUR") == "Turkey"
    assert canonical_team_name("Czechia") == "Czech Republic"


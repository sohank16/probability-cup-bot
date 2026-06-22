from pathlib import Path

from src.starting_xi import key_for, load_starting_xi


def test_load_starting_xi_reads_confirmed_status(tmp_path: Path) -> None:
    path = tmp_path / "starting_xi.csv"
    path.write_text(
        "match_name,player,status,confidence,source\n"
        "A vs B,Example Player,bench,confirmed,official\n",
        encoding="utf-8",
    )

    statuses = load_starting_xi(path)

    status = statuses[key_for("A vs B", "Example Player")]
    assert status.status == "bench"
    assert status.is_confirmed

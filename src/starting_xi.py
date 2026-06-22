from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.name_normalization import normalize_player_name


@dataclass(frozen=True)
class StartingXIStatus:
    match_name: str
    player: str
    status: str
    confidence: str
    source: str

    @property
    def is_confirmed(self) -> bool:
        return self.confidence == "confirmed"


VALID_STATUSES = {"starting", "bench", "out", "unknown"}
VALID_CONFIDENCE = {"confirmed", "probable"}


def key_for(match_name: str, player: str) -> tuple[str, str]:
    return match_name.strip().lower(), normalize_player_name(player)


def load_starting_xi(path: Path | None) -> dict[tuple[str, str], StartingXIStatus]:
    if path is None or not path.exists():
        return {}

    statuses: dict[tuple[str, str], StartingXIStatus] = {}
    with path.open("r", newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            match_name = (row.get("match_name") or "").strip()
            player = (row.get("player") or "").strip()
            status = (row.get("status") or "").strip().lower()
            confidence = (row.get("confidence") or "").strip().lower()
            if not match_name or not player:
                continue
            if status not in VALID_STATUSES or confidence not in VALID_CONFIDENCE:
                continue
            statuses[key_for(match_name, player)] = StartingXIStatus(
                match_name=match_name,
                player=player,
                status=status,
                confidence=confidence,
                source=(row.get("source") or "").strip(),
            )
    return statuses

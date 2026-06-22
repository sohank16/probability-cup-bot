from __future__ import annotations

import unicodedata


def normalize_player_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in ascii_name)
    return " ".join(cleaned.split())

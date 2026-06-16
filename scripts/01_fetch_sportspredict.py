from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.api_client import SportsPredictClient, SportsPredictAPIError
from src.config import load_settings
from src.storage import Storage, utc_now_iso


def find_probability_cup_event(events: list[dict]) -> dict:
    for event in events:
        title = (event.get("title") or "").lower()
        event_type = (event.get("type") or "").lower()
        if "probability cup" in title or event_type == "probability":
            return event
    raise RuntimeError("Could not find a Probability Cup event in API response.")


def write_raw_file(raw_data_dir: Path, name: str, payload: object) -> Path:
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now_iso().replace(":", "-")
    path = raw_data_dir / f"{timestamp}_{name}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    settings = load_settings()

    if not settings.has_api_key:
        print("Missing SPORTSPREDICT_API_KEY.")
        print("Create a .env file from .env.example and add your bot API key.")
        return 1

    storage = Storage(settings.database_path)
    storage.initialize()

    client = SportsPredictClient(
        api_key=settings.sportspredict_api_key,
        base_url=settings.sportspredict_base_url,
    )

    try:
        events = client.get_events()
        event = find_probability_cup_event(events)
        event_id = event["id"]

        lobbies = client.get_lobbies(event_id)
        joined_lobbies = [lobby for lobby in lobbies if lobby.get("joined")]
        lobby = joined_lobbies[0] if joined_lobbies else lobbies[0]
        lobby_id = lobby["id"]

        matches = client.get_matches(event_id=event_id, lobby_id=lobby_id)
        markets = client.get_markets(lobby_id=lobby_id)
    except SportsPredictAPIError as exc:
        print(f"SportsPredict API error: {exc}")
        return 1

    payload = {
        "events": events,
        "selected_event": event,
        "lobbies": lobbies,
        "selected_lobby": lobby,
        "matches": matches,
        "markets": markets,
    }

    raw_path = write_raw_file(settings.raw_data_dir, "sportspredict_snapshot", payload)
    storage.save_snapshot("sportspredict_snapshot", payload)
    storage.upsert_events(events)
    storage.upsert_lobbies(event_id, lobbies)
    storage.upsert_matches(matches)
    storage.upsert_markets(markets)

    print("SportsPredict fetch complete.")
    print(f"Event: {event.get('title')} ({event_id})")
    print(f"Lobby: {lobby.get('name')} ({lobby_id})")
    print(f"Matches fetched: {len(matches)}")
    print(f"Markets fetched: {len(markets)}")
    print(f"Raw snapshot: {raw_path}")
    print(f"SQLite database: {settings.database_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Storage:
    """SQLite storage for raw API snapshots and structured Probability Cup records."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS api_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    captured_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    type TEXT,
                    status TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    raw_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS lobbies (
                    id TEXT PRIMARY KEY,
                    event_id TEXT,
                    name TEXT,
                    type TEXT,
                    joined INTEGER,
                    raw_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS matches (
                    id TEXT PRIMARY KEY,
                    event_id TEXT,
                    name TEXT,
                    opening_time TEXT,
                    closing_time TEXT,
                    open_market_count INTEGER,
                    raw_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS markets (
                    id TEXT PRIMARY KEY,
                    lobby_id TEXT,
                    match_id TEXT,
                    question TEXT,
                    status TEXT,
                    closing_time TEXT,
                    raw_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def save_snapshot(self, name: str, payload: Any) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO api_snapshots (name, captured_at, payload_json)
                VALUES (?, ?, ?)
                """,
                (name, utc_now_iso(), json.dumps(payload, sort_keys=True)),
            )

    def upsert_events(self, events: Iterable[dict[str, Any]]) -> None:
        rows = []
        now = utc_now_iso()
        for event in events:
            rows.append(
                (
                    event.get("id"),
                    event.get("title"),
                    event.get("type"),
                    event.get("status"),
                    event.get("start_date"),
                    event.get("end_date"),
                    json.dumps(event, sort_keys=True),
                    now,
                )
            )

        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO events (id, title, type, status, start_date, end_date, raw_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    type=excluded.type,
                    status=excluded.status,
                    start_date=excluded.start_date,
                    end_date=excluded.end_date,
                    raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

    def upsert_lobbies(self, event_id: str, lobbies: Iterable[dict[str, Any]]) -> None:
        rows = []
        now = utc_now_iso()
        for lobby in lobbies:
            rows.append(
                (
                    lobby.get("id"),
                    event_id,
                    lobby.get("name"),
                    lobby.get("type"),
                    int(bool(lobby.get("joined"))),
                    json.dumps(lobby, sort_keys=True),
                    now,
                )
            )

        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO lobbies (id, event_id, name, type, joined, raw_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    event_id=excluded.event_id,
                    name=excluded.name,
                    type=excluded.type,
                    joined=excluded.joined,
                    raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

    def upsert_matches(self, matches: Iterable[dict[str, Any]]) -> None:
        rows = []
        now = utc_now_iso()
        for match in matches:
            rows.append(
                (
                    match.get("id"),
                    match.get("event_id"),
                    match.get("name"),
                    match.get("opening_time"),
                    match.get("closing_time"),
                    match.get("open_market_count"),
                    json.dumps(match, sort_keys=True),
                    now,
                )
            )

        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO matches (id, event_id, name, opening_time, closing_time, open_market_count, raw_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    event_id=excluded.event_id,
                    name=excluded.name,
                    opening_time=excluded.opening_time,
                    closing_time=excluded.closing_time,
                    open_market_count=excluded.open_market_count,
                    raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

    def upsert_markets(self, markets: Iterable[dict[str, Any]]) -> None:
        rows = []
        now = utc_now_iso()
        for market in markets:
            match = market.get("match") or {}
            rows.append(
                (
                    market.get("id"),
                    market.get("lobby_id"),
                    match.get("id"),
                    market.get("question"),
                    market.get("status"),
                    match.get("closing_time") or market.get("closing_time"),
                    json.dumps(market, sort_keys=True),
                    now,
                )
            )

        with self.connect() as connection:
            connection.executemany(
                """
                INSERT INTO markets (id, lobby_id, match_id, question, status, closing_time, raw_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    lobby_id=excluded.lobby_id,
                    match_id=excluded.match_id,
                    question=excluded.question,
                    status=excluded.status,
                    closing_time=excluded.closing_time,
                    raw_json=excluded.raw_json,
                    updated_at=excluded.updated_at
                """,
                rows,
            )

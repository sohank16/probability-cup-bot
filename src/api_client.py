from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class SportsPredictAPIError(RuntimeError):
    """Raised when the SportsPredict API returns an error response."""


@dataclass
class SportsPredictClient:
    """Small REST client for the SportsPredict Model API."""

    api_key: str
    base_url: str = "https://api.sportspredict.com/api/v1"
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        response = self.session.request(
            method,
            url,
            timeout=self.timeout_seconds,
            **kwargs,
        )

        if response.status_code >= 400:
            raise SportsPredictAPIError(
                f"{method} {path} failed with {response.status_code}: {response.text}"
            )

        if not response.content:
            return None

        return response.json()

    def get_events(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._request("GET", "/events", params={"limit": limit})

    def get_lobbies(self, event_id: str) -> list[dict[str, Any]]:
        return self._request("GET", "/lobbies", params={"event_id": event_id})

    def join_lobby(self, lobby_id: str) -> dict[str, Any] | None:
        return self._request("POST", f"/lobbies/{lobby_id}/join")

    def get_matches(
        self,
        event_id: str | None = None,
        lobby_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {k: v for k, v in {"event_id": event_id, "lobby_id": lobby_id}.items() if v}
        return self._request("GET", "/matches", params=params)

    def get_markets(
        self,
        lobby_id: str | None = None,
        match_id: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {k: v for k, v in {"lobby_id": lobby_id, "match_id": match_id}.items() if v}
        return self._request("GET", "/markets", params=params)

    def get_predictions(self, lobby_id: str | None = None) -> list[dict[str, Any]]:
        params = {k: v for k, v in {"lobby_id": lobby_id}.items() if v}
        return self._request("GET", "/predictions", params=params)

    def submit_predictions_batch(
        self,
        predictions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/predictions/batch",
            json={"predictions": predictions},
        )

    def update_prediction(self, prediction_id: str, probability: int) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/predictions/{prediction_id}",
            json={"probability": probability},
        )

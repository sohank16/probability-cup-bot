from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from src.betting_odds import implied_probability_from_decimal_odds, parse_float
from src.probability_utils import clamp_probability


@dataclass(frozen=True)
class ExactMarketOdds:
    market_id: str
    probability: float
    sample_size: int
    source: str


def parse_probability(row: dict[str, str]) -> float | None:
    fair_probability = parse_float(row.get("fair_probability"))
    if fair_probability is not None:
        if fair_probability > 1.0:
            fair_probability /= 100.0
        return clamp_probability(fair_probability)

    yes_odds = parse_float(row.get("yes_odds"))
    no_odds = parse_float(row.get("no_odds"))
    fair = implied_probability_from_decimal_odds([yes_odds, no_odds])
    if fair is None:
        return None
    return clamp_probability(fair[0])


def load_exact_market_odds(path: Path | None) -> dict[str, ExactMarketOdds]:
    if path is None or not path.exists():
        return {}

    grouped: dict[str, list[float]] = {}
    sources: dict[str, set[str]] = {}
    with path.open("r", newline="", encoding="utf-8") as input_file:
        for row in csv.DictReader(input_file):
            market_id = (row.get("market_id") or "").strip()
            if not market_id:
                continue
            probability = parse_probability(row)
            if probability is None:
                continue
            grouped.setdefault(market_id, []).append(probability)
            source = (row.get("source") or row.get("bookmaker") or "exact market odds").strip()
            sources.setdefault(market_id, set()).add(source)

    return {
        market_id: ExactMarketOdds(
            market_id=market_id,
            probability=sum(probabilities) / len(probabilities),
            sample_size=len(probabilities),
            source=", ".join(sorted(sources.get(market_id, {"exact market odds"}))),
        )
        for market_id, probabilities in grouped.items()
    }


def blend_with_exact_odds(model_probability: float, odds: ExactMarketOdds) -> float:
    odds_weight = 0.70 if odds.sample_size >= 2 else 0.60
    return clamp_probability((model_probability * (1.0 - odds_weight)) + (odds.probability * odds_weight))

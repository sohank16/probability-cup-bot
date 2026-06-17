from __future__ import annotations

import csv
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import requests


FOOTBALL_DATA_BASE_URL = "https://www.football-data.co.uk/mmz4281"
DEFAULT_SEASONS = ["2021", "2122", "2223", "2324", "2425", "2526"]
DEFAULT_LEAGUE_CODES = ["E0", "D1", "I1", "SP1", "F1", "N1", "P1", "B1", "T1", "SC0"]


@dataclass(frozen=True)
class OddsBaseline:
    market_type: str
    metric: str
    period: str
    threshold: int
    probability: float
    sample_size: int
    source: str


def football_data_url(season: str, league_code: str) -> str:
    return f"{FOOTBALL_DATA_BASE_URL}/{season}/{league_code}.csv"


def fetch_football_data_csv(season: str, league_code: str, timeout_seconds: int = 8) -> str:
    response = requests.get(
        football_data_url(season, league_code),
        timeout=timeout_seconds,
        headers={"User-Agent": "Mozilla/5.0 probability-cup-bot/0.1"},
    )
    response.raise_for_status()
    return response.text


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_rows(csv_text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(csv_text)))


def valid_number(row: dict[str, str], column: str) -> float | None:
    return parse_float(row.get(column))


def implied_probability_from_decimal_odds(odds: list[float | None]) -> list[float] | None:
    if any(item is None or item <= 1.0 for item in odds):
        return None
    raw = [1.0 / float(item) for item in odds]
    total = sum(raw)
    if total <= 0.0:
        return None
    return [item / total for item in raw]


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def probability_over(values: list[float], threshold: int) -> float:
    if not values:
        return 0.0
    return sum(1 for value in values if value >= threshold) / len(values)


def add_threshold_baselines(
    baselines: list[OddsBaseline],
    market_type: str,
    metric: str,
    values: list[float],
    thresholds: list[int],
    source: str,
) -> None:
    for threshold in thresholds:
        baselines.append(
            OddsBaseline(
                market_type=market_type,
                metric=metric,
                period="match",
                threshold=threshold,
                probability=probability_over(values, threshold),
                sample_size=len(values),
                source=source,
            )
        )


def build_baselines_from_rows(rows: list[dict[str, str]], source: str = "Football-Data") -> list[OddsBaseline]:
    match_home_probs: list[float] = []
    match_draw_probs: list[float] = []
    match_away_probs: list[float] = []
    over_2_5_probs: list[float] = []

    team_shots_on_target: list[float] = []
    total_shots_on_target: list[float] = []
    team_corners: list[float] = []
    total_corners: list[float] = []
    team_fouls: list[float] = []
    total_fouls: list[float] = []
    team_offsides: list[float] = []
    total_offsides: list[float] = []
    team_cards: list[float] = []
    total_cards: list[float] = []

    for row in rows:
        one_x_two = implied_probability_from_decimal_odds(
            [
                valid_number(row, "AvgH") or valid_number(row, "B365H"),
                valid_number(row, "AvgD") or valid_number(row, "B365D"),
                valid_number(row, "AvgA") or valid_number(row, "B365A"),
            ]
        )
        if one_x_two:
            match_home_probs.append(one_x_two[0])
            match_draw_probs.append(one_x_two[1])
            match_away_probs.append(one_x_two[2])

        totals = implied_probability_from_decimal_odds(
            [
                valid_number(row, "Avg>2.5") or valid_number(row, "B365>2.5"),
                valid_number(row, "Avg<2.5") or valid_number(row, "B365<2.5"),
            ]
        )
        if totals:
            over_2_5_probs.append(totals[0])

        hst, ast = valid_number(row, "HST"), valid_number(row, "AST")
        if hst is not None and ast is not None:
            team_shots_on_target.extend([hst, ast])
            total_shots_on_target.append(hst + ast)

        hc, ac = valid_number(row, "HC"), valid_number(row, "AC")
        if hc is not None and ac is not None:
            team_corners.extend([hc, ac])
            total_corners.append(hc + ac)

        hf, af = valid_number(row, "HF"), valid_number(row, "AF")
        if hf is not None and af is not None:
            team_fouls.extend([hf, af])
            total_fouls.append(hf + af)

        ho, ao = valid_number(row, "HO"), valid_number(row, "AO")
        if ho is not None and ao is not None:
            team_offsides.extend([ho, ao])
            total_offsides.append(ho + ao)

        hy, ay = valid_number(row, "HY"), valid_number(row, "AY")
        hr, ar = valid_number(row, "HR") or 0.0, valid_number(row, "AR") or 0.0
        if hy is not None and ay is not None:
            home_cards = hy + hr
            away_cards = ay + ar
            team_cards.extend([home_cards, away_cards])
            total_cards.append(home_cards + away_cards)

    baselines = [
        OddsBaseline("match_winner", "home_win", "match", 1, average(match_home_probs), len(match_home_probs), source),
        OddsBaseline("match_winner", "draw", "match", 1, average(match_draw_probs), len(match_draw_probs), source),
        OddsBaseline("match_winner", "away_win", "match", 1, average(match_away_probs), len(match_away_probs), source),
        OddsBaseline("total_goals_over", "goals", "match", 3, average(over_2_5_probs), len(over_2_5_probs), source),
    ]

    add_threshold_baselines(baselines, "team_metric_over", "shots_on_target", team_shots_on_target, [2, 3, 4, 5, 6], source)
    add_threshold_baselines(baselines, "total_metric_over", "shots_on_target", total_shots_on_target, [6, 7, 8, 9, 10, 11], source)
    add_threshold_baselines(baselines, "team_metric_over", "corners", team_corners, [3, 4, 5, 6, 7], source)
    add_threshold_baselines(baselines, "total_metric_over", "corners", total_corners, [7, 8, 9, 10, 11, 12], source)
    add_threshold_baselines(baselines, "team_metric_over", "fouls", team_fouls, [8, 10, 12, 14, 16], source)
    add_threshold_baselines(baselines, "total_metric_over", "fouls", total_fouls, [18, 22, 26, 30], source)
    add_threshold_baselines(baselines, "team_metric_over", "offsides", team_offsides, [1, 2, 3, 4], source)
    add_threshold_baselines(baselines, "total_metric_over", "offsides", total_offsides, [2, 3, 4, 5, 6], source)
    add_threshold_baselines(baselines, "team_metric_over", "cards", team_cards, [1, 2, 3, 4], source)
    add_threshold_baselines(baselines, "total_metric_over", "cards", total_cards, [2, 3, 4, 5, 6], source)

    return [baseline for baseline in baselines if baseline.sample_size > 0]


def baseline_key(market_type: str, metric: str, period: str, threshold: int) -> tuple[str, str, str, int]:
    return market_type, metric, period, threshold


def baseline_lookup(baselines: list[OddsBaseline]) -> dict[tuple[str, str, str, int], OddsBaseline]:
    return {
        baseline_key(
            baseline.market_type,
            baseline.metric,
            baseline.period,
            baseline.threshold,
        ): baseline
        for baseline in baselines
    }


def write_baselines_csv(path: Path, baselines: list[OddsBaseline]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["market_type", "metric", "period", "threshold", "probability", "sample_size", "source"],
        )
        writer.writeheader()
        for baseline in baselines:
            writer.writerow(
                {
                    "market_type": baseline.market_type,
                    "metric": baseline.metric,
                    "period": baseline.period,
                    "threshold": baseline.threshold,
                    "probability": round(baseline.probability, 6),
                    "sample_size": baseline.sample_size,
                    "source": baseline.source,
                }
            )


def load_baselines_csv(path: Path) -> list[OddsBaseline]:
    with path.open("r", newline="", encoding="utf-8") as input_file:
        return [
            OddsBaseline(
                market_type=row["market_type"],
                metric=row["metric"],
                period=row["period"],
                threshold=int(row["threshold"]),
                probability=float(row["probability"]),
                sample_size=int(row["sample_size"]),
                source=row["source"],
            )
            for row in csv.DictReader(input_file)
        ]

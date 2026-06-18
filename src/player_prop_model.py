from __future__ import annotations

import unicodedata

from src.market_priors import MarketPrior
from src.probability_utils import add_logit_adjustment, clamp_probability, shrink_toward
from src.question_parser import ParsedMarket


PLAYER_PROP_MARKETS = {
    "player_shot_on_target",
    "player_goal",
    "player_goal_or_assist",
}


def normalize_player_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in ascii_name)
    return " ".join(cleaned.split())


def base_player_probability(parsed: ParsedMarket) -> float:
    if parsed.market_type == "player_shot_on_target":
        if parsed.period == "second_half":
            return 0.20
        threshold = parsed.threshold or 1
        return max(0.18, 0.40 - max(0, threshold - 1) * 0.10)
    if parsed.market_type == "player_goal":
        return 0.16
    if parsed.market_type == "player_goal_or_assist":
        return 0.28
    return 0.50


def club_goal_logit_adjustment(club_goals_2025_26: int | None, market_type: str) -> float:
    if club_goals_2025_26 is None:
        return 0.0
    if club_goals_2025_26 >= 25:
        return 0.95 if market_type == "player_shot_on_target" else 1.05
    if club_goals_2025_26 >= 15:
        return 0.70 if market_type == "player_shot_on_target" else 0.78
    if club_goals_2025_26 >= 8:
        return 0.42 if market_type == "player_shot_on_target" else 0.46
    if club_goals_2025_26 >= 3:
        return 0.20
    if club_goals_2025_26 == 0:
        return -0.45 if market_type == "player_shot_on_target" else -0.60
    return 0.0


def country_start_logit_adjustment(country_starts_last_10: int | None, market_type: str) -> float:
    if country_starts_last_10 is None:
        return 0.0
    starts = max(0, min(10, country_starts_last_10))
    if starts >= 9:
        return 0.62
    if starts >= 7:
        return 0.45
    if starts >= 5:
        return 0.22
    if starts >= 3:
        return -0.15
    if starts >= 1:
        return -0.45
    return -0.95 if market_type == "player_shot_on_target" else -0.85


def player_data_strength(form: object | None) -> float:
    if form is None:
        return 0.50
    strength = 0.50
    if getattr(form, "club_goals_2025_26", None) is not None:
        strength += 0.20
    if getattr(form, "country_starts_last_10", None) is not None:
        strength += 0.25
    return min(0.92, strength)


def predict_player_prop(
    parsed: ParsedMarket,
    player_forms: dict[str, object],
) -> MarketPrior | None:
    if parsed.market_type not in PLAYER_PROP_MARKETS or not parsed.player:
        return None

    base = base_player_probability(parsed)
    form = player_forms.get(normalize_player_name(parsed.player))
    if not form or (
        getattr(form, "club_goals_2025_26", None) is None
        and getattr(form, "country_starts_last_10", None) is None
    ):
        return MarketPrior(
            shrink_toward(base, 0.50, 0.65),
            "low",
            "Player prop routed to player model, but no player form data was available.",
        )

    club_goals = getattr(form, "club_goals_2025_26", None)
    country_starts = getattr(form, "country_starts_last_10", None)
    adjustment = club_goal_logit_adjustment(club_goals, parsed.market_type)
    adjustment += country_start_logit_adjustment(country_starts, parsed.market_type)
    raw_probability = add_logit_adjustment(base, adjustment)
    probability = shrink_toward(raw_probability, base, player_data_strength(form))

    parts = []
    if club_goals is not None:
        parts.append(f"{club_goals} club goals in 2025/26")
    if country_starts is not None:
        parts.append(f"{country_starts}/10 country starts")
    detail = ", ".join(parts) if parts else "player form fields present"
    source = getattr(form, "source", "")
    source_text = f" Source: {source}." if source else ""
    return MarketPrior(
        clamp_probability(probability),
        "medium",
        (
            "Player prop model uses logit scoring from player form "
            f"({detail}) with confidence shrinkage.{source_text}"
        ),
    )

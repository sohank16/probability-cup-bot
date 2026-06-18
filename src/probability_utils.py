from __future__ import annotations

import math


def clamp_probability(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


def logit(probability: float) -> float:
    probability = clamp_probability(probability)
    return math.log(probability / (1.0 - probability))


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def add_logit_adjustment(probability: float, adjustment: float) -> float:
    return clamp_probability(sigmoid(logit(probability) + adjustment))


def shrink_toward(probability: float, anchor: float, strength: float) -> float:
    strength = max(0.0, min(1.0, strength))
    return clamp_probability(
        sigmoid((logit(probability) * strength) + (logit(anchor) * (1.0 - strength)))
    )

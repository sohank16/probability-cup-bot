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


def poisson_pmf(lam: float, value: int) -> float:
    return math.exp(-lam) * (lam**value) / math.factorial(value)


def poisson_cdf(lam: float, value: int) -> float:
    return sum(poisson_pmf(lam, current) for current in range(value + 1))


def poisson_ge(lam: float, threshold: int) -> float:
    if threshold <= 0:
        return 1.0
    return 1.0 - poisson_cdf(lam, threshold - 1)


def poisson_greater_than(left_lambda: float, right_lambda: float, max_value: int = 14) -> float:
    probability = 0.0
    for left_value in range(max_value + 1):
        for right_value in range(max_value + 1):
            if left_value > right_value:
                probability += poisson_pmf(left_lambda, left_value) * poisson_pmf(
                    right_lambda,
                    right_value,
                )
    return probability

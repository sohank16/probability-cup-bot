import importlib.util
from pathlib import Path

from src.api_client import SportsPredictAPIError


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "06_submit_predictions.py"
SPEC = importlib.util.spec_from_file_location("submit_predictions_script", SCRIPT_PATH)
submit_predictions = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(submit_predictions)


def test_probability_to_api_int_accepts_fraction_and_percent() -> None:
    assert submit_predictions.probability_to_api_int("0.541") == 54
    assert submit_predictions.probability_to_api_int("54") == 54
    assert submit_predictions.probability_to_api_int("0") == 1
    assert submit_predictions.probability_to_api_int("100") == 99


def test_parse_existing_probability_uses_known_probability_keys() -> None:
    assert submit_predictions.parse_existing_probability({"probability": 0.541}) == 54
    assert submit_predictions.parse_existing_probability({"probability_percent": 54}) == 54
    assert submit_predictions.parse_existing_probability({"probabilityPercent": "54"}) == 54
    assert submit_predictions.parse_existing_probability({"other": "54"}) is None


def test_is_rate_limit_error_detects_429() -> None:
    error = SportsPredictAPIError("PATCH /predictions/id failed with 429: Too Many Requests")

    assert submit_predictions.is_rate_limit_error(error)

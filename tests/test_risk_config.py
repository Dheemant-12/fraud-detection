
import pytest

from src.risk_assessment import (
    get_risk_level,
    validate_risk_thresholds,
)


def test_valid_thresholds():
    validate_risk_thresholds(0.40, 0.70)


@pytest.mark.parametrize(
    "low, high",
    [
        (-0.1, 0.70),
        (0.40, 1.1),
        (0.70, 0.40),
        (0.50, 0.50),
    ],
)
def test_invalid_thresholds(low, high):
    with pytest.raises(ValueError):
        validate_risk_thresholds(low, high)


def test_default_risk_classification():
    assert get_risk_level(0.20) == "low"
    assert get_risk_level(0.50) == "medium"
    assert get_risk_level(0.80) == "high"

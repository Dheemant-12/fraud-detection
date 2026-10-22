
import pytest

from src.risk_assessment import get_risk_level


@pytest.mark.parametrize(
    "probability, expected",
    [
        (0.0, "low"),
        (0.39, "low"),
        (0.40, "medium"),
        (0.69, "medium"),
        (0.70, "high"),
        (1.0, "high"),
    ],
)
def test_risk_level(probability, expected):
    assert get_risk_level(probability) == expected


@pytest.mark.parametrize("probability", [-0.1, 1.1])
def test_invalid_probability(probability):
    with pytest.raises(ValueError):
        get_risk_level(probability)

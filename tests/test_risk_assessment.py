
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

def test_medium_risk_lower_boundary():
    assert get_risk_level(0.40) == "medium"


def test_high_risk_lower_boundary():
    assert get_risk_level(0.70) == "high"


def test_low_risk_upper_boundary():
    assert get_risk_level(0.3999) == "low"


def test_medium_risk_upper_boundary():
    assert get_risk_level(0.6999) == "medium"

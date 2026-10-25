
from src.risk_assessment import get_risk_level


def test_configured_low_risk_boundary():
    assert get_risk_level(0.40) == "medium"


def test_configured_high_risk_boundary():
    assert get_risk_level(0.70) == "high"


def test_probability_below_low_boundary():
    assert get_risk_level(0.39) == "low"


def test_probability_between_boundaries():
    assert get_risk_level(0.55) == "medium"


def test_probability_above_high_boundary():
    assert get_risk_level(0.90) == "high"

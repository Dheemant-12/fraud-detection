
from src.config import (
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)


def validate_risk_thresholds(
    low_threshold: float = LOW_RISK_THRESHOLD,
    high_threshold: float = HIGH_RISK_THRESHOLD,
) -> None:
    if not 0 <= low_threshold <= 1:
        raise ValueError(
            "Low risk threshold must be between 0 and 1"
        )

    if not 0 <= high_threshold <= 1:
        raise ValueError(
            "High risk threshold must be between 0 and 1"
        )

    if low_threshold >= high_threshold:
        raise ValueError(
            "Low risk threshold must be less than high risk threshold"
        )


def get_risk_level(fraud_probability: float) -> str:
    validate_risk_thresholds()

    if not 0 <= fraud_probability <= 1:
        raise ValueError(
            "Fraud probability must be between 0 and 1"
        )

    if fraud_probability >= HIGH_RISK_THRESHOLD:
        return "high"

    if fraud_probability >= LOW_RISK_THRESHOLD:
        return "medium"

    return "low"

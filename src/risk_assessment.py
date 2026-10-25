
from src.config import (
    LOW_RISK_THRESHOLD,
    HIGH_RISK_THRESHOLD,
)


def get_risk_level(fraud_probability: float) -> str:
    if not 0 <= fraud_probability <= 1:
        raise ValueError(
            "Fraud probability must be between 0 and 1"
        )

    if fraud_probability >= HIGH_RISK_THRESHOLD:
        return "high"

    if fraud_probability >= LOW_RISK_THRESHOLD:
        return "medium"

    return "low"

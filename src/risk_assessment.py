
def get_risk_level(fraud_probability: float) -> str:
    if not 0 <= fraud_probability <= 1:
        raise ValueError(
            "Fraud probability must be between 0 and 1"
        )

    if fraud_probability >= 0.70:
        return "high"

    if fraud_probability >= 0.40:
        return "medium"

    return "low"

from src.predict import predict_transaction


REQUIRED_FIELDS = {
    "transaction_amount",
    "transaction_type",
    "payment_mode",
    "device_type",
    "device_location",
    "account_age_days",
    "transaction_hour",
    "previous_failed_attempts",
    "avg_transaction_amount",
    "is_international",
    "ip_risk_score",
    "login_attempts_last_24h",
    "hist_mean",
    "hist_std",
    "amount_deviation",
    "user_transaction_count",
    "amount_ratio",
    "risk_login_interaction",
}


def predict(transaction: dict):
    missing_fields = REQUIRED_FIELDS - transaction.keys()

    if missing_fields:
        raise ValueError(
            f"Missing required fields: {sorted(missing_fields)}"
        )

    return predict_transaction(transaction)
import pandas as pd

from src.config import PREDICTION_THRESHOLD
from src.model_loader import load_model


def predict_transaction(transaction: dict):
    saved_model = load_model()

    model = saved_model["model"]
    feature_columns = saved_model["features"]

    df = pd.DataFrame([transaction])

    df = df.drop(
        columns=["transaction_id", "user_id"],
        errors="ignore",
    )

    df = pd.get_dummies(
        df,
        drop_first=True,
    )

    df = df.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    fraud_probability = model.predict_proba(df)[0][1]

    prediction = int(
        fraud_probability >= PREDICTION_THRESHOLD
    )

    return {
        "fraud_probability": fraud_probability,
        "prediction": prediction,
    }


if __name__ == "__main__":
    sample_transaction = {
        "transaction_amount": 2500,
        "transaction_type": "online",
        "payment_mode": "card",
        "device_type": "mobile",
        "device_location": "foreign",
        "account_age_days": 100,
        "transaction_hour": 2,
        "previous_failed_attempts": 3,
        "avg_transaction_amount": 500,
        "is_international": 1,
        "ip_risk_score": 80,
        "login_attempts_last_24h": 5,
        "hist_mean": 500,
        "hist_std": 100,
        "amount_deviation": 20,
        "user_transaction_count": 10,
        "amount_ratio": 5,
        "risk_login_interaction": 400,
    }

    result = predict_transaction(sample_transaction)

    print("\nPrediction:")
    print(result)
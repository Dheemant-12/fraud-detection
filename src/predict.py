import joblib
import pandas as pd

from src.config import MODEL_DIR


def predict_transaction(transaction: dict):
    # Load saved model
    saved_model = joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    )

    model = saved_model["model"]
    feature_columns = saved_model["features"]

    # Convert transaction into DataFrame
    df = pd.DataFrame([transaction])

    # Remove identifiers
    df = df.drop(
        columns=["transaction_id", "user_id"],
        errors="ignore",
    )

    # Convert categorical columns
    df = pd.get_dummies(
        df,
        drop_first=True,
    )

    # Match training features
    df = df.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    # Make prediction
    fraud_probability = model.predict_proba(
        df
    )[0][1]

    prediction = int(
        fraud_probability >= 0.5
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

    result = predict_transaction(
        sample_transaction
    )

    print("\nPrediction:")
    print(result)
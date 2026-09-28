from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == (
        "Fraud Detection API is running"
    )


def test_predict():
    transaction = {
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

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 200

    result = response.json()

    assert "fraud_probability" in result
    assert "prediction" in result

    assert 0 <= result["fraud_probability"] <= 1
    assert result["prediction"] in [0, 1]
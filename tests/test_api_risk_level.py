
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


@pytest.mark.parametrize(
    "probability, expected_risk",
    [
        (0.20, "low"),
        (0.50, "medium"),
        (0.85, "high"),
    ],
)
def test_prediction_returns_correct_risk_level(
    probability,
    expected_risk,
):
    mocked_result = {
        "fraud_probability": probability,
        "prediction": int(probability >= 0.5),
    }

    with patch(
        "src.api.predict_service",
        return_value=mocked_result.copy(),
    ):
        response = client.post(
            "/predict",
            json={
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
            },
        )

    assert response.status_code == 200
    assert response.json()["risk_level"] == expected_risk

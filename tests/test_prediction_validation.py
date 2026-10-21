
import pytest
from unittest.mock import patch

from src.prediction_service import predict


def test_missing_required_field():
    transaction = {
        "transaction_amount": 2500,
    }

    with pytest.raises(
        ValueError,
        match="Missing required fields",
    ):
        predict(transaction)


def test_multiple_missing_fields():
    transaction = {
        "transaction_amount": 2500,
        "transaction_type": "online",
        "payment_mode": "card",
    }

    with pytest.raises(
        ValueError,
        match="Missing required fields",
    ):
        predict(transaction)


@patch("src.prediction_service.predict_transaction")
def test_invalid_input_does_not_call_model(mock_predict):
    transaction = {
        "transaction_amount": 2500,
    }

    with pytest.raises(ValueError, match="Missing required fields"):
        predict(transaction)

    mock_predict.assert_not_called()

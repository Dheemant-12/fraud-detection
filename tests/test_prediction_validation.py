import pytest

from src.prediction_service import predict


def test_missing_required_field():
    transaction = {
        "transaction_amount": 2500,
    }

    with pytest.raises(ValueError, match="Missing required fields"):
        predict(transaction)
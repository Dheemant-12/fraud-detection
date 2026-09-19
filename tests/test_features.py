import pandas as pd

from src.features import FeatureBuilder


def test_amount_deviation():
    df = pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U2"],
            "transaction_amount": [100.0, 200.0, 500.0],
        }
    )

    builder = FeatureBuilder()
    result = builder.fit_transform(df)

    assert "hist_mean" in result.columns
    assert "hist_std" in result.columns
    assert "amount_deviation" in result.columns


def test_user_transaction_count():
    df = pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U2"],
            "transaction_amount": [100.0, 200.0, 500.0],
        }
    )

    builder = FeatureBuilder()
    result = builder.fit_transform(df)

    assert "user_transaction_count" in result.columns
    assert result.loc[result["user_id"] == "U1", "user_transaction_count"].iloc[0] == 2
    assert result.loc[result["user_id"] == "U2", "user_transaction_count"].iloc[0] == 1
def test_amount_ratio():
    df = pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U2"],
            "transaction_amount": [100.0, 200.0, 500.0],
        }
    )

    builder = FeatureBuilder()
    result = builder.fit_transform(df)

    assert "amount_ratio" in result.columns
    assert result["amount_ratio"].notna().all()
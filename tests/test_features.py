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
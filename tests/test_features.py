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
def test_risk_login_interaction():
    df = pd.DataFrame(
        {
            "user_id": ["U1", "U1"],
            "transaction_amount": [100.0, 200.0],
            "ip_risk_score": [10, 20],
            "login_attempts_last_24h": [2, 3],
        }
    )

    builder = FeatureBuilder()
    result = builder.fit_transform(df)

    assert "risk_login_interaction" in result.columns
    assert result.loc[0, "risk_login_interaction"] == 20
    assert result.loc[1, "risk_login_interaction"] == 60
def test_all_features_are_generated():
    df = pd.DataFrame(
        {
            "user_id": ["U1", "U1", "U2"],
            "transaction_amount": [100.0, 200.0, 500.0],
            "ip_risk_score": [10, 20, 30],
            "login_attempts_last_24h": [1, 2, 3],
        }
    )

    builder = FeatureBuilder()
    result = builder.fit_transform(df)

    expected_features = [
        "hist_mean",
        "hist_std",
        "amount_deviation",
        "user_transaction_count",
        "amount_ratio",
        "risk_login_interaction",
    ]

    for feature in expected_features:
        assert feature in result.columns
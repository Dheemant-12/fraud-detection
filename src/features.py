import pandas as pd


class FeatureBuilder:
    def __init__(self):
        self.user_history_ = None
        self.user_transaction_count_ = None

    def fit(self, df: pd.DataFrame):
        # Calculate historical transaction statistics per user
        self.user_history_ = (
            df.groupby("user_id")["transaction_amount"]
            .agg(["mean", "std"])
            .rename(
                columns={
                    "mean": "hist_mean",
                    "std": "hist_std",
                }
            )
        )

        # Calculate transaction count per user
        self.user_transaction_count_ = (
            df.groupby("user_id")
            .size()
            .rename("user_transaction_count")
        )

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # Add historical amount statistics
        out = out.merge(
            self.user_history_,
            on="user_id",
            how="left",
        )

        # Add amount deviation
        out["amount_deviation"] = (
            (out["transaction_amount"] - out["hist_mean"])
            / out["hist_std"].replace(0, 1)
        )

        # Add transaction count
        out = out.merge(
            self.user_transaction_count_,
            on="user_id",
            how="left",
        )

        # Add amount ratio
        out["amount_ratio"] = (
            out["transaction_amount"]
            / out["hist_mean"].replace(0, 1)
        )

        # Add risk and login interaction feature
        if {
            "ip_risk_score",
            "login_attempts_last_24h",
        }.issubset(out.columns):
            out["risk_login_interaction"] = (
                out["ip_risk_score"]
                * out["login_attempts_last_24h"]
            )

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
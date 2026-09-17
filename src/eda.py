from src.data_loader import load_raw_data
from src.config import TARGET_COL


def main():
    df = load_raw_data()

    print("\n===== DATASET OVERVIEW =====")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\n===== FRAUD DISTRIBUTION =====")
    fraud_counts = df[TARGET_COL].value_counts()
    print(fraud_counts)

    fraud_rate = df[TARGET_COL].mean() * 100
    print(f"\nFraud rate: {fraud_rate:.2f}%")

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== NUMERIC FEATURES =====")

    numeric_df = df.select_dtypes(include="number")

    correlations = (
        numeric_df.corr()[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .sort_values(ascending=False)
    )

    print(correlations)

    print("\n===== FRAUD BY TRANSACTION HOUR =====")

    fraud_by_hour = (
        df.groupby("transaction_hour")[TARGET_COL]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    print(fraud_by_hour)

    print("\n===== CATEGORICAL FEATURES =====")

    categorical_columns = [
        "transaction_type",
        "payment_mode",
        "device_type",
        "device_location",
    ]

    for column in categorical_columns:
        print(f"\n--- {column} ---")

        fraud_by_category = (
            df.groupby(column)[TARGET_COL]
            .mean()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
        )

        print(fraud_by_category)

    print("\n===== EDA COMPLETE =====")


if __name__ == "__main__":
    main()
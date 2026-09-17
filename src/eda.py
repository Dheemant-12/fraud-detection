from src.data_loader import load_raw_data
from src.config import TARGET_COL


def main():
    df = load_raw_data()

    print("\n===== DATA INFO =====")
    df.info()

    print("\n===== DATA DESCRIPTION =====")
    print(df.describe())

    print("\n===== FIRST 5 ROWS =====")
    print(df.head())

    print("\n===== DATASET SHAPE =====")
    print(df.shape)

    print("\n===== FRAUD COUNTS =====")
    print(df[TARGET_COL].value_counts())

    print("\n===== FRAUD RATE =====")
    fraud_rate = df[TARGET_COL].mean() * 100
    print(f"Fraud rate: {fraud_rate:.2f}%")

    print("\n===== FRAUD BY TRANSACTION HOUR =====")
    fraud_by_hour = (
        df.groupby("transaction_hour")[TARGET_COL]
        .mean()
        .mul(100)
        .round(2)
    )
    print(fraud_by_hour)

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== DATA TYPES =====")
    print(df.dtypes)

    print("\n===== UNIQUE VALUES =====")
    print(df.nunique())

    print("\n===== TOP 10 NUMERIC FEATURES BY CORRELATION =====")

    numeric_df = df.select_dtypes(include="number")

    correlations = (
        numeric_df.corr()[TARGET_COL]
        .drop(TARGET_COL)
        .abs()
        .sort_values(ascending=False)
        .head(10)
    )

    print(correlations)

    print("\n===== CATEGORICAL FEATURE ANALYSIS =====")

    categorical_columns = [
        "transaction_type",
        "payment_mode",
        "device_type",
        "device_location",
    ]

    for column in categorical_columns:
        print(f"\n--- {column} ---")

        print("Number of categories:", df[column].nunique())

        fraud_by_category = (
            df.groupby(column)[TARGET_COL]
            .mean()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
        )

        print("Fraud rate by category:")
        print(fraud_by_category)


if __name__ == "__main__":
    main()
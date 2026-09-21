import pandas as pd

from src.config import PROCESSED_DATA_DIR, TARGET_COL
from src.data_loader import time_based_split


def main():
    # Load processed dataset
    df = pd.read_csv(
        PROCESSED_DATA_DIR / "transactions_features.csv"
    )

    # Split dataset
    train, val, test = time_based_split(df)

    # Save splits
    train.to_csv(
        PROCESSED_DATA_DIR / "train.csv",
        index=False,
    )

    val.to_csv(
        PROCESSED_DATA_DIR / "validation.csv",
        index=False,
    )

    test.to_csv(
        PROCESSED_DATA_DIR / "test.csv",
        index=False,
    )

    print("Data splitting complete")
    print(f"Train shape: {train.shape}")
    print(f"Validation shape: {val.shape}")
    print(f"Test shape: {test.shape}")

    print("\nFraud distribution:")
    print(f"Train fraud rate: {train[TARGET_COL].mean():.4f}")
    print(f"Validation fraud rate: {val[TARGET_COL].mean():.4f}")
    print(f"Test fraud rate: {test[TARGET_COL].mean():.4f}")


if __name__ == "__main__":
    main()
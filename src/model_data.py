import pandas as pd

from src.config import PROCESSED_DATA_DIR, TARGET_COL


def load_model_data(split_name: str):
    file_path = PROCESSED_DATA_DIR / f"{split_name}.csv"

    df = pd.read_csv(file_path)

    # Remove target from input features
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Remove identifier columns
    columns_to_drop = [
        "transaction_id",
        "user_id",
    ]

    X = X.drop(
        columns=columns_to_drop,
        errors="ignore",
    )

    # Convert categorical columns into dummy variables
    X = pd.get_dummies(
        X,
        drop_first=True,
    )

    return X, y


def main():
    X_train, y_train = load_model_data("train")

    print("Model data preparation complete")
    print(f"X shape: {X_train.shape}")
    print(f"y shape: {y_train.shape}")

    print("\nFeature columns:")
    print(X_train.columns.tolist())

    print("\nTarget distribution:")
    print(y_train.value_counts())


if __name__ == "__main__":
    main()
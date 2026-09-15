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


if __name__ == "__main__":
    main()
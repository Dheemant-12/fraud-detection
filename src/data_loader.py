import pandas as pd

from src.config import RAW_DATA_DIR


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_DIR / "transactions.csv")

    return df.reset_index(drop=True)


def time_based_split(df: pd.DataFrame, train_frac=0.7, val_frac=0.15):
    n = len(df)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    return train, val, test
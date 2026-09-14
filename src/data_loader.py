import pandas as pd
from src.config import RAW_DATA_DIR, TIME_COL
def load_raw_data()-> pd.DataFrame:
    df=pd.read_csv(RAW_DATA_DIR/"transactions.csv")
    df[TIME_COL]=pd.to_datetime(df[TIME_COL])
    return df.sort_values(TIME_COL).reset_index(drop=True)
def time_based_split(df:pd.DataFrame,train_frac=0.7,val_frac=0.15):
    n=len(df)
    train_end=int(n*train_frac)
    val_end=train_end+int(n*val_frac)
    train_df=df.iloc[:train_end]
    val_df=df.iloc[train_end:val_end]
    test_df=df.iloc[val_end:]
    return train_df,val_df,test_df
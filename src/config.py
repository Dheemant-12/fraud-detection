from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
RAW_DATA_DIR=ROOT/'data/raw'
PROCESSED_DATA_DIR=ROOT/'data/processed'
MODEL_DIR=ROOT/'models'
RANDOM_SEED=42
TARGET_COL="is_fraud"
TIME_COL="traansaction_time"
#cost matrix_dollars
COST_FALSE_NEGATIVE=500
COST_FALSE_POSITIVE=50

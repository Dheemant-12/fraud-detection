from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT / "data" / "raw"
PROCESSED_DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

RANDOM_SEED = 42

TARGET_COL = "fraud_label"

# Cost matrix — dollars
COST_FALSE_NEGATIVE = 500
COST_FALSE_POSITIVE = 50
import joblib

from src.config import MODEL_DIR


def load_model():
    model_path = MODEL_DIR / "logistic_regression.pkl"

    return joblib.load(model_path)
import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from src.config import MODEL_DIR
from src.model_data import load_model_data


def main():
    # Load saved model
    saved_model = joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    )

    model = saved_model["model"]
    feature_columns = saved_model["features"]

    # Load test data
    X_test, y_test = load_model_data("test")

    # Match test columns with training columns
    X_test = X_test.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    # Get fraud probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
                "f1_score": f1_score(
                    y_test,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    results_df = pd.DataFrame(results)

    print("\nThreshold Analysis:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
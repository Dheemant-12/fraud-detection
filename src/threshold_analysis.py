import joblib
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.config import (
    MODEL_DIR,
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
)

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

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
        ).ravel()

        total_cost = (
            fn * COST_FALSE_NEGATIVE
            + fp * COST_FALSE_POSITIVE
        )

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
                "false_positive_cost": (
                    fp * COST_FALSE_POSITIVE
                ),
                "false_negative_cost": (
                    fn * COST_FALSE_NEGATIVE
                ),
                "total_cost": (
                    fp * COST_FALSE_POSITIVE
                    + fn * COST_FALSE_NEGATIVE
                ),
                "false_positives": fp,
                "false_negatives": fn,
                "total_cost": total_cost,
            }
        )

    results_df = pd.DataFrame(results)

    print("\nCost-Sensitive Threshold Analysis:")
    print(
        results_df.to_string(index=False)
    )


if __name__ == "__main__":
    main()
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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

    # Generate predictions
    predictions = model.predict(X_test)

    # Display evaluation results
    print("Accuracy:")
    print(accuracy_score(y_test, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
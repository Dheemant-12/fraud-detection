import json

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
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

    # Generate fraud probabilities
    probabilities = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    results = {
        "accuracy": accuracy_score(y_test, predictions),
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
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            predictions,
        ).tolist(),
    }

    # Print results
    print("Model Evaluation Results:")

    for metric, value in results.items():
        print(f"{metric}: {value}")

    # Save results
    output_path = MODEL_DIR / "evaluation_results.json"

    with open(output_path, "w") as file:
        json.dump(results, file, indent=4)

    print(
        f"\nEvaluation results saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
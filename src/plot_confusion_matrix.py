import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

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

    # Plot confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Legitimate", "Fraud"],
    )

    plt.title("Fraud Detection Confusion Matrix")
    plt.tight_layout()

    # Save plot
    output_path = MODEL_DIR / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300)

    print(f"Confusion matrix saved to: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
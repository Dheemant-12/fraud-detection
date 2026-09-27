import joblib
import pandas as pd

from src.config import MODEL_DIR


def main():
    # Load saved model
    saved_model = joblib.load(
        MODEL_DIR / "logistic_regression.pkl"
    )

    model = saved_model["model"]
    feature_columns = saved_model["features"]

    # Get model coefficients
    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": model.coef_[0],
        }
    )

    # Absolute coefficient = strength of influence
    importance["absolute_coefficient"] = (
        importance["coefficient"].abs()
    )

    # Sort by importance
    importance = importance.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    print("\nTop 15 Features:")
    print(
        importance.head(15).to_string(
            index=False
        )
    )

    # Save results
    output_path = (
        MODEL_DIR / "feature_importance.csv"
    )

    importance.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nFeature importance saved to: {output_path}"
    )


if __name__ == "__main__":
    main()
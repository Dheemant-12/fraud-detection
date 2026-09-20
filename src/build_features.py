from src.data_loader import load_raw_data
from src.features import FeatureBuilder
from src.config import PROCESSED_DATA_DIR


def main():
    # Load raw dataset
    df = load_raw_data()

    # Generate features
    builder = FeatureBuilder()
    feature_df = builder.fit_transform(df)

    # Create processed directory if needed
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save processed dataset
    output_path = PROCESSED_DATA_DIR / "transactions_features.csv"
    feature_df.to_csv(output_path, index=False)

    print(f"Processed dataset saved to: {output_path}")
    print(f"Original shape: {df.shape}")
    print(f"Processed shape: {feature_df.shape}")
    print("\nGenerated columns:")
    print(feature_df.columns.tolist())


if __name__ == "__main__":
    main()
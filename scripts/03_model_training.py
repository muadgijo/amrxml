"""Phase 1: train XGBoost model using the unified pipeline."""

from scripts.pipeline import PATHS, full_preprocess, train_model


def main() -> None:
    print("=" * 70)
    print("AMR-X PHASE 1: Model Training")
    print("=" * 70)

    # Ensure data is ready, then train
    full_preprocess(PATHS) if not PATHS.ready.exists() else None
    metadata = train_model(PATHS)

    print("\nTest metrics:")
    for key, value in metadata["test_metrics"].items():
        print(f"  {key}: {value:.3f}")
    print(f"Accuracy 95% CI: [{metadata['accuracy_ci'][0]:.3f}, {metadata['accuracy_ci'][1]:.3f}]")
    print("✅ Training complete (artifacts under models/)")


if __name__ == "__main__":
    main()

"""Phase 2: create model visualizations using the unified pipeline."""

from scripts.pipeline import PATHS, full_train, generate_visualizations


def main() -> None:
    print("=" * 70)
    print("AMR-X PHASE 2: Visualization")
    print("=" * 70)

    # Ensure model exists, then render plots
    full_train(PATHS) if not PATHS.model_path.exists() else None
    generate_visualizations(PATHS)
    print("✅ Visualization complete (files under outputs/)")


if __name__ == "__main__":
    main()

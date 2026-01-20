"""Phase 3: inference helpers using the unified pipeline artifacts."""

from scripts.pipeline import PATHS, demo_predictions, full_train, load_inference_assets


def main() -> None:
    print("=" * 70)
    print("AMR-X PHASE 3: Inference Engine")
    print("=" * 70)

    full_train(PATHS) if not PATHS.model_path.exists() else None
    assets = load_inference_assets(PATHS)
    print(
        f"✓ Loaded model, {len(assets['organism_lookup']):,} organisms, {len(assets['antibiotic_lookup']):,} antibiotics"
    )
    demo_predictions(PATHS)
    print("✅ Inference ready (use predict_resistance and rank_antibiotics from pipeline.py)")


if __name__ == "__main__":
    main()

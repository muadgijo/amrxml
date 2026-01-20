"""Phase 2: run baseline models on the unified splits."""

from scripts.pipeline import PATHS, full_preprocess, run_baselines


def main() -> None:
    print("=" * 70)
    print("AMR-X PHASE 2: Baseline Comparisons")
    print("=" * 70)

    full_preprocess(PATHS) if not PATHS.ready.exists() else None
    results = run_baselines(PATHS)
    print("\n✅ Baseline comparison complete")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

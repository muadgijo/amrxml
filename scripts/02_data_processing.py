"""Phase 0: combine raw files, clean, encode, and save processed data."""

from scripts.pipeline import PATHS, combine_raw_datasets, ensure_directories, normalize_and_encode


def main() -> None:
    print("=" * 70)
    print("AMR-X PHASE 0: Processing & Encoding")
    print("=" * 70)

    ensure_directories()
    combine_raw_datasets()
    processed = normalize_and_encode()

    print("\nDataset summary:")
    print(f"  rows: {len(processed):,}")
    print(f"  unique organisms: {processed['organism_code'].nunique():,}")
    print(f"  unique antibiotics: {processed['antibiotic_code'].nunique():,}")
    print(f"  resistance rate: {(processed['label_binary'] == 1).mean():.1%}")
    print("✅ Processing complete (artifacts under data/processed)")


if __name__ == "__main__":
    main()

"""Phase 0: download raw datasets into the unified pipeline layout."""

from scripts.pipeline import PATHS, download_datasets, ensure_directories


def main() -> None:
    print("=" * 70)
    print("AMR-X PHASE 0: Download")
    print("=" * 70)
    ensure_directories(PATHS)
    download_datasets(PATHS)
    print("✅ Download complete (stored under data/raw)")


if __name__ == "__main__":
    main()

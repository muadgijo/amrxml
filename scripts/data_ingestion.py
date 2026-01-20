"""Deprecated wrapper. Use scripts/01_data_download.py or pipeline.download_datasets."""

from scripts.pipeline import PATHS, download_datasets, ensure_directories


def main() -> None:
    ensure_directories(PATHS)
    download_datasets(PATHS)


if __name__ == "__main__":
    main()
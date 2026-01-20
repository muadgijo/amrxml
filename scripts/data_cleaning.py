"""Compatibility wrapper that now delegates to the unified pipeline."""

from scripts.pipeline import PATHS, full_preprocess


def main() -> None:
    full_preprocess(PATHS)


if __name__ == "__main__":
    main()
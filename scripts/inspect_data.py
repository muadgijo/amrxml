import pandas as pd

from scripts.pipeline import PATHS


def inspect_dataset(path: str, name: str) -> None:
    print("\n" + "=" * 60)
    print(f"Inspecting: {name}")
    print("=" * 60)

    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("\nColumns:")
    for col in df.columns:
        print("-", col)
    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    inspect_dataset(str(PATHS["raw_main"]), "Main AMR Dataset")
    inspect_dataset(str(PATHS["raw_implied"]), "Implied Susceptibility Dataset")

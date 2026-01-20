"""Ablation: train model using only antibiotic_code (to measure feature importance)."""

import json
from pathlib import Path
from scripts.pipeline import PATHS, _train_val_test_split, train_ablation_model
import pandas as pd


def main():
    print("=" * 50)
    print("ABLATION: Antibiotic-Only Model")
    print("=" * 50)

    df = pd.read_csv(PATHS["ready"])
    X = df[["antibiotic_code"]].to_numpy()
    y = df["label_binary"].to_numpy(dtype=int)

    X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split(X, y)

    print("\nTraining...")
    results = train_ablation_model(X_train, y_train, X_test, y_test, feature_names=["antibiotic_code"])

    print(f"\nResults: Accuracy={results['accuracy']:.4f}, ROC-AUC={results['roc_auc']:.4f}")

    output = PATHS["outputs_dir"] / "ablation_antibiotic_only.json"
    PATHS["outputs_dir"].mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved: {output}")


if __name__ == "__main__":
    main()

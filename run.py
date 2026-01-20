#!/usr/bin/env python
"""Simple entry point to run the full pipeline."""

from scripts.pipeline import full_preprocess, full_train, generate_visualizations, run_baselines
from pathlib import Path

if __name__ == "__main__":
    # Check if data exists, if not preprocess
    if not Path("data/processed/AMR_X_ML_ready.csv").exists():
        print("\n📥 Preprocessing data...")
        full_preprocess()
    
    print("\n🤖 Training model...")
    full_train()
    
    print("\n📊 Generating visualizations...")
    generate_visualizations()
    
    print("\n📈 Running baselines...")
    run_baselines()
    
    print("\n✓ All done!")

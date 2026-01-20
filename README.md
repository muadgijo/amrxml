# AMR-X: Antimicrobial Resistance Prediction

A machine learning model to predict antimicrobial resistance from organism and antibiotic data.

## Quick Start (For Teammates)

### First Time Setup

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies (if needed)
pip install -r requirements.txt
```

### Run Everything

```powershell
python run.py
```

Done. The script will:
1. Download datasets (first time only)
2. Clean and prepare data
3. Train the XGBoost model
4. Generate plots (ROC, PR curve, confusion matrix, feature importance)
5. Compare against baseline models

**Output:**
- `models/amr_xgb_model.json` — trained model
- `models/model_metadata.json` — metrics
- `outputs/` — visualization plots

## Project Structure

```
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned & encoded
├── models/               # Saved model + metadata
├── outputs/              # Plots
├── scripts/
│   └── pipeline.py       # Main ML code
├── run.py                # Simple entry point
└── requirements.txt
```

## What It Does

**Data:** Loads organism, antibiotic, and resistance labels. Cleans, normalizes, and encodes as integers.

**Model:** XGBoost binary classifier trained on organism + antibiotic features using stratified train-val-test split.

**Output:** Probability of resistance + optimal classification threshold.

**Results:**
- Accuracy: 68.25%
- ROC-AUC: 0.8063
- Sensitivity: 87% (catches resistant cases)

## Using the Model

```python
from scripts.pipeline import load_inference_assets, predict_resistance

assets = load_inference_assets()
result = predict_resistance("E. coli", "ciprofloxacin", assets)
print(result)
# {'organism': 'ESCHERICHIA COLI', 'antibiotic': 'CIPROFLOXACIN', 
#  'probability': 0.8234, 'prediction': 'Resistant'}
```

## Streamlit App

Interactive UI for predictions and antibiotic ranking.

```powershell
streamlit run streamlit_app.py
```

- Run `python run.py` once first so the model, maps, and threshold are saved.
- Pick an organism and antibiotic from dropdowns to see resistance probability and label.
- Move the slider to view the top-N antibiotics ranked by predicted effectiveness.

## Requirements

Python 3.11+, dependencies in `requirements.txt`


## Design Rationale

**Why XGBoost over frequency tables?**
- Frequency tables provide point estimates but fail under sparsity and rare combinations.
- XGBoost offers smoothing, regularization, and calibrated probabilities.
- Extensible to richer features if data becomes available.
- Enables principled uncertainty quantification.

**Why low precision (0.32)?**
- Resistance prevalence is 16.6%; threshold optimized for screening sensitivity (0.82).
- Clinical use requires confirmatory testing; system explicitly communicates uncertainty.
- Trade-off is intentional and documented.

## Notes
- Splits are fixed with seed 42 to keep results comparable across runs.
- Inference uses fuzzy matching for organism/antibiotic names and the saved optimal threshold from metadata.
- Deprecated helper scripts (`data_ingestion.py`, `data_processing.py`, `data_cleaning.py`) now forward to the unified pipeline to avoid divergent outputs.

# AMR-X: Antimicrobial Resistance Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amrxml-93kfeuxsjop6g5g5yy4ydn.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning model to predict antimicrobial resistance (AMR) using XGBoost. Trained on real-world clinical data to help identify effective antibiotics for bacterial infections.

##  Live Demo

**Try it now:** [https://amrxml.streamlit.app/](https://amrxml.streamlit.app/)

Select an organism and antibiotic to predict resistance probability and see recommended treatment options.

---

##  Features

- **Resistance Prediction**: Predict if a bacteria will be resistant to a specific antibiotic
- **Treatment Ranking**: Get ranked list of most effective antibiotics for an organism
- **Fast Inference**: Lightweight XGBoost model with instant predictions
- **Real Clinical Data**: Trained on curated antimicrobial susceptibility datasets

---

##  Model Scope & Intended Use

**What This Model Does:**
- Estimates global resistance patterns for organism-antibiotic pairs
- Ranks treatment options based on resistance probability
- Provides awareness and education about antimicrobial resistance trends
- Supports surveillance and decision-making in data-limited regions

**What This Model Doesn't Do:**
- Make patient-specific clinical diagnoses
- Predict regional resistance rates or local patterns
- Replace culture and sensitivity testing
- Forecast resistance trends over time

**Important Context:**
This model reflects global resistance patterns aggregated across clinical datasets. It's designed as a comparative tool and awareness resource, especially useful where local resistance data isn't available. Always contextualize results with your local antibiogram and clinical judgment.

**When to Use:**
- Educational purposes about AMR
- Awareness and surveillance support
- Treatment ranking in resource-limited settings
- Decision support when local data isn't available

**When NOT to Use:**
- As a sole basis for clinical decisions
- Without confirming with culture testing
- For region-specific resistance predictions

---

##  How It Works

1. **Input**: Select organism (e.g., *E. coli*) and antibiotic (e.g., Ciprofloxacin)
2. **Model**: XGBoost binary classifier trained on organism-antibiotic pairs
3. **Output**: Probability of resistance + ranked treatment alternatives

**Model Details:**
- Algorithm: XGBoost (Gradient Boosting)
- Features: Organism code, Antibiotic code (encoded)
- Training Data: ~1.4M clinical observations
- Performance: 68.2% accuracy, 0.81 ROC-AUC, 82% sensitivity

---

##  What This Model Does (and Doesn't)

**AMR-X generates a baseline resistance risk signal** based on global clinical data patterns. It's designed for surveillance, awareness, and comparative analysis in data-scarce regions.

### What It Learns
- Structural resistance relationships (organism-antibiotic mechanisms)
- Comparative risk across antibiotic classes
- Treatment ranking for specific organisms
- Organism vulnerability profiles

### What It Doesn't Predict
- Regional resistance rates (e.g., "% E. coli resistance in Kerala")
- Temporal resistance evolution or future trends
- Patient-specific clinical outcomes
- Healthcare system-specific patterns

### Why It Matters for LMICs
Many low-resource regions lack:
- Standardized AMR surveillance systems
- Open resistance datasets
- Resources for comprehensive culture testing

AMR-X fills this gap by providing a **starting point** for treatment awareness and a **comparative framework** to be validated against local data.

**Key Principle:** Structural resistance patterns often transfer across regions; absolute percentages do not. This model provides the former.

---

##  Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/muadgijo/amrxml.git
cd amrxml

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Train Model from Scratch (Optional)

```bash
# Download data, preprocess, and train
python run.py
```

This will:
- Download datasets from Hugging Face
- Preprocess and encode data
- Train XGBoost model
- Generate evaluation plots

---

##  Project Structure

```
amrxml/
├── streamlit_app.py          # Web interface
├── run.py                     # Full pipeline runner
├── requirements.txt           # Dependencies
├── scripts/
│   ├── pipeline.py           # Core ML pipeline
│   ├── 01_data_download.py   # Data ingestion
│   ├── 02_data_processing.py # Preprocessing
│   ├── 03_model_training.py  # Model training
│   ├── 04_model_evaluation.py # Evaluation
│   └── 05_visualization.py   # Plots generation
├── data/
│   ├── processed/            # Processed datasets & lookups
│   └── raw/                  # (Downloaded automatically)
├── models/
│   ├── amr_xgb_model.json   # Trained model
│   └── model_metadata.json   # Model info
└── outputs/                  # Visualizations (ROC, confusion matrix, etc.)
```

---

##  Model Performance

The model is evaluated on held-out test data:

- **Accuracy**: 68.2%
- **ROC-AUC**: 0.81
- **Sensitivity**: 82.0% (catches resistant cases)
- **Specificity**: 65.5% (correctly identifies susceptible cases)

The 68% accuracy reflects the real-world difficulty of AMR prediction. High sensitivity (82%) means the model prioritizes safety by being conservative — it errs on the side of warning about resistance rather than missing it.

---

##  Data Sources

Datasets are automatically downloaded from Hugging Face:
- [AMR Dataset (Cleaned & Preprocessed)](https://huggingface.co/datasets/Muadgijo/amrx-datasets)
- Microbiology cultures with implied susceptibility

---

##  Usage Example

```python
from scripts.pipeline import load_inference_assets, predict_resistance

# Load model and lookups
assets = load_inference_assets()

# Predict resistance
result = predict_resistance(
    organism="Escherichia coli",
    antibiotic="Ciprofloxacin",
    assets=assets
)

print(result)
# {'organism': 'Escherichia coli', 
#  'antibiotic': 'Ciprofloxacin',
#  'prediction': 'Resistant', 
#  'probability': 0.8234}
```

---

##  Future Work & Limitations

### Current Constraints
1. **Regional Calibration**: Model trained on US-centric data
   - Captures global baseline patterns
   - Requires local calibration for region-specific prevalence
   
2. **Temporal Dynamics**: Dataset lacks time-indexed labels
   - Cannot model resistance evolution over time
   - Temporal awareness possible at system level (query pattern analysis)

3. **Feature Sparsity**: Binary organism–antibiotic classification
   - No patient demographics, sample source, or geographic stratification

### Planned Enhancements
- [ ] Local calibration with ICMR AMR surveillance data
- [ ] Ensemble modeling combining US baseline + WHO GLASS global datasets
- [ ] Temporal retraining pipeline when timestamped data becomes available
- [ ] Regional stratification for resistance ecology differences
- [ ] Uncertainty quantification using prediction intervals

### Contributing Data
If you have access to regional AMR datasets (especially India, Africa, Southeast Asia), please consider contributing to improve local calibration. Contact [@muadgijo](https://github.com/muadgijo).

---

##  Contributing

Contributions are welcome! Feel free to:
- Report bugs or issues
- Suggest new features
- Improve documentation
- Add new datasets

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Citation

If you use this work in your research or project, please cite:

```bibtex
@software{amrxml2026,
  author = {[Your Name]},
  title = {AMR-X: Machine Learning Model for Antimicrobial Resistance Prediction},
  year = {2026},
  url = {https://github.com/muadgijo/amrxml},
  version = {1.0.0}
}
```

---

##  Acknowledgments

- Data sourced from public antimicrobial resistance databases
- Built with Streamlit, XGBoost, pandas, and scikit-learn

---

##  Contact

Created by [@muadgijo](https://github.com/muadgijo)

**Live Demo**: [https://amrxml-93kfeuxsjop6g5g5yy4ydn.streamlit.app/](https://amrxml-93kfeuxsjop6g5g5yy4ydn.streamlit.app/)

# AMR-X: Antimicrobial Resistance Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://amrxml-93kfeuxsjop6g5g5yy4ydn.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning model to predict antimicrobial resistance (AMR) using XGBoost. Trained on real-world clinical data to help identify effective antibiotics for bacterial infections.

##  Live Demo

**Try it now:** [https://amrxml-93kfeuxsjop6g5g5yy4ydn.streamlit.app/](https://amrxml-93kfeuxsjop6g5g5yy4ydn.streamlit.app/)

Select an organism and antibiotic to predict resistance probability and see recommended treatment options.

---

##  Features

- **Resistance Prediction**: Predict if a bacteria will be resistant to a specific antibiotic
- **Treatment Ranking**: Get ranked list of most effective antibiotics for an organism
- **Fast Inference**: Lightweight XGBoost model with instant predictions
- **Real Clinical Data**: Trained on curated antimicrobial susceptibility datasets

---

##  How It Works

1. **Input**: Select organism (e.g., *E. coli*) and antibiotic (e.g., Ciprofloxacin)
2. **Model**: XGBoost binary classifier trained on organism-antibiotic pairs
3. **Output**: Probability of resistance + ranked treatment alternatives

**Model Details:**
- Algorithm: XGBoost (Gradient Boosting)
- Features: Organism code, Antibiotic code (encoded)
- Training Data: ~[insert number] clinical observations
- Performance: [insert accuracy/AUC if you have it]

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

The model is evaluated on held-out test data. Key metrics:

- **Accuracy**: [add your metric]
- **ROC-AUC**: [add your metric]
- **Precision/Recall**: See `outputs/` for plots

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

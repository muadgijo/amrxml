"""
AMR-X machine learning pipeline: data processing, model training, inference.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Config
RANDOM_SEED = 42
XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "eta": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
}
NUM_BOOST_ROUNDS = 500
EARLY_STOPPING_ROUNDS = 30

# Paths (simple dict instead of frozen dataclass)
PATHS = {
    "raw_dir": Path("data/raw"),
    "processed_dir": Path("data/processed"),
    "artifacts_dir": Path("models"),
    "outputs_dir": Path("outputs"),
    "raw_main": Path("data/raw/AMR_Dataset_Cleaned_Preprocessed.csv"),
    "raw_implied": Path("data/raw/microbiology_cultures_implied_susceptibility.csv"),
    "combined": Path("data/processed/data_combined_raw.csv"),
    "ready": Path("data/processed/AMR_X_ML_ready.csv"),
    "organism_map": Path("data/processed/organism_map.csv"),
    "antibiotic_map": Path("data/processed/antibiotic_map.csv"),
    "pair_counts": Path("data/processed/pair_counts.csv"),
    "model_path": Path("models/amr_xgb_model.json"),
    "metadata_path": Path("models/model_metadata.json"),
    "fig_confusion": Path("outputs/confusion_matrix.png"),
    "fig_roc": Path("outputs/roc_curve.png"),
    "fig_pr": Path("outputs/pr_curve.png"),
    "fig_feature_gain": Path("outputs/feature_importance_gain.png"),
}

IMPLIED_USECOLS = ("organism", "antibiotic", "susceptibility", "implied_susceptibility")


DATASET_URLS = {
    "AMR_Dataset_Cleaned_Preprocessed.csv": "https://huggingface.co/datasets/Muadgijo/amrx-datasets/resolve/main/AMR_Dataset_Cleaned_Preprocessed.csv",
    "microbiology_cultures_implied_susceptibility.csv": "https://huggingface.co/datasets/Muadgijo/amrx-datasets/resolve/main/microbiology_cultures_implied_susceptibility.csv",
}


def load_spectrum_rules():
    """Load biological filtering rules for antibiotics and organisms.
    
    Returns dict with spectrum and classification data, or None if files missing.
    """
    spectrum_path = Path("data/antibiotic_spectrum.json")
    organism_path = Path("data/organism_classification.json")
    
    try:
        if not spectrum_path.exists() or not organism_path.exists():
            return None
        
        with open(spectrum_path) as f:
            spectrum_data = json.load(f)
        
        with open(organism_path) as f:
            organism_data = json.load(f)
        
        return {
            "spectrum": spectrum_data,
            "organisms": organism_data,
        }
    except Exception:
        return None


def filter_by_spectrum(organism_name: str, antibiotic_list: list, spectrum_rules: dict | None) -> list:
    """Filter antibiotics based on organism type and spectrum of activity.
    
    Args:
        organism_name: Name of organism (e.g., "ESCHERICHIA COLI")
        antibiotic_list: List of antibiotic names to filter
        spectrum_rules: Rules dict from load_spectrum_rules() or None
    
    Returns:
        Filtered list of antibiotics appropriate for organism, or original list if filtering unavailable
    """
    if not spectrum_rules:
        return antibiotic_list
    
    try:
        org_upper = str(organism_name).upper().strip()
        organisms = spectrum_rules.get("organisms", {})
        spectrum = spectrum_rules.get("spectrum", {})
        
        # Determine organism type
        is_gram_neg = org_upper in organisms.get("gram_negative", [])
        is_gram_pos = org_upper in organisms.get("gram_positive", [])
        is_anaerobe = org_upper in organisms.get("anaerobe", [])
        
        # Get appropriate antibiotics
        appropriate = set()
        
        # Always include broad-spectrum
        appropriate.update(abx.upper() for abx in spectrum.get("broad_spectrum", []))
        
        if is_gram_neg:
            appropriate.update(abx.upper() for abx in spectrum.get("gram_negative_active", []))
        
        if is_gram_pos:
            appropriate.update(abx.upper() for abx in spectrum.get("gram_positive_only", []))
        
        if is_anaerobe:
            appropriate.update(abx.upper() for abx in spectrum.get("anaerobe_primary", []))
        
        # If organism type unknown, return all (fail-safe)
        if not (is_gram_neg or is_gram_pos or is_anaerobe):
            return antibiotic_list
        
        # Filter the list
        filtered = [abx for abx in antibiotic_list if abx.upper() in appropriate]
        
        # Return filtered list or original if filtering removed everything
        return filtered if filtered else antibiotic_list
        
    except Exception:
        # Fail gracefully - return original list
        return antibiotic_list


def ensure_directories():
    """Create required pipeline directories."""
    for path in [PATHS["raw_dir"], PATHS["processed_dir"], PATHS["artifacts_dir"], PATHS["outputs_dir"]]:
        path.mkdir(parents=True, exist_ok=True)


def _download_file(url, dest):
    """Download file from URL if it doesn't exist."""
    if dest.exists():
        print(f"✓ {dest.name} already exists")
        return
    print(f"Downloading {dest.name}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with dest.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print(f"✓ Saved to {dest}")


def download_datasets():
    """Download raw datasets from HuggingFace."""
    ensure_directories()
    for filename, url in DATASET_URLS.items():
        dest = PATHS["raw_dir"] / filename
        _download_file(url, dest)


def _clean_main_dataset(df):
    """Clean main AMR dataset: extract organism, antibiotic, susceptibility; map to binary labels."""
    if {"organism", "antibiotic", "susceptibility"}.issubset(df.columns):
        tidy = df[["organism", "antibiotic", "susceptibility"]].copy()
    else:
        org_col = df.columns[0]
        tidy = df.melt(id_vars=[org_col], var_name="antibiotic", value_name="susceptibility")
        tidy.rename(columns={org_col: "organism"}, inplace=True)

    tidy = tidy[tidy["susceptibility"].notna()].copy()
    tidy["sus"] = tidy["susceptibility"].astype(str).str.upper().str.strip().str[:1]
    tidy = tidy[tidy["sus"].isin(["S", "I", "R"])]
    tidy[["label_multi", "label_binary"]] = tidy["sus"].map({"S": (0, 0), "I": (1, 0), "R": (2, 1)}).apply(pd.Series)
    return tidy[["organism", "antibiotic", "sus", "label_multi", "label_binary"]]


def _clean_implied_dataset(path: Path | str):
    """Clean implied susceptibility dataset."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=250000, low_memory=True, usecols=IMPLIED_USECOLS):  # type: ignore[arg-type]
        chunk = chunk[chunk["susceptibility"].notna() | chunk["implied_susceptibility"].notna()].copy()
        chunk["final_sus"] = chunk["susceptibility"].fillna(chunk["implied_susceptibility"])
        chunk["sus"] = chunk["final_sus"].astype(str).str.upper().str[0]
        chunk["label_multi"] = chunk["sus"].map({"S": 0, "R": 2})
        chunk["label_binary"] = chunk["sus"].map({"S": 0, "R": 1})
        chunks.append(chunk[["organism", "antibiotic", "sus", "label_multi", "label_binary"]])
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["organism", "antibiotic", "sus", "label_multi", "label_binary"])


def combine_raw_datasets():
    """Load, clean, and combine main and implied datasets."""
    ensure_directories()
    if not PATHS["raw_main"].exists() or not PATHS["raw_implied"].exists():
        raise FileNotFoundError("Raw datasets missing. Run download_datasets() first.")
    
    print("Loading main dataset...")
    df_main = pd.read_csv(PATHS["raw_main"], low_memory=True)
    df_main = _clean_main_dataset(df_main)
    
    print("Loading implied dataset...")
    df_implied = _clean_implied_dataset(PATHS["raw_implied"])
    
    combined = pd.concat([df_main, df_implied], ignore_index=True)
    combined.to_csv(PATHS["combined"], index=False)
    print(f"✓ Combined dataset: {PATHS['combined']} ({len(combined):,} rows)")
    return combined


def _normalize_organism(name):
    """Normalize organism names."""
    if pd.isna(name):
        return None
    name = str(name).strip().upper()
    replacements = {
        "ESCHERICHIA COLI (CARBAPENEM RESISTANT)": "ESCHERICHIA COLI",
        "E. COLI": "ESCHERICHIA COLI",
        "E.COLI": "ESCHERICHIA COLI",
    }
    return replacements.get(name, name)


def normalize_and_encode():
    """Normalize organism/antibiotic names and encode as integers. Save maps."""
    ensure_directories()
    if not PATHS["combined"].exists():
        raise FileNotFoundError("Combined dataset missing. Run combine_raw_datasets() first.")
    
    df = pd.read_csv(PATHS["combined"])
    df["organism"] = df["organism"].apply(_normalize_organism)
    df["antibiotic"] = df["antibiotic"].astype(str).str.strip()
    df = df[df["organism"].notna()]
    df = df[df["organism"].str.contains(r"[A-Z]{3,}", na=False)]
    
    organism_cat = pd.Categorical(df["organism"])
    antibiotic_cat = pd.Categorical(df["antibiotic"])
    
    df["organism_code"] = organism_cat.codes.astype(int)
    df["antibiotic_code"] = antibiotic_cat.codes.astype(int)
    
    processed = df[["organism", "antibiotic", "sus", "label_multi", "label_binary", "organism_code", "antibiotic_code"]].copy()
    
    # Basic validation
    assert processed["organism_code"].isna().sum() == 0, "Found NaN organism codes"
    assert processed["antibiotic_code"].isna().sum() == 0, "Found NaN antibiotic codes"
    assert (processed["label_binary"].isin([0, 1])).all(), "Invalid labels found"
    
    PATHS["processed_dir"].mkdir(parents=True, exist_ok=True)
    processed.to_csv(PATHS["ready"], index=False)
    
    organism_map = pd.DataFrame({"organism": organism_cat.categories, "organism_code": range(len(organism_cat.categories))})
    antibiotic_map = pd.DataFrame({"antibiotic": antibiotic_cat.categories, "antibiotic_code": range(len(antibiotic_cat.categories))})
    
    organism_map.to_csv(PATHS["organism_map"], index=False)
    antibiotic_map.to_csv(PATHS["antibiotic_map"], index=False)
    
    pair_counts = processed.groupby(["organism_code", "antibiotic_code"]).size().reset_index(name="count")
    pair_counts.to_csv(PATHS["pair_counts"], index=False)
    
    print(f"✓ Processed dataset: {PATHS['ready']} ({len(processed):,} rows)")
    return processed


def _train_val_test_split(X, y, seed=RANDOM_SEED):
    """Stratified train-val-test split (64-16-20%)."""
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, stratify=y_temp, random_state=seed)
    return X_train, X_val, X_test, y_train, y_val, y_test


def _compute_threshold(y_true, y_prob):
    """Compute optimal threshold using Youden's J statistic (TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr - fpr)
    return float(thresholds[optimal_idx])


def train_model():
    """Train XGBoost on processed data. Save model and metrics."""
    ensure_directories()
    if not PATHS["ready"].exists():
        raise FileNotFoundError("Processed dataset missing. Run normalize_and_encode() first.")
    
    df = pd.read_csv(PATHS["ready"])
    X = df[["organism_code", "antibiotic_code"]].to_numpy()
    y = df["label_binary"].to_numpy(dtype=np.int32)
    
    X_train, X_val, X_test, y_train, y_val, y_test = _train_val_test_split(X, y)
    
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=["organism_code", "antibiotic_code"])
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=["organism_code", "antibiotic_code"])
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=["organism_code", "antibiotic_code"])
    
    params = {
        **XGBOOST_PARAMS,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "seed": RANDOM_SEED,
    }
    
    print("Training XGBoost...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )
    
    y_val_prob = model.predict(dval)
    threshold = _compute_threshold(y_val, y_val_prob)
    
    y_test_prob = model.predict(dtest)
    y_test_pred = (y_test_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_test, y_test_pred)
    roc_auc = roc_auc_score(y_test, y_test_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-9)
    specificity = tn / (tn + fp + 1e-9)
    
    PATHS["artifacts_dir"].mkdir(parents=True, exist_ok=True)
    model.save_model(PATHS["model_path"])
    
    metadata = {
        "optimal_threshold": float(threshold),
        "test_metrics": {
            "accuracy": float(acc),
            "roc_auc": float(roc_auc),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        },
        "n_samples": int(len(df)),
        "n_test": int(len(X_test)),
    }
    
    with open(PATHS["metadata_path"], "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Model saved: {PATHS['model_path']}")
    print(f"  Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}, Threshold: {threshold:.3f}")
    return metadata


def run_baselines():
    """Run baseline models: majority class, logistic regression."""
    if not PATHS["ready"].exists():
        raise FileNotFoundError("Processed dataset missing. Run normalize_and_encode() first.")
    
    df = pd.read_csv(PATHS["ready"])
    X = df[["organism_code", "antibiotic_code"]].to_numpy()
    y = df["label_binary"].to_numpy(dtype=np.int32)
    X_train, _, X_test, y_train, _, y_test = _train_val_test_split(X, y)
    
    # Majority class baseline
    majority_label = int(pd.Series(y_train).mode()[0])
    y_pred_maj = np.full_like(y_test, majority_label)
    acc_maj = accuracy_score(y_test, y_pred_maj)
    auc_maj = roc_auc_score(y_test, y_pred_maj) if len(np.unique(y_test)) > 1 else float("nan")
    
    # Logistic regression baseline
    lr = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight="balanced")
    lr.fit(X_train, y_train)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_pred_lr = (y_prob_lr >= 0.5).astype(int)
    acc_lr = accuracy_score(y_test, y_pred_lr)
    auc_lr = roc_auc_score(y_test, y_prob_lr)
    
    results = pd.DataFrame({
        "Model": ["Majority Class", "Logistic Regression"],
        "Accuracy": [acc_maj, acc_lr],
        "ROC-AUC": [auc_maj, auc_lr],
    })
    print("\nBaseline Results:")
    print(results.to_string(index=False))
    return results


def generate_visualizations():
    """Generate confusion matrix, ROC, PR, and feature importance plots."""
    if not PATHS["ready"].exists() or not PATHS["model_path"].exists():
        raise FileNotFoundError("Run train_model() first.")
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    df = pd.read_csv(PATHS["ready"])
    X = df[["organism_code", "antibiotic_code"]].to_numpy()
    y = df["label_binary"].to_numpy(dtype=np.int32)
    _, _, X_test, _, _, y_test = _train_val_test_split(X, y)
    
    model = xgb.Booster()
    model.load_model(PATHS["model_path"])
    
    with open(PATHS["metadata_path"]) as f:
        metadata = json.load(f)
    threshold = metadata.get("optimal_threshold", 0.5)
    
    dtest = xgb.DMatrix(X_test, feature_names=["organism_code", "antibiotic_code"])
    y_test_prob = model.predict(dtest)
    y_test_pred = (y_test_prob >= threshold).astype(int)
    
    PATHS["outputs_dir"].mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Susceptible", "Resistant"], yticklabels=["Susceptible", "Resistant"])
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(PATHS["fig_confusion"], dpi=100, bbox_inches="tight")
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    auc = roc_auc_score(y_test, y_test_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PATHS["fig_roc"], dpi=100, bbox_inches="tight")
    plt.close()
    
    # PR curve
    prec, rec, _ = precision_recall_curve(y_test, y_test_prob)
    ap = average_precision_score(y_test, y_test_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="purple", lw=2, label=f"PR (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PATHS["fig_pr"], dpi=100, bbox_inches="tight")
    plt.close()
    
    # Feature importance
    importance = model.get_score(importance_type="gain")
    if importance:
        items = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        names, gains = zip(*items)
        plt.figure(figsize=(6, 4))
        plt.barh(names, gains, color="teal")
        plt.xlabel("Gain")
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig(PATHS["fig_feature_gain"], dpi=100, bbox_inches="tight")
        plt.close()
    
    print("✓ Plots saved to outputs/")


def load_inference_assets():
    """Load trained model and lookup maps for inference."""
    if not (PATHS["model_path"].exists() and PATHS["organism_map"].exists() and PATHS["antibiotic_map"].exists()):
        raise FileNotFoundError("Model or maps missing. Run train_model() first.")
    
    model = xgb.Booster()
    model.load_model(PATHS["model_path"])
    
    with open(PATHS["metadata_path"]) as f:
        metadata = json.load(f)
    
    org_map = pd.read_csv(PATHS["organism_map"])
    abx_map = pd.read_csv(PATHS["antibiotic_map"])
    
    org_lookup = {str(row["organism"]).lower(): (int(row["organism_code"]), str(row["organism"])) for _, row in org_map.iterrows()}
    abx_lookup = {str(row["antibiotic"]).lower(): (int(row["antibiotic_code"]), str(row["antibiotic"])) for _, row in abx_map.iterrows()}
    
    return {
        "model": model,
        "metadata": metadata,
        "org_lookup": org_lookup,
        "abx_lookup": abx_lookup,
    }


def predict_resistance(organism, antibiotic, assets):
    """Predict resistance probability for organism-antibiotic pair."""
    org_lookup = assets["org_lookup"]
    abx_lookup = assets["abx_lookup"]
    model = assets["model"]
    
    org_key = str(organism).lower().strip()
    abx_key = str(antibiotic).lower().strip()
    
    if org_key not in org_lookup:
        return {"error": f"Organism '{organism}' not found"}
    if abx_key not in abx_lookup:
        return {"error": f"Antibiotic '{antibiotic}' not found"}
    
    org_code, org_name = org_lookup[org_key]
    abx_code, abx_name = abx_lookup[abx_key]
    
    X_input = np.array([[org_code, abx_code]], dtype=np.float32)
    dmat = xgb.DMatrix(X_input, feature_names=["organism_code", "antibiotic_code"])
    prob = float(model.predict(dmat)[0])
    pred = "Resistant" if prob >= 0.5 else "Susceptible"
    
    return {
        "organism": org_name,
        "antibiotic": abx_name,
        "probability": round(prob, 4),
        "prediction": pred,
    }


def rank_antibiotics(organism, assets, top_n=10, apply_filtering=True):
    """Rank antibiotics by effectiveness for organism.
    
    Args:
        organism: Organism name
        assets: Dict with model, lookups
        top_n: Number of top antibiotics to return
        apply_filtering: Whether to apply biological spectrum filtering
    
    Returns:
        DataFrame with ranked antibiotics or error dict
    """
    org_lookup = assets["org_lookup"]
    abx_lookup = assets["abx_lookup"]
    model = assets["model"]
    
    org_key = str(organism).lower().strip()
    if org_key not in org_lookup:
        return {"error": f"Organism '{organism}' not found"}
    
    org_code, org_name = org_lookup[org_key]
    
    # Load spectrum rules if filtering enabled
    spectrum_rules = load_spectrum_rules() if apply_filtering else None
    
    # Get all antibiotic names
    all_antibiotics = [abx_name for _, (_, abx_name) in abx_lookup.items()]
    
    # Apply biological filtering
    if apply_filtering and spectrum_rules:
        filtered_names = filter_by_spectrum(org_name, all_antibiotics, spectrum_rules)
    else:
        filtered_names = all_antibiotics
    
    rows = []
    for abx_lower, (abx_code, abx_name) in abx_lookup.items():
        # Skip if filtered out
        if abx_name not in filtered_names:
            continue
            
        X_input = np.array([[org_code, abx_code]], dtype=np.float32)
        dmat = xgb.DMatrix(X_input, feature_names=["organism_code", "antibiotic_code"])
        prob_resistant = float(model.predict(dmat)[0])
        rows.append({
            "antibiotic": abx_name,
            "prob_effective": round(1 - prob_resistant, 4),
            "prob_resistant": round(prob_resistant, 4),
        })
    
    df = pd.DataFrame(rows).sort_values("prob_effective", ascending=False)
    return df.head(top_n)


def train_ablation_model(X_train, y_train, X_test, y_test, feature_names):
    """Train XGBoost on subset of features (ablation study)."""
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)
    
    params = {
        **XGBOOST_PARAMS,
        "scale_pos_weight": (y_train == 0).sum() / max((y_train == 1).sum(), 1),
        "seed": RANDOM_SEED,
    }
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(dtrain, "train")],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=50,
    )
    
    y_prob = model.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)
    
    acc = float(accuracy_score(y_test, y_pred))
    auc = float(roc_auc_score(y_test, y_prob))
    
    return {
        "accuracy": acc,
        "roc_auc": auc,
        "features": feature_names,
    }


def full_preprocess():
    """Download, combine, and encode data in one call."""
    download_datasets()
    combine_raw_datasets()
    return normalize_and_encode()


def full_train():
    """Ensure data ready then train model."""
    if not PATHS["ready"].exists():
        full_preprocess()
    return train_model()

if __name__ == "__main__":
    ensure_directories()
    print("Pipeline ready. Call: full_preprocess() -> full_train() -> generate_visualizations()")
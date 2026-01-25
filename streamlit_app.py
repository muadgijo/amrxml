"""Enhanced Streamlit UI for AMR-X predictions with analytics, insights, and clinical guidance."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st  # type: ignore[import-not-found]

from scripts.pipeline import (
    load_inference_assets,
    predict_resistance,
    rank_antibiotics,
    load_spectrum_rules,
)

# Page configuration
st.set_page_config(
    page_title="AMR-X Resistance Predictor", 
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-themed styling
st.markdown("""
<style>
    /* Minimal, accessible palette */
    :root {
        --primary: #0D6EFD; /* blue */
        --success: #198754; /* green */
        --warning: #FFC107; /* amber */
        --danger:  #DC3545; /* red */
    }
    /* Let Streamlit manage text colors for light/dark themes */
    .stButton>button { background: var(--primary); color: #fff; border-radius: 6px; border: none; }
    .stButton>button:hover { filter: brightness(0.95); }
    .disclaimer-box { background: #FFF3CD; border-left: 4px solid #FF9800; padding: 12px; border-radius: 4px; color: #333; }
    .status-resistant { color: var(--danger); font-weight: 600; }
    .status-susceptible { color: var(--success); font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_assets():
    """Load inference assets with caching."""
    return load_inference_assets()


@st.cache_data(show_spinner=False)
def load_model_metadata():
    """Load model metadata for analytics display."""
    metadata_path = Path("models/model_metadata.json")
    try:
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return None
    except Exception:
        return None


def format_probability(prob: float) -> str:
    """Format probability as percentage."""
    return f"{prob * 100:.1f}%"


def get_confidence_level(prob: float) -> tuple[str, str]:
    """Get confidence level and color based on probability.
    
    Returns: (level_text, color)
    """
    if prob < 0.3 or prob > 0.7:
        return "High Confidence", "#16a34a"
    elif 0.3 <= prob < 0.4 or 0.6 < prob <= 0.7:
        return "Moderate Confidence", "#ea580c"
    else:
        return "Low Confidence", "#dc2626"


def get_risk_level(prob: float) -> tuple[str, str, str]:
    """Get risk stratification for resistance probability.
    
    Returns: (level, emoji, description)
    """
    if prob < 0.3:
        return "Low Risk", "🟢", "Low probability of resistance - likely effective"
    elif prob < 0.5:
        return "Moderate-Low Risk", "🟡", "Some resistance possible - consider with caution"
    elif prob < 0.7:
        return "Moderate-High Risk", "🟠", "Significant resistance risk - alternative recommended"
    else:
        return "High Risk", "🔴", "High probability of resistance - avoid if possible"


def render_disclaimer():
    """Render medical disclaimer banner."""
    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Medical Disclaimer</strong>
        <p>This tool is for educational and research use only and does not replace professional medical advice or clinical judgment.</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with navigation and info."""
    with st.sidebar:
        st.title("AMR-X")
        st.caption("Antimicrobial Resistance Predictor")

        st.markdown("---")
        st.subheader("About")

        # Load metadata for dynamic stats
        metadata = load_model_metadata()

        if metadata:
            test_metrics = metadata.get("test_metrics", {})
            n_samples = metadata.get("n_samples", 0)
            accuracy = test_metrics.get("accuracy", 0) * 100
            st.markdown(
                f"""
                **AMR-X** predicts antimicrobial resistance and ranks treatment options.

                - Model: XGBoost
                - Training Samples: **{n_samples:,}+**
                - Accuracy: **{accuracy:.1f}%**
                """
            )
        else:
            st.markdown(
                """
                **AMR-X** predicts antimicrobial resistance and ranks treatment options.

                - Model: XGBoost
                - Research-grade, inference-only
                """
            )

        st.markdown("---")
        st.subheader("Resources")
        st.markdown(
            """
            - [WHO AMR Resources](https://www.who.int/health-topics/antimicrobial-resistance)
            - [CDC AMR Info](https://www.cdc.gov/drugresistance/)
            - [GitHub Repository](https://github.com/muadgijo/amrxml)
            """
        )

        st.markdown("---")
        st.caption("© 2026 AMR-X · MIT License")


def render_analytics_tab(metadata):
    """Render analytics and model performance tab."""
    st.header("Model Analytics & Performance")
    
    if metadata is None:
        st.warning("Model metadata not available. Analytics cannot be displayed.")
        return
    
    # Model Performance Metrics
    st.subheader("Model Performance Metrics")
    
    test_metrics = metadata.get("test_metrics", {})
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy = test_metrics.get("accuracy", 0) * 100
        st.metric("Accuracy", f"{accuracy:.2f}%", help="Overall prediction accuracy on test set")
    
    with col2:
        roc_auc = test_metrics.get("roc_auc", 0)
        st.metric("ROC-AUC", f"{roc_auc:.4f}", help="Area Under ROC Curve - model discrimination ability")
    
    with col3:
        sensitivity = test_metrics.get("sensitivity", 0) * 100
        st.metric("Sensitivity", f"{sensitivity:.2f}%", help="True Positive Rate - correctly identified resistant cases")
    
    with col4:
        specificity = test_metrics.get("specificity", 0) * 100
        st.metric("Specificity", f"{specificity:.2f}%", help="True Negative Rate - correctly identified susceptible cases")
    
    # Dataset Statistics
    st.markdown("---")
    st.subheader("Dataset Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_samples = metadata.get("n_samples", 0)
        st.metric("Total Training Samples", f"{n_samples:,}", help="Total organism-antibiotic pairs used for training")
    
    with col2:
        n_test = metadata.get("n_test", 0)
        st.metric("Test Samples", f"{n_test:,}", help="Samples reserved for model evaluation")
    
    with col3:
        threshold = metadata.get("optimal_threshold", 0.5)
        st.metric("Optimal Threshold", f"{threshold:.3f}", help="Decision threshold optimized using Youden's J statistic")
    
    # Model Information
    st.markdown("---")
    st.subheader("Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Algorithm:** XGBoost (Gradient Boosting)
        
        **Features:**
        - Organism Code (encoded)
        - Antibiotic Code (encoded)
        
        **Objective:** Binary Classification (Resistant/Susceptible)
        """)
    
    with col2:
        st.markdown("""
        **Key Parameters:**
        - Max Depth: 6
        - Learning Rate: 0.1
        - Subsample: 0.8
        - Tree Method: Histogram-based
        
        **Training:** 500 rounds with early stopping
        """)
    
    # Visualizations (if available)
    st.markdown("---")
    st.subheader("Performance Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    roc_path = Path("outputs/roc_curve.png")
    cm_path = Path("outputs/confusion_matrix.png")
    
    with viz_col1:
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curve", use_container_width=True)
        else:
            st.info("ROC curve visualization not available")
    
    with viz_col2:
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix", use_container_width=True)
        else:
            st.info("Confusion matrix visualization not available")
    
    # Interpretation Guide
    with st.expander("How to interpret these metrics"):
        st.markdown("""
        **Accuracy**: Percentage of correct predictions (both resistant and susceptible).
        
        **ROC-AUC**: Measures the model's ability to distinguish between resistant and susceptible cases. 
        Values closer to 1.0 indicate better discrimination. 0.5 would be random guessing.
        
        **Sensitivity (Recall)**: Proportion of actual resistant cases correctly identified. 
        High sensitivity means the model is good at catching resistant bacteria.
        
        **Specificity**: Proportion of actual susceptible cases correctly identified. 
        High specificity means the model rarely misidentifies susceptible bacteria as resistant.
        
        **Optimal Threshold**: The probability cutoff used to classify predictions. Optimized to 
        balance sensitivity and specificity using Youden's J statistic.
        """)


def render_insights_tab():
    """Render clinical insights and educational content tab."""
    st.header("🩺 Clinical Insights & Education")
    
    # Risk Interpretation
    st.subheader("📊 Understanding Resistance Risk")
    st.markdown("""
    The model provides a **probability score** (0-100%) indicating the likelihood of resistance:
    
    - 🟢 **0-30%**: Low resistance risk - antibiotic likely effective
    - 🟡 **30-50%**: Moderate-low risk - use with clinical judgment
    - 🟠 **50-70%**: Moderate-high risk - consider alternatives
    - 🔴 **70-100%**: High risk - avoid if possible, seek alternatives
    
    **Note:** These are predictions based on historical data. Actual susceptibility should be confirmed 
    with culture and sensitivity testing when possible.
    """)
    
    st.markdown("---")
    
    # Spectrum of Activity
    st.subheader("🎯 Antibiotic Spectrum Filtering")
    st.markdown("""
    The rankings apply **biological filtering** based on spectrum of activity:
    
    - **Gram-negative bacteria** → Filtered to show antibiotics active against Gram-negative organisms
    - **Gram-positive bacteria** → Includes Gram-positive-specific and broad-spectrum agents
    - **Anaerobes** → Filtered to anaerobe-active antibiotics
    
    This ensures recommendations are biologically appropriate, not just statistically likely.
    """)
    
    st.markdown("---")
    
    # Educational Content
    st.subheader("📚 Learn More")
    
    with st.expander("🦠 What is Antimicrobial Resistance (AMR)?"):
        st.markdown("""
        **Antimicrobial Resistance** occurs when bacteria evolve to resist the effects of antibiotics 
        that once killed them or stopped their growth.
        
        **Why it matters:**
        - Makes infections harder to treat
        - Increases healthcare costs
        - Leads to longer hospital stays
        - Can result in treatment failure
        
        **Key causes:**
        - Overuse and misuse of antibiotics
        - Poor infection control
        - Inadequate sanitation
        - Lack of new antibiotic development
        
        **Learn more:** [WHO AMR Factsheet](https://www.who.int/news-room/fact-sheets/detail/antimicrobial-resistance)
        """)
    
    with st.expander("🔬 How Does This Model Work?"):
        st.markdown("""
        **AMR-X** is a machine learning model that:
        
        1. **Learns from historical data**: Trained on 1.4+ million organism-antibiotic susceptibility test results
        2. **Identifies patterns**: Uses XGBoost algorithm to find resistance patterns
        3. **Makes predictions**: Estimates resistance probability for any organism-antibiotic pair
        4. **Ranks alternatives**: Suggests most effective treatment options
        
        **Input features:**
        - Organism species (encoded as numeric code)
        - Antibiotic agent (encoded as numeric code)
        
        **Output:**
        - Probability of resistance (0-100%)
        - Binary classification (Resistant/Susceptible)
        - Ranked treatment alternatives
        
        **Limitations:**
        - Does not account for patient-specific factors
        - Based on population-level data
        - Cannot replace laboratory testing
        - May not reflect local resistance patterns
        """)
    
    with st.expander("⚕️ When to Consult Healthcare Providers?"):
        st.markdown("""
        **Always consult qualified healthcare providers** for:
        
        ✅ **Clinical diagnosis** of infections
        
        ✅ **Treatment decisions** and antibiotic selection
        
        ✅ **Interpreting** culture and sensitivity results
        
        ✅ **Patient-specific** considerations (allergies, drug interactions, pregnancy, etc.)
        
        ✅ **Monitoring** treatment response and adverse effects
        
        ⚠️ **This tool does NOT:**
        - Diagnose infections
        - Prescribe medications
        - Replace clinical judgment
        - Account for individual patient factors
        - Substitute for culture and sensitivity testing
        """)
    
    with st.expander("🌍 Antimicrobial Stewardship"):
        st.markdown("""
        **Antimicrobial stewardship** is a coordinated effort to use antibiotics responsibly:
        
        **For Healthcare Providers:**
        - Prescribe antibiotics only when needed
        - Choose the right drug, dose, and duration
        - Use culture-guided therapy when possible
        - Follow local antibiograms and guidelines
        
        **For Patients:**
        - Take antibiotics exactly as prescribed
        - Complete the full course
        - Never share or save antibiotics
        - Prevent infections through hygiene and vaccination
        
        **Resources:**
        - [CDC: Antibiotic Prescribing and Use](https://www.cdc.gov/antibiotic-use/)
        - [WHO: Antimicrobial Stewardship](https://www.who.int/teams/integrated-health-services/infection-prevention-control/antimicrobial-stewardship)
        """)


def render_about_tab():
    """Render about/help section."""
    st.header("ℹ️ About AMR-X")
    
    st.markdown("""
    ### 🔬 What is AMR-X?
    
    **AMR-X** (Antimicrobial Resistance - eXplorer) is a machine learning-powered tool for predicting 
    antimicrobial resistance patterns and identifying effective treatment options for bacterial infections.
    
    ### 🎯 Key Features
    
    - **Resistance Prediction**: Estimate probability of resistance for any organism-antibiotic combination
    - **Treatment Rankings**: Get biologically-filtered, ranked lists of effective antibiotics
    - **Clinical Insights**: Understand risk levels and treatment implications
    - **Comprehensive Analytics**: View model performance metrics and dataset statistics
    - **Educational Content**: Learn about AMR, model methodology, and antibiotic stewardship
    
    ### 📊 Data Sources
    
    The model is trained on curated antimicrobial susceptibility datasets:
    - **1.4+ million** organism-antibiotic test results
    - Real-world clinical data from multiple sources
    - Standardized susceptibility interpretations (S/I/R)
    
    **Data available on:** [Hugging Face - Muadgijo/amrx-datasets](https://huggingface.co/datasets/Muadgijo/amrx-datasets)
    
    ### 🤖 Model Details
    
    - **Algorithm**: XGBoost (Gradient Boosted Trees)
    - **Task**: Binary classification (Resistant vs Susceptible)
    - **Features**: Organism code, Antibiotic code (encoded)
    - **Performance**: 68% accuracy, 0.81 ROC-AUC
    - **Validation**: Stratified train-validation-test split
    - **Optimization**: Youden's J statistic for threshold selection
    
    ### 🛠️ How to Use
    
    1. **Select Organism**: Choose the bacterial species from the dropdown
    2. **Select Antibiotic**: Choose the antibiotic agent to test
    3. **View Prediction**: See resistance probability and classification
    4. **Check Rankings**: Review biologically-appropriate treatment alternatives
    5. **Explore Analytics**: Dive into model performance and statistics
    6. **Learn More**: Read clinical insights and educational content
    
    ### ⚠️ Important Limitations
    
    - **Not for clinical use**: This is an educational/research tool
    - **Population-based**: Does not account for individual patient factors
    - **Requires validation**: Culture and sensitivity testing recommended
    - **Local patterns**: May not reflect resistance in your specific area
    - **Biological filtering**: Spectrum-based filtering applied but not comprehensive
    
    ### 📖 Citation
    
    If you use this tool in your research or educational materials, please cite:
    
    ```
    AMR-X: Machine Learning Model for Antimicrobial Resistance Prediction
    https://github.com/muadgijo/amrxml
    ```
    
    ### 🤝 Contributing
    
    This is an open-source project. Contributions welcome!
    - Report issues on [GitHub](https://github.com/muadgijo/amrxml/issues)
    - Submit pull requests
    - Share feedback and suggestions
    
    ### 📝 License
    
    MIT License - See repository for details
    
    ### 👨‍💻 Developer
    
    Created by [@muadgijo](https://github.com/muadgijo)
    """)


def main():
    """Main Streamlit application."""
    # Render sidebar
    render_sidebar()
    
    # Main header
    st.title("AMR-X: Antimicrobial Resistance Predictor")
    st.caption("Predict resistance and review treatment rankings.")
    
    # Disclaimer banner
    # Concise disclaimer
    render_disclaimer()
    
    # Load assets
    assets = None
    try:
        with st.spinner("Loading model and data..."):
            assets = get_assets()
    except FileNotFoundError:
        st.error("❌ Model or lookup files missing. Run `python run.py` once to train and save assets.")
        st.stop()
    
    if assets is None:
        st.error("❌ Failed to load model assets.")
        st.stop()
    
    # Load metadata for analytics
    metadata = load_model_metadata()
    
    # Get options
    org_options = sorted({name for _, name in assets["org_lookup"].values()})
    abx_options = sorted({name for _, name in assets["abx_lookup"].values()})
    
    if not org_options or not abx_options:
        st.error("Lookup tables are empty. Re-run preprocessing/training.")
        st.stop()
    
    # Create tabs
    tab1, tab2 = st.tabs(["Prediction", "Analytics"])
    
    # Tab 1: Prediction
    with tab1:
        st.header("Resistance Prediction & Treatment Rankings")
        
        # Prediction form
        st.subheader("Test Resistance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            organism = st.selectbox(
                "Select Organism", 
                org_options,
                help="Choose the bacterial species to test"
            )
        
        with col2:
            antibiotic = st.selectbox(
                "Select Antibiotic", 
                abx_options,
                help="Choose the antibiotic agent"
            )
        
        col1, col2 = st.columns([1, 3])
        with col1:
            predict_btn = st.button("Predict Resistance", type="primary", use_container_width=True)
        
        # Prediction results
        if predict_btn:
            with st.spinner("Analyzing..."):
                result = predict_resistance(organism, antibiotic, assets)
            
            if "error" in result:
                st.warning(result["error"])
            else:
                prob = float(result.get("probability", 0.0))
                pred = result.get("prediction", "Unknown")
                
                st.markdown("---")
                st.subheader("Prediction Results")
                
                # Display prediction with color coding
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Organism", organism)
                
                with col2:
                    st.metric("Antibiotic", antibiotic)
                
                with col3:
                    st.metric("Prediction", pred)
                
                # Probability and confidence
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Resistance Probability", 
                        format_probability(prob),
                        help="Likelihood that the organism is resistant to this antibiotic"
                    )
                    st.progress(min(max(prob, 0.0), 1.0))
                
                with col2:
                    confidence, conf_color = get_confidence_level(prob)
                    st.metric(
                        "Prediction Confidence",
                        confidence,
                        help="Confidence level based on probability distribution"
                    )
                
                # Risk stratification
                st.markdown("---")
                # Keep results minimal without extra badges/emojis
                
                # Clinical interpretation
                with st.expander("📖 Clinical Interpretation"):
                    if prob < 0.3:
                        st.success(f"""
                        **Low resistance probability ({format_probability(prob)})**
                        
                        This organism-antibiotic combination shows low predicted resistance. 
                        The antibiotic is likely to be effective, though culture confirmation is recommended.
                        """)
                    elif prob < 0.5:
                        st.warning(f"""
                        **Moderate-low resistance probability ({format_probability(prob)})**
                        
                        There is some predicted resistance. Consider this antibiotic with caution and 
                        review patient factors, local resistance patterns, and alternative options.
                        """)
                    elif prob < 0.7:
                        st.warning(f"""
                        **Moderate-high resistance probability ({format_probability(prob)})**
                        
                        Significant resistance is predicted. Alternative antibiotics are recommended. 
                        If used, close monitoring and susceptibility testing are essential.
                        """)
                    else:
                        st.error(f"""
                        **High resistance probability ({format_probability(prob)})**
                        
                        This combination shows high predicted resistance. Avoid this antibiotic if possible 
                        and select alternatives from the rankings below. Confirm with culture and sensitivity.
                        """)
        
        # Treatment Rankings
        st.markdown("---")
        st.subheader("Recommended Treatment Options")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            rank_organism = st.selectbox(
                "Organism for Rankings",
                org_options,
                index=org_options.index(organism) if organism in org_options else 0,
                key="rank_org",
                help="Select organism to see treatment rankings"
            )
        
        with col2:
            top_n = st.slider(
                "Show Top N",
                min_value=5,
                max_value=min(30, len(abx_options)),
                value=15,
                help="Number of antibiotics to display"
            )
        
        with st.spinner("Ranking antibiotics..."):
            rank_df = rank_antibiotics(rank_organism, assets, top_n=top_n, apply_filtering=True)
        
        if isinstance(rank_df, dict) and "error" in rank_df:
            st.warning(rank_df["error"])
        elif isinstance(rank_df, pd.DataFrame) and not rank_df.empty:
            # Format the dataframe
            display_df = rank_df.copy()
            display_df["Effectiveness"] = display_df["prob_effective"].apply(lambda x: format_probability(x))
            display_df["Resistance"] = display_df["prob_resistant"].apply(lambda x: format_probability(x))
            
            # Rename and reorder columns
            display_df = display_df[["antibiotic", "Effectiveness", "Resistance"]]
            display_df.columns = ["Antibiotic", "Effectiveness", "Resistance"]
            display_df.index = range(1, len(display_df) + 1)
            
            st.dataframe(display_df, use_container_width=True)
        else:
            st.warning("No ranking data available")
    
    # Tab 2: Analytics
    with tab2:
        render_analytics_tab(metadata)
    
    # Additional tabs removed for a cleaner layout


if __name__ == "__main__":
    main()

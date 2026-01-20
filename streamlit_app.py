"""Simple Streamlit UI for AMR-X predictions."""

from __future__ import annotations

import pandas as pd
import streamlit as st  # type: ignore[import-not-found]

from scripts.pipeline import load_inference_assets, predict_resistance, rank_antibiotics

st.set_page_config(page_title="AMR-X Resistance Predictor", layout="centered")


@st.cache_resource(show_spinner=False)
def get_assets():
    return load_inference_assets()


def format_probability(prob: float) -> str:
    return f"{prob:.3f}"


def main():
    st.title("AMR-X Resistance Predictor")
    st.caption("Pick an organism and antibiotic to estimate resistance, then see the top options.")

    assets = None
    try:
        assets = get_assets()
    except FileNotFoundError:
        st.error("Model or lookup files missing. Run `python run.py` once to train and save assets.")
        st.stop()

    assert assets is not None

    org_options = sorted({name for _, name in assets["org_lookup"].values()})
    abx_options = sorted({name for _, name in assets["abx_lookup"].values()})

    if not org_options or not abx_options:
        st.error("Lookup tables are empty. Re-run preprocessing/training.")
        st.stop()

    with st.form("predict_form", clear_on_submit=False):
        organism = st.selectbox("Organism", org_options)
        antibiotic = st.selectbox("Antibiotic", abx_options)
        top_n = st.slider("Top antibiotics to rank", min_value=3, max_value=min(20, len(abx_options)), value=10)
        submitted = st.form_submit_button("Predict resistance")

    if submitted:
        result = predict_resistance(organism, antibiotic, assets)
        if "error" in result:
            st.warning(result["error"])
        else:
            prob = float(result.get("probability", 0.0))
            st.subheader("Prediction")
            st.metric("Predicted label", result["prediction"], delta=None)
            st.write(f"Probability of resistance: {format_probability(prob)}")
            st.progress(min(max(prob, 0.0), 1.0))

    st.subheader("Top antibiotics for this organism")
    rank_df = rank_antibiotics(organism if submitted else org_options[0], assets, top_n=top_n)
    if isinstance(rank_df, dict) and "error" in rank_df:
        st.warning(rank_df["error"])
    elif isinstance(rank_df, pd.DataFrame):
        display_df = rank_df.rename(columns={"prob_effective": "Prob effective", "prob_resistant": "Prob resistant"})
        st.dataframe(display_df.reset_index(drop=True), use_container_width=True)


if __name__ == "__main__":
    main()

"""
app.py
-------
Streamlit web application for the
"Explainable AI Framework for Medical Diagnosis" project.

Workflow implemented on this single page:
    Patient Input -> AI Diagnosis -> Explainable Prediction

Run with (from the project folder, inside the virtual environment):
    streamlit run app.py

See README.md for full setup instructions on Windows / VS Code.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import shap

from explainability import (
    build_shap_explainer, _class1_values,
    generate_natural_language_explanation,
)

# ---------------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Explainable AI Framework for Medical Diagnosis",
    page_icon=None,
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

CLASS_NAMES = ["Benign", "Malignant"]

FEATURE_GROUPS = {
    "Mean Values": [
        "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
        "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
        "mean fractal dimension",
    ],
    "Standard Error Values": [
        "radius error", "texture error", "perimeter error", "area error", "smoothness error",
        "compactness error", "concavity error", "concave points error", "symmetry error",
        "fractal dimension error",
    ],
    "Worst (Largest) Values": [
        "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
        "worst compactness", "worst concavity", "worst concave points", "worst symmetry",
        "worst fractal dimension",
    ],
}


# ---------------------------------------------------------------------------
# Cached loading of model artifacts (runs once per session, not on every click)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained model and explainer...")
def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    X_train_raw = pd.read_csv(os.path.join(MODELS_DIR, "X_train_background.csv"))
    X_train_scaled = scaler.transform(X_train_raw)
    explainer = build_shap_explainer(model, pd.DataFrame(X_train_scaled, columns=feature_names))

    metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)

    sample_patients = None
    sample_path = os.path.join(MODELS_DIR, "X_test_sample.csv")
    if os.path.exists(sample_path):
        sample_patients = pd.read_csv(sample_path)

    return model, scaler, feature_names, X_train_raw, explainer, metadata, sample_patients


try:
    model, scaler, feature_names, X_train_raw, explainer, metadata, sample_patients = load_artifacts()
except FileNotFoundError:
    st.error(
        "Trained model files were not found in the models/ folder.\n\n"
        "Please run the following commands first, from the project folder:\n\n"
        "    python train_model.py\n"
        "    python generate_explanations.py\n\n"
        "Then restart this app."
    )
    st.stop()

feature_stats = X_train_raw.describe()  # used for input defaults / ranges


# ---------------------------------------------------------------------------
# Header + mandatory academic/research disclaimer
# ---------------------------------------------------------------------------
st.title("Explainable AI Framework for Medical Diagnosis")
st.caption(
    "Breast Cancer Diagnosis Prediction with SHAP-Based Explainability "
    f"(Model: {metadata.get('best_model_name', 'N/A')})"
)

st.warning(
    "**Disclaimer:** This system is developed for academic and research purposes only. "
    "It is not intended to replace professional medical diagnosis or clinical decision-making. "
    "Always consult a qualified healthcare professional for an actual diagnosis."
)

st.markdown(
    "Enter the patient's cytological measurements below (obtained from a digitized image of a "
    "fine needle aspirate / FNA of a breast mass), then click **Predict Diagnosis** to get an "
    "AI-assisted prediction together with a transparent, feature-level explanation of *why* the "
    "model made that prediction."
)

# ---------------------------------------------------------------------------
# Sample patient quick-fill (kept OUTSIDE the form so it can trigger a rerun)
# ---------------------------------------------------------------------------
if "feature_values" not in st.session_state:
    st.session_state.feature_values = {
        f: float(feature_stats.loc["mean", f]) for f in feature_names
    }


def _apply_sample(row: pd.Series):
    for f in feature_names:
        st.session_state.feature_values[f] = float(row[f])
    for f in feature_names:
        st.session_state[f"input_{f}"] = float(row[f])


col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    st.markdown("**Quick-fill options** (for demo/testing purposes):")
with col_b:
    if sample_patients is not None and st.button("Load Sample: Benign patient", use_container_width=True):
        row = sample_patients[sample_patients["diagnosis"] == 0].iloc[0]
        _apply_sample(row)
        st.rerun()
with col_c:
    if sample_patients is not None and st.button("Load Sample: Malignant patient", use_container_width=True):
        row = sample_patients[sample_patients["diagnosis"] == 1].iloc[0]
        _apply_sample(row)
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# Patient input form
# ---------------------------------------------------------------------------
with st.form("patient_form"):
    st.subheader("Patient Medical Parameters")

    for group_name, group_features in FEATURE_GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "Mean Values")):
            cols = st.columns(2)
            for i, feat in enumerate(group_features):
                col = cols[i % 2]
                min_v = float(feature_stats.loc["min", feat])
                max_v = float(feature_stats.loc["max", feat])
                span = max_v - min_v if max_v > min_v else 1.0
                default_val = st.session_state.feature_values.get(feat, float(feature_stats.loc["mean", feat]))
                col.number_input(
                    label=feat,
                    min_value=round(min_v - 0.1 * span, 4),
                    max_value=round(max_v + 0.1 * span, 4),
                    value=round(default_val, 4),
                    step=round(span / 100 if span > 0 else 0.01, 5),
                    format="%.4f",
                    key=f"input_{feat}",
                    help=f"Typical range in dataset: {min_v:.3f} to {max_v:.3f}",
                )

    submitted = st.form_submit_button("Predict Diagnosis", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Prediction + Explanation
# ---------------------------------------------------------------------------
if submitted:
    input_values = {f: st.session_state[f"input_{f}"] for f in feature_names}
    st.session_state.feature_values = input_values

    input_df = pd.DataFrame([input_values])[feature_names]
    input_scaled = scaler.transform(input_df)

    proba_malignant = float(model.predict_proba(input_scaled)[0][1])
    prediction = int(proba_malignant >= 0.5)
    prediction_label = CLASS_NAMES[prediction]

    st.divider()
    st.header("Diagnosis Result")

    res_col1, res_col2 = st.columns([1, 1])
    with res_col1:
        if prediction_label == "Malignant":
            st.error(f"### Predicted Diagnosis: {prediction_label}")
        else:
            st.success(f"### Predicted Diagnosis: {prediction_label}")
    with res_col2:
        st.metric("Estimated Probability of Malignancy", f"{proba_malignant * 100:.1f}%")
        st.progress(min(max(proba_malignant, 0.0), 1.0))

    # --- Local SHAP explanation for this specific patient ---
    st.subheader("Why did the model predict this? (SHAP Local Explanation)")

    with st.spinner("Computing SHAP explanation for this patient..."):
        input_scaled_df = pd.DataFrame(input_scaled, columns=feature_names)
        shap_values_instance = explainer(input_scaled_df)
        values, base_values = _class1_values(shap_values_instance)
        row_values = values[0]
        base_value = base_values[0] if np.ndim(base_values) > 0 else base_values

    exp_col1, exp_col2 = st.columns([1.2, 1])

    with exp_col1:
        explanation_obj = shap.Explanation(
            values=row_values,
            base_values=base_value,
            data=input_scaled[0],
            feature_names=feature_names,
        )
        fig = plt.figure()
        shap.plots.waterfall(explanation_obj, max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(
            "Waterfall plot: starts at the model's average/base prediction and shows how each "
            "feature pushes the output up (red, toward Malignant) or down (blue, toward Benign) "
            "for THIS patient."
        )

    with exp_col2:
        contrib_df = pd.DataFrame({
            "Feature": feature_names,
            "Patient Value": [input_values[f] for f in feature_names],
            "SHAP Contribution": row_values,
        })
        contrib_df["Direction"] = np.where(
            contrib_df["SHAP Contribution"] > 0, "Toward Malignant", "Toward Benign"
        )
        contrib_df["Abs Impact"] = contrib_df["SHAP Contribution"].abs()
        contrib_df = contrib_df.sort_values("Abs Impact", ascending=False).drop(columns="Abs Impact")
        contrib_df["SHAP Contribution"] = contrib_df["SHAP Contribution"].round(4)
        contrib_df["Patient Value"] = contrib_df["Patient Value"].round(4)

        st.markdown("**Top contributing features (positive & negative):**")
        st.dataframe(
            contrib_df.head(10).style.background_gradient(
                cmap="RdBu_r", subset=["SHAP Contribution"], vmin=-max(1e-6, contrib_df["SHAP Contribution"].abs().max()),
                vmax=max(1e-6, contrib_df["SHAP Contribution"].abs().max()),
            ),
            use_container_width=True, hide_index=True,
        )

    # --- Natural-language explanation ---
    st.subheader("Plain-Language Interpretation")
    explanation_text = generate_natural_language_explanation(
        feature_names, [input_values[f] for f in feature_names], row_values,
        prediction_label, proba_malignant, top_n=5,
    )
    st.markdown(explanation_text)

    st.caption(
        "Note: SHAP contributions are additive on the model's raw output (log-odds / margin) "
        "scale, so individual values are not literal percentages, but their sign (direction) "
        "and relative magnitude are directly meaningful and comparable."
    )

# ---------------------------------------------------------------------------
# Global model insights (precomputed SHAP plots, always available)
# ---------------------------------------------------------------------------
st.divider()
with st.expander("Global Model Insights (SHAP) - How the model behaves across ALL patients"):
    st.markdown(
        "These plots summarize the trained model's behavior across the full test set "
        "(generated by `generate_explanations.py`), independent of the single patient above."
    )
    g1, g2 = st.columns(2)
    summary_path = os.path.join(FIG_DIR, "shap_summary_plot.png")
    bar_path = os.path.join(FIG_DIR, "shap_bar_plot.png")
    if os.path.exists(summary_path):
        g1.image(summary_path, caption="SHAP Summary (Beeswarm) Plot", use_container_width=True)
    if os.path.exists(bar_path):
        g2.image(bar_path, caption="SHAP Feature Importance (Bar Plot)", use_container_width=True)
    if not os.path.exists(summary_path):
        st.info("Run `python generate_explanations.py` once to generate these global plots.")

st.divider()
st.caption(
    "Explainable AI Framework for Medical Diagnosis - Academic/Research Prototype. "
    "Model trained on the Breast Cancer Wisconsin (Diagnostic) Dataset. "
    "Explainability powered by SHAP."
)

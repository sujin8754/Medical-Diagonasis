"""
explainability.py
-------------------
Explainable AI (XAI) utilities for the medical diagnosis model.

Primary technique : SHAP (SHapley Additive exPlanations)
Secondary/optional : LIME (Local Interpretable Model-agnostic Explanations)

SHAP is used as the primary technique because it:
  * Has a strong game-theoretic foundation (Shapley values from cooperative
    game theory) -> each feature's contribution is fairly attributed.
  * Provides BOTH a global view (which features matter most across all
    patients) and a local view (why THIS patient got THIS prediction),
    from the exact same underlying values -- unlike many other methods.
  * Is consistent: if a feature's effect increases, its SHAP value never
    decreases, which is an important trust property in a medical setting.

This module provides small, well-documented helper functions that
train_model.py's saved artifacts can be plugged into, and that both
app.py (Streamlit) and generate_explanations.py (batch script) reuse.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

CLASS_NAMES = ["Benign", "Malignant"]


# ---------------------------------------------------------------------------
# Building the SHAP explainer
# ---------------------------------------------------------------------------
def build_shap_explainer(model, background_data):
    """
    Builds a SHAP explainer for `model` using `background_data` (a small
    reference sample of SCALED training data, as a DataFrame or ndarray).

    shap.Explainer() automatically picks the fastest exact/near-exact
    algorithm for the model type:
        - Linear models (e.g. Logistic Regression) -> LinearExplainer
        - Tree ensembles (Random Forest, XGBoost)   -> TreeExplainer
    Both are efficient enough to run on a normal laptop in real time, which
    is why SHAP (rather than a purely sampling-based method) was chosen.
    """
    # A background sample of ~100 rows keeps computation fast while still
    # giving a representative "baseline" for the model.
    if len(background_data) > 100:
        background_data = shap.sample(background_data, 100, random_state=42)
    explainer = shap.Explainer(model, background_data)
    return explainer


def compute_shap_values(explainer, X):
    """Computes SHAP values for the rows in X (scaled features)."""
    return explainer(X)


def _class1_values(shap_values):
    """
    Some explainers (e.g. TreeExplainer on a binary classifier) return SHAP
    values with an extra dimension for each class. This helper normalizes
    the output so we always work with the SHAP values for class 1
    (Malignant) -- i.e. how much each feature pushes the prediction TOWARD
    a malignant diagnosis.
    """
    values = shap_values.values
    base_values = shap_values.base_values
    if values.ndim == 3:  # (n_samples, n_features, n_classes)
        values = values[:, :, 1]
        base_values = base_values[:, 1] if np.ndim(base_values) > 1 else base_values
    return values, base_values


# ---------------------------------------------------------------------------
# Global explanations
# ---------------------------------------------------------------------------
def plot_global_summary(shap_values, X, feature_names, save_path):
    """
    SHAP summary (beeswarm) plot: shows, for every feature, how its value
    (color) relates to its impact on the prediction (x-axis position) across
    ALL patients in X. This is the main "global explanation" visualization.
    """
    values, _ = _class1_values(shap_values)
    plt.figure()
    shap.summary_plot(
        values, X, feature_names=feature_names, show=False, plot_size=(9, 8)
    )
    plt.title("SHAP Summary Plot - Global Feature Impact (toward Malignant)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_global_bar(shap_values, feature_names, save_path, max_display=15):
    """
    SHAP bar plot: mean(|SHAP value|) per feature -- a simple ranked bar
    chart of overall feature importance, easier to read at a glance than
    the beeswarm plot.
    """
    values, _ = _class1_values(shap_values)
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]

    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_names[i] for i in order][::-1],
        mean_abs[order][::-1],
        color="#3B7DD8",
    )
    plt.xlabel("Mean |SHAP value|  (average impact on model output)")
    plt.title("SHAP Feature Importance (Global)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    return [(feature_names[i], float(mean_abs[i])) for i in order]


# ---------------------------------------------------------------------------
# Local (per-patient) explanations
# ---------------------------------------------------------------------------
def get_instance_shap(shap_values, index):
    """Extracts the SHAP values, base value, and feature contributions for a
    single row (patient) at position `index`."""
    values, base_values = _class1_values(shap_values)
    row_values = values[index]
    base_value = base_values[index] if np.ndim(base_values) > 0 else base_values
    return row_values, base_value


def plot_local_waterfall(shap_values, index, feature_names, save_path, max_display=10):
    """
    SHAP waterfall plot for ONE patient: shows exactly how each feature
    pushed the prediction up (toward Malignant) or down (toward Benign),
    starting from the model's base/expected value.
    """
    values, base_values = _class1_values(shap_values)
    base_value = base_values[index] if np.ndim(base_values) > 0 else base_values

    explanation = shap.Explanation(
        values=values[index],
        base_values=base_value,
        data=shap_values.data[index] if shap_values.data is not None else None,
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_natural_language_explanation(
    feature_names, feature_values, shap_row_values, prediction_label,
    prediction_proba, top_n=5,
):
    """
    Converts raw SHAP numbers into a short, plain-English explanation that a
    non-technical user (e.g. a doctor without an ML background) can read.

    prediction_label : "Malignant" or "Benign"
    prediction_proba : model's predicted probability of the MALIGNANT class
    """
    contributions = list(zip(feature_names, feature_values, shap_row_values))
    # Sort by absolute impact, most influential first
    contributions.sort(key=lambda t: abs(t[2]), reverse=True)
    top = contributions[:top_n]

    lines = []
    lines.append(
        f"The model predicts **{prediction_label}** with a "
        f"{prediction_proba * 100:.1f}% estimated probability of malignancy."
    )
    lines.append("The most influential factors behind this prediction were:")

    for feat, val, shap_val in top:
        direction = "increased" if shap_val > 0 else "decreased"
        toward = "toward Malignant" if shap_val > 0 else "toward Benign"
        lines.append(
            f"  - **{feat}** = {val:.3f} {direction} the malignancy risk "
            f"({toward}), contributing {shap_val:+.3f} to the prediction score."
        )

    positive = [c for c in top if c[2] > 0]
    negative = [c for c in top if c[2] < 0]
    lines.append(
        f"\nSummary: {len(positive)} of the top {top_n} factors pushed the "
        f"prediction toward Malignant, while {len(negative)} pushed it "
        f"toward Benign. The final decision reflects the combined (additive) "
        f"effect of all 30 input features, not just these top factors."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Optional secondary technique: LIME
# ---------------------------------------------------------------------------
def build_lime_explainer(X_train_scaled, feature_names):
    """Builds a LIME tabular explainer using the scaled training data as the
    reference distribution for generating local perturbations."""
    from lime.lime_tabular import LimeTabularExplainer
    return LimeTabularExplainer(
        training_data=np.array(X_train_scaled),
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        mode="classification",
        random_state=42,
    )


def explain_instance_lime(lime_explainer, model, instance_scaled, num_features=10):
    """
    Returns LIME's local explanation for one patient as a list of
    (feature_description, weight) tuples, e.g. ("worst radius > 16.8", 0.12).
    """
    exp = lime_explainer.explain_instance(
        np.array(instance_scaled),
        model.predict_proba,
        num_features=num_features,
        labels=(1,),
    )
    return exp.as_list(label=1)


def compare_shap_lime_top_features(shap_top_features, lime_pairs, top_n=5):
    """
    Compares the top-N most influential features identified by SHAP vs LIME
    for the same patient, and reports the overlap -- used to sanity-check
    that both explanation methods broadly agree (a good interpretability
    "consistency" indicator to discuss in the report).
    """
    shap_top = set(f for f, _ in shap_top_features[:top_n])
    # LIME feature descriptions look like "worst radius > 16.80"; extract the
    # base feature name (strip the inequality/threshold part).
    lime_top = set()
    for desc, _ in lime_pairs[:top_n]:
        for fname in shap_top:
            if fname in desc:
                lime_top.add(fname)
                break
    overlap = shap_top & lime_top
    return {
        "shap_top_features": sorted(shap_top),
        "lime_top_features": sorted(lime_top),
        "overlap": sorted(overlap),
        "overlap_ratio": len(overlap) / top_n if top_n else 0.0,
    }

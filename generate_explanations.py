"""
generate_explanations.py
--------------------------
Standalone script that loads the trained best model (saved by
train_model.py) and produces all the SHAP/LIME explainability artifacts
used in the project report:

    reports/figures/shap_summary_plot.png
    reports/figures/shap_bar_plot.png
    reports/figures/shap_waterfall_malignant_example.png
    reports/figures/shap_waterfall_benign_example.png
    reports/explanations_sample_output.txt   (natural-language explanations
                                               + SHAP vs LIME comparison)

Run this AFTER train_model.py:
    python train_model.py
    python generate_explanations.py

This script is separate from app.py so that the heavier, one-time SHAP
computation over the whole test set (used for the global plots) doesn't need
to be repeated every time the Streamlit app starts.
"""

import os
import joblib
import numpy as np
import pandas as pd

from explainability import (
    build_shap_explainer, compute_shap_values,
    plot_global_summary, plot_global_bar,
    plot_local_waterfall, get_instance_shap,
    generate_natural_language_explanation,
    build_lime_explainer, explain_instance_lime,
    compare_shap_lime_top_features,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CLASS_NAMES = ["Benign", "Malignant"]


def main():
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))

    X_train_raw = pd.read_csv(os.path.join(MODELS_DIR, "X_train_background.csv"))
    test_df = pd.read_csv(os.path.join(MODELS_DIR, "X_test_sample.csv"))
    y_test = test_df["diagnosis"].values
    X_test_raw = test_df[feature_names]

    X_train_scaled = scaler.transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    print("Building SHAP explainer...")
    explainer = build_shap_explainer(model, pd.DataFrame(X_train_scaled, columns=feature_names))

    print("Computing SHAP values for the test set (this is the slowest step)...")
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    shap_values = compute_shap_values(explainer, X_test_scaled_df)

    print("Saving global SHAP summary (beeswarm) plot...")
    plot_global_summary(
        shap_values, X_test_scaled_df, feature_names,
        os.path.join(FIG_DIR, "shap_summary_plot.png"),
    )

    print("Saving global SHAP bar plot...")
    top_features = plot_global_bar(
        shap_values, feature_names, os.path.join(FIG_DIR, "shap_bar_plot.png"),
    )
    print("Top global features by mean |SHAP value|:")
    for f, v in top_features[:10]:
        print(f"  {f:30s} {v:.4f}")

    # ------------------------------------------------------------------
    # Local explanations for two representative patients:
    # one true Malignant case, one true Benign case.
    # ------------------------------------------------------------------
    malignant_idx = int(np.where(y_test == 1)[0][0])
    benign_idx = int(np.where(y_test == 0)[0][0])

    output_lines = []
    lime_explainer = build_lime_explainer(X_train_scaled, feature_names)

    for label, idx, fname in [
        ("Malignant example (true label = Malignant)", malignant_idx, "shap_waterfall_malignant_example.png"),
        ("Benign example (true label = Benign)", benign_idx, "shap_waterfall_benign_example.png"),
    ]:
        proba = model.predict_proba(X_test_scaled[idx:idx + 1])[0][1]
        pred_label = CLASS_NAMES[int(proba >= 0.5)]

        plot_local_waterfall(shap_values, idx, feature_names, os.path.join(FIG_DIR, fname))

        row_values, base_value = get_instance_shap(shap_values, idx)
        explanation_text = generate_natural_language_explanation(
            feature_names, X_test_raw.iloc[idx].values, row_values,
            pred_label, proba, top_n=5,
        )

        # SHAP vs LIME comparison for this same patient
        shap_top_sorted = sorted(
            zip(feature_names, np.abs(row_values)), key=lambda t: t[1], reverse=True
        )
        lime_pairs = explain_instance_lime(lime_explainer, model, X_test_scaled[idx])
        comparison = compare_shap_lime_top_features(shap_top_sorted, lime_pairs, top_n=5)

        output_lines.append("=" * 80)
        output_lines.append(label)
        output_lines.append("=" * 80)
        output_lines.append(explanation_text)
        output_lines.append("\nLIME top contributing features (for comparison):")
        for desc, weight in lime_pairs[:5]:
            output_lines.append(f"  {desc:40s} weight={weight:+.4f}")
        output_lines.append(
            f"\nSHAP vs LIME agreement on top-5 features: "
            f"{len(comparison['overlap'])}/5 features overlap "
            f"({comparison['overlap_ratio'] * 100:.0f}%)"
        )
        output_lines.append(f"  SHAP top-5: {comparison['shap_top_features']}")
        output_lines.append(f"  LIME top-5 (matched to SHAP names): {comparison['lime_top_features']}")
        output_lines.append("")

    out_path = os.path.join(REPORTS_DIR, "explanations_sample_output.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"\nSaved local explanation plots + text report to: {REPORTS_DIR}")
    print("\n".join(output_lines))


if __name__ == "__main__":
    main()

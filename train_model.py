"""
train_model.py
----------------
Trains and compares three classification models for the
"Explainable AI Framework for Medical Diagnosis" project, then saves the
best-performing model (plus its scaler and metadata) to the models/ folder
so app.py can load it for live predictions.

Models compared:
    1. Logistic Regression  -> simple, inherently interpretable baseline
    2. Random Forest        -> robust bagged-tree ensemble
    3. XGBoost               -> gradient-boosted tree ensemble, usually the
                                strongest performer on tabular medical data

Run with:  python train_model.py
(Takes well under a minute on a normal laptop -- the dataset only has 569
rows and 30 columns.)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed, just save PNG files
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

from preprocessing import load_raw_dataframe, preprocess_and_split, RANDOM_STATE

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports", "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

CLASS_NAMES = ["Benign", "Malignant"]  # index 0, 1  (see preprocessing.py)


def get_models():
    """Defines the three candidate models with sensible, beginner-friendly
    hyperparameters (no heavy tuning, kept deliberately simple)."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=RANDOM_STATE,
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    """Computes the standard classification metrics + confusion matrix and
    saves a confusion-matrix heatmap for this model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
    )
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(os.path.join(REPORTS_DIR, f"confusion_matrix_{safe_name}.png"), dpi=150)
    plt.close()

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    return metrics, y_proba


def plot_roc_curves(results, y_test):
    """Overlays the ROC curve of all three models on a single plot."""
    plt.figure(figsize=(6, 5))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, label=f"{name} (AUC = {res['metrics']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "roc_curve_comparison.png"), dpi=150)
    plt.close()


def plot_metric_comparison(results):
    """Grouped bar chart comparing all metrics across all models."""
    metric_names = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    df = pd.DataFrame({
        name: [res["metrics"][m] for m in metric_names]
        for name, res in results.items()
    }, index=metric_names)

    ax = df.plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()


def main():
    print("Loading and preprocessing the Breast Cancer Wisconsin (Diagnostic) dataset...")
    df = load_raw_dataframe()
    (
        X_train, X_test, y_train, y_test,
        scaler, feature_names, X_train_raw, X_test_raw,
    ) = preprocess_and_split(df)

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]} | Features: {len(feature_names)}")

    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        metrics, y_proba = evaluate_model(name, model, X_test, y_test)
        results[name] = {"model": model, "metrics": metrics, "y_proba": y_proba}
        # Save each individual trained model too (useful for the notebook / comparison)
        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, os.path.join(MODELS_DIR, f"{safe_name}.pkl"))

    plot_roc_curves(results, y_test)
    plot_metric_comparison(results)

    # ------------------------------------------------------------------
    # Select the best model.
    # Primary criterion: ROC-AUC (threshold-independent, standard choice
    # for medical diagnosis tasks where both classes matter).
    # Tie-break: F1-score (balances precision & recall for the positive /
    # malignant class, which is the clinically important class to catch).
    # ------------------------------------------------------------------
    best_name = max(
        results,
        key=lambda n: (results[n]["metrics"]["roc_auc"], results[n]["metrics"]["f1_score"]),
    )
    best_model = results[best_name]["model"]
    print(f"\n>>> Best model selected: {best_name} "
          f"(ROC-AUC = {results[best_name]['metrics']['roc_auc']:.4f}, "
          f"F1 = {results[best_name]['metrics']['f1_score']:.4f})")

    # Save artifacts needed by app.py / explainability.py
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, "feature_names.pkl"))

    # Save a small unscaled background/reference sample for SHAP + UI defaults
    X_train_raw.reset_index(drop=True).to_csv(
        os.path.join(MODELS_DIR, "X_train_background.csv"), index=False
    )
    X_test_raw.assign(diagnosis=y_test).reset_index(drop=True).to_csv(
        os.path.join(MODELS_DIR, "X_test_sample.csv"), index=False
    )

    comparison_table = {
        name: res["metrics"] for name, res in results.items()
    }
    metadata = {
        "best_model_name": best_name,
        "class_mapping": {"0": "Benign", "1": "Malignant"},
        "selection_criterion": "Highest ROC-AUC, tie-break by F1-score",
        "all_model_metrics": comparison_table,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "n_features": len(feature_names),
        "random_state": RANDOM_STATE,
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved best model, scaler, and metadata to: {MODELS_DIR}")
    print(f"Saved evaluation plots to: {REPORTS_DIR}")
    print("\nFull comparison table:")
    print(pd.DataFrame(comparison_table).T.round(4))


if __name__ == "__main__":
    main()

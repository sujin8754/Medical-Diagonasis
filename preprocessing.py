"""
preprocessing.py
-----------------
Data loading and preprocessing utilities for the
"Explainable AI Framework for Medical Diagnosis" project.

Dataset : Breast Cancer Wisconsin (Diagnostic) Data Set
Source  : Built into scikit-learn (sklearn.datasets.load_breast_cancer),
          originally from the UCI Machine Learning Repository:
          https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
          Also mirrored on UCI ML Repo id 17 and commonly hosted on Kaggle as
          "Breast Cancer Wisconsin (Diagnostic) Data Set".

Target  : 0 = malignant, 1 = benign  (sklearn's own encoding)
          For readability in this project we RELABEL so that:
              1 = Malignant (disease present)
              0 = Benign    (disease absent)
          This matches the common convention in medical-diagnosis papers
          where the "positive" class (1) represents the disease of interest.

Why this dataset needs no manual download:
    scikit-learn ships the data locally as part of the library, so
    `load_breast_cancer()` works completely offline. A CSV copy is also
    saved to data/breast_cancer.csv the first time this module runs, purely
    for transparency/reproducibility (so students can open it in Excel).

This file is intentionally simple and heavily commented so it is easy to
follow for a final-year student.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_CSV_PATH = os.path.join(DATA_DIR, "breast_cancer.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_raw_dataframe(save_csv: bool = True) -> pd.DataFrame:
    """
    Loads the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn
    and returns it as a single pandas DataFrame containing the 30 feature
    columns plus a 'diagnosis' target column.

    diagnosis = 1 -> Malignant (cancer present)
    diagnosis = 0 -> Benign    (cancer absent)

    If save_csv is True, a copy is written to data/breast_cancer.csv so the
    raw data is visible/inspectable outside of Python as well.
    """
    raw = load_breast_cancer()
    df = pd.DataFrame(raw.data, columns=raw.feature_names)

    # sklearn encodes 0=malignant, 1=benign. We flip it so that
    # 1 = malignant (disease present), which is the more intuitive
    # convention for a "positive diagnosis" in medical ML papers.
    df["diagnosis"] = 1 - raw.target  # 1 = malignant, 0 = benign

    if save_csv:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(DATA_CSV_PATH, index=False)

    return df


def get_feature_names() -> list:
    """Returns the list of 30 clinical/cytological feature names, in order."""
    raw = load_breast_cancer()
    return list(raw.feature_names)


def get_feature_display_groups() -> dict:
    """
    Groups the 30 features into the three natural categories used by the
    Breast Cancer Wisconsin dataset, so the Streamlit UI can present them in
    tidy sections instead of one long list of 30 boxes.

    Each measurement (radius, texture, perimeter, ...) was computed from a
    digitized image of a fine needle aspirate (FNA) of a breast mass, and is
    reported as: the MEAN value, the STANDARD ERROR, and the WORST
    (largest/most severe) value across the cell nuclei in the image.
    """
    base_features = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave points", "symmetry",
        "fractal dimension",
    ]
    groups = {
        "Mean Values": [f"mean {f}" for f in base_features],
        "Standard Error Values": [f"{f} error" for f in base_features],
        "Worst (Largest) Values": [f"worst {f}" for f in base_features],
    }
    return groups


def preprocess_and_split(df: pd.DataFrame = None):
    """
    Splits the dataframe into train/test sets and scales the features with
    StandardScaler (zero mean, unit variance). The SAME scaler is reused for
    all three models (Logistic Regression, Random Forest, XGBoost) purely to
    keep the pipeline simple: scaling does not hurt tree-based models, and it
    lets the Streamlit app use a single, shared preprocessing step regardless
    of which trained model ends up being selected as "best".

    Returns
    -------
    X_train_scaled, X_test_scaled : np.ndarray
    y_train, y_test               : np.ndarray
    scaler                        : fitted StandardScaler
    feature_names                 : list[str]
    X_train_raw, X_test_raw       : pd.DataFrame (unscaled, for SHAP display)
    """
    if df is None:
        df = load_raw_dataframe()

    feature_names = [c for c in df.columns if c != "diagnosis"]
    X = df[feature_names]
    y = df["diagnosis"].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,  # keep the malignant/benign ratio equal in train & test
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return (
        X_train_scaled, X_test_scaled,
        y_train, y_test,
        scaler, feature_names,
        X_train_raw, X_test_raw,
    )


if __name__ == "__main__":
    # Quick manual test: `python preprocessing.py`
    dataframe = load_raw_dataframe()
    print(f"Dataset shape: {dataframe.shape}")
    print(f"Class balance:\n{dataframe['diagnosis'].value_counts()}")
    print(f"Saved CSV to: {DATA_CSV_PATH}")

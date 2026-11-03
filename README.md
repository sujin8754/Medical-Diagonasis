# Explainable AI Framework for Medical Diagnosis

An academic/research prototype that predicts breast cancer diagnosis (Malignant vs Benign)
from clinical/cytological measurements, and **explains why** the model made each prediction
using SHAP (SHapley Additive exPlanations), with an optional LIME comparison.

> **Disclaimer:** This system is developed for academic and research purposes only. It is
> not intended to replace professional medical diagnosis or clinical decision-making. Always
> consult a qualified healthcare professional for an actual diagnosis.

---

## 1. Project Overview

| | |
|---|---|
| **Problem** | Binary medical diagnosis classification (Malignant vs Benign) |
| **Dataset** | Breast Cancer Wisconsin (Diagnostic) Data Set |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Best model** | Logistic Regression (ROC-AUC = 0.996, selected automatically by `train_model.py`) |
| **Explainability** | SHAP (primary), LIME (secondary, for comparison) |
| **Interface** | Streamlit single-page web app |

### Why this dataset?

The dataset was chosen over the alternative (Pima Indians Diabetes) because it offers the best
balance of the four criteria requested for this project:

- **Simplicity** – ships built into scikit-learn (`sklearn.datasets.load_breast_cancer`), so
  there is zero manual download step and the project works fully offline.
- **Accuracy** – clean, complete data (no missing values, no placeholder zeros to impute) lets
  all three models reach 96–97% accuracy and ROC-AUC above 0.99, so the explainability results
  are built on a genuinely trustworthy classifier rather than a noisy one.
- **Explainability** – 30 well-defined, continuous cytological measurements (radius, texture,
  concavity, etc.) give SHAP clean, medically meaningful features to attribute importance to.
- **IEEE-paper suitability** – it is one of the most widely used tabular benchmarks in
  published explainable-AI / medical-ML literature, making comparisons with existing work easy.

### Contribution (for the IEEE-style paper)

This project's contribution is **not** "a new classifier" — Logistic Regression, Random Forest
and XGBoost are all standard, well-known algorithms. The contribution is the **explainability
layer built around the classifier**:

1. A unified pipeline that trains and fairly compares three model families and automatically
   selects the best one by ROC-AUC.
2. Patient-level (local) SHAP explanations that convert raw model output into human-readable,
   feature-by-feature justifications for a single prediction.
3. A quantitative SHAP-vs-LIME agreement check on individual patients, as a simple
   interpretability-consistency indicator.
4. A working, end-to-end transparent AI-assisted diagnosis interface (Streamlit) that displays
   the prediction, its confidence, and its explanation together — not the prediction alone.

We deliberately make **no** claims that the system replaces doctors, is clinically validated,
or achieves "100% accurate diagnosis" — it is a research/academic prototype demonstrating an
XAI methodology, evaluated on a public benchmark dataset.

---

## 2. Project Architecture

```
Patient Input (Streamlit form)
        |
        v
Data Preprocessing (preprocessing.py -> StandardScaler)
        |
        v
Trained ML Model (models/best_model.pkl)
        |
        v
Disease Prediction (Malignant / Benign + probability)
        |
        v
SHAP Explainability (explainability.py -> shap.Explainer)
        |
        v
Feature Contributions (positive / negative, per feature)
        |
        v
Human-Readable Explanation (plain-language summary)
        |
        v
Streamlit Interface (app.py) -- displays all of the above
```

---

## 3. Project Structure

```
xai-medical-diagnosis/
├── app.py                      # Streamlit web application (the main UI)
├── train_model.py              # Loads data, trains & compares 3 models, saves the best one
├── explainability.py           # SHAP (+ optional LIME) helper functions
├── preprocessing.py            # Dataset loading, cleaning, train/test split, scaling
├── generate_explanations.py    # Batch script: produces global SHAP plots + sample explanations
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/
│   └── breast_cancer.csv       # Auto-generated CSV copy of the dataset (for inspection)
├── models/                     # Auto-generated: trained models, scaler, metadata
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   ├── logistic_regression.pkl / random_forest.pkl / xgboost.pkl
│   ├── model_metadata.json
│   ├── X_train_background.csv
│   └── X_test_sample.csv
├── reports/
│   ├── figures/                # Auto-generated: confusion matrices, ROC curves, SHAP plots
│   └── explanations_sample_output.txt
└── notebooks/
    └── eda_and_experiments.ipynb   # Optional exploratory data analysis notebook
```

### What each file does (plain-language summary)

- **`preprocessing.py`** – Loads the Breast Cancer Wisconsin dataset, relabels it so `1 =
  Malignant` and `0 = Benign` (more intuitive than sklearn's default), splits it into an 80%
  training set and 20% test set (stratified so both sets have the same malignant/benign ratio),
  and scales all 30 features with `StandardScaler`.
- **`train_model.py`** – Trains Logistic Regression, Random Forest and XGBoost on the same
  data, evaluates each with accuracy/precision/recall/F1/ROC-AUC, saves a confusion matrix and
  comparison chart for each, and automatically picks the best model (highest ROC-AUC) to save
  as `models/best_model.pkl` for the app to use.
- **`explainability.py`** – A library of reusable functions: build a SHAP explainer for any of
  the three model types, compute global (all-patients) and local (one-patient) SHAP values,
  draw the summary/bar/waterfall plots, and turn SHAP numbers into a plain-English paragraph.
  Also includes optional LIME functions for a secondary explanation method.
- **`generate_explanations.py`** – Runs once after training to produce the global SHAP summary
  and bar plots (used in the report and shown in the app), plus two worked examples (one
  malignant, one benign patient) with full natural-language explanations and a SHAP-vs-LIME
  comparison, saved to `reports/explanations_sample_output.txt`.
- **`app.py`** – The Streamlit web app. Patient enters values -> clicks "Predict Diagnosis" ->
  app shows the prediction, confidence, a SHAP waterfall plot, a feature-contribution table, and
  a plain-language explanation, all on one page.

---

## 4. Dataset Details

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository
  (<https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic>), bundled
  directly inside scikit-learn (`sklearn.datasets.load_breast_cancer`) — **no manual download
  is required**; running `preprocessing.py` or `train_model.py` loads it automatically and also
  saves a CSV copy to `data/breast_cancer.csv`.
- **Records:** 569 patients
- **Features:** 30 numeric features, computed from a digitized image of a fine needle aspirate
  (FNA) of a breast mass. Each of 10 base measurements (radius, texture, perimeter, area,
  smoothness, compactness, concavity, concave points, symmetry, fractal dimension) is reported
  as its **mean**, **standard error**, and **"worst" (largest)** value — giving 10 × 3 = 30
  features.
- **Target variable:** `diagnosis` — 1 = Malignant, 0 = Benign (212 malignant / 357 benign
  cases in this project's relabeling)
- **Preprocessing:** No missing values in this dataset (verified). 80/20 stratified train/test
  split (random_state=42) + `StandardScaler` feature scaling.

---

## 5. Machine Learning Results

All three models were trained with the same 80/20 stratified split and evaluated on the same
held-out test set (114 patients):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** (selected) | 0.965 | 0.975 | 0.929 | 0.951 | **0.996** |
| Random Forest | 0.974 | 1.000 | 0.929 | 0.963 | 0.995 |
| XGBoost | 0.965 | 1.000 | 0.905 | 0.950 | 0.993 |

**Selection criterion:** highest ROC-AUC (threshold-independent, the standard metric for
medical diagnosis where both classes matter), with F1-score as a tie-break. All three models
are very close in performance on this dataset — this is itself a useful observation for the
report: a simple, inherently interpretable model (Logistic Regression) performs competitively
with more complex ensembles here, which strengthens the case for transparent AI in this domain.

All evaluation plots (confusion matrices, ROC curve comparison, metric comparison bar chart)
are saved to `reports/figures/` by `train_model.py`.

---

## 6. Explainable AI (SHAP)

`generate_explanations.py` produces:

- **`shap_summary_plot.png`** – global beeswarm plot: which features matter most across all
  patients, and whether high/low values push predictions toward Malignant or Benign.
- **`shap_bar_plot.png`** – global bar chart ranking features by mean |SHAP value|.
- **`shap_waterfall_*_example.png`** – local, per-patient waterfall plots showing exactly how
  each feature pushed one specific patient's prediction.
- **`explanations_sample_output.txt`** – plain-language explanations for two example patients,
  plus a SHAP-vs-LIME top-feature agreement check (both methods agreed on 3 of the top 5
  features for both example patients in our test run — i.e., 60% overlap, a reasonable
  consistency result worth discussing/limitations in the report).

Inside the Streamlit app, the same SHAP machinery runs **live** on whatever values the user
enters, so the waterfall plot, contribution table, and plain-language paragraph always reflect
the exact patient currently being predicted.

### Why XAI is needed here

A black-box "Malignant / Benign" label alone gives a clinician nothing to check the model's
reasoning against. SHAP decomposes the prediction into additive, per-feature contributions
grounded in cooperative game theory (Shapley values), so a user can see, for example, that a
high `radius error` and high `mean concave points` were the dominant reasons a specific patient
was flagged — reasoning that can be sanity-checked against known clinical intuition (larger,
more irregular/concave cell nuclei are associated with malignancy).

### Limitations of AI explanations in medical diagnosis (for the report)

- SHAP explains **what the model learned**, not **medical ground truth** — a feature can be
  influential to the model without being causally relevant to the disease.
- Explanations are only as trustworthy as the underlying model; a model with hidden bias will
  produce confidently-explained but biased predictions.
- SHAP values are computed relative to a background/reference distribution — different
  background samples can shift the numeric contributions somewhat, even though the ranking is
  usually stable.
- SHAP and LIME do not always agree perfectly (see the ~60% top-feature overlap above), which
  is itself an important honesty check: no single XAI method should be treated as absolute
  ground truth.
- None of this replaces clinical judgment, biopsy confirmation, or a qualified radiologist/
  pathologist's opinion — this system is a decision-support research prototype only.

---

## 7. How to Run This Project (Windows / VS Code)

### Prerequisites
- Windows 10/11 with **Python 3.10, 3.11, or 3.12** installed
  (download from <https://www.python.org/downloads/> — tick "Add Python to PATH" during install)
- Visual Studio Code with the Python extension (optional but recommended)

### Step-by-step

1. **Unzip the project** and open the folder in VS Code
   (`File > Open Folder... > xai-medical-diagnosis`).

2. **Open a terminal in VS Code** (`` Terminal > New Terminal `` , or `` Ctrl+` ``).

3. **Create and activate a virtual environment:**
   ```powershell
   python -m venv venv
   venv\Scripts\activate
   ```
   (In VS Code, you can also select this `venv` as the Python interpreter via
   `Ctrl+Shift+P` → "Python: Select Interpreter".)

4. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

5. **Train the models** (creates the `models/` and part of `reports/` folders):
   ```powershell
   python train_model.py
   ```

6. **Generate the SHAP/LIME explanation artifacts:**
   ```powershell
   python generate_explanations.py
   ```

7. **Launch the Streamlit app:**
   ```powershell
   streamlit run app.py
   ```
   This opens the app automatically in your default browser at `http://localhost:8501`.
   If it doesn't open automatically, copy that URL into your browser.

8. **Use the app:** expand a section (Mean / Standard Error / Worst Values), adjust the numbers
   (or click **"Load Sample: Benign patient"** / **"Load Sample: Malignant patient"** to
   auto-fill realistic example values), then click **Predict Diagnosis**.

To stop the app, go back to the VS Code terminal and press `Ctrl+C`.

---

## 8. Testing / Sample Patient Inputs

The app includes two built-in "quick-fill" buttons that load real (held-out test set) patient
records so you can test the full pipeline in seconds without typing 30 numbers by hand:

| Button | Expected model behaviour |
|---|---|
| **Load Sample: Benign patient** | After clicking Predict Diagnosis, expect **Predicted Diagnosis: Benign** with a low (near 0%) estimated probability of malignancy, and a waterfall plot where most bars point toward Benign (blue, negative SHAP values). |
| **Load Sample: Malignant patient** | After clicking Predict Diagnosis, expect **Predicted Diagnosis: Malignant** with a high (near 100%) estimated probability of malignancy, and a waterfall plot dominated by bars pointing toward Malignant (red, positive SHAP values), typically led by features like `radius error`, `mean concave points`, or `worst concavity`. |

You can also manually edit any individual value after loading a sample (e.g. lower `worst
radius` and `worst area` on the malignant sample) and re-click Predict Diagnosis to see the
prediction and explanation update — a good way to demonstrate the model's sensitivity to
specific features during a project demo/viva.

If model files are missing (fresh clone, before running `train_model.py`), the app shows a
clear on-screen instruction instead of crashing.

---

## 9. Evaluation Metrics Reference

- **Accuracy** – overall fraction of correct predictions.
- **Precision** – of patients predicted Malignant, how many actually were (minimizes false
  alarms).
- **Recall (Sensitivity)** – of patients who are actually Malignant, how many the model caught
  (minimizes missed cancers — usually the more clinically critical metric).
- **F1-score** – harmonic mean of precision and recall.
- **ROC-AUC** – overall ability to rank malignant patients higher than benign ones, independent
  of the classification threshold.
- **Confusion Matrix** – full breakdown of true/false positives and negatives.

---

## 10. Notes on Scope

This is a final-year / MCA academic project, intentionally kept to classic, well-understood
ML models and a single, clean tabular dataset rather than deep learning or medical imaging, so
that every part of the pipeline (data → model → explanation → UI) remains understandable,
reproducible, and runnable on a normal laptop in well under a minute of training time.

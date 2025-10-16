# CHDP: Cancer & Heart Disease Prediction — Technical Learning Report

Author: Alenyorige Obed Alemya  
Project type: Learning/portfolio (educational; not for clinical use)

---

## Abstract
This learning project builds end-to-end, interpretable tabular ML pipelines for two clinical prediction tasks: heart disease classification and breast cancer diagnosis. The goal is to practice reproducible ML workflows (EDA → preprocessing → model selection → evaluation → explainability → simple app) and to document methods, metrics, and limitations clearly. Models are trained with scikit-learn pipelines, task-specific preprocessing, cross-validated hyperparameter search, and SHAP explainability. A Streamlit app provides a form-based interface for making predictions using the trained artifacts.

## Datasets
- Heart Disease (UCI Cleveland) — tabular clinical features (e.g., age, BP, cholesterol).  
  Target: presence of heart disease (binary).
- Wisconsin Diagnostic Breast Cancer (WDBC) — tabular features derived from digitized images (30 numeric features).  
  Target: malignant vs. benign (binary).

Note: These are public benchmark datasets suitable for learning; results here are not validated for clinical deployment.

## Methods
- Preprocessing: Column-wise transformers with numeric/categorical handling; per-task feature schema persisted to `data/processed/{task}_schema.json` for app alignment.
- Models: Baselines and class-weighted Logistic Regression (best for both tasks in this run) with GridSearchCV (StratifiedKFold) for hyperparameter selection; class imbalance handled with `class_weight='balanced'` where applicable.
- Evaluation: Holdout test metrics (ROC-AUC, accuracy, precision, recall, F1) and cross-validation best score.
- Explainability: SHAP-based global feature importance; local explanations available in app.
- App: Streamlit form-based UI for structured inputs (no text parsing), loading `preprocessor.joblib` and `model.joblib` per task under `artifacts/{task}/`.

## Experiments
- Training and evaluation notebook: `CHDP_workbook.ipynb`  
- Artifacts directory: `artifacts/{task}/` with `model.joblib`, `preprocessor.joblib`, `metrics.json`, and SHAP outputs under `shap/`.
- Re-runs will regenerate artifacts and schemas in a reproducible manner (versions pinned in `requirements.txt`).

## Results
Summary of test-set performance for the best model per task (current run):

Heart Disease (task = `heart`) — best_model: Logistic Regression
- ROC-AUC: 0.962
- Accuracy: 0.933
- Precision: 0.909
- Recall: 0.952
- F1: 0.930
- CV best score (mean CV metric): 0.905
- Metrics source: `artifacts/heart/metrics.json`

Breast Cancer (task = `cancer`) — best_model: Logistic Regression
- ROC-AUC: 0.996
- Accuracy: 0.953
- Precision: 0.912
- Recall: 0.969
- F1: 0.939
- CV best score (mean CV metric): 0.995
- Metrics source: `artifacts/cancer/metrics.json`

## Explainability
- SHAP global feature importance computed per task; CSVs stored at:
  - `artifacts/heart/shap/feature_importance.csv`
  - `artifacts/cancer/shap/feature_importance.csv`
- The Streamlit app can display local feature attributions for individual predictions when SHAP is installed.

## Reproducibility
- Environment: see `requirements.txt`.
- End-to-end steps:
  1. Open and run `CHDP_workbook.ipynb` to train models, compute SHAP, and export artifacts/schemas.
  2. Launch the app to interact with trained models:
     - `streamlit run src/app/streamlit_app.py`
- Key outputs:
  - `artifacts/{task}/model.joblib`, `preprocessor.joblib`, `metrics.json`, `shap/feature_importance.csv`
  - `data/processed/{task}_schema.json`

## Limitations (Learning Context)
- Small, public benchmark datasets; results vary with splits and random seeds.
- Minimal clinical validation; feature semantics for cancer are imaging-derived and not clinician-curated for deployment.
- Limited model family exploration in this snapshot (logistic regression performed best); no calibration or subgroup fairness analysis yet.
- Streamlit UI is a learning-focused demo, not a clinical interface.

## Ethical/Use Note
This work is strictly educational and not intended for diagnosis or treatment. Any real-world clinical use would require rigorous validation, calibration, fairness assessment, and regulatory review.

## Possible Extensions (Future Learning)
- Probability calibration (Platt/Isotonic) and decision-threshold tuning.
- Subgroup/fairness analysis and robustness to domain shift.
- Expanded model families (tree ensembles, gradient boosting) with SHAP comparison.
- Batch inference and experiment tracking (e.g., MLflow) for more systematic re-runs.

## Acknowledgements
- UCI Machine Learning Repository (Heart Cleveland).  
- Wisconsin Diagnostic Breast Cancer dataset providers.

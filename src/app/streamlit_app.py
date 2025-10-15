import streamlit as st

# Configure page ASAP: must be the first Streamlit command
st.set_page_config(
    page_title="CHDP: Cancer & Heart Disease Prediction",
    page_icon="🏥",
    layout="wide",
)

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

"""CHDP Streamlit App: Form-based UI for Cancer & Heart predictions.

This app loads trained artifacts and presents a clean form-based interface
for entering model features. SHAP feature importance is shown when available.
"""

# Import SHAP for explanations (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# (page config set above)

# Constants (mirror notebook config)
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SCHEMA_DIR = DATA_DIR / "processed"

@st.cache_data
def load_schema(task: str):
    """Load the cached schema for a task (cancer/heart)."""
    path = SCHEMA_DIR / f"{task}_schema.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_resource
def load_artifacts(task: str):
    """Load preprocessor and model for a task"""
    try:
        preprocessor = joblib.load(ARTIFACTS_DIR / task / 'preprocessor.joblib')
        model = joblib.load(ARTIFACTS_DIR / task / 'model.joblib')
        return preprocessor, model
    except Exception as e:
        st.error(f"Failed to load artifacts for {task}: {e}")
        st.info("Run the notebook first to train models and generate artifacts.")
        return None, None

# ---------------------------------
# Field metadata and helpers
# ---------------------------------

HEART_FIELD_HELP = {
    "age": "Age in years.",
    "sex": "Sex: 1 = male, 0 = female.",
    "cp": "Chest pain type: 0=typical angina, 1=atypical angina, 2=non-anginal, 3=asymptomatic.",
    "trestbps": "Resting blood pressure (mm Hg).",
    "chol": "Serum cholesterol (mg/dl).",
    "fbs": "Fasting blood sugar > 120 mg/dl: 1=true, 0=false.",
    "restecg": "Resting ECG: 0=normal, 1=ST-T abnormality, 2=LVH.",
    "thalach": "Max heart rate achieved (bpm).",
    "exang": "Exercise-induced angina: 1=yes, 0=no.",
    "oldpeak": "ST depression induced by exercise relative to rest.",
    "slope": "Slope of peak exercise ST: 0=upsloping, 1=flat, 2=downsloping.",
    "ca": "Number of major vessels colored by fluoroscopy (0-3).",
    "thal": "Thalassemia: 1=normal, 2=fixed defect, 3=reversible defect.",
}

HEART_CATEGORICAL_OPTIONS = {
    "sex": [("Female", 0), ("Male", 1)],
    "cp": [
        ("Typical angina", 0),
        ("Atypical angina", 1),
        ("Non-anginal pain", 2),
        ("Asymptomatic", 3),
    ],
    "fbs": [("False", 0), ("True", 1)],
    "restecg": [("Normal", 0), ("ST-T abnormality", 1), ("LVH", 2)],
    "exang": [("No", 0), ("Yes", 1)],
    "slope": [("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
    "ca": [(str(i), i) for i in range(0, 4)],
    "thal": [("Normal", 1), ("Fixed defect", 2), ("Reversible defect", 3)],
}

CANCER_FIELD_HELP_DEFAULT = "Imaging-derived numeric feature from nuclei measurements."

def predict_with_model(features_df: pd.DataFrame, task: str):
    """Make prediction using loaded artifacts"""
    preprocessor, model = load_artifacts(task)
    if preprocessor is None or model is None:
        return None
    
    try:
        X_vec = preprocessor.transform(features_df)
        proba = float(model.predict_proba(X_vec)[:, 1][0])
        pred = int(proba >= 0.5)
        return {"probability": proba, "prediction": pred}
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def get_shap_explanation(features_df: pd.DataFrame, task: str, max_features: int = 5):
    """Get SHAP explanation for the prediction"""
    if not SHAP_AVAILABLE:
        return None
        
    preprocessor, model = load_artifacts(task)
    if preprocessor is None or model is None:
        return None
        
    try:
        X_vec = preprocessor.transform(features_df)
        
        # Simple explainer selection (matches notebook logic)
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        if isinstance(model, (RandomForestClassifier,)) or hasattr(model, 'get_booster'):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_vec)
        else:
            # Kernel explainer fallback
            explainer = shap.KernelExplainer(model.predict_proba, X_vec)
        
        shap_values = explainer.shap_values(X_vec)
        values = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        # Get feature names from preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feature_{i}" for i in range(values.shape[1])]
        
        # Get top contributing features
        importance = np.abs(values[0])
        top_indices = np.argsort(importance)[-max_features:][::-1]
        
        explanations = []
        for idx in top_indices:
            explanations.append({
                "feature": feature_names[idx],
                "shap_value": float(values[0][idx]),
                "importance": float(importance[idx])
            })
        
        return explanations
        
    except Exception as e:
        st.warning(f"SHAP explanation failed: {e}")
        return None

# App title and description
st.title("🏥 CHDP: Cancer & Heart Disease Prediction")
st.caption("Form-based input for fast, reliable predictions. No text parsing.")

# Sidebar for task selection
st.sidebar.title("Configuration")
task = st.sidebar.selectbox("Select Prediction Task", ["heart", "cancer"])

# Check if artifacts exist
preprocessor, model = load_artifacts(task)
if preprocessor is None or model is None:
    st.stop()

# Load metrics if available
metrics_file = ARTIFACTS_DIR / task / "metrics.json"
if metrics_file.exists():
    with open(metrics_file, encoding='utf-8') as f:
        metrics = json.load(f)
    st.sidebar.metric("Best Model", metrics.get("best_model", "Unknown"))
    st.sidebar.metric("Test ROC AUC", f"{metrics.get('roc_auc', 0):.3f}")

# Main interface: form-based input
st.header("🧾 Input Features")
schema = load_schema(task)
if not schema:
    st.warning(f"Schema not found. Run the notebook to generate schemas: data/processed/{task}_schema.json")
    st.stop()

numeric_cols = schema.get("numeric", [])
categorical_cols = schema.get("categorical", [])

# Short task description
with st.expander("ℹ️ Task & Model Description", expanded=True):
    if task == "heart":
        st.markdown(
            "Predicts presence of heart disease (1=yes, 0=no) based on clinical features like age, blood pressure, cholesterol, etc."
        )
    else:
        st.markdown(
            "Predicts malignant vs benign diagnosis from imaging-derived numeric features (30 columns)."
        )

with st.form("feature_form"):
    cols = st.columns(2)
    inputs = {}

    # Numeric features
    with cols[0]:
        st.subheader("Numeric Features")
        for col in numeric_cols:
            help_text = HEART_FIELD_HELP.get(col, CANCER_FIELD_HELP_DEFAULT) if task == "heart" else CANCER_FIELD_HELP_DEFAULT
            # Provide some sensible ranges where known
            if col in ("age",):
                val = st.number_input(col, min_value=0, max_value=120, value=50, help=help_text)
            elif col in ("trestbps",):
                val = st.number_input(col, min_value=0, max_value=250, value=120, help=help_text)
            elif col in ("chol",):
                val = st.number_input(col, min_value=0, max_value=800, value=200, help=help_text)
            elif col in ("thalach",):
                val = st.number_input(col, min_value=0, max_value=250, value=150, help=help_text)
            elif col in ("oldpeak",):
                val = st.number_input(col, min_value=0.0, max_value=10.0, value=1.0, step=0.1, help=help_text)
            else:
                # Generic numeric
                val = st.number_input(col, value=0.0, help=help_text)
            inputs[col] = val

    # Categorical features
    with cols[1]:
        st.subheader("Categorical Features")
        for col in categorical_cols:
            if task == "heart" and col in HEART_CATEGORICAL_OPTIONS:
                options = HEART_CATEGORICAL_OPTIONS[col]
                labels = [label for label, _ in options]
                values = [value for _, value in options]
                idx = st.selectbox(col, options=list(range(len(labels))), format_func=lambda i: labels[i], help=HEART_FIELD_HELP.get(col, ""))
                inputs[col] = values[idx]
            else:
                # Default binary categorical
                val = st.selectbox(col, options=[0, 1], format_func=lambda x: str(x), help=(HEART_FIELD_HELP.get(col, "") if task == "heart" else ""))
                inputs[col] = val

    submitted = st.form_submit_button("🔮 Make Prediction")

# Build features_df from inputs
features_df = None
if schema:
    cols_all = numeric_cols + categorical_cols
    row = {c: inputs.get(c, np.nan) for c in cols_all}
    features_df = pd.DataFrame([row])

# Prediction section
if submitted and features_df is not None:
    st.header("🎯 Prediction Results")
    prediction_result = predict_with_model(features_df, task)
        
    if prediction_result:
        prob = prediction_result["probability"]
        pred_class = prediction_result["prediction"]

        # Display prediction with color coding
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Probability", f"{prob:.3f}")
        with c2:
            risk_level = "HIGH" if prob > 0.7 else "MEDIUM" if prob > 0.3 else "LOW"
            color = "🔴" if risk_level == "HIGH" else "🟡" if risk_level == "MEDIUM" else "🟢"
            st.metric("Risk Level", f"{color} {risk_level}")
        with c3:
            prediction_text = "POSITIVE" if pred_class == 1 else "NEGATIVE"
            st.metric("Prediction", prediction_text)

        # SHAP Explanation
        if SHAP_AVAILABLE:
            st.subheader("📊 Feature Importance (SHAP)")
            explanations = get_shap_explanation(features_df, task)
            if explanations:
                for exp in explanations:
                    impact = "↗️" if exp["shap_value"] > 0 else "↘️"
                    st.write(f"{impact} **{exp['feature']}**: {exp['shap_value']:.3f}")
            else:
                st.info("SHAP explanations not available for this model type.")
        else:
            st.info("Install SHAP for feature importance explanations: `pip install shap`")

        # Model info
        with st.expander("ℹ️ Model Information"):
            st.write(f"**Task**: {task.title()}")
            st.write(f"**Features used**: {len(features_df.columns)}")
            if metrics_file.exists():
                st.write(f"**Best Model**: {metrics.get('best_model', 'Unknown')}")
                st.write(f"**Test ROC AUC**: {metrics.get('roc_auc', 0):.3f}")

# Footer
st.markdown("---")
st.markdown(
    "🔬 **CHDP Project** | Built with Streamlit | "
    "⚠️ *For educational purposes only - not for medical diagnosis*"
)
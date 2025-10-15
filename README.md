# CHDP

Cancer & Heart Disease Prediction

## Quick Start (Notebook-first)

This repo is set up for a notebook-first workflow with minimal folders. You can train, evaluate, run SHAP, and export artifacts directly from the notebook.

### Requirements file location

- The dependencies are listed in `requirements.txt` at the repository root (`CHDP/requirements.txt`).

### Create and use a virtual environment (recommended)

Linux/macOS (bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

After installing, ensure your Jupyter kernel points to this venv. In VS Code, use the "Python: Select Interpreter" command and pick `.venv`.

### Open and run the notebook

1. Open `CHDP_workbook.ipynb`.
2. Run the first cell to create `data/` and `artifacts/` folders and verify imports.
3. If imports fail, run the provided install cell in the notebook (it uses your current interpreter).
4. Proceed through the sections to train models, evaluate on a holdout set, and generate SHAP explanations.

Outputs (artifacts) will be saved under:
- `artifacts/cancer/` and `artifacts/heart/`

## Run the Streamlit app

After training models in the notebook, launch the app:

```bash
streamlit run src/app/streamlit_app.py
```

The app provides:
- **Form-based Input**: Enter model features via friendly inputs (no text parsing)
- **Predictions**: View probability, risk level, and class prediction
- **Explainability**: Optional SHAP-based feature importance
- **Task Selection**: Switch between heart disease and cancer

## Roadmap / Future Improvements

- Optional offline LLM extraction (heart only): parse clinician notes into structured features using a small local model (e.g., Ollama + Llama). This will be an optional enhancement on top of the form-based UI.
- The cancer task will remain form-based (features are imaging-derived, not natural language).

If you prefer a more modular layout later, the plan in `plan.md` describes a `src/` structure and tests you can promote code into.

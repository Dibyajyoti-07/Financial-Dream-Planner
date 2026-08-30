import json
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent.parent / "models" / "salary_model.pkl"
METADATA_PATH = Path(__file__).parent.parent / "models" / "model_metadata.json"

_model = None
_metadata = None


def _load():
    global _model, _metadata
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} not found - run models/train_and_compare.py first")
        _model = joblib.load(MODEL_PATH)
        _metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return _model, _metadata


def is_loaded():
    try:
        _load()
        return True
    except FileNotFoundError:
        return False


def predict_salary(age, city, education, job_role):
    model, metadata = _load()
    X = pd.DataFrame([{
        "Age": age,
        "City": city,
        "Education": education,
        "Job_Role": job_role,
    }])
    raw = float(model.predict(X)[0])
    salary = max(raw, 1.0)
    salary_range = metadata["training_salary_range"]
    low_confidence = salary < salary_range["min"] or salary > salary_range["max"]
    return {
        "predicted_monthly_salary": round(salary, 2),
        "low_confidence": low_confidence,
    }

"""Carga del pipeline entrenado y ejecucion de una prediccion."""

import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "bank_marketing_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"

FEATURE_ORDER = [
    "age",
    "balance",
    "campaign",
    "job",
    "marital",
    "education",
    "housing",
    "loan",
]

THRESHOLD = 0.5

_model = None
_metrics = None


def get_model():
    """Lee el .joblib una sola vez y lo deja en memoria."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No existe el modelo en {MODEL_PATH}. "
                "Corre primero: python training/train.py"
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metrics() -> dict:
    global _metrics
    if _metrics is None:
        if METRICS_PATH.exists():
            _metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        else:
            _metrics = {}
    return _metrics


def is_model_available() -> bool:
    return MODEL_PATH.exists()


def classify(probability: float) -> str:
    if probability >= THRESHOLD:
        return "Potencialmente interesado"
    return "Poco probable"


def predict(payload: dict) -> dict:
    model = get_model()

    row = {campo: payload[campo] for campo in FEATURE_ORDER}
    sample = pd.DataFrame([row], columns=FEATURE_ORDER)

    probability = float(model.predict_proba(sample)[0, 1])
    prediction = "yes" if probability >= THRESHOLD else "no"

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "classification": classify(probability),
        "threshold": THRESHOLD,
    }

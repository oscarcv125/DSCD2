"""
Entrenamiento del modelo de propension a contratar un deposito a plazo.

Este script se corre una sola vez (o cada vez que se quiera reentrenar).
La API nunca lo ejecuta: solo carga el .joblib que este script deja en models/.

Uso:
    python training/train.py
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "bank.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "bank_marketing_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

NUMERIC_FEATURES = ["age", "balance", "campaign"]
CATEGORICAL_FEATURES = ["job", "marital", "education", "housing", "loan"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "y"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"No encontre {DATA_PATH}. Corre primero: python training/get_data.py"
        )

    df = pd.read_csv(DATA_PATH, sep=";")

    faltantes = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if faltantes:
        raise ValueError(f"Al csv le faltan columnas: {faltantes}")

    return df[FEATURES + [TARGET]].copy()


def build_pipeline() -> Pipeline:
    """Preprocesamiento y modelo en un solo objeto, que es lo que se serializa."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Dataset: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Tasa de contratacion: {(df[TARGET] == 'yes').mean():.2%}\n")

    X = df[FEATURES]
    y = (df[TARGET] == "yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }

    cm = confusion_matrix(y_test, pred)

    print("Metricas sobre el conjunto de prueba")
    for nombre, valor in metrics.items():
        print(f"  {nombre:10s} {valor}")

    print("\nMatriz de confusion")
    print(
        pd.DataFrame(
            cm,
            index=["Real: no contrata", "Real: contrata"],
            columns=["Pred: no contrata", "Pred: contrata"],
        ).to_string()
    )

    print("\nReporte por clase")
    print(classification_report(y_test, pred, target_names=["no", "yes"], zero_division=0))

    joblib.dump(pipeline, MODEL_PATH)

    salida = {
        "model": "LogisticRegression",
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_features": ["duration"],
        "n_rows": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate": round(float(y.mean()), 4),
        "threshold": 0.5,
        "metrics": metrics,
        "confusion_matrix": {
            "true_negative": int(cm[0, 0]),
            "false_positive": int(cm[0, 1]),
            "false_negative": int(cm[1, 0]),
            "true_positive": int(cm[1, 1]),
        },
        "categories": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
    }
    METRICS_PATH.write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nPipeline guardado en: {MODEL_PATH}")
    print(f"Metricas guardadas en: {METRICS_PATH}")


if __name__ == "__main__":
    main()

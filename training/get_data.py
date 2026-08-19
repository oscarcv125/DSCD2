"""Deja data/bank.csv, bajado de UCI o generado localmente si no hay red.

Se usa bank.csv, la muestra del 10% (4521 registros):
https://archive.ics.uci.edu/dataset/222/bank+marketing

    python training/get_data.py

data/DATA_SOURCE.txt queda diciendo cual de las dos rutas se tomo.
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CSV_PATH = DATA_DIR / "bank.csv"
SOURCE_NOTE = DATA_DIR / "DATA_SOURCE.txt"


def download_from_uci() -> pd.DataFrame:
    """El zip de UCI trae otro zip adentro, de ahi los dos pasos."""
    print(f"Descargando {UCI_URL}")
    with urllib.request.urlopen(UCI_URL, timeout=60) as response:
        outer_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
        names = outer.namelist()

        if "bank.csv" in names:
            with outer.open("bank.csv") as f:
                return pd.read_csv(f, sep=";")

        if "bank.zip" in names:
            with outer.open("bank.zip") as f:
                inner_bytes = f.read()
            with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
                with inner.open("bank.csv") as f:
                    return pd.read_csv(f, sep=";")

    raise FileNotFoundError(f"No encontre bank.csv dentro del zip. Contenido: {names}")


def build_offline_substitute(n_rows: int = 4521, seed: int = 42) -> pd.DataFrame:
    """Sustituto con el esquema de bank.csv, para que el proyecto corra sin red.

    Proporciones y rangos salen de las estadisticas publicadas del original.
    Los datos no son reales y no deben usarse para reportar resultados.
    """
    rng = np.random.default_rng(seed)

    jobs = {
        "blue-collar": 0.215, "management": 0.209, "technician": 0.168,
        "admin.": 0.114, "services": 0.092, "retired": 0.050,
        "self-employed": 0.035, "entrepreneur": 0.033, "unemployed": 0.029,
        "housemaid": 0.027, "student": 0.021, "unknown": 0.006,
    }
    maritals = {"married": 0.602, "single": 0.283, "divorced": 0.115}
    educations = {"secondary": 0.513, "tertiary": 0.294, "primary": 0.152, "unknown": 0.041}

    def pick(mapping):
        keys = list(mapping)
        probs = np.array(list(mapping.values()), dtype=float)
        return rng.choice(keys, size=n_rows, p=probs / probs.sum())

    age = np.clip(rng.normal(41.2, 10.6, n_rows).round(), 19, 87).astype(int)
    balance = (rng.lognormal(6.42, 1.36, n_rows) - rng.exponential(200, n_rows)).round()
    balance = np.clip(balance, -3313, 71188).astype(int)
    campaign = rng.geometric(0.36, n_rows)
    cola = rng.random(n_rows) < 0.006
    campaign[cola] = rng.integers(15, 51, cola.sum())
    campaign = np.clip(campaign, 1, 50).astype(int)

    job = pick(jobs)
    marital = pick(maritals)
    education = pick(educations)
    housing = rng.choice(["yes", "no"], n_rows, p=[0.556, 0.444])
    loan = rng.choice(["yes", "no"], n_rows, p=[0.160, 0.840])

    job_effect = {
        "student": 1.05, "retired": 0.95, "unemployed": 0.20, "management": 0.15,
        "admin.": 0.05, "self-employed": 0.0, "technician": -0.05, "unknown": 0.0,
        "housemaid": -0.15, "services": -0.20, "entrepreneur": -0.30, "blue-collar": -0.40,
    }
    education_effect = {"tertiary": 0.30, "secondary": 0.0, "primary": -0.20, "unknown": 0.10}
    marital_effect = {"single": 0.25, "divorced": 0.05, "married": 0.0}

    logit = (
        -1.62
        + np.vectorize(job_effect.get)(job)
        + np.vectorize(education_effect.get)(education)
        + np.vectorize(marital_effect.get)(marital)
        - 0.62 * (housing == "yes")
        - 0.55 * (loan == "yes")
        - 0.22 * (campaign - 1)
        + 0.00006 * balance
        + 0.010 * np.abs(age - 41)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = np.where(rng.binomial(1, prob) == 1, "yes", "no")

    return pd.DataFrame({
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "balance": balance,
        "housing": housing,
        "loan": loan,
        "campaign": campaign,
        "y": y,
    })


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    try:
        df = download_from_uci()
        origen = "UCI Machine Learning Repository (bank.csv original)"
        print(f"Descarga correcta: {df.shape[0]} filas, {df.shape[1]} columnas")
    except Exception as exc:
        print(f"No pude descargar desde UCI: {exc}", file=sys.stderr)
        print("Genero un dataset sustituto para que el proyecto siga corriendo.", file=sys.stderr)
        df = build_offline_substitute()
        origen = "SUSTITUTO GENERADO LOCALMENTE (no son los datos originales de UCI)"

    df.to_csv(CSV_PATH, sep=";", index=False)
    SOURCE_NOTE.write_text(
        f"Origen de data/bank.csv: {origen}\n"
        f"Filas: {len(df)}\n"
        f"Tasa de 'yes': {(df['y'] == 'yes').mean():.4f}\n",
        encoding="utf-8",
    )

    print(f"Guardado en: {CSV_PATH}")
    print(f"Origen: {origen}")
    print(f"Tasa de contratacion (y=yes): {(df['y'] == 'yes').mean():.2%}")


if __name__ == "__main__":
    main()

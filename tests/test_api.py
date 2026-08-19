"""Pruebas de la API: caso feliz, los dos casos de error de la rubrica y rangos."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

CLIENTE_VALIDO = {
    "age": 41,
    "job": "technician",
    "marital": "married",
    "education": "secondary",
    "balance": 3200,
    "housing": "yes",
    "loan": "no",
    "campaign": 2,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_prediccion_valida():
    response = client.post("/predict", json=CLIENTE_VALIDO)
    assert response.status_code == 200

    body = response.json()
    assert body["prediction"] in {"yes", "no"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["classification"]


def test_probabilidad_y_clase_son_consistentes():
    """La clase tiene que salir de comparar la probabilidad contra el umbral."""
    response = client.post("/predict", json=CLIENTE_VALIDO)
    body = response.json()

    esperado = "yes" if body["probability"] >= body["threshold"] else "no"
    assert body["prediction"] == esperado


def test_tipo_incorrecto():
    """Caso 1: age llega como texto."""
    payload = {**CLIENTE_VALIDO, "age": "hola"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "age" in response.json()["detail"]


def test_valor_fuera_de_rango():
    """Caso 2: age negativa."""
    payload = {**CLIENTE_VALIDO, "age": -10}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "18" in response.json()["detail"]


def test_categoria_desconocida():
    """El catalogo tambien es parte del contrato."""
    payload = {**CLIENTE_VALIDO, "job": "ingeniero"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_campo_faltante():
    payload = {k: v for k, v in CLIENTE_VALIDO.items() if k != "balance"}
    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    "campo,valor",
    [("campaign", 0), ("campaign", 500), ("age", 150), ("balance", 99_999_999)],
)
def test_rangos_numericos(campo, valor):
    payload = {**CLIENTE_VALIDO, campo: valor}
    assert client.post("/predict", json=payload).status_code == 422


def test_jubilado_tiene_mas_propension_que_obrero():
    jubilado = {
        **CLIENTE_VALIDO,
        "age": 68, "job": "retired", "education": "tertiary",
        "balance": 6000, "housing": "no", "loan": "no", "campaign": 1,
    }
    obrero = {
        **CLIENTE_VALIDO,
        "age": 38, "job": "blue-collar", "education": "primary",
        "balance": 100, "housing": "yes", "loan": "yes", "campaign": 6,
    }

    p_jubilado = client.post("/predict", json=jubilado).json()["probability"]
    p_obrero = client.post("/predict", json=obrero).json()["probability"]

    assert p_jubilado > p_obrero

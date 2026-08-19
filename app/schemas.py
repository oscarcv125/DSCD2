"""Contratos de entrada y salida de la API."""

from typing import Literal

from pydantic import BaseModel, Field

Job = Literal[
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
]

Marital = Literal["divorced", "married", "single"]
Education = Literal["primary", "secondary", "tertiary", "unknown"]
YesNo = Literal["yes", "no"]


class ClientInput(BaseModel):
    """Los 8 datos que el asesor captura en la interfaz."""

    age: int = Field(
        ...,
        ge=18,
        le=100,
        description="Edad del cliente. Menores de 18 no pueden contratar el producto.",
    )
    job: Job = Field(..., description="Ocupacion del cliente.")
    marital: Marital = Field(..., description="Estado civil.")
    education: Education = Field(..., description="Nivel educativo.")
    balance: float = Field(
        ...,
        ge=-100_000,
        le=1_000_000,
        description="Balance anual promedio en euros. Puede ser negativo.",
    )
    housing: YesNo = Field(..., description="Tiene credito hipotecario.")
    loan: YesNo = Field(..., description="Tiene prestamo personal.")
    campaign: int = Field(
        ...,
        ge=1,
        le=60,
        description="Numero de contactos realizados durante la campana. Minimo 1.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 41,
                    "job": "technician",
                    "marital": "married",
                    "education": "secondary",
                    "balance": 3200,
                    "housing": "yes",
                    "loan": "no",
                    "campaign": 2,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    prediction: Literal["yes", "no"]
    probability: float = Field(..., description="Probabilidad estimada de contratacion.")
    classification: str = Field(..., description="Etiqueta legible para el asesor.")
    threshold: float = Field(..., description="Umbral usado para convertir probabilidad en clase.")


class ErrorResponse(BaseModel):
    error: str
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    features: list[str]

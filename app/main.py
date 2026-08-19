"""API de inferencia. El modelo se carga desde disco, nunca se entrena aqui."""

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.inference import get_metrics, is_model_available, predict
from app.schemas import ClientInput, HealthResponse, PredictionResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

app = FastAPI(
    title="Bank Marketing Predictor",
    description="Estima la propension de un cliente a contratar un deposito a plazo.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Aplana el 422 de Pydantic a un solo string que el frontend pueda pintar."""
    problemas = []
    for error in exc.errors():
        campo = ".".join(str(p) for p in error["loc"] if p != "body")
        problemas.append(f"{campo}: {error['msg']}")

    return JSONResponse(
        status_code=422,
        content={
            "error": "Datos invalidos",
            "detail": " | ".join(problemas),
        },
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    metrics = get_metrics()
    return HealthResponse(
        status="ok",
        model_loaded=is_model_available(),
        features=metrics.get("features", []),
    )


@app.get("/model-info")
def model_info() -> dict:
    """Metricas y catalogos de categorias. El frontend llena sus menus con esto."""
    metrics = get_metrics()
    if not metrics:
        raise HTTPException(
            status_code=503,
            detail="Todavia no hay modelo entrenado. Corre python training/train.py",
        )
    return metrics


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(client: ClientInput) -> PredictionResponse:
    try:
        resultado = predict(client.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error inesperado: {exc}") from exc

    return PredictionResponse(**resultado)


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

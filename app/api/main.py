"""ChargeSentinel FastAPI entry point."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routers import anomaly, chargers, demand, digital_twin, health, solar
from src.integration.adapters.base import ModelNotReadyError

app = FastAPI(title="ChargeSentinel Twin", version="0.1.0")


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(
    _request: Request, exc: ModelNotReadyError
) -> JSONResponse:
    """Translate unavailable model adapters into an explicit service response."""

    return JSONResponse(
        status_code=503,
        content={"status": "model_not_ready", "detail": str(exc)},
    )


app.include_router(health.router)
app.include_router(solar.router)
app.include_router(demand.router)
app.include_router(anomaly.router)
app.include_router(chargers.router)
app.include_router(digital_twin.router)

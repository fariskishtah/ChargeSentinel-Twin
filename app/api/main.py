"""ChargeSentinel FastAPI entry point."""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routers import (
    anomaly,
    chargers,
    demand,
    digital_twin,
    health,
    simulator,
    solar,
)
from src.integration.adapters.base import ModelNotReadyError

API_V1_PREFIX = "/api/v1"
logger = logging.getLogger(__name__)

app = FastAPI(title="ChargeSentinel Twin", version="1.0.0")


@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(
    _request: Request, exc: ModelNotReadyError
) -> JSONResponse:
    """Translate unavailable model adapters into an explicit service response."""

    return JSONResponse(
        status_code=503,
        content={
            "error": "model_not_ready",
            "message": "Model or component is not ready.",
            "component": exc.component,
        },
    )


@app.exception_handler(RequestValidationError)
async def invalid_payload_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return useful validation errors without echoing submitted values."""

    issues = [
        {
            "location": error["loc"],
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": "invalid_payload", "issues": issues},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Keep internal exception details out of public responses."""

    logger.error(
        "Unhandled API exception",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "Unexpected internal error.",
        },
    )


app.include_router(health.router)
app.include_router(solar.router, prefix=API_V1_PREFIX)
app.include_router(demand.router, prefix=API_V1_PREFIX)
app.include_router(anomaly.router, prefix=API_V1_PREFIX)
app.include_router(chargers.router, prefix=API_V1_PREFIX)
app.include_router(digital_twin.router, prefix=API_V1_PREFIX)
app.include_router(simulator.router, prefix=API_V1_PREFIX)

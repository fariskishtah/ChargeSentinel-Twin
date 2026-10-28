"""Anomaly scoring route."""

from fastapi import APIRouter, HTTPException

from app.api.schemas import AnomalyRequest, AnomalyResponse, GridLoadRequest, GridLoadResponse
from app.api.services import schema_to_payload
from src.integration.adapters.anomaly import AnomalyUnavailableError, predict_anomaly
from src.grid_load.detector import GridLoadUnavailableError, detect

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


@router.post("/predict", response_model=AnomalyResponse)
def score_anomaly(request: AnomalyRequest) -> AnomalyResponse:
    try:
        return AnomalyResponse(**predict_anomaly(schema_to_payload(request)))
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session-start features.") from None
    except AnomalyUnavailableError:
        raise HTTPException(status_code=503, detail="Anomaly component unavailable.") from None


@router.post("/grid-load", response_model=GridLoadResponse)
def score_grid_load(request: GridLoadRequest) -> GridLoadResponse:
    try:
        return GridLoadResponse(**detect(request.timestamp))
    except GridLoadUnavailableError:
        raise HTTPException(status_code=503, detail="Grid-load detector unavailable.") from None
    except ValueError:
        raise HTTPException(status_code=422, detail="Unsupported historical grid-load interval.") from None

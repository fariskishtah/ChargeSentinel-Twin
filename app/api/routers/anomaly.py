"""Anomaly scoring route."""

from fastapi import APIRouter, HTTPException

from app.api.schemas import AnomalyRequest, AnomalyResponse
from app.api.services import schema_to_payload
from src.integration.adapters.anomaly import AnomalyUnavailableError, predict_anomaly

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


@router.post("/predict", response_model=AnomalyResponse)
def score_anomaly(request: AnomalyRequest) -> AnomalyResponse:
    try:
        return AnomalyResponse(**predict_anomaly(schema_to_payload(request)))
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session-start features.") from None
    except AnomalyUnavailableError:
        raise HTTPException(status_code=503, detail="Anomaly component unavailable.") from None

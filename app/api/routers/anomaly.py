"""Anomaly scoring route."""

from fastapi import APIRouter

from app.api.schemas import AnomalyRequest, AnomalyResponse
from app.api.services import schema_to_payload
from src.integration.adapters.anomaly import predict_anomaly

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


@router.post("/predict", response_model=AnomalyResponse)
def score_anomaly(request: AnomalyRequest) -> AnomalyResponse:
    return AnomalyResponse(**predict_anomaly(schema_to_payload(request)))

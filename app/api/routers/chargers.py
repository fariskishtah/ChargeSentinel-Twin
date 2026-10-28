"""Charger availability and recommendation routes."""

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    AvailabilityRequest,
    AvailabilityResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from app.api.services import schema_to_payload
from src.integration.adapters.availability import predict_availability, recommend

router = APIRouter(prefix="/chargers", tags=["chargers"])


@router.post("/availability", response_model=AvailabilityResponse)
def charger_availability(request: AvailabilityRequest) -> AvailabilityResponse:
    try:
        return AvailabilityResponse(**predict_availability(schema_to_payload(request)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/recommend", response_model=RecommendationResponse)
def recommend_charger(request: RecommendationRequest) -> RecommendationResponse:
    try:
        return RecommendationResponse(**recommend(schema_to_payload(request)))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

"""Charger availability and recommendation routes."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.schemas import (
    AvailabilityRequest,
    AvailabilityResponse,
    RecommendationRequest,
    RecommendationResponse,
)
from app.api.services import schema_to_payload
from src.integration.adapters.availability import predict_availability, recommend

router = APIRouter(prefix="/chargers", tags=["chargers"])


@router.get("/availability", response_model=AvailabilityResponse)
def charger_availability(
    request: Annotated[AvailabilityRequest, Depends()],
) -> AvailabilityResponse:
    return AvailabilityResponse(**predict_availability(schema_to_payload(request)))


@router.post("/recommend", response_model=RecommendationResponse)
def recommend_charger(request: RecommendationRequest) -> RecommendationResponse:
    return RecommendationResponse(**recommend(schema_to_payload(request)))

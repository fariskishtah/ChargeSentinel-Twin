"""EV demand forecast route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.schemas import DemandForecastRequest, DemandForecastResponse
from app.api.services import schema_to_payload
from src.integration.adapters.demand import predict_demand

router = APIRouter(prefix="/demand", tags=["demand"])


@router.get("/forecast", response_model=DemandForecastResponse)
def forecast_demand(
    request: Annotated[DemandForecastRequest, Depends()],
) -> DemandForecastResponse:
    return DemandForecastResponse(**predict_demand(schema_to_payload(request)))

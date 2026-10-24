"""EV demand forecast route."""

from fastapi import APIRouter

from app.api.schemas import DemandForecastRequest, DemandForecastResponse
from app.api.services import schema_to_payload
from src.integration.adapters.demand import predict_demand

router = APIRouter(prefix="/demand", tags=["demand"])


@router.post("/forecast", response_model=DemandForecastResponse)
def forecast_demand(request: DemandForecastRequest) -> DemandForecastResponse:
    return DemandForecastResponse(**predict_demand(schema_to_payload(request)))

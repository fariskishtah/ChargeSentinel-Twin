"""EV demand forecast route."""

from fastapi import APIRouter, HTTPException

from app.api.schemas import DemandForecastRequest, DemandForecastResponse
from app.api.services import schema_to_payload
from src.integration.adapters.demand import predict_demand
from src.integration.demand_history import DemandHistoryError, DemandUnavailableError

router = APIRouter(prefix="/demand", tags=["demand"])


@router.post("/forecast", response_model=DemandForecastResponse)
def forecast_demand(request: DemandForecastRequest) -> DemandForecastResponse:
    try:
        return DemandForecastResponse(**predict_demand(schema_to_payload(request)))
    except DemandHistoryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except DemandUnavailableError:
        raise HTTPException(status_code=503, detail="Demand component unavailable.") from None

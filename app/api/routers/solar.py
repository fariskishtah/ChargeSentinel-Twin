"""Solar forecast route."""

from fastapi import APIRouter, HTTPException

from app.api.schemas import SolarForecastRequest, SolarForecastResponse
from app.api.services import schema_to_payload
from src.integration.adapters.solar import SolarInputError, SolarUnavailableError, predict_solar

router = APIRouter(prefix="/solar", tags=["solar"])


@router.post("/forecast", response_model=SolarForecastResponse)
def forecast_solar(request: SolarForecastRequest) -> SolarForecastResponse:
    try:
        return SolarForecastResponse(**predict_solar(schema_to_payload(request)))
    except SolarInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except SolarUnavailableError:
        raise HTTPException(status_code=503, detail="Solar component unavailable.") from None

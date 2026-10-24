"""Solar forecast route."""

from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.schemas import SolarForecastRequest, SolarForecastResponse
from app.api.services import schema_to_payload
from src.integration.adapters.solar import predict_solar

router = APIRouter(prefix="/solar", tags=["solar"])


@router.get("/forecast", response_model=SolarForecastResponse)
def forecast_solar(
    request: Annotated[SolarForecastRequest, Depends()],
) -> SolarForecastResponse:
    return SolarForecastResponse(**predict_solar(schema_to_payload(request)))

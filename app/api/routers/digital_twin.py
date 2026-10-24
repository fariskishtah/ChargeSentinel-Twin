"""Digital Twin route."""

from fastapi import APIRouter

from app.api.schemas import DigitalTwinRequest, DigitalTwinResponse
from app.api.services import schema_to_payload
from src.digital_twin.simulator import simulate

router = APIRouter(prefix="/digital-twin", tags=["digital-twin"])


@router.post("/simulate", response_model=DigitalTwinResponse)
def simulate_scenario(request: DigitalTwinRequest) -> DigitalTwinResponse:
    return DigitalTwinResponse(**simulate(schema_to_payload(request)))

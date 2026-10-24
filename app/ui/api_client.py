"""HTTP-only client for the ChargeSentinel FastAPI service."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.api.schemas import (
    AnomalyResponse,
    AvailabilityResponse,
    DemandForecastResponse,
    DigitalTwinResponse,
    HealthResponse,
    RecommendationResponse,
    SolarForecastResponse,
)

DEFAULT_API_URL = "http://localhost:8000"
ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class ResultStatus(str, Enum):
    READY = "READY"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    INVALID_INPUT = "INVALID_INPUT"
    API_OFFLINE = "API_OFFLINE"
    BACKEND_ERROR = "BACKEND_ERROR"


@dataclass(frozen=True)
class APIResult:
    status: ResultStatus
    data: dict[str, object] | None = None
    component: str | None = None

    @property
    def is_ready(self) -> bool:
        return self.status is ResultStatus.READY


class ChargeSentinelAPIClient:
    """Call FastAPI and normalize public response states for the UI."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout_seconds: float = 5.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        configured_url = (
            base_url or os.getenv("CHARGESENTINEL_API_URL", DEFAULT_API_URL)
        ).strip()
        self.base_url = (configured_url or DEFAULT_API_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport

    def _request(
        self,
        method: str,
        path: str,
        response_model: type[ResponseModel],
        payload: Mapping[str, object] | None = None,
    ) -> APIResult:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, json=payload)
        except (httpx.InvalidURL, httpx.RequestError):
            return APIResult(status=ResultStatus.API_OFFLINE)

        try:
            body: object = response.json()
        except ValueError:
            return APIResult(status=ResultStatus.BACKEND_ERROR)

        if response.status_code == 200:
            try:
                validated = response_model.model_validate(body)
            except ValidationError:
                return APIResult(status=ResultStatus.BACKEND_ERROR)
            return APIResult(
                status=ResultStatus.READY,
                data=validated.model_dump(mode="json"),
            )

        if response.status_code == 422:
            return APIResult(status=ResultStatus.INVALID_INPUT)

        if response.status_code == 503:
            component = None
            if isinstance(body, dict) and isinstance(body.get("component"), str):
                component = body["component"]
            return APIResult(
                status=ResultStatus.MODEL_NOT_READY,
                component=component,
            )

        return APIResult(status=ResultStatus.BACKEND_ERROR)

    def health(self) -> APIResult:
        return self._request("GET", "/health", HealthResponse)

    def get_solar_forecast(self, payload: Mapping[str, object]) -> APIResult:
        return self._request(
            "POST", "/api/v1/solar/forecast", SolarForecastResponse, payload
        )

    def get_demand_forecast(self, payload: Mapping[str, object]) -> APIResult:
        return self._request(
            "POST", "/api/v1/demand/forecast", DemandForecastResponse, payload
        )

    def predict_anomaly(self, payload: Mapping[str, object]) -> APIResult:
        return self._request(
            "POST", "/api/v1/anomaly/predict", AnomalyResponse, payload
        )

    def get_availability(self, payload: Mapping[str, object]) -> APIResult:
        return self._request(
            "POST", "/api/v1/chargers/availability", AvailabilityResponse, payload
        )

    def get_recommendation(self, payload: Mapping[str, object]) -> APIResult:
        return self._request(
            "POST", "/api/v1/chargers/recommend", RecommendationResponse, payload
        )

    def simulate_digital_twin(self, payload: Mapping[str, object]) -> APIResult:
        return self._request(
            "POST", "/api/v1/digital-twin/simulate", DigitalTwinResponse, payload
        )

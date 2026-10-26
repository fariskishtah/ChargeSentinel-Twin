"""
FastAPI Service for EV Demand Forecasting (Person 2 Integration / Person 5 Ready).
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.demand.inference import predict_demand

app = FastAPI(
    title="EV Smart Energy & Demand Forecasting API",
    description="FastAPI service serving Person 2 EV demand forecasting models.",
    version="1.0.0"
)


class DemandForecastRequest(BaseModel):
    timestamp: str = Field(..., example="2022-01-01T14:00:00", description="Target forecast hour")
    energy_lag_1h: Optional[float] = Field(None, example=420.5, description="Energy delivered in previous hour")
    energy_lag_2h: Optional[float] = Field(None, example=410.0)
    energy_lag_24h: Optional[float] = Field(None, example=450.0, description="Energy delivered same hour yesterday")
    energy_lag_168h: Optional[float] = Field(None, example=430.0, description="Energy delivered same hour last week")
    sessions_lag_1h: Optional[float] = Field(None, example=26.0)
    sessions_lag_2h: Optional[float] = Field(None, example=25.0)
    sessions_lag_24h: Optional[float] = Field(None, example=28.0)
    sessions_lag_168h: Optional[float] = Field(None, example=27.0)
    energy_rolling_mean_24h: Optional[float] = Field(None, example=415.0)
    sessions_rolling_mean_24h: Optional[float] = Field(None, example=25.5)
    temp_c: Optional[float] = Field(18.0, example=12.5)
    humidity: Optional[float] = Field(65.0, example=55.0)
    precip_mm: Optional[float] = Field(0.0, example=0.0)
    history: Optional[List[Dict[str, Any]]] = Field(None, description="Optional raw recent hourly observations")


class DemandForecastResponse(BaseModel):
    timestamp: str
    expected_energy_kwh: float
    expected_sessions: float
    model_name: str
    model_version: str
    horizon: str


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ev_demand_forecasting", "version": "1.0.0"}


@app.post("/demand/forecast", response_model=DemandForecastResponse)
def forecast_demand(payload: DemandForecastRequest):
    """
    Generate next-hour EV energy and session demand forecast.
    """
    try:
        data_dict = payload.model_dump(exclude_unset=True)
        result = predict_demand(data_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/demand/forecast", response_model=DemandForecastResponse)
def get_sample_forecast(timestamp: str = "2022-01-01T14:00:00"):
    """
    GET endpoint for quick browser testing / health preview as required in Person 5 spec.
    """
    try:
        sample_payload = {"timestamp": timestamp}
        result = predict_demand(sample_payload)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8000, reload=True)

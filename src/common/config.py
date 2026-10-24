"""Environment-backed shared configuration."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"


def get_settings() -> Settings:
    return Settings(
        mlflow_tracking_uri=os.getenv(
            "CHARGESENTINEL_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
        )
    )

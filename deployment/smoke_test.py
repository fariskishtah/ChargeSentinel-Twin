"""Smoke-check a running ChargeSentinel Docker Compose stack."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
API_URL = "http://localhost:8000"
UI_URL = "http://localhost:8501"
MLFLOW_URL = os.getenv("DOCKER_MLFLOW_URL", "http://localhost:5000")


def request(
    name: str,
    url: str,
    *,
    payload: dict[str, object] | None = None,
    expected_status: int = 200,
) -> dict[str, object] | None:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    http_request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=10) as response:
            status = response.status
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        response_body = exc.read()

    if status != expected_status:
        raise RuntimeError(f"{name}: expected HTTP {expected_status}, got {status}")

    print(f"PASS: {name}")
    if not response_body:
        return None
    try:
        parsed = json.loads(response_body)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def check_internal_api_connection() -> None:
    result = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "ui",
            "python",
            "-c",
            (
                "import urllib.request; "
                "response=urllib.request.urlopen('http://api:8000/health', timeout=5); "
                "raise SystemExit(0 if response.status == 200 else 1)"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"UI-to-API check failed: {result.stderr.strip()}")
    print("PASS: UI resolves API at http://api:8000")


def check_ui_application_startup() -> None:
    import_check = subprocess.run(
        [
            "docker",
            "compose",
            "exec",
            "-T",
            "--workdir",
            "/tmp",
            "ui",
            "python",
            "-c",
            (
                "import app; "
                "from app.ui.api_client import ChargeSentinelAPIClient; "
                "print('IMPORT_OK'); print('CLIENT_IMPORT_OK')"
            ),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if import_check.returncode != 0:
        raise RuntimeError(f"UI application imports failed: {import_check.stderr.strip()}")

    logs = subprocess.run(
        ["docker", "compose", "logs", "--no-color", "ui"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if logs.returncode != 0:
        raise RuntimeError(f"Could not inspect UI logs: {logs.stderr.strip()}")
    startup_failures = (
        "ModuleNotFoundError",
        "ImportError",
        "PermissionError",
        "Traceback",
    )
    found = [marker for marker in startup_failures if marker in logs.stdout]
    if found:
        raise RuntimeError(f"UI logs contain startup failure markers: {', '.join(found)}")

    print("PASS: UI package imports from outside the project root")
    print("PASS: UI logs contain no import traceback")


def main() -> int:
    try:
        request("API health", f"{API_URL}/health")
        request("API OpenAPI schema", f"{API_URL}/openapi.json")
        request("API documentation", f"{API_URL}/docs")
        request("Streamlit health", f"{UI_URL}/_stcore/health")
        request("MLflow health", f"{MLFLOW_URL}/health")
        demand = request("Real demand inference", f"{API_URL}/api/v1/demand/forecast",
                         payload={"forecast_origin": "2021-10-01T00:00:00"})
        if not demand or demand.get("model_version") != "demand-completion-v1":
            raise RuntimeError("Demand endpoint did not return the repaired model")
        if not isinstance(demand.get("expected_sessions"), float):
            raise RuntimeError("Demand expected sessions were not preserved as float")
        check_ui_application_startup()
        twin = request(
            "Digital Twin simulation",
            f"{API_URL}/api/v1/digital-twin/simulate",
            payload={
                "baseline_solar_kw": 60.0,
                "baseline_demand_kw": 100.0,
                "total_chargers": 10,
                "active_chargers": 10,
                "average_session_minutes": 60.0,
                "demand_multiplier": 1.3,
                "solar_multiplier": 0.5,
                "offline_chargers": ["scenario-charger-1"],
            },
        )
        if not twin or twin.get("is_simulation") is not True:
            raise RuntimeError("Digital Twin response is not labeled as simulation")
        for horizon in (24, 48):
            solar = request(f"Real solar {horizon}h inference",
                f"{API_URL}/api/v1/solar/forecast",
                payload={"forecast_origin":"2024-06-15T00:00:00+00:00", "horizon_hours":horizon})
            if not solar or len(solar.get("points", [])) != horizon:
                raise RuntimeError(f"Solar {horizon}h output length mismatch")
        anomaly = request("Real anomaly inference", f"{API_URL}/api/v1/anomaly/predict",
            payload={"charger_id":"12", "features":{"location":"Financial Industrial Park",
                "district":"Nanhu", "start_time":"2021-10-16T12:02:44"}})
        if not anomaly or anomaly.get("model_version") != "anomaly-v1" or not anomaly.get("reason"):
            raise RuntimeError("Anomaly model or explanation unavailable")
        grid = request("Historical grid-load interval", f"{API_URL}/api/v1/anomaly/grid-load",
            payload={"timestamp": "2021-10-01T00:00:00"})
        if not grid or grid.get("detector_version") != "grid-load-seasonal-rule-v1" or grid.get("is_simulated") is not False:
            raise RuntimeError("Grid-load detector or historical labeling unavailable")
        check_internal_api_connection()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("Docker stack smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

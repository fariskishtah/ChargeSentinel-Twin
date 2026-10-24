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


def main() -> int:
    try:
        request("API health", f"{API_URL}/health")
        request("API OpenAPI schema", f"{API_URL}/openapi.json")
        request("API documentation", f"{API_URL}/docs")
        request("Streamlit health", f"{UI_URL}/_stcore/health")
        request("MLflow health", f"{MLFLOW_URL}/health")
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
        solar = request(
            "Unavailable solar model",
            f"{API_URL}/api/v1/solar/forecast",
            payload={
                "forecast_origin": "2026-10-24T12:00:00+02:00",
                "horizon_hours": 24,
            },
            expected_status=503,
        )
        if not solar or solar.get("component") != "solar":
            raise RuntimeError("Solar endpoint did not preserve model-not-ready state")
        check_internal_api_connection()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print("Docker stack smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

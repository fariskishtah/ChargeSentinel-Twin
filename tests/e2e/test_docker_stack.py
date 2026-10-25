"""Opt-in smoke tests for the running Docker Compose stack."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(
        os.getenv("RUN_DOCKER_TESTS") != "1",
        reason="set RUN_DOCKER_TESTS=1 after starting Docker Compose",
    ),
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
API_URL = os.getenv("DOCKER_API_URL", "http://localhost:8000")
UI_URL = os.getenv("DOCKER_UI_URL", "http://localhost:8501")
MLFLOW_URL = os.getenv("DOCKER_MLFLOW_URL", "http://localhost:5000")

SCENARIO = {
    "baseline_solar_kw": 60.0,
    "baseline_demand_kw": 100.0,
    "total_chargers": 10,
    "active_chargers": 10,
    "average_session_minutes": 60.0,
    "demand_multiplier": 1.3,
    "solar_multiplier": 0.5,
    "offline_chargers": ["scenario-charger-1"],
}


def _request(
    url: str,
    *,
    payload: dict[str, object] | None = None,
) -> tuple[int, bytes]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method="POST" if body else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _compose(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_api_health_returns_200() -> None:
    status, body = _request(f"{API_URL}/health")

    assert status == 200
    assert json.loads(body)["status"] == "ok"


def test_openapi_endpoint_responds() -> None:
    status, body = _request(f"{API_URL}/openapi.json")
    docs_status, _ = _request(f"{API_URL}/docs")

    assert status == 200
    assert docs_status == 200
    assert "/api/v1/digital-twin/simulate" in json.loads(body)["paths"]


def test_ui_health_endpoint_responds() -> None:
    status, _ = _request(f"{UI_URL}/_stcore/health")

    assert status == 200


def test_mlflow_health_endpoint_responds() -> None:
    status, _ = _request(f"{MLFLOW_URL}/health")

    assert status == 200


def test_digital_twin_works_through_containerized_api() -> None:
    status, body = _request(
        f"{API_URL}/api/v1/digital-twin/simulate",
        payload=SCENARIO,
    )
    result = json.loads(body)

    assert status == 200
    assert result["is_simulation"] is True
    assert result["simulation_version"] == "1.0.0"


@pytest.mark.parametrize(
    ("path", "payload", "component"),
    [
        (
            "/api/v1/solar/forecast",
            {
                "forecast_origin": "2026-10-24T12:00:00+02:00",
                "horizon_hours": 24,
            },
            "solar",
        ),
        (
            "/api/v1/demand/forecast",
            {"forecast_origin": "2026-10-24T12:00:00+02:00"},
            "demand",
        ),
        (
            "/api/v1/anomaly/predict",
            {"charger_id": "scenario-charger-1", "features": {"power_kw": 7}},
            "anomaly",
        ),
        (
            "/api/v1/chargers/availability",
            {
                "charger_id": "scenario-charger-1",
                "query_time": "2026-10-24T12:00:00+02:00",
            },
            "availability",
        ),
        (
            "/api/v1/chargers/recommend",
            {"user_eta": "2026-10-24T12:00:00+02:00"},
            "recommendation",
        ),
    ],
)
def test_unavailable_models_still_return_503(
    path: str,
    payload: dict[str, object],
    component: str,
) -> None:
    status, body = _request(f"{API_URL}{path}", payload=payload)

    assert status == 503
    assert json.loads(body)["component"] == component


def test_ui_container_resolves_api_service_hostname() -> None:
    result = _compose(
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
    )

    assert result.returncode == 0, result.stderr


def test_ui_package_imports_outside_project_workdir() -> None:
    result = _compose(
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
            "assert ChargeSentinelAPIClient"
        ),
    )

    assert result.returncode == 0, result.stderr


def test_streamlit_renders_online_and_model_not_ready_states() -> None:
    result = _compose(
        "exec",
        "-T",
        "--workdir",
        "/tmp",
        "ui",
        "python",
        "-c",
        (
            "from streamlit.testing.v1 import AppTest; "
            "app=AppTest.from_file('/app/app/ui/streamlit_app.py').run(timeout=15); "
            "next(item for item in app.button "
            "if item.label == 'Run Solar Forecast').click().run(timeout=15); "
            "success=[item.value for item in app.success]; "
            "warnings=[item.value for item in app.warning]; "
            "raise SystemExit(0 if not app.exception and 'API Online' in success "
            "and any('MODEL NOT READY' in value for value in warnings) else 1)"
        ),
    )

    assert result.returncode == 0, result.stderr


def test_streamlit_digital_twin_form_calls_api() -> None:
    result = _compose(
        "exec",
        "-T",
        "--workdir",
        "/tmp",
        "ui",
        "python",
        "-c",
        (
            "from streamlit.testing.v1 import AppTest; "
            "app=AppTest.from_file('/app/app/ui/streamlit_app.py').run(timeout=15); "
            "next(item for item in app.button "
            "if item.label == 'Run Simulation').click().run(timeout=15); "
            "info=[item.value for item in app.info]; "
            "raise SystemExit(0 if not app.exception "
            "and any('Deterministic what-if result' in value for value in info) "
            "else 1)"
        ),
    )

    assert result.returncode == 0, result.stderr


def test_ui_logs_have_no_application_startup_import_failure() -> None:
    result = _compose("logs", "--no-color", "ui")

    assert result.returncode == 0, result.stderr
    for marker in (
        "ModuleNotFoundError",
        "ImportError",
        "PermissionError",
        "Traceback",
    ):
        assert marker not in result.stdout


def test_services_are_running_without_restarts() -> None:
    running = _compose("ps", "--services", "--status", "running")
    assert running.returncode == 0, running.stderr
    assert set(running.stdout.splitlines()) == {"api", "ui", "mlflow"}

    for service in ("api", "ui", "mlflow"):
        container_id = _compose("ps", "-q", service)
        assert container_id.returncode == 0, container_id.stderr
        inspect = subprocess.run(
            [
                "docker",
                "inspect",
                "--format={{.RestartCount}}",
                container_id.stdout.strip(),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert inspect.returncode == 0, inspect.stderr
        assert inspect.stdout.strip() == "0"

"""Digital Twin interface placeholder."""

from typing import Any, Mapping

from src.integration.adapters.base import ModelNotReadyError


def simulate(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Run a bounded what-if scenario after its contract is implemented."""

    del payload
    raise ModelNotReadyError("Digital Twin simulator is not ready.")

"""Digital Twin interface placeholder."""

from collections.abc import Mapping

from src.integration.adapters.base import ModelNotReadyError


def simulate(payload: Mapping[str, object]) -> dict[str, object]:
    """Run a bounded what-if scenario after its contract is implemented."""

    del payload
    raise ModelNotReadyError("digital_twin")

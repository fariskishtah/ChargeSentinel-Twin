"""Shared adapter errors."""


class ModelNotReadyError(RuntimeError):
    """Raised when an approved model handoff is not available."""

    def __init__(self, component: str) -> None:
        self.component = component
        super().__init__(f"{component} is not ready")

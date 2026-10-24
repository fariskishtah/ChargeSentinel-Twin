"""Shared API service helpers."""

from typing import Any

from pydantic import BaseModel


def schema_to_payload(request: BaseModel) -> dict[str, Any]:
    """Convert a validated request to the adapter payload contract."""

    return request.model_dump(exclude_none=True)


__all__ = ["schema_to_payload"]

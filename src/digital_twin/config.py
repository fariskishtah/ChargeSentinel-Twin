"""Version and named thresholds for deterministic Digital Twin scenarios."""

from typing import Final, Literal, TypeAlias

Severity: TypeAlias = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]

DIGITAL_TWIN_VERSION: Final = "1.0.0"
OUTPUT_PRECISION: Final = 4

# A neutral reference for a normalized load index; it is not measured utilization.
NORMALIZED_BASELINE_UTILIZATION: Final = 0.50

MEDIUM_UTILIZATION: Final = 0.65
HIGH_UTILIZATION: Final = 0.80
CRITICAL_UTILIZATION: Final = 0.95

MEDIUM_RENEWABLE_DEFICIT: Final = 0.50
HIGH_RENEWABLE_DEFICIT: Final = 0.75
CRITICAL_RENEWABLE_DEFICIT: Final = 0.90

MEDIUM_CHARGER_AVAILABILITY_FLOOR: Final = 1.00
HIGH_CHARGER_AVAILABILITY_FLOOR: Final = 0.50
CRITICAL_CHARGER_AVAILABILITY_FLOOR: Final = 0.25

RECOMMENDATIONS: Final[dict[Severity, str]] = {
    "LOW": "System operating within normal simulated range.",
    "MEDIUM": "Monitor charger utilization and renewable coverage.",
    "HIGH": "Consider shifting flexible charging sessions to lower-demand periods.",
    "CRITICAL": (
        "Prioritize critical charging demand and restore offline chargers if possible."
    ),
}

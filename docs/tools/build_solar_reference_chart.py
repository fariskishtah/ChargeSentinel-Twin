"""Plot one recorded modelled PVGIS reference week from the committed derived file."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "team/mazen_solar/bundle/solar-v1/pvgis_h1_history.csv"
DESTINATION = ROOT / "docs/assets/solar_reference_week.png"


def main() -> None:
    times: list[datetime] = []
    values: list[float] = []
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stamp = datetime.fromisoformat(row["timestamp"])
            if datetime(2024, 6, 15) <= stamp < datetime(2024, 6, 22) and row["power_kw"]:
                times.append(stamp)
                values.append(float(row["power_kw"]))
    if len(times) < 150:
        raise ValueError("Expected approximately one week of committed PVGIS reference hours")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7faff")
    ax.fill_between(times, values, color="#b9d8ed", alpha=0.55)
    ax.plot(times, values, color="#15508d", lw=1.8)
    ax.set_title("New Cairo modelled PVGIS reference — June 15–21, 2024", loc="left",
                 color="#143e7d", fontsize=15, weight="bold", pad=14)
    ax.set_ylabel("Modelled PV output (kW)")
    ax.set_xlabel("UTC reference hour")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", color="#5b93c7", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.08, 0.02, "Reference series only · Not measured station output or a forecast trace",
             color="#526578", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINATION, dpi=160, facecolor="white")
    plt.close(fig)
    print(DESTINATION)


if __name__ == "__main__":
    main()

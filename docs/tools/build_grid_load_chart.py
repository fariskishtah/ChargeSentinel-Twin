"""Render a historical Jiaxing charging-load proxy example from committed data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from src.grid_load.detector import interval_features, load_history, score


def main() -> None:
    config, rows = load_history()
    focus = datetime.fromisoformat("2021-10-02T13:00:00")
    first = rows[0]["timestamp"]
    center = int((focus - first).total_seconds() // 3600)
    indices = range(center - 36, center + 13)
    records = [(rows[i]["timestamp"], interval_features(rows, i)) for i in indices]
    timestamps = [stamp for stamp, _ in records]
    observed = [feature["energy_kwh"] for _, feature in records]
    baseline = [feature["expected_energy_kwh"] for _, feature in records]
    alerts = [(stamp, feature["energy_kwh"]) for stamp, feature in records
              if score(feature) >= config["threshold"]]
    focus_feature = interval_features(rows, center)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, ax = plt.subplots(figsize=(11.5, 5.3), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f7faff")
    ax.plot(timestamps, observed, color="#143e7d", lw=2, label="Recorded completed-session energy")
    ax.plot(timestamps, baseline, color="#5b93c7", lw=1.8, ls="--",
            label="Median of eight earlier same-week hours")
    if alerts:
        ax.scatter([x for x, _ in alerts], [y for _, y in alerts], s=45,
                   color="#ce694c", edgecolors="white", linewidths=0.8, zorder=4,
                   label="Rule alert on recorded hour")
    ax.axvspan(focus, focus + timedelta(hours=1), color="#dfb56b", alpha=0.28, zorder=0)
    ax.annotate(
        f"2021-10-02 13:00\n{focus_feature['energy_kwh']:.0f} kWh recorded; "
        f"{focus_feature['expected_energy_kwh']:.0f} kWh baseline\n"
        f"score {score(focus_feature):.2f} (threshold {config['threshold']:.2f})",
        xy=(focus, focus_feature["energy_kwh"]), xytext=(0.53, 0.88),
        textcoords="axes fraction", ha="left", va="top", fontsize=9,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": "#b9cee4"},
        arrowprops={"arrowstyle": "->", "color": "#ce694c", "lw": 1.5},
    )
    ax.set_title("Historical charging-load proxy — Jiaxing hourly energy", loc="left",
                 color="#143e7d", fontsize=15, weight="bold", pad=16)
    ax.set_ylabel("Completed-session energy (kWh)")
    ax.set_xlabel("Source operational time (timezone unverified)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.grid(axis="y", alpha=0.25, color="#5b93c7")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, ncol=1)
    fig.text(0.075, 0.02, "Recorded history only · Alerts are unusual intervals, not confirmed grid faults",
             color="#526578", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    destination = Path(__file__).resolve().parents[1] / "assets" / "grid_load_anomaly_example.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, facecolor="white")
    plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()

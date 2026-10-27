"""Reproducibly package authorized derived history; raw transactions stay local."""
import argparse
import json
from pathlib import Path
import pandas as pd

from team.waseem_demand.demand.build_hourly import build_hourly, sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "64eadd73175f1a063e134ecc7e346b628b8309cf"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    hourly, audit = build_hourly(args.raw)
    model = json.loads((ROOT / "bundle/v1/metadata.json").read_text())
    if model["dataset_version"] != "sha256:" + audit["raw_sha256"]:
        raise ValueError("Source differs from repaired model dataset")
    if len(hourly) != 17547 or audit["raw_rows"] != 441077:
        raise ValueError("Unexpected source coverage")
    destination = ROOT / "runtime_data" / "jiaxing-hourly-demand-v1"
    destination.mkdir(parents=True, exist_ok=True)
    asset = destination / "hourly_history.csv"
    hourly.to_csv(asset, index=False, float_format="%.17g", date_format="%Y-%m-%dT%H:%M:%S")
    metadata = {"data_version": "jiaxing-hourly-demand-v1", "source_dataset": "Jiaxing Charging_Data.csv",
                "source_rows": audit["raw_rows"], "source_sha256": audit["raw_sha256"],
                "aggregation_source_commit": SOURCE_COMMIT,
                "aggregation_method": "Completed session energy at End Time hour; arrivals at Start Time hour",
                "output_rows": len(hourly), "columns": list(hourly.columns),
                "first_hour": hourly.timestamp.iloc[0].isoformat(), "last_hour": hourly.timestamp.iloc[-1].isoformat(),
                "timezone_policy": "Source timezone not independently verified; naive operational time preserved",
                "zero_policy": "No recorded event => zero under complete source capture assumption; no forward filling",
                "derived_sha256": sha256(asset), "file_size_bytes": asset.stat().st_size,
                "earliest_forecast_origin": (hourly.timestamp.iloc[0] + pd.Timedelta(hours=192)).isoformat(),
                "latest_forecast_origin": (hourly.timestamp.iloc[-1] + pd.Timedelta(hours=1)).isoformat()}
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    assert sha256(args.raw) == audit["raw_sha256"]
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

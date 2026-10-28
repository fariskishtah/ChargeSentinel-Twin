"""Training-period-only historical charger abnormal-rate lookup."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


TEAM_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RISK_TABLE_PATH = TEAM_DIR / "bundle" / "availability-v1" / "charger_risk_table.json"
DEFAULT_FEATURES_PATH = TEAM_DIR / "artifacts" / "features_all.pkl"
TRAIN_END = pd.Timestamp("2021-07-01")
RISK_TABLE_VERSION = "historical-charger-risk-v1"


def build_risk_table(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    output_path: str | Path = DEFAULT_RISK_TABLE_PATH,
) -> dict[str, Any]:
    """Build and save the risk table using only sessions before 2021-07-01."""
    frame = joblib.load(Path(features_path))
    required = {"charger_id", "start", "is_abnormal"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Saved features artifact is missing columns: {sorted(missing)}")
    start = pd.to_datetime(frame["start"], errors="coerce")
    abnormal = pd.to_numeric(frame["is_abnormal"], errors="coerce")
    train = frame.loc[(start < TRAIN_END) & abnormal.isin([0, 1]), ["charger_id"]].copy()
    train["is_abnormal"] = abnormal.loc[train.index].astype(int)
    train = train.dropna(subset=["charger_id"])
    if train.empty:
        raise ValueError("No valid pre-2021-07-01 sessions found in saved features")

    grouped = train.groupby("charger_id", sort=True)["is_abnormal"].agg(["mean", "size"])
    table = {
        "version": RISK_TABLE_VERSION,
        "built_from_end": TRAIN_END.date().isoformat(),
        "global_rate": float(train["is_abnormal"].mean()),
        "per_charger": {
            str(charger_id): {"rate": float(row["mean"]), "n_sessions": int(row["size"])}
            for charger_id, row in grouped.iterrows()
        },
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
    return table


def load_risk_table(
    path: str | Path = DEFAULT_RISK_TABLE_PATH,
    features_path: str | Path = DEFAULT_FEATURES_PATH,
) -> dict[str, Any]:
    """Load a saved table, generating it from saved features only if missing."""
    table_path = Path(path)
    if not table_path.is_file():
        return build_risk_table(features_path, table_path)
    table = json.loads(table_path.read_text(encoding="utf-8"))
    if not isinstance(table, dict) or not isinstance(table.get("per_charger"), dict):
        raise ValueError(f"Invalid historical charger risk table: {table_path}")
    rate = float(table.get("global_rate", -1))
    if not 0 <= rate <= 1:
        raise ValueError(f"Invalid global_rate in historical risk table: {table_path}")
    for charger_id, record in table["per_charger"].items():
        charger_rate = float(record.get("rate", -1))
        n_sessions = int(record.get("n_sessions", -1))
        if not 0 <= charger_rate <= 1 or n_sessions < 1:
            raise ValueError(f"Invalid risk record for charger {charger_id!r}")
    return table


def charger_risk(charger_id: Any, table: dict[str, Any]) -> float:
    """Return a charger's training rate, falling back globally when unseen."""
    record = table["per_charger"].get(str(charger_id))
    return float(record["rate"] if record is not None else table["global_rate"])

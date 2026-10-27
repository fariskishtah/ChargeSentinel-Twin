import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import jsonschema
import numpy as np
import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE))
from demand.build_hourly import aggregate, build_hourly, sha256
from demand.features import FEATURES, TARGETS, extract_features, sequence_arrays, split_masks
from demand.inference import predict_demand, load_bundle
from demand.train import EXPERIMENT, fit_scalers, metrics


@pytest.fixture
def hourly():
    return pd.DataFrame({"timestamp": pd.date_range("2021-01-01", periods=400, freq="h"),
                         "energy_kwh": np.arange(400, dtype=float), "sessions": np.arange(400)})


@pytest.fixture
def transactions():
    return pd.DataFrame({"Start Time": ["2020-01-01 00:10:00", "2020-01-01 03:00:00"],
                         "End Time": ["2020-01-01 01:30:00", "2020-01-01 03:30:00"],
                         "Transaction power/kwh": [10., 20.], "UserID": [1, 1], "Charging Post ID": [1, 2]})


def fixture():
    return json.loads((MODULE / "fixtures/request.json").read_text())


def test_raw_file_immutable(tmp_path, transactions):
    path = tmp_path / "raw.csv"
    transactions.to_csv(path, index=False, encoding="gb18030")
    original = path.read_bytes()
    _, audit = build_hourly(path)
    assert path.read_bytes() == original
    assert audit["raw_sha256"] == hashlib.sha256(original).hexdigest()


def test_raw_dataframe_immutable(transactions):
    original = transactions.copy(deep=True)
    aggregate(transactions)
    pd.testing.assert_frame_equal(original, transactions)


def test_continuous_and_empty_hours(transactions):
    d = aggregate(transactions)
    assert d.timestamp.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all()
    assert len(d) == 4
    assert d.iloc[2]["energy_kwh"] == d.iloc[2]["sessions"] == 0


def test_completion_energy_and_start_arrivals(transactions):
    d = aggregate(transactions)
    assert d.energy_kwh.tolist() == [0., 10., 0., 20.]
    assert d.sessions.tolist() == [1, 0, 0, 1]
    assert d.energy_kwh.sum() == transactions["Transaction power/kwh"].sum()
    assert d.sessions.sum() == len(transactions)


def test_no_valid_edge_hours_dropped(transactions):
    transactions.loc[0, "Start Time"] = "2019-12-31 21:27:03"
    assert aggregate(transactions).timestamp.min() == pd.Timestamp("2019-12-31 21:00:00")


@pytest.mark.parametrize("change", ["negative", "backwards", "duplicate"])
def test_bad_raw_rejected(transactions, change):
    if change == "negative": transactions.loc[0, "Transaction power/kwh"] = -1
    if change == "backwards": transactions.loc[0, "End Time"] = "2019-01-01"
    if change == "duplicate": transactions = pd.concat([transactions, transactions.iloc[:1]])
    with pytest.raises(ValueError): aggregate(transactions)


def test_target_row_and_lag_alignment(hourly):
    d = extract_features(hourly)
    row = d.iloc[200]
    assert row.energy_kwh == 200
    assert row.energy_kwh_lag_1 == 199
    assert row.energy_kwh_lag_24 == 176
    assert row.energy_kwh_lag_168 == 32


def test_shift_before_rolling(hourly):
    d = extract_features(hourly)
    assert d.iloc[200].energy_kwh_mean_6 == np.arange(194, 200).mean()
    assert d.iloc[200].energy_kwh_std_24 == pytest.approx(np.arange(176, 200).std(ddof=1))


def test_current_and_future_cannot_change_features(hourly):
    before = extract_features(hourly).iloc[200][FEATURES]
    hourly.loc[200:, TARGETS] = 999999
    after = extract_features(hourly).iloc[200][FEATURES]
    pd.testing.assert_series_equal(before, after)


def test_weather_and_outcomes_excluded(hourly):
    for field in ["temp_c", "humidity", "precip_mm", "end_cause", "is_abnormal", "is_full_charge",
                  "is_user_stop", "is_nonsystem_fault", "is_system_issue", "is_EV_fault", "is_other_fault"]:
        hourly[field] = 999999
    assert not any("weather" in c or "is_" in c or c in hourly.columns[3:] for c in FEATURES)
    original = extract_features(hourly)[FEATURES]
    hourly.iloc[:, 3:] = 0
    pd.testing.assert_frame_equal(original, extract_features(hourly)[FEATURES])


def test_chronology_and_zero_target_overlap():
    t = pd.Series(pd.date_range("2021-01-01", "2021-12-31 23:00:00", freq="h"))
    masks = split_masks(t)
    sets = {s: set(t[m]) for s, m in masks.items()}
    assert sets["train"].isdisjoint(sets["validation"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert max(sets["train"]) < min(sets["validation"]) < min(sets["test"])


def test_scalers_fit_train_only():
    x = np.array([[0., 1.], [2., 3.], [9999., 9999.]])
    sx, sy = fit_scalers(x, x, np.array([True, True, False]))
    assert sx.mean_.tolist() == sy.mean_.tolist() == [1., 2.]
    assert sx.n_samples_seen_ == sy.n_samples_seen_ == 2


def test_lstm_sequence_ends_at_correct_target_row():
    x = np.arange(30).reshape(-1, 1)
    y = np.arange(30) + 100
    a, b = sequence_arrays(x, y, length=24)
    assert len(a) == len(b) == 7
    assert a[0, -1, 0] == 23 and b[0] == 123
    assert a[-1, -1, 0] == 29 and b[-1] == 129


def test_sequence_latest_row_has_only_previous_hour(hourly):
    d = extract_features(hourly).dropna(subset=FEATURES)
    a, b = sequence_arrays(d[FEATURES].to_numpy(), d[TARGETS].to_numpy())
    lag_index = FEATURES.index("energy_kwh_lag_1")
    assert a[0, -1, lag_index] == b[0, 0] - 1


@pytest.mark.parametrize("payload", [{}, {"forecast_origin": "2021-10-01T00:00:00"},
                                      {"forecast_origin": "2021-10-01T00:00:00", "history": []}])
def test_missing_history_no_defaults(payload):
    with pytest.raises(ValueError, match="history"): predict_demand(payload)


@pytest.mark.parametrize("corruption", ["gap", "future", "nan", "negative", "fractional", "aware"])
def test_invalid_real_history_rejected(corruption):
    p = fixture()
    if corruption == "gap": p["history"][1]["timestamp"] = p["history"][0]["timestamp"]
    if corruption == "future": p["forecast_origin"] = p["history"][-1]["timestamp"]
    if corruption == "nan": p["history"][0]["energy_kwh"] = float("nan")
    if corruption == "negative": p["history"][0]["sessions"] = -1
    if corruption == "fractional": p["history"][0]["sessions"] = 1.5
    if corruption == "aware": p["forecast_origin"] += "Z"
    with pytest.raises(ValueError): predict_demand(p)


def test_schema_fixture_prediction_and_version():
    p = fixture()
    jsonschema.validate(p, json.loads((MODULE / "schemas/input.json").read_text()))
    result = predict_demand(p)
    jsonschema.validate(result, json.loads((MODULE / "schemas/output.json").read_text()))
    expected = json.loads((MODULE / "fixtures/expected_response.json").read_text())
    assert result["model_version"] == expected["model_version"]
    assert result["expected_energy_kwh"] == pytest.approx(expected["expected_energy_kwh"], rel=1e-5)
    assert result["expected_sessions"] == pytest.approx(expected["expected_sessions"], rel=1e-5)
    assert isinstance(result["expected_sessions"], float)
    assert result["expected_energy_kwh"] >= 0 and result["expected_sessions"] >= 0


@pytest.mark.parametrize("outside", [False, True])
def test_clean_process_cwd_independence(tmp_path, outside):
    code = "import json;from demand import predict_demand;from pathlib import Path;print(json.dumps(predict_demand(json.loads(Path(" + repr(str(MODULE / "fixtures/request.json")) + ").read_text()))))"
    env = {**os.environ, "PYTHONPATH": str(MODULE)}
    result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path if outside else MODULE.parents[1],
                            env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["model_version"]


def test_bundle_checksums_and_train_scaling_metadata():
    path, meta = load_bundle()
    for filename, digest in json.loads((path / "checksums.json").read_text()).items():
        assert sha256(path / filename) == digest
    scale = json.loads((path / "scaling.json").read_text())
    assert pd.Timestamp(scale["fit_last_target"]) < pd.Timestamp("2021-07-01")
    for c in TARGETS:
        assert meta["champions"][c] == min(meta["results"], key=lambda f: meta["results"][f]["validation"][c]["MAE"])


def test_mlflow_experiment_and_real_run_receipts():
    assert EXPERIMENT == "chargesentinel-demand"
    receipt = json.loads((MODULE / "reports/mlflow_receipt.json").read_text())
    assert receipt["experiment"] == EXPERIMENT
    assert set(receipt["runs"]) == {"Baseline", "XGBoost", "LSTM"}
    for run in receipt["runs"].values():
        assert len(run["run_id"]) == 32
        assert run["bundle_artifact_uri"] == f"runs:/{run['run_id']}/bundle"


def test_smape_both_zero_and_nonnegative_clipping():
    assert metrics(np.array([0., 0.]), np.array([0., -1.])) == {"MAE": 0., "RMSE": 0., "sMAPE": 0.}

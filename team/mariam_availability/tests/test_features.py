"""Run from team/mariam_availability:  pytest tests/test_features.py -v"""
import numpy as np
import pandas as pd
import pytest

from src.features import (build_features, split_by_time, FeatureEncoder, FORBIDDEN,
                          CATEGORICAL, NUMERIC, TRAIN_END, VAL_END)

NON_FEATURE = ("start", "target_min", "is_abnormal")


def toy(n=40):
    """Two chargers at two different sites, sessions in time order, no overlaps."""
    rng = np.random.default_rng(0)
    rows = []
    for cid, loc in [(1, "A"), (2, "B")]:
        t = pd.Timestamp("2021-01-01 08:00")
        for _ in range(n):
            t += pd.Timedelta(minutes=int(rng.integers(20, 200)))
            dur = float(rng.integers(5, 120))
            rows.append(dict(charger_id=cid, location=loc, district="D", start=t,
                             end=t + pd.Timedelta(minutes=dur), duration_min=dur,
                             is_abnormal=int(rng.random() < 0.2)))
            t += pd.Timedelta(minutes=dur)
    return pd.DataFrame(rows)


def feature_cols(f):
    return [c for c in f.columns if c not in NON_FEATURE]


# ---------------------------------------------------------------- leakage
def test_feature_lists_contain_no_forbidden_column():
    assert not (set(CATEGORICAL) | set(NUMERIC)) & FORBIDDEN


def test_end_time_is_target_only_and_never_an_inference_feature():
    features = build_features(toy())
    inference_columns = set(feature_cols(features))
    encoder = FeatureEncoder().fit(features)
    assert "end" not in inference_columns
    assert "end" not in encoder.transform(features).columns


def test_encoder_output_contains_no_forbidden_column():
    f = build_features(toy())
    enc = FeatureEncoder().fit(f)
    assert not set(enc.feature_names) & FORBIDDEN
    assert not set(enc.transform(f).columns) & FORBIDDEN


def test_feature_table_has_no_forbidden_or_outcome_features():
    f = build_features(toy())
    features = set(feature_cols(f))
    assert not features & FORBIDDEN
    assert not {c for c in features if c.startswith("is_")} - {"is_weekend"}


def test_a_sessions_own_outcome_never_leaks_into_its_own_features():
    """Change the duration, end and abnormal flag of session k. Its own features and every
    EARLIER session's features must stay identical. Later sessions may change (that is fine)."""
    d = toy()
    f1 = build_features(d)
    d2 = d.copy()
    k = 15                                              # a session of charger 1
    d2.loc[k, "duration_min"] += 300
    d2.loc[k, "end"] += pd.Timedelta(minutes=300)
    d2.loc[k, "is_abnormal"] = 1 - d2.loc[k, "is_abnormal"]
    f2 = build_features(d2)
    cols = feature_cols(f1)
    pd.testing.assert_frame_equal(f1.loc[:k, cols], f2.loc[:k, cols])          # up to and including k
    pd.testing.assert_frame_equal(f1.loc[40:, cols], f2.loc[40:, cols])        # the other charger is untouched
    assert not f1.loc[k + 1, cols].equals(f2.loc[k + 1, cols]), "test could not detect a change"


def test_history_uses_only_previous_sessions():
    d = toy()
    f = build_features(d)
    # first session of a charger has no history, third has min_periods reached on the 4th
    first = f[f["charger_id"] == 1].iloc[0]
    assert first["sessions_before"] == 0 and first["has_history"] == 0 and np.isnan(first["prev_duration_min"])
    row = f[f["charger_id"] == 1].iloc[5]
    prev = d[d["charger_id"] == 1].iloc[4]["duration_min"]
    assert row["prev_duration_min"] == prev


# ---------------------------------------------------------------- split
def test_split_is_chronological_and_boundaries_are_correct():
    s = pd.DataFrame({"start": [pd.Timestamp(x) for x in [
        "2020-01-05 00:00:00", "2021-06-30 23:59:59", "2021-07-01 00:00:00",
        "2021-09-30 23:59:59", "2021-10-01 00:00:00", "2021-12-31 12:00:00"]]})
    tr, va, te = split_by_time(s)
    assert len(tr) == 2 and len(va) == 2 and len(te) == 2
    assert tr["start"].max() < TRAIN_END <= va["start"].min()
    assert va["start"].max() < VAL_END <= te["start"].min()


# ---------------------------------------------------------------- encoder fitted on train only
def test_unseen_category_becomes_minus_one_and_no_nans():
    f = build_features(toy())
    enc = FeatureEncoder().fit(f[f["charger_id"] == 1])      # never sees charger 2 or site B
    X = enc.transform(f[f["charger_id"] == 2])
    assert (X["charger_id_code"] == -1).all() and (X["location_code"] == -1).all()
    assert not X.isna().any().any()


def test_encoder_has_no_missing_values_across_all_splits():
    f = build_features(toy())
    # Give each partition enough normal sessions to fit numeric medians.
    f["start"] = pd.to_datetime(list(pd.date_range("2021-01-01", periods=30, freq="D")) +
                                 list(pd.date_range("2021-07-01", periods=20, freq="D")) +
                                 list(pd.date_range("2021-10-01", periods=30, freq="D")))
    train, val, test = split_by_time(f)
    enc = FeatureEncoder().fit(train[train["is_abnormal"] == 0])
    assert min(len(train), len(val), len(test)) > 0
    for part in (train, val, test):
        assert not enc.transform(part).isna().any().any()


def test_encoder_state_round_trip():
    f = build_features(toy())
    enc = FeatureEncoder().fit(f)
    enc2 = FeatureEncoder.from_state(enc.get_state())
    pd.testing.assert_frame_equal(enc.transform(f), enc2.transform(f))


# ---------------------------------------------------------------- real data
@pytest.fixture(scope="module")
def real():
    from src.occupancy import load_sessions
    try:
        return build_features(load_sessions())
    except FileNotFoundError:
        pytest.skip("clean dataset not found, run with the data in place")


def test_real_split_sizes_and_order(real):
    tr, va, te = split_by_time(real)
    assert min(len(tr), len(va), len(te)) > 1000
    assert tr["start"].max() < va["start"].min() and va["start"].max() < te["start"].min()
    assert len(tr) + len(va) + len(te) == len(real)


def test_real_encoder_has_no_missing_values(real):
    tr, va, te = split_by_time(real)
    enc = FeatureEncoder().fit(tr[tr["is_abnormal"] == 0])
    for part in (tr, va, te):
        assert not enc.transform(part).isna().any().any()

"""Tests for `muenster_bike_forecast.modeling.lag_features`.

All tests use small, hand-built synthetic data - no live network calls and
no dependency on the real `data/raw/` files. Includes a synthetic data gap
to verify the exact-timestamp lag lookup does not silently pair a row with
the wrong one across it, and a check that rolling features never leak the
current row's own value into its own window.
"""

from __future__ import annotations

import pandas as pd
import pytest

from muenster_bike_forecast.modeling.lag_features import (
    LagFeatureError,
    add_lag_feature,
    add_rolling_feature,
)

# ---------------------------------------------------------------------------
# add_lag_feature
# ---------------------------------------------------------------------------


def test_add_lag_feature_uses_exact_timestamp_not_position() -> None:
    # Station S1 has a gap at 00:30 (missing 15-minute interval). Looking
    # back 30 minutes from 01:00 should hit the missing 00:30 row and get a
    # null lag, NOT be silently paired with the nearest available row.
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S1"],
            "datetime": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 01:00"]
            ),
            "total_count": [10.0, 20.0, 40.0],
        }
    )

    out = add_lag_feature(df, lag=pd.Timedelta(minutes=30), feature_col="lag_30m")

    # 00:15 - 30min = 23:45 (prior day) -> not in data -> null lag.
    assert pd.isna(out.loc[out["datetime"] == "2024-01-01 00:15", "lag_30m"].iloc[0])
    # 01:00 - 30min = 00:30 -> missing -> null lag, not paired with 00:15.
    assert pd.isna(out.loc[out["datetime"] == "2024-01-01 01:00", "lag_30m"].iloc[0])


def test_add_lag_feature_looks_up_exact_earlier_value() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "datetime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:15"]),
            "total_count": [10.0, 20.0],
        }
    )

    out = add_lag_feature(df, lag=pd.Timedelta(minutes=15), feature_col="lag_15m")

    assert out.loc[out["datetime"] == "2024-01-01 00:15", "lag_15m"].iloc[0] == 10.0


def test_add_lag_feature_does_not_cross_station_boundaries() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "datetime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:15"]),
            "total_count": [10.0, 99.0],
        }
    )

    out = add_lag_feature(df, lag=pd.Timedelta(minutes=15), feature_col="lag_15m")

    # S2's 00:15 - 15min = 00:00, which exists only for S1 - must not match.
    assert pd.isna(out.loc[out["station_id"] == "S2", "lag_15m"].iloc[0])


def test_add_lag_feature_raises_on_duplicate_station_timestamp_pairs() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "datetime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00"]),
            "total_count": [10.0, 11.0],
        }
    )
    with pytest.raises(LagFeatureError):
        add_lag_feature(df, lag=pd.Timedelta(minutes=15), feature_col="lag_15m")


def test_add_lag_feature_raises_on_missing_column() -> None:
    df = pd.DataFrame({"station_id": ["S1"], "datetime": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(LagFeatureError):
        add_lag_feature(df, lag=pd.Timedelta(minutes=15), feature_col="lag_15m")


def test_add_lag_feature_returns_valid_empty_result_on_empty_input() -> None:
    # pd.MultiIndex.from_tuples([]) raises TypeError on an empty key list -
    # this must not propagate as a bare, uninformative exception.
    df = pd.DataFrame({"station_id": [], "datetime": [], "total_count": []})

    out = add_lag_feature(df, lag=pd.Timedelta(minutes=15), feature_col="lag_15m")

    assert out.empty
    assert "lag_15m" in out.columns


# ---------------------------------------------------------------------------
# add_rolling_feature
# ---------------------------------------------------------------------------


def test_add_rolling_feature_excludes_current_row() -> None:
    # closed="left" means the window is (t - window, t) - the current row's
    # own value must never appear in its own rolling mean.
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "datetime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:15"]),
            "total_count": [10.0, 999.0],
        }
    )

    out = add_rolling_feature(
        df,
        window=pd.Timedelta(minutes=15),
        feature_col="roll_mean",
        stat="mean",
        min_periods=1,
    )

    assert out.loc[out["datetime"] == "2024-01-01 00:15", "roll_mean"].iloc[0] == 10.0


def test_add_rolling_feature_respects_min_periods() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["S1"],
            "datetime": pd.to_datetime(["2024-01-01 00:00"]),
            "total_count": [10.0],
        }
    )

    out = add_rolling_feature(
        df,
        window=pd.Timedelta(hours=1),
        feature_col="roll_mean",
        stat="mean",
        min_periods=1,
    )

    # No prior data at all within the window -> null, not fabricated.
    assert pd.isna(out["roll_mean"].iloc[0])


def test_add_rolling_feature_does_not_cross_station_boundaries() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S2", "S2"],
            "datetime": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:15"]
            ),
            "total_count": [500.0, 10.0, 20.0],
        }
    )

    out = add_rolling_feature(
        df,
        window=pd.Timedelta(minutes=15),
        feature_col="roll_mean",
        stat="mean",
        min_periods=1,
    )

    # S2's 00:15 rolling mean must only see S2's own 00:00 value (10.0),
    # never S1's 500.0.
    s2_row = out[(out["station_id"] == "S2") & (out["datetime"] == "2024-01-01 00:15")]
    assert s2_row["roll_mean"].iloc[0] == 10.0


def test_add_rolling_feature_raises_on_unsupported_stat() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["S1"],
            "datetime": pd.to_datetime(["2024-01-01 00:00"]),
            "total_count": [10.0],
        }
    )
    with pytest.raises(LagFeatureError):
        add_rolling_feature(
            df, window=pd.Timedelta(hours=1), feature_col="x", stat="median"
        )


def test_add_rolling_feature_raises_on_missing_column() -> None:
    df = pd.DataFrame({"station_id": ["S1"], "datetime": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(LagFeatureError):
        add_rolling_feature(df, window=pd.Timedelta(hours=1), feature_col="x")

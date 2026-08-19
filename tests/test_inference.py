"""Tests for `muenster_bike_forecast.inference`.

All tests use small, hand-built synthetic data - no live network calls and
no dependency on the real `data/raw/`/`models/` files. Bike-count
timestamps are chosen in June (no DST transition in Germany that month) so
their UTC-localized equivalents are simple to compute by hand for building
matching synthetic weather rows.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from muenster_bike_forecast.data.join import localize_bike_timestamps
from muenster_bike_forecast.inference import (
    FEATURE_COLS,
    InferenceError,
    assemble_feature_history,
    latest_feature_row,
    months_needed,
    predict_24h_ahead,
    predict_forecast_curve,
)

STATION_ID = 12345


def _raw_bike_df() -> pd.DataFrame:
    """Three daily rows (10 June - 12 June 2024, 08:00 local) for one station."""
    return pd.DataFrame(
        {
            "station_id": [str(STATION_ID)] * 3,
            "datetime": pd.to_datetime(
                ["2024-06-10 08:00", "2024-06-11 08:00", "2024-06-12 08:00"]
            ),
            "12345 (Test)": [10, 20, 30],
            "12345-status": [0, 0, 0],
        }
    )


def _weather_wide_df(bike_datetimes: pd.Series) -> pd.DataFrame:
    """Hourly weather rows exactly matching each bike timestamp's UTC equivalent."""
    localized = localize_bike_timestamps(pd.DataFrame({"datetime": bike_datetimes}))[
        "timestamp"
    ]
    return pd.DataFrame(
        {
            "station_id": ["01766"] * len(localized),
            "timestamp": localized,
            "air_temperature_c": [15.0, 16.0, 17.0][: len(localized)],
            "relative_humidity_pct": [70.0, 71.0, 72.0][: len(localized)],
            "precipitation_mm": [0.0, 0.0, 0.2][: len(localized)],
            "wind_speed_ms": [3.0, 3.5, 4.0][: len(localized)],
        }
    )


def _public_holidays_df() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime([]), "name": pd.Series(dtype="object")})


def _school_holidays_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_date": pd.to_datetime([]),
            "end_date": pd.to_datetime([]),
        }
    )


def _ratio_table(station_id: int = STATION_ID, ratio: float = 0.65) -> pd.DataFrame:
    return pd.DataFrame({"station_id": [station_id], "weekend_weekday_ratio": [ratio]})


# ---------------------------------------------------------------------------
# assemble_feature_history
# ---------------------------------------------------------------------------


def test_assemble_feature_history_computes_total_count_and_features() -> None:
    raw = _raw_bike_df()
    weather = _weather_wide_df(raw["datetime"])

    out = assemble_feature_history(
        raw, weather, _public_holidays_df(), _school_holidays_df(), _ratio_table()
    )

    assert list(out["total_count"]) == [10.0, 20.0, 30.0]
    # 12 June's lag_1d looks up 11 June's total_count exactly.
    last_row = out.iloc[-1]
    assert last_row["lag_1d"] == 20.0
    # lag_1w has no data 7 days back within this short window -> null, not fabricated.
    assert pd.isna(last_row["lag_1w"])
    for col in FEATURE_COLS:
        assert col in out.columns


def test_assemble_feature_history_attaches_weather_with_prefix() -> None:
    raw = _raw_bike_df()
    weather = _weather_wide_df(raw["datetime"])

    out = assemble_feature_history(
        raw, weather, _public_holidays_df(), _school_holidays_df(), _ratio_table()
    )

    assert list(out["weather_air_temperature_c"]) == [15.0, 16.0, 17.0]


def test_assemble_feature_history_casts_station_id_to_match_training_dtype() -> None:
    raw = _raw_bike_df()  # station_id passed in as str
    weather = _weather_wide_df(raw["datetime"])

    out = assemble_feature_history(
        raw, weather, _public_holidays_df(), _school_holidays_df(), _ratio_table()
    )

    assert out["station_id"].dtype == np.dtype("int64")
    assert out["station_id"].iloc[0] == STATION_ID


def test_assemble_feature_history_raises_on_empty_input() -> None:
    with pytest.raises(InferenceError):
        assemble_feature_history(
            pd.DataFrame(),
            _weather_wide_df(pd.Series(dtype="datetime64[ns]")),
            _public_holidays_df(),
            _school_holidays_df(),
            _ratio_table(),
        )


def test_assemble_feature_history_raises_on_non_numeric_station_id() -> None:
    # `_validate_station_id` in `data.bike_counts` allows any
    # `[A-Za-z0-9_-]+` id (it only guards against path traversal), so a
    # station id like "ABC-01" can reach here even though it can't be cast
    # to the int64 dtype the production model expects.
    raw = _raw_bike_df()
    raw["station_id"] = "ABC-01"
    weather = _weather_wide_df(raw["datetime"])

    with pytest.raises(InferenceError):
        assemble_feature_history(
            raw, weather, _public_holidays_df(), _school_holidays_df(), _ratio_table()
        )


def test_assemble_feature_history_raises_on_out_of_range_numeric_station_id() -> None:
    # A purely numeric id can still be too large for int64 - `astype`
    # raises `OverflowError` for this case, not `ValueError`/`TypeError`.
    raw = _raw_bike_df()
    raw["station_id"] = "999999999999999999999999999"
    weather = _weather_wide_df(raw["datetime"])

    with pytest.raises(InferenceError):
        assemble_feature_history(
            raw, weather, _public_holidays_df(), _school_holidays_df(), _ratio_table()
        )


def test_assemble_feature_history_raises_when_station_missing_from_ratio_table() -> (
    None
):
    raw = _raw_bike_df()
    weather = _weather_wide_df(raw["datetime"])

    with pytest.raises(InferenceError):
        assemble_feature_history(
            raw,
            weather,
            _public_holidays_df(),
            _school_holidays_df(),
            _ratio_table(station_id=99999),
        )


# ---------------------------------------------------------------------------
# latest_feature_row
# ---------------------------------------------------------------------------


def test_latest_feature_row_picks_most_recent_non_null_total_count() -> None:
    history = pd.DataFrame(
        {
            "station_id": [STATION_ID, STATION_ID, STATION_ID],
            "datetime": pd.to_datetime(
                ["2024-06-10 08:00", "2024-06-11 08:00", "2024-06-12 08:00"]
            ),
            "total_count": [10.0, 20.0, np.nan],
        }
    )

    row = latest_feature_row(history, station_id=STATION_ID)

    assert row["total_count"] == 20.0


def test_latest_feature_row_matches_station_after_dtype_cast() -> None:
    history = pd.DataFrame(
        {
            "station_id": [STATION_ID],
            "datetime": pd.to_datetime(["2024-06-10 08:00"]),
            "total_count": [10.0],
        }
    )

    row = latest_feature_row(history, station_id=str(STATION_ID))

    assert row["total_count"] == 10.0


def test_latest_feature_row_raises_when_all_null() -> None:
    history = pd.DataFrame(
        {
            "station_id": [STATION_ID],
            "datetime": pd.to_datetime(["2024-06-10 08:00"]),
            "total_count": [np.nan],
        }
    )
    with pytest.raises(InferenceError):
        latest_feature_row(history, station_id=STATION_ID)


# ---------------------------------------------------------------------------
# predict_24h_ahead
# ---------------------------------------------------------------------------


class _StubModel:
    """Records the DataFrame it was called with and returns a fixed value."""

    def __init__(self, value: float) -> None:
        self.value = value
        self.received: pd.DataFrame | None = None

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.received = X
        return np.array([self.value])


def test_predict_24h_ahead_passes_exact_feature_cols_and_returns_float() -> None:
    row = pd.Series({col: 1.0 for col in FEATURE_COLS} | {"extra_col": "ignored"})
    model = _StubModel(42.5)

    result = predict_24h_ahead(model, row)

    assert result == 42.5
    assert isinstance(result, float)
    assert list(model.received.columns) == FEATURE_COLS


def test_predict_24h_ahead_raises_on_missing_columns() -> None:
    row = pd.Series({"total_count": 1.0})
    with pytest.raises(InferenceError):
        predict_24h_ahead(_StubModel(0.0), row)


# ---------------------------------------------------------------------------
# predict_forecast_curve
# ---------------------------------------------------------------------------


class _SequentialStubModel:
    """Returns 0, 1, 2, ... for however many rows it's asked to predict."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.arange(len(X), dtype=float)


def _hourly_history(n: int = 30) -> pd.DataFrame:
    datetimes = pd.date_range("2024-06-01 00:00", periods=n, freq="h")
    history = pd.DataFrame({col: [1.0] * n for col in FEATURE_COLS})
    history["station_id"] = STATION_ID
    history["datetime"] = datetimes
    history["total_count"] = 1.0
    return history


def test_predict_forecast_curve_covers_exactly_the_horizon_window() -> None:
    history = _hourly_history(n=30)
    current_datetime = history["datetime"].iloc[-1]
    horizon = pd.Timedelta(hours=24)

    curve = predict_forecast_curve(
        _SequentialStubModel(), history, STATION_ID, current_datetime, horizon=horizon
    )

    assert len(curve) == 24
    assert list(curve["target_datetime"]) == sorted(curve["target_datetime"])
    assert curve["target_datetime"].min() > current_datetime
    assert curve["target_datetime"].max() == current_datetime + horizon
    # Predictions come back in the same (ascending) order as their source rows.
    assert list(curve["predicted_total_count"]) == list(range(24))


def test_predict_forecast_curve_raises_when_window_is_empty() -> None:
    history = _hourly_history(n=30)
    far_future = history["datetime"].iloc[-1] + pd.Timedelta(days=100)

    with pytest.raises(InferenceError):
        predict_forecast_curve(_SequentialStubModel(), history, STATION_ID, far_future)


def test_predict_forecast_curve_raises_on_missing_columns() -> None:
    history = pd.DataFrame(
        {
            "station_id": [STATION_ID],
            "datetime": pd.to_datetime(["2024-06-01 00:00"]),
            "total_count": [1.0],
        }
    )
    with pytest.raises(InferenceError):
        predict_forecast_curve(
            _SequentialStubModel(), history, STATION_ID, history["datetime"].iloc[-1]
        )


# ---------------------------------------------------------------------------
# months_needed
# ---------------------------------------------------------------------------


def test_months_needed_spans_a_month_boundary() -> None:
    result = months_needed(date(2024, 1, 5), lookback=pd.Timedelta(days=10))
    assert result == [(2023, 12), (2024, 1)]


def test_months_needed_within_a_single_month() -> None:
    result = months_needed(date(2024, 3, 15), lookback=pd.Timedelta(days=10))
    assert result == [(2024, 3)]

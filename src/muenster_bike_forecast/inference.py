"""Live 24h-ahead forecasting: turns freshly fetched data into a prediction.

This module assembles exactly the feature row `notebooks/17_final_production_model.ipynb`
trained `models/production_random_forest.joblib` on, but from a short recent
window of live data rather than the full historical `model_table.csv`. Its
functions are pure transforms over already-fetched DataFrames; fetching the
raw bike-count/weather/calendar data (via `data.bike_counts`, `data.weather`,
`data.calendar`) and calling the model itself are left to the caller (the
Streamlit app), consistent with this project's convention of keeping I/O out
of reusable transform code.

`FEATURE_COLS` (and the `LAG_SPECS`/`ROLLING_SPECS`/`NUMERIC_FEATURES`/
`CATEGORICAL_FEATURES` it is built from) must stay in sync with notebook 17's
own definitions — they are duplicated here because notebooks are not
importable modules. If the production model is ever retrained with a
different feature set, update both places.
"""

from __future__ import annotations

from datetime import date
from typing import Final

import pandas as pd

from muenster_bike_forecast.data.join import join_station_weather, localize_bike_timestamps
from muenster_bike_forecast.modeling.feature_followups import add_weekend_weekday_ratio_feature
from muenster_bike_forecast.modeling.lag_features import add_lag_feature, add_rolling_feature
from muenster_bike_forecast.modeling.model_table import add_calendar_features, compute_total_count

LAG_SPECS: Final[dict[str, pd.Timedelta]] = {
    "lag_1h": pd.Timedelta(hours=1),
    "lag_1d": pd.Timedelta(days=1),
    "lag_1w": pd.Timedelta(weeks=1),
}
ROLLING_SPECS: Final[dict[str, pd.Timedelta]] = {
    "rolling_mean_2h": pd.Timedelta(hours=2),
    "rolling_mean_24h": pd.Timedelta(hours=24),
}
HISTORY_FEATURE_COLS: Final[list[str]] = list(LAG_SPECS) + list(ROLLING_SPECS)

CATEGORICAL_FEATURES: Final[list[str]] = [
    "station_id",
    "hour",
    "day_of_week",
    "month",
    "is_public_holiday",
    "is_school_holiday",
    "is_lecture_period",
]
NUMERIC_FEATURES: Final[list[str]] = [
    "total_count",
    "weather_air_temperature_c",
    "weather_relative_humidity_pct",
    "weather_precipitation_mm",
    "weather_wind_speed_ms",
    *HISTORY_FEATURE_COLS,
    "weekend_weekday_ratio",
]
FEATURE_COLS: Final[list[str]] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Deliberately more than one calendar month (31 days max) so that
# `months_needed(as_of, MIN_HISTORY_LOOKBACK)` always spans at least two
# distinct calendar months regardless of `as_of`'s day-of-month. A tighter
# bound (e.g. 10 days) can fetch only the current month if `as_of` is late
# in it, which silently shortens the available history whenever the source
# itself has stalled (its real latest data can then be *earlier* in that
# same month, leaving too little trailing history for lag_1w or a 7-day
# chart window). Callers fetching live bike-count history should cover at
# least this much time before `as_of`.
MIN_HISTORY_LOOKBACK: Final[pd.Timedelta] = pd.Timedelta(days=35)

# As-of joining bike-count rows to hourly weather: a row further than this
# past the nearest weather reading is left with null weather columns rather
# than matched to stale data (same tolerance as notebooks 03/06/17).
WEATHER_JOIN_TOLERANCE: Final[pd.Timedelta] = pd.Timedelta(hours=2)


class InferenceError(Exception):
    """Raised when a live feature row cannot be assembled or predicted from.

    Covers empty/unusable input data, a station missing from the
    weekend/weekday ratio table, and a station with no evaluable (non-null
    `total_count`) row to predict from.
    """


def months_needed(as_of: date, lookback: pd.Timedelta = MIN_HISTORY_LOOKBACK) -> list[tuple[int, int]]:
    """Lists the distinct calendar (year, month) pairs a lookback window spans.

    Pure helper for callers deciding which monthly bike-count files to fetch
    (e.g. via `data.bike_counts.fetch_station_month`) to cover
    ``[as_of - lookback, as_of]`` without fetching a station's entire
    history.

    Args:
        as_of: Last date (inclusive) the window should cover.
        lookback: How far back before `as_of` the window should reach.

    Returns:
        Sorted list of unique ``(year, month)`` tuples, oldest first.
    """
    start = pd.Timestamp(as_of) - lookback
    end = pd.Timestamp(as_of)
    months = pd.period_range(start=start.strftime("%Y-%m"), end=end.strftime("%Y-%m"), freq="M")
    return [(int(period.year), int(period.month)) for period in months]


def assemble_feature_history(
    raw_bike_df: pd.DataFrame,
    weather_wide_df: pd.DataFrame,
    public_holidays_df: pd.DataFrame,
    school_holidays_df: pd.DataFrame,
    ratio_table: pd.DataFrame,
) -> pd.DataFrame:
    """Builds the full `FEATURE_COLS` history for one station's recent data.

    Mirrors, for a short live window, exactly what
    ``notebooks/06_baseline_model.ipynb``'s ``_load_station_slim`` plus
    notebooks 08-17's lag/rolling/calendar/ratio steps do for the full
    historical table: coalesce+sum count channels into `total_count`,
    as-of join hourly weather, add lag/rolling history features, add
    calendar features, and broadcast the static per-station
    `weekend_weekday_ratio`.

    Args:
        raw_bike_df: One station's raw rows, as returned by
            `data.bike_counts.fetch_station_month`/`fetch_station_data`
            (columns: ``station_id``, ``datetime``, plus per-channel count/
            status columns). ``station_id`` may be `str` or numeric; it is
            cast to `int64` here to match the dtype the production model
            was trained on (`data/raw/model_table/model_table.csv`, read via
            `pandas.read_csv` with no dtype override).
        weather_wide_df: Combined hourly weather, as returned by
            `data.join.combine_weather_parameters`.
        public_holidays_df: As returned by `data.calendar.public_holidays`.
        school_holidays_df: As returned by
            `data.calendar.fetch_school_holidays`/`load_school_holidays`.
        ratio_table: Static per-station `weekend_weekday_ratio` lookup, as
            returned by `modeling.feature_followups.compute_weekend_weekday_ratio`
            or loaded from the committed ``models/weekend_weekday_ratio.csv``.

    Returns:
        DataFrame with columns ``station_id``, ``datetime``, and every
        column in `FEATURE_COLS`, one row per distinct bike-count timestamp
        in `raw_bike_df`, sorted by ``datetime``. Lag/rolling features are
        null for timestamps near the start of `raw_bike_df`'s own coverage
        (not enough history within the fetched window), consistent with how
        the training pipeline handles the same near-start-of-coverage case.

    Raises:
        InferenceError: if `raw_bike_df` is empty, or its station has no
            matching row in `ratio_table`.
    """
    if raw_bike_df.empty:
        raise InferenceError(
            "raw_bike_df is empty; no bike-count data to build features from."
        )

    working = raw_bike_df.copy()
    working["station_id"] = working["station_id"].astype("int64")

    localized = localize_bike_timestamps(working)
    joined = join_station_weather(localized, weather_wide_df, tolerance=WEATHER_JOIN_TOLERANCE)
    total_count = compute_total_count(joined)

    weather_columns = [
        column
        for column in joined.columns
        if column.startswith("weather_")
        and column not in ("weather_station_id", "weather_timestamp")
    ]
    slim = joined[["station_id", "datetime", *weather_columns]].copy()
    slim["total_count"] = total_count

    for feature_col, lag in LAG_SPECS.items():
        slim = add_lag_feature(slim, lag=lag, feature_col=feature_col)
    for feature_col, window in ROLLING_SPECS.items():
        slim = add_rolling_feature(slim, window=window, feature_col=feature_col, stat="mean")

    slim = add_calendar_features(slim, public_holidays_df, school_holidays_df)

    try:
        slim = add_weekend_weekday_ratio_feature(slim, ratio_table)
    except Exception as exc:  # noqa: BLE001 - re-raised as this module's own error type
        raise InferenceError(
            f"Could not attach weekend_weekday_ratio for station "
            f"{working['station_id'].iloc[0]!r}: {exc}"
        ) from exc

    return slim.sort_values("datetime").reset_index(drop=True)


def latest_feature_row(feature_history_df: pd.DataFrame, station_id: int) -> pd.Series:
    """Picks the most recent fully-evaluable row to predict "now" from.

    The production model predicts `total_count` 24h ahead of a row's own
    timestamp using that row's own (current) feature values, so the most
    recent row with a non-null `total_count` represents "right now" for a
    live forecast.

    Args:
        feature_history_df: As returned by `assemble_feature_history`.
        station_id: Station to select (matched after casting to `int64`,
            consistent with `assemble_feature_history`).

    Returns:
        The selected row, as a `pandas.Series`.

    Raises:
        InferenceError: if `feature_history_df` has no row for `station_id`
            with a non-null `total_count`.
    """
    station_rows = feature_history_df.loc[
        feature_history_df["station_id"] == int(station_id)
    ]
    evaluable = station_rows.dropna(subset=["total_count"])
    if evaluable.empty:
        raise InferenceError(
            f"No row with a non-null total_count for station {station_id!r}; "
            "cannot build a current feature row to forecast from."
        )
    return evaluable.sort_values("datetime").iloc[-1]


def predict_forecast_curve(
    model: object,
    feature_history_df: pd.DataFrame,
    station_id: int,
    current_datetime: pd.Timestamp,
    horizon: pd.Timedelta = pd.Timedelta(hours=24),
) -> pd.DataFrame:
    """Predicts a continuous forecast curve for the next `horizon`.

    The production model always maps one row's own (current) features to
    `total_count` exactly `horizon` after that row's own timestamp. Running
    it on every available row in the `horizon` window *before*
    `current_datetime` therefore yields one real, independently-valid
    24h-ahead prediction per point spanning
    ``(current_datetime, current_datetime + horizon]`` — a genuine
    continuous forecast curve for "the next 24 hours from now", built
    entirely from single valid hops on real historical data rather than
    chaining the model's own predictions back into itself (which was not
    how it was trained or validated).

    Args:
        model: A fitted scikit-learn `Pipeline` (or any object exposing
            `.predict`).
        feature_history_df: As returned by `assemble_feature_history`.
        station_id: Station to select (matched after casting to `int64`).
        current_datetime: The "now" timestamp the curve is anchored to —
            typically the timestamp of `latest_feature_row`'s row.
        horizon: How far ahead each row's own target lies (must match what
            the model was trained to predict; 24h for the production
            model).

    Returns:
        DataFrame with columns ``source_datetime`` (each row's own
        timestamp), ``target_datetime`` (``source_datetime + horizon``),
        and ``predicted_total_count``, sorted by ``target_datetime``.

    Raises:
        InferenceError: if there is no row with a non-null `total_count`
            for `station_id` in ``(current_datetime - horizon,
            current_datetime]``, or `feature_history_df` is missing any of
            `FEATURE_COLS`.
    """
    missing = [col for col in FEATURE_COLS if col not in feature_history_df.columns]
    if missing:
        raise InferenceError(f"feature_history_df is missing column(s): {missing}.")

    station_rows = feature_history_df.loc[
        feature_history_df["station_id"] == int(station_id)
    ]
    window = station_rows.loc[
        (station_rows["datetime"] > current_datetime - horizon)
        & (station_rows["datetime"] <= current_datetime)
    ]
    evaluable = window.dropna(subset=["total_count"]).sort_values("datetime")
    if evaluable.empty:
        raise InferenceError(
            f"No row with a non-null total_count for station {station_id!r} in "
            f"the {horizon} before {current_datetime}; cannot build a forecast curve."
        )

    predictions = model.predict(evaluable[FEATURE_COLS])
    curve = pd.DataFrame(
        {
            "source_datetime": evaluable["datetime"].to_numpy(),
            "target_datetime": (evaluable["datetime"] + horizon).to_numpy(),
            "predicted_total_count": predictions,
        }
    )
    return curve.sort_values("target_datetime").reset_index(drop=True)


def predict_24h_ahead(model: object, feature_row: pd.Series) -> float:
    """Runs the production model on one feature row.

    Args:
        model: A fitted scikit-learn `Pipeline` (or any object exposing
            `.predict`), as loaded from
            ``models/production_random_forest.joblib``.
        feature_row: A row containing at least every column in
            `FEATURE_COLS`, e.g. as returned by `latest_feature_row`.

    Returns:
        The predicted `total_count` 24 hours after `feature_row`'s own
        timestamp.

    Raises:
        InferenceError: if `feature_row` is missing any of `FEATURE_COLS`.
    """
    missing = [col for col in FEATURE_COLS if col not in feature_row.index]
    if missing:
        raise InferenceError(f"feature_row is missing column(s): {missing}.")

    X = pd.DataFrame([feature_row[FEATURE_COLS]])
    prediction = model.predict(X)
    return float(prediction[0])

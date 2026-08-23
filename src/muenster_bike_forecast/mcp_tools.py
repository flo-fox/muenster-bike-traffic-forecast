"""Reusable logic behind the local MCP forecast-query server.

Kept separate from `scripts/mcp_forecast_server.py` (the thin MCP-protocol
entry point) so these functions are directly unit-testable and importable
without spinning up a server - same "orchestration script + reusable
module" split as `daily_report.py`/`scripts/send_daily_email.py`.

Deliberately does NOT import `dashboard_common` (it imports `streamlit`,
irrelevant/unavailable outside the dashboard process) or
`scripts/send_daily_email` (a one-shot script, not meant to be imported,
and pulls in email/Anthropic-API dependencies this server doesn't need).
Instead this module duplicates the small "fetch one station's raw
history" helper and fetch-error tuple, matching the same deliberate
decoupling `send_daily_email.py` already does for the same reason (see
its own `SCRIPT_FETCH_ERRORS`/`fetch_station_raw_history` docstrings).
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Final

import joblib
import pandas as pd

from muenster_bike_forecast import daily_report, inference
from muenster_bike_forecast.data.bike_counts import (
    BikeCountDataError,
    Station,
    fetch_station_month,
    list_stations,
)
from muenster_bike_forecast.data.calendar import (
    DEFAULT_PUBLIC_HOLIDAY_SUBDIV,
    SchoolHolidayFetchError,
    SchoolHolidaySchemaError,
    fetch_school_holidays,
    public_holidays,
)
from muenster_bike_forecast.data.join import JoinError, combine_weather_parameters
from muenster_bike_forecast.data.semester_dates import SemesterDateRangeError
from muenster_bike_forecast.data.weather import (
    PARAMETER_SPECS,
    WeatherFetchError,
    WeatherSchemaError,
    fetch_hourly_weather,
)
from muenster_bike_forecast.modeling.lag_features import LagFeatureError
from muenster_bike_forecast.modeling.model_table import ModelTableError

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent.parent
MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "production_lightgbm.joblib"

# Same exception family as dashboard_common.FETCH_ERRORS / send_daily_email's
# SCRIPT_FETCH_ERRORS, duplicated for the same "don't import a module with
# an unrelated runtime dependency" reason - see module docstring.
MCP_FETCH_ERRORS: Final[tuple[type[Exception], ...]] = (
    BikeCountDataError,
    WeatherFetchError,
    WeatherSchemaError,
    SchoolHolidayFetchError,
    SchoolHolidaySchemaError,
    SemesterDateRangeError,
    JoinError,
    ModelTableError,
    LagFeatureError,
    inference.InferenceError,
)


class McpToolError(Exception):
    """Raised for any tool-facing failure (unknown station, fetch/model
    error, etc.) - MCP tool functions catch `MCP_FETCH_ERRORS` and
    station-lookup failures and re-raise as this one type, so
    `scripts/mcp_forecast_server.py` only needs to translate a single
    exception family into an MCP tool error response.
    """


@lru_cache(maxsize=1)
def _load_model() -> object:
    """Loads the committed production model once per server process.

    A plain `functools.lru_cache`, not `streamlit.cache_resource` - this
    process has no Streamlit runtime. The model never changes without a
    server restart, so caching for the process lifetime is correct.
    """
    return joblib.load(MODEL_PATH)


def _parse_as_of(as_of: str | None) -> date:
    """Parses an MCP client's optional ``as_of`` argument into a `date`.

    Args:
        as_of: ``YYYY-MM-DD`` string, or `None` to mean today.

    Raises:
        McpToolError: if `as_of` is not `None` and not a valid ISO date -
            this is client-supplied input (unlike the rest of the
            pipeline's dates, which come from Streamlit widgets/cron), so
            a malformed value must surface as a clean tool error rather
            than an unhandled `ValueError`.
    """
    if as_of is None:
        return date.today()
    try:
        return date.fromisoformat(as_of)
    except ValueError as exc:
        raise McpToolError(
            f"Invalid as_of date {as_of!r}; expected YYYY-MM-DD format."
        ) from exc


def _find_station(stations: list[Station], station_id: str) -> Station:
    """Looks up one station by id in an already-fetched station list.

    Args:
        stations: As returned by `list_stations_tool`/`data.bike_counts.list_stations`.
        station_id: Station id to find.

    Raises:
        McpToolError: if no station in `stations` has this id - message
            lists the valid ids, so a typo'd id is obvious rather than a
            generic "not found".
    """
    for station in stations:
        if station.station_id == station_id:
            return station
    valid_ids = ", ".join(sorted(s.station_id for s in stations))
    raise McpToolError(
        f"Unknown station_id {station_id!r}. Known station ids: {valid_ids}."
    )


def fetch_station_raw_history(station: Station, as_of: date) -> pd.DataFrame:
    """Fetches one station's raw bike-count rows for the last ~35 days.

    Duplicated from `scripts/send_daily_email.py`'s identical helper
    (itself already a deliberate duplicate of `dashboard_common.
    build_forecast`'s inline fetch logic) - see module docstring for why
    this module keeps its own copy rather than importing either.

    Args:
        station: Station to fetch.
        as_of: Last date (inclusive) the fetched window should cover.

    Returns:
        Concatenated raw rows across every needed month.

    Raises:
        inference.InferenceError: if no month in the window returned any
            data.
    """
    frames = []
    for year, month in inference.months_needed(as_of):
        if year < station.start_year:
            continue
        frame = fetch_station_month(station.station_id, year, month)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise inference.InferenceError(
            f"No recent bike-count data available for {station.name} "
            f"({station.station_id}) in the last "
            f"{inference.MIN_HISTORY_LOOKBACK.days} days."
        )
    return pd.concat(frames, ignore_index=True)


def _fetch_shared_context(
    as_of: date,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetches the weather/calendar context every tool call needs.

    Returns:
        `(weather_wide_df, public_holidays_df, school_holidays_df)`.
    """
    weather_wide_df = combine_weather_parameters(
        {
            parameter: fetch_hourly_weather(parameter, period="recent")
            for parameter in PARAMETER_SPECS
        }
    )
    public_holidays_df = public_holidays(
        as_of.year - 1, as_of.year, subdiv=DEFAULT_PUBLIC_HOLIDAY_SUBDIV
    )
    school_holidays_df = fetch_school_holidays(as_of.year - 1, as_of.year)
    return weather_wide_df, public_holidays_df, school_holidays_df


def list_stations_tool() -> list[dict[str, object]]:
    """Lists every bike-counting station this project has live data for.

    Returns:
        One dict per station with keys ``station_id`` (str), ``name``
        (str), and ``start_year`` (int, first year the source repo has
        data for that station).

    Raises:
        McpToolError: if the station index cannot be fetched.
    """
    try:
        stations = list_stations()
    except MCP_FETCH_ERRORS as exc:
        raise McpToolError(f"Could not fetch the station list: {exc}") from exc
    return [
        {
            "station_id": station.station_id,
            "name": station.name,
            "start_year": station.start_year,
        }
        for station in stations
    ]


def get_forecast_tool(station_id: str, as_of: str | None = None) -> dict[str, object]:
    """Gets a live 24h-ahead bike-traffic forecast for one station.

    Fetches fresh bike-count/weather/calendar data and runs the committed
    production model (`models/production_lightgbm.joblib`) - the exact
    same live-inference pipeline the dashboard's Station forecast page
    uses (`inference.assemble_feature_history`/`predict_24h_ahead`/
    `predict_forecast_curve`).

    Args:
        station_id: Station to forecast (see `list_stations_tool` for
            valid ids).
        as_of: Date (``YYYY-MM-DD``) to anchor the live-data fetch window
            to; defaults to today if omitted. This does not let you query
            an arbitrary *past* forecast - see `get_actual_vs_predicted_tool`'s
            docstring for why this project has no forecast-history
            storage at all.

    Returns:
        Dict with keys ``station_id``, ``station_name``, ``as_of_datetime``
        (ISO timestamp of the most recent bike-count reading used),
        ``predicted_total_next_24h`` (rolling next-24h total),
        ``peak_datetime``/``peak_value`` (when/how high the forecast
        curve peaks), and ``actual_total_last_24h`` (observed total over
        the last 24h, for context).

    Raises:
        McpToolError: if `station_id` is unknown, or live data/the model
            can't be fetched/loaded/scored for this station right now.
    """
    as_of_date = _parse_as_of(as_of)
    try:
        stations = list_stations()
        station = _find_station(stations, station_id)
        raw_bike_df = fetch_station_raw_history(station, as_of_date)
        weather_wide_df, public_holidays_df, school_holidays_df = _fetch_shared_context(
            as_of_date
        )
        history = inference.assemble_feature_history(
            raw_bike_df, weather_wide_df, public_holidays_df, school_holidays_df
        )
        current_row = inference.latest_feature_row(history, station.station_id)
        model = _load_model()
        forecast_curve = inference.predict_forecast_curve(
            model, history, station.station_id, current_row["datetime"]
        )
        summary = inference.summarize_forecast_curve(forecast_curve)
        window_rows = inference.select_window_rows(
            history, station.station_id, current_row["datetime"]
        )
        actual_total_24h = float(window_rows["total_count"].sum())
    except McpToolError:
        raise
    except MCP_FETCH_ERRORS as exc:
        raise McpToolError(
            f"Could not build a forecast for station {station_id!r}: {exc}"
        ) from exc
    except FileNotFoundError as exc:
        raise McpToolError(
            "The production model artifact is missing on this machine; "
            "cannot serve forecasts until it is restored."
        ) from exc

    return {
        "station_id": station.station_id,
        "station_name": station.name,
        "as_of_datetime": current_row["datetime"].isoformat(),
        "predicted_total_next_24h": summary.total_predicted_count,
        "peak_datetime": summary.peak_datetime.isoformat(),
        "peak_value": summary.peak_value,
        "actual_total_last_24h": actual_total_24h,
    }


def get_actual_vs_predicted_tool(
    station_id: str, as_of: str | None = None
) -> dict[str, object]:
    """Compares yesterday's predicted traffic total to what actually happened.

    Wraps `daily_report.build_station_report` - the same accuracy check
    the daily forecast-accuracy email sends. **Important limitation**:
    this project has no forecast-history storage (see `daily_report.py`'s
    own module docstring on its "recompute, not persist" design) - there
    is no way to query an arbitrary past date's accuracy. This always
    compares "yesterday" (the 24h window ending at the most recent
    available bike-count reading) against the actual total over that
    same window, reconstructed fresh from live data every call. `as_of`
    only controls which live-data window is fetched (defaults to today),
    not which historical date is being checked.

    Args:
        station_id: Station to check (see `list_stations_tool` for valid
            ids).
        as_of: Date (``YYYY-MM-DD``) to anchor the live-data fetch window
            to; defaults to today if omitted.

    Returns:
        Dict with keys ``station_id``, ``station_name``, ``now_datetime``
        (ISO timestamp the comparison window ends at), ``predicted_total``,
        ``actual_total``, ``abs_error``, ``pct_diff`` (all `float | None`
        together - `None` if the ~24h-back comparison couldn't be
        resolved, e.g. not enough trailing history yet), and
        ``unresolved_reason`` (explains a `None` result, or `None` if
        resolved).

    Raises:
        McpToolError: if `station_id` is unknown, or live data/the model
            can't be fetched/loaded/scored for this station right now.
    """
    as_of_date = _parse_as_of(as_of)
    try:
        stations = list_stations()
        station = _find_station(stations, station_id)
        raw_bike_df = fetch_station_raw_history(station, as_of_date)
        weather_wide_df, public_holidays_df, school_holidays_df = _fetch_shared_context(
            as_of_date
        )
        model = _load_model()
        report = daily_report.build_station_report(
            station,
            model,
            public_holidays_df,
            school_holidays_df,
            weather_wide_df,
            raw_bike_df,
        )
    except McpToolError:
        raise
    except MCP_FETCH_ERRORS as exc:
        raise McpToolError(
            f"Could not build an accuracy report for station {station_id!r}: {exc}"
        ) from exc
    except FileNotFoundError as exc:
        raise McpToolError(
            "The production model artifact is missing on this machine; "
            "cannot serve forecasts until it is restored."
        ) from exc

    return {
        "station_id": report.station_id,
        "station_name": report.station_name,
        "now_datetime": report.now_datetime.isoformat(),
        "predicted_total": report.predicted_total,
        "actual_total": report.actual_total,
        "abs_error": report.abs_error,
        "pct_diff": report.pct_diff,
        "unresolved_reason": report.unresolved_reason,
    }

"""Shared caching, live-data fetching, and chart-building for the dashboard.

Split out of `app.py` so the same cached resources (model, station list,
weather, calendar data) and the same fetch/predict/chart logic are reused
across every page of the multipage app (`pages/`), rather than duplicated
or re-fetched per page. `app.py` is the only entry point Streamlit runs
directly; it sets up `sys.path` before any page (and therefore this
module) is imported.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
from muenster_bike_forecast import inference

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "production_random_forest.joblib"
RATIO_TABLE_PATH = PROJECT_ROOT / "models" / "weekend_weekday_ratio.csv"
STATION_LOCATIONS_PATH = PROJECT_ROOT / "data" / "processed" / "station_locations.csv"

# The source repo publishes new data roughly daily (verified via its own
# commit history); flag data older than this as stale rather than silently
# presenting it as fresh.
STALENESS_WARNING_THRESHOLD = pd.Timedelta(hours=36)

# How much recent history the forecast chart displays.
CHART_HISTORY_WINDOW = pd.Timedelta(days=7)

# Same reference point/zoom as notebooks/07_descriptive_analysis.ipynb's
# station map (Prinzipalmarkt/Dom area), for a consistent view of the city.
MAP_CENTER = {"lat": 51.9625, "lon": 7.6256}
MAP_ZOOM = 11.3

# Every exception a live fetch/feature-assembly/prediction call can raise,
# for pages to catch in one place and show as a friendly st.error/st.info.
# Covers the full call graph of build_forecast/build_fleet_snapshot: fetch
# errors from each live source, schema-validation errors on what they
# return, the calendar-table lookup range, and feature/model-table assembly.
FETCH_ERRORS = (
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


@st.cache_resource(show_spinner=False)
def load_model() -> object:
    """Loads the committed production model once per app process."""
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_ratio_table() -> pd.DataFrame:
    """Loads the committed static per-station weekend/weekday ratio table."""
    return pd.read_csv(RATIO_TABLE_PATH)


@st.cache_resource(show_spinner=False)
def load_station_locations() -> pd.DataFrame:
    """Loads the committed station-coordinate table (see notebook 07)."""
    return pd.read_csv(STATION_LOCATIONS_PATH, dtype={"station_id": str})


@st.cache_data(ttl=3600, show_spinner="Loading station list…")
def cached_list_stations() -> list[Station]:
    return list_stations()


@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_bike_month(
    station_id: str, year: int, month: int
) -> pd.DataFrame | None:
    return fetch_station_month(station_id, year, month)


@st.cache_data(ttl=900, show_spinner="Fetching recent weather…")
def cached_recent_weather() -> pd.DataFrame:
    frames = {
        parameter: fetch_hourly_weather(parameter, period="recent")
        for parameter in PARAMETER_SPECS
    }
    return combine_weather_parameters(frames)


@st.cache_data(ttl=86400, show_spinner="Fetching calendar data…")
def cached_calendar_tables(year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    holidays_df = public_holidays(year - 1, year, subdiv=DEFAULT_PUBLIC_HOLIDAY_SUBDIV)
    school_holidays_df = fetch_school_holidays(year - 1, year)
    return holidays_df, school_holidays_df


def build_forecast(station: Station, as_of: date) -> dict[str, object]:
    """Fetches live data and runs the production model for one station.

    Raises:
        inference.InferenceError: if no recent bike-count data is available
            for `station`, the assembled feature row cannot be scored, or a
            committed model artifact (the model file, the ratio table) is
            missing on this server - translated from the underlying
            `FileNotFoundError` so callers only need to catch one error
            family, and so the artifact's server-local path never reaches a
            page's error message.
    """
    frames = []
    for year, month in inference.months_needed(as_of):
        if year < station.start_year:
            continue
        frame = cached_fetch_bike_month(station.station_id, year, month)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise inference.InferenceError(
            f"No recent bike-count data available for {station.name} "
            f"({station.station_id}) in the last {inference.MIN_HISTORY_LOOKBACK.days} days."
        )
    raw_bike_df = pd.concat(frames, ignore_index=True)

    weather_wide_df = cached_recent_weather()
    public_holidays_df, school_holidays_df = cached_calendar_tables(as_of.year)
    try:
        ratio_table = load_ratio_table()

        history = inference.assemble_feature_history(
            raw_bike_df,
            weather_wide_df,
            public_holidays_df,
            school_holidays_df,
            ratio_table,
        )
        current_row = inference.latest_feature_row(history, station.station_id)
        model = load_model()
        forecast_value = inference.predict_24h_ahead(model, current_row)
        forecast_curve = inference.predict_forecast_curve(
            model, history, station.station_id, current_row["datetime"]
        )
    except FileNotFoundError as exc:
        raise inference.InferenceError(
            "A required model artifact is missing on this server; the app "
            "cannot serve forecasts until it is restored."
        ) from exc

    return {
        "history": history,
        "current_row": current_row,
        "forecast_value": forecast_value,
        "forecast_curve": forecast_curve,
    }


@dataclass(frozen=True)
class FleetSnapshot:
    """Result of `build_fleet_snapshot`: the usable rows plus what was dropped.

    Attributes:
        data: One row per station with usable recent data.
        dropped_stations: Names of stations skipped because no usable
            recent data was available, in `cached_list_stations()` order.
            Empty when every station had usable data.
    """

    data: pd.DataFrame
    dropped_stations: list[str]


@st.cache_data(ttl=900, show_spinner="Building city-wide snapshot…")
def build_fleet_snapshot(as_of: date) -> FleetSnapshot:
    """Builds a current-count + 24h-forecast snapshot across every station.

    Stations with no usable recent data are skipped (rather than failing
    the whole snapshot) but named in the returned `dropped_stations`, so a
    caller can surface that some stations are missing instead of the
    comparison/map silently looking complete with fewer rows than
    stations. The per-station detail page separately surfaces the
    underlying error for whichever single station the user has selected.
    """
    rows = []
    dropped = []
    for station in cached_list_stations():
        try:
            result = build_forecast(station, as_of)
        except FETCH_ERRORS as exc:
            logger.warning(
                "Dropping station %s (%s) from fleet snapshot: %s",
                station.station_id,
                station.name,
                exc,
            )
            dropped.append(station.name)
            continue
        current_row = result["current_row"]
        rows.append(
            {
                "station_id": station.station_id,
                "name": station.name,
                "current_total_count": current_row["total_count"],
                "current_datetime": current_row["datetime"],
                "forecast_value": result["forecast_value"],
            }
        )
    return FleetSnapshot(data=pd.DataFrame(rows), dropped_stations=dropped)


def render_dropped_stations_warning(fleet_snapshot: FleetSnapshot) -> None:
    """Shows an `st.warning` naming skipped stations, if any were dropped.

    Shared by every page that calls `build_fleet_snapshot` so the wording
    can't drift between pages; the underlying reason for each drop is only
    logged server-side (see `build_fleet_snapshot`), not shown here, since
    it isn't this project's convention to introduce per-station diagnostic
    UI beyond naming what's missing.
    """
    if not fleet_snapshot.dropped_stations:
        return
    st.warning(
        f"{len(fleet_snapshot.dropped_stations)} station(s) have no recent "
        f"enough data and are not shown: "
        f"{', '.join(fleet_snapshot.dropped_stations)}."
    )


def render_forecast_chart(
    history: pd.DataFrame, current_row: pd.Series, forecast_curve: pd.DataFrame
) -> go.Figure:
    """Builds a recent-history + rolling-average + forecast-curve chart."""
    window_start = current_row["datetime"] - CHART_HISTORY_WINDOW
    windowed = history[history["datetime"] >= window_start]
    recent = windowed.dropna(subset=["total_count"])
    rolling = windowed.dropna(subset=["rolling_mean_24h"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["datetime"],
            y=recent["total_count"],
            mode="lines",
            name="Observed traffic",
            line=dict(color="#4C78A8"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=rolling["datetime"],
            y=rolling["rolling_mean_24h"],
            mode="lines",
            name="Rolling 24h average",
            line=dict(color="#72B7B2", dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[current_row["datetime"]],
            y=[current_row["total_count"]],
            mode="markers",
            name="Now",
            marker=dict(color="#4C78A8", size=10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=forecast_curve["target_datetime"],
            y=forecast_curve["predicted_total_count"],
            mode="lines+markers",
            name="24h-ahead forecast",
            line=dict(color="#E45756"),
            marker=dict(size=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[forecast_curve["target_datetime"].iloc[-1]],
            y=[forecast_curve["predicted_total_count"].iloc[-1]],
            mode="markers",
            name="Forecast for +24h",
            marker=dict(color="#E45756", size=12, symbol="star"),
        )
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Bike count (per 15 min)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=80),
    )
    fig.add_annotation(
        text="Source: bike counts od-ms/radverkehr-zaehlstellen (dl-de/by-2-0); weather DWD Open Data (CC BY 4.0)",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.25,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )
    return fig


MARKER_SIZE_RANGE = (14, 34)
MARKER_HALO_PADDING = 8
# ~30m at Münster's latitude - enough to visually separate two coincident
# markers without materially misrepresenting either station's location.
COINCIDENT_MARKER_OFFSET_DEGREES = 0.00025


def _spread_coincident_markers(locations: pd.DataFrame) -> pd.DataFrame:
    """Nudges markers that share exact coordinates apart in a small circle.

    Some station coordinates are a documented, intentional approximation
    (e.g. `notebooks/07_descriptive_analysis.ipynb`'s
    `GEOCODE_QUERY_OVERRIDES` maps two distinct "Kanalpromenade" path
    segments, station ids 100053305 and 300037936, to the same generic
    fallback query) rather than an error - so the fix belongs here, in how
    coincident points are drawn, not in the cached coordinates themselves.
    Without this, one marker fully occludes the other on the map.

    Args:
        locations: Station coordinates with ``lat``/``lon`` columns.

    Returns:
        Copy of `locations` with `lat`/`lon` perturbed for every station
        that shares its exact coordinates with at least one other row;
        stations with a unique coordinate are returned unchanged.
    """
    result = locations.copy()
    for _, group in result.groupby(["lat", "lon"]):
        if len(group) < 2:
            continue
        n = len(group)
        for offset, idx in enumerate(group.index):
            angle = 2 * math.pi * offset / n
            result.loc[idx, "lat"] += COINCIDENT_MARKER_OFFSET_DEGREES * math.cos(angle)
            result.loc[idx, "lon"] += COINCIDENT_MARKER_OFFSET_DEGREES * math.sin(angle)
    return result


def render_station_map(snapshot: pd.DataFrame, locations: pd.DataFrame) -> go.Figure:
    """Plots every station on OpenStreetMap, sized/colored by its live forecast.

    `go.Scattermap` markers have no `line`/border property (unlike a plain
    `go.Scatter` marker), and OpenStreetMap tiles have no single fixed
    background color to contrast against (parks are green, roads pale
    yellow/white, water blue, buildings tan) — a sequential colorscale's
    light end can disappear into whichever tile color happens to match. The
    fix used everywhere on the web for point-on-map markers: a solid, dark
    "halo" trace drawn underneath the real (colored) markers, slightly
    larger, so every marker keeps contrast regardless of both its own color
    and the tile color beneath it.

    Marker sizes are computed manually (not via `px.scatter_map`'s
    automatic bubble sizing) so the halo can be sized in lockstep with the
    data markers it sits behind.
    """
    spread_locations = _spread_coincident_markers(
        locations[["station_id", "lat", "lon"]]
    )
    merged = snapshot.merge(spread_locations, on="station_id")
    merged["label"] = merged["name"] + " (" + merged["station_id"] + ")"

    value = merged["forecast_value"]
    value_range = value.max() - value.min()
    size_min, size_max = MARKER_SIZE_RANGE
    if value_range > 0:
        merged["marker_size"] = (
            size_min + (size_max - size_min) * (value - value.min()) / value_range
        )
    else:
        merged["marker_size"] = (size_min + size_max) / 2

    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=merged["lat"],
            lon=merged["lon"],
            mode="markers",
            marker=dict(
                size=merged["marker_size"] + MARKER_HALO_PADDING,
                color="#26264d",
                opacity=0.9,
            ),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scattermap(
            lat=merged["lat"],
            lon=merged["lon"],
            mode="markers",
            marker=dict(
                size=merged["marker_size"],
                color=merged["forecast_value"],
                colorscale="YlOrRd",
                cmin=value.min(),
                cmax=value.max(),
                colorbar=dict(title="24h forecast"),
                opacity=1.0,
            ),
            text=merged["label"],
            customdata=merged[["current_total_count", "forecast_value"]],
            hovertemplate=(
                "%{text}<br>Current: %{customdata[0]:.0f}<br>"
                "Forecast +24h: %{customdata[1]:.0f}<extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.update_layout(
        map=dict(style="open-street-map", center=MAP_CENTER, zoom=MAP_ZOOM),
        height=550,
        margin={"l": 0, "r": 0, "t": 10, "b": 40},
    )
    fig.add_annotation(
        text=(
            "Source: bike counts od-ms/radverkehr-zaehlstellen; weather DWD Open Data; "
            "station coordinates OpenStreetMap Nominatim; map tiles © OpenStreetMap contributors"
        ),
        xref="paper",
        yref="paper",
        x=0,
        y=-0.06,
        showarrow=False,
        font=dict(size=10, color="gray"),
    )
    return fig


def render_footer() -> None:
    st.divider()
    st.caption(
        "**Impressum & Datenschutz** (Platzhalter — vor Veröffentlichung vervollständigen): "
        "Verantwortlich gemäß § 5 DDG: [Name], [Postanschrift], [Kontakt-E-Mail]. "
        "Diese Seite läuft auf Streamlit Community Cloud; Ihr Hoster protokolliert "
        "Besucher-IP-Adressen serverseitig. Es werden keine Cookies gesetzt und keine "
        "personenbezogenen Daten der Nutzer:innen dieser App gespeichert."
    )
    st.caption(
        "Data sources: bike counts © Stadt Münster, "
        "[od-ms/radverkehr-zaehlstellen](https://github.com/od-ms/radverkehr-zaehlstellen) "
        "(dl-de/by-2-0) · weather © Deutscher Wetterdienst, "
        "[DWD Open Data](https://opendata.dwd.de) (CC BY 4.0) · school holidays via "
        "[OpenHolidays API](https://openholidaysapi.org/) (ODbL-1.0) · public holidays via the "
        "[`holidays`](https://pypi.org/project/holidays/) Python library."
    )

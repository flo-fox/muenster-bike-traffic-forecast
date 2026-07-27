"""Shared caching, live-data fetching, and chart-building for the dashboard.

Split out of `app.py` so the same cached resources (model, station list,
weather, calendar data) and the same fetch/predict/chart logic are reused
across every page of the multipage app (`pages/`), rather than duplicated
or re-fetched per page. `app.py` is the only entry point Streamlit runs
directly; it sets up `sys.path` before any page (and therefore this
module) is imported.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
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
    fetch_school_holidays,
    public_holidays,
)
from muenster_bike_forecast.data.join import JoinError, combine_weather_parameters
from muenster_bike_forecast.data.weather import (
    PARAMETER_SPECS,
    WeatherFetchError,
    fetch_hourly_weather,
)
from muenster_bike_forecast import inference

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
FETCH_ERRORS = (
    BikeCountDataError,
    WeatherFetchError,
    SchoolHolidayFetchError,
    JoinError,
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
def cached_fetch_bike_month(station_id: str, year: int, month: int) -> pd.DataFrame | None:
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
            for `station`, or the assembled feature row cannot be scored.
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
    ratio_table = load_ratio_table()

    history = inference.assemble_feature_history(
        raw_bike_df, weather_wide_df, public_holidays_df, school_holidays_df, ratio_table
    )
    current_row = inference.latest_feature_row(history, station.station_id)
    model = load_model()
    forecast_value = inference.predict_24h_ahead(model, current_row)
    forecast_curve = inference.predict_forecast_curve(
        model, history, station.station_id, current_row["datetime"]
    )

    return {
        "history": history,
        "current_row": current_row,
        "forecast_value": forecast_value,
        "forecast_curve": forecast_curve,
    }


@st.cache_data(ttl=900, show_spinner="Building city-wide snapshot…")
def build_fleet_snapshot(as_of: date) -> pd.DataFrame:
    """Builds a current-count + 24h-forecast snapshot across every station.

    Stations with no usable recent data are silently skipped (rather than
    failing the whole snapshot) — the per-station detail page already
    surfaces that error clearly for whichever station the user has selected.
    """
    rows = []
    for station in cached_list_stations():
        try:
            result = build_forecast(station, as_of)
        except FETCH_ERRORS:
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
    return pd.DataFrame(rows)


def render_forecast_chart(
    history: pd.DataFrame, current_row: pd.Series, forecast_curve: pd.DataFrame
) -> go.Figure:
    """Builds a recent-history + rolling-average + forecast-curve chart."""
    window_periods = int(CHART_HISTORY_WINDOW / pd.Timedelta(minutes=15))
    recent = history.dropna(subset=["total_count"]).tail(window_periods)
    rolling = history.dropna(subset=["rolling_mean_24h"]).tail(window_periods)

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


def render_station_map(snapshot: pd.DataFrame, locations: pd.DataFrame) -> go.Figure:
    """Plots every station on OpenStreetMap, sized/colored by its live forecast.

    Same map style/center/zoom/colorscale as
    `notebooks/07_descriptive_analysis.ipynb`'s station map, swapping that
    notebook's static all-time mean for this dashboard's live 24h-ahead
    forecast.
    """
    merged = snapshot.merge(locations[["station_id", "lat", "lon"]], on="station_id")
    merged["label"] = merged["name"] + " (" + merged["station_id"] + ")"

    fig = px.scatter_map(
        merged,
        lat="lat",
        lon="lon",
        size="forecast_value",
        color="forecast_value",
        color_continuous_scale="YlOrRd",
        hover_name="label",
        hover_data={
            "lat": False,
            "lon": False,
            "current_total_count": ":.0f",
            "forecast_value": ":.0f",
        },
        zoom=MAP_ZOOM,
        center=MAP_CENTER,
        height=550,
        map_style="open-street-map",
    )
    fig.update_traces(marker={"opacity": 0.9})
    fig.update_layout(
        coloraxis_colorbar_title="24h forecast",
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

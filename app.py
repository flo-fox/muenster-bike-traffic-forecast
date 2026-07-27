"""Streamlit dashboard: live 24h-ahead bike-traffic forecast per station.

Fetches fresh bike-count (od-ms/radverkehr-zaehlstellen) and weather (DWD
Open Data) data at request time, builds the same feature row
`notebooks/17_final_production_model.ipynb` trained on, and runs the
committed production model (`models/production_random_forest.joblib`) to
predict traffic 24 hours ahead for the selected station.

All data fetching and feature assembly is delegated to
`src/muenster_bike_forecast/` (data + inference modules); this file only
handles Streamlit wiring (caching, layout, error display).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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

MODEL_PATH = PROJECT_ROOT / "models" / "production_random_forest.joblib"
RATIO_TABLE_PATH = PROJECT_ROOT / "models" / "weekend_weekday_ratio.csv"

st.set_page_config(page_title="Münster Bike Traffic Forecast", page_icon="🚲", layout="wide")


@st.cache_resource(show_spinner=False)
def load_model() -> object:
    """Loads the committed production model once per app process."""
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_ratio_table() -> pd.DataFrame:
    """Loads the committed static per-station weekend/weekday ratio table."""
    return pd.read_csv(RATIO_TABLE_PATH)


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

    return {
        "history": history,
        "current_row": current_row,
        "forecast_value": forecast_value,
    }


def render_forecast_chart(history: pd.DataFrame, current_row: pd.Series, forecast_value: float) -> go.Figure:
    """Builds a recent-history + forecast-point chart for one station."""
    recent = history.dropna(subset=["total_count"]).tail(4 * 24 * 3)  # last ~3 days
    target_time = current_row["datetime"] + pd.Timedelta(hours=24)

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
            x=[current_row["datetime"]],
            y=[current_row["total_count"]],
            mode="markers",
            name="Now",
            marker=dict(color="#4C78A8", size=10),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[target_time],
            y=[forecast_value],
            mode="markers",
            name="24h-ahead forecast",
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


def main() -> None:
    st.title("🚲 Münster Bike Traffic Forecast")
    st.caption(
        "24h-ahead bike-traffic prediction per counting station, from live "
        "bike-count and weather data. Production model: random forest, "
        "MAE 27.07 / RMSE 53.70 on held-out data (see `notebooks/17_final_production_model.ipynb`)."
    )

    try:
        stations = cached_list_stations()
    except BikeCountDataError as exc:
        st.error(f"Could not load the station list: {exc}")
        return

    stations_by_label = {f"{s.name} ({s.station_id})": s for s in stations}
    selected_label = st.selectbox("Counting station", sorted(stations_by_label))
    station = stations_by_label[selected_label]

    as_of = date.today()
    try:
        with st.spinner(f"Building live forecast for {station.name}…"):
            result = build_forecast(station, as_of)
    except (
        BikeCountDataError,
        WeatherFetchError,
        SchoolHolidayFetchError,
        JoinError,
        inference.InferenceError,
    ) as exc:
        st.error(f"Could not build a forecast for {station.name}: {exc}")
        return

    current_row = result["current_row"]
    forecast_value = result["forecast_value"]
    target_time = current_row["datetime"] + pd.Timedelta(hours=24)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current count (per 15 min)", f"{current_row['total_count']:.0f}", help=f"As of {current_row['datetime']}")
    col2.metric(
        "Forecast in 24h (per 15 min)",
        f"{forecast_value:.0f}",
        delta=f"{forecast_value - current_row['total_count']:+.0f}",
        help=f"Target time: {target_time}",
    )
    col3.metric("Station", station.station_id, help=station.name)

    st.plotly_chart(
        render_forecast_chart(result["history"], current_row, forecast_value),
        width="stretch",
    )

    render_footer()


if __name__ == "__main__":
    main()

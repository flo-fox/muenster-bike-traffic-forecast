"""Page: per-station live 24h-ahead forecast (metrics + 7-day chart)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from dashboard_common import (
    FETCH_ERRORS,
    STALENESS_WARNING_THRESHOLD,
    build_forecast,
    cached_list_stations,
    render_footer,
    render_forecast_chart,
)
from muenster_bike_forecast.data.bike_counts import BikeCountDataError

st.title("📍 Station forecast")
st.caption(
    "24h-ahead bike-traffic prediction per counting station, from live "
    "bike-count and weather data. Production model: random forest, "
    "MAE 27.14 / RMSE 53.71 on held-out data (see `notebooks/17_final_production_model.ipynb`)."
)

try:
    stations = cached_list_stations()
except BikeCountDataError as exc:
    st.error(f"Could not load the station list: {exc}")
    st.stop()

stations_by_label = {f"{s.name} ({s.station_id})": s for s in stations}
selected_label = st.selectbox("Counting station", sorted(stations_by_label))
station = stations_by_label[selected_label]

as_of = date.today()
try:
    with st.spinner(f"Building live forecast for {station.name}…"):
        result = build_forecast(station, as_of)
except FETCH_ERRORS as exc:
    st.error(f"Could not build a forecast for {station.name}: {exc}")
    st.stop()

current_row = result["current_row"]
forecast_value = result["forecast_value"]
target_time = current_row["datetime"] + pd.Timedelta(hours=24)
data_age = (
    pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None) - current_row["datetime"]
)

col1, col2, col3 = st.columns(3)
col1.metric("Current count (per 15 min)", f"{current_row['total_count']:.0f}")
col2.metric(
    "Forecast in 24h (per 15 min)",
    f"{forecast_value:.0f}",
    delta=f"{forecast_value - current_row['total_count']:+.0f}",
)
col3.metric("Station", station.station_id, help=station.name)

st.caption(
    f"Latest available data: **{current_row['datetime']:%Y-%m-%d %H:%M}** (Europe/Berlin) "
    f"→ forecast target: **{target_time:%Y-%m-%d %H:%M}**"
)
if data_age > STALENESS_WARNING_THRESHOLD:
    st.warning(
        f"⚠️ The source hasn't published new data in {data_age.days}d "
        f"{data_age.components.hours}h (it normally updates about daily). "
        "This forecast is anchored to the most recent data available, not "
        "necessarily to right now."
    )

st.plotly_chart(
    render_forecast_chart(result["history"], current_row, result["forecast_curve"]),
    width="stretch",
)

render_footer()

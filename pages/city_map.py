"""Page: city-wide snapshot of every station, mapped on OpenStreetMap."""

from __future__ import annotations

from datetime import date

import streamlit as st

from dashboard_common import (
    FETCH_ERRORS,
    build_fleet_snapshot,
    load_station_locations,
    render_dropped_stations_warning,
    render_footer,
    render_station_map,
)

st.title("🗺️ City map")
st.caption("Current reading and 24h-ahead forecast for every station, mapped.")

as_of = date.today()
try:
    locations = load_station_locations()
except FileNotFoundError:
    st.info("Station location data not available.")
    st.stop()

try:
    fleet_snapshot = build_fleet_snapshot(as_of)
except FETCH_ERRORS as exc:
    st.error(f"Could not build the city-wide snapshot: {exc}")
    st.stop()

render_dropped_stations_warning(fleet_snapshot)

if fleet_snapshot.data.empty:
    st.info("No station currently has recent enough data to show.")
else:
    st.plotly_chart(render_station_map(fleet_snapshot.data, locations), width="stretch")

render_footer()

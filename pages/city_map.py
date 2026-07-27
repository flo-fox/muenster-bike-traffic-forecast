"""Page: city-wide snapshot of every station, mapped on OpenStreetMap."""

from __future__ import annotations

from datetime import date

import streamlit as st

from dashboard_common import (
    build_fleet_snapshot,
    load_station_locations,
    render_footer,
    render_station_map,
)

st.title("🗺️ City map")
st.caption("Current reading and 24h-ahead forecast for every station, mapped.")

as_of = date.today()
try:
    locations = load_station_locations()
    snapshot = build_fleet_snapshot(as_of)
except FileNotFoundError:
    st.info("Station location data not available.")
else:
    if snapshot.empty:
        st.info("No station currently has recent enough data to show.")
    else:
        st.plotly_chart(render_station_map(snapshot, locations), width="stretch")

render_footer()

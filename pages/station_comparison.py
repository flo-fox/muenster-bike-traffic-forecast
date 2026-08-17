"""Page: cross-sectional comparison of every station right now.

Unlike the station-forecast page (one station over time), this compares all
23 stations against each other at a single point in time: their current
reading and 24h-ahead forecast, side by side.
"""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import streamlit as st

from dashboard_common import FETCH_ERRORS, build_fleet_snapshot, render_footer

st.title("📊 Station comparison")
st.caption(
    "Cross-sectional comparison: how every station's current reading and "
    "24h-ahead forecast compare to each other right now (not a time series)."
)

as_of = date.today()
try:
    snapshot = build_fleet_snapshot(as_of)
except FETCH_ERRORS as exc:
    st.error(f"Could not build the station comparison: {exc}")
    st.stop()
if snapshot.empty:
    st.info("No station currently has recent enough data to compare.")
    st.stop()

ranked = snapshot.sort_values("forecast_value", ascending=True)
labels = ranked["name"] + " (" + ranked["station_id"] + ")"

fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=labels,
        x=ranked["current_total_count"],
        name="Current (per 15 min)",
        orientation="h",
        marker=dict(color="#4C78A8"),
    )
)
fig.add_trace(
    go.Bar(
        y=labels,
        x=ranked["forecast_value"],
        name="Forecast +24h (per 15 min)",
        orientation="h",
        marker=dict(color="#E45756"),
    )
)
fig.update_layout(
    barmode="group",
    xaxis_title="Bike count (per 15 min)",
    height=max(500, 28 * len(ranked)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    margin=dict(t=60, b=60, l=10, r=10),
)
fig.add_annotation(
    text="Source: bike counts od-ms/radverkehr-zaehlstellen (dl-de/by-2-0); weather DWD Open Data (CC BY 4.0)",
    xref="paper",
    yref="paper",
    x=0,
    y=-0.06,
    showarrow=False,
    font=dict(size=10, color="gray"),
)
st.plotly_chart(fig, width="stretch")

st.dataframe(
    snapshot.sort_values("forecast_value", ascending=False)[
        [
            "name",
            "station_id",
            "current_total_count",
            "forecast_value",
            "current_datetime",
        ]
    ].rename(
        columns={
            "name": "Station",
            "station_id": "ID",
            "current_total_count": "Current",
            "forecast_value": "Forecast +24h",
            "current_datetime": "As of",
        }
    ),
    width="stretch",
    hide_index=True,
)

render_footer()

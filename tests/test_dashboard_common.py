"""Tests for `dashboard_common`'s pure chart-building/geometry logic, plus
`build_fleet_snapshot`'s station-dropping branch.

Most tests here transform already-fetched data into a `go.Figure` or a
coordinate table - no live fetches, no Streamlit runtime exercised. The
`build_fleet_snapshot` tests are the one exception: `cached_list_stations`/
`build_forecast` are monkeypatched so the per-station loop and its
`FleetSnapshot.dropped_stations` bookkeeping run for real, without a live
fetch or a Streamlit script context - `st.cache_data`/`st.warning` work
fine called directly from a plain process. Each test clears the cache
first so it isn't served a previous test's cached result for the same
`as_of` date.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import dashboard_common
from dashboard_common import (
    MARKER_HALO_PADDING,
    FleetSnapshot,
    _spread_coincident_markers,
    build_fleet_snapshot,
    render_forecast_chart,
    render_station_map,
)
from muenster_bike_forecast.data.bike_counts import BikeCountDataError, Station

# ---------------------------------------------------------------------------
# _spread_coincident_markers
# ---------------------------------------------------------------------------


def test_spread_coincident_markers_separates_exact_duplicates() -> None:
    # Mirrors the real case: two distinct "Kanalpromenade" station ids
    # geocoded to the same fallback coordinate (see notebook 07).
    locations = pd.DataFrame(
        {
            "station_id": ["100053305", "300037936"],
            "lat": [51.9594524, 51.9594524],
            "lon": [7.6624286, 7.6624286],
        }
    )
    result = _spread_coincident_markers(locations)
    assert (result["lat"].iloc[0], result["lon"].iloc[0]) != (
        result["lat"].iloc[1],
        result["lon"].iloc[1],
    )
    # The nudge should be small - stations must not appear to move meaningfully.
    assert (result["lat"] - 51.9594524).abs().max() < 0.01
    assert (result["lon"] - 7.6624286).abs().max() < 0.01


def test_spread_coincident_markers_leaves_unique_coordinates_untouched() -> None:
    locations = pd.DataFrame(
        {
            "station_id": ["a", "b"],
            "lat": [51.95, 51.96],
            "lon": [7.60, 7.61],
        }
    )
    result = _spread_coincident_markers(locations)
    pd.testing.assert_frame_equal(result, locations)


def test_spread_coincident_markers_handles_three_way_tie() -> None:
    locations = pd.DataFrame(
        {
            "station_id": ["a", "b", "c"],
            "lat": [51.95, 51.95, 51.95],
            "lon": [7.60, 7.60, 7.60],
        }
    )
    result = _spread_coincident_markers(locations)
    points = list(zip(result["lat"], result["lon"]))
    assert len(set(points)) == 3


# ---------------------------------------------------------------------------
# render_forecast_chart
# ---------------------------------------------------------------------------


def _sample_history() -> pd.DataFrame:
    times = pd.date_range("2026-08-01 00:00", periods=8, freq="15min")
    return pd.DataFrame(
        {
            "datetime": times,
            "total_count": [10, 12, 9, 11, 14, 13, 15, 16],
            "rolling_mean_24h": [None, None, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
        }
    )


def test_render_forecast_chart_includes_data_source_caption() -> None:
    history = _sample_history()
    current_row = history.iloc[-1]
    forecast_curve = pd.DataFrame(
        {
            "target_datetime": pd.date_range(
                current_row["datetime"] + pd.Timedelta(hours=1), periods=4, freq="6h"
            ),
            "predicted_total_count": [17, 18, 16, 19],
        }
    )
    fig = render_forecast_chart(history, current_row, forecast_curve)
    caption_texts = [ann.text for ann in fig.layout.annotations]
    assert any("Source" in text for text in caption_texts)


def test_render_forecast_chart_plots_every_forecast_point() -> None:
    history = _sample_history()
    current_row = history.iloc[-1]
    forecast_curve = pd.DataFrame(
        {
            "target_datetime": pd.date_range(
                current_row["datetime"] + pd.Timedelta(hours=1), periods=4, freq="6h"
            ),
            "predicted_total_count": [17, 18, 16, 19],
        }
    )
    fig = render_forecast_chart(history, current_row, forecast_curve)
    forecast_trace = next(t for t in fig.data if t.name == "24h-ahead forecast")
    assert list(forecast_trace.y) == [17, 18, 16, 19]


# ---------------------------------------------------------------------------
# render_station_map
# ---------------------------------------------------------------------------


def _sample_snapshot_and_locations() -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshot = pd.DataFrame(
        {
            "station_id": ["100053305", "300037936", "300038855"],
            "name": ["Kanalpromenade 5", "Kanalpromenade 6", "Bismarckallee"],
            "current_total_count": [5, 8, 20],
            "forecast_value": [6, 9, 25],
        }
    )
    locations = pd.DataFrame(
        {
            "station_id": ["100053305", "300037936", "300038855"],
            "lat": [51.9594524, 51.9594524, 51.9557872],
            "lon": [7.6624286, 7.6624286, 7.6170896],
        }
    )
    return snapshot, locations


def test_render_station_map_has_halo_and_data_traces_for_every_station() -> None:
    snapshot, locations = _sample_snapshot_and_locations()
    fig = render_station_map(snapshot, locations)
    assert len(fig.data) == 2
    halo, data_trace = fig.data
    assert len(halo.lat) == len(snapshot)
    assert len(data_trace.lat) == len(snapshot)


def test_render_station_map_halo_is_larger_than_data_marker() -> None:
    snapshot, locations = _sample_snapshot_and_locations()
    fig = render_station_map(snapshot, locations)
    halo, data_trace = fig.data
    assert list(halo.marker.size) == pytest.approx(
        [size + MARKER_HALO_PADDING for size in data_trace.marker.size]
    )


def test_render_station_map_separates_coincident_stations() -> None:
    # 100053305 and 300037936 share an exact coordinate in the fixture (see
    # `_spread_coincident_markers`'s docstring for why this happens for real
    # stations too) - the rendered map must not plot them on top of each other.
    snapshot, locations = _sample_snapshot_and_locations()
    fig = render_station_map(snapshot, locations)
    _, data_trace = fig.data
    first_point = (data_trace.lat[0], data_trace.lon[0])
    second_point = (data_trace.lat[1], data_trace.lon[1])
    assert first_point != second_point


# ---------------------------------------------------------------------------
# build_fleet_snapshot
# ---------------------------------------------------------------------------


def _station(station_id: str, name: str) -> Station:
    return Station(station_id=station_id, name=name, start_year=2023, channels=())


def test_build_fleet_snapshot_reports_dropped_stations(monkeypatch) -> None:
    build_fleet_snapshot.clear()
    stations = [
        _station("100020113", "Good Station"),
        _station("999999999", "Broken Station"),
    ]
    monkeypatch.setattr(dashboard_common, "cached_list_stations", lambda: stations)

    def fake_build_forecast(station: Station, as_of: date) -> dict[str, object]:
        if station.station_id == "999999999":
            raise BikeCountDataError("no usable rows")
        return {
            "current_row": pd.Series(
                {"total_count": 5.0, "datetime": pd.Timestamp(as_of)}
            ),
            "forecast_value": 6.0,
        }

    monkeypatch.setattr(dashboard_common, "build_forecast", fake_build_forecast)

    result = build_fleet_snapshot(date(2026, 8, 19))

    assert isinstance(result, FleetSnapshot)
    assert result.dropped_stations == ["Broken Station"]
    assert list(result.data["station_id"]) == ["100020113"]


def test_build_fleet_snapshot_has_no_dropped_stations_when_all_succeed(
    monkeypatch,
) -> None:
    build_fleet_snapshot.clear()
    stations = [_station("100020113", "Good Station")]
    monkeypatch.setattr(dashboard_common, "cached_list_stations", lambda: stations)

    def fake_build_forecast(station: Station, as_of: date) -> dict[str, object]:
        return {
            "current_row": pd.Series(
                {"total_count": 5.0, "datetime": pd.Timestamp(as_of)}
            ),
            "forecast_value": 6.0,
        }

    monkeypatch.setattr(dashboard_common, "build_forecast", fake_build_forecast)

    result = build_fleet_snapshot(date(2026, 8, 20))

    assert result.dropped_stations == []
    assert len(result.data) == 1

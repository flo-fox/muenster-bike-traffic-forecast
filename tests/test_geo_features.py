"""Tests for `muenster_bike_forecast.modeling.geo_features`.

All tests use small, hand-built synthetic data - no live network calls and
no dependency on the real `data/raw/` files.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from muenster_bike_forecast.modeling.geo_features import (
    CENTER_LAT,
    CENTER_LON,
    GeoFeatureError,
    add_distance_from_center,
    haversine_distance_km,
)

# ---------------------------------------------------------------------------
# haversine_distance_km
# ---------------------------------------------------------------------------


def test_haversine_distance_km_same_point_is_zero() -> None:
    assert haversine_distance_km(51.9625, 7.6256, 51.9625, 7.6256) == pytest.approx(0.0)


def test_haversine_distance_km_known_reference_distance() -> None:
    # Berlin (Brandenburg Gate) to Munich (Marienplatz) is a well-known
    # great-circle distance of roughly 505km - a sanity check that the
    # formula and EARTH_RADIUS_KM constant are wired up correctly, not just
    # internally self-consistent.
    berlin_lat, berlin_lon = 52.5163, 13.3777
    munich_lat, munich_lon = 48.1374, 11.5755
    distance = haversine_distance_km(berlin_lat, berlin_lon, munich_lat, munich_lon)
    assert distance == pytest.approx(504, abs=10)


def test_haversine_distance_km_matches_flat_earth_at_short_range() -> None:
    # At the few-km scale relevant to this project, the exact haversine
    # distance should closely match notebook 07's flat-earth approximation
    # (1 degree latitude ~= 111km, longitude scaled by cos(latitude)).
    lat, lon = 51.9777, 7.6156  # a station a few km from the center
    exact = haversine_distance_km(lat, lon, CENTER_LAT, CENTER_LON)
    approx = math.sqrt(
        ((lat - CENTER_LAT) * 111) ** 2
        + ((lon - CENTER_LON) * 111 * math.cos(math.radians(CENTER_LAT))) ** 2
    )
    assert exact == pytest.approx(approx, rel=0.01)


def test_haversine_distance_km_vectorized_over_series() -> None:
    lat = pd.Series([51.9625, 51.9777])
    lon = pd.Series([7.6256, 7.6156])
    result = haversine_distance_km(lat, lon, CENTER_LAT, CENTER_LON)
    assert isinstance(result, pd.Series)
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[1] > 0


# ---------------------------------------------------------------------------
# add_distance_from_center
# ---------------------------------------------------------------------------


def test_add_distance_from_center_adds_expected_column() -> None:
    station_locations = pd.DataFrame(
        {
            "station_id": [1, 2],
            "lat": [CENTER_LAT, 51.9777],
            "lon": [CENTER_LON, 7.6156],
        }
    )
    out = add_distance_from_center(station_locations)
    assert "distance_from_center_km" in out.columns
    assert out.loc[out["station_id"] == 1, "distance_from_center_km"].iloc[
        0
    ] == pytest.approx(0.0)
    assert out.loc[out["station_id"] == 2, "distance_from_center_km"].iloc[0] > 0
    # No rows dropped or added - safe to merge onto a model table.
    assert len(out) == len(station_locations)


def test_add_distance_from_center_missing_column_raises() -> None:
    station_locations = pd.DataFrame({"station_id": [1], "lat": [51.96]})
    with pytest.raises(GeoFeatureError):
        add_distance_from_center(station_locations)

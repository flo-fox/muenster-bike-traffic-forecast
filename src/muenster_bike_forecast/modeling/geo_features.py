"""Distance-from-city-center feature for the bike-traffic model.

`notebooks/07_descriptive_analysis.ipynb` (section 5) found a moderate
negative correlation (r ~= -0.51) between a station's distance from the
Prinzipalmarkt/Dom city center and its all-time mean traffic — busier
stations cluster on the central Altstadt ring, quieter ones sit 2-4km out
(with exceptions). That notebook computed the distance purely for the
correlation check, using a flat-earth approximation (1 degree latitude ~=
111km everywhere; 1 degree longitude scaled by cos(latitude)) and never fed
it into a model. This module turns the same idea into a reusable, exact
model feature: `haversine_distance_km` computes true great-circle distance
(no flat-earth approximation), and `add_distance_from_center` applies it
per station using the identical Prinzipalmarkt/Dom reference point notebook
07 used, so the "city center" definition stays consistent between that
descriptive finding and this feature.

All I/O (reading `data/raw/bike_counts/station_locations.csv`) is left to
the caller; the functions here only transform DataFrames/values already in
memory.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

# Mean earth radius (km), WGS84 authalic radius - standard constant for
# haversine great-circle distance.
EARTH_RADIUS_KM: Final[float] = 6371.0088

# Prinzipalmarkt/Dom Münster city-center reference point - identical to the
# CENTER_LAT/CENTER_LON used in notebooks/07_descriptive_analysis.ipynb's
# distance-from-center exploration, reused as-is (not re-derived) so this
# feature and that descriptive finding refer to the same "center".
CENTER_LAT: Final[float] = 51.9625
CENTER_LON: Final[float] = 7.6256


class GeoFeatureError(Exception):
    """Raised when a distance-from-center feature cannot be computed.

    Covers missing required columns in the input station-locations table.
    """


def haversine_distance_km(
    lat1: float | pd.Series,
    lon1: float | pd.Series,
    lat2: float | pd.Series,
    lon2: float | pd.Series,
) -> float | pd.Series:
    """Computes great-circle distance(s) in km between WGS84 coordinates.

    Uses the haversine formula, which is exact for a spherical-earth model
    (accurate to well under the ~0.3% oblateness error of the earth, more
    than precise enough at the few-km scale relevant here) — unlike the
    flat-earth approximation `07_descriptive_analysis.ipynb` used for its
    purely descriptive correlation check.

    Args:
        lat1: Latitude (degrees) of the first point(s). Scalar or
            elementwise-aligned with `lon1`/`lat2`/`lon2` if any is a
            `pandas.Series`.
        lon1: Longitude (degrees) of the first point(s).
        lat2: Latitude (degrees) of the second point(s).
        lon2: Longitude (degrees) of the second point(s).

    Returns:
        Distance(s) in km, same shape as the (broadcast) inputs — a
        `float` if all inputs are scalars, a `pandas.Series` if any input
        is one.
    """
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arcsin(np.sqrt(a))
    return EARTH_RADIUS_KM * c


def add_distance_from_center(
    station_locations: pd.DataFrame,
    station_col: str = "station_id",
    lat_col: str = "lat",
    lon_col: str = "lon",
    center_lat: float = CENTER_LAT,
    center_lon: float = CENTER_LON,
    feature_col: str = "distance_from_center_km",
) -> pd.DataFrame:
    """Adds a per-station great-circle distance from the city center.

    Args:
        station_locations: One row per station, as loaded from
            `data/raw/bike_counts/station_locations.csv` (see
            `muenster_bike_forecast.data.geocode`); needs `station_col`,
            `lat_col`, `lon_col` columns.
        station_col: Name of the station-id column.
        lat_col: Name of the latitude column (degrees).
        lon_col: Name of the longitude column (degrees).
        center_lat: Reference latitude for "city center" (degrees).
            Defaults to `CENTER_LAT` (Prinzipalmarkt/Dom), matching
            `07_descriptive_analysis.ipynb`.
        center_lon: Reference longitude for "city center" (degrees).
            Defaults to `CENTER_LON`, matching
            `07_descriptive_analysis.ipynb`.
        feature_col: Name of the new distance column to add.

    Returns:
        Copy of `station_locations` with `feature_col` (km) added,
        containing exactly one row per input row (no rows dropped or
        added) so it can be merged onto a model table on `station_col`.

    Raises:
        GeoFeatureError: if `station_col`, `lat_col`, or `lon_col` is
            missing from `station_locations`.
    """
    required = {station_col, lat_col, lon_col}
    missing = required - set(station_locations.columns)
    if missing:
        raise GeoFeatureError(
            f"station_locations is missing column(s): {sorted(missing)}."
        )

    out = station_locations.copy()
    out[feature_col] = haversine_distance_km(
        out[lat_col], out[lon_col], center_lat, center_lon
    )
    return out

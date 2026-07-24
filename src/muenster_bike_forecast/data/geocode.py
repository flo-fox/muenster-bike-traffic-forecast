"""Geocode bike-counting station names to coordinates via Nominatim.

Station metadata from the source repo (``data/raw/bike_counts/stations.json``,
see ``muenster_bike_forecast.data.bike_counts``) has a ``name`` (e.g.
``"Bismarckallee"``) but no coordinates. This module resolves each station
name to a ``(lat, lon)`` pair using the OpenStreetMap Nominatim search API
(https://nominatim.openstreetmap.org/), so the descriptive-analysis notebook
can plot station locations on a map.

Nominatim usage policy (https://operations.osmfoundation.org/policies/nominatim/)
requires: at most one request per second, and a descriptive ``User-Agent``
identifying the client (not a spoofed browser string) — both enforced here
(`MIN_REQUEST_INTERVAL_SECONDS`, `USER_AGENT`).

Several station names are ambiguous or too generic to geocode reliably on
their own (e.g. plain ``"Neutor"`` or ``"Bohlweg"`` could in principle match
a similarly-named place elsewhere), so every query is qualified with
``", Münster, Germany"`` and every result is checked against a Münster
bounding box (`MUENSTER_BOUNDS`) before being accepted — a result outside the
box, or no match at all, is recorded with ``resolved=False`` rather than
silently guessed at. A few station names are canal-promenade path segments
(e.g. ``"Kanalpromenade, Abschnitt 5"``) that do not geocode as a single
point via a plain name search; callers may supply a `query_overrides`
mapping (station_id -> a nearby cross-street or the general
``"Kanalpromenade, Münster, Germany"`` query) as a documented approximation
for these — the resulting `geocode_query` column records exactly what was
searched, so an approximated row can always be told apart from an exact
name match.

Results are cached to disk (CSV, columns `CACHE_COLUMNS`) so repeated
notebook runs do not re-hit the API: `geocode_stations` only geocodes
station ids missing from the cache, making it idempotent (see
``muenster_bike_forecast.data.weather.fetch_hourly_weather`` for the same
convention on the weather-fetch side).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import requests

NOMINATIM_SEARCH_URL: Final[str] = "https://nominatim.openstreetmap.org/search"
USER_AGENT: Final[str] = "muenster-bike-traffic-forecast/1.0 (research project)"
# Nominatim's usage policy caps request rate at 1/second; this is the
# minimum time between the *start* of consecutive requests.
MIN_REQUEST_INTERVAL_SECONDS: Final[float] = 1.0
DEFAULT_QUERY_SUFFIX: Final[str] = ", Münster, Germany"
# Münster city bounding box (lat_min, lat_max, lon_min, lon_max), used to
# reject a geocoding result that falls outside the city entirely - e.g. a
# generic station name like "Neutor" matching a similarly-named place in a
# different town.
MUENSTER_BOUNDS: Final[tuple[float, float, float, float]] = (51.90, 52.02, 7.55, 7.70)

CACHE_COLUMNS: Final[list[str]] = [
    "station_id",
    "name",
    "lat",
    "lon",
    "geocode_query",
    "resolved",
]


class GeocodeError(Exception):
    """Raised when station geocoding or its cache cannot proceed as expected.

    Covers Nominatim request/response failures, and cache files that are
    missing required columns - shape problems that would otherwise silently
    produce a wrong or empty result.
    """


@dataclass(frozen=True)
class GeocodeResult:
    """One station's geocoding outcome.

    Attributes:
        station_id: Station identifier (as in ``stations.json``).
        name: Station's human-readable name.
        lat: Latitude of the resolved point, or ``None`` if unresolved.
        lon: Longitude of the resolved point, or ``None`` if unresolved.
        geocode_query: Exact query string sent to Nominatim.
        resolved: ``True`` if Nominatim returned a match that fell within
            `MUENSTER_BOUNDS`; ``False`` if there was no match, or the top
            match fell outside the box.
    """

    station_id: str
    name: str
    lat: float | None
    lon: float | None
    geocode_query: str
    resolved: bool


def is_within_bounds(
    lat: float,
    lon: float,
    bounds: tuple[float, float, float, float] = MUENSTER_BOUNDS,
) -> bool:
    """Checks whether a point falls within a lat/lon bounding box.

    Args:
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
        bounds: ``(lat_min, lat_max, lon_min, lon_max)``.

    Returns:
        ``True`` if `lat`/`lon` fall within `bounds` (inclusive).
    """
    lat_min, lat_max, lon_min, lon_max = bounds
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def build_geocode_query(name: str, suffix: str = DEFAULT_QUERY_SUFFIX) -> str:
    """Builds the Nominatim query string for a station name.

    Args:
        name: Station's human-readable name.
        suffix: Location-qualifying suffix appended to disambiguate generic
            names (e.g. plain ``"Neutor"``) from similarly-named places
            elsewhere.

    Returns:
        `name` with `suffix` appended.
    """
    return f"{name}{suffix}"


def load_location_cache(path: Path) -> pd.DataFrame:
    """Loads a station-location cache CSV, or an empty frame if absent.

    Args:
        path: Path to the cache CSV (see `CACHE_COLUMNS`).

    Returns:
        DataFrame with `CACHE_COLUMNS`, ``station_id`` read back as ``str``
        (so leading-digit ids are never misinterpreted). Empty (zero rows,
        correct columns) if `path` does not exist yet - the first run of a
        fresh notebook is expected to start from nothing.

    Raises:
        GeocodeError: if the file exists but is missing an expected column.
    """
    if not path.exists():
        empty = pd.DataFrame(columns=CACHE_COLUMNS)
        return empty.astype({"resolved": bool})

    df = pd.read_csv(path, dtype={"station_id": str})
    missing = set(CACHE_COLUMNS) - set(df.columns)
    if missing:
        raise GeocodeError(f"{path} is missing expected column(s): {sorted(missing)}.")
    return df[CACHE_COLUMNS]


def save_location_cache(df: pd.DataFrame, path: Path) -> Path:
    """Saves a station-location cache DataFrame to CSV.

    Args:
        df: DataFrame with (at least) `CACHE_COLUMNS`.
        path: Destination CSV path; parent directory created if missing.

    Returns:
        `path`.

    Raises:
        GeocodeError: if `df` is missing an expected column.
    """
    missing = set(CACHE_COLUMNS) - set(df.columns)
    if missing:
        raise GeocodeError(
            f"DataFrame is missing expected column(s): {sorted(missing)}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df[CACHE_COLUMNS].to_csv(path, index=False)
    return path


def merge_location_cache(
    existing: pd.DataFrame, new_results: pd.DataFrame
) -> pd.DataFrame:
    """Merges newly geocoded rows into an existing cache, keyed by station_id.

    Args:
        existing: Cache as returned by `load_location_cache` (possibly
            empty).
        new_results: Newly geocoded rows (`CACHE_COLUMNS`) to merge in.

    Returns:
        Combined DataFrame, one row per distinct ``station_id``. Where a
        ``station_id`` appears in both inputs, the `new_results` row wins
        (allows deliberately re-geocoding and overwriting a previously
        cached station), sorted by `station_id` for a deterministic row
        order regardless of input order.

    Raises:
        GeocodeError: if either input is missing an expected column.
    """
    for frame, label in ((existing, "existing"), (new_results, "new_results")):
        missing = set(CACHE_COLUMNS) - set(frame.columns)
        if missing:
            raise GeocodeError(
                f"{label} is missing expected column(s): {sorted(missing)}."
            )
    combined = pd.concat([existing, new_results], ignore_index=True)
    combined = combined.drop_duplicates(subset="station_id", keep="last")
    return combined.sort_values("station_id").reset_index(drop=True)


def _query_nominatim(
    query: str, session: requests.Session, timeout: float
) -> list[dict[str, object]]:
    """Sends one Nominatim search request and returns its parsed results.

    Not unit-tested directly against the live API (see module docstring);
    tests stub `session` with a fake object exposing a compatible `get`.

    Args:
        query: Free-text search query.
        session: ``requests.Session``-like object (must expose `.get`).
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response: a list of candidate match dicts (Nominatim's
        ``jsonv2`` format), ranked most-likely first. Empty list if no
        candidate was found.

    Raises:
        GeocodeError: on any request/HTTP failure, or a non-JSON response.
    """
    headers = {"User-Agent": USER_AGENT}
    params = {"q": query, "format": "jsonv2", "limit": 1}
    try:
        response = session.get(
            NOMINATIM_SEARCH_URL, params=params, headers=headers, timeout=timeout
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise GeocodeError(
            f"Nominatim request failed for query {query!r}: {exc}"
        ) from exc
    try:
        return response.json()
    except ValueError as exc:
        raise GeocodeError(
            f"Nominatim response for query {query!r} is not valid JSON."
        ) from exc


def geocode_station(
    station_id: str,
    name: str,
    query: str,
    session: requests.Session,
    timeout: float = 10.0,
    bounds: tuple[float, float, float, float] = MUENSTER_BOUNDS,
) -> GeocodeResult:
    """Geocodes a single station name via Nominatim, one HTTP request.

    Callers doing multiple stations must space calls at least
    `MIN_REQUEST_INTERVAL_SECONDS` apart themselves (see `geocode_stations`
    for the rate-limited batch version); this function makes exactly one
    request and does not sleep.

    Args:
        station_id: Station identifier, carried through to the result.
        name: Station's human-readable name, carried through to the result.
        query: Exact query string to send to Nominatim (see
            `build_geocode_query`, or a manually-chosen override for a
            station that does not geocode cleanly on its own name).
        session: ``requests.Session``-like object.
        timeout: Request timeout in seconds.
        bounds: Münster bounding box a match must fall within to be
            accepted (see `is_within_bounds`).

    Returns:
        A `GeocodeResult`. ``resolved=False`` (with ``lat``/``lon`` set to
        the out-of-bounds coordinates, or ``None`` if there was no match at
        all) rather than raising, so a batch call can report all stations
        - including unresolved ones - in one pass.

    Raises:
        GeocodeError: if the Nominatim request itself fails (network/HTTP/
            JSON-parsing error) - as opposed to a clean "no match" or
            out-of-bounds result, which is not an error.
    """
    results = _query_nominatim(query, session, timeout)
    if not results:
        return GeocodeResult(
            station_id=station_id,
            name=name,
            lat=None,
            lon=None,
            geocode_query=query,
            resolved=False,
        )

    top = results[0]
    lat = float(top["lat"])
    lon = float(top["lon"])
    return GeocodeResult(
        station_id=station_id,
        name=name,
        lat=lat,
        lon=lon,
        geocode_query=query,
        resolved=is_within_bounds(lat, lon, bounds),
    )


def geocode_stations(
    stations: pd.DataFrame,
    cache_path: Path,
    session: requests.Session | None = None,
    timeout: float = 10.0,
    min_request_interval: float = MIN_REQUEST_INTERVAL_SECONDS,
    query_overrides: dict[str, str] | None = None,
    bounds: tuple[float, float, float, float] = MUENSTER_BOUNDS,
) -> pd.DataFrame:
    """Geocodes stations missing from the cache, then updates the cache file.

    Idempotent: a station id already present in the cache at `cache_path` is
    never re-geocoded, so a fresh ``Restart & Run All`` of a notebook only
    hits the network for genuinely new stations, not all 23 every time.

    Args:
        stations: DataFrame with (at least) ``station_id`` and ``name``
            columns (e.g. loaded from ``stations.json``).
        cache_path: Path to the on-disk cache CSV (`CACHE_COLUMNS`); read at
            the start and overwritten (merged, not truncated) at the end.
        session: Optional ``requests.Session`` for connection reuse or
            mocking in tests. A new one is created if not given.
        timeout: Per-request timeout in seconds.
        min_request_interval: Minimum seconds between the start of
            consecutive Nominatim requests (Nominatim's usage policy caps
            this at 1 request/second; do not lower below
            `MIN_REQUEST_INTERVAL_SECONDS` without re-reading that policy).
        query_overrides: Optional mapping from ``station_id`` to a manually
            chosen query string, for stations whose name does not geocode
            cleanly on its own (see module docstring) - e.g. a canal-
            promenade segment geocoded via a nearby cross-street or the
            general "Kanalpromenade, Münster, Germany" instead of the exact
            segment name.
        bounds: Münster bounding box a match must fall within to be
            accepted.

    Returns:
        The full merged cache (previously-cached rows plus any newly
        geocoded ones), `CACHE_COLUMNS`, one row per station in `stations`
        that has ever been geocoded (a superset of `stations` if the cache
        already held other stations; a subset if some `stations` rows have
        not been geocoded yet due to an error partway through - already-
        geocoded rows from this call are still saved).

    Raises:
        GeocodeError: if `stations` is missing ``station_id``/``name``, or
            an individual `geocode_station` call raises (a network/HTTP/
            JSON failure - not a clean unresolved result). Rows
            successfully geocoded before the failure are still written to
            `cache_path` before the exception propagates, so a re-run does
            not lose that progress.
    """
    missing_columns = {"station_id", "name"} - set(stations.columns)
    if missing_columns:
        raise GeocodeError(f"stations is missing column(s): {sorted(missing_columns)}.")

    existing = load_location_cache(cache_path)
    already_cached = set(existing["station_id"])
    to_geocode = stations[~stations["station_id"].astype(str).isin(already_cached)]

    if to_geocode.empty:
        return existing

    http = session if session is not None else requests.Session()
    overrides = query_overrides or {}
    new_rows: list[GeocodeResult] = []
    try:
        for i, row in enumerate(to_geocode.itertuples(index=False)):
            if i > 0:
                time.sleep(min_request_interval)
            station_id = str(row.station_id)
            query = overrides.get(station_id, build_geocode_query(row.name))
            new_rows.append(
                geocode_station(
                    station_id, row.name, query, http, timeout=timeout, bounds=bounds
                )
            )
    finally:
        if new_rows:
            new_df = pd.DataFrame([r.__dict__ for r in new_rows], columns=CACHE_COLUMNS)
            existing = merge_location_cache(existing, new_df)
            save_location_cache(existing, cache_path)

    return existing

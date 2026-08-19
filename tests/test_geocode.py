"""Tests for `muenster_bike_forecast.data.geocode`.

Pure/validation logic (bounding-box check, query building, cache-merge
logic) is tested directly. `geocode_station`/`geocode_stations` are tested
against a fake in-memory session — no live network calls are made.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import muenster_bike_forecast.data.geocode as geocode_module
from muenster_bike_forecast.data.geocode import (
    CACHE_COLUMNS,
    GeocodeError,
    build_geocode_query,
    geocode_station,
    geocode_stations,
    is_within_bounds,
    load_location_cache,
    merge_location_cache,
    save_location_cache,
)

MUENSTER_BOUNDS = (51.90, 52.02, 7.55, 7.70)

# ---------------------------------------------------------------------------
# is_within_bounds
# ---------------------------------------------------------------------------


def test_is_within_bounds_true_for_point_inside_muenster() -> None:
    assert is_within_bounds(51.96, 7.62, MUENSTER_BOUNDS) is True


def test_is_within_bounds_false_for_point_outside_muenster() -> None:
    # Berlin, roughly - well outside the Münster box.
    assert is_within_bounds(52.52, 13.40, MUENSTER_BOUNDS) is False


def test_is_within_bounds_true_on_box_edge() -> None:
    assert is_within_bounds(51.90, 7.55, MUENSTER_BOUNDS) is True


# ---------------------------------------------------------------------------
# build_geocode_query
# ---------------------------------------------------------------------------


def test_build_geocode_query_appends_default_suffix() -> None:
    assert build_geocode_query("Bismarckallee") == "Bismarckallee, Münster, Germany"


def test_build_geocode_query_accepts_custom_suffix() -> None:
    assert build_geocode_query("Foo", suffix=", Bar") == "Foo, Bar"


# ---------------------------------------------------------------------------
# load_location_cache / save_location_cache
# ---------------------------------------------------------------------------


def test_load_location_cache_returns_empty_frame_if_file_missing(
    tmp_path: Path,
) -> None:
    cache = load_location_cache(tmp_path / "does_not_exist.csv")

    assert list(cache.columns) == CACHE_COLUMNS
    assert cache.empty


def test_save_then_load_location_cache_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "station_locations.csv"
    df = pd.DataFrame(
        [
            {
                "station_id": "300038855",
                "name": "Bismarckallee",
                "lat": 51.9508714,
                "lon": 7.6070912,
                "geocode_query": "Bismarckallee, Münster, Germany",
                "resolved": True,
            }
        ]
    )

    save_location_cache(df, path)
    loaded = load_location_cache(path)

    assert loaded.loc[0, "station_id"] == "300038855"
    assert loaded.loc[0, "resolved"] == True  # noqa: E712
    assert loaded.loc[0, "lat"] == pytest.approx(51.9508714)


def test_save_location_cache_raises_on_missing_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"station_id": ["1"]})
    with pytest.raises(GeocodeError):
        save_location_cache(df, tmp_path / "out.csv")


def test_load_location_cache_raises_on_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"station_id": ["1"]}).to_csv(path, index=False)
    with pytest.raises(GeocodeError):
        load_location_cache(path)


# ---------------------------------------------------------------------------
# merge_location_cache
# ---------------------------------------------------------------------------


def _row(
    station_id: str, lat: float = 51.96, resolved: bool = True
) -> dict[str, object]:
    return {
        "station_id": station_id,
        "name": f"Station {station_id}",
        "lat": lat,
        "lon": 7.62,
        "geocode_query": f"Station {station_id}, Münster, Germany",
        "resolved": resolved,
    }


def test_merge_location_cache_combines_disjoint_station_ids() -> None:
    existing = pd.DataFrame([_row("1")])
    new = pd.DataFrame([_row("2")])

    merged = merge_location_cache(existing, new)

    assert sorted(merged["station_id"]) == ["1", "2"]


def test_merge_location_cache_new_result_overwrites_existing_row() -> None:
    existing = pd.DataFrame([_row("1", lat=0.0, resolved=False)])
    new = pd.DataFrame([_row("1", lat=51.96, resolved=True)])

    merged = merge_location_cache(existing, new)

    assert len(merged) == 1
    assert merged.loc[0, "lat"] == pytest.approx(51.96)
    assert merged.loc[0, "resolved"] == True  # noqa: E712


def test_merge_location_cache_raises_on_missing_column() -> None:
    existing = pd.DataFrame({"station_id": ["1"]})
    new = pd.DataFrame([_row("2")])
    with pytest.raises(GeocodeError):
        merge_location_cache(existing, new)


# ---------------------------------------------------------------------------
# geocode_station / geocode_stations (fake session, no live network calls)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: list[dict[str, object]]):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list[dict[str, object]]:
        return self._payload


class _FakeSession:
    """Routes GET requests by the query string's ``q`` param."""

    def __init__(self, routes: dict[str, list[dict[str, object]]]):
        self._routes = routes
        self.requests: list[str] = []

    def get(self, url, params=None, headers=None, timeout=None):  # noqa: ANN001
        query = params["q"]
        self.requests.append(query)
        assert "User-Agent" in headers
        if query not in self._routes:
            raise AssertionError(f"Unexpected query in test: {query!r}")
        return _FakeResponse(self._routes[query])


def _nominatim_hit(lat: str, lon: str) -> list[dict[str, object]]:
    return [{"lat": lat, "lon": lon, "display_name": "some place"}]


def test_geocode_station_resolves_a_clean_match() -> None:
    session = _FakeSession(
        {"Bismarckallee, Münster, Germany": _nominatim_hit("51.9508714", "7.6070912")}
    )

    result = geocode_station(
        "300038855",
        "Bismarckallee",
        "Bismarckallee, Münster, Germany",
        session,
        bounds=MUENSTER_BOUNDS,
    )

    assert result.resolved is True
    assert result.lat == pytest.approx(51.9508714)
    assert result.lon == pytest.approx(7.6070912)


def test_geocode_station_flags_out_of_bounds_match_as_unresolved() -> None:
    # Berlin coordinates - a plausible mismatch for a too-generic name.
    session = _FakeSession(
        {"Neutor, Münster, Germany": _nominatim_hit("52.52", "13.40")}
    )

    result = geocode_station(
        "100035541",
        "Neutor",
        "Neutor, Münster, Germany",
        session,
        bounds=MUENSTER_BOUNDS,
    )

    assert result.resolved is False
    assert result.lat == pytest.approx(52.52)


def test_geocode_station_flags_no_match_as_unresolved_with_null_coords() -> None:
    session = _FakeSession({"Nowhere, Münster, Germany": []})

    result = geocode_station(
        "999", "Nowhere", "Nowhere, Münster, Germany", session, bounds=MUENSTER_BOUNDS
    )

    assert result.resolved is False
    assert result.lat is None
    assert result.lon is None


def test_geocode_station_raises_on_non_list_response() -> None:
    class _BadResponse(_FakeResponse):
        def json(self) -> dict[str, object]:  # type: ignore[override]
            return {"error": "not a list"}

    class _BadSession(_FakeSession):
        def get(self, url, params=None, headers=None, timeout=None):  # noqa: ANN001
            return _BadResponse({})

    with pytest.raises(GeocodeError):
        geocode_station("1", "Bad", "Bad, Münster, Germany", _BadSession({}))


def test_geocode_station_raises_on_result_missing_lat_lon() -> None:
    class _BadSession(_FakeSession):
        def get(self, url, params=None, headers=None, timeout=None):  # noqa: ANN001
            return _FakeResponse([{"display_name": "no coords here"}])

    with pytest.raises(GeocodeError):
        geocode_station("1", "Bad", "Bad, Münster, Germany", _BadSession({}))


def test_geocode_station_raises_on_non_numeric_lat_lon() -> None:
    class _BadSession(_FakeSession):
        def get(self, url, params=None, headers=None, timeout=None):  # noqa: ANN001
            return _FakeResponse([{"lat": "not-a-number", "lon": "7.6"}])

    with pytest.raises(GeocodeError):
        geocode_station("1", "Bad", "Bad, Münster, Germany", _BadSession({}))


def test_geocode_stations_only_queries_cache_misses(tmp_path: Path) -> None:
    cache_path = tmp_path / "station_locations.csv"
    save_location_cache(pd.DataFrame([_row("1")]), cache_path)

    stations = pd.DataFrame(
        [
            {"station_id": "1", "name": "Station 1"},
            {"station_id": "2", "name": "Station 2"},
        ]
    )
    session = _FakeSession(
        {"Station 2, Münster, Germany": _nominatim_hit("51.96", "7.62")}
    )

    result = geocode_stations(
        stations, cache_path, session=session, min_request_interval=0
    )

    # Only the cache miss ("2") should have triggered a request.
    assert session.requests == ["Station 2, Münster, Germany"]
    assert sorted(result["station_id"]) == ["1", "2"]
    reloaded = load_location_cache(cache_path)
    assert sorted(reloaded["station_id"]) == ["1", "2"]


def test_geocode_stations_is_a_noop_when_everything_is_cached(tmp_path: Path) -> None:
    cache_path = tmp_path / "station_locations.csv"
    save_location_cache(pd.DataFrame([_row("1")]), cache_path)

    stations = pd.DataFrame([{"station_id": "1", "name": "Station 1"}])
    session = _FakeSession({})

    result = geocode_stations(
        stations, cache_path, session=session, min_request_interval=0
    )

    assert session.requests == []
    assert list(result["station_id"]) == ["1"]


def test_geocode_stations_applies_query_overrides(tmp_path: Path) -> None:
    cache_path = tmp_path / "station_locations.csv"
    stations = pd.DataFrame(
        [{"station_id": "1", "name": "Kanalpromenade, Abschnitt 5"}]
    )
    session = _FakeSession(
        {"Kanalpromenade, Münster, Germany": _nominatim_hit("51.95", "7.62")}
    )

    result = geocode_stations(
        stations,
        cache_path,
        session=session,
        min_request_interval=0,
        query_overrides={"1": "Kanalpromenade, Münster, Germany"},
    )

    assert result.loc[0, "geocode_query"] == "Kanalpromenade, Münster, Germany"
    assert result.loc[0, "resolved"] == True  # noqa: E712


def test_geocode_stations_raises_on_missing_column(tmp_path: Path) -> None:
    stations = pd.DataFrame({"station_id": ["1"]})  # no "name" column
    with pytest.raises(GeocodeError):
        geocode_stations(stations, tmp_path / "out.csv", session=_FakeSession({}))


class _TrackingSession(_FakeSession):
    """Same as `_FakeSession`, but records whether `.close()` was called."""

    def __init__(self, routes: dict[str, list[dict[str, object]]]):
        super().__init__(routes)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_geocode_stations_closes_the_session_it_creates_itself(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "station_locations.csv"
    stations = pd.DataFrame([{"station_id": "1", "name": "Station 1"}])
    tracking_session = _TrackingSession(
        {"Station 1, Münster, Germany": _nominatim_hit("51.96", "7.62")}
    )
    monkeypatch.setattr(geocode_module.requests, "Session", lambda: tracking_session)

    geocode_stations(stations, cache_path, min_request_interval=0)

    assert tracking_session.closed is True


def test_geocode_stations_does_not_close_a_caller_supplied_session(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "station_locations.csv"
    stations = pd.DataFrame([{"station_id": "1", "name": "Station 1"}])
    tracking_session = _TrackingSession(
        {"Station 1, Münster, Germany": _nominatim_hit("51.96", "7.62")}
    )

    geocode_stations(
        stations, cache_path, session=tracking_session, min_request_interval=0
    )

    assert tracking_session.closed is False

"""Tests for `muenster_bike_forecast.data.bike_counts`.

All tests use small, local, hand-written sample data (CSV text / JSON-like
dicts). Where HTTP-boundary functions are exercised, `requests.get` is
monkeypatched to a fake response - no live network calls are made anywhere
in this file.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import requests

from muenster_bike_forecast.data import bike_counts
from muenster_bike_forecast.data.bike_counts import (
    BikeCountDataError,
    Channel,
    Station,
    find_missing_intervals,
    load_station_data,
    parse_station_csv,
    parse_station_index,
    save_station_data,
    save_stations_index,
    station_csv_url,
    summarize_coverage,
)

VALID_CSV = """Datetime,300038855 (Bismarckallee),353420516 (Bismarckallee Fahrräder OUT),353420517 (Bismarckallee Fahrräder IN),300038855-status,353420516-status,353420517-status
2025-01-01 00:00,2,1,1,0,0,0
2025-01-01 00:15,18,14,4,0,0,0
2025-01-01 00:30,22,17,5,0,0,0
2025-01-01 01:00,33,12,21,0,0,0
"""
# Note: 2025-01-01 00:45 is intentionally missing above -> one gap.


# ---------------------------------------------------------------------------
# parse_station_csv
# ---------------------------------------------------------------------------


def test_parse_station_csv_valid_returns_expected_shape() -> None:
    df = parse_station_csv(VALID_CSV, station_id="300038855")

    assert list(df.columns) == [
        "station_id",
        "datetime",
        "300038855 (Bismarckallee)",
        "353420516 (Bismarckallee Fahrräder OUT)",
        "353420517 (Bismarckallee Fahrräder IN)",
        "300038855-status",
        "353420516-status",
        "353420517-status",
    ]
    assert len(df) == 4
    assert (df["station_id"] == "300038855").all()
    assert pd.api.types.is_datetime64_any_dtype(df["datetime"])
    assert df["300038855 (Bismarckallee)"].tolist() == [2, 18, 22, 33]
    # Sorted by datetime.
    assert df["datetime"].is_monotonic_increasing


def test_parse_station_csv_missing_datetime_column_raises() -> None:
    csv_text = "300038855 (Bismarckallee),300038855-status\n1,0\n"
    with pytest.raises(BikeCountDataError, match="Datetime"):
        parse_station_csv(csv_text, station_id="300038855")


def test_parse_station_csv_count_without_status_raises() -> None:
    csv_text = "Datetime,300038855 (Bismarckallee)\n2025-01-01 00:00,2\n"
    with pytest.raises(BikeCountDataError, match="mismatched"):
        parse_station_csv(csv_text, station_id="300038855")


def test_parse_station_csv_status_without_count_raises() -> None:
    # A file with only a status column and no count column at all is caught
    # by the "no count columns" check before the pairing check runs.
    csv_text = "Datetime,300038855-status\n2025-01-01 00:00,0\n"
    with pytest.raises(BikeCountDataError, match="no count columns"):
        parse_station_csv(csv_text, station_id="300038855")


def test_parse_station_csv_status_without_matching_count_raises() -> None:
    # A status column whose id has no corresponding count column, but the
    # file does have *some* count column - hits the pairing check.
    csv_text = "Datetime,1 (Foo),1-status,2-status\n" "2025-01-01 00:00,2,0,0\n"
    with pytest.raises(BikeCountDataError, match="mismatched"):
        parse_station_csv(csv_text, station_id="300038855")


def test_parse_station_csv_unrecognized_column_raises() -> None:
    csv_text = (
        "Datetime,300038855 (Bismarckallee),300038855-status,extra_column\n"
        "2025-01-01 00:00,2,0,foo\n"
    )
    with pytest.raises(BikeCountDataError, match="unrecognized column"):
        parse_station_csv(csv_text, station_id="300038855")


def test_parse_station_csv_unparsable_datetime_raises() -> None:
    csv_text = (
        "Datetime,300038855 (Bismarckallee),300038855-status\n" "not-a-date,2,0\n"
    )
    with pytest.raises(BikeCountDataError, match="unparsable"):
        parse_station_csv(csv_text, station_id="300038855")


def test_parse_station_csv_non_numeric_value_raises() -> None:
    csv_text = (
        "Datetime,300038855 (Bismarckallee),300038855-status\n"
        "2025-01-01 00:00,not_a_number,0\n"
    )
    with pytest.raises(BikeCountDataError, match="non-numeric"):
        parse_station_csv(csv_text, station_id="300038855")


def test_parse_station_csv_duplicate_timestamp_raises() -> None:
    csv_text = (
        "Datetime,300038855 (Bismarckallee),300038855-status\n"
        "2025-01-01 00:00,2,0\n"
        "2025-01-01 00:00,3,0\n"
    )
    with pytest.raises(BikeCountDataError, match="duplicate"):
        parse_station_csv(csv_text, station_id="300038855")


def test_parse_station_csv_empty_cell_is_missing_value_not_error() -> None:
    # An empty count value (real sensor gap) must not be mistaken for a
    # schema violation - only genuinely non-numeric garbage should raise.
    csv_text = (
        "Datetime,300038855 (Bismarckallee),300038855-status\n" "2025-01-01 00:00,,0\n"
    )
    df = parse_station_csv(csv_text, station_id="300038855")
    assert pd.isna(df["300038855 (Bismarckallee)"].iloc[0])


def test_parse_station_csv_unparseable_garbage_raises() -> None:
    with pytest.raises(BikeCountDataError):
        parse_station_csv("not,a,csv,at,all\n\x00\x00", station_id="x")


# ---------------------------------------------------------------------------
# parse_station_index
# ---------------------------------------------------------------------------


def test_parse_station_index_valid() -> None:
    raw = [
        {
            "name": "Bismarckallee",
            "directory": "300038855",
            "start": 2023,
            "channels": [[300038855, "Bismarckallee"], [353420516, "OUT"]],
        }
    ]
    stations = parse_station_index(raw)
    assert stations == [
        Station(
            station_id="300038855",
            name="Bismarckallee",
            start_year=2023,
            channels=(
                Channel(channel_id=300038855, description="Bismarckallee"),
                Channel(channel_id=353420516, description="OUT"),
            ),
        )
    ]


def test_parse_station_index_not_a_list_raises() -> None:
    with pytest.raises(BikeCountDataError, match="non-empty JSON list"):
        parse_station_index({"not": "a list"})


def test_parse_station_index_empty_list_raises() -> None:
    with pytest.raises(BikeCountDataError, match="non-empty JSON list"):
        parse_station_index([])


def test_parse_station_index_missing_key_raises() -> None:
    raw = [{"name": "X", "directory": "1", "start": 2023}]
    with pytest.raises(BikeCountDataError, match="missing required keys"):
        parse_station_index(raw)


def test_parse_station_index_malformed_channels_raises() -> None:
    raw = [{"name": "X", "directory": "1", "start": 2023, "channels": "not-a-list"}]
    with pytest.raises(BikeCountDataError, match="malformed 'channels'"):
        parse_station_index(raw)


def test_parse_station_index_no_channels_raises() -> None:
    raw = [{"name": "X", "directory": "1", "start": 2023, "channels": []}]
    with pytest.raises(BikeCountDataError, match="no channels"):
        parse_station_index(raw)


@pytest.mark.parametrize(
    "directory",
    ["../../etc/evil", "..", "a/b", "a\\b", ""],
)
def test_parse_station_index_rejects_unsafe_directory(directory: str) -> None:
    raw = [
        {
            "name": "X",
            "directory": directory,
            "start": 2023,
            "channels": [[1, "Foo"]],
        }
    ]
    with pytest.raises(BikeCountDataError, match="safe identifier"):
        parse_station_index(raw)


@pytest.mark.parametrize("start_year", [1900, 1999, 2999])
def test_parse_station_index_rejects_implausible_start_year(start_year: int) -> None:
    raw = [
        {
            "name": "X",
            "directory": "1",
            "start": start_year,
            "channels": [[1, "Foo"]],
        }
    ]
    with pytest.raises(BikeCountDataError, match="implausible"):
        parse_station_index(raw)


# ---------------------------------------------------------------------------
# station_csv_url
# ---------------------------------------------------------------------------


def test_station_csv_url_format() -> None:
    url = station_csv_url("300038855", 2025, 1)
    assert url == (
        "https://raw.githubusercontent.com/od-ms/radverkehr-zaehlstellen/"
        "master/300038855/2025-01.csv"
    )


def test_station_csv_url_zero_pads_month() -> None:
    url = station_csv_url("1", 2025, 3)
    assert url.endswith("/1/2025-03.csv")


@pytest.mark.parametrize("month", [0, 13, -1])
def test_station_csv_url_invalid_month_raises(month: int) -> None:
    with pytest.raises(ValueError):
        station_csv_url("300038855", 2025, month)


@pytest.mark.parametrize("station_id", ["../../etc/evil", "..", "a/b", "a\\b", ""])
def test_station_csv_url_rejects_unsafe_station_id(station_id: str) -> None:
    with pytest.raises(BikeCountDataError, match="safe identifier"):
        station_csv_url(station_id, 2025, 1)


# ---------------------------------------------------------------------------
# find_missing_intervals / summarize_coverage
# ---------------------------------------------------------------------------


def test_find_missing_intervals_detects_gap() -> None:
    df = parse_station_csv(VALID_CSV, station_id="300038855")
    missing = find_missing_intervals(df)
    assert list(missing) == [pd.Timestamp("2025-01-01 00:45")]


def test_find_missing_intervals_no_gap_returns_empty() -> None:
    df = pd.DataFrame(
        {"datetime": pd.date_range("2025-01-01 00:00", periods=4, freq="15min")}
    )
    missing = find_missing_intervals(df)
    assert len(missing) == 0


def test_find_missing_intervals_missing_column_raises() -> None:
    with pytest.raises(BikeCountDataError, match="datetime"):
        find_missing_intervals(pd.DataFrame({"foo": [1, 2]}))


def test_find_missing_intervals_empty_df_raises() -> None:
    with pytest.raises(BikeCountDataError, match="empty"):
        find_missing_intervals(pd.DataFrame({"datetime": pd.to_datetime([])}))


def test_summarize_coverage_reports_expected_counts() -> None:
    df = parse_station_csv(VALID_CSV, station_id="300038855")
    summary = summarize_coverage(df, station_id="300038855")
    assert summary["station_id"] == "300038855"
    assert summary["n_records"] == 4
    assert summary["n_missing"] == 1
    assert summary["n_expected"] == 5
    assert summary["missing_timestamps"] == [pd.Timestamp("2025-01-01 00:45")]


# ---------------------------------------------------------------------------
# save_station_data / load_station_data / save_stations_index
# ---------------------------------------------------------------------------


def test_save_and_load_station_data_roundtrip(tmp_path: Path) -> None:
    df = parse_station_csv(VALID_CSV, station_id="300038855")
    path = save_station_data(df, tmp_path, "300038855")
    assert path == tmp_path / "300038855.csv"

    loaded = load_station_data(path)
    assert len(loaded) == len(df)
    assert pd.api.types.is_datetime64_any_dtype(loaded["datetime"])


def test_save_station_data_is_idempotent(tmp_path: Path) -> None:
    df = parse_station_csv(VALID_CSV, station_id="300038855")
    path_a = save_station_data(df, tmp_path, "300038855")
    content_a = path_a.read_text(encoding="utf-8")

    # Re-run with rows shuffled and duplicated - output must be identical.
    shuffled = pd.concat([df, df]).sample(frac=1, random_state=0)
    path_b = save_station_data(shuffled, tmp_path, "300038855")
    content_b = path_b.read_text(encoding="utf-8")

    assert content_a == content_b


def test_save_station_data_missing_datetime_raises(tmp_path: Path) -> None:
    with pytest.raises(BikeCountDataError):
        save_station_data(pd.DataFrame({"foo": [1]}), tmp_path, "x")


@pytest.mark.parametrize("station_id", ["../../etc/evil", "..", "a/b", "a\\b", ""])
def test_save_station_data_rejects_unsafe_station_id(
    station_id: str, tmp_path: Path
) -> None:
    df = pd.DataFrame({"datetime": pd.to_datetime(["2025-01-01 00:00"])})
    with pytest.raises(BikeCountDataError, match="safe identifier"):
        save_station_data(df, tmp_path, station_id)


def test_save_station_data_sanitizes_formula_like_headers(tmp_path: Path) -> None:
    # A malicious/untrusted column name starting with a spreadsheet
    # formula-trigger character must not reach the output CSV as-is.
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2025-01-01 00:00"]),
            "=cmd|'/c calc'!A1": [1],
        }
    )
    path = save_station_data(df, tmp_path, "300038855")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "'=cmd|'/c calc'!A1" in header
    assert header.split(",")[1].startswith("'")


# ---------------------------------------------------------------------------
# HTTP-boundary functions (list_stations / fetch_station_month /
# fetch_station_data), exercised with a mocked `requests.get` - no live
# network calls.
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Minimal stand-in for `requests.Response` used to mock HTTP calls."""

    def __init__(
        self, status_code: int = 200, text: str = "", json_data: object = None
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json_data = json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> object:
        return self._json_data


def test_list_stations_uses_mocked_index(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = [
        {
            "name": "Bismarckallee",
            "directory": "300038855",
            "start": 2023,
            "channels": [[300038855, "Bismarckallee"]],
        }
    ]
    monkeypatch.setattr(
        bike_counts.requests,
        "get",
        lambda *a, **k: _FakeResponse(200, json_data=raw),
    )
    stations = bike_counts.list_stations()
    assert stations == parse_station_index(raw)


def test_list_stations_http_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bike_counts.requests, "get", lambda *a, **k: _FakeResponse(500))
    with pytest.raises(BikeCountDataError, match="Failed to fetch"):
        bike_counts.list_stations()


def test_fetch_station_month_404_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bike_counts.requests, "get", lambda *a, **k: _FakeResponse(404))
    assert bike_counts.fetch_station_month("300038855", 2019, 1) is None


def test_fetch_station_month_success_returns_parsed_df(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bike_counts.requests,
        "get",
        lambda *a, **k: _FakeResponse(200, text=VALID_CSV),
    )
    df = bike_counts.fetch_station_month("300038855", 2025, 1)
    assert df is not None
    assert len(df) == 4


def test_fetch_station_month_oversized_response_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bike_counts, "MAX_DOWNLOAD_BYTES", 10)
    monkeypatch.setattr(
        bike_counts.requests,
        "get",
        lambda *a, **k: _FakeResponse(200, text=VALID_CSV),
    )
    with pytest.raises(BikeCountDataError, match="exceeds"):
        bike_counts.fetch_station_month("300038855", 2025, 1)


def test_fetch_station_data_combines_months_skips_404s_and_dedupes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jan_csv = (
        "Datetime,1 (Foo),1-status\n" "2025-01-01 00:00,1,0\n" "2025-01-01 00:15,2,0\n"
    )
    feb_csv = "Datetime,1 (Foo),1-status\n" "2025-02-01 00:00,3,0\n"

    def fake_get(url: str, *args, **kwargs) -> _FakeResponse:
        if url.endswith("2025-01.csv"):
            return _FakeResponse(200, text=jan_csv)
        if url.endswith("2025-02.csv"):
            return _FakeResponse(200, text=feb_csv)
        return _FakeResponse(404)

    monkeypatch.setattr(bike_counts.requests, "get", fake_get)

    station = Station(station_id="1", name="Test", start_year=2025, channels=())
    df = bike_counts.fetch_station_data(station, as_of=date(2025, 2, 15))

    assert len(df) == 3
    assert df["datetime"].is_monotonic_increasing
    assert df["datetime"].nunique() == 3


def test_fetch_station_data_no_months_returns_empty_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bike_counts.requests, "get", lambda *a, **k: _FakeResponse(404))
    station = Station(station_id="1", name="Test", start_year=2025, channels=())
    df = bike_counts.fetch_station_data(station, as_of=date(2025, 3, 1))
    assert df.empty
    assert list(df.columns) == ["station_id", "datetime"]


def test_save_stations_index_writes_json(tmp_path: Path) -> None:
    stations = [
        Station(
            station_id="1",
            name="Test",
            start_year=2023,
            channels=(Channel(channel_id=1, description="Test"),),
        )
    ]
    path = save_stations_index(stations, tmp_path)
    assert path == tmp_path / "stations.json"
    assert "Test" in path.read_text(encoding="utf-8")

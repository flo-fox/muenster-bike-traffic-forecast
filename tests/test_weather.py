"""Tests for muenster_bike_forecast.data.weather.

These tests use small, local/in-memory sample data (built to mirror the
real DWD ``produkt_*.txt`` format) and mocked HTTP responses — no live
network calls are made.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from muenster_bike_forecast.data import weather

# ---------------------------------------------------------------------------
# Helpers for building fake DWD payloads
# ---------------------------------------------------------------------------


def _tu_product_text(rows: list[str]) -> str:
    header = "STATIONS_ID;MESS_DATUM;QN_9;TT_TU;RF_TU;eor"
    return "\n".join([header, *rows, ""])


def _make_zip_bytes(product_filename: str, product_text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr(product_filename, product_text)
        archive.writestr("Metadaten_Geographie_01766.txt", "irrelevant metadata")
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes = b"", text: str = "", status_code: int = 200):
        self.content = content
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise weather.requests.HTTPError(f"status {self.status_code}")


class _FakeSession:
    """Minimal stand-in for requests.Session, routes GET by URL suffix."""

    def __init__(self, routes: dict[str, _FakeResponse]):
        self._routes = routes

    def get(self, url: str, timeout: float = 30.0) -> _FakeResponse:
        for suffix, response in self._routes.items():
            if url.endswith(suffix):
                return response
        raise AssertionError(f"Unexpected URL requested in test: {url}")


# ---------------------------------------------------------------------------
# validate_weather_schema
# ---------------------------------------------------------------------------


def test_validate_weather_schema_accepts_valid_air_temperature_data() -> None:
    df = pd.DataFrame(
        {
            "STATIONS_ID": [1766, 1766],
            "MESS_DATUM": [2025011900, 2025011901],
            "QN_9": [3, 3],
            "TT_TU": [-2.5, -2.8],
            "RF_TU": [100.0, 100.0],
        }
    )
    # Should not raise.
    weather.validate_weather_schema(df, "air_temperature")


def test_validate_weather_schema_accepts_matching_station_id() -> None:
    df = pd.DataFrame(
        {
            "STATIONS_ID": [1766],
            "MESS_DATUM": [2025011900],
            "QN_9": [3],
            "TT_TU": [-2.5],
            "RF_TU": [100.0],
        }
    )
    weather.validate_weather_schema(df, "air_temperature", expected_station_id="01766")


def test_validate_weather_schema_rejects_mismatched_station_id() -> None:
    df = pd.DataFrame(
        {
            "STATIONS_ID": [9999],
            "MESS_DATUM": [2025011900],
            "QN_9": [3],
            "TT_TU": [-2.5],
            "RF_TU": [100.0],
        }
    )
    with pytest.raises(weather.WeatherSchemaError):
        weather.validate_weather_schema(
            df, "air_temperature", expected_station_id="01766"
        )


def test_validate_weather_schema_rejects_missing_columns() -> None:
    df = pd.DataFrame(
        {
            "STATIONS_ID": [1766],
            "MESS_DATUM": [2025011900],
            # TT_TU and RF_TU are missing.
            "QN_9": [3],
        }
    )
    with pytest.raises(weather.WeatherSchemaError, match="missing expected columns"):
        weather.validate_weather_schema(df, "air_temperature")


def test_validate_weather_schema_rejects_non_numeric_value_column() -> None:
    df = pd.DataFrame(
        {
            "STATIONS_ID": [1766],
            "MESS_DATUM": [2025011900],
            "QN_9": [3],
            "TT_TU": ["not-a-number"],
            "RF_TU": [100.0],
        }
    )
    with pytest.raises(weather.WeatherSchemaError):
        weather.validate_weather_schema(df, "air_temperature")


def test_validate_weather_schema_rejects_bad_timestamp_format() -> None:
    df = pd.DataFrame(
        {
            "STATIONS_ID": [1766],
            "MESS_DATUM": ["not-a-timestamp"],
            "QN_9": [3],
            "TT_TU": [-2.5],
            "RF_TU": [100.0],
        }
    )
    with pytest.raises(weather.WeatherSchemaError):
        weather.validate_weather_schema(df, "air_temperature")


def test_validate_weather_schema_rejects_empty_dataframe() -> None:
    df = pd.DataFrame(columns=["STATIONS_ID", "MESS_DATUM", "QN_9", "TT_TU", "RF_TU"])
    with pytest.raises(weather.WeatherSchemaError):
        weather.validate_weather_schema(df, "air_temperature")


def test_unknown_parameter_raises_value_error() -> None:
    with pytest.raises(ValueError):
        weather._parameter_spec("solar_flux")


# ---------------------------------------------------------------------------
# find_missing_hours
# ---------------------------------------------------------------------------


def test_find_missing_hours_detects_a_gap() -> None:
    timestamps = pd.to_datetime(
        [
            "2025-01-19 00:00",
            "2025-01-19 01:00",
            # 02:00 and 03:00 missing
            "2025-01-19 04:00",
        ],
        utc=True,
    )
    df = pd.DataFrame({"timestamp": timestamps})

    missing = weather.find_missing_hours(df)

    assert list(missing["timestamp"]) == list(
        pd.to_datetime(["2025-01-19 02:00", "2025-01-19 03:00"], utc=True)
    )


def test_find_missing_hours_returns_empty_for_contiguous_data() -> None:
    timestamps = pd.date_range("2025-01-19", periods=5, freq="h", tz="UTC")
    df = pd.DataFrame({"timestamp": timestamps})

    missing = weather.find_missing_hours(df)

    assert missing.empty


def test_find_missing_hours_raises_on_missing_column() -> None:
    df = pd.DataFrame({"not_timestamp": [1, 2, 3]})
    with pytest.raises(ValueError):
        weather.find_missing_hours(df)


def test_find_missing_hours_raises_on_empty_dataframe() -> None:
    df = pd.DataFrame({"timestamp": pd.to_datetime([], utc=True)})
    with pytest.raises(ValueError):
        weather.find_missing_hours(df)


# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------


def test_build_recent_zip_url_is_deterministic() -> None:
    url = weather.build_recent_zip_url("air_temperature", station_id="01766")
    assert url == (
        "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
        "climate/hourly/air_temperature/recent/stundenwerte_TU_01766_akt.zip"
    )


def test_build_recent_zip_url_rejects_unknown_parameter() -> None:
    with pytest.raises(ValueError):
        weather.build_recent_zip_url("unknown_parameter")


def test_resolve_historical_zip_url_parses_directory_index() -> None:
    index_html = (
        "<a href='stundenwerte_TU_00001_19500101_20241231_hist.zip'>x</a>"
        "<a href='stundenwerte_TU_01766_19891001_20251231_hist.zip'>x</a>"
    )
    session = _FakeSession(
        {"air_temperature/historical/": _FakeResponse(text=index_html)}
    )

    url = weather.resolve_historical_zip_url(
        "air_temperature", station_id="01766", session=session
    )

    assert url.endswith("stundenwerte_TU_01766_19891001_20251231_hist.zip")


def test_resolve_historical_zip_url_raises_when_station_not_found() -> None:
    index_html = "<a href='stundenwerte_TU_00001_19500101_20241231_hist.zip'>x</a>"
    session = _FakeSession(
        {"air_temperature/historical/": _FakeResponse(text=index_html)}
    )

    with pytest.raises(weather.WeatherFetchError):
        weather.resolve_historical_zip_url(
            "air_temperature", station_id="01766", session=session
        )


def test_resolve_historical_zip_url_escapes_station_id_regex_metacharacters() -> None:
    # Without escaping, "." in station_id acts as a regex wildcard: a
    # crafted/malformed id like "01.66" would incorrectly match the real
    # "01766" file below instead of failing to find "01.66".
    index_html = "<a href='stundenwerte_TU_01766_19891001_20251231_hist.zip'>x</a>"
    session = _FakeSession(
        {"air_temperature/historical/": _FakeResponse(text=index_html)}
    )

    with pytest.raises(weather.WeatherFetchError):
        weather.resolve_historical_zip_url(
            "air_temperature", station_id="01.66", session=session
        )


# ---------------------------------------------------------------------------
# fetch_hourly_weather (fully mocked HTTP)
# ---------------------------------------------------------------------------


def test_fetch_hourly_weather_recent_end_to_end_with_mocked_http() -> None:
    product_text = _tu_product_text(
        [
            "      1766;2025011900;    3;  -2.5; 100.0;eor",
            "      1766;2025011901;    3;  -999;-999;eor",  # missing sentinel
            "      1766;2025011902;    3;  -3.2;  99.0;eor",
        ]
    )
    zip_bytes = _make_zip_bytes(
        "produkt_tu_stunde_20250119_20260722_01766.txt", product_text
    )
    session = _FakeSession(
        {"stundenwerte_TU_01766_akt.zip": _FakeResponse(content=zip_bytes)}
    )

    df = weather.fetch_hourly_weather(
        "air_temperature", period="recent", station_id="01766", session=session
    )

    assert list(df.columns) == [
        "station_id",
        "timestamp",
        "quality_level",
        "air_temperature_c",
        "relative_humidity_pct",
    ]
    assert len(df) == 3
    assert (df["station_id"] == "01766").all()
    # The -999 sentinel row should be converted to NA, not silently dropped.
    missing_row = df[df["timestamp"] == pd.Timestamp("2025-01-19 01:00", tz="UTC")]
    assert missing_row["air_temperature_c"].isna().all()
    assert df["air_temperature_c"].iloc[0] == -2.5


def test_fetch_hourly_weather_rejects_invalid_period() -> None:
    with pytest.raises(ValueError):
        weather.fetch_hourly_weather("air_temperature", period="future")


def test_fetch_hourly_weather_wraps_http_errors() -> None:
    session = _FakeSession(
        {"stundenwerte_TU_01766_akt.zip": _FakeResponse(status_code=500)}
    )
    with pytest.raises(weather.WeatherFetchError):
        weather.fetch_hourly_weather(
            "air_temperature", period="recent", station_id="01766", session=session
        )


def test_fetch_hourly_weather_rejects_non_zip_content() -> None:
    session = _FakeSession(
        {"stundenwerte_TU_01766_akt.zip": _FakeResponse(content=b"not a zip file")}
    )
    with pytest.raises(weather.WeatherFetchError):
        weather.fetch_hourly_weather(
            "air_temperature", period="recent", station_id="01766", session=session
        )


# ---------------------------------------------------------------------------
# Download / zip-bomb size guards
# ---------------------------------------------------------------------------


def test_download_zip_bytes_rejects_declared_oversized_content_length() -> None:
    response = _FakeResponse(content=b"small")
    response.headers = {"Content-Length": str(weather.MAX_DOWNLOAD_BYTES + 1)}
    session = _FakeSession({"foo.zip": response})

    with pytest.raises(weather.WeatherFetchError, match="exceeds"):
        weather._download_zip_bytes("https://example.invalid/foo.zip", session=session)


def test_download_zip_bytes_rejects_oversized_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(weather, "MAX_DOWNLOAD_BYTES", 3)
    session = _FakeSession({"foo.zip": _FakeResponse(content=b"too long")})

    with pytest.raises(weather.WeatherFetchError, match="exceeds"):
        weather._download_zip_bytes("https://example.invalid/foo.zip", session=session)


def test_extract_product_text_rejects_oversized_zip_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(weather, "MAX_ZIP_MEMBER_BYTES", 10)
    zip_bytes = _make_zip_bytes("produkt_test.txt", "x" * 100)

    with pytest.raises(weather.WeatherFetchError, match="exceeds"):
        weather._extract_product_text(zip_bytes, "https://example.invalid/foo.zip")


# ---------------------------------------------------------------------------
# save_raw_weather
# ---------------------------------------------------------------------------


def test_save_raw_weather_is_idempotent(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "station_id": ["01766"],
            "timestamp": pd.to_datetime(["2025-01-19 00:00"], utc=True),
            "air_temperature_c": [-2.5],
        }
    )

    path_first = weather.save_raw_weather(df, "air_temperature", "01766", tmp_path)
    path_second = weather.save_raw_weather(df, "air_temperature", "01766", tmp_path)

    assert path_first == path_second
    assert len(list(tmp_path.glob("*.csv"))) == 1
    saved = pd.read_csv(path_first)
    assert len(saved) == 1


def test_save_raw_weather_rejects_unknown_parameter(tmp_path: Path) -> None:
    df = pd.DataFrame({"station_id": ["01766"]})
    with pytest.raises(ValueError):
        weather.save_raw_weather(df, "unknown_parameter", "01766", tmp_path)

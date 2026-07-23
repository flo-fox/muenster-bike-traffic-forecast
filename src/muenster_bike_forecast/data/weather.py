"""Fetch and validate DWD hourly weather observations for Münster.

Data source: the DWD (Deutscher Wetterdienst) Open Data portal at
https://opendata.dwd.de/ — specifically the hourly climate observations
under ``climate_environment/CDC/observations_germany/climate/hourly/``.
No credentials are required.

The nearest DWD hourly weather station to Münster with a long, continuous
observation record is **Münster/Osnabrück (airport)**, DWD station id
``01766`` (WMO/ICAO station at Münster Osnabrück International Airport,
~52.13N 7.70E). It is used as the default station throughout this module.

DWD publishes each parameter (temperature, precipitation, wind, ...) as two
separate download products per station:

- ``recent``: a rolling window of roughly the last 500 days, updated
  continuously. Filename does not encode a date range, e.g.
  ``stundenwerte_TU_01766_akt.zip``.
- ``historical``: the complete record up to the end of the previous
  calendar year, frozen and only extended once a year. The filename embeds
  the covered date range, e.g.
  ``stundenwerte_TU_01766_19891001_20251231_hist.zip``, so it cannot be
  hardcoded and is resolved from the live directory listing instead.

Both products are zip archives containing metadata files plus one
semicolon-separated ``produkt_*.txt`` file with the actual observations.
Missing values are encoded with the sentinel ``-999`` and timestamps
(``MESS_DATUM``) are hourly, in the format ``YYYYMMDDHH``, UTC (per DWD's
own parameter documentation for post-2001 SYNOP-derived data).

Open question / known limitation: DWD hourly data is genuinely hourly,
while bike counts are 15-minute. This module intentionally does **not**
resample or align the two; that is left to a later feature-engineering
step.
"""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd
import requests

DWD_BASE_URL: Final[str] = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/hourly"
)
DEFAULT_STATION_ID: Final[str] = "01766"
DEFAULT_STATION_NAME: Final[str] = "Münster/Osnabrück"
MISSING_VALUE_SENTINEL: Final[float] = -999.0
VALID_PERIODS: Final[tuple[str, ...]] = ("recent", "historical")
# Generous ceiling for a single DWD download (decades of hourly data for one
# station/parameter is a few MB). Guards against an oversized/runaway
# response, e.g. from a compromised or misbehaving host.
MAX_DOWNLOAD_BYTES: Final[int] = 200 * 1024 * 1024
# Ceiling on a zip member's declared *uncompressed* size, checked before
# reading it into memory - guards against zip-bomb-style archives where a
# small download decompresses into something huge.
MAX_ZIP_MEMBER_BYTES: Final[int] = 500 * 1024 * 1024


class WeatherFetchError(Exception):
    """Raised when weather data cannot be retrieved from DWD Open Data."""


class WeatherSchemaError(Exception):
    """Raised when fetched weather data does not match the expected schema."""


@dataclass(frozen=True)
class ParameterSpec:
    """Describes one DWD hourly weather parameter/product.

    Attributes:
        subdir: Sub-directory name under ``.../climate/hourly/``, e.g.
            ``"wind"``.
        file_code: Two-letter code DWD uses in filenames, e.g. ``"FF"``.
        value_columns: Mapping from raw DWD column name to a descriptive
            snake_case column name used in the returned DataFrame.
    """

    subdir: str
    file_code: str
    value_columns: dict[str, str]


PARAMETER_SPECS: Final[dict[str, ParameterSpec]] = {
    "air_temperature": ParameterSpec(
        subdir="air_temperature",
        file_code="TU",
        value_columns={
            "QN_9": "quality_level",
            "TT_TU": "air_temperature_c",
            "RF_TU": "relative_humidity_pct",
        },
    ),
    "precipitation": ParameterSpec(
        subdir="precipitation",
        file_code="RR",
        value_columns={
            "QN_8": "quality_level",
            "R1": "precipitation_mm",
            "RS_IND": "precipitation_indicator",
            "WRTR": "precipitation_form",
        },
    ),
    "wind": ParameterSpec(
        subdir="wind",
        file_code="FF",
        value_columns={
            "QN_3": "quality_level",
            "F": "wind_speed_ms",
            "D": "wind_direction_deg",
        },
    ),
}


def _parameter_spec(parameter: str) -> ParameterSpec:
    """Looks up the ``ParameterSpec`` for a parameter key.

    Args:
        parameter: Weather parameter key, e.g. ``"air_temperature"``.

    Returns:
        The matching ``ParameterSpec``.

    Raises:
        ValueError: if `parameter` is not a known key of ``PARAMETER_SPECS``.
    """
    try:
        return PARAMETER_SPECS[parameter]
    except KeyError as exc:
        raise ValueError(
            f"Unknown weather parameter {parameter!r}. "
            f"Expected one of {sorted(PARAMETER_SPECS)}."
        ) from exc


def build_recent_zip_url(parameter: str, station_id: str = DEFAULT_STATION_ID) -> str:
    """Builds the download URL for the rolling "recent" observation file.

    Args:
        parameter: Weather parameter key, e.g. ``"air_temperature"``.
        station_id: Five-digit zero-padded DWD station id.

    Returns:
        Full HTTPS URL to the ``_akt.zip`` file. This URL is stable/
        deterministic (no date range embedded), so building it never
        requires a network call.

    Raises:
        ValueError: if `parameter` is not a known parameter key.
    """
    spec = _parameter_spec(parameter)
    return (
        f"{DWD_BASE_URL}/{spec.subdir}/recent/"
        f"stundenwerte_{spec.file_code}_{station_id}_akt.zip"
    )


def resolve_historical_zip_url(
    parameter: str,
    station_id: str = DEFAULT_STATION_ID,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> str:
    """Resolves the exact "historical" zip URL from the live DWD directory.

    The historical filename embeds a start/end date
    (``..._<start>_<end>_hist.zip``) that DWD updates roughly once a year,
    so it cannot be hardcoded. This fetches the directory index page and
    extracts the filename matching the given station.

    Args:
        parameter: Weather parameter key, e.g. ``"air_temperature"``.
        station_id: Five-digit zero-padded DWD station id.
        session: Optional ``requests.Session`` to reuse connections / allow
            mocking in tests.
        timeout: Per-request timeout in seconds.

    Returns:
        Full HTTPS URL to the resolved ``_hist.zip`` file.

    Raises:
        ValueError: if `parameter` is not a known parameter key.
        WeatherFetchError: if the directory listing cannot be fetched, or
            no matching file is found for the station.
    """
    spec = _parameter_spec(parameter)
    index_url = f"{DWD_BASE_URL}/{spec.subdir}/historical/"
    http = session if session is not None else requests

    try:
        response = http.get(index_url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherFetchError(
            f"Could not list DWD historical directory at {index_url!r}: {exc}"
        ) from exc

    # spec.file_code is a hardcoded constant, but station_id can be caller-
    # supplied - escape both so neither can inject regex metacharacters
    # (e.g. cause catastrophic backtracking) into the compiled pattern.
    pattern = re.compile(
        rf"stundenwerte_{re.escape(spec.file_code)}_{re.escape(station_id)}"
        rf"_\d{{8}}_\d{{8}}_hist\.zip"
    )
    matches = sorted(set(pattern.findall(response.text)))
    if not matches:
        raise WeatherFetchError(
            f"No historical file found for parameter {parameter!r} and "
            f"station {station_id!r} at {index_url!r}."
        )
    # Filenames sort lexicographically the same as by end date (YYYYMMDD),
    # so the last match is the most complete/current one.
    return f"{DWD_BASE_URL}/{spec.subdir}/historical/{matches[-1]}"


def _download_zip_bytes(
    url: str, session: requests.Session | None = None, timeout: float = 60.0
) -> bytes:
    """Downloads a URL and returns the raw response body.

    Args:
        url: URL to download.
        session: Optional ``requests.Session``.
        timeout: Request timeout in seconds.

    Returns:
        Raw response body bytes.

    Raises:
        WeatherFetchError: on any network/HTTP error, or if the response
            declares (via ``Content-Length``) or delivers a body larger than
            `MAX_DOWNLOAD_BYTES`.
    """
    http = session if session is not None else requests
    try:
        response = http.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise WeatherFetchError(f"Could not download {url!r}: {exc}") from exc

    content_length = (
        response.headers.get("Content-Length") if hasattr(response, "headers") else None
    )
    if content_length is not None and int(content_length) > MAX_DOWNLOAD_BYTES:
        raise WeatherFetchError(
            f"Refusing to download {url!r}: declared size {content_length} "
            f"bytes exceeds the {MAX_DOWNLOAD_BYTES}-byte limit."
        )
    if len(response.content) > MAX_DOWNLOAD_BYTES:
        raise WeatherFetchError(
            f"Refusing to use response from {url!r}: body of "
            f"{len(response.content)} bytes exceeds the "
            f"{MAX_DOWNLOAD_BYTES}-byte limit."
        )
    return response.content


def _extract_product_text(zip_bytes: bytes, source_url: str) -> str:
    """Extracts the ``produkt_*.txt`` payload text from a DWD zip archive.

    Args:
        zip_bytes: Raw bytes of a DWD observation zip archive.
        source_url: Original URL, used only for error messages.

    Returns:
        Decoded text content of the ``produkt_*.txt`` member.

    Raises:
        WeatherFetchError: if the bytes are not a valid zip archive, the
            archive does not contain a ``produkt_*.txt`` member, or that
            member's declared uncompressed size exceeds
            `MAX_ZIP_MEMBER_BYTES` (zip-bomb guard).
    """
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            product_names = [
                name for name in archive.namelist() if name.startswith("produkt_")
            ]
            if not product_names:
                raise WeatherFetchError(
                    f"Zip archive from {source_url!r} does not contain a "
                    "produkt_*.txt data file."
                )
            info = archive.getinfo(product_names[0])
            if info.file_size > MAX_ZIP_MEMBER_BYTES:
                raise WeatherFetchError(
                    f"Refusing to extract {product_names[0]!r} from "
                    f"{source_url!r}: declared uncompressed size "
                    f"{info.file_size} bytes exceeds the "
                    f"{MAX_ZIP_MEMBER_BYTES}-byte limit."
                )
            with archive.open(product_names[0]) as fh:
                # DWD text files are plain ASCII/Latin-1; decoding as
                # latin-1 never raises, unlike utf-8, on any byte sequence.
                return fh.read().decode("latin-1")
    except zipfile.BadZipFile as exc:
        raise WeatherFetchError(
            f"Content downloaded from {source_url!r} is not a valid zip archive: {exc}"
        ) from exc


def _parse_product_text(text: str, source_url: str) -> pd.DataFrame:
    """Parses DWD's semicolon-separated ``produkt_*.txt`` content into a DataFrame.

    Args:
        text: Raw file content (untrusted external input).
        source_url: Original URL, used only for error messages.

    Returns:
        DataFrame with raw DWD column names (whitespace-stripped), and the
        trailing ``eor`` end-of-record marker column dropped if present.

    Raises:
        WeatherSchemaError: if the content cannot be parsed as the expected
            semicolon-separated table.
    """
    try:
        df = pd.read_csv(io.StringIO(text), sep=";", skipinitialspace=True)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise WeatherSchemaError(
            f"Could not parse DWD product file from {source_url!r} as CSV: {exc}"
        ) from exc

    df.columns = [str(c).strip() for c in df.columns]
    if "eor" in df.columns:
        df = df.drop(columns=["eor"])
    return df


def validate_weather_schema(
    df: pd.DataFrame, parameter: str, expected_station_id: str | None = None
) -> None:
    """Validates that a raw DWD DataFrame has the expected columns and dtypes.

    Treats the input as untrusted (parsed from an external HTTP download):
    checks structure and types explicitly rather than assuming them.

    Args:
        df: DataFrame as parsed from a DWD ``produkt_*.txt`` file, with raw
            DWD column names (before renaming to descriptive names).
        parameter: Weather parameter key, e.g. ``"air_temperature"``.
        expected_station_id: If given, verifies every row's ``STATIONS_ID``
            matches this station id (guards against silently mixing up
            data from a different station).

    Raises:
        ValueError: if `parameter` is not a known parameter key.
        WeatherSchemaError: if required columns are missing, contain
            values that don't match the expected type/format, or (when
            `expected_station_id` is given) contain an unexpected station.
    """
    spec = _parameter_spec(parameter)
    expected_columns = {"STATIONS_ID", "MESS_DATUM", *spec.value_columns}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise WeatherSchemaError(
            f"Weather data for {parameter!r} is missing expected columns "
            f"{sorted(missing_columns)}. Found columns: {list(df.columns)}."
        )

    if df.empty:
        raise WeatherSchemaError(f"Weather data for {parameter!r} has no data rows.")

    if not pd.api.types.is_numeric_dtype(df["STATIONS_ID"]):
        raise WeatherSchemaError(
            "Column STATIONS_ID is not numeric, cannot identify station."
        )

    try:
        pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", errors="raise")
    except (ValueError, TypeError) as exc:
        raise WeatherSchemaError(
            "Column MESS_DATUM does not match the expected YYYYMMDDHH "
            f"timestamp format: {exc}"
        ) from exc

    for column in spec.value_columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise WeatherSchemaError(f"Column {column!r} is not numeric as expected.")

    if expected_station_id is not None:
        found_ids = set(df["STATIONS_ID"].astype(int).astype(str).str.zfill(5))
        expected = str(expected_station_id).zfill(5)
        if found_ids != {expected}:
            raise WeatherSchemaError(
                f"Expected only station {expected!r} in data for "
                f"{parameter!r}, found station id(s) {sorted(found_ids)}."
            )


def _standardize(
    df: pd.DataFrame, spec: ParameterSpec, station_id: str
) -> pd.DataFrame:
    """Converts a validated raw DWD DataFrame into a standardized shape.

    Args:
        df: Raw, already schema-validated DataFrame (DWD column names).
        spec: ``ParameterSpec`` describing the value columns to keep/rename.
        station_id: Station id to stamp onto every row (zero-padded).

    Returns:
        DataFrame with columns ``station_id``, ``timestamp`` (UTC,
        hourly), and the renamed value columns from `spec`. DWD's ``-999``
        missing-value sentinel is replaced with ``pandas.NA``. Rows are
        sorted by timestamp and de-duplicated on
        (``station_id``, ``timestamp``), keeping the last occurrence.
    """
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["MESS_DATUM"], format="%Y%m%d%H", utc=True)
    # Assigning the station_id scalar only broadcasts correctly once `out`
    # already has an index (i.e. after the first real column is set).
    out["station_id"] = station_id
    for raw_col, new_col in spec.value_columns.items():
        values = pd.to_numeric(df[raw_col], errors="coerce")
        out[new_col] = values.mask(values <= MISSING_VALUE_SENTINEL)

    out = out[["station_id", "timestamp", *spec.value_columns.values()]]
    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["station_id", "timestamp"], keep="last")
    return out.reset_index(drop=True)


def fetch_hourly_weather(
    parameter: str,
    period: str = "recent",
    station_id: str = DEFAULT_STATION_ID,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetches hourly DWD weather observations for one parameter/period.

    Idempotent: repeated calls download the same DWD source file(s) and
    return an equivalent DataFrame; nothing is appended to or mutated on
    disk by this function (persistence is a separate, explicit step, see
    ``save_raw_weather``).

    Args:
        parameter: One of ``PARAMETER_SPECS`` keys — ``"air_temperature"``,
            ``"precipitation"``, or ``"wind"``.
        period: ``"recent"`` (rolling ~500-day window, updated
            continuously) or ``"historical"`` (complete frozen record up to
            the end of the previous calendar year).
        station_id: Five-digit zero-padded DWD station id. Defaults to
            ``01766`` (Münster/Osnabrück airport), the nearest DWD hourly
            station to Münster.
        session: Optional ``requests.Session`` for connection reuse or
            mocking in tests.
        timeout: Per-request timeout in seconds.

    Returns:
        Standardized DataFrame — see ``_standardize`` — with hourly
        resolution and UTC timestamps.

    Raises:
        ValueError: if `parameter` or `period` is not recognized.
        WeatherFetchError: if the DWD file cannot be located, downloaded,
            or extracted.
        WeatherSchemaError: if the downloaded data does not match DWD's
            documented hourly-observation schema.
    """
    if period not in VALID_PERIODS:
        raise ValueError(f"period must be one of {VALID_PERIODS}, got {period!r}.")
    spec = _parameter_spec(parameter)

    if period == "recent":
        url = build_recent_zip_url(parameter, station_id)
    else:
        url = resolve_historical_zip_url(
            parameter, station_id, session=session, timeout=timeout
        )

    zip_bytes = _download_zip_bytes(url, session=session, timeout=timeout)
    text = _extract_product_text(zip_bytes, url)
    raw_df = _parse_product_text(text, url)
    validate_weather_schema(raw_df, parameter, expected_station_id=station_id)
    return _standardize(raw_df, spec, station_id)


def fetch_weather_history(
    parameter: str,
    station_id: str = DEFAULT_STATION_ID,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> pd.DataFrame:
    """Fetches and combines the "historical" and "recent" DWD periods.

    DWD splits each station's record into a frozen ``historical`` file
    (up to the end of the previous calendar year) and a rolling ``recent``
    file (last ~500 days), which overlap near the boundary. This combines
    both into one continuous series, preferring ``recent`` values on any
    overlapping timestamp since it is the more current source.

    Args:
        parameter: One of ``PARAMETER_SPECS`` keys.
        station_id: Five-digit zero-padded DWD station id.
        session: Optional ``requests.Session``.
        timeout: Per-request timeout in seconds.

    Returns:
        Combined, de-duplicated, timestamp-sorted DataFrame spanning the
        full available DWD record for the station.

    Raises:
        ValueError: if `parameter` is not a known parameter key.
        WeatherFetchError: if either period cannot be downloaded.
        WeatherSchemaError: if either period's data fails schema validation.
    """
    historical = fetch_hourly_weather(
        parameter,
        period="historical",
        station_id=station_id,
        session=session,
        timeout=timeout,
    )
    recent = fetch_hourly_weather(
        parameter,
        period="recent",
        station_id=station_id,
        session=session,
        timeout=timeout,
    )
    combined = pd.concat([historical, recent], ignore_index=True)
    combined = combined.sort_values("timestamp")
    combined = combined.drop_duplicates(subset=["station_id", "timestamp"], keep="last")
    return combined.reset_index(drop=True)


def find_missing_hours(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """Reports missing hourly timestamps within an otherwise-hourly series.

    Gaps are computed against a complete hourly range spanning the data's
    own min/max timestamp — missing hours are never silently dropped, they
    are returned explicitly so callers can report/handle them.

    Args:
        df: DataFrame containing a timestamp column at (nominally) hourly
            resolution.
        timestamp_col: Name of the timestamp column.

    Returns:
        DataFrame with a single column (`timestamp_col`) listing every
        missing hourly timestamp between the data's min and max timestamp.
        Empty (zero rows, same dtype) if no gaps are found.

    Raises:
        ValueError: if `timestamp_col` is not a column of `df`, or `df` is
            empty.
    """
    if timestamp_col not in df.columns:
        raise ValueError(f"Column {timestamp_col!r} not found in DataFrame.")
    if df.empty:
        raise ValueError("Cannot detect gaps in an empty DataFrame.")

    observed = pd.DatetimeIndex(df[timestamp_col]).drop_duplicates().sort_values()
    full_range = pd.date_range(start=observed.min(), end=observed.max(), freq="h")
    missing = full_range.difference(observed)
    return pd.DataFrame({timestamp_col: missing})


def load_weather_data(path: Path) -> pd.DataFrame:
    """Load a weather CSV previously written by `save_raw_weather`.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with ``station_id`` read back as ``str`` (so a leading
        zero, e.g. ``"01766"``, is preserved instead of being silently
        dropped by integer reinterpretation) and ``timestamp`` parsed back
        to a UTC-aware ``datetime64[ns, UTC]`` column.

    Raises:
        WeatherSchemaError: if the file is missing the ``station_id`` or
            ``timestamp`` column.
    """
    df = pd.read_csv(path, dtype={"station_id": str})
    missing_columns = {"station_id", "timestamp"} - set(df.columns)
    if missing_columns:
        raise WeatherSchemaError(
            f"{path} is missing expected column(s): {sorted(missing_columns)}."
        )
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def save_raw_weather(
    df: pd.DataFrame,
    parameter: str,
    station_id: str,
    raw_dir: Path,
) -> Path:
    """Saves fetched weather data to `raw_dir` as CSV, idempotently.

    The output filename encodes only the parameter and station id (never a
    fetch timestamp), so re-running the fetch overwrites the same file
    with the latest data instead of accumulating duplicate/stale files.

    Args:
        df: Standardized weather DataFrame, as returned by
            ``fetch_hourly_weather`` / ``fetch_weather_history``.
        parameter: Weather parameter key used to name the output file.
        station_id: DWD station id used to name the output file.
        raw_dir: Directory to write into (created if it doesn't exist).

    Returns:
        Path to the written CSV file.

    Raises:
        ValueError: if `parameter` is not a known parameter key.
    """
    _parameter_spec(parameter)  # validates parameter name, raises if unknown
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"dwd_{parameter}_{station_id}.csv"
    df.to_csv(out_path, index=False)
    return out_path

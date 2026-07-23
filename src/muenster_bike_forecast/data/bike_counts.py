"""Fetch and validate bike-traffic count data.

Source: the `od-ms/radverkehr-zaehlstellen` GitHub repository, which
publishes daily-updated 15-minute bike (and, at some stations, car) counts
for counting stations across Münster. Layout, discovered by inspecting the
live repo:

- ``site_min.json`` at the repo root lists every station: its directory
  name (``station_id``), display name, first year of data (``start_year``),
  and measurement channels.
- Each station has its own top-level directory (e.g. ``300038855/``)
  containing one CSV file per calendar month, named ``YYYY-MM.csv``.
- Each CSV has a ``Datetime`` column (format ``YYYY-MM-DD HH:MM``, 15-minute
  steps) plus, per channel, a count column named
  ``"<channel_id> (<description>)"`` and a matching status column named
  ``"<channel_id>-status"``.
- The source data is explicitly documented as raw/uncleaned and may contain
  extended gaps (sensor outages, construction). Missing months simply have
  no CSV file (HTTP 404); this is a legitimate absence, not an error.

All HTTP/CSV content from the source repo is treated as untrusted input and
is schema-validated before use.

Licensed under Datenlizenz Deutschland – Namensnennung 2.0 (dl-de/by-2-0)
via the City of Münster's open-data portal; redistribution requires
attribution. See the README's "Data sources & attribution" section for the
full citation.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd
import requests

GITHUB_RAW_BASE: Final[str] = (
    "https://raw.githubusercontent.com/od-ms/radverkehr-zaehlstellen/master"
)
SITE_INDEX_URL: Final[str] = f"{GITHUB_RAW_BASE}/site_min.json"
EXPECTED_INTERVAL_MINUTES: Final[int] = 15
DATETIME_COLUMN: Final[str] = "Datetime"
# Generous ceiling for a single station-month CSV (a full month of 15-minute
# data for one station is a few hundred KB at most). Guards against an
# oversized/runaway response, e.g. from a compromised or misbehaving host.
MAX_DOWNLOAD_BYTES: Final[int] = 50 * 1024 * 1024
EARLIEST_PLAUSIBLE_START_YEAR: Final[int] = 2000

_COUNT_COLUMN_RE: Final[re.Pattern[str]] = re.compile(r"^(\d+)\s*\(.*\)$")
_STATUS_COLUMN_RE: Final[re.Pattern[str]] = re.compile(r"^(\d+)-status$")
_SOURCE_DATETIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M"

_STATION_ID_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9_-]+$")
_FORMULA_TRIGGER_PREFIXES: Final[tuple[str, ...]] = ("=", "+", "-", "@", "\t", "\r")


class BikeCountDataError(Exception):
    """Raised when bike-count data from the source repo fails validation.

    Covers HTTP/CSV content that does not match the expected schema:
    missing columns, mismatched count/status channel pairs, unparsable
    timestamps, duplicate timestamps, or non-numeric count/status values.
    """


def _validate_station_id(station_id: str) -> None:
    """Reject station ids that are unsafe to use as a filename component.

    `station_id` originates from the source repo's ``site_min.json``
    (untrusted input) and is later joined onto a local output directory to
    build a file path. Without this check, a crafted id such as
    ``"../../etc/evil"`` could escape the intended output directory.

    Args:
        station_id: Station identifier to validate.

    Raises:
        BikeCountDataError: if `station_id` is empty or contains any
            character other than an ASCII letter, digit, ``_``, or ``-``.
    """
    if not _STATION_ID_RE.match(station_id):
        raise BikeCountDataError(
            f"station_id {station_id!r} is not a safe identifier; expected "
            "only letters, digits, '_', and '-'."
        )


def _sanitize_csv_header(name: str) -> str:
    """Neutralize a column name that could be read as a spreadsheet formula.

    Count-column names embed a free-text ``description`` from the source
    repo (untrusted input). Spreadsheet applications (Excel, Google Sheets)
    treat a cell as a formula if it starts with ``=``, ``+``, ``-``, ``@``,
    tab, or carriage return, so this defuses that regardless of where the
    triggering character came from.

    Args:
        name: Column name as it will be written to a CSV header.

    Returns:
        `name` unchanged, or prefixed with a single quote if it starts with
        a formula-triggering character.
    """
    if name and name[0] in _FORMULA_TRIGGER_PREFIXES:
        return f"'{name}"
    return name


@dataclass(frozen=True)
class Channel:
    """A single measurement channel at a counting station.

    Attributes:
        channel_id: Numeric channel identifier used in CSV column headers.
        description: Human-readable description (often includes direction,
            e.g. "Stadteinwärts").
    """

    channel_id: int
    description: str


@dataclass(frozen=True)
class Station:
    """Metadata for one bike-counting station in the source repo.

    Attributes:
        station_id: Directory name / station identifier as used in the
            source repo (e.g. ``"300038855"``).
        name: Human-readable station name (e.g. ``"Bismarckallee"``).
        start_year: First year for which the source repo has data.
        channels: Measurement channels available at this station.
    """

    station_id: str
    name: str
    start_year: int
    channels: tuple[Channel, ...]


def list_stations(timeout: float = 30.0) -> list[Station]:
    """Fetch the list of bike-counting stations from the source repo's index.

    Args:
        timeout: HTTP request timeout in seconds.

    Returns:
        One `Station` per counting location, in source order.

    Raises:
        BikeCountDataError: if the index cannot be fetched, is not valid
            JSON, or its shape does not match the expected station schema.
    """
    try:
        response = requests.get(SITE_INDEX_URL, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BikeCountDataError(
            f"Failed to fetch station index from {SITE_INDEX_URL}: {exc}"
        ) from exc
    try:
        raw = response.json()
    except ValueError as exc:
        raise BikeCountDataError(
            f"Station index at {SITE_INDEX_URL} is not valid JSON."
        ) from exc
    return parse_station_index(raw)


def parse_station_index(raw: object) -> list[Station]:
    """Validate and convert raw station-index JSON into `Station` objects.

    Args:
        raw: Parsed JSON content of ``site_min.json`` (expected: a list of
            dicts with keys ``name``, ``directory``, ``start``, ``channels``,
            where ``channels`` is a list of ``[channel_id, description]``
            pairs).

    Returns:
        One `Station` per entry in `raw`, in the same order.

    Raises:
        BikeCountDataError: if `raw` is not a non-empty list of station
            dicts with the expected keys, types, and at least one channel
            each, or if a ``directory`` value is not a safe identifier.
    """
    if not isinstance(raw, list) or not raw:
        raise BikeCountDataError(
            "Expected station index to be a non-empty JSON list, got "
            f"{type(raw).__name__}."
        )
    stations: list[Station] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise BikeCountDataError(
                f"Expected station entry to be a JSON object, got {entry!r}."
            )
        missing_keys = {"name", "directory", "start", "channels"} - entry.keys()
        if missing_keys:
            raise BikeCountDataError(
                f"Station entry {entry!r} is missing required keys: "
                f"{sorted(missing_keys)}."
            )
        try:
            channels = tuple(
                Channel(channel_id=int(channel[0]), description=str(channel[1]))
                for channel in entry["channels"]
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise BikeCountDataError(
                f"Station entry {entry!r} has malformed 'channels'."
            ) from exc
        if not channels:
            raise BikeCountDataError(f"Station entry {entry!r} has no channels.")
        station_id = str(entry["directory"])
        _validate_station_id(station_id)
        try:
            start_year = int(entry["start"])
        except (TypeError, ValueError) as exc:
            raise BikeCountDataError(f"Station entry {entry!r} is malformed.") from exc
        current_year = date.today().year
        if not EARLIEST_PLAUSIBLE_START_YEAR <= start_year <= current_year:
            raise BikeCountDataError(
                f"Station entry {entry!r} has an implausible 'start' year "
                f"{start_year}; expected between {EARLIEST_PLAUSIBLE_START_YEAR} "
                f"and {current_year}. A bogus year would make "
                "`fetch_station_data` iterate an unbounded number of months."
            )
        try:
            station = Station(
                station_id=station_id,
                name=str(entry["name"]),
                start_year=start_year,
                channels=channels,
            )
        except (TypeError, ValueError) as exc:
            raise BikeCountDataError(f"Station entry {entry!r} is malformed.") from exc
        stations.append(station)
    return stations


def station_csv_url(station_id: str, year: int, month: int) -> str:
    """Build the raw-content URL for one station's monthly CSV file.

    Args:
        station_id: Station directory id, e.g. ``"300038855"``.
        year: Calendar year, e.g. 2025.
        month: Calendar month (1-12).

    Returns:
        The full ``raw.githubusercontent.com`` URL for that station/month.

    Raises:
        ValueError: if `month` is not in 1..12.
        BikeCountDataError: if `station_id` is not a safe identifier (see
            `_validate_station_id`) - this function builds a URL directly
            from `station_id`, so it is validated here too rather than
            relying on callers to have done so.
    """
    if not 1 <= month <= 12:
        raise ValueError(f"month must be in 1..12, got {month}.")
    _validate_station_id(station_id)
    return f"{GITHUB_RAW_BASE}/{station_id}/{year:04d}-{month:02d}.csv"


def parse_station_csv(csv_text: str, station_id: str) -> pd.DataFrame:
    """Parse and schema-validate one station-month CSV payload.

    Expects the source format: a ``Datetime`` column plus, per measurement
    channel, a count column named ``"<channel_id> (<description>)"`` and a
    matching status column named ``"<channel_id>-status"``.

    Args:
        csv_text: Raw CSV text as returned by the source repo.
        station_id: Station id this payload belongs to (used for error
            messages and to tag the returned rows).

    Returns:
        DataFrame with columns ``station_id``, ``datetime`` (datetime64[ns])
        and one nullable-integer (``Int64``) column per source count/status
        column, sorted by ``datetime`` with no duplicate timestamps.

    Raises:
        BikeCountDataError: if the CSV cannot be parsed, is missing the
            ``Datetime`` column, has an unrecognized column, has count
            columns without a matching status column (or vice versa),
            contains unparsable or duplicate timestamps, or contains
            non-numeric count/status values.
    """
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise BikeCountDataError(
            f"Could not parse CSV for station {station_id!r}: {exc}"
        ) from exc

    if DATETIME_COLUMN not in df.columns:
        raise BikeCountDataError(
            f"Station {station_id!r} CSV is missing the '{DATETIME_COLUMN}' "
            f"column; found columns: {list(df.columns)}."
        )

    count_columns: dict[str, str] = {}
    status_columns: dict[str, str] = {}
    for column in df.columns:
        if column == DATETIME_COLUMN:
            continue
        count_match = _COUNT_COLUMN_RE.match(column)
        status_match = _STATUS_COLUMN_RE.match(column)
        if count_match:
            count_columns[count_match.group(1)] = column
        elif status_match:
            status_columns[status_match.group(1)] = column
        else:
            raise BikeCountDataError(
                f"Station {station_id!r} CSV has an unrecognized column "
                f"'{column}'; expected '<id> (<description>)' or "
                f"'<id>-status'."
            )

    if not count_columns:
        raise BikeCountDataError(f"Station {station_id!r} CSV has no count columns.")

    missing_status = count_columns.keys() - status_columns.keys()
    missing_count = status_columns.keys() - count_columns.keys()
    if missing_status or missing_count:
        raise BikeCountDataError(
            f"Station {station_id!r} CSV has mismatched count/status channel "
            f"pairs. Count channels without status: {sorted(missing_status)}. "
            f"Status channels without count: {sorted(missing_count)}."
        )

    try:
        datetimes = pd.to_datetime(
            df[DATETIME_COLUMN], format=_SOURCE_DATETIME_FORMAT, errors="raise"
        )
    except (ValueError, TypeError) as exc:
        raise BikeCountDataError(
            f"Station {station_id!r} CSV has unparsable '{DATETIME_COLUMN}' "
            f"values: {exc}"
        ) from exc
    if datetimes.isna().any():
        raise BikeCountDataError(
            f"Station {station_id!r} CSV has null timestamps after parsing."
        )

    value_columns = list(count_columns.values()) + list(status_columns.values())
    numeric = df[value_columns].apply(pd.to_numeric, errors="coerce")
    invalid_mask = df[value_columns].notna() & numeric.isna()
    if invalid_mask.to_numpy().any():
        bad_columns = invalid_mask.columns[invalid_mask.any()].tolist()
        raise BikeCountDataError(
            f"Station {station_id!r} CSV has non-numeric values in columns: "
            f"{bad_columns}."
        )

    result = numeric.astype("Int64")
    result.insert(0, "datetime", datetimes)
    result.insert(0, "station_id", station_id)

    if result["datetime"].duplicated().any():
        n_dupes = int(result["datetime"].duplicated().sum())
        raise BikeCountDataError(
            f"Station {station_id!r} CSV has {n_dupes} duplicate timestamp(s)."
        )

    return result.sort_values("datetime").reset_index(drop=True)


def fetch_station_month(
    station_id: str, year: int, month: int, timeout: float = 30.0
) -> pd.DataFrame | None:
    """Fetch and validate one month of raw count data for one station.

    Args:
        station_id: Station directory id, e.g. ``"300038855"``.
        year: Calendar year to fetch.
        month: Calendar month to fetch (1-12).
        timeout: HTTP request timeout in seconds.

    Returns:
        Validated DataFrame for that month (see `parse_station_csv`), or
        `None` if the source repo has no file for that station/month
        (HTTP 404) — a legitimate absence, e.g. before the station existed
        or during a known sensor outage.

    Raises:
        BikeCountDataError: on any HTTP error other than 404, if the response
            body exceeds `MAX_DOWNLOAD_BYTES`, or if the fetched CSV fails
            schema validation.
    """
    url = station_csv_url(station_id, year, month)
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        raise BikeCountDataError(f"Failed to fetch {url}: {exc}") from exc
    if response.status_code == 404:
        return None
    try:
        response.raise_for_status()
    except requests.RequestException as exc:
        raise BikeCountDataError(f"Failed to fetch {url}: {exc}") from exc
    content_length = (
        response.headers.get("Content-Length") if hasattr(response, "headers") else None
    )
    if content_length is not None and int(content_length) > MAX_DOWNLOAD_BYTES:
        raise BikeCountDataError(
            f"Refusing to download {url}: declared size {content_length} "
            f"bytes exceeds the {MAX_DOWNLOAD_BYTES}-byte limit."
        )
    body_size = len(response.text.encode("utf-8"))
    if body_size > MAX_DOWNLOAD_BYTES:
        raise BikeCountDataError(
            f"Refusing to use response from {url}: body of {body_size} "
            f"bytes exceeds the {MAX_DOWNLOAD_BYTES}-byte limit."
        )
    return parse_station_csv(response.text, station_id=station_id)


def fetch_station_data(
    station: Station, as_of: date | None = None, timeout: float = 30.0
) -> pd.DataFrame:
    """Fetch all available months of raw count data for one station.

    Iterates every calendar month from the station's `start_year` (January)
    through `as_of` (default: today), skipping months the source repo has
    no file for. The result is deduplicated by timestamp and sorted, so
    fetching the same station repeatedly (or with months arriving out of
    order) always yields the same DataFrame — safe to re-run.

    Args:
        station: Station to fetch, as returned by `list_stations`.
        as_of: Last month (inclusive) to fetch; defaults to today.
        timeout: HTTP request timeout in seconds, applied per request.

    Returns:
        Concatenated DataFrame across all available months, sorted by
        `datetime`, with duplicate timestamps removed. Has just the
        columns ``station_id`` and ``datetime`` (no rows) if no month had
        data.

    Raises:
        BikeCountDataError: if any fetched month fails schema validation or
            a non-404 HTTP error occurs.
    """
    as_of = as_of or date.today()
    months = pd.period_range(
        start=f"{station.start_year}-01", end=as_of.strftime("%Y-%m"), freq="M"
    )
    frames = [
        frame
        for period in months
        if (
            frame := fetch_station_month(
                station.station_id, period.year, period.month, timeout=timeout
            )
        )
        is not None
    ]
    if not frames:
        return pd.DataFrame(columns=["station_id", "datetime"])
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="datetime", keep="last")
    return combined.sort_values("datetime").reset_index(drop=True)


def find_missing_intervals(
    df: pd.DataFrame, freq_minutes: int = EXPECTED_INTERVAL_MINUTES
) -> pd.DatetimeIndex:
    """Find missing 15-minute timestamps within a station's covered range.

    Args:
        df: DataFrame with a ``datetime`` column (as produced by
            `fetch_station_data` / `parse_station_csv`); need not be sorted
            or deduplicated.
        freq_minutes: Expected spacing between consecutive intervals, in
            minutes.

    Returns:
        Sorted `DatetimeIndex` of timestamps that fall within
        ``[df["datetime"].min(), df["datetime"].max()]`` at the expected
        cadence but are absent from `df`. Empty if there are no gaps.

    Raises:
        BikeCountDataError: if `df` has no ``datetime`` column or is empty.
    """
    if "datetime" not in df.columns:
        raise BikeCountDataError("DataFrame has no 'datetime' column.")
    if df.empty:
        raise BikeCountDataError(
            "DataFrame is empty; cannot determine a coverage range."
        )
    present = pd.DatetimeIndex(df["datetime"]).unique()
    expected = pd.date_range(
        start=present.min(), end=present.max(), freq=f"{freq_minutes}min"
    )
    return expected.difference(present)


def summarize_coverage(
    df: pd.DataFrame,
    station_id: str,
    freq_minutes: int = EXPECTED_INTERVAL_MINUTES,
) -> dict[str, object]:
    """Summarize date coverage and missing-interval counts for one station.

    Args:
        df: Station data as returned by `fetch_station_data`.
        station_id: Station identifier, included in the returned report.
        freq_minutes: Expected spacing between consecutive intervals, in
            minutes.

    Returns:
        Dict with keys ``station_id``, ``first_timestamp``, ``last_timestamp``,
        ``n_records`` (distinct timestamps present), ``n_expected``
        (timestamps expected across the covered range), ``n_missing``, and
        ``missing_timestamps`` (list of the missing timestamps).

    Raises:
        BikeCountDataError: if `df` is empty or has no ``datetime`` column.
    """
    missing = find_missing_intervals(df, freq_minutes=freq_minutes)
    n_records = int(df["datetime"].nunique())
    return {
        "station_id": station_id,
        "first_timestamp": df["datetime"].min(),
        "last_timestamp": df["datetime"].max(),
        "n_records": n_records,
        "n_expected": n_records + len(missing),
        "n_missing": int(len(missing)),
        "missing_timestamps": list(missing),
    }


def save_station_data(df: pd.DataFrame, output_dir: Path, station_id: str) -> Path:
    """Save one station's data as CSV, deterministically.

    Always writes the full, sorted, deduplicated DataFrame in one shot
    (rather than appending), so calling this repeatedly with the same
    logical data produces byte-identical output regardless of call order.

    Args:
        df: Station data as returned by `fetch_station_data`.
        output_dir: Directory to write into; created if missing.
        station_id: Station identifier, used to name the output file.

    Returns:
        Path to the written CSV file.

    Raises:
        BikeCountDataError: if `df` has no ``datetime`` column, or if
            `station_id` is not a safe filename component.
    """
    if "datetime" not in df.columns:
        raise BikeCountDataError("DataFrame has no 'datetime' column.")
    _validate_station_id(station_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{station_id}.csv"
    ordered = df.drop_duplicates(subset="datetime").sort_values("datetime")
    ordered = ordered.rename(columns=_sanitize_csv_header)
    ordered.reset_index(drop=True).to_csv(path, index=False)
    return path


def load_station_data(path: Path) -> pd.DataFrame:
    """Load a station CSV previously written by `save_station_data`.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with ``datetime`` parsed back to ``datetime64[ns]``.

    Raises:
        BikeCountDataError: if the file is missing the ``datetime`` column.
    """
    df = pd.read_csv(path)
    if "datetime" not in df.columns:
        raise BikeCountDataError(f"{path} is missing the 'datetime' column.")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def save_stations_index(stations: list[Station], output_dir: Path) -> Path:
    """Save station metadata (id, name, start year, channels) as JSON.

    Args:
        stations: Stations as returned by `list_stations`.
        output_dir: Directory to write into; created if missing.

    Returns:
        Path to the written ``stations.json`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "stations.json"
    payload = [
        {
            "station_id": station.station_id,
            "name": station.name,
            "start_year": station.start_year,
            "channels": [
                {"channel_id": channel.channel_id, "description": channel.description}
                for channel in station.channels
            ],
        }
        for station in stations
    ]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

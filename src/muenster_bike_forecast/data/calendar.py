"""Calendar/holiday feature data: public holidays and NRW school holidays.

Two independent data sources feed this module, deliberately kept separate
because they have very different reliability characteristics:

- **Public holidays** (gesetzliche Feiertage) are rule-computable for any
  year via the `holidays <https://pypi.org/project/holidays/>`_ Python
  library (``holidays.Germany(subdiv="NW")``). No network call is needed
  and there is no staleness risk.
- **NRW school holidays** (Schulferien) are set per-year by the state
  ministry of education and are *not* rule-computable, so they are fetched
  once from the `OpenHolidays API <https://openholidaysapi.org/>`_
  (``GET /SchoolHolidays``, no API key required) and cached as a static
  file rather than re-fetched on every run.

  Discovered live from the API (as of 2026-07): the endpoint accepts query
  parameters ``countryIsoCode`` (e.g. ``"DE"``), ``subdivisionCode`` (e.g.
  ``"DE-NW"``), ``validFrom`` / ``validTo`` (``YYYY-MM-DD``), and
  ``languageIsoCode`` (e.g. ``"DE"``). It returns a JSON array of objects
  shaped like::

      {
        "id": "29bd10aa-e6b8-4760-a44e-6a88ab783f03",
        "startDate": "2024-12-23",
        "endDate": "2025-01-06",
        "type": "School",
        "name": [{"language": "DE", "text": "Weihnachtsferien"}],
        "regionalScope": "Regional",
        "temporalScope": "FullDay",
        "nationwide": false,
        "subdivisions": [{"code": "DE-NW", "shortName": "NW"}]
      }

  The API rejects a ``validFrom``/``validTo`` span wider than 1095 days
  (HTTP 400, ``"The maximum date range is 1095 days."``), so multi-year
  fetches are chunked one calendar year per request here.

  School-holiday data from OpenHolidays API is licensed **ODbL-1.0**;
  redistributing it (including the cached file this module writes)
  requires attribution to OpenHolidays API (https://openholidaysapi.org/).

All HTTP/JSON content from the OpenHolidays API is treated as untrusted
input and is schema-validated before use, consistent with
`muenster_bike_forecast.data.bike_counts` and
`muenster_bike_forecast.data.weather`.

Public holidays via the `holidays` library are MIT-licensed (no
attribution required). See the README's "Data sources & attribution"
section for the full citation of both sources.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import holidays
import pandas as pd
import requests

OPENHOLIDAYS_BASE_URL: Final[str] = "https://openholidaysapi.org"
DEFAULT_COUNTRY_ISO_CODE: Final[str] = "DE"
DEFAULT_SUBDIVISION_CODE: Final[str] = "DE-NW"
DEFAULT_LANGUAGE_ISO_CODE: Final[str] = "DE"
DEFAULT_PUBLIC_HOLIDAY_SUBDIV: Final[str] = "NW"
SCHOOL_HOLIDAY_TYPE: Final[str] = "School"
# The OpenHolidays API rejects validFrom/validTo spans wider than this many
# days (HTTP 400), so fetch_school_holidays chunks requests one calendar
# year at a time, which is always well within the limit.
MAX_API_RANGE_DAYS: Final[int] = 1095


class SchoolHolidayFetchError(Exception):
    """Raised when NRW school holidays cannot be retrieved from OpenHolidays API."""


class SchoolHolidaySchemaError(Exception):
    """Raised when OpenHolidays API data does not match the expected schema."""


def public_holidays(
    start_year: int, end_year: int, subdiv: str = DEFAULT_PUBLIC_HOLIDAY_SUBDIV
) -> pd.DataFrame:
    """Computes German public holidays for a subdivision over a year range.

    Pure function: uses the `holidays` library's rule-based calendar, no
    network call and no side effects.

    Args:
        start_year: First calendar year (inclusive) to include.
        end_year: Last calendar year (inclusive) to include.
        subdiv: ISO 3166-2 subdivision code without the country prefix,
            e.g. ``"NW"`` for Nordrhein-Westfalen (used as
            ``holidays.Germany(subdiv=subdiv, ...)``).

    Returns:
        DataFrame with columns ``date`` (``datetime64[ns]``) and ``name``
        (str, German holiday name), one row per holiday, sorted by
        ``date``.

    Raises:
        ValueError: if `start_year` is greater than `end_year`.
    """
    if start_year > end_year:
        raise ValueError(f"start_year ({start_year}) must be <= end_year ({end_year}).")
    de_holidays = holidays.Germany(
        subdiv=subdiv, years=range(start_year, end_year + 1), language="de"
    )
    rows = sorted(de_holidays.items())
    return pd.DataFrame(
        {
            "date": pd.to_datetime([day for day, _ in rows]),
            "name": [name for _, name in rows],
        }
    )


def _parse_school_holidays_json(raw: object, subdivision_code: str) -> pd.DataFrame:
    """Validates and converts a raw OpenHolidays API JSON payload.

    Args:
        raw: Parsed JSON body of a ``GET /SchoolHolidays`` response
            (expected: a list of objects with keys ``id``, ``startDate``,
            ``endDate``, ``type``, ``name``).
        subdivision_code: Subdivision code the request was made for (used
            for error messages and stamped onto every returned row).

    Returns:
        DataFrame with columns ``id`` (str), ``start_date``
        (``datetime64[ns]``), ``end_date`` (``datetime64[ns]``), ``name``
        (str) and ``subdivision_code`` (str). Empty (zero rows, correct
        columns) if `raw` is an empty list — a legitimate response for a
        year with no fetched periods yet.

    Raises:
        SchoolHolidaySchemaError: if `raw` is not a list, an entry is not
            an object with the expected keys, ``type`` is not
            ``"School"``, ``startDate``/``endDate`` cannot be parsed as
            dates, or ``name`` is not a non-empty list of
            ``{"language": ..., "text": ...}`` objects.
    """
    columns = ["id", "start_date", "end_date", "name", "subdivision_code"]
    if not isinstance(raw, list):
        raise SchoolHolidaySchemaError(
            f"Expected OpenHolidays API response to be a JSON list, got "
            f"{type(raw).__name__}."
        )
    if not raw:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise SchoolHolidaySchemaError(
                f"Expected school-holiday entry to be a JSON object, got {entry!r}."
            )
        missing_keys = {"id", "startDate", "endDate", "type", "name"} - entry.keys()
        if missing_keys:
            raise SchoolHolidaySchemaError(
                f"School-holiday entry {entry!r} is missing required keys: "
                f"{sorted(missing_keys)}."
            )
        if entry["type"] != SCHOOL_HOLIDAY_TYPE:
            raise SchoolHolidaySchemaError(
                f"Expected entry type {SCHOOL_HOLIDAY_TYPE!r}, got "
                f"{entry['type']!r} in entry {entry!r}."
            )
        names = entry["name"]
        if not isinstance(names, list) or not names:
            raise SchoolHolidaySchemaError(
                f"School-holiday entry {entry!r} has no usable 'name' list."
            )
        try:
            name_text = str(names[0]["text"])
        except (TypeError, KeyError) as exc:
            raise SchoolHolidaySchemaError(
                f"School-holiday entry {entry!r} has a malformed 'name' entry."
            ) from exc
        rows.append(
            {
                "id": str(entry["id"]),
                "start_date": entry["startDate"],
                "end_date": entry["endDate"],
                "name": name_text,
                "subdivision_code": subdivision_code,
            }
        )

    df = pd.DataFrame(rows, columns=columns)
    for date_column in ("start_date", "end_date"):
        try:
            df[date_column] = pd.to_datetime(
                df[date_column], format="%Y-%m-%d", errors="raise"
            )
        except (ValueError, TypeError) as exc:
            raise SchoolHolidaySchemaError(
                f"School-holiday data has unparsable '{date_column}' values: {exc}"
            ) from exc
    if (df["end_date"] < df["start_date"]).any():
        raise SchoolHolidaySchemaError(
            "School-holiday data has one or more entries with end_date before "
            "start_date."
        )
    return df


def fetch_school_holidays_for_year(
    year: int,
    country_iso_code: str = DEFAULT_COUNTRY_ISO_CODE,
    subdivision_code: str = DEFAULT_SUBDIVISION_CODE,
    language_iso_code: str = DEFAULT_LANGUAGE_ISO_CODE,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetches and validates NRW school-holiday periods for one calendar year.

    Args:
        year: Calendar year to fetch (queried as ``validFrom=<year>-01-01``
            through ``validTo=<year>-12-31``).
        country_iso_code: ISO 3166-1 alpha-2 country code.
        subdivision_code: ISO 3166-2 subdivision code, e.g. ``"DE-NW"``.
        language_iso_code: Language for the ``name`` field, e.g. ``"DE"``.
        session: Optional ``requests.Session`` to reuse connections / allow
            mocking in tests.
        timeout: HTTP request timeout in seconds.

    Returns:
        DataFrame as returned by `_parse_school_holidays_json`.

    Raises:
        SchoolHolidayFetchError: on any network/HTTP error, or if the
            response body is not valid JSON.
        SchoolHolidaySchemaError: if the response does not match the
            expected school-holiday schema.
    """
    http = session if session is not None else requests
    params = {
        "countryIsoCode": country_iso_code,
        "subdivisionCode": subdivision_code,
        "validFrom": f"{year:04d}-01-01",
        "validTo": f"{year:04d}-12-31",
        "languageIsoCode": language_iso_code,
    }
    url = f"{OPENHOLIDAYS_BASE_URL}/SchoolHolidays"
    try:
        response = http.get(url, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SchoolHolidayFetchError(
            f"Failed to fetch school holidays for {year} from {url}: {exc}"
        ) from exc
    try:
        raw = response.json()
    except ValueError as exc:
        raise SchoolHolidayFetchError(
            f"Response for school holidays for {year} from {url} is not valid JSON."
        ) from exc
    return _parse_school_holidays_json(raw, subdivision_code=subdivision_code)


def fetch_school_holidays(
    start_year: int,
    end_year: int,
    country_iso_code: str = DEFAULT_COUNTRY_ISO_CODE,
    subdivision_code: str = DEFAULT_SUBDIVISION_CODE,
    language_iso_code: str = DEFAULT_LANGUAGE_ISO_CODE,
    session: requests.Session | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Fetches NRW school-holiday periods for a range of calendar years.

    Issues one request per calendar year (see module docstring: the
    OpenHolidays API rejects a ``validFrom``/``validTo`` span wider than
    `MAX_API_RANGE_DAYS` days), then combines and deduplicates the result.
    Idempotent: repeated calls re-fetch the same source data and return an
    equivalent DataFrame.

    Args:
        start_year: First calendar year (inclusive) to fetch.
        end_year: Last calendar year (inclusive) to fetch.
        country_iso_code: ISO 3166-1 alpha-2 country code.
        subdivision_code: ISO 3166-2 subdivision code, e.g. ``"DE-NW"``.
        language_iso_code: Language for the ``name`` field, e.g. ``"DE"``.
        session: Optional ``requests.Session``.
        timeout: HTTP request timeout in seconds, applied per request.

    Returns:
        Combined DataFrame across all requested years (see
        `_parse_school_holidays_json`), deduplicated by ``id`` and sorted
        by ``start_date``.

    Raises:
        ValueError: if `start_year` is greater than `end_year`.
        SchoolHolidayFetchError: if any year's request fails.
        SchoolHolidaySchemaError: if any year's response fails validation.
    """
    if start_year > end_year:
        raise ValueError(f"start_year ({start_year}) must be <= end_year ({end_year}).")
    frames = [
        fetch_school_holidays_for_year(
            year,
            country_iso_code=country_iso_code,
            subdivision_code=subdivision_code,
            language_iso_code=language_iso_code,
            session=session,
            timeout=timeout,
        )
        for year in range(start_year, end_year + 1)
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="id", keep="last")
    return combined.sort_values("start_date").reset_index(drop=True)


def save_school_holidays(df: pd.DataFrame, output_dir: Path) -> Path:
    """Saves fetched school-holiday periods as CSV, idempotently.

    Always writes the full, sorted, deduplicated DataFrame in one shot
    (rather than appending), so calling this repeatedly with the same
    logical data produces byte-identical output regardless of call order.

    Args:
        df: School-holiday data as returned by `fetch_school_holidays`.
        output_dir: Directory to write into; created if missing.

    Returns:
        Path to the written ``school_holidays_nw.csv`` file.

    Raises:
        SchoolHolidaySchemaError: if `df` is missing any of the expected
            columns (``id``, ``start_date``, ``end_date``, ``name``,
            ``subdivision_code``).
    """
    expected_columns = {"id", "start_date", "end_date", "name", "subdivision_code"}
    missing_columns = expected_columns - set(df.columns)
    if missing_columns:
        raise SchoolHolidaySchemaError(
            f"DataFrame is missing required columns: {sorted(missing_columns)}."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "school_holidays_nw.csv"
    ordered = df.drop_duplicates(subset="id").sort_values("start_date")
    ordered.reset_index(drop=True).to_csv(path, index=False)
    return path


def load_school_holidays(path: Path) -> pd.DataFrame:
    """Loads a school-holiday CSV previously written by `save_school_holidays`.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame with ``start_date``/``end_date`` parsed back to
        ``datetime64[ns]``.

    Raises:
        SchoolHolidaySchemaError: if the file is missing the ``start_date``
            or ``end_date`` column.
    """
    df = pd.read_csv(path)
    if "start_date" not in df.columns or "end_date" not in df.columns:
        raise SchoolHolidaySchemaError(
            f"{path} is missing the 'start_date' and/or 'end_date' column."
        )
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    return df

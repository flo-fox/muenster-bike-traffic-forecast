"""Tests for `muenster_bike_forecast.data.calendar`.

`public_holidays` is tested directly (deterministic, rule-based, no
network). The OpenHolidays API fetch/validation logic is tested against a
small, hand-written fake payload built to mirror the real API's response
shape, with `requests.get` replaced by a fake session - no live network
calls are made anywhere in this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from muenster_bike_forecast.data import calendar
from muenster_bike_forecast.data.calendar import (
    SchoolHolidayFetchError,
    SchoolHolidaySchemaError,
    fetch_school_holidays,
    fetch_school_holidays_for_year,
    load_school_holidays,
    public_holidays,
    save_school_holidays,
)

# ---------------------------------------------------------------------------
# Fakes for the OpenHolidays API HTTP boundary
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, json_body: object = None, status_code: int = 200):
        self._json_body = json_body
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise calendar.requests.HTTPError(f"status {self.status_code}")

    def json(self) -> object:
        return self._json_body


class _FakeSession:
    """Minimal stand-in for requests.Session, routes GET by validFrom year."""

    def __init__(self, by_year: dict[int, _FakeResponse]):
        self._by_year = by_year

    def get(self, url: str, params: dict, timeout: float = 30.0) -> _FakeResponse:
        year = int(params["validFrom"][:4])
        try:
            return self._by_year[year]
        except KeyError:
            raise AssertionError(f"Unexpected year requested in test: {year}")


def _entry(
    entry_id: str,
    start_date: str,
    end_date: str,
    name_text: str,
    entry_type: str = "School",
) -> dict:
    """Builds one OpenHolidays-shaped SchoolHolidays entry."""
    return {
        "id": entry_id,
        "startDate": start_date,
        "endDate": end_date,
        "type": entry_type,
        "name": [{"language": "DE", "text": name_text}],
        "regionalScope": "Regional",
        "temporalScope": "FullDay",
        "nationwide": False,
        "subdivisions": [{"code": "DE-NW", "shortName": "NW"}],
    }


_SAMPLE_2025_ENTRIES = [
    _entry(
        "29bd10aa-e6b8-4760-a44e-6a88ab783f03",
        "2024-12-23",
        "2025-01-06",
        "Weihnachtsferien",
    ),
    _entry(
        "c45b7237-ad6d-40ce-ba16-735541b1be82",
        "2025-04-14",
        "2025-04-26",
        "Osterferien",
    ),
    _entry(
        "589f490d-9bbb-48ef-a402-ae9af623edfa",
        "2025-06-10",
        "2025-06-10",
        "Pfingstferien",
    ),
]


# ---------------------------------------------------------------------------
# public_holidays
# ---------------------------------------------------------------------------


def test_public_holidays_contains_known_fixed_dates() -> None:
    df = public_holidays(2025, 2025)

    dates = set(df["date"].dt.date.astype(str))
    assert "2025-01-01" in dates  # Neujahr
    assert "2025-10-03" in dates  # Tag der Deutschen Einheit
    assert "2025-12-25" in dates  # 1. Weihnachtstag


def test_public_holidays_nrw_includes_allerheiligen() -> None:
    # Allerheiligen (All Saints' Day, Nov 1) is a public holiday in NRW
    # (a majority-Catholic state) but not in every German state.
    df = public_holidays(2025, 2025, subdiv="NW")

    dates = set(df["date"].dt.date.astype(str))
    assert "2025-11-01" in dates


def test_public_holidays_spans_full_year_range() -> None:
    df = public_holidays(2019, 2020)

    years = set(df["date"].dt.year)
    assert years == {2019, 2020}


def test_public_holidays_returns_expected_columns() -> None:
    df = public_holidays(2025, 2025)

    assert list(df.columns) == ["date", "name"]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing


def test_public_holidays_rejects_start_after_end() -> None:
    with pytest.raises(ValueError):
        public_holidays(2025, 2019)


# ---------------------------------------------------------------------------
# fetch_school_holidays_for_year / fetch_school_holidays
# ---------------------------------------------------------------------------


def test_fetch_school_holidays_for_year_parses_valid_response() -> None:
    session = _FakeSession({2025: _FakeResponse(_SAMPLE_2025_ENTRIES)})

    df = fetch_school_holidays_for_year(2025, session=session)

    assert list(df.columns) == [
        "id",
        "start_date",
        "end_date",
        "name",
        "subdivision_code",
    ]
    assert len(df) == 3
    assert pd.api.types.is_datetime64_any_dtype(df["start_date"])
    assert pd.api.types.is_datetime64_any_dtype(df["end_date"])
    assert "Osterferien" in df["name"].tolist()
    assert (df["subdivision_code"] == "DE-NW").all()


def test_fetch_school_holidays_for_year_empty_list_returns_empty_frame() -> None:
    session = _FakeSession({2025: _FakeResponse([])})

    df = fetch_school_holidays_for_year(2025, session=session)

    assert df.empty
    assert list(df.columns) == [
        "id",
        "start_date",
        "end_date",
        "name",
        "subdivision_code",
    ]


def test_fetch_school_holidays_for_year_wraps_http_errors() -> None:
    session = _FakeSession({2025: _FakeResponse(status_code=500)})

    with pytest.raises(SchoolHolidayFetchError):
        fetch_school_holidays_for_year(2025, session=session)


def test_fetch_school_holidays_for_year_rejects_non_list_payload() -> None:
    session = _FakeSession(
        {2025: _FakeResponse({"title": "Bad Request", "status": 400})}
    )

    with pytest.raises(SchoolHolidaySchemaError):
        fetch_school_holidays_for_year(2025, session=session)


def test_fetch_school_holidays_for_year_rejects_missing_keys() -> None:
    bad_entry = {"id": "x", "startDate": "2025-01-01"}  # missing endDate/type/name
    session = _FakeSession({2025: _FakeResponse([bad_entry])})

    with pytest.raises(SchoolHolidaySchemaError, match="missing required keys"):
        fetch_school_holidays_for_year(2025, session=session)


def test_fetch_school_holidays_for_year_rejects_wrong_type() -> None:
    entry = _entry("x", "2025-01-01", "2025-01-05", "Not a school holiday")
    entry["type"] = "Public"
    session = _FakeSession({2025: _FakeResponse([entry])})

    with pytest.raises(SchoolHolidaySchemaError):
        fetch_school_holidays_for_year(2025, session=session)


def test_fetch_school_holidays_for_year_rejects_unparsable_dates() -> None:
    entry = _entry("x", "not-a-date", "2025-01-05", "Ferien")
    session = _FakeSession({2025: _FakeResponse([entry])})

    with pytest.raises(SchoolHolidaySchemaError):
        fetch_school_holidays_for_year(2025, session=session)


def test_fetch_school_holidays_for_year_rejects_end_before_start() -> None:
    entry = _entry("x", "2025-06-10", "2025-06-01", "Ferien")
    session = _FakeSession({2025: _FakeResponse([entry])})

    with pytest.raises(SchoolHolidaySchemaError, match="end_date before"):
        fetch_school_holidays_for_year(2025, session=session)


def test_fetch_school_holidays_chunks_by_year_and_dedupes_overlap() -> None:
    # The Christmas period spans two calendar years and is a legitimate
    # duplicate hit (same id) across the two per-year requests.
    christmas = _entry(
        "29bd10aa-e6b8-4760-a44e-6a88ab783f03",
        "2024-12-23",
        "2025-01-06",
        "Weihnachtsferien",
    )
    session = _FakeSession(
        {
            2024: _FakeResponse([christmas]),
            2025: _FakeResponse([christmas, _SAMPLE_2025_ENTRIES[1]]),
        }
    )

    df = fetch_school_holidays(2024, 2025, session=session)

    assert len(df) == 2  # deduped by id, not 3
    assert df["start_date"].is_monotonic_increasing


def test_fetch_school_holidays_rejects_start_after_end() -> None:
    with pytest.raises(ValueError):
        fetch_school_holidays(2025, 2019)


# ---------------------------------------------------------------------------
# save_school_holidays / load_school_holidays
# ---------------------------------------------------------------------------


def test_save_school_holidays_is_idempotent(tmp_path: Path) -> None:
    session = _FakeSession({2025: _FakeResponse(_SAMPLE_2025_ENTRIES)})
    df = fetch_school_holidays_for_year(2025, session=session)

    path_first = save_school_holidays(df, tmp_path)
    path_second = save_school_holidays(df, tmp_path)

    assert path_first == path_second
    assert len(list(tmp_path.glob("*.csv"))) == 1
    saved = pd.read_csv(path_first)
    assert len(saved) == 3


def test_save_school_holidays_rejects_missing_columns(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": ["x"]})
    with pytest.raises(SchoolHolidaySchemaError):
        save_school_holidays(df, tmp_path)


def test_load_school_holidays_round_trips(tmp_path: Path) -> None:
    session = _FakeSession({2025: _FakeResponse(_SAMPLE_2025_ENTRIES)})
    df = fetch_school_holidays_for_year(2025, session=session)
    path = save_school_holidays(df, tmp_path)

    loaded = load_school_holidays(path)

    assert pd.api.types.is_datetime64_any_dtype(loaded["start_date"])
    assert pd.api.types.is_datetime64_any_dtype(loaded["end_date"])
    assert len(loaded) == 3


def test_load_school_holidays_rejects_missing_date_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"id": ["x"], "name": ["y"]}).to_csv(path, index=False)

    with pytest.raises(SchoolHolidaySchemaError):
        load_school_holidays(path)

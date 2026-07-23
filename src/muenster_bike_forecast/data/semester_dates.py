"""University semester / lecture-period feature data for WWU Münster.

This is deliberately **not** a fetch/scrape module. Both WWU Münster's and
FH Münster's own semester-date pages are HTML/PDF only with no structured
export, but the NRW Ministry of Culture and Science (MKW) publishes a
standardized calendar of KMK-recommended lecture periods
(*Vorlesungszeiten*) for all NRW universities at
https://www.mkw.nrw/service/vorlesungszeiten, which WWU Münster follows
closely. That page lists roughly 20 semesters, so a small hand-transcribed
lookup table is simpler and more robust than a scraper.

Two kinds of date range are distinguished per semester:

- **Semester start/end**: the full, legally-fixed half-year boundary used
  in German higher education (Wintersemester = 1 Oct - 31 Mar,
  Sommersemester = 1 Apr - 30 Sep). This is a fixed convention, not
  ministry-published data, so it is the same rule for every year in the
  table.
- **Lecture-period start/end** (*Vorlesungszeit*): the actual teaching
  period within a semester, outside of which is a semester break
  (*vorlesungsfreie Zeit*, e.g. over Christmas or in late summer). This is
  the ministry-published figure and the one most relevant to bike-traffic
  demand, since commuting patterns differ between lecture periods and
  breaks.

**Data provenance (verified live 2026-07-23):**

- **2022-2030** (Wintersemester 2022/2023 through Sommersemester 2030):
  lecture-period dates transcribed directly from the "Universitäten" table
  on https://www.mkw.nrw/service/vorlesungszeiten (WWU Münster is a
  Universität, not a Fachhochschule; Fachhochschule Münster's lecture
  periods differ by roughly 1-3 weeks and are out of scope here). Rows are
  tagged ``source="mkw_nrw"``.
- **2018/2019-2022 (WS2018/19 through SS2022)**: the ministry page's
  window starts at Wintersemester 2022/2023, so earlier semesters are not
  ministry-published. These are extrapolated from the stable NRW pattern
  visible in the sourced years (lecture periods starting ~7-13 Oct /
  ~3-17 Apr and ending ~31 Jan-6 Feb / ~12-28 Jul), rounded to
  representative approximate dates (14 Oct - 3 Feb for winter, 15 Apr -
  15 Jul for summer). These are **not exact** — real historical dates
  (including any COVID-era shifts in 2020/2021) may differ by up to a
  couple of weeks — and are tagged ``source="extrapolated"`` so callers
  can filter or treat them with less confidence if needed.

Semester boundaries and lecture periods are contiguous across the whole
table (one semester's end date is always the day before the next
semester's start date), so every date from the first semester's start to
the last semester's end falls inside exactly one semester.

**Known limitation**: the ministry page (and therefore this table) gives
only the single outer start/end date of each lecture period. In practice,
WWU Münster (like other German universities) observes a short lecture-free
recess around Christmas/New Year *within* that outer window (exact dates
vary by year and are not ministry-published in this dataset). This module
does **not** model that inner recess separately — a date such as 24
December therefore classifies as ``is_lecture_period=True`` if it falls
within the winter semester's outer lecture-period window, which
understates real semester breaks around the turn of the year. Treat
`is_lecture_period` as "within the official teaching-period window", not
as a guarantee that lectures are actually held that day.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

MKW_SOURCE: str = "mkw_nrw"
EXTRAPOLATED_SOURCE: str = "extrapolated"


class SemesterDateRangeError(Exception):
    """Raised when a date falls outside the range covered by the semester table."""


@dataclass(frozen=True)
class SemesterPeriod:
    """One semester's boundary and lecture-period dates.

    Attributes:
        semester_id: Short label, e.g. ``"WS2024/25"`` or ``"SS2025"``.
        semester_start: First calendar day of the semester (fixed
            convention: 1 Oct for winter, 1 Apr for summer).
        semester_end: Last calendar day of the semester (fixed convention:
            31 Mar for winter, 30 Sep for summer).
        lecture_start: First day of the lecture period (*Vorlesungszeit*).
        lecture_end: Last day of the lecture period.
        source: Provenance of the lecture-period dates: `MKW_SOURCE` if
            transcribed from the NRW ministry page, `EXTRAPOLATED_SOURCE`
            if extrapolated from the stable NRW pattern (see module
            docstring).
    """

    semester_id: str
    semester_start: date
    semester_end: date
    lecture_start: date
    lecture_end: date
    source: str


def _winter(
    start_year: int,
    lecture_start: tuple[int, int],
    lecture_end: tuple[int, int],
    source: str,
) -> SemesterPeriod:
    """Builds a Wintersemester `SemesterPeriod` (1 Oct - 31 Mar).

    Args:
        start_year: Calendar year the winter semester begins in, e.g. 2024
            for "WS2024/25".
        lecture_start: ``(month, day)`` of the lecture period's start,
            within `start_year`.
        lecture_end: ``(month, day)`` of the lecture period's end, within
            `start_year + 1`.
        source: `MKW_SOURCE` or `EXTRAPOLATED_SOURCE`.

    Returns:
        The corresponding `SemesterPeriod`.
    """
    end_year = start_year + 1
    return SemesterPeriod(
        semester_id=f"WS{start_year}/{end_year % 100:02d}",
        semester_start=date(start_year, 10, 1),
        semester_end=date(end_year, 3, 31),
        lecture_start=date(start_year, *lecture_start),
        lecture_end=date(end_year, *lecture_end),
        source=source,
    )


def _summer(
    year: int,
    lecture_start: tuple[int, int],
    lecture_end: tuple[int, int],
    source: str,
) -> SemesterPeriod:
    """Builds a Sommersemester `SemesterPeriod` (1 Apr - 30 Sep).

    Args:
        year: Calendar year of the summer semester, e.g. 2025 for "SS2025".
        lecture_start: ``(month, day)`` of the lecture period's start.
        lecture_end: ``(month, day)`` of the lecture period's end.
        source: `MKW_SOURCE` or `EXTRAPOLATED_SOURCE`.

    Returns:
        The corresponding `SemesterPeriod`.
    """
    return SemesterPeriod(
        semester_id=f"SS{year}",
        semester_start=date(year, 4, 1),
        semester_end=date(year, 9, 30),
        lecture_start=date(year, *lecture_start),
        lecture_end=date(year, *lecture_end),
        source=source,
    )


# Extrapolated approximate pattern used for semesters before the ministry
# page's published window (see module docstring): winter lecture periods
# ~14 Oct - 3 Feb, summer lecture periods ~15 Apr - 15 Jul.
_EXTRAPOLATED_WINTER_LECTURE = ((10, 14), (2, 3))
_EXTRAPOLATED_SUMMER_LECTURE = ((4, 15), (7, 15))

SEMESTER_PERIODS: tuple[SemesterPeriod, ...] = (
    # --- Extrapolated (ministry page does not cover before WS2022/23) ---
    _winter(2018, *_EXTRAPOLATED_WINTER_LECTURE, EXTRAPOLATED_SOURCE),
    _summer(2019, *_EXTRAPOLATED_SUMMER_LECTURE, EXTRAPOLATED_SOURCE),
    _winter(2019, *_EXTRAPOLATED_WINTER_LECTURE, EXTRAPOLATED_SOURCE),
    _summer(2020, *_EXTRAPOLATED_SUMMER_LECTURE, EXTRAPOLATED_SOURCE),
    _winter(2020, *_EXTRAPOLATED_WINTER_LECTURE, EXTRAPOLATED_SOURCE),
    _summer(2021, *_EXTRAPOLATED_SUMMER_LECTURE, EXTRAPOLATED_SOURCE),
    _winter(2021, *_EXTRAPOLATED_WINTER_LECTURE, EXTRAPOLATED_SOURCE),
    _summer(2022, *_EXTRAPOLATED_SUMMER_LECTURE, EXTRAPOLATED_SOURCE),
    # --- Sourced from https://www.mkw.nrw/service/vorlesungszeiten (Universitäten table) ---
    _winter(2022, (10, 10), (2, 3), MKW_SOURCE),
    _summer(2023, (4, 3), (7, 14), MKW_SOURCE),
    _winter(2023, (10, 9), (2, 2), MKW_SOURCE),
    _summer(2024, (4, 8), (7, 19), MKW_SOURCE),
    _winter(2024, (10, 7), (1, 31), MKW_SOURCE),
    _summer(2025, (4, 7), (7, 18), MKW_SOURCE),
    _winter(2025, (10, 13), (2, 6), MKW_SOURCE),
    _summer(2026, (4, 13), (7, 24), MKW_SOURCE),
    _winter(2026, (10, 12), (2, 5), MKW_SOURCE),
    _summer(2027, (4, 5), (7, 16), MKW_SOURCE),
    _winter(2027, (10, 11), (2, 4), MKW_SOURCE),
    _summer(2028, (4, 17), (7, 28), MKW_SOURCE),
    _winter(2028, (10, 9), (2, 2), MKW_SOURCE),
    _summer(2029, (4, 9), (7, 20), MKW_SOURCE),
    _winter(2029, (10, 8), (2, 1), MKW_SOURCE),
    _summer(2030, (4, 1), (7, 12), MKW_SOURCE),
)


@dataclass(frozen=True)
class DateSemesterClassification:
    """Semester classification for a single date.

    Attributes:
        target_date: The classified date.
        semester_id: The enclosing semester's `SemesterPeriod.semester_id`.
        is_lecture_period: `True` if `target_date` falls within that
            semester's lecture period, `False` if it falls within a
            semester break (vorlesungsfreie Zeit).
        source: The enclosing semester's `SemesterPeriod.source`.
    """

    target_date: date
    semester_id: str
    is_lecture_period: bool
    source: str


def _to_date(value: date | pd.Timestamp | str) -> date:
    """Normalizes a date-like value to a plain `datetime.date`.

    Args:
        value: A `datetime.date`, `datetime.datetime`, `pandas.Timestamp`,
            or ISO-format date string.

    Returns:
        The corresponding `datetime.date`, with any time-of-day component
        dropped.

    Raises:
        SemesterDateRangeError: if `value` cannot be interpreted as a date.
    """
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError) as exc:
        raise SemesterDateRangeError(
            f"Could not interpret {value!r} as a date."
        ) from exc


def covered_range(
    periods: Sequence[SemesterPeriod] = SEMESTER_PERIODS,
) -> tuple[date, date]:
    """Returns the earliest and latest date covered by a semester table.

    Args:
        periods: Semester table to inspect, sorted or not.

    Returns:
        ``(earliest_semester_start, latest_semester_end)``.

    Raises:
        SemesterDateRangeError: if `periods` is empty.
    """
    if not periods:
        raise SemesterDateRangeError("Semester table is empty; no covered range.")
    return (
        min(period.semester_start for period in periods),
        max(period.semester_end for period in periods),
    )


def classify_date(
    target_date: date | pd.Timestamp | str,
    periods: Sequence[SemesterPeriod] = SEMESTER_PERIODS,
) -> DateSemesterClassification:
    """Classifies a single date as lecture period or semester break.

    Pure function: looks up `target_date` in the static `periods` table; no
    network call, no side effects.

    Args:
        target_date: Date to classify (any of `datetime.date`,
            `datetime.datetime`, `pandas.Timestamp`, ISO date string; any
            time-of-day component is ignored).
        periods: Semester table to use, sorted or not (defaults to the
            module-level `SEMESTER_PERIODS`).

    Returns:
        The `DateSemesterClassification` for `target_date`.

    Raises:
        SemesterDateRangeError: if `target_date` falls before the earliest
            or after the latest date covered by `periods` — the table has
            no data for that date, so it is not guessed at.
    """
    normalized = _to_date(target_date)
    earliest_start, latest_end = covered_range(periods)
    if normalized < earliest_start or normalized > latest_end:
        raise SemesterDateRangeError(
            f"{normalized.isoformat()} is outside the semester table's covered "
            f"range [{earliest_start.isoformat()}, {latest_end.isoformat()}]."
        )
    for period in periods:
        if period.semester_start <= normalized <= period.semester_end:
            is_lecture_period = period.lecture_start <= normalized <= period.lecture_end
            return DateSemesterClassification(
                target_date=normalized,
                semester_id=period.semester_id,
                is_lecture_period=is_lecture_period,
                source=period.source,
            )
    # Unreachable if `periods` is contiguous (as SEMESTER_PERIODS is), but
    # guards against a caller-supplied table with a gap inside its range.
    raise SemesterDateRangeError(
        f"{normalized.isoformat()} falls within the table's covered range but "
        "no semester period contains it (the supplied table has a gap)."
    )


def classify_dates(
    dates: Iterable[date | pd.Timestamp | str],
    periods: Sequence[SemesterPeriod] = SEMESTER_PERIODS,
) -> pd.DataFrame:
    """Classifies many dates as lecture period or semester break.

    Vectorized convenience wrapper around `classify_date`, suitable for
    tagging a bike-count or weather DataFrame's timestamps with a
    lecture-period feature.

    Args:
        dates: Date-like values to classify (see `classify_date` for
            accepted types per element).
        periods: Semester table to use (defaults to the module-level
            `SEMESTER_PERIODS`).

    Returns:
        DataFrame with one row per input date and columns ``date``
        (datetime64), ``semester_id`` (str), ``is_lecture_period`` (bool),
        and ``source`` (str), in input order.

    Raises:
        SemesterDateRangeError: if any date falls outside the range
            covered by `periods`.
    """
    classifications = [classify_date(value, periods=periods) for value in dates]
    return pd.DataFrame(
        {
            "date": pd.to_datetime([c.target_date for c in classifications]),
            "semester_id": [c.semester_id for c in classifications],
            "is_lecture_period": np.array(
                [c.is_lecture_period for c in classifications], dtype=bool
            ),
            "source": [c.source for c in classifications],
        }
    )

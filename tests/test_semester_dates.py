"""Tests for `muenster_bike_forecast.data.semester_dates`.

Deterministic, no network calls: exercises the static semester-period
table directly.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from muenster_bike_forecast.data.semester_dates import (
    EXTRAPOLATED_SOURCE,
    MKW_SOURCE,
    SEMESTER_PERIODS,
    SemesterDateRangeError,
    classify_date,
    classify_dates,
    covered_range,
)


def test_covered_range_matches_table_extremes() -> None:
    earliest, latest = covered_range(SEMESTER_PERIODS)
    assert earliest == date(2018, 10, 1)
    assert latest == date(2030, 9, 30)


def test_known_lecture_period_date_classifies_as_lecture() -> None:
    # 2024-11-12 is a Tuesday within WS2024/25's ministry-sourced lecture
    # period (07.10.2024 - 31.01.2025).
    result = classify_date(date(2024, 11, 12))
    assert result.is_lecture_period is True
    assert result.semester_id == "WS2024/25"
    assert result.source == MKW_SOURCE


def test_late_august_is_a_semester_break() -> None:
    # Late August sits in the SS/WS gap-free calendar, well after any
    # summer lecture period end and before the next winter lecture start.
    result = classify_date(date(2024, 8, 20))
    assert result.semester_id == "SS2024"
    assert result.is_lecture_period is False
    assert result.source == MKW_SOURCE


def test_mid_semester_break_date_between_semesters() -> None:
    # 15 Sep 2024 falls in SS2024 (semester runs 1 Apr - 30 Sep) after its
    # lecture period (08.04.2024 - 19.07.2024) ended.
    result = classify_date(date(2024, 9, 15))
    assert result.semester_id == "SS2024"
    assert result.is_lecture_period is False


def test_random_november_weekday_is_lecture_period() -> None:
    # 6 Nov 2023 (Monday) within WS2023/24's lecture period
    # (09.10.2023 - 02.02.2024).
    result = classify_date(date(2023, 11, 6))
    assert result.is_lecture_period is True
    assert result.semester_id == "WS2023/24"


def test_extrapolated_period_is_flagged() -> None:
    # 2019 predates the ministry page's published window (starts
    # WS2022/23), so it must come from the extrapolated portion.
    result = classify_date(date(2019, 11, 15))
    assert result.source == EXTRAPOLATED_SOURCE
    assert result.semester_id == "WS2019/20"


def test_semester_boundary_is_inclusive() -> None:
    start_result = classify_date(date(2024, 10, 1))
    assert start_result.semester_id == "WS2024/25"
    end_result = classify_date(date(2025, 3, 31))
    assert end_result.semester_id == "WS2024/25"


def test_date_before_table_range_raises() -> None:
    with pytest.raises(SemesterDateRangeError):
        classify_date(date(2000, 1, 1))


def test_date_after_table_range_raises() -> None:
    with pytest.raises(SemesterDateRangeError):
        classify_date(date(2031, 1, 1))


def test_classify_date_accepts_timestamp_and_string() -> None:
    from_timestamp = classify_date(pd.Timestamp("2024-11-12 08:30"))
    from_string = classify_date("2024-11-12")
    assert from_timestamp.semester_id == from_string.semester_id == "WS2024/25"
    assert from_timestamp.is_lecture_period is True


def test_classify_dates_returns_dataframe_in_input_order() -> None:
    dates = [date(2024, 11, 12), date(2024, 8, 20), date(2023, 11, 6)]
    df = classify_dates(dates)
    assert list(df["semester_id"]) == ["WS2024/25", "SS2024", "WS2023/24"]
    assert list(df["is_lecture_period"]) == [True, False, True]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_classify_dates_out_of_range_raises() -> None:
    with pytest.raises(SemesterDateRangeError):
        classify_dates([date(2024, 1, 1), date(1999, 1, 1)])


def test_table_is_contiguous_with_no_gaps_or_overlaps() -> None:
    sorted_periods = sorted(SEMESTER_PERIODS, key=lambda p: p.semester_start)
    for earlier, later in zip(sorted_periods, sorted_periods[1:]):
        assert later.semester_start == date(
            earlier.semester_end.year,
            earlier.semester_end.month,
            earlier.semester_end.day,
        ) + pd.Timedelta(days=1)


def test_every_lecture_period_is_within_its_semester() -> None:
    for period in SEMESTER_PERIODS:
        assert period.semester_start <= period.lecture_start
        assert period.lecture_end <= period.semester_end
        assert period.lecture_start <= period.lecture_end

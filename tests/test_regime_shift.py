"""Tests for `muenster_bike_forecast.analysis.regime_shift`.

All tests use small, hand-built synthetic data - no dependency on the real
`data/raw/` files.
"""

from __future__ import annotations

import pandas as pd
import pytest

from muenster_bike_forecast.analysis.regime_shift import (
    RegimeShiftError,
    daily_coverage_and_mean,
    find_constant_runs,
    window_summary,
)

# ---------------------------------------------------------------------------
# daily_coverage_and_mean
# ---------------------------------------------------------------------------


def _quarter_hour_index(start: str, n_periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_periods, freq="15min")


def test_daily_coverage_and_mean_distinguishes_gap_from_zero_run() -> None:
    # Day 1: normal data (mean 10). Day 2: fully missing (a gap - dropped
    # from the index entirely, not just null). Day 3: fully present but
    # every value is zero (a zero-run).
    day1 = pd.Series(10.0, index=_quarter_hour_index("2026-01-01", 96))
    day3 = pd.Series(0.0, index=_quarter_hour_index("2026-01-03", 96))
    series = pd.concat([day1, day3])

    daily = daily_coverage_and_mean(series)

    assert daily.loc["2026-01-01", "n_present"] == 96
    assert daily.loc["2026-01-01", "mean"] == pytest.approx(10.0)
    assert daily.loc["2026-01-02", "n_present"] == 0
    assert pd.isna(daily.loc["2026-01-02", "mean"])
    assert daily.loc["2026-01-03", "n_present"] == 96
    assert daily.loc["2026-01-03", "mean"] == pytest.approx(0.0)


def test_daily_coverage_and_mean_raises_on_empty_series() -> None:
    with pytest.raises(RegimeShiftError):
        daily_coverage_and_mean(pd.Series(dtype=float))


def test_daily_coverage_and_mean_raises_on_non_datetime_index() -> None:
    with pytest.raises(RegimeShiftError):
        daily_coverage_and_mean(pd.Series([1.0, 2.0]))


# ---------------------------------------------------------------------------
# find_constant_runs
# ---------------------------------------------------------------------------


def test_find_constant_runs_finds_gap_and_drops_short_runs() -> None:
    daily = pd.DataFrame(
        {"n_present": [96, 0, 0, 0, 96, 0, 96]},
        index=pd.date_range("2026-01-01", periods=7, freq="D"),
    )

    runs = find_constant_runs(daily, column="n_present", value=0, min_run_days=2)

    assert len(runs) == 1
    assert runs.loc[0, "start"] == pd.Timestamp("2026-01-02")
    assert runs.loc[0, "end"] == pd.Timestamp("2026-01-04")
    assert runs.loc[0, "n_days"] == 3


def test_find_constant_runs_includes_run_extending_to_series_end() -> None:
    daily = pd.DataFrame(
        {"mean": [10.0, 0.0, 0.0]},
        index=pd.date_range("2026-01-01", periods=3, freq="D"),
    )

    runs = find_constant_runs(daily, column="mean", value=0.0, min_run_days=1)

    assert len(runs) == 1
    assert runs.loc[0, "end"] == pd.Timestamp("2026-01-03")


def test_find_constant_runs_raises_on_missing_column() -> None:
    daily = pd.DataFrame({"mean": [0.0]}, index=pd.date_range("2026-01-01", periods=1))
    with pytest.raises(RegimeShiftError):
        find_constant_runs(daily, column="n_present", value=0, min_run_days=1)


# ---------------------------------------------------------------------------
# window_summary
# ---------------------------------------------------------------------------


def test_window_summary_computes_stats_over_a_slice() -> None:
    series = pd.Series(
        [10.0, 20.0, 30.0], index=pd.date_range("2026-01-01", periods=3, freq="D")
    )

    summary = window_summary(series, "2026-01-01", "2026-01-02")

    assert summary["n_rows"] == 2
    assert summary["mean"] == pytest.approx(15.0)
    assert summary["median"] == pytest.approx(15.0)


def test_window_summary_raises_on_empty_window() -> None:
    series = pd.Series([10.0], index=pd.date_range("2026-01-01", periods=1, freq="D"))
    with pytest.raises(RegimeShiftError):
        window_summary(series, "2027-01-01", "2027-01-02")


def test_window_summary_raises_on_non_datetime_index() -> None:
    with pytest.raises(RegimeShiftError):
        window_summary(pd.Series([1.0, 2.0]), 0, 1)

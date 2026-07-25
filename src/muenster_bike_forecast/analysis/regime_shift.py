"""Diagnostics for level shifts, gaps, and zero-runs in a per-station series.

This module supports diagnostic work (see
``notebooks/12_regime_shift_investigation.ipynb``) into *why* a station's
traffic level changes abruptly rather than drifting gradually - specifically
distinguishing three different things a naive "the mean dropped" observation
could actually be:

1. A **data gap**: no rows at all for a stretch of time (sensor offline,
   station not yet reporting to the source repo). `daily_coverage_and_mean`
   makes this visible by reindexing onto the *full* expected 15-minute grid
   first, so a day with zero rows shows `n_present == 0` rather than being
   silently absent from a `groupby`.
2. A **zero-run**: rows *are* present (full or near-full coverage) but every
   value is exactly zero - a sensor reporting a fault/idle state as `0`
   rather than going silent, or a genuine full closure. `find_constant_runs`
   with `column="mean", value=0.0` finds these.
3. A **genuine level shift**: real, non-zero traffic that settles at a
   durably different mean - what `window_summary` is for, comparing
   before/after (or train/test) windows once gaps and zero-runs have been
   ruled in or out as the actual explanation.

Every function is pure: it accepts a DataFrame/Series already in memory and
returns a new DataFrame/dict; no file I/O happens here (left to the
notebook, matching the convention in
``muenster_bike_forecast.analysis.descriptive``).
"""

from __future__ import annotations

from typing import Final

import pandas as pd

DEFAULT_FREQ_MINUTES: Final[int] = 15


class RegimeShiftError(Exception):
    """Raised when regime-shift diagnostics cannot be computed as requested.

    Covers missing required columns/index types and other shape problems
    that would otherwise silently produce a misleading or empty result.
    """


def daily_coverage_and_mean(
    series: pd.Series, freq_minutes: int = DEFAULT_FREQ_MINUTES
) -> pd.DataFrame:
    """Summarizes a datetime-indexed count series to one row per calendar day.

    Reindexes `series` onto the *full* expected grid at `freq_minutes`
    spacing between its own min and max timestamp before aggregating, so a
    day with no rows at all in the source data (a real gap) is distinguished
    from a day with rows present but every value zero (a zero-run) - both
    would otherwise look identical after a plain ``groupby`` on the raw
    (already-gappy) index.

    Args:
        series: Count values indexed by a `datetime64` index (not
            necessarily sorted or gap-free); typically `total_count` for one
            station.
        freq_minutes: Expected spacing between consecutive intervals, in
            minutes.

    Returns:
        DataFrame indexed by calendar day (`datetime64`, midnight-aligned)
        with columns:

        - ``n_expected``: intervals expected that day at `freq_minutes`
          spacing (96 for a full day at the default 15 minutes).
        - ``n_present``: intervals with a non-null value.
        - ``mean``: mean of the non-null values (``NaN`` if `n_present` is
          0 - never silently treated as zero).

    Raises:
        RegimeShiftError: if `series` is empty or its index is not a
            `DatetimeIndex`.
    """
    if series.empty:
        raise RegimeShiftError("series is empty; nothing to summarize.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise RegimeShiftError("series must have a DatetimeIndex.")

    full_index = pd.date_range(
        series.index.min(), series.index.max(), freq=f"{freq_minutes}min"
    )
    reindexed = series.reindex(full_index)

    n_expected = pd.Series(1, index=full_index).resample("D").sum()
    n_present = reindexed.resample("D").apply(lambda x: int(x.notna().sum()))
    mean = reindexed.resample("D").mean()

    return pd.DataFrame(
        {"n_expected": n_expected, "n_present": n_present, "mean": mean}
    )


def find_constant_runs(
    daily: pd.DataFrame, column: str, value: float, min_run_days: int = 3
) -> pd.DataFrame:
    """Finds runs of consecutive days where `column` equals `value`.

    Used, e.g., with ``column="n_present", value=0`` to find data-gap runs,
    or ``column="mean", value=0.0`` to find zero-traffic runs (see module
    docstring).

    Args:
        daily: Per-day DataFrame as returned by `daily_coverage_and_mean`
            (or any DataFrame indexed by day).
        column: Column to test against `value`.
        value: Value that must hold for every day in a run.
        min_run_days: Minimum run length (in days) to include in the
            result; shorter runs are dropped as noise.

    Returns:
        DataFrame with one row per qualifying run, columns ``start``,
        ``end`` (both inclusive, `Timestamp`), and ``n_days``, sorted by
        `start`.

    Raises:
        RegimeShiftError: if `column` is not a column of `daily`.
    """
    if column not in daily.columns:
        raise RegimeShiftError(f"Column {column!r} not found in `daily`.")

    is_match = daily[column] == value
    runs: list[dict[str, object]] = []
    run_start: pd.Timestamp | None = None
    for day, matched in is_match.items():
        if matched and run_start is None:
            run_start = day
        elif not matched and run_start is not None:
            runs.append(_close_run(run_start, day, daily.index))
            run_start = None
    if run_start is not None:
        runs.append(_close_run(run_start, None, daily.index))

    result = pd.DataFrame(runs, columns=["start", "end", "n_days"])
    return result.loc[result["n_days"] >= min_run_days].reset_index(drop=True)


def _close_run(
    run_start: pd.Timestamp, next_day: pd.Timestamp | None, index: pd.DatetimeIndex
) -> dict[str, object]:
    """Builds one run record ending the day before `next_day` (or at the end)."""
    run_end = (next_day - pd.Timedelta(days=1)) if next_day is not None else index[-1]
    n_days = int((run_end - run_start).days) + 1
    return {"start": run_start, "end": run_end, "n_days": n_days}


def window_summary(
    series: pd.Series, start: str | pd.Timestamp, end: str | pd.Timestamp
) -> dict[str, object]:
    """Summarizes a count series over a fixed time window.

    Args:
        series: Count values indexed by a `datetime64` index.
        start: Window start (inclusive), as anything `pandas` can slice a
            `DatetimeIndex` with.
        end: Window end (inclusive), same rules as `start`.

    Returns:
        Dict with keys ``start``, ``end``, ``n_rows`` (rows present in the
        window, i.e. non-null - a gap contributes nothing here), ``mean``,
        ``median``, ``std``.

    Raises:
        RegimeShiftError: if `series` has no `DatetimeIndex`, or the window
            contains zero rows.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise RegimeShiftError("series must have a DatetimeIndex.")
    windowed = series.loc[start:end].dropna()
    if windowed.empty:
        raise RegimeShiftError(f"No rows found in window [{start}, {end}].")
    return {
        "start": start,
        "end": end,
        "n_rows": int(len(windowed)),
        "mean": float(windowed.mean()),
        "median": float(windowed.median()),
        "std": float(windowed.std()),
    }

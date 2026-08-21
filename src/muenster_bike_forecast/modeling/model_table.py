"""Assemble a modeling-ready feature table and score a baseline forecaster.

This module turns the per-station joined bike-count/weather CSVs (see
``muenster_bike_forecast.data.join``) into a single table suitable for
24h-ahead traffic forecasting, plus a seasonal-naive baseline to compare
future models against. Responsibilities, kept as pure functions:

1. **Channel coalescing + total-count summation.** The source repo has, for
   some stations, renamed a channel's description mid-history while keeping
   its numeric channel id, which shows up in the joined CSVs as two
   differently-named columns sharing the same leading channel id (e.g.
   ``"101020113 (FR stdteinwärts)"`` and
   ``"101020113 (FR stadteinwärts)"``  the first is a typo, fixed
   later). Naively summing every count-shaped column would double-count
   that channel. `compute_total_count` groups count columns by channel id,
   coalesces same-id columns into one series, and only then sums across
   distinct channel ids.
2. **Exact-timestamp 24h-ahead target construction.** `add_forecast_target`
   looks up, for each ``(station, t)`` row, the `total_count` value at the
   row whose timestamp is exactly ``t + horizon`` for the same station  a
   real timestamp lookup, not a positional shift, so real gaps in the
   15-minute data never silently pair a row with something that isn't
   actually `horizon` later.
3. **Calendar-feature merge** (`add_calendar_features`): public holidays,
   NRW school holidays, WWU lecture-period classification, and simple
   timestamp-derived features (`hour`, `day_of_week`, `month`).
4. **Chronological train/test split** (`chronological_split`): a single
   global cutoff applied uniformly across all stations.
5. **Seasonal-naive baseline** (`add_baseline_prediction`) and its
   evaluation (`compute_baseline_metrics`, plus the
   `summarize_target_nulls` / `summarize_baseline_evaluable_rows` data-
   quality reports).

All I/O (reading the joined CSVs, the school-holiday CSV, writing the
assembled table) is left to the caller (see
``notebooks/06_baseline_model.ipynb``); the functions here only transform
DataFrames already in memory.
"""

from __future__ import annotations

import re
from typing import Final, Sequence

import pandas as pd

from muenster_bike_forecast.data.semester_dates import (
    SEMESTER_PERIODS,
    SemesterPeriod,
    classify_dates,
)

# Matches a channel count column such as "101020113 (FR stadteinwärts)".
# Greedy `.*` correctly handles descriptions that themselves contain
# parentheses (e.g. "100031297 (Promenade (nördl. Salzstraße))") because it
# captures everything up to the *last* closing paren at the end of the
# column name.
_COUNT_COLUMN_RE: Final[re.Pattern[str]] = re.compile(r"^(\d+)\s*\(.*\)$")

DEFAULT_HORIZON: Final[pd.Timedelta] = pd.Timedelta(hours=24)
DEFAULT_TEST_PERIOD: Final[pd.Timedelta] = pd.Timedelta(weeks=8)


class ModelTableError(Exception):
    """Raised when the feature table cannot be assembled or evaluated.

    Covers missing required columns, empty input frames, channel columns
    that conflict (same channel id, same row, different non-null values),
    and other shape problems that would otherwise silently produce a wrong
    or empty result.
    """


def identify_channel_count_columns(columns: Sequence[str]) -> dict[str, list[str]]:
    """Groups count-column names by their leading numeric channel id.

    Args:
        columns: Column names to inspect (typically `DataFrame.columns`).
            Non-count columns (e.g. ``station_id``, ``datetime``, any
            ``"<id>-status"`` or ``weather_*`` column) are ignored.

    Returns:
        Mapping from channel id (as it appears in the column name, e.g.
        ``"101020113"``) to the list of matching column names, in input
        order. A channel id maps to more than one column exactly when the
        source repo has renamed that channel's description mid-history
        (see module docstring).
    """
    channel_columns: dict[str, list[str]] = {}
    for column in columns:
        match = _COUNT_COLUMN_RE.match(column)
        if match is None:
            continue
        channel_columns.setdefault(match.group(1), []).append(column)
    return channel_columns


def coalesce_channel_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Coalesces same-channel-id count columns into a single series.

    For each row, takes whichever of `columns` is non-null. If more than
    one of `columns` is non-null for the same row, this is only accepted
    when they agree; a genuine conflict (same row, different non-null
    values) is a real data question and is raised rather than silently
    resolved (e.g. by summing, which would double-count).

    Args:
        df: DataFrame containing `columns`.
        columns: One or more column names believed to describe the same
            physical channel (see `identify_channel_count_columns`).

    Returns:
        A single coalesced series, aligned to `df`'s index.

    Raises:
        ModelTableError: if `columns` is empty, any name is missing from
            `df`, or any row has more than one non-null value among
            `columns` that disagree with each other.
    """
    if not columns:
        raise ModelTableError("coalesce_channel_columns received no columns.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ModelTableError(f"Column(s) not found in DataFrame: {missing}.")

    coalesced = df[columns[0]].copy()
    for column in columns[1:]:
        other = df[column]
        conflict = coalesced.notna() & other.notna() & (coalesced != other)
        if conflict.any():
            raise ModelTableError(
                f"Columns {columns[0]!r} and {column!r} both have non-null, "
                f"disagreeing values in {int(conflict.sum())} row(s); cannot "
                "coalesce them into a single channel series without a "
                "human decision on which value is correct."
            )
        coalesced = coalesced.combine_first(other)
    return coalesced


def compute_total_count(df: pd.DataFrame) -> pd.Series:
    """Computes total bike count per row from a station's count columns.

    Identifies distinct channels via `identify_channel_count_columns`,
    coalesces any duplicate same-id columns (see module docstring), then
    sums across distinct channels with ``min_count=1`` so a row where every
    channel is null sums to ``NaN`` rather than a fabricated 0  missing
    data must never silently become "zero traffic".

    Args:
        df: One station's rows, containing one or more
            ``"<channel_id> (<description>)"`` count columns (e.g. as
            loaded from ``data/raw/joined/<station_id>.csv``).

    Returns:
        Series of total counts, aligned to `df`'s index, dtype float
        (nullable via `NaN`).

    Raises:
        ModelTableError: if `df` has no recognizable count columns, or a
            duplicate-channel-id conflict is found (see
            `coalesce_channel_columns`).
    """
    channel_columns = identify_channel_count_columns(list(df.columns))
    if not channel_columns:
        raise ModelTableError("DataFrame has no recognizable channel count columns.")

    coalesced = pd.DataFrame(
        {
            channel_id: coalesce_channel_columns(df, columns)
            for channel_id, columns in channel_columns.items()
        },
        index=df.index,
    )
    return coalesced.sum(axis=1, min_count=1)


def combined_channel_matches_directional_sum(
    df: pd.DataFrame, station_id: int | str
) -> pd.Series:
    """Checks, per row, whether the station's combined channel count equals
    the sum of its other (directional) channel counts for that row.

    Every station in this dataset publishes one "combined" count column
    whose numeric channel id equals the station's own `station_id`,
    alongside two or more "directional" sub-channels (e.g. `stadteinwärts`/
    `stadtauswärts`, sometimes reissued under new channel ids over the
    station's history). Read-only diagnostic: it does not change how
    `compute_total_count` behaves, and is not wired into the modeling
    pipeline. It exists to confirm (or refute), per station, the
    suspicion that the combined channel is not an independent sensor
    reading but already equals the sum of whichever directional channels
    are active for that row - which would mean `compute_total_count`
    currently double-counts every such station's traffic by summing the
    combined channel on top of the directional ones instead of treating
    it as redundant.

    Args:
        df: One station's rows, containing channel count columns as
            recognized by `identify_channel_count_columns`.
        station_id: The station's own id - identifies which channel
            column is "combined" (the one whose numeric id matches this
            value, compared as a string).

    Returns:
        Nullable boolean Series aligned to `df`'s index: `True` where the
        combined channel's value equals the sum of every other channel's
        value for that row, `False` where they disagree, and `pd.NA`
        where there is nothing to compare (the combined channel is null
        for that row, or *every* directional channel is null for that
        row - distinguishing "checked and disagreed" from "nothing to
        check" matters here, since a naive comparison against a
        fabricated `0` for missing directional data would misreport a
        data gap as a mismatch).

        Known accepted limitation: a *partial* directional gap (some, not
        all, directional channels null on a row) is not detected as
        "nothing to check" - the sum is computed from whichever channels
        are present and compared as if complete, which can misreport
        `False` on a row that's actually just missing one reading. This
        can't cheaply be distinguished from the normal, expected case of
        several *inactive* directional-channel generations being null on
        every row outside their own vintage (see the module docstring on
        reissued channel ids) - requiring every directional column
        non-null would make this function reject nearly every row of a
        multi-generation station. Accepted because the actual 23-station
        confirmation run this function was built for came back
        unambiguous (100% match, zero `False`s) despite this gap.

    Raises:
        ModelTableError: if `df` has no channel column whose id matches
            `station_id` (nothing to treat as "combined").
    """
    channel_columns = identify_channel_count_columns(list(df.columns))
    combined_id = str(station_id)
    if combined_id not in channel_columns:
        raise ModelTableError(
            f"No channel column found with id matching station_id {station_id!r}."
        )
    combined = coalesce_channel_columns(df, channel_columns[combined_id])

    directional_ids = [cid for cid in channel_columns if cid != combined_id]
    if not directional_ids:
        raise ModelTableError(
            f"Station {station_id!r} has only the combined channel "
            f"{combined_id!r} - no directional channels to compare it against."
        )
    directional = pd.DataFrame(
        {
            channel_id: coalesce_channel_columns(df, channel_columns[channel_id])
            for channel_id in directional_ids
        },
        index=df.index,
    )
    directional_sum = directional.sum(axis=1, min_count=1)

    has_data = combined.notna() & directional_sum.notna()
    result = pd.Series(pd.NA, index=df.index, dtype="boolean")
    result[has_data] = combined[has_data] == directional_sum[has_data]
    return result


def add_forecast_target(
    df: pd.DataFrame,
    value_col: str = "total_count",
    timestamp_col: str = "datetime",
    station_col: str = "station_id",
    horizon: pd.Timedelta = DEFAULT_HORIZON,
    target_col: str = "target_total_count",
) -> pd.DataFrame:
    """Adds a 24h-ahead forecast target via exact-timestamp lookup.

    For each row at time ``t``, the target is `value_col`'s value at the
    row whose `timestamp_col` is exactly ``t + horizon`` for the *same*
    station  found via an exact timestamp match, not a positional shift.
    Positional shifting would be wrong here: the data has real gaps
    (missing 15-minute intervals), so shifting by a fixed number of rows
    across a gap would silently pair a row with something that isn't
    actually `horizon` later. Rows with no data at exactly ``t + horizon``
    get a null target.

    Args:
        df: Rows for one or more stations, with `station_col`,
            `timestamp_col`, and `value_col` columns.
        value_col: Column whose future value becomes the target.
        timestamp_col: Column with (per-station) unique timestamps.
        station_col: Column identifying the station.
        horizon: How far ahead to look up the target value.
        target_col: Name of the new target column to add.

    Returns:
        Copy of `df` with `target_col` added.

    Raises:
        ModelTableError: if any of `station_col`, `timestamp_col`,
            `value_col` is missing from `df`, or `df` has duplicate
            ``(station_col, timestamp_col)`` pairs (the lookup requires a
            unique key per station/timestamp).
    """
    required = {station_col, timestamp_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise ModelTableError(f"DataFrame is missing column(s): {sorted(missing)}.")

    if df.empty:
        # `pd.MultiIndex.from_tuples([])` below raises `TypeError: Cannot
        # infer number of levels from empty list` on an empty key list -
        # short-circuit with a valid empty result instead.
        out = df.copy()
        out[target_col] = pd.Series(dtype="float64")
        return out

    key_pairs = list(zip(df[station_col], df[timestamp_col]))
    lookup = pd.Series(
        df[value_col].to_numpy(), index=pd.MultiIndex.from_tuples(key_pairs)
    )
    if lookup.index.duplicated().any():
        n_dupes = int(lookup.index.duplicated().sum())
        raise ModelTableError(
            f"DataFrame has {n_dupes} duplicate (station, timestamp) pair(s); "
            "the exact-timestamp target lookup requires a unique key."
        )

    out = df.copy()
    lookup_keys = pd.MultiIndex.from_arrays(
        [out[station_col], out[timestamp_col] + horizon]
    )
    out[target_col] = lookup.reindex(lookup_keys).to_numpy()
    return out


def summarize_target_nulls(
    df: pd.DataFrame, target_col: str = "target_total_count"
) -> dict[str, object]:
    """Reports what fraction of rows have no 24h-ahead target.

    Args:
        df: DataFrame as returned by `add_forecast_target`.
        target_col: Name of the target column.

    Returns:
        Dict with keys ``n_rows``, ``n_null_target``, ``pct_null_target``.

    Raises:
        ModelTableError: if `target_col` is not a column of `df`.
    """
    if target_col not in df.columns:
        raise ModelTableError(f"Column {target_col!r} not found in DataFrame.")
    n_rows = len(df)
    n_null = int(df[target_col].isna().sum())
    return {
        "n_rows": n_rows,
        "n_null_target": n_null,
        "pct_null_target": 100 * n_null / n_rows if n_rows else float("nan"),
    }


def add_calendar_features(
    df: pd.DataFrame,
    public_holidays: pd.DataFrame,
    school_holidays: pd.DataFrame,
    timestamp_col: str = "datetime",
    semester_periods: Sequence[SemesterPeriod] = SEMESTER_PERIODS,
) -> pd.DataFrame:
    """Adds calendar and time-of-day features, computed once per unique date.

    Adds:

    - `hour`, `day_of_week` (Monday=0), `month`: derived directly from
      `timestamp_col`.
    - `is_public_holiday`: whether the row's date is a German/NRW public
      holiday (from ``data.calendar.public_holidays``).
    - `is_school_holiday`: whether the row's date falls within any
      ``[start_date, end_date]`` period in `school_holidays` (from
      ``data.calendar.load_school_holidays``).
    - `is_lecture_period`: whether the row's date falls within a WWU
      lecture period, per ``data.semester_dates.classify_dates``.

    Args:
        df: Rows with a `timestamp_col` column.
        public_holidays: As returned by
            ``muenster_bike_forecast.data.calendar.public_holidays`` (needs
            a ``date`` column).
        school_holidays: As returned by
            ``muenster_bike_forecast.data.calendar.load_school_holidays``
            (needs ``start_date``/``end_date`` columns).
        timestamp_col: Name of the timestamp column to derive dates from.
        semester_periods: Semester table passed through to
            ``classify_dates``.

    Returns:
        Copy of `df` with the columns above added.

    Raises:
        ModelTableError: if `timestamp_col` is missing from `df`, or
            `public_holidays`/`school_holidays` is missing its required
            column(s).
        SemesterDateRangeError: propagated from `classify_dates` if any
            row's date falls outside the range covered by
            `semester_periods`.
    """
    if timestamp_col not in df.columns:
        raise ModelTableError(f"Column {timestamp_col!r} not found in DataFrame.")
    if "date" not in public_holidays.columns:
        raise ModelTableError("public_holidays is missing a 'date' column.")
    missing_school_cols = {"start_date", "end_date"} - set(school_holidays.columns)
    if missing_school_cols:
        raise ModelTableError(
            f"school_holidays is missing column(s): {sorted(missing_school_cols)}."
        )

    out = df.copy()
    dt = pd.to_datetime(out[timestamp_col])
    out["hour"] = dt.dt.hour
    out["day_of_week"] = dt.dt.dayofweek
    out["month"] = dt.dt.month

    dates = dt.dt.normalize()

    holiday_dates = set(pd.to_datetime(public_holidays["date"]).dt.normalize())
    out["is_public_holiday"] = dates.isin(holiday_dates)

    is_school_holiday = pd.Series(False, index=out.index)
    starts = pd.to_datetime(school_holidays["start_date"]).dt.normalize()
    ends = pd.to_datetime(school_holidays["end_date"]).dt.normalize()
    for start, end in zip(starts, ends):
        is_school_holiday |= (dates >= start) & (dates <= end)
    out["is_school_holiday"] = is_school_holiday

    unique_dates = pd.DatetimeIndex(dates.unique())
    classification = classify_dates(unique_dates, periods=semester_periods)
    lecture_map = dict(zip(classification["date"], classification["is_lecture_period"]))
    out["is_lecture_period"] = dates.map(lecture_map)

    return out


def chronological_split(
    df: pd.DataFrame,
    timestamp_col: str = "datetime",
    test_period: pd.Timedelta = DEFAULT_TEST_PERIOD,
    embargo: pd.Timedelta = DEFAULT_HORIZON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Splits rows into train/test by a single global chronological cutoff.

    The cutoff is ``max(timestamp_col) - test_period`` computed over the
    *whole* input (all stations together), then applied uniformly: every
    row before ``cutoff - embargo`` is train, every row at/after the
    cutoff is test. This respects time order (no future leaking into the
    past) and station boundaries (the same cutoff date is used for every
    station, rather than a per-station cutoff which would let one
    station's "future" leak relative to another's).

    `embargo` exists because time order alone isn't enough here: a row's
    *label* (as added by `add_forecast_target`) is observed `horizon`
    after its own timestamp, not at it. Without an embargo, a train row
    timestamped just before `cutoff` would carry a label timestamped at
    or after `cutoff` - i.e. inside the nominal test window - so the
    model would train on information from the period it's meant to be
    evaluated against. The `[cutoff - embargo, cutoff)` window is dropped
    from train entirely (it's in neither split) to close this gap;
    defaults to `DEFAULT_HORIZON`, matching `add_forecast_target`'s
    default horizon.

    Args:
        df: Rows for one or more stations, with a `timestamp_col` column.
        timestamp_col: Name of the timestamp column.
        test_period: Length of the held-out test window at the end of the
            data.
        embargo: Width of the excluded window immediately before the
            cutoff. Pass `pd.Timedelta(0)` to disable it and reproduce
            the pre-embargo behavior.

    Returns:
        ``(train_df, test_df, cutoff)`` where `cutoff` is the timestamp
        used as the train/test boundary (test rows have
        ``timestamp_col >= cutoff``; train rows have
        ``timestamp_col < cutoff - embargo``).

    Raises:
        ModelTableError: if `timestamp_col` is missing from `df`, or `df`
            is empty.
    """
    if timestamp_col not in df.columns:
        raise ModelTableError(f"Column {timestamp_col!r} not found in DataFrame.")
    if df.empty:
        raise ModelTableError("DataFrame is empty; cannot determine a split cutoff.")

    cutoff = df[timestamp_col].max() - test_period
    train = df.loc[df[timestamp_col] < cutoff - embargo].copy()
    test = df.loc[df[timestamp_col] >= cutoff].copy()
    return train, test, cutoff


def add_baseline_prediction(
    df: pd.DataFrame,
    current_col: str = "total_count",
    prediction_col: str = "baseline_prediction",
) -> pd.DataFrame:
    """Adds a seasonal-naive ("same time next day as today") prediction.

    The baseline needs no fitting: the predicted value 24h ahead is simply
    the row's own current value. This is a well-established baseline for
    daily-seasonal time series.

    Args:
        df: Rows with a `current_col` column.
        current_col: Column holding the current (t) value.
        prediction_col: Name of the new prediction column to add.

    Returns:
        Copy of `df` with `prediction_col` added (identical to
        `current_col`).

    Raises:
        ModelTableError: if `current_col` is not a column of `df`.
    """
    if current_col not in df.columns:
        raise ModelTableError(f"Column {current_col!r} not found in DataFrame.")
    out = df.copy()
    out[prediction_col] = out[current_col]
    return out


def summarize_baseline_evaluable_rows(
    df: pd.DataFrame,
    prediction_col: str = "baseline_prediction",
    target_col: str = "target_total_count",
) -> dict[str, object]:
    """Reports how many rows can be scored (both prediction and target present).

    Args:
        df: DataFrame with `prediction_col` and `target_col` columns.
        prediction_col: Name of the prediction column.
        target_col: Name of the true-target column.

    Returns:
        Dict with keys ``n_rows``, ``n_evaluable``, ``n_excluded``,
        ``pct_excluded``.

    Raises:
        ModelTableError: if either column is missing from `df`.
    """
    missing = {prediction_col, target_col} - set(df.columns)
    if missing:
        raise ModelTableError(f"DataFrame is missing column(s): {sorted(missing)}.")
    n_rows = len(df)
    evaluable = df[prediction_col].notna() & df[target_col].notna()
    n_evaluable = int(evaluable.sum())
    n_excluded = n_rows - n_evaluable
    return {
        "n_rows": n_rows,
        "n_evaluable": n_evaluable,
        "n_excluded": n_excluded,
        "pct_excluded": 100 * n_excluded / n_rows if n_rows else float("nan"),
    }


def compute_baseline_metrics(
    df: pd.DataFrame,
    prediction_col: str = "baseline_prediction",
    target_col: str = "target_total_count",
    group_col: str | None = None,
) -> pd.DataFrame:
    """Computes MAE and RMSE for a prediction column against the target.

    Only rows where both `prediction_col` and `target_col` are non-null are
    scored (see `summarize_baseline_evaluable_rows` for how many rows that
    excludes).

    Args:
        df: DataFrame with `prediction_col`, `target_col`, and (if given)
            `group_col` columns  typically a test-set slice.
        prediction_col: Name of the prediction column.
        target_col: Name of the true-target column.
        group_col: If given, compute metrics separately per distinct value
            of this column (e.g. ``"station_id"``) in addition to one
            overall row. If `None`, only the overall row is returned.

    Returns:
        DataFrame with columns ``group`` (``"overall"``, or the
        `group_col` value), ``mae``, ``rmse``, ``n_rows`` (rows actually
        scored for that group).

    Raises:
        ModelTableError: if `prediction_col`/`target_col` (or `group_col`,
            if given) is missing from `df`, or there are zero evaluable
            rows.
    """
    required = {prediction_col, target_col} | ({group_col} if group_col else set())
    missing = required - set(df.columns)
    if missing:
        raise ModelTableError(f"DataFrame is missing column(s): {sorted(missing)}.")

    evaluable = df.loc[df[prediction_col].notna() & df[target_col].notna()]
    if evaluable.empty:
        raise ModelTableError(
            "No rows with both prediction and target present; cannot compute "
            "metrics."
        )

    def _metrics(frame: pd.DataFrame) -> pd.Series:
        error = frame[prediction_col] - frame[target_col]
        return pd.Series(
            {
                "mae": error.abs().mean(),
                "rmse": (error**2).mean() ** 0.5,
                "n_rows": len(frame),
            }
        )

    if group_col is None:
        overall = _metrics(evaluable)
        result = pd.DataFrame([{"group": "overall", **overall.to_dict()}])
    else:
        grouped = evaluable.groupby(group_col, sort=True).apply(
            _metrics, include_groups=False
        )
        result = grouped.reset_index().rename(columns={group_col: "group"})

    result["n_rows"] = result["n_rows"].astype(int)
    return result[["group", "mae", "rmse", "n_rows"]]

"""Adds lag and rolling-window history features to the model table.

Two kinds of "what happened before now" features, both computed strictly
from data at or before each row's own timestamp (never the future, so
these are safe to use alongside `target_total_count` without leakage):

1. **Lag features** (`add_lag_feature`): the value at an exact earlier
   timestamp (e.g. "1 hour ago", "same time yesterday"), found via an
   exact-timestamp lookup rather than a positional shift — the same
   reasoning as `modeling.model_table.add_forecast_target`: the data has
   real gaps (missing 15-minute intervals), so shifting by a fixed number
   of rows across a gap would silently pair a row with something that
   isn't actually `lag` earlier.
2. **Rolling features** (`add_rolling_feature`): a time-windowed
   aggregate (e.g. mean count over the past 2 hours) computed with a
   time-based (not row-count-based) window so it is robust to gaps, using
   `closed="left"` so the current row's own value is never included in
   its own rolling statistic.

All I/O is left to the caller; the functions here only transform
DataFrames already in memory.
"""

from __future__ import annotations

import pandas as pd


class LagFeatureError(Exception):
    """Raised when a lag/rolling feature cannot be computed as expected.

    Covers missing required columns, duplicate ``(station, timestamp)``
    pairs, and unsupported rolling statistics.
    """


def add_lag_feature(
    df: pd.DataFrame,
    lag: pd.Timedelta,
    feature_col: str,
    value_col: str = "total_count",
    timestamp_col: str = "datetime",
    station_col: str = "station_id",
) -> pd.DataFrame:
    """Adds `value_col`'s value from exactly `lag` earlier, per station.

    For each row at time ``t``, the new feature is `value_col`'s value at
    the row whose `timestamp_col` is exactly ``t - lag`` for the *same*
    station. Rows with no data at exactly ``t - lag`` (including near the
    start of a station's coverage) get a null feature value, which is
    fine: `HistGradientBoostingRegressor` and similar tree models handle
    missing feature values natively.

    Args:
        df: Rows for one or more stations, with `station_col`,
            `timestamp_col`, and `value_col` columns.
        lag: How far back to look up the feature value.
        feature_col: Name of the new feature column to add.
        value_col: Column whose past value becomes the feature.
        timestamp_col: Column with (per-station) unique timestamps.
        station_col: Column identifying the station.

    Returns:
        Copy of `df` with `feature_col` added.

    Raises:
        LagFeatureError: if any of `station_col`, `timestamp_col`,
            `value_col` is missing from `df`, or `df` has duplicate
            ``(station_col, timestamp_col)`` pairs.
    """
    required = {station_col, timestamp_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise LagFeatureError(f"DataFrame is missing column(s): {sorted(missing)}.")

    if df.empty:
        # `pd.MultiIndex.from_tuples([])` below raises `TypeError: Cannot
        # infer number of levels from empty list` on an empty key list -
        # short-circuit with the same "valid empty result" behavior
        # `add_rolling_feature` already gets for free from its groupby.
        out = df.copy()
        out[feature_col] = pd.Series(dtype="float64")
        return out

    key_pairs = list(zip(df[station_col], df[timestamp_col]))
    lookup = pd.Series(
        df[value_col].to_numpy(), index=pd.MultiIndex.from_tuples(key_pairs)
    )
    if lookup.index.duplicated().any():
        n_dupes = int(lookup.index.duplicated().sum())
        raise LagFeatureError(
            f"DataFrame has {n_dupes} duplicate (station, timestamp) pair(s); "
            "the exact-timestamp lag lookup requires a unique key."
        )

    out = df.copy()
    lookup_keys = pd.MultiIndex.from_arrays(
        [out[station_col], out[timestamp_col] - lag]
    )
    out[feature_col] = lookup.reindex(lookup_keys).to_numpy()
    return out


def add_rolling_feature(
    df: pd.DataFrame,
    window: pd.Timedelta,
    feature_col: str,
    stat: str = "mean",
    value_col: str = "total_count",
    timestamp_col: str = "datetime",
    station_col: str = "station_id",
    min_periods: int = 1,
) -> pd.DataFrame:
    """Adds a per-station, time-windowed rolling statistic of `value_col`.

    Uses a time-based window (not a row-count window), so it stays
    correct across the data's real gaps (missing 15-minute intervals),
    and ``closed="left"`` so the window covers ``(t - window, t)`` and
    never includes the current row's own value.

    Args:
        df: Rows for one or more stations, with `station_col`,
            `timestamp_col`, and `value_col` columns.
        window: Length of the trailing time window.
        feature_col: Name of the new feature column to add.
        stat: Aggregate to compute over the window; one of ``"mean"`` or
            ``"std"``.
        value_col: Column to aggregate.
        timestamp_col: Column with (per-station) unique timestamps.
        station_col: Column identifying the station.
        min_periods: Minimum number of observations in the window
            required to produce a non-null value (passed through to
            `pandas.DataFrame.rolling`).

    Returns:
        Copy of `df` with `feature_col` added.

    Raises:
        LagFeatureError: if any of `station_col`, `timestamp_col`,
            `value_col` is missing from `df`, or `stat` is not one of
            ``"mean"``/``"std"``.
    """
    required = {station_col, timestamp_col, value_col}
    missing = required - set(df.columns)
    if missing:
        raise LagFeatureError(f"DataFrame is missing column(s): {sorted(missing)}.")
    if stat not in {"mean", "std"}:
        raise LagFeatureError(f"Unsupported stat {stat!r}; use 'mean' or 'std'.")

    out = df.copy()
    result = pd.Series(index=out.index, dtype="float64")
    for _station, group in out.groupby(station_col, sort=False):
        ordered = group.sort_values(timestamp_col)
        indexed = ordered.set_index(timestamp_col)[value_col]
        rolling = indexed.rolling(window, closed="left", min_periods=min_periods)
        rolled = getattr(rolling, stat)()
        result.loc[ordered.index] = rolled.to_numpy()
    out[feature_col] = result
    return out

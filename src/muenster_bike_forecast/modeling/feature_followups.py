"""Three follow-up features identified by `07_descriptive_analysis.ipynb`.

Notebooks 08-11/14 deliberately share one identical base feature set (no
distance feature, no bucketed precipitation, no explicit hour x
day-of-week interaction, no weekend/weekday ratio) so their model-*class*
comparisons stay apples-to-apples. `13_distance_feature_test.ipynb`
already tested the distance-from-center idea in isolation (result: exactly
redundant with `station_id`, zero effect). This module does the same for
notebook 07's three remaining deferred findings, each as a pure,
independently-testable transform:

1. **Bucketed precipitation** (`add_precipitation_bucket`): notebook 07
   found raw `weather_precipitation_mm` has a near-zero linear correlation
   with traffic (r ~= -0.03), but a real *threshold* effect once bucketed
   (near-zero until it actually rains, then a step down). Reuses
   `analysis.descriptive.DEFAULT_PRECIPITATION_BINS`/`_LABELS` exactly
   (not re-derived), so this feature and that descriptive finding define
   "bucket" identically.
2. **Explicit hour x day-of-week interaction** (`add_hour_dow_interaction`):
   a single combined categorical for the (hour, day_of_week) pair,
   testing whether an explicit interaction column measurably helps a tree
   model over the two separate columns it can already split on
   independently (or is redundant, the way distance-from-center turned
   out to be for `station_id`).
3. **Per-station weekend/weekday ratio** (`compute_weekend_weekday_ratio`
   + `add_weekend_weekday_ratio_feature`): a static per-station
   `mean(weekend total_count) / mean(weekday total_count)`, split into two
   steps specifically so the caller can (and must) compute the ratio from
   training-period rows only, then broadcast that static value onto both
   train and test rows -- computing it from the full dataset (including
   the test period) would leak future average-traffic information into a
   feature applied to training rows too.

All I/O is left to the caller; the functions here only transform
DataFrames already in memory.
"""

from __future__ import annotations

from typing import Final, Sequence

import pandas as pd

from muenster_bike_forecast.analysis.descriptive import (
    DEFAULT_PRECIPITATION_BINS,
    DEFAULT_PRECIPITATION_LABELS,
    bucket_numeric_column,
)

# Saturday, Sunday (day_of_week is Monday=0, per
# `modeling.model_table.add_calendar_features`).
DEFAULT_WEEKEND_DAYS: Final[tuple[int, ...]] = (5, 6)


class FeatureFollowupError(Exception):
    """Raised when a follow-up feature cannot be computed as requested.

    Covers missing required columns, empty input frames, and a
    weekend/weekday ratio that cannot be computed or broadcast cleanly
    (a station with zero weekday or weekend rows, or a row whose station
    has no matching ratio).
    """


def add_precipitation_bucket(
    df: pd.DataFrame,
    precip_col: str = "weather_precipitation_mm",
    bins: Sequence[float] | None = None,
    labels: Sequence[str] | None = None,
    feature_col: str = "precipitation_bucket",
) -> pd.DataFrame:
    """Adds a bucketed-precipitation categorical feature.

    Reuses `analysis.descriptive.bucket_numeric_column` with the exact
    same bin edges notebook 07 used
    (`analysis.descriptive.DEFAULT_PRECIPITATION_BINS`/`_LABELS` by
    default), so this feature's "bucket" definition matches that
    descriptive finding exactly rather than inventing new boundaries.

    Args:
        df: Rows with a `precip_col` column.
        precip_col: Name of the raw precipitation (mm) column.
        bins: Bucket edges; defaults to
            `analysis.descriptive.DEFAULT_PRECIPITATION_BINS` if `None`.
        labels: One label per bucket; defaults to
            `analysis.descriptive.DEFAULT_PRECIPITATION_LABELS` if `None`.
        feature_col: Name of the new categorical feature column to add.

    Returns:
        Copy of `df` with `feature_col` added (a pandas `Categorical`;
        `NaN` preserved for null `precip_col` values).

    Raises:
        FeatureFollowupError: if `precip_col` is missing from `df`.
    """
    if bins is None:
        bins = DEFAULT_PRECIPITATION_BINS
    if labels is None:
        labels = DEFAULT_PRECIPITATION_LABELS

    if precip_col not in df.columns:
        raise FeatureFollowupError(f"Column {precip_col!r} not found in DataFrame.")

    out = df.copy()
    out[feature_col] = bucket_numeric_column(out[precip_col], bins=bins, labels=labels)
    return out


def add_hour_dow_interaction(
    df: pd.DataFrame,
    hour_col: str = "hour",
    dow_col: str = "day_of_week",
    feature_col: str = "hour_dow",
) -> pd.DataFrame:
    """Adds a combined (hour, day_of_week) categorical feature.

    Tree models can already split on `hour_col` and `dow_col` separately
    across multiple splits/boosting rounds to approximate an interaction;
    this makes the pairing explicit as a single category (e.g. ``"7_0"``
    for 07:00 on a Monday) so the open empirical question -- whether an
    explicit combined category measurably helps over the two separate
    columns, or is redundant -- can actually be tested rather than
    assumed either way.

    Args:
        df: Rows with `hour_col` and `dow_col` columns.
        hour_col: Name of the hour-of-day column (0-23).
        dow_col: Name of the day-of-week column (0-6, Monday=0).
        feature_col: Name of the new combined categorical column to add.

    Returns:
        Copy of `df` with `feature_col` added, as a string category of the
        form ``f"{hour}_{day_of_week}"`` (up to 24 x 7 = 168 distinct
        values).

    Raises:
        FeatureFollowupError: if `hour_col` or `dow_col` is missing from
            `df`.
    """
    required = {hour_col, dow_col}
    missing = required - set(df.columns)
    if missing:
        raise FeatureFollowupError(
            f"DataFrame is missing column(s): {sorted(missing)}."
        )

    out = df.copy()
    out[feature_col] = (
        out[hour_col].astype("Int64").astype(str)
        + "_"
        + out[dow_col].astype("Int64").astype(str)
    )
    return out


def compute_weekend_weekday_ratio(
    df: pd.DataFrame,
    station_col: str = "station_id",
    value_col: str = "total_count",
    dow_col: str = "day_of_week",
    weekend_days: Sequence[int] = DEFAULT_WEEKEND_DAYS,
) -> pd.DataFrame:
    """Computes each station's mean(weekend `value_col`) / mean(weekday `value_col`).

    **Leakage warning**: this is a per-station aggregate computed from
    `value_col`'s own history. Pass only **training-period** rows (rows
    before whatever cutoff `modeling.model_table.chronological_split`
    produces) -- computing it over the full dataset, including the test
    period, would leak future average-traffic information into a feature
    applied to training rows too. Broadcast the resulting static
    per-station value onto both train and test rows with
    `add_weekend_weekday_ratio_feature`.

    Args:
        df: Rows (training-period only -- see warning above) with
            `station_col`, `value_col`, `dow_col` columns.
        station_col: Column identifying the station.
        value_col: Column to average (typically ``total_count``).
        dow_col: Day-of-week column (0-6, Monday=0) used to classify each
            row as weekend or weekday.
        weekend_days: Which `dow_col` values count as "weekend". Defaults
            to `DEFAULT_WEEKEND_DAYS` (Saturday, Sunday).

    Returns:
        DataFrame with columns `station_col`, ``weekend_weekday_ratio``,
        one row per distinct station in `df`.

    Raises:
        FeatureFollowupError: if a required column is missing, `df` has
            zero non-null `value_col` rows, or any station has zero
            weekday or zero weekend rows (a ratio would be undefined or
            infinite).
    """
    required = {station_col, value_col, dow_col}
    missing = required - set(df.columns)
    if missing:
        raise FeatureFollowupError(
            f"DataFrame is missing column(s): {sorted(missing)}."
        )

    valid = df.dropna(subset=[value_col])
    if valid.empty:
        raise FeatureFollowupError(
            f"No rows with a non-null {value_col!r}; cannot compute a ratio."
        )

    is_weekend = valid[dow_col].isin(weekend_days)
    means = (
        valid.assign(_is_weekend=is_weekend)
        .groupby([station_col, "_is_weekend"], sort=True)[value_col]
        .mean()
        .unstack("_is_weekend")
    )
    if True not in means.columns or False not in means.columns:
        raise FeatureFollowupError(
            "Every station must have at least one weekday and one weekend "
            "row with a non-null value; got no rows of one kind at all "
            "across the whole input."
        )

    incomplete = means[True].isna() | means[False].isna()
    if incomplete.any():
        bad_stations = means.index[incomplete].tolist()
        raise FeatureFollowupError(
            f"Station(s) {bad_stations} have zero weekday or zero weekend "
            f"rows with a non-null {value_col!r}; cannot compute a ratio "
            "for them."
        )

    ratio = (means[True] / means[False]).rename("weekend_weekday_ratio")
    return ratio.reset_index().rename(columns={station_col: station_col})


def add_weekend_weekday_ratio_feature(
    df: pd.DataFrame,
    ratio_table: pd.DataFrame,
    station_col: str = "station_id",
    feature_col: str = "weekend_weekday_ratio",
) -> pd.DataFrame:
    """Broadcasts a per-station ratio table onto every row of `df`.

    `ratio_table` (as returned by `compute_weekend_weekday_ratio`) holds
    one static value per station; this merges it onto every row of `df`
    by `station_col`, so `df` may be train rows, test rows, or both --
    the leakage guard lives in *how* `ratio_table` itself was computed
    (training-period rows only), not in which rows this function is
    applied to.

    Args:
        df: Rows to add the feature to, with a `station_col` column.
        ratio_table: As returned by `compute_weekend_weekday_ratio`, with
            `station_col` and `feature_col` columns.
        station_col: Column identifying the station, shared by both
            frames.
        feature_col: Name of the ratio column in `ratio_table`, and of the
            new column added to `df`.

    Returns:
        Copy of `df` with `feature_col` added.

    Raises:
        FeatureFollowupError: if `station_col` is missing from `df`,
            `station_col`/`feature_col` is missing from `ratio_table`, or
            any row of `df` has a station absent from `ratio_table` (e.g.
            a station with zero training-period rows).
    """
    if station_col not in df.columns:
        raise FeatureFollowupError(f"Column {station_col!r} not found in df.")
    missing_ratio_cols = {station_col, feature_col} - set(ratio_table.columns)
    if missing_ratio_cols:
        raise FeatureFollowupError(
            f"ratio_table is missing column(s): {sorted(missing_ratio_cols)}."
        )

    out = df.merge(
        ratio_table[[station_col, feature_col]],
        on=station_col,
        how="left",
        validate="many_to_one",
    )
    n_missing = int(out[feature_col].isna().sum())
    if n_missing:
        missing_stations = sorted(
            out.loc[out[feature_col].isna(), station_col].unique().tolist()
        )
        raise FeatureFollowupError(
            f"{n_missing} row(s) had no matching station in ratio_table "
            f"(station(s) {missing_stations}); every station in df must "
            "have a ratio (typically computed from training-period rows "
            "covering all stations)."
        )
    return out

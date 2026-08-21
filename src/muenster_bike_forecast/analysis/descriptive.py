"""Descriptive statistics for the assembled model table.

This module supports exploratory data analysis (see
``notebooks/07_descriptive_analysis.ipynb``) over
``data/raw/model_table/model_table.csv`` (one row per
``(station_id, datetime)`` at 15-minute resolution, see
``muenster_bike_forecast.modeling.model_table``). It is deliberately *not*
modeling: no train/test split, no forecasting, only pure summary functions
that answer "what does the data look like":

1. **Station ranking** (`rank_stations`, `rank_stations_by_group`): which
   stations see the most/least traffic, overall or split by some other
   column (e.g. weekday vs. weekend), so a ranking-change can be read off
   directly rather than eyeballed from charts.
2. **Time-of-day / day-of-week / month patterns** (`average_by_time_feature`,
   `average_by_time_feature_per_station`): mean traffic grouped by a
   timestamp-derived feature, optionally broken out per station to compare
   commuter-pattern vs. leisure-pattern stations.
3. **Weather-bucketed averages** (`bucket_numeric_column`,
   `average_by_weather_bucket`) and **linear correlation**
   (`weather_correlations`): how traffic differs across temperature/
   precipitation/wind ranges, in both a concrete "% different from the
   overall mean" form and a single correlation coefficient.
4. **Boolean calendar-flag comparison** (`compare_boolean_flag`): mean
   traffic when a flag (`is_public_holiday`, `is_school_holiday`,
   `is_lecture_period`) is true vs. false.
5. **Directional (inbound/outbound) imbalance** (`classify_channel_direction`,
   `compute_directional_totals`): unlike 1-4, these work over a station's
   *raw channel* data (e.g. ``data/raw/joined/<station_id>.csv``), not
   ``model_table.csv`` - they classify each non-combined channel's
   description as inbound/outbound and sum by direction, to compare a
   station's inbound vs. outbound traffic (not something `total_count`
   alone can answer, since it collapses both directions into one number -
   see `muenster_bike_forecast.modeling.model_table.compute_total_count`).

Every function skips rows with a null `total_count` (or other relevant
column) rather than treating a missing 15-minute interval as zero traffic -
silently coercing a gap to 0 would understate real averages. All functions
are pure: they accept DataFrames/Series already in memory and return new
DataFrames/Series/dicts; no file I/O happens here (left to the notebook, see
``muenster_bike_forecast.modeling.model_table`` for the same convention).
"""

from __future__ import annotations

import re
from typing import Final, Sequence

import numpy as np
import pandas as pd

from muenster_bike_forecast.modeling.model_table import (
    coalesce_channel_columns,
    identify_channel_count_columns,
)

# Default weather-column bucket edges/labels, chosen to give roughly
# concrete, human-readable ranges rather than statistically-derived
# quantiles (quantile bins would shift meaning as the underlying data
# grows, e.g. after re-fetching more months of weather).
DEFAULT_TEMPERATURE_BINS: Final[list[float]] = [-np.inf, 0, 10, 20, 30, np.inf]
DEFAULT_TEMPERATURE_LABELS: Final[list[str]] = [
    "< 0°C",
    "0-10°C",
    "10-20°C",
    "20-30°C",
    "> 30°C",
]

DEFAULT_PRECIPITATION_BINS: Final[list[float]] = [0, 0.1, 1.0, 5.0, np.inf]
DEFAULT_PRECIPITATION_LABELS: Final[list[str]] = [
    "dry (0-0.1mm)",
    "light (0.1-1mm)",
    "moderate (1-5mm)",
    "heavy (> 5mm)",
]

DEFAULT_WIND_BINS: Final[list[float]] = [0, 3, 6, 10, np.inf]
DEFAULT_WIND_LABELS: Final[list[str]] = [
    "calm (0-3m/s)",
    "moderate (3-6m/s)",
    "brisk (6-10m/s)",
    "strong (> 10m/s)",
]


class DescriptiveAnalysisError(Exception):
    """Raised when descriptive statistics cannot be computed as requested.

    Covers missing required columns, empty input frames, frames with no
    non-null observations for the column being summarized, and mismatched
    bucket bin/label counts - shape problems that would otherwise silently
    produce a misleading or empty result.
    """


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    """Raises `DescriptiveAnalysisError` if any of `columns` is missing from `df`."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise DescriptiveAnalysisError(f"DataFrame is missing column(s): {missing}.")


def _dropna_or_raise(df: pd.DataFrame, subset: Sequence[str]) -> pd.DataFrame:
    """Drops rows with a null value in any of `subset`; raises if none remain."""
    valid = df.dropna(subset=list(subset))
    if valid.empty:
        raise DescriptiveAnalysisError(
            f"No rows with non-null values for {list(subset)}."
        )
    return valid


def rank_stations(
    df: pd.DataFrame,
    value_col: str = "total_count",
    station_col: str = "station_id",
) -> pd.DataFrame:
    """Ranks stations by mean traffic, busiest first.

    Rows with a null `value_col` (missing 15-minute interval) are skipped
    entirely rather than treated as zero traffic, so a station with many
    data gaps is not penalized for missingness relative to one with fewer
    gaps.

    Args:
        df: Rows with `station_col` and `value_col`.
        value_col: Column to average (typically ``total_count``).
        station_col: Column identifying the station.

    Returns:
        DataFrame with columns `station_col`, ``mean``, ``median``,
        ``n_obs``, sorted by ``mean`` descending, plus a 1-based ``rank``
        column (1 = busiest).

    Raises:
        DescriptiveAnalysisError: if a required column is missing, or `df`
            has zero non-null `value_col` observations.
    """
    _require_columns(df, [station_col, value_col])
    valid = _dropna_or_raise(df, [value_col])
    grouped = valid.groupby(station_col)[value_col].agg(
        mean="mean", median="median", n_obs="count"
    )
    ranked = grouped.sort_values("mean", ascending=False).reset_index()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def rank_stations_by_group(
    df: pd.DataFrame,
    period_col: str,
    value_col: str = "total_count",
    station_col: str = "station_id",
) -> pd.DataFrame:
    """Ranks stations by mean traffic separately within each `period_col` value.

    Useful for checking whether the busiest-station ranking changes across
    e.g. weekday vs. weekend, or month.

    Args:
        df: Rows with `period_col`, `station_col`, and `value_col`.
        period_col: Column to split the ranking by (e.g. ``"is_weekend"``
            or ``"month"``).
        value_col: Column to average (typically ``total_count``).
        station_col: Column identifying the station.

    Returns:
        DataFrame with columns `period_col`, `station_col`, ``mean``,
        ``n_obs``, ``rank`` (1 = busiest, computed within that period),
        sorted by `period_col` then ``rank``.

    Raises:
        DescriptiveAnalysisError: if a required column is missing, or `df`
            has zero non-null `value_col` observations.
    """
    _require_columns(df, [period_col, station_col, value_col])
    valid = _dropna_or_raise(df, [value_col])
    grouped = (
        valid.groupby([period_col, station_col])[value_col]
        .agg(mean="mean", n_obs="count")
        .reset_index()
    )
    grouped["rank"] = (
        grouped.groupby(period_col)["mean"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    return grouped.sort_values([period_col, "rank"]).reset_index(drop=True)


def average_by_time_feature(
    df: pd.DataFrame,
    time_col: str,
    value_col: str = "total_count",
) -> pd.DataFrame:
    """Computes mean traffic grouped by a single time feature.

    Args:
        df: Rows with `time_col` and `value_col` (e.g. `time_col` is
            ``"hour"``, ``"day_of_week"``, or ``"month"``).
        time_col: Column to group by.
        value_col: Column to average (typically ``total_count``).

    Returns:
        DataFrame with columns `time_col`, ``mean``, ``n_obs``, sorted by
        `time_col` ascending.

    Raises:
        DescriptiveAnalysisError: if a required column is missing, or `df`
            has zero non-null `value_col` observations.
    """
    _require_columns(df, [time_col, value_col])
    valid = _dropna_or_raise(df, [value_col])
    result = (
        valid.groupby(time_col)[value_col].agg(mean="mean", n_obs="count").reset_index()
    )
    return result.sort_values(time_col).reset_index(drop=True)


def average_by_time_feature_per_station(
    df: pd.DataFrame,
    time_col: str,
    value_col: str = "total_count",
    station_col: str = "station_id",
) -> pd.DataFrame:
    """Computes mean traffic grouped by a time feature, broken out per station.

    Lets a caller compare each station's own hourly/daily/monthly profile
    (e.g. to spot commuter-pattern stations with weekday rush-hour peaks
    vs. leisure-route stations with a flatter or afternoon-weighted
    profile), and can be called on an already-filtered subset of `df`
    (e.g. only weekend rows) to compare profiles across two subsets.

    Args:
        df: Rows with `time_col`, `station_col`, and `value_col`.
        time_col: Column to group by (e.g. ``"hour"``).
        value_col: Column to average (typically ``total_count``).
        station_col: Column identifying the station.

    Returns:
        Pivot-table DataFrame indexed by `station_col`, columns being the
        distinct values of `time_col`, values being mean `value_col`.

    Raises:
        DescriptiveAnalysisError: if a required column is missing, or `df`
            has zero non-null `value_col` observations.
    """
    _require_columns(df, [time_col, station_col, value_col])
    valid = _dropna_or_raise(df, [value_col])
    return valid.pivot_table(
        index=station_col, columns=time_col, values=value_col, aggfunc="mean"
    )


def bucket_numeric_column(
    series: pd.Series,
    bins: Sequence[float],
    labels: Sequence[str],
) -> pd.Series:
    """Buckets a numeric series into labeled ranges.

    Args:
        series: Numeric values to bucket (e.g. a weather column).
        bins: Bin edges, passed through to `pandas.cut` (length
            ``len(labels) + 1``).
        labels: One label per bucket.

    Returns:
        Categorical series of the same length/index as `series`, with `NaN`
        preserved for out-of-range or null input values.

    Raises:
        DescriptiveAnalysisError: if ``len(labels) != len(bins) - 1``.
    """
    if len(labels) != len(bins) - 1:
        raise DescriptiveAnalysisError(
            f"labels must have len(bins) - 1 = {len(bins) - 1} entries, "
            f"got {len(labels)}."
        )
    return pd.cut(series, bins=bins, labels=labels, right=False, include_lowest=True)


def average_by_weather_bucket(
    df: pd.DataFrame,
    weather_col: str,
    bins: Sequence[float],
    labels: Sequence[str],
    value_col: str = "total_count",
) -> pd.DataFrame:
    """Computes mean traffic per weather-value bucket, plus % vs. the overall mean.

    Args:
        df: Rows with `weather_col` and `value_col`.
        weather_col: Weather column to bucket (e.g.
            ``"weather_air_temperature_c"``).
        bins: Bucket edges, see `bucket_numeric_column`.
        labels: Bucket labels, see `bucket_numeric_column`.
        value_col: Column to average (typically ``total_count``).

    Returns:
        DataFrame with columns ``bucket``, ``mean``, ``n_obs``,
        ``pct_diff_from_overall`` (mean of that bucket vs. the overall mean
        across all buckets, as a percentage; positive = busier than
        average). Ordered by `labels`' input order.

    Raises:
        DescriptiveAnalysisError: if a required column is missing, `df` has
            zero rows with both `weather_col` and `value_col` non-null, or
            the bins/labels length mismatch (see `bucket_numeric_column`).
    """
    _require_columns(df, [weather_col, value_col])
    valid = _dropna_or_raise(df, [weather_col, value_col])
    overall_mean = valid[value_col].mean()

    bucketed = bucket_numeric_column(valid[weather_col], bins, labels)
    result = (
        valid.assign(bucket=bucketed)
        .groupby("bucket", observed=True)[value_col]
        .agg(mean="mean", n_obs="count")
        .reindex(labels)
        .reset_index(names="bucket")
    )
    result["pct_diff_from_overall"] = (
        100 * (result["mean"] - overall_mean) / overall_mean
    )
    return result


def weather_correlations(
    df: pd.DataFrame,
    weather_cols: Sequence[str],
    value_col: str = "total_count",
) -> pd.DataFrame:
    """Computes the Pearson correlation of traffic with each weather column.

    Each column is correlated on its own pairwise-complete subset (rows
    where both that weather column and `value_col` are non-null), so one
    sparsely-populated weather column does not shrink the sample used for
    another.

    Args:
        df: Rows with every column in `weather_cols` and `value_col`.
        weather_cols: Weather columns to correlate against `value_col`.
        value_col: Column to correlate against (typically ``total_count``).

    Returns:
        DataFrame with columns ``weather_col``, ``correlation`` (`NaN` if
        no overlapping non-null rows), ``n_obs``.

    Raises:
        DescriptiveAnalysisError: if a required column is missing, or
            `weather_cols` is empty.
    """
    if not weather_cols:
        raise DescriptiveAnalysisError("weather_cols must not be empty.")
    _require_columns(df, [*weather_cols, value_col])

    rows = []
    for col in weather_cols:
        pairwise = df[[col, value_col]].dropna()
        correlation = (
            pairwise[col].corr(pairwise[value_col]) if len(pairwise) else np.nan
        )
        rows.append(
            {"weather_col": col, "correlation": correlation, "n_obs": len(pairwise)}
        )
    return pd.DataFrame(rows)


def compare_boolean_flag(
    df: pd.DataFrame,
    flag_col: str,
    value_col: str = "total_count",
) -> dict[str, object]:
    """Compares mean traffic when a boolean flag is true vs. false.

    Args:
        df: Rows with `flag_col` and `value_col`.
        flag_col: Boolean column to split on (e.g. ``"is_public_holiday"``).
        value_col: Column to average (typically ``total_count``).

    Returns:
        Dict with keys ``mean_true``, ``mean_false``, ``pct_difference``
        (mean_true vs. mean_false, as a percentage; negative = lower
        traffic when the flag is true), ``n_true``, ``n_false`` (row
        counts with non-null `value_col`, by flag value).

    Raises:
        DescriptiveAnalysisError: if a required column is missing, `df` has
            zero non-null `value_col` observations, or either flag value
            (true/false) has zero observations.
    """
    _require_columns(df, [flag_col, value_col])
    valid = _dropna_or_raise(df, [value_col])

    is_true = valid[flag_col].astype(bool)
    true_rows = valid.loc[is_true, value_col]
    false_rows = valid.loc[~is_true, value_col]
    if true_rows.empty or false_rows.empty:
        raise DescriptiveAnalysisError(
            f"{flag_col!r} must have at least one true and one false row with "
            f"a non-null {value_col!r}; got {len(true_rows)} true, "
            f"{len(false_rows)} false."
        )

    mean_true = true_rows.mean()
    mean_false = false_rows.mean()
    return {
        "mean_true": mean_true,
        "mean_false": mean_false,
        "pct_difference": 100 * (mean_true - mean_false) / mean_false,
        "n_true": int(len(true_rows)),
        "n_false": int(len(false_rows)),
    }


# Channel-description phrasing is not uniform across this project's 23
# stations - four families observed (see notebook 07's directional-
# imbalance section): German "stadteinwärts"/"stadtauswärts" (several
# casings/brackets, and one real typo, "stdteinwärts", which still
# contains "einwärts" as a substring so the pattern below catches it
# unmodified), bare "einwärts"/"auswärts" with no "stadt" prefix, English
# "IN"/"OUT" ("Fahrräder IN/OUT" or "[Bike IN/OUT]"), and a few stations
# with no directional semantic at all (named by cross-street/landmark
# instead, e.g. "Richtung Osttor") - `classify_channel_direction` returns
# `None` for those, not a guess.
_INBOUND_RE: Final[re.Pattern[str]] = re.compile(r"einw[äa]rts", re.IGNORECASE)
_OUTBOUND_RE: Final[re.Pattern[str]] = re.compile(r"ausw[äa]rts", re.IGNORECASE)
# Case-sensitive on purpose: every English-phrased channel in this dataset
# uses uppercase "IN"/"OUT" specifically ("Fahrräder IN", "[Bike OUT]");
# matching lowercase too would risk a false positive on an unrelated
# German word that happens to contain "in"/"out" as a whole token.
_INBOUND_EN_RE: Final[re.Pattern[str]] = re.compile(r"\bIN\b")
_OUTBOUND_EN_RE: Final[re.Pattern[str]] = re.compile(r"\bOUT\b")

_CHANNEL_DESCRIPTION_RE: Final[re.Pattern[str]] = re.compile(r"^\d+\s*\((.*)\)$")


def classify_channel_direction(description: str) -> str | None:
    """Classifies a channel description as inbound, outbound, or neither.

    Args:
        description: A channel's description text (e.g. the parenthesized
            part of a ``"<channel_id> (<description>)"`` column name).

    Returns:
        ``"in"``, ``"out"``, or `None` if the description matches neither
        pattern (e.g. a station whose sub-channels are named by cross-
        street/landmark, with no inbound/outbound semantic at all - not
        every station in this dataset is classifiable, and this is not
        forced to guess for those).
    """
    if _INBOUND_RE.search(description) or _INBOUND_EN_RE.search(description):
        return "in"
    if _OUTBOUND_RE.search(description) or _OUTBOUND_EN_RE.search(description):
        return "out"
    return None


def compute_directional_totals(
    df: pd.DataFrame, station_id: int | str
) -> dict[str, float] | None:
    """Sums a station's directional channels into inbound/outbound totals.

    Excludes the combined channel (id matching `station_id` - see
    `muenster_bike_forecast.modeling.model_table.compute_total_count`,
    which already treats it as redundant with the directional channels
    summed here) and classifies every remaining channel by parsing its
    description (`classify_channel_direction`), summing all channels
    sharing a classification - including across different channel ids
    that share a direction classification, which `coalesce_channel_columns`
    alone does not merge (that only merges same-id duplicates).

    Known, deliberately-unresolved limitation: at least two real stations
    (Kanalpromenade Abschnitt 6, Gasselstiege) have multiple channel ids
    classified to the same direction that are *concurrently* populated
    for months at a time with different, only moderately correlated
    values (confirmed by direct inspection, not sequential "one retired,
    one active" reissued-sensor generations as originally assumed) -
    whether summing them is correct (e.g. genuinely separate lanes) or
    double-counts (e.g. overlapping sensor feeds) is **not verified
    either way**, unlike `compute_total_count`'s combined-vs-directional
    relationship, which is cross-checked by
    `combined_channel_matches_directional_sum` against real data. Treat
    `total_in`/`total_out` for any station with more than one channel id
    per direction as a best-effort estimate, not a verified total - see
    the caveats in ``notebooks/07_descriptive_analysis.ipynb``'s
    directional-imbalance section.

    Args:
        df: One station's rows, containing channel count columns (e.g. as
            loaded from ``data/raw/joined/<station_id>.csv``).
        station_id: The station's own id, to exclude its combined channel.

    Returns:
        `None` if no channel could be classified as inbound or outbound at
        all (a station with no directional semantic in its channel names).
        Otherwise a dict with keys ``total_in``, ``total_out`` (each a
        float total summed across the whole `df`, `NaN` if that direction
        was classified as present but every value across every matching
        channel was null - callers must not treat that `NaN` as "0
        traffic").
    """
    channel_columns = identify_channel_count_columns(list(df.columns))
    combined_id = str(station_id)

    in_ids: list[str] = []
    out_ids: list[str] = []
    for channel_id, columns in channel_columns.items():
        if channel_id == combined_id:
            continue
        # Guaranteed to match: `columns[0]` already passed
        # `identify_channel_count_columns`'s equivalent-shaped filter.
        description = _CHANNEL_DESCRIPTION_RE.match(columns[0]).group(1)
        direction = classify_channel_direction(description)
        if direction == "in":
            in_ids.append(channel_id)
        elif direction == "out":
            out_ids.append(channel_id)

    if not in_ids and not out_ids:
        return None

    def _sum_ids(ids: list[str]) -> float:
        if not ids:
            return float("nan")
        per_row = pd.DataFrame(
            {cid: coalesce_channel_columns(df, channel_columns[cid]) for cid in ids}
        ).sum(axis=1, min_count=1)
        # min_count=1 here too: without it, a direction classified as
        # present but with every row null would sum to a fabricated 0.0
        # instead of NaN (pandas' default `skipna=True` on an all-NaN
        # Series returns 0.0, not NaN) - silently misreporting "no data"
        # as "confirmed zero traffic", the same gotcha documented on
        # `muenster_bike_forecast.daily_report._sum_or_none`.
        return float(per_row.sum(min_count=1))

    return {"total_in": _sum_ids(in_ids), "total_out": _sum_ids(out_ids)}

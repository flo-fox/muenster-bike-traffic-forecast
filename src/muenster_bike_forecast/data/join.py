"""Join 15-minute bike counts with hourly weather into one dataset.

This module has three responsibilities, kept as pure functions:

1. Localizing bike-count timestamps (naive, Europe/Berlin local time) to
   UTC, so they are directly comparable with weather timestamps.
2. Combining the three per-parameter hourly weather DataFrames
   (``air_temperature``, ``precipitation``, ``wind``) into one wide hourly
   weather DataFrame.
3. As-of joining one station's 15-minute bike-count rows against the
   combined hourly weather DataFrame, using the most recent *past* weather
   reading for each bike-count row — never a future one, since this is a
   forecasting pipeline and a future weather value leaking onto a past
   bike-count row would invalidate any model trained on the result.

All I/O (reading/writing CSVs) is left to the caller (see
``notebooks/03_join_bike_weather.ipynb``); the functions here only
transform DataFrames already in memory.
"""

from __future__ import annotations

import pandas as pd


class JoinError(Exception):
    """Raised when bike-count/weather data cannot be joined as expected.

    Covers missing required columns, empty input frames, weather frames
    that don't all describe the same single station, and other shape
    problems that would otherwise silently produce a wrong or empty join.
    """


def localize_bike_timestamps(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    source_tz: str = "Europe/Berlin",
    result_col: str = "timestamp",
) -> pd.DataFrame:
    """Localizes naive bike-count datetimes to UTC.

    Bike-count timestamps are naive and represent `source_tz` local time.
    This localizes them to `source_tz` and converts to UTC, so they become
    directly comparable with weather timestamps (already UTC-aware).

    The two DST edge cases are made explicit rather than silently guessed
    at:

    - **Ambiguous** times (the repeated hour during the autumn fall-back,
      e.g. 02:30 occurring twice) become ``NaT``.
    - **Nonexistent** times (the skipped hour during the spring
      forward-jump, e.g. 02:30 never occurring) become ``NaT``.

    Callers should count ``df[result_col].isna().sum()`` (or use
    `summarize_dst_edge_cases`) to report how many rows were affected,
    rather than letting them disappear silently.

    Args:
        df: DataFrame with a naive-datetime column.
        datetime_col: Name of the naive-datetime column.
        source_tz: IANA timezone name the naive datetimes represent.
        result_col: Name of the new UTC-aware timestamp column to add.

    Returns:
        Copy of `df` with an added `result_col` column, dtype
        ``datetime64[ns, UTC]``, containing ``NaT`` for any row that fell
        on a DST ambiguous/nonexistent local time.

    Raises:
        JoinError: if `datetime_col` is not a column of `df`, or is already
            timezone-aware (localizing an already-aware series would raise
            inside pandas anyway, but this gives a clearer message).
    """
    if datetime_col not in df.columns:
        raise JoinError(f"Column {datetime_col!r} not found in DataFrame.")
    naive = pd.to_datetime(df[datetime_col])
    if naive.dt.tz is not None:
        raise JoinError(
            f"Column {datetime_col!r} is already timezone-aware; expected "
            "naive local-time values."
        )

    out = df.copy()
    localized = naive.dt.tz_localize(source_tz, ambiguous="NaT", nonexistent="NaT")
    out[result_col] = localized.dt.tz_convert("UTC")
    return out


def summarize_dst_edge_cases(
    df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> dict[str, object]:
    """Reports how many rows hit a DST ambiguous/nonexistent edge case.

    Args:
        df: DataFrame as returned by `localize_bike_timestamps`.
        timestamp_col: Name of the localized UTC timestamp column (`NaT`
            marks a DST edge case).

    Returns:
        Dict with keys ``n_rows``, ``n_dst_edge_case``, and
        ``pct_dst_edge_case``.

    Raises:
        JoinError: if `timestamp_col` is not a column of `df`.
    """
    if timestamp_col not in df.columns:
        raise JoinError(f"Column {timestamp_col!r} not found in DataFrame.")
    n_rows = len(df)
    n_dst_edge_case = int(df[timestamp_col].isna().sum())
    return {
        "n_rows": n_rows,
        "n_dst_edge_case": n_dst_edge_case,
        "pct_dst_edge_case": 100 * n_dst_edge_case / n_rows if n_rows else float("nan"),
    }


def combine_weather_parameters(
    parameter_frames: dict[str, pd.DataFrame],
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Combines per-parameter hourly weather DataFrames into one wide frame.

    Args:
        parameter_frames: Mapping from parameter key (e.g.
            ``"air_temperature"``) to its standardized DataFrame — columns
            ``station_id``, `timestamp_col`, plus parameter-specific value
            columns — as returned by ``weather.fetch_hourly_weather`` /
            ``weather.load_weather_data``. All frames must describe the
            same single weather station.
        timestamp_col: Name of the shared timestamp column to join on.

    Returns:
        Wide DataFrame with columns ``station_id``, `timestamp_col`, and
        every value column from every input frame, outer-joined on
        `timestamp_col` so no parameter's coverage is truncated to another
        parameter's shorter range. Value-column names that collide across
        parameters (e.g. every parameter has its own ``quality_level``) are
        prefixed with the parameter key to disambiguate. Sorted by
        `timestamp_col`.

    Raises:
        JoinError: if `parameter_frames` is empty, any frame is missing
            ``station_id`` or `timestamp_col`, or the frames don't all
            share the same single ``station_id``.
    """
    if not parameter_frames:
        raise JoinError("parameter_frames is empty; nothing to combine.")

    station_ids: set[str] = set()
    combined: pd.DataFrame | None = None
    seen_value_columns: set[str] = set()

    for parameter, frame in parameter_frames.items():
        missing_columns = {"station_id", timestamp_col} - set(frame.columns)
        if missing_columns:
            raise JoinError(
                f"Weather frame for parameter {parameter!r} is missing "
                f"column(s): {sorted(missing_columns)}."
            )
        station_ids.update(frame["station_id"].unique())

        value_columns = [
            c for c in frame.columns if c not in ("station_id", timestamp_col)
        ]
        rename_map = {
            col: f"{parameter}_{col}" if col in seen_value_columns else col
            for col in value_columns
        }
        seen_value_columns.update(rename_map.values())

        renamed = frame[[timestamp_col, *value_columns]].rename(columns=rename_map)
        combined = (
            renamed
            if combined is None
            else combined.merge(renamed, on=timestamp_col, how="outer")
        )

    if len(station_ids) != 1:
        raise JoinError(
            "Expected all weather frames to describe the same single "
            f"station, found station id(s): {sorted(station_ids)}."
        )

    assert combined is not None  # loop runs at least once (checked above)
    combined.insert(0, "station_id", next(iter(station_ids)))
    combined = combined.sort_values(timestamp_col).reset_index(drop=True)
    return combined


def join_station_weather(
    bike_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    tolerance: pd.Timedelta = pd.Timedelta(hours=2),
) -> pd.DataFrame:
    """As-of joins one station's 15-minute bike counts to hourly weather.

    Each bike-count row is matched to the most recent *past* weather
    reading within `tolerance`, via
    ``pandas.merge_asof(..., direction="backward")``. A bike-count row is
    never matched to a *future* weather reading — this is a forecasting
    pipeline, and doing so would leak information from the future into a
    training row. A row further than `tolerance` past the nearest weather
    reading (e.g. spanning a weather data gap) is left with null weather
    columns rather than silently matched to stale data.

    Rows whose `timestamp_col` is ``NaT`` (bike-count DST edge cases from
    `localize_bike_timestamps`) cannot be matched at all — `merge_asof`
    rejects null join keys — so they are carried through unmatched, with
    null weather columns, instead of raising or being dropped.

    Args:
        bike_df: One station's bike-count rows, with a UTC-aware
            `timestamp_col` column (see `localize_bike_timestamps`). Not
            required to be pre-sorted.
        weather_df: Combined hourly weather DataFrame (see
            `combine_weather_parameters`), with a UTC-aware `timestamp_col`
            column. Not required to be pre-sorted.
        timestamp_col: Name of the shared UTC timestamp column.
        tolerance: Maximum age of a matched weather reading.

    Returns:
        `bike_df`, in its original row order, with weather columns
        appended (all columns of `weather_df` except `timestamp_col`,
        renamed with a ``weather_`` prefix) plus a ``weather_timestamp``
        column recording which weather reading was actually matched (null
        if none was, within `tolerance`).

    Raises:
        JoinError: if `timestamp_col` is missing from either frame, or
            either DataFrame is empty.
    """
    if timestamp_col not in bike_df.columns:
        raise JoinError(f"bike_df is missing column {timestamp_col!r}.")
    if timestamp_col not in weather_df.columns:
        raise JoinError(f"weather_df is missing column {timestamp_col!r}.")
    if bike_df.empty:
        raise JoinError("bike_df is empty; nothing to join.")
    if weather_df.empty:
        raise JoinError("weather_df is empty; nothing to join.")

    weather_value_columns = [c for c in weather_df.columns if c != timestamp_col]
    weather_prepared = weather_df[[timestamp_col, *weather_value_columns]].copy()
    weather_prepared["weather_timestamp"] = weather_prepared[timestamp_col]
    weather_prepared = weather_prepared.rename(
        columns={c: f"weather_{c}" for c in weather_value_columns}
    )
    weather_sorted = weather_prepared.sort_values(timestamp_col).reset_index(drop=True)

    # merge_asof rejects null keys on the left side, so DST-edge-case rows
    # (timestamp_col == NaT) are split off, joined back afterwards with
    # null weather columns. A dedicated row-id column (rather than relying
    # on bike_df's own index being unique) restores the original row order
    # at the end regardless of what index bike_df came in with.
    working = bike_df.reset_index(drop=True)
    working["_join_row_id"] = working.index

    is_matchable = working[timestamp_col].notna()
    matchable = working.loc[is_matchable].sort_values(timestamp_col)
    unmatchable = working.loc[~is_matchable].copy()

    matched = pd.merge_asof(
        matchable,
        weather_sorted,
        on=timestamp_col,
        direction="backward",
        tolerance=tolerance,
    )

    new_weather_columns = [c for c in weather_sorted.columns if c != timestamp_col]
    for col in new_weather_columns:
        unmatchable[col] = pd.NA

    result = pd.concat([matched, unmatchable], ignore_index=True)
    result = result.sort_values("_join_row_id").drop(columns="_join_row_id")
    return result.reset_index(drop=True)


def summarize_weather_coverage(
    joined_df: pd.DataFrame,
    station_id: str | None = None,
    matched_col: str = "weather_timestamp",
) -> dict[str, object]:
    """Reports what fraction of joined rows have no matched weather reading.

    A row counts as missing weather if and only if `matched_col` is null —
    i.e. `join_station_weather` found no weather reading within tolerance
    (or the row's own timestamp was itself a DST edge case) — as opposed to
    a matched weather reading that happens to have a null *value* in one
    particular column (e.g. a DWD sentinel gap), which is a separate,
    already-reported concern (see ``weather.fetch_hourly_weather``).

    Args:
        joined_df: Result of `join_station_weather`.
        station_id: Optional station identifier, included in the report.
        matched_col: Name of the column recording the matched weather
            timestamp (null when no weather reading was matched).

    Returns:
        Dict with keys ``station_id`` (if given), ``n_rows``,
        ``n_missing_weather``, ``pct_missing_weather``.

    Raises:
        JoinError: if `joined_df` is empty or `matched_col` is missing.
    """
    if joined_df.empty:
        raise JoinError("joined_df is empty; cannot summarize coverage.")
    if matched_col not in joined_df.columns:
        raise JoinError(f"Column {matched_col!r} not found in DataFrame.")

    n_rows = len(joined_df)
    n_missing = int(joined_df[matched_col].isna().sum())
    report: dict[str, object] = {
        "n_rows": n_rows,
        "n_missing_weather": n_missing,
        "pct_missing_weather": 100 * n_missing / n_rows if n_rows else float("nan"),
    }
    if station_id is not None:
        report = {"station_id": station_id, **report}
    return report

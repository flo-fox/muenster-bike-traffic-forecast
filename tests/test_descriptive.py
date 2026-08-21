"""Tests for `muenster_bike_forecast.analysis.descriptive`.

All tests use small, hand-built synthetic data - no live network calls and
no dependency on the real `data/raw/` files.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from muenster_bike_forecast.analysis.descriptive import (
    DescriptiveAnalysisError,
    average_by_time_feature,
    average_by_time_feature_per_station,
    average_by_weather_bucket,
    bucket_numeric_column,
    classify_channel_direction,
    compare_boolean_flag,
    compute_directional_totals,
    rank_stations,
    rank_stations_by_group,
    weather_correlations,
)

# ---------------------------------------------------------------------------
# rank_stations
# ---------------------------------------------------------------------------


def test_rank_stations_sorts_busiest_first_and_skips_nulls() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["A", "A", "A", "B", "B", "B"],
            "total_count": [10.0, 20.0, None, 100.0, 100.0, 100.0],
        }
    )

    result = rank_stations(df)

    assert result["station_id"].tolist() == ["B", "A"]
    assert result["rank"].tolist() == [1, 2]
    assert result.loc[result["station_id"] == "A", "n_obs"].iloc[0] == 2
    assert result.loc[result["station_id"] == "A", "mean"].iloc[0] == pytest.approx(
        15.0
    )


def test_rank_stations_raises_on_missing_column() -> None:
    with pytest.raises(DescriptiveAnalysisError):
        rank_stations(pd.DataFrame({"station_id": ["A"]}))


def test_rank_stations_raises_when_all_values_null() -> None:
    df = pd.DataFrame({"station_id": ["A", "B"], "total_count": [None, None]})
    with pytest.raises(DescriptiveAnalysisError):
        rank_stations(df)


# ---------------------------------------------------------------------------
# rank_stations_by_group
# ---------------------------------------------------------------------------


def test_rank_stations_by_group_ranks_within_each_period() -> None:
    # Station A busiest on weekdays, station B busiest on weekends.
    df = pd.DataFrame(
        {
            "station_id": ["A", "A", "B", "B"],
            "is_weekend": [False, True, False, True],
            "total_count": [100.0, 10.0, 20.0, 200.0],
        }
    )

    result = rank_stations_by_group(df, period_col="is_weekend")

    weekday = result.loc[~result["is_weekend"]].set_index("station_id")
    weekend = result.loc[result["is_weekend"]].set_index("station_id")
    assert weekday.loc["A", "rank"] == 1
    assert weekday.loc["B", "rank"] == 2
    assert weekend.loc["B", "rank"] == 1
    assert weekend.loc["A", "rank"] == 2


def test_rank_stations_by_group_raises_on_missing_column() -> None:
    df = pd.DataFrame({"station_id": ["A"], "total_count": [1.0]})
    with pytest.raises(DescriptiveAnalysisError):
        rank_stations_by_group(df, period_col="does_not_exist")


# ---------------------------------------------------------------------------
# average_by_time_feature / average_by_time_feature_per_station
# ---------------------------------------------------------------------------


def test_average_by_time_feature_groups_and_sorts() -> None:
    df = pd.DataFrame(
        {"hour": [1, 1, 0, 0, 0], "total_count": [10.0, 20.0, None, 5.0, 15.0]}
    )

    result = average_by_time_feature(df, time_col="hour")

    assert result["hour"].tolist() == [0, 1]
    assert result.loc[result["hour"] == 0, "mean"].iloc[0] == pytest.approx(10.0)
    assert result.loc[result["hour"] == 0, "n_obs"].iloc[0] == 2
    assert result.loc[result["hour"] == 1, "mean"].iloc[0] == pytest.approx(15.0)


def test_average_by_time_feature_raises_on_missing_column() -> None:
    with pytest.raises(DescriptiveAnalysisError):
        average_by_time_feature(pd.DataFrame({"total_count": [1.0]}), time_col="hour")


def test_average_by_time_feature_per_station_pivots_by_station() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["A", "A", "B", "B"],
            "hour": [0, 1, 0, 1],
            "total_count": [10.0, 30.0, 100.0, 300.0],
        }
    )

    result = average_by_time_feature_per_station(df, time_col="hour")

    assert result.loc["A", 0] == pytest.approx(10.0)
    assert result.loc["A", 1] == pytest.approx(30.0)
    assert result.loc["B", 0] == pytest.approx(100.0)
    assert result.loc["B", 1] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# bucket_numeric_column / average_by_weather_bucket
# ---------------------------------------------------------------------------


def test_bucket_numeric_column_assigns_expected_labels() -> None:
    series = pd.Series([-5.0, 5.0, 25.0])
    result = bucket_numeric_column(
        series, bins=[-np.inf, 0, 20, np.inf], labels=["cold", "mild", "hot"]
    )
    assert list(result) == ["cold", "mild", "hot"]


def test_bucket_numeric_column_raises_on_mismatched_lengths() -> None:
    with pytest.raises(DescriptiveAnalysisError):
        bucket_numeric_column(pd.Series([1.0]), bins=[0, 1, 2], labels=["only_one"])


def test_average_by_weather_bucket_reports_pct_diff_from_overall() -> None:
    df = pd.DataFrame(
        {
            "weather_air_temperature_c": [-5.0, -5.0, 25.0, 25.0],
            "total_count": [10.0, 10.0, 30.0, 30.0],
        }
    )

    result = average_by_weather_bucket(
        df,
        weather_col="weather_air_temperature_c",
        bins=[-np.inf, 0, np.inf],
        labels=["cold", "warm"],
    )

    # Overall mean is 20; cold bucket mean is 10 -> -50%, warm is 30 -> +50%.
    result = result.set_index("bucket")
    assert result.loc["cold", "mean"] == pytest.approx(10.0)
    assert result.loc["cold", "pct_diff_from_overall"] == pytest.approx(-50.0)
    assert result.loc["warm", "pct_diff_from_overall"] == pytest.approx(50.0)
    assert result.loc["cold", "n_obs"] == 2


def test_average_by_weather_bucket_raises_on_missing_column() -> None:
    with pytest.raises(DescriptiveAnalysisError):
        average_by_weather_bucket(
            pd.DataFrame({"total_count": [1.0]}),
            weather_col="does_not_exist",
            bins=[0, 1],
            labels=["x"],
        )


# ---------------------------------------------------------------------------
# weather_correlations
# ---------------------------------------------------------------------------


def test_weather_correlations_computes_pearson_per_column() -> None:
    df = pd.DataFrame(
        {
            "weather_air_temperature_c": [0.0, 10.0, 20.0, 30.0],
            "weather_precipitation_mm": [30.0, 20.0, 10.0, 0.0],
            "total_count": [10.0, 20.0, 30.0, 40.0],
        }
    )

    result = weather_correlations(
        df, weather_cols=["weather_air_temperature_c", "weather_precipitation_mm"]
    ).set_index("weather_col")

    assert result.loc["weather_air_temperature_c", "correlation"] == pytest.approx(1.0)
    assert result.loc["weather_precipitation_mm", "correlation"] == pytest.approx(-1.0)
    assert result.loc["weather_air_temperature_c", "n_obs"] == 4


def test_weather_correlations_ignores_rows_null_only_in_other_columns() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, None],
            "b": [1.0, None, 3.0, 4.0],
            "total_count": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = weather_correlations(df, weather_cols=["a", "b"]).set_index("weather_col")

    assert result.loc["a", "n_obs"] == 3
    assert result.loc["b", "n_obs"] == 3


def test_weather_correlations_raises_on_empty_weather_cols() -> None:
    with pytest.raises(DescriptiveAnalysisError):
        weather_correlations(pd.DataFrame({"total_count": [1.0]}), weather_cols=[])


# ---------------------------------------------------------------------------
# compare_boolean_flag
# ---------------------------------------------------------------------------


def test_compare_boolean_flag_computes_pct_difference() -> None:
    df = pd.DataFrame(
        {
            "is_public_holiday": [True, True, False, False],
            "total_count": [50.0, 50.0, 100.0, 100.0],
        }
    )

    result = compare_boolean_flag(df, flag_col="is_public_holiday")

    assert result["mean_true"] == pytest.approx(50.0)
    assert result["mean_false"] == pytest.approx(100.0)
    assert result["pct_difference"] == pytest.approx(-50.0)
    assert result["n_true"] == 2
    assert result["n_false"] == 2


def test_compare_boolean_flag_raises_when_one_side_is_empty() -> None:
    df = pd.DataFrame({"is_public_holiday": [False, False], "total_count": [1.0, 2.0]})
    with pytest.raises(DescriptiveAnalysisError):
        compare_boolean_flag(df, flag_col="is_public_holiday")


def test_compare_boolean_flag_raises_on_missing_column() -> None:
    with pytest.raises(DescriptiveAnalysisError):
        compare_boolean_flag(pd.DataFrame({"total_count": [1.0]}), flag_col="x")


# ---------------------------------------------------------------------------
# classify_channel_direction
# ---------------------------------------------------------------------------


def test_classify_channel_direction_handles_stadteinwaerts_variants() -> None:
    assert classify_channel_direction("Neutor stadteinwärts") == "in"
    assert classify_channel_direction("[Bike Stadteinwärts]") == "in"
    assert classify_channel_direction("FR stdteinwärts") == "in"  # real typo, no "a"
    assert classify_channel_direction("Gartenstraße einwärts") == "in"


def test_classify_channel_direction_handles_stadtauswaerts_variants() -> None:
    assert classify_channel_direction("Neutor stadtauswärts") == "out"
    assert classify_channel_direction("[Bike Stadtauswärts]") == "out"
    assert classify_channel_direction("Gartenstraße auswärts") == "out"


def test_classify_channel_direction_handles_english_in_out() -> None:
    assert classify_channel_direction("Bismarckallee Fahrräder IN") == "in"
    assert classify_channel_direction("Hafenstraße Fahrräder OUT") == "out"
    assert classify_channel_direction("[Bike IN]") == "in"
    assert classify_channel_direction("[Bike OUT]") == "out"


def test_classify_channel_direction_returns_none_for_placename_only() -> None:
    assert classify_channel_direction("Fahrräder Richtung Osttor") is None
    assert classify_channel_direction("Radfahrer FR Mauritztor") is None
    assert classify_channel_direction("Neutor") is None  # the combined channel itself


# ---------------------------------------------------------------------------
# compute_directional_totals
# ---------------------------------------------------------------------------


def _neutor_style_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "100035541 (Neutor)": [4, 7, 99],
            "101035541 (Neutor stadteinwärts)": [1, 5, 40],
            "102035541 (Neutor stadtauswärts)": [3, 2, 59],
        }
    )


def test_compute_directional_totals_sums_by_direction_and_excludes_combined() -> None:
    df = _neutor_style_df()

    result = compute_directional_totals(df, station_id="100035541")

    assert result == {"total_in": 46.0, "total_out": 64.0}  # (1+5+40), (3+2+59)


def test_compute_directional_totals_sums_sequential_channel_ids() -> None:
    # Two channel ids classified to the same direction, non-overlapping in
    # time (one null wherever the other is present) - must be summed as
    # one "in" total, not treated as two separate unclassified channels.
    df = pd.DataFrame(
        {
            "1 (Station)": [10, 10],
            "2 (Station Stadteinwärts)": [4, None],
            "3 (Station [Bike IN])": [None, 6],
            "4 (Station Stadtauswärts)": [6, 4],
        }
    )

    result = compute_directional_totals(df, station_id="1")

    assert result == {"total_in": 10.0, "total_out": 10.0}


def test_compute_directional_totals_sums_concurrently_overlapping_channel_ids() -> None:
    # Pins a documented, unverified-but-current behavior (see the
    # function's docstring): two channel ids sharing a direction
    # classification are summed even when *concurrently* populated with
    # different values, not just when sequentially non-overlapping - the
    # real pattern found at Kanalpromenade Abschnitt 6/Gasselstiege
    # (confirmed 2026-08-21), which is genuinely different from the
    # "reissued sensor, one retired" assumption this function's summing
    # was originally designed around. Whether concurrent-summing is
    # correct is not verified either way; this test only pins the current
    # behavior so a future change to it is deliberate.
    df = pd.DataFrame(
        {
            "1 (Station)": [20.0, 20.0],
            "2 (Station Stadteinwärts)": [4.0, 3.0],
            "3 (Station [Bike IN])": [2.0, 5.0],
        }
    )

    result = compute_directional_totals(df, station_id="1")

    assert result["total_in"] == 14.0  # (4+2) + (3+5)
    assert math.isnan(result["total_out"])  # no "out" channel present at all here


def test_compute_directional_totals_all_null_direction_is_nan_not_zero() -> None:
    # A classified direction with zero real readings anywhere must read
    # as "no data" (NaN), not a fabricated "confirmed zero traffic" (0.0)
    # - mirrors the same gotcha already fixed for
    # `muenster_bike_forecast.daily_report._sum_or_none`.
    df = pd.DataFrame(
        {
            "1 (Station)": [10.0, 10.0],
            "2 (Station stadteinwärts)": [None, None],
            "3 (Station stadtauswärts)": [None, None],
        }
    )

    result = compute_directional_totals(df, station_id="1")

    assert math.isnan(result["total_in"])
    assert math.isnan(result["total_out"])


def test_compute_directional_totals_returns_none_when_unclassifiable() -> None:
    df = pd.DataFrame(
        {
            "1 (Station)": [10.0],
            "2 (Station Richtung Osttor)": [4.0],
            "3 (Station Richtung Zentrum)": [6.0],
        }
    )
    assert compute_directional_totals(df, station_id="1") is None

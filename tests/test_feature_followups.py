"""Tests for `muenster_bike_forecast.modeling.feature_followups`.

All tests use small, hand-built synthetic data - no live network calls and
no dependency on the real `data/raw/` files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from muenster_bike_forecast.modeling.feature_followups import (
    FeatureFollowupError,
    add_hour_dow_interaction,
    add_precipitation_bucket,
    add_weekend_weekday_ratio_feature,
    compute_weekend_weekday_ratio,
)

# ---------------------------------------------------------------------------
# add_precipitation_bucket
# ---------------------------------------------------------------------------


def test_add_precipitation_bucket_uses_default_bins() -> None:
    df = pd.DataFrame({"weather_precipitation_mm": [0.0, 0.05, 0.5, 3.0, 10.0, np.nan]})

    result = add_precipitation_bucket(df)

    assert result["precipitation_bucket"].tolist() == [
        "dry (0-0.1mm)",
        "dry (0-0.1mm)",
        "light (0.1-1mm)",
        "moderate (1-5mm)",
        "heavy (> 5mm)",
        np.nan,
    ]


def test_add_precipitation_bucket_missing_column_raises() -> None:
    df = pd.DataFrame({"other": [1.0]})
    with pytest.raises(FeatureFollowupError):
        add_precipitation_bucket(df)


def test_add_precipitation_bucket_custom_bins() -> None:
    df = pd.DataFrame({"weather_precipitation_mm": [0.0, 5.0]})
    result = add_precipitation_bucket(df, bins=[0, 2.0, np.inf], labels=["low", "high"])
    assert result["precipitation_bucket"].tolist() == ["low", "high"]


# ---------------------------------------------------------------------------
# add_hour_dow_interaction
# ---------------------------------------------------------------------------


def test_add_hour_dow_interaction_combines_columns() -> None:
    df = pd.DataFrame({"hour": [7, 16, 7], "day_of_week": [0, 0, 5]})

    result = add_hour_dow_interaction(df)

    assert result["hour_dow"].tolist() == ["7_0", "16_0", "7_5"]


def test_add_hour_dow_interaction_distinguishes_swapped_values() -> None:
    # Guards against an accidental string-concat bug where (7, 16) and
    # (1, 6) or similar collide - hour and day_of_week ranges don't overlap
    # here, but the separator must still make each pair unique.
    df = pd.DataFrame({"hour": [1, 12], "day_of_week": [2, 1]})
    result = add_hour_dow_interaction(df)
    assert result["hour_dow"].tolist() == ["1_2", "12_1"]
    assert len(set(result["hour_dow"])) == 2


def test_add_hour_dow_interaction_missing_column_raises() -> None:
    df = pd.DataFrame({"hour": [1]})
    with pytest.raises(FeatureFollowupError):
        add_hour_dow_interaction(df)


# ---------------------------------------------------------------------------
# compute_weekend_weekday_ratio
# ---------------------------------------------------------------------------


def _synthetic_station_rows(
    station_id: int, weekday_values: list[float], weekend_values: list[float]
) -> pd.DataFrame:
    weekday_dows = [0, 1, 2, 3, 4] * (len(weekday_values) // 5 + 1)
    weekend_dows = [5, 6] * (len(weekend_values) // 2 + 1)
    return pd.DataFrame(
        {
            "station_id": station_id,
            "total_count": weekday_values + weekend_values,
            "day_of_week": weekday_dows[: len(weekday_values)]
            + weekend_dows[: len(weekend_values)],
        }
    )


def test_compute_weekend_weekday_ratio_commuter_station_below_one() -> None:
    # Commuter route: busier on weekdays, so weekend/weekday < 1.
    df = _synthetic_station_rows(
        1, weekday_values=[100.0] * 5, weekend_values=[40.0] * 2
    )

    result = compute_weekend_weekday_ratio(df)

    assert result.loc[result["station_id"] == 1, "weekend_weekday_ratio"].iloc[
        0
    ] == pytest.approx(0.4)


def test_compute_weekend_weekday_ratio_leisure_station_above_one() -> None:
    # Leisure route: busier on weekends, so weekend/weekday > 1.
    df = _synthetic_station_rows(
        2, weekday_values=[50.0] * 5, weekend_values=[80.0] * 2
    )

    result = compute_weekend_weekday_ratio(df)

    assert result.loc[result["station_id"] == 2, "weekend_weekday_ratio"].iloc[
        0
    ] == pytest.approx(1.6)


def test_compute_weekend_weekday_ratio_multi_station() -> None:
    df = pd.concat(
        [
            _synthetic_station_rows(1, [100.0] * 5, [40.0] * 2),
            _synthetic_station_rows(2, [50.0] * 5, [80.0] * 2),
        ],
        ignore_index=True,
    )

    result = compute_weekend_weekday_ratio(df)

    assert set(result["station_id"]) == {1, 2}
    assert len(result) == 2


def test_compute_weekend_weekday_ratio_ignores_nulls() -> None:
    df = _synthetic_station_rows(1, [100.0] * 5, [40.0] * 2)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {"station_id": [1], "total_count": [np.nan], "day_of_week": [6]}
            ),
        ],
        ignore_index=True,
    )

    result = compute_weekend_weekday_ratio(df)

    assert result.loc[result["station_id"] == 1, "weekend_weekday_ratio"].iloc[
        0
    ] == pytest.approx(0.4)


def test_compute_weekend_weekday_ratio_missing_weekend_data_raises() -> None:
    df = pd.DataFrame(
        {
            "station_id": [1] * 5,
            "total_count": [10.0] * 5,
            "day_of_week": [0, 1, 2, 3, 4],
        }
    )
    with pytest.raises(FeatureFollowupError):
        compute_weekend_weekday_ratio(df)


def test_compute_weekend_weekday_ratio_missing_column_raises() -> None:
    df = pd.DataFrame({"station_id": [1], "total_count": [1.0]})
    with pytest.raises(FeatureFollowupError):
        compute_weekend_weekday_ratio(df)


# ---------------------------------------------------------------------------
# add_weekend_weekday_ratio_feature
# ---------------------------------------------------------------------------


def test_add_weekend_weekday_ratio_feature_broadcasts_static_value() -> None:
    ratio_table = pd.DataFrame(
        {"station_id": [1, 2], "weekend_weekday_ratio": [0.4, 1.6]}
    )
    df = pd.DataFrame({"station_id": [1, 1, 2], "other": [10, 20, 30]})

    result = add_weekend_weekday_ratio_feature(df, ratio_table)

    assert result["weekend_weekday_ratio"].tolist() == [0.4, 0.4, 1.6]


def test_add_weekend_weekday_ratio_feature_does_not_mutate_row_count() -> None:
    ratio_table = pd.DataFrame(
        {"station_id": [1, 2], "weekend_weekday_ratio": [0.4, 1.6]}
    )
    df = pd.DataFrame({"station_id": [1, 2, 1, 2, 1]})

    result = add_weekend_weekday_ratio_feature(df, ratio_table)

    assert len(result) == len(df)


def test_add_weekend_weekday_ratio_feature_unmatched_station_raises() -> None:
    ratio_table = pd.DataFrame({"station_id": [1], "weekend_weekday_ratio": [0.4]})
    df = pd.DataFrame({"station_id": [1, 2]})  # station 2 absent from ratio_table

    with pytest.raises(FeatureFollowupError):
        add_weekend_weekday_ratio_feature(df, ratio_table)


def test_add_weekend_weekday_ratio_feature_missing_column_raises() -> None:
    ratio_table = pd.DataFrame({"station_id": [1], "weekend_weekday_ratio": [0.4]})
    df = pd.DataFrame({"other": [1]})
    with pytest.raises(FeatureFollowupError):
        add_weekend_weekday_ratio_feature(df, ratio_table)


def test_compute_and_broadcast_ratio_train_only_guards_against_leakage() -> None:
    # Simulates the leakage guard end-to-end: a station whose weekend
    # traffic changes sharply between train and test periods should have
    # its ratio driven only by the train-period rows, not shifted by
    # what's in the (excluded) test period.
    train_rows = _synthetic_station_rows(1, [100.0] * 5, [40.0] * 2)
    test_rows = _synthetic_station_rows(1, [100.0] * 5, [400.0] * 2)  # test-only spike

    ratio_table = compute_weekend_weekday_ratio(train_rows)
    full_df = pd.concat([train_rows, test_rows], ignore_index=True)
    result = add_weekend_weekday_ratio_feature(full_df, ratio_table)

    # Every row (train and test) gets the *train-derived* ratio (0.4), not
    # one influenced by the test-period spike to 400.0.
    assert np.allclose(result["weekend_weekday_ratio"].to_numpy(), 0.4)

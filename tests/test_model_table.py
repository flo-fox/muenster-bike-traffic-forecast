"""Tests for `muenster_bike_forecast.modeling.model_table`.

All tests use small, hand-built synthetic data - no live network calls and
no dependency on the real `data/raw/` files. Includes a synthetic
reproduction of the duplicate-channel-column scenario (a channel id whose
description was renamed mid-history, see the module docstring) and a
synthetic data gap to verify the exact-timestamp target lookup does not
silently pair a row with the wrong one across it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from muenster_bike_forecast.modeling.model_table import (
    ModelTableError,
    add_baseline_prediction,
    add_calendar_features,
    add_forecast_target,
    chronological_split,
    coalesce_channel_columns,
    combined_channel_matches_directional_sum,
    compute_baseline_metrics,
    compute_total_count,
    identify_channel_count_columns,
    summarize_baseline_evaluable_rows,
    summarize_target_nulls,
)

# ---------------------------------------------------------------------------
# identify_channel_count_columns
# ---------------------------------------------------------------------------


def test_identify_channel_count_columns_groups_by_leading_id() -> None:
    columns = [
        "station_id",
        "datetime",
        "101 (FR stdteinwärts)",
        "101 (FR stadteinwärts)",
        "102 (FR stadtauswärts)",
        "101-status",
        "weather_air_temperature_c",
    ]

    result = identify_channel_count_columns(columns)

    assert result == {
        "101": ["101 (FR stdteinwärts)", "101 (FR stadteinwärts)"],
        "102": ["102 (FR stadtauswärts)"],
    }


def test_identify_channel_count_columns_handles_nested_parens_in_description() -> None:
    columns = ["100031297 (Promenade (nördl. Salzstraße))"]

    result = identify_channel_count_columns(columns)

    assert result == {"100031297": ["100031297 (Promenade (nördl. Salzstraße))"]}


def test_identify_channel_count_columns_ignores_non_count_columns() -> None:
    result = identify_channel_count_columns(["station_id", "datetime", "weather_x"])
    assert result == {}


# ---------------------------------------------------------------------------
# coalesce_channel_columns / compute_total_count
# ---------------------------------------------------------------------------


def _duplicate_channel_df() -> pd.DataFrame:
    # Reproduces the real typo-rename scenario for the *combined* channel
    # (id "101", matching this fixture's own station_id): its description
    # changed mid-history, so the two "101" columns are never both
    # non-null for the same row - except the last row, added deliberately
    # to exercise the "both null" (missing data) case. Channel "102" is a
    # directional sub-channel, present for realism but must be ignored by
    # compute_total_count (which selects only the channel matching
    # station_id).
    return pd.DataFrame(
        {
            "station_id": ["101"] * 4,
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 00:00",
                    "2024-01-01 00:15",
                    "2024-01-01 00:30",
                    "2024-01-01 00:45",
                ]
            ),
            "101 (FR stdteinwärts)": [5, None, None, None],
            "101 (FR stadteinwärts)": [None, 7, None, None],
            "102 (FR stadtauswärts)": [1, 2, 3, None],
        }
    )


def test_coalesce_channel_columns_merges_non_overlapping_values() -> None:
    df = _duplicate_channel_df()
    coalesced = coalesce_channel_columns(
        df, ["101 (FR stdteinwärts)", "101 (FR stadteinwärts)"]
    )
    assert coalesced.iloc[0] == 5
    assert coalesced.iloc[1] == 7
    assert pd.isna(coalesced.iloc[2])
    assert pd.isna(coalesced.iloc[3])


def test_coalesce_channel_columns_raises_on_genuine_conflict() -> None:
    df = pd.DataFrame({"a": [5, 1], "b": [9, 1]})
    with pytest.raises(ModelTableError):
        coalesce_channel_columns(df, ["a", "b"])


def test_coalesce_channel_columns_raises_on_missing_column() -> None:
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ModelTableError):
        coalesce_channel_columns(df, ["a", "does_not_exist"])


def test_compute_total_count_selects_combined_channel_and_ignores_directional() -> None:
    # compute_total_count no longer sums channels - it selects the
    # combined channel (id matching station_id) directly, coalescing that
    # channel's own duplicate-description columns (the real mid-history
    # rename scenario) but ignoring unrelated directional channels
    # entirely. Channel "102" here (values [1, 2, 3, None]) would have
    # been wrongly folded in by the old sum-based behavior - it must not
    # affect the result at all now.
    df = _duplicate_channel_df()

    total = compute_total_count(df, station_id="101")

    assert total.iloc[0] == 5.0
    assert total.iloc[1] == 7.0
    assert pd.isna(total.iloc[2])
    assert pd.isna(total.iloc[3])


def test_compute_total_count_accepts_int_station_id() -> None:
    df = _duplicate_channel_df()
    total = compute_total_count(df, station_id=101)
    assert total.iloc[0] == 5.0


def test_compute_total_count_selected_channel_keeps_null_rows_as_nan() -> None:
    df = pd.DataFrame(
        {
            "101 (a)": [None, 1.0],
            "102 (b)": [None, 2.0],
        }
    )

    total = compute_total_count(df, station_id="101")

    assert pd.isna(total.iloc[0])
    assert total.iloc[1] == 1.0


def test_compute_total_count_raises_when_no_count_columns() -> None:
    df = pd.DataFrame({"station_id": ["S1"], "datetime": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ModelTableError):
        compute_total_count(df, station_id="S1")


def test_compute_total_count_raises_when_no_channel_matches_station_id() -> None:
    df = pd.DataFrame({"101 (a)": [1.0]})
    with pytest.raises(ModelTableError):
        compute_total_count(df, station_id="999")


# ---------------------------------------------------------------------------
# combined_channel_matches_directional_sum
# ---------------------------------------------------------------------------


def _combined_plus_directional_df() -> pd.DataFrame:
    # Reproduces the real "Neutor" pattern confirmed 2026-08-21: the
    # combined channel (id "100035541") already equals the sum of its two
    # directional channels for the first two rows; the third row is a
    # deliberate mismatch (to prove the check can fail, not just pass);
    # the fourth row has no directional data at all (a gap), which must
    # read as "nothing to check", not "mismatch".
    return pd.DataFrame(
        {
            "100035541 (Neutor)": [4, 7, 99, 5],
            "101035541 (Neutor stadteinwärts)": [1, 5, 1, None],
            "102035541 (Neutor stadtauswärts)": [3, 2, 1, None],
        }
    )


def test_combined_channel_matches_directional_sum_confirms_real_pattern() -> None:
    df = _combined_plus_directional_df()

    result = combined_channel_matches_directional_sum(df, station_id="100035541")

    assert bool(result.iloc[0]) is True  # 4 == 1 + 3
    assert bool(result.iloc[1]) is True  # 7 == 5 + 2
    assert bool(result.iloc[2]) is False  # 99 != 1 + 1
    assert result.iloc[3] is pd.NA  # no directional data to compare against


def test_combined_channel_matches_directional_sum_accepts_int_station_id() -> None:
    df = _combined_plus_directional_df()
    result = combined_channel_matches_directional_sum(df, station_id=100035541)
    assert bool(result.iloc[0]) is True


def test_combined_channel_matches_directional_sum_null_combined_is_not_checked() -> (
    None
):
    df = pd.DataFrame(
        {
            "1 (combined)": [None],
            "2 (dir a)": [3.0],
            "3 (dir b)": [4.0],
        }
    )
    result = combined_channel_matches_directional_sum(df, station_id="1")
    assert result.iloc[0] is pd.NA


def test_combined_channel_matches_directional_sum_partial_directional_gap_is_a_known_limitation() -> (
    None
):
    # Pins a documented, accepted limitation (see the function's Returns
    # docstring): a *partial* directional gap (one of two channels null,
    # the other present) is NOT detected as "nothing to check" - it's
    # scored as a disagreement instead, because requiring every
    # directional column non-null would break multi-generation stations
    # (several reissued-channel-id columns are null outside their own
    # vintage on every row, not just gap rows). This test exists so a
    # future change to this behavior is a deliberate decision, not an
    # accidental one.
    df = pd.DataFrame(
        {
            "1 (combined)": [10.0],
            "2 (dir a)": [None],  # a genuine gap on this one channel
            "3 (dir b)": [4.0],
        }
    )
    result = combined_channel_matches_directional_sum(df, station_id="1")
    assert bool(result.iloc[0]) is False  # not pd.NA, despite the real gap


def test_combined_channel_matches_directional_sum_raises_when_no_combined_channel() -> (
    None
):
    df = pd.DataFrame({"2 (dir a)": [1.0], "3 (dir b)": [2.0]})
    with pytest.raises(ModelTableError):
        combined_channel_matches_directional_sum(df, station_id="1")


def test_combined_channel_matches_directional_sum_raises_when_no_directional_channels() -> (
    None
):
    df = pd.DataFrame({"1 (combined)": [5.0]})
    with pytest.raises(ModelTableError):
        combined_channel_matches_directional_sum(df, station_id="1")


# ---------------------------------------------------------------------------
# add_forecast_target / summarize_target_nulls
# ---------------------------------------------------------------------------


def test_add_forecast_target_uses_exact_timestamp_not_position() -> None:
    # Station S1 has a gap at 00:30 (missing 15-minute interval). With a
    # 30-minute horizon, the row at 00:00 should look up 00:30 - which is
    # missing - and get a null target, NOT be silently paired with the next
    # available row (00:45) as a naive positional shift would do.
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S1", "S1"],
            "datetime": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:15", "2024-01-01 00:45"]
            ),
            "total_count": [10.0, 20.0, 40.0],
        }
    )

    out = add_forecast_target(df, horizon=pd.Timedelta(minutes=30))

    # 00:00 + 30min = 00:30 -> missing -> null target.
    assert pd.isna(
        out.loc[out["datetime"] == "2024-01-01 00:00", "target_total_count"].iloc[0]
    )
    # 00:15 + 30min = 00:45 -> present -> target is that row's total_count.
    assert (
        out.loc[out["datetime"] == "2024-01-01 00:15", "target_total_count"].iloc[0]
        == 40.0
    )
    # 00:45 + 30min = 01:15 -> not in data at all -> null target.
    assert pd.isna(
        out.loc[out["datetime"] == "2024-01-01 00:45", "target_total_count"].iloc[0]
    )


def test_add_forecast_target_does_not_cross_station_boundaries() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S2"],
            "datetime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:30"]),
            "total_count": [10.0, 99.0],
        }
    )

    out = add_forecast_target(df, horizon=pd.Timedelta(minutes=30))

    # S1's 00:00 + 30min = 00:30, which exists only for S2 - must not match.
    assert pd.isna(out.loc[out["station_id"] == "S1", "target_total_count"].iloc[0])


def test_add_forecast_target_raises_on_duplicate_station_timestamp_pairs() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "datetime": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 00:00"]),
            "total_count": [10.0, 11.0],
        }
    )
    with pytest.raises(ModelTableError):
        add_forecast_target(df)


def test_add_forecast_target_raises_on_missing_column() -> None:
    df = pd.DataFrame({"station_id": ["S1"], "datetime": [pd.Timestamp("2024-01-01")]})
    with pytest.raises(ModelTableError):
        add_forecast_target(df)


def test_add_forecast_target_returns_valid_empty_result_on_empty_input() -> None:
    # pd.MultiIndex.from_tuples([]) raises TypeError on an empty key list -
    # this must not propagate as a bare, uninformative exception.
    df = pd.DataFrame({"station_id": [], "datetime": [], "total_count": []})

    out = add_forecast_target(df)

    assert out.empty
    assert "target_total_count" in out.columns


def test_summarize_target_nulls_reports_fraction() -> None:
    df = pd.DataFrame({"target_total_count": [1.0, None, 3.0, None]})

    summary = summarize_target_nulls(df)

    assert summary == {"n_rows": 4, "n_null_target": 2, "pct_null_target": 50.0}


def test_summarize_target_nulls_raises_on_missing_column() -> None:
    with pytest.raises(ModelTableError):
        summarize_target_nulls(pd.DataFrame({"x": [1]}))


# ---------------------------------------------------------------------------
# add_calendar_features
# ---------------------------------------------------------------------------


def test_add_calendar_features_adds_expected_columns() -> None:
    # 2024-11-12 is a Tuesday within WS2024/25's lecture period.
    # 2024-01-01 is a German public holiday (Neujahr) and also falls inside
    # WS2023/24's lecture-period window (09.10.2023-02.02.2024) - the
    # semester table does not model the inner Christmas/New Year recess
    # (see semester_dates module docstring), so this is expected True.
    # 2024-08-15 falls after SS2024's lecture period ends (19.07.2024) and
    # before WS2024/25's starts (07.10.2024) - squarely a semester break -
    # and is also inside a made-up school-holiday range.
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-01-01 08:00", "2024-11-12 14:30", "2024-08-15 09:00"]
            )
        }
    )
    public_holidays = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])})
    school_holidays = pd.DataFrame(
        {
            "start_date": pd.to_datetime(["2024-07-08"]),
            "end_date": pd.to_datetime(["2024-08-20"]),
        }
    )

    out = add_calendar_features(df, public_holidays, school_holidays)

    assert list(out["hour"]) == [8, 14, 9]
    assert list(out["month"]) == [1, 11, 8]
    assert list(out["day_of_week"]) == [0, 1, 3]  # Mon, Tue, Thu
    assert list(out["is_public_holiday"]) == [True, False, False]
    assert list(out["is_school_holiday"]) == [False, False, True]
    assert list(out["is_lecture_period"]) == [True, True, False]


def test_add_calendar_features_raises_on_missing_timestamp_column() -> None:
    with pytest.raises(ModelTableError):
        add_calendar_features(
            pd.DataFrame({"x": [1]}),
            pd.DataFrame({"date": pd.to_datetime(["2024-01-01"])}),
            pd.DataFrame(
                {
                    "start_date": pd.to_datetime(["2024-01-01"]),
                    "end_date": pd.to_datetime(["2024-01-02"]),
                }
            ),
        )


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------


def test_chronological_split_uses_a_single_global_cutoff_across_stations() -> None:
    # Station A has 10 days of data, station B only 5 - the cutoff is
    # derived from the *global* max timestamp, applied identically to both,
    # not a per-station cutoff.
    dates_a = pd.date_range("2024-01-01", periods=10, freq="D")
    dates_b = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.concat(
        [
            pd.DataFrame({"station_id": "A", "datetime": dates_a}),
            pd.DataFrame({"station_id": "B", "datetime": dates_b}),
        ],
        ignore_index=True,
    )

    train, test, cutoff = chronological_split(df, test_period=pd.Timedelta(days=3))

    assert cutoff == pd.Timestamp("2024-01-10") - pd.Timedelta(days=3)
    assert (train["datetime"] < cutoff).all()
    assert (test["datetime"] >= cutoff).all()
    # Station B never reaches the cutoff -> zero test rows for B.
    assert (test["station_id"] == "B").sum() == 0
    assert (train["station_id"] == "B").sum() == 5


def test_chronological_split_embargoes_rows_whose_target_reaches_into_test() -> None:
    # Hourly data; default embargo (24h, matching add_forecast_target's
    # default horizon) must exclude rows in [cutoff - 24h, cutoff) from
    # train, since add_forecast_target would label them from inside test.
    dates = pd.date_range("2024-01-01", periods=24 * 5, freq="h")
    df = pd.DataFrame({"station_id": "A", "datetime": dates})

    train, test, cutoff = chronological_split(df, test_period=pd.Timedelta(days=2))

    assert (train["datetime"] < cutoff - pd.Timedelta(hours=24)).all()
    assert (test["datetime"] >= cutoff).all()
    embargoed = df[
        (df["datetime"] >= cutoff - pd.Timedelta(hours=24)) & (df["datetime"] < cutoff)
    ]
    assert len(embargoed) > 0
    assert not embargoed["datetime"].isin(train["datetime"]).any()
    assert not embargoed["datetime"].isin(test["datetime"]).any()


def test_chronological_split_embargo_zero_reproduces_pre_embargo_behavior() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    df = pd.DataFrame({"station_id": "A", "datetime": dates})

    train, test, cutoff = chronological_split(
        df, test_period=pd.Timedelta(days=3), embargo=pd.Timedelta(0)
    )

    assert len(train) + len(test) == len(df)
    assert (train["datetime"] < cutoff).all()
    assert (test["datetime"] >= cutoff).all()


def test_chronological_split_raises_on_empty_dataframe() -> None:
    with pytest.raises(ModelTableError):
        chronological_split(pd.DataFrame({"datetime": pd.to_datetime([])}))


def test_chronological_split_raises_on_missing_column() -> None:
    with pytest.raises(ModelTableError):
        chronological_split(pd.DataFrame({"x": [1]}))


# ---------------------------------------------------------------------------
# add_baseline_prediction / summarize_baseline_evaluable_rows / compute_baseline_metrics
# ---------------------------------------------------------------------------


def test_add_baseline_prediction_copies_current_value() -> None:
    df = pd.DataFrame({"total_count": [1.0, 2.0, None]})
    out = add_baseline_prediction(df)
    assert out["baseline_prediction"].iloc[0] == 1.0
    assert out["baseline_prediction"].iloc[1] == 2.0
    assert pd.isna(out["baseline_prediction"].iloc[2])


def test_summarize_baseline_evaluable_rows_counts_exclusions() -> None:
    df = pd.DataFrame(
        {
            "baseline_prediction": [1.0, None, 3.0, 4.0],
            "target_total_count": [1.0, 2.0, None, 5.0],
        }
    )
    summary = summarize_baseline_evaluable_rows(df)
    assert summary == {
        "n_rows": 4,
        "n_evaluable": 2,
        "n_excluded": 2,
        "pct_excluded": 50.0,
    }


def test_compute_baseline_metrics_overall() -> None:
    df = pd.DataFrame(
        {
            "baseline_prediction": [10.0, 20.0],
            "target_total_count": [12.0, 16.0],
        }
    )
    result = compute_baseline_metrics(df)

    assert result["group"].tolist() == ["overall"]
    assert result["mae"].iloc[0] == pytest.approx((2 + 4) / 2)
    assert result["rmse"].iloc[0] == pytest.approx(np.sqrt((4 + 16) / 2))
    assert result["n_rows"].iloc[0] == 2


def test_compute_baseline_metrics_per_group_excludes_null_rows() -> None:
    df = pd.DataFrame(
        {
            "station_id": ["A", "A", "B", "B"],
            "baseline_prediction": [10.0, 20.0, 5.0, None],
            "target_total_count": [12.0, 16.0, 5.0, 9.0],
        }
    )
    result = compute_baseline_metrics(df, group_col="station_id")

    result = result.set_index("group")
    assert result.loc["A", "n_rows"] == 2
    assert result.loc["B", "n_rows"] == 1
    assert result.loc["B", "mae"] == pytest.approx(0.0)


def test_compute_baseline_metrics_raises_when_no_evaluable_rows() -> None:
    df = pd.DataFrame({"baseline_prediction": [None], "target_total_count": [None]})
    with pytest.raises(ModelTableError):
        compute_baseline_metrics(df)


def test_compute_baseline_metrics_raises_on_missing_column() -> None:
    with pytest.raises(ModelTableError):
        compute_baseline_metrics(pd.DataFrame({"x": [1]}))

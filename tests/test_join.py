"""Tests for `muenster_bike_forecast.data.join`.

All tests use small, hand-built sample data spanning DST transitions and a
deliberate weather gap — no live network calls and no dependency on the
real `data/raw/` files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from muenster_bike_forecast.data.join import (
    JoinError,
    combine_weather_parameters,
    join_station_weather,
    localize_bike_timestamps,
    summarize_dst_edge_cases,
    summarize_weather_coverage,
)

# ---------------------------------------------------------------------------
# localize_bike_timestamps / summarize_dst_edge_cases
# ---------------------------------------------------------------------------


def test_localize_bike_timestamps_converts_ordinary_time_to_utc() -> None:
    # 2024-01-01 is CET (UTC+1), no DST.
    df = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-01 12:00"])})

    out = localize_bike_timestamps(df)

    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 11:00", tz="UTC")


def test_localize_bike_timestamps_marks_spring_forward_gap_as_nat() -> None:
    # 2024-03-31 02:00-03:00 Europe/Berlin local time does not exist
    # (clocks jump from 02:00 to 03:00).
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-03-31 01:45", "2024-03-31 02:30", "2024-03-31 03:15"]
            )
        }
    )

    out = localize_bike_timestamps(df)

    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-03-31 00:45", tz="UTC")
    assert pd.isna(out["timestamp"].iloc[1])
    assert out["timestamp"].iloc[2] == pd.Timestamp("2024-03-31 01:15", tz="UTC")


def test_localize_bike_timestamps_marks_fall_back_ambiguous_time_as_nat() -> None:
    # 2024-10-27 02:00-03:00 Europe/Berlin local time occurs twice
    # (clocks fall back from 03:00 to 02:00).
    df = pd.DataFrame({"datetime": pd.to_datetime(["2024-10-27 02:30"])})

    out = localize_bike_timestamps(df)

    assert pd.isna(out["timestamp"].iloc[0])


def test_localize_bike_timestamps_raises_on_missing_column() -> None:
    df = pd.DataFrame({"not_datetime": [1, 2]})
    with pytest.raises(JoinError):
        localize_bike_timestamps(df)


def test_localize_bike_timestamps_raises_on_already_aware_column() -> None:
    df = pd.DataFrame({"datetime": pd.to_datetime(["2024-01-01 12:00"], utc=True)})
    with pytest.raises(JoinError):
        localize_bike_timestamps(df)


def test_summarize_dst_edge_cases_counts_nat_rows() -> None:
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-03-31 01:45", "2024-03-31 02:30", "2024-10-27 02:30"]
            )
        }
    )
    localized = localize_bike_timestamps(df)

    summary = summarize_dst_edge_cases(localized)

    assert summary["n_rows"] == 3
    assert summary["n_dst_edge_case"] == 2
    assert summary["pct_dst_edge_case"] == pytest.approx(200 / 3)


def test_summarize_dst_edge_cases_raises_on_missing_column() -> None:
    df = pd.DataFrame({"not_timestamp": [1, 2]})
    with pytest.raises(JoinError):
        summarize_dst_edge_cases(df)


# ---------------------------------------------------------------------------
# combine_weather_parameters
# ---------------------------------------------------------------------------


def _weather_frame(value_col: str, values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["01766"] * len(values),
            "timestamp": pd.date_range(
                "2024-01-01", periods=len(values), freq="h", tz="UTC"
            ),
            "quality_level": [3] * len(values),
            value_col: values,
        }
    )


def test_combine_weather_parameters_joins_on_timestamp() -> None:
    frames = {
        "air_temperature": _weather_frame("air_temperature_c", [1.0, 2.0]),
        "precipitation": _weather_frame("precipitation_mm", [0.0, 0.5]),
        "wind": _weather_frame("wind_speed_ms", [3.0, 4.0]),
    }

    combined = combine_weather_parameters(frames)

    assert len(combined) == 2
    assert (combined["station_id"] == "01766").all()
    assert list(combined["air_temperature_c"]) == [1.0, 2.0]
    assert list(combined["precipitation_mm"]) == [0.0, 0.5]
    assert list(combined["wind_speed_ms"]) == [3.0, 4.0]


def test_combine_weather_parameters_prefixes_colliding_column_names() -> None:
    # Every parameter frame has its own "quality_level" column - this must
    # not silently overwrite the others.
    frames = {
        "air_temperature": _weather_frame("air_temperature_c", [1.0]),
        "precipitation": _weather_frame("precipitation_mm", [0.0]),
    }

    combined = combine_weather_parameters(frames)

    assert "quality_level" in combined.columns  # first parameter, unprefixed
    assert "precipitation_quality_level" in combined.columns


def test_combine_weather_parameters_outer_joins_mismatched_ranges() -> None:
    short = _weather_frame("air_temperature_c", [1.0])  # only 2024-01-01 00:00
    long = _weather_frame("wind_speed_ms", [3.0, 4.0, 5.0])  # 3 hours

    combined = combine_weather_parameters({"air_temperature": short, "wind": long})

    assert len(combined) == 3
    assert combined["air_temperature_c"].isna().sum() == 2


def test_combine_weather_parameters_raises_on_empty_input() -> None:
    with pytest.raises(JoinError):
        combine_weather_parameters({})


def test_combine_weather_parameters_raises_on_mismatched_station_ids() -> None:
    frame_a = _weather_frame("air_temperature_c", [1.0])
    frame_b = _weather_frame("wind_speed_ms", [3.0])
    frame_b["station_id"] = "99999"

    with pytest.raises(JoinError):
        combine_weather_parameters({"air_temperature": frame_a, "wind": frame_b})


def test_combine_weather_parameters_raises_on_missing_column() -> None:
    frame = pd.DataFrame({"station_id": ["01766"], "not_timestamp": [1]})
    with pytest.raises(JoinError):
        combine_weather_parameters({"air_temperature": frame})


# ---------------------------------------------------------------------------
# join_station_weather / summarize_weather_coverage
# ---------------------------------------------------------------------------


def _bike_df(timestamps_utc: list[str], counts: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["S1"] * len(timestamps_utc),
            "timestamp": pd.to_datetime(timestamps_utc, utc=True),
            "count": counts,
        }
    )


def _weather_df(timestamps_utc: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": ["01766"] * len(timestamps_utc),
            "timestamp": pd.to_datetime(timestamps_utc, utc=True),
            "air_temperature_c": values,
        }
    )


def test_join_station_weather_matches_most_recent_past_reading() -> None:
    bike = _bike_df(["2024-01-01 01:15+00:00"], [10])
    weather = _weather_df(
        ["2024-01-01 01:00+00:00", "2024-01-01 02:00+00:00"], [1.0, 2.0]
    )

    joined = join_station_weather(bike, weather)

    assert joined["weather_air_temperature_c"].iloc[0] == 1.0
    assert joined["weather_timestamp"].iloc[0] == pd.Timestamp(
        "2024-01-01 01:00", tz="UTC"
    )


def test_join_station_weather_never_matches_a_future_reading() -> None:
    # Bike row is right before the only weather reading - must stay
    # unmatched (null), not "round" to the nearby future value.
    bike = _bike_df(["2024-01-01 00:59:00+00:00"], [10])
    weather = _weather_df(["2024-01-01 01:00:00+00:00"], [1.0])

    joined = join_station_weather(bike, weather, tolerance=pd.Timedelta(hours=2))

    assert pd.isna(joined["weather_air_temperature_c"].iloc[0])


def test_join_station_weather_respects_tolerance_across_a_gap() -> None:
    # Weather gap between 01:00 and 05:00. A bike row at 04:00 is 3 hours
    # past the last reading before the gap - outside a 2-hour tolerance -
    # so it must be left unmatched rather than matched to stale data.
    bike = _bike_df(["2024-01-01 02:30:00+00:00", "2024-01-01 04:00:00+00:00"], [1, 2])
    weather = _weather_df(
        ["2024-01-01 01:00:00+00:00", "2024-01-01 05:00:00+00:00"], [1.0, 5.0]
    )

    joined = join_station_weather(bike, weather, tolerance=pd.Timedelta(hours=2))

    # 02:30 is 1.5h after 01:00 -> within tolerance -> matched.
    assert joined["weather_air_temperature_c"].iloc[0] == 1.0
    # 04:00 is 3h after 01:00 -> outside tolerance -> unmatched.
    assert pd.isna(joined["weather_air_temperature_c"].iloc[1])


def test_join_station_weather_leaves_dst_edge_case_rows_unmatched() -> None:
    bike_naive = pd.DataFrame(
        {
            "station_id": ["S1"],
            "datetime": pd.to_datetime(["2024-03-31 02:30"]),  # nonexistent local time
        }
    )
    bike = localize_bike_timestamps(bike_naive)
    weather = _weather_df(["2024-03-31 00:00:00+00:00"], [1.0])

    joined = join_station_weather(bike, weather)

    assert pd.isna(joined["timestamp"].iloc[0])
    assert pd.isna(joined["weather_air_temperature_c"].iloc[0])


def test_join_station_weather_keeps_numeric_dtype_when_mixing_matched_and_unmatched_rows() -> (
    None
):
    """Regression test: mixing matched/unmatched rows must not degrade dtype.

    A prior implementation assigned `pd.NA` directly to the unmatched
    half's weather columns, which forced them to `object` dtype - and
    `pd.concat` with the matched half (float64) then silently degraded
    the *whole* combined column to `object`, even for genuinely-matched
    numeric values. Downstream consumers that don't tolerate `object`
    dtype (e.g. `LGBMRegressor.predict`, which requires int/float/bool)
    would fail on this, even though every individual value was still
    numerically correct.
    """
    bike_naive = pd.DataFrame(
        {
            "station_id": ["S1", "S1"],
            "datetime": pd.to_datetime(
                [
                    "2024-01-01 09:00",
                    "2024-03-31 02:30",
                ]  # one normal, one DST edge case
            ),
        }
    )
    bike = localize_bike_timestamps(bike_naive)
    weather = _weather_df(["2024-01-01 08:00:00+00:00"], [15.0])

    joined = join_station_weather(bike, weather)

    assert joined["weather_air_temperature_c"].dtype == np.dtype("float64")
    assert joined["weather_air_temperature_c"].iloc[0] == 15.0
    assert pd.isna(joined["weather_air_temperature_c"].iloc[1])


def test_join_station_weather_preserves_original_row_order() -> None:
    bike = _bike_df(
        [
            "2024-01-01 03:00:00+00:00",
            "2024-01-01 01:00:00+00:00",
            "2024-01-01 02:00:00+00:00",
        ],
        [30, 10, 20],
    )
    weather = _weather_df(
        ["2024-01-01 01:00:00+00:00", "2024-01-01 02:00:00+00:00"], [1.0, 2.0]
    )

    joined = join_station_weather(bike, weather)

    assert list(joined["count"]) == [30, 10, 20]


def test_join_station_weather_raises_on_missing_column() -> None:
    bike = pd.DataFrame({"not_timestamp": [1]})
    weather = _weather_df(["2024-01-01 00:00:00+00:00"], [1.0])
    with pytest.raises(JoinError):
        join_station_weather(bike, weather)


def test_join_station_weather_raises_on_empty_input() -> None:
    bike = _bike_df([], [])
    weather = _weather_df(["2024-01-01 00:00:00+00:00"], [1.0])
    with pytest.raises(JoinError):
        join_station_weather(bike, weather)


def test_summarize_weather_coverage_reports_missing_fraction() -> None:
    bike = _bike_df(["2024-01-01 01:00:00+00:00", "2024-01-01 10:00:00+00:00"], [1, 2])
    weather = _weather_df(["2024-01-01 01:00:00+00:00"], [1.0])

    joined = join_station_weather(bike, weather, tolerance=pd.Timedelta(hours=2))
    summary = summarize_weather_coverage(joined, station_id="S1")

    assert summary["station_id"] == "S1"
    assert summary["n_rows"] == 2
    assert summary["n_missing_weather"] == 1
    assert summary["pct_missing_weather"] == pytest.approx(50.0)


def test_summarize_weather_coverage_raises_on_empty_input() -> None:
    with pytest.raises(JoinError):
        summarize_weather_coverage(pd.DataFrame(columns=["weather_timestamp"]))


def test_summarize_weather_coverage_raises_on_missing_column() -> None:
    df = pd.DataFrame({"count": [1, 2]})
    with pytest.raises(JoinError):
        summarize_weather_coverage(df)

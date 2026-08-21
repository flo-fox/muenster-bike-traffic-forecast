"""Tests for `muenster_bike_forecast.daily_report`.

Small hand-built synthetic data, same style as `tests/test_inference.py` -
no live network calls, no real Anthropic API calls (a fake client double
stands in for `generate_explanation`).
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from muenster_bike_forecast import daily_report, inference
from muenster_bike_forecast.data.bike_counts import Station
from muenster_bike_forecast.data.join import localize_bike_timestamps

STATION_ID = 12345


class _FakeModel:
    """`.predict` triples each row's `total_count` - lets tests assert
    exact expected predictions instead of depending on a real model."""

    def predict(self, X: pd.DataFrame):
        return (X["total_count"] * 3).to_numpy()


def _raw_bike_df() -> pd.DataFrame:
    """Two readings 24h apart (10, then 20) for one station."""
    return pd.DataFrame(
        {
            "station_id": [str(STATION_ID)] * 2,
            "datetime": pd.to_datetime(["2024-06-10 08:00", "2024-06-11 08:00"]),
            f"{STATION_ID} (Test)": [10, 20],
            f"{STATION_ID}-status": [0, 0],
        }
    )


def _weather_wide_df(bike_datetimes: pd.Series) -> pd.DataFrame:
    localized = localize_bike_timestamps(pd.DataFrame({"datetime": bike_datetimes}))[
        "timestamp"
    ]
    return pd.DataFrame(
        {
            "station_id": ["01766"] * len(localized),
            "timestamp": localized,
            "air_temperature_c": [15.0, 16.0][: len(localized)],
            "relative_humidity_pct": [70.0, 71.0][: len(localized)],
            "precipitation_mm": [0.0, 2.0][: len(localized)],
            "wind_speed_ms": [3.0, 3.5][: len(localized)],
        }
    )


def _public_holidays_df() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime([]), "name": pd.Series(dtype="object")})


def _school_holidays_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"start_date": pd.to_datetime([]), "end_date": pd.to_datetime([])}
    )


def _ratio_table() -> pd.DataFrame:
    return pd.DataFrame({"station_id": [STATION_ID], "weekend_weekday_ratio": [0.65]})


def _feature_history(times: list[str], total_counts: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": [STATION_ID] * len(times),
            "datetime": pd.to_datetime(times),
            "total_count": total_counts,
        }
    )


def _dummy_context(dt: str = "2024-06-10 08:00") -> daily_report.RowContext:
    return daily_report.RowContext(
        datetime=pd.Timestamp(dt),
        hour=8,
        day_of_week=0,
        is_public_holiday=False,
        is_school_holiday=False,
        is_lecture_period=False,
        weather_air_temperature_c=15.0,
        weather_precipitation_mm=0.0,
        weather_wind_speed_ms=3.0,
        weather_relative_humidity_pct=70.0,
    )


def _report(
    station_id: str,
    abs_error: float | None,
    pct_diff: float = 10.0,
    name: str | None = None,
    data_age: pd.Timedelta = pd.Timedelta(hours=1),
    prediction_timing_gap: pd.Timedelta = pd.Timedelta(0),
) -> daily_report.StationAccuracyReport:
    resolved = abs_error is not None
    return daily_report.StationAccuracyReport(
        station_id=station_id,
        station_name=name or f"Station {station_id}",
        now_datetime=pd.Timestamp("2024-06-11 08:00"),
        data_age=data_age,
        actual_now=20.0,
        forecast_24h_ahead=25.0,
        now_context=_dummy_context("2024-06-11 08:00"),
        predicted_for_now=20.0 + abs_error if resolved else None,
        abs_error=abs_error,
        pct_diff=pct_diff if resolved else None,
        prediction_basis_context=(
            _dummy_context("2024-06-10 08:00") if resolved else None
        ),
        prediction_timing_gap=prediction_timing_gap if resolved else None,
        unresolved_reason=None if resolved else "no bike-count reading near 24h ago",
    )


# ---------------------------------------------------------------------------
# RowContext
# ---------------------------------------------------------------------------


def test_row_context_from_feature_row_maps_fields_and_handles_missing_weather() -> None:
    row = pd.Series(
        {
            "datetime": pd.Timestamp("2024-06-10 08:00"),
            "hour": 8,
            "day_of_week": 0,
            "is_public_holiday": True,
            "is_school_holiday": False,
            "is_lecture_period": True,
            "weather_air_temperature_c": 15.5,
            "weather_precipitation_mm": float("nan"),
            "weather_wind_speed_ms": 3.0,
            "weather_relative_humidity_pct": 70.0,
        }
    )
    context = daily_report.RowContext.from_feature_row(row)
    assert context.hour == 8
    assert context.is_public_holiday is True
    assert context.weather_air_temperature_c == 15.5
    assert context.weather_precipitation_mm is None


# ---------------------------------------------------------------------------
# select_row_near
# ---------------------------------------------------------------------------


def test_select_row_near_picks_closest_within_tolerance() -> None:
    history = _feature_history(
        ["2024-06-10 07:40", "2024-06-10 08:15", "2024-06-10 09:30"], [1.0, 2.0, 3.0]
    )
    row = daily_report.select_row_near(
        history, STATION_ID, pd.Timestamp("2024-06-10 08:00")
    )
    assert row["datetime"] == pd.Timestamp("2024-06-10 08:15")


def test_select_row_near_raises_when_nothing_within_tolerance() -> None:
    history = _feature_history(["2024-06-10 05:00"], [1.0])
    with pytest.raises(inference.InferenceError):
        daily_report.select_row_near(
            history, STATION_ID, pd.Timestamp("2024-06-10 08:00")
        )


def test_select_row_near_raises_when_no_evaluable_rows() -> None:
    history = _feature_history(["2024-06-10 08:00"], [float("nan")])
    with pytest.raises(inference.InferenceError):
        daily_report.select_row_near(
            history, STATION_ID, pd.Timestamp("2024-06-10 08:00")
        )


# ---------------------------------------------------------------------------
# build_station_report
# ---------------------------------------------------------------------------


def test_build_station_report_computes_accuracy_and_forecast() -> None:
    station = Station(
        station_id=str(STATION_ID), name="Test Station", start_year=2020, channels=()
    )
    raw = _raw_bike_df()
    weather = _weather_wide_df(raw["datetime"])

    report = daily_report.build_station_report(
        station,
        _FakeModel(),
        _ratio_table(),
        _public_holidays_df(),
        _school_holidays_df(),
        weather,
        raw,
    )

    assert report.station_name == "Test Station"
    assert report.actual_now == 20.0
    assert report.forecast_24h_ahead == pytest.approx(60.0)  # 20 * 3
    assert report.predicted_for_now == pytest.approx(30.0)  # 10 * 3
    assert report.abs_error == pytest.approx(10.0)  # |30 - 20|
    assert report.pct_diff == pytest.approx(50.0)  # (30 - 20) / 20 * 100
    assert report.unresolved_reason is None
    assert report.prediction_basis_context is not None


def test_build_station_report_flags_stale_data() -> None:
    # _raw_bike_df's fixed 2024 dates are always far in the past relative
    # to whenever this test actually runs, so is_stale must be True.
    station = Station(
        station_id=str(STATION_ID), name="Test Station", start_year=2020, channels=()
    )
    raw = _raw_bike_df()
    weather = _weather_wide_df(raw["datetime"])

    report = daily_report.build_station_report(
        station,
        _FakeModel(),
        _ratio_table(),
        _public_holidays_df(),
        _school_holidays_df(),
        weather,
        raw,
    )

    assert report.is_stale is True
    assert report.data_age > daily_report.STALENESS_WARNING_THRESHOLD


def test_build_station_report_computes_timing_gap_when_basis_row_is_offset() -> None:
    # Basis reading is 20 minutes earlier than exactly "24h before now".
    station = Station(
        station_id=str(STATION_ID), name="Test Station", start_year=2020, channels=()
    )
    raw = pd.DataFrame(
        {
            "station_id": [str(STATION_ID)] * 2,
            "datetime": pd.to_datetime(["2024-06-10 07:40", "2024-06-11 08:00"]),
            f"{STATION_ID} (Test)": [10, 20],
            f"{STATION_ID}-status": [0, 0],
        }
    )
    weather = _weather_wide_df(raw["datetime"])

    report = daily_report.build_station_report(
        station,
        _FakeModel(),
        _ratio_table(),
        _public_holidays_df(),
        _school_holidays_df(),
        weather,
        raw,
    )

    assert report.prediction_timing_gap == pd.Timedelta(minutes=20)


def test_build_station_report_degrades_gracefully_without_yesterday_row() -> None:
    # Only one reading, so there is nothing near "24h ago" - the station's
    # today's forecast should still be computed.
    station = Station(
        station_id=str(STATION_ID), name="Test Station", start_year=2020, channels=()
    )
    raw = _raw_bike_df().iloc[[1]]  # keep only the "now" reading
    weather = _weather_wide_df(raw["datetime"])

    report = daily_report.build_station_report(
        station,
        _FakeModel(),
        _ratio_table(),
        _public_holidays_df(),
        _school_holidays_df(),
        weather,
        raw,
    )

    assert report.forecast_24h_ahead == pytest.approx(60.0)
    assert report.predicted_for_now is None
    assert report.abs_error is None
    assert report.pct_diff is None
    assert report.prediction_basis_context is None
    assert report.unresolved_reason is not None


# ---------------------------------------------------------------------------
# select_top_deviations
# ---------------------------------------------------------------------------


def test_select_top_deviations_sorts_by_abs_error_desc_and_respects_n() -> None:
    reports = [
        _report("1", 5.0),
        _report("2", 20.0),
        _report("3", 10.0),
        _report("4", None),
    ]
    top = daily_report.select_top_deviations(reports, n=2)
    assert [r.station_id for r in top] == ["2", "3"]


def test_select_top_deviations_tiebreaks_by_station_id() -> None:
    reports = [_report("3", 10.0), _report("1", 10.0), _report("2", 10.0)]
    top = daily_report.select_top_deviations(reports, n=3)
    assert [r.station_id for r in top] == ["1", "2", "3"]


def test_select_top_deviations_excludes_none_abs_error() -> None:
    reports = [_report("1", None), _report("2", 5.0)]
    top = daily_report.select_top_deviations(reports)
    assert [r.station_id for r in top] == ["2"]


# ---------------------------------------------------------------------------
# build_explanation_prompt
# ---------------------------------------------------------------------------


def test_build_explanation_prompt_includes_report_numbers() -> None:
    report = _report("300038855", 10.0, pct_diff=50.0, name="Bismarckallee")
    system, user = daily_report.build_explanation_prompt(report)
    assert "Bismarckallee" in user
    assert "300038855" in user
    assert "+50.0%" in user
    assert "invent" in system.lower()


def test_build_explanation_prompt_omits_caveats_when_fresh_and_exact() -> None:
    report = _report("1", 10.0, name="Station A")  # fresh, zero timing gap
    _, user = daily_report.build_explanation_prompt(report)
    assert "Caveat" not in user


def test_build_explanation_prompt_includes_staleness_caveat_when_stale() -> None:
    report = _report("1", 10.0, name="Station A", data_age=pd.Timedelta(days=2))
    _, user = daily_report.build_explanation_prompt(report)
    assert "Caveat" in user
    assert "2 days" in user


def test_build_explanation_prompt_includes_timing_gap_caveat_when_notable() -> None:
    report = _report(
        "1", 10.0, name="Station A", prediction_timing_gap=pd.Timedelta(minutes=30)
    )
    _, user = daily_report.build_explanation_prompt(report)
    assert "Caveat" in user
    assert "offset" in user


# ---------------------------------------------------------------------------
# format_email_subject / format_email_body
# ---------------------------------------------------------------------------


def test_format_email_subject_includes_date() -> None:
    subject = daily_report.format_email_subject(date(2026, 8, 19))
    assert subject == "Bike traffic forecast report - 2026-08-19"


def test_format_email_body_lists_every_station_and_flags_only_selected() -> None:
    station_a = _report("1", 5.0, name="Station A")
    station_b = _report("2", 20.0, name="Station B")
    body = daily_report.format_email_body(
        date(2026, 8, 19),
        [station_a, station_b],
        [station_b],
        {"2": "It rained a lot."},
        ["Station C"],
    )
    assert "Station A" in body
    assert "Station B" in body
    assert body.count("It rained a lot.") == 1
    assert "Station C" in body


def test_format_email_body_shows_na_and_reason_for_unresolved_station() -> None:
    unresolved = _report("1", None, name="Station A")
    body = daily_report.format_email_body(date(2026, 8, 19), [unresolved], [], {}, [])
    assert "n/a" in body
    assert "no bike-count reading near 24h ago" in body


def test_format_email_body_flags_stale_station_in_table_and_summary() -> None:
    stale = _report("1", 5.0, name="Station A", data_age=pd.Timedelta(days=2))
    body = daily_report.format_email_body(date(2026, 8, 19), [stale], [stale], {}, [])
    assert "STALE" in body
    assert "1 stale" in body
    assert "Note:" in body  # per-station staleness note in "Notable deviations"


def test_format_email_body_flags_notable_timing_gap() -> None:
    offset = _report(
        "1", 5.0, name="Station A", prediction_timing_gap=pd.Timedelta(minutes=30)
    )
    body = daily_report.format_email_body(date(2026, 8, 19), [offset], [], {}, [])
    assert "offset by" in body


def test_format_email_body_omits_timing_gap_note_within_grid_tolerance() -> None:
    exact = _report(
        "1", 5.0, name="Station A", prediction_timing_gap=pd.Timedelta(minutes=5)
    )
    body = daily_report.format_email_body(date(2026, 8, 19), [exact], [], {}, [])
    assert "offset by" not in body


# ---------------------------------------------------------------------------
# format_email_body_html
# ---------------------------------------------------------------------------


def test_format_email_body_html_lists_every_station_and_flags_only_selected() -> None:
    station_a = _report("1", 5.0, name="Station A")
    station_b = _report("2", 20.0, name="Station B")
    body = daily_report.format_email_body_html(
        date(2026, 8, 19),
        [station_a, station_b],
        [station_b],
        {"2": "It rained a lot."},
        ["Station C"],
    )
    assert "<html>" in body
    assert "Station A" in body
    assert "Station B" in body
    assert body.count("It rained a lot.") == 1
    assert "Station C" in body


def test_format_email_body_html_shows_na_and_reason_for_unresolved_station() -> None:
    unresolved = _report("1", None, name="Station A")
    body = daily_report.format_email_body_html(
        date(2026, 8, 19), [unresolved], [], {}, []
    )
    assert "n/a" in body
    assert "no bike-count reading near 24h ago" in body


def test_format_email_body_html_flags_stale_station_in_table_and_summary() -> None:
    stale = _report("1", 5.0, name="Station A", data_age=pd.Timedelta(days=2))
    body = daily_report.format_email_body_html(
        date(2026, 8, 19), [stale], [stale], {}, []
    )
    assert "STALE" in body
    assert "1 stale" in body
    assert "Note:" in body


def test_format_email_body_html_escapes_untrusted_text() -> None:
    # station_name and the AI explanation are both rendered verbatim into
    # HTML - a literal "<script>" must come out escaped, not executable.
    malicious = _report("1", 5.0, name="<script>alert(1)</script>")
    body = daily_report.format_email_body_html(
        date(2026, 8, 19),
        [malicious],
        [malicious],
        {"1": "<img src=x onerror=alert(1)>"},
        [],
    )
    assert "<script>" not in body
    assert "&lt;script&gt;" in body
    assert "<img src=x onerror=alert(1)>" not in body
    assert "&lt;img" in body


# ---------------------------------------------------------------------------
# generate_explanation
# ---------------------------------------------------------------------------


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> _FakeResponse:
        self.last_kwargs = kwargs
        return _FakeResponse("  A short explanation.  ")


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


def test_generate_explanation_extracts_text_and_passes_model_and_prompt() -> None:
    report = _report("1", 10.0, name="Station A")
    client = _FakeClient()

    result = daily_report.generate_explanation(client, report, model="claude-haiku-4-5")

    assert result == "A short explanation."
    assert client.messages.last_kwargs["model"] == "claude-haiku-4-5"
    assert "Station A" in client.messages.last_kwargs["messages"][0]["content"]

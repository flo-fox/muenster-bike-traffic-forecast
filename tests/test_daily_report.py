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


def _raw_bike_df_spanning_49h(count: float = 10.0) -> pd.DataFrame:
    """49 hourly readings (a constant `count`) for one station, spanning 48h.

    `build_station_report` needs source rows reaching back ~48h from "now"
    to reconstruct both "today's forecast" (last 24h of rows) and
    "yesterday's forecast for the last 24h" (the 24h of rows before that) -
    this fixture provides exactly that.
    """
    times = pd.date_range("2024-06-10 00:00", periods=49, freq="h")
    return pd.DataFrame(
        {
            "station_id": [str(STATION_ID)] * len(times),
            "datetime": times,
            f"{STATION_ID} (Test)": [count] * len(times),
            f"{STATION_ID}-status": [0] * len(times),
        }
    )


def _weather_wide_df(bike_datetimes: pd.Series) -> pd.DataFrame:
    localized = localize_bike_timestamps(pd.DataFrame({"datetime": bike_datetimes}))[
        "timestamp"
    ]
    n = len(localized)
    return pd.DataFrame(
        {
            "station_id": ["01766"] * n,
            "timestamp": localized,
            "air_temperature_c": [15.0] * n,
            "relative_humidity_pct": [70.0] * n,
            "precipitation_mm": [0.0] * n,
            "wind_speed_ms": [3.0] * n,
        }
    )


def _public_holidays_df() -> pd.DataFrame:
    return pd.DataFrame({"date": pd.to_datetime([]), "name": pd.Series(dtype="object")})


def _school_holidays_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"start_date": pd.to_datetime([]), "end_date": pd.to_datetime([])}
    )


def _dummy_window_context(
    window_start: str = "2024-06-10 08:00", window_end: str = "2024-06-11 08:00"
) -> daily_report.WindowContext:
    return daily_report.WindowContext(
        window_start=pd.Timestamp(window_start),
        window_end=pd.Timestamp(window_end),
        day_of_week=0,
        is_public_holiday=False,
        is_school_holiday=False,
        is_lecture_period=False,
        mean_air_temperature_c=15.0,
        total_precipitation_mm=0.0,
        mean_wind_speed_ms=3.0,
        mean_relative_humidity_pct=70.0,
    )


def _dummy_forecast_summary(total: float = 90.0) -> inference.ForecastSummary:
    return inference.ForecastSummary(
        total_predicted_count=total,
        peak_datetime=pd.Timestamp("2024-06-11 08:15"),
        peak_value=30.0,
    )


def _report(
    station_id: str,
    abs_error: float | None,
    pct_diff: float = 10.0,
    name: str | None = None,
    data_age: pd.Timedelta = pd.Timedelta(hours=1),
) -> daily_report.StationAccuracyReport:
    resolved = abs_error is not None
    return daily_report.StationAccuracyReport(
        station_id=station_id,
        station_name=name or f"Station {station_id}",
        now_datetime=pd.Timestamp("2024-06-11 08:00"),
        data_age=data_age,
        forecast_summary=_dummy_forecast_summary(),
        predicted_total=220.0 + abs_error if resolved else None,
        actual_total=220.0 if resolved else None,
        abs_error=abs_error,
        pct_diff=pct_diff if resolved else None,
        window_context=_dummy_window_context() if resolved else None,
        unresolved_reason=None if resolved else "no bike-count reading near 24h ago",
    )


# ---------------------------------------------------------------------------
# WindowContext
# ---------------------------------------------------------------------------


def test_window_context_from_window_rows_aggregates_weather_and_uses_latest_calendar_flags() -> (
    None
):
    rows = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2024-06-10 08:00", "2024-06-10 09:00", "2024-06-10 10:00"]
            ),
            "day_of_week": [0, 0, 0],
            "is_public_holiday": [False, False, True],
            "is_school_holiday": [False, False, False],
            "is_lecture_period": [True, True, True],
            "weather_air_temperature_c": [10.0, 20.0, float("nan")],
            "weather_precipitation_mm": [1.0, 2.0, 3.0],
            "weather_wind_speed_ms": [2.0, 4.0, 6.0],
            "weather_relative_humidity_pct": [60.0, 70.0, 80.0],
        }
    )

    context = daily_report.WindowContext.from_window_rows(rows)

    assert context.window_start == pd.Timestamp("2024-06-10 08:00")
    assert context.window_end == pd.Timestamp("2024-06-10 10:00")
    # Representative row is the latest one (10:00), which is a holiday.
    assert context.is_public_holiday is True
    assert context.is_lecture_period is True
    assert context.mean_air_temperature_c == pytest.approx(15.0)  # mean of 10, 20
    assert context.total_precipitation_mm == pytest.approx(6.0)  # sum of 1, 2, 3
    assert context.mean_wind_speed_ms == pytest.approx(4.0)
    assert context.mean_relative_humidity_pct == pytest.approx(70.0)


def test_window_context_from_window_rows_handles_all_null_weather_field() -> None:
    rows = pd.DataFrame(
        {
            "datetime": pd.to_datetime(["2024-06-10 08:00"]),
            "day_of_week": [0],
            "is_public_holiday": [False],
            "is_school_holiday": [False],
            "is_lecture_period": [False],
            "weather_air_temperature_c": [float("nan")],
            "weather_precipitation_mm": [float("nan")],
            "weather_wind_speed_ms": [float("nan")],
            "weather_relative_humidity_pct": [float("nan")],
        }
    )

    context = daily_report.WindowContext.from_window_rows(rows)

    assert context.mean_air_temperature_c is None
    assert context.total_precipitation_mm is None


# ---------------------------------------------------------------------------
# build_station_report
# ---------------------------------------------------------------------------


def test_build_station_report_computes_daily_totals() -> None:
    station = Station(
        station_id=str(STATION_ID), name="Test Station", start_year=2020, channels=()
    )
    raw = _raw_bike_df_spanning_49h(count=10.0)
    weather = _weather_wide_df(raw["datetime"])

    report = daily_report.build_station_report(
        station,
        _FakeModel(),
        _public_holidays_df(),
        _school_holidays_df(),
        weather,
        raw,
    )

    assert report.station_name == "Test Station"
    # forecast_summary: last 24h of source rows (24 rows, count=10 each),
    # each tripled by _FakeModel -> 24 * 30 = 720.
    assert report.forecast_summary.total_predicted_count == pytest.approx(720.0)
    # predicted_total: yesterday-anchored curve over the same 24 source
    # rows one window earlier -> also 24 * 30 = 720.
    assert report.predicted_total == pytest.approx(720.0)
    # actual_total: sum of actual total_count over the last 24h window
    # (24 rows, count=10 each) -> 240.
    assert report.actual_total == pytest.approx(240.0)
    assert report.abs_error == pytest.approx(480.0)
    assert report.pct_diff == pytest.approx(200.0)
    assert report.unresolved_reason is None
    assert report.window_context is not None


def test_build_station_report_flags_stale_data() -> None:
    # The fixture's fixed 2024 dates are always far in the past relative to
    # whenever this test actually runs, so is_stale must be True.
    station = Station(
        station_id=str(STATION_ID), name="Test Station", start_year=2020, channels=()
    )
    raw = _raw_bike_df_spanning_49h()
    weather = _weather_wide_df(raw["datetime"])

    report = daily_report.build_station_report(
        station,
        _FakeModel(),
        _public_holidays_df(),
        _school_holidays_df(),
        weather,
        raw,
    )

    assert report.is_stale is True
    assert report.data_age > daily_report.STALENESS_WARNING_THRESHOLD


def test_build_station_report_degrades_gracefully_without_yesterday_window() -> None:
    # Only the last 24h of rows exist, so there's no ~48h-back history to
    # reconstruct yesterday's forecast - today's forecast should still be
    # computed.
    station = Station(
        station_id=str(STATION_ID), name="Test Station", start_year=2020, channels=()
    )
    full = _raw_bike_df_spanning_49h()
    raw = full.iloc[-24:].reset_index(drop=True)  # keep only the last 24h
    weather = _weather_wide_df(raw["datetime"])

    report = daily_report.build_station_report(
        station,
        _FakeModel(),
        _public_holidays_df(),
        _school_holidays_df(),
        weather,
        raw,
    )

    # forecast_summary still resolves: it only needs the last 24h of
    # source rows (24 rows, count=10 each, tripled -> 720), which are all
    # present even with the ~48h-back history missing.
    assert report.forecast_summary.total_predicted_count == pytest.approx(720.0)
    assert report.predicted_total is None
    assert report.actual_total is None
    assert report.abs_error is None
    assert report.pct_diff is None
    assert report.window_context is None
    assert report.unresolved_reason is not None


def test_build_station_report_actual_total_skips_gaps_within_window() -> None:
    # A few missing 15-min readings inside the last-24h window (not at its
    # "now" boundary, which is always non-null by construction) should just
    # be excluded from the sum, not treated as unresolved.
    station = Station(
        station_id=str(STATION_ID), name="Test Station", start_year=2020, channels=()
    )
    raw = _raw_bike_df_spanning_49h()
    # Null out 4 of the 24 readings inside the window, keeping the last
    # ("now") row intact.
    gap_positions = raw.index[-10:-6]
    raw.loc[gap_positions, f"{STATION_ID} (Test)"] = pd.NA
    weather = _weather_wide_df(raw["datetime"])

    report = daily_report.build_station_report(
        station,
        _FakeModel(),
        _public_holidays_df(),
        _school_holidays_df(),
        weather,
        raw,
    )

    assert report.unresolved_reason is None
    # 20 remaining valid readings (count=10 each) instead of 24.
    assert report.actual_total == pytest.approx(200.0)


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
    assert "daily-total" in system.lower()


def test_build_explanation_prompt_omits_caveats_when_fresh() -> None:
    report = _report("1", 10.0, name="Station A")  # fresh
    _, user = daily_report.build_explanation_prompt(report)
    assert "Caveat" not in user


def test_build_explanation_prompt_includes_staleness_caveat_when_stale() -> None:
    report = _report("1", 10.0, name="Station A", data_age=pd.Timedelta(days=2))
    _, user = daily_report.build_explanation_prompt(report)
    assert "Caveat" in user
    assert "2 days" in user


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
    assert "Predicted total (last 24h)" in body
    assert "Predicted total (next 24h)" in body


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


def test_format_email_body_includes_repo_link() -> None:
    body = daily_report.format_email_body(date(2026, 8, 19), [], [], {}, [])
    assert daily_report.PROJECT_REPO_URL in body


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


def test_format_email_body_html_includes_repo_link() -> None:
    body = daily_report.format_email_body_html(date(2026, 8, 19), [], [], {}, [])
    assert f'href="{daily_report.PROJECT_REPO_URL}"' in body


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

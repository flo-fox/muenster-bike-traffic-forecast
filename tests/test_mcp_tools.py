"""Tests for `muenster_bike_forecast.mcp_tools`.

Same hand-built-synthetic-data style as `tests/test_daily_report.py`/
`tests/test_inference.py` - no live network calls, no real model load.
Every underlying fetch/model call is monkeypatched at the `mcp_tools`
module level (these functions are called directly, not injected as
parameters, unlike `daily_report.build_station_report`).
"""

from __future__ import annotations

import pandas as pd
import pytest

from muenster_bike_forecast import mcp_tools
from muenster_bike_forecast.data.bike_counts import BikeCountDataError, Station
from muenster_bike_forecast.data.join import localize_bike_timestamps

STATION = Station(station_id="12345", name="Test Station", start_year=2020, channels=())

# Must fall within _raw_bike_df_spanning_49h's fixed 2024-06-10..06-12
# window, so the fake fetch_station_month (which filters by real
# year/month) actually has data to return for the requested window.
AS_OF = "2024-06-12"


class _FakeModel:
    """`.predict` triples each row's `total_count` - matches
    `tests/test_daily_report.py`'s `_FakeModel` convention."""

    def predict(self, X: pd.DataFrame):
        return (X["total_count"] * 3).to_numpy()


def _raw_bike_df_spanning_49h(count: float = 10.0) -> pd.DataFrame:
    """49 hourly readings for one station, spanning 48h - enough history
    for both a forecast and an actual-vs-predicted comparison."""
    times = pd.date_range("2024-06-10 00:00", periods=49, freq="h")
    return pd.DataFrame(
        {
            "station_id": [STATION.station_id] * len(times),
            "datetime": times,
            f"{STATION.station_id} (Test)": [count] * len(times),
            f"{STATION.station_id}-status": [0] * len(times),
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


def _empty_calendar_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    holidays_df = pd.DataFrame({"date": pd.to_datetime([]), "name": []})
    school_holidays_df = pd.DataFrame(
        {"start_date": pd.to_datetime([]), "end_date": pd.to_datetime([])}
    )
    return holidays_df, school_holidays_df


def _patch_common_fetches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patches every module-level fetch `mcp_tools` calls with a fixed
    fake, so tool functions run on deterministic synthetic data."""
    raw_bike_df = _raw_bike_df_spanning_49h()
    weather_wide_df = _weather_wide_df(raw_bike_df["datetime"])
    holidays_df, school_holidays_df = _empty_calendar_dfs()

    def _fake_fetch_station_month(
        station_id: str, year: int, month: int
    ) -> pd.DataFrame:
        # months_needed() asks for several (year, month) pairs across the
        # lookback window; returning the full fixture for every one of them
        # would duplicate rows on concat, so filter to the requested month
        # like the real per-month CSV source would.
        in_month = (raw_bike_df["datetime"].dt.year == year) & (
            raw_bike_df["datetime"].dt.month == month
        )
        return raw_bike_df[in_month]

    monkeypatch.setattr(mcp_tools, "list_stations", lambda: [STATION])
    monkeypatch.setattr(mcp_tools, "fetch_station_month", _fake_fetch_station_month)
    monkeypatch.setattr(
        mcp_tools, "fetch_hourly_weather", lambda *a, **k: weather_wide_df
    )
    monkeypatch.setattr(mcp_tools, "public_holidays", lambda *a, **k: holidays_df)
    monkeypatch.setattr(
        mcp_tools, "fetch_school_holidays", lambda *a, **k: school_holidays_df
    )
    monkeypatch.setattr(mcp_tools, "_load_model", lambda: _FakeModel())


def test_list_stations_tool_returns_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_tools, "list_stations", lambda: [STATION])
    result = mcp_tools.list_stations_tool()
    assert result == [
        {"station_id": "12345", "name": "Test Station", "start_year": 2020}
    ]


def test_list_stations_tool_raises_mcp_tool_error_on_fetch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise():
        raise BikeCountDataError("index unreachable")

    monkeypatch.setattr(mcp_tools, "list_stations", _raise)
    with pytest.raises(mcp_tools.McpToolError, match="index unreachable"):
        mcp_tools.list_stations_tool()


def test_find_station_raises_with_known_ids_on_typo() -> None:
    with pytest.raises(mcp_tools.McpToolError, match="12345"):
        mcp_tools._find_station([STATION], "99999")


def test_get_forecast_tool_raises_mcp_tool_error_on_malformed_as_of(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_fetches(monkeypatch)
    with pytest.raises(mcp_tools.McpToolError, match="Invalid as_of date"):
        mcp_tools.get_forecast_tool(STATION.station_id, as_of="not-a-date")


def test_get_forecast_tool_returns_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_fetches(monkeypatch)
    result = mcp_tools.get_forecast_tool(STATION.station_id, as_of=AS_OF)
    assert result["station_id"] == STATION.station_id
    assert result["station_name"] == STATION.name
    assert isinstance(result["predicted_total_next_24h"], float)
    assert isinstance(result["peak_datetime"], str)
    assert isinstance(result["peak_value"], float)
    assert isinstance(result["actual_total_last_24h"], float)
    # _FakeModel triples total_count; a constant count=10.0 predicts 30.0
    # per point, so the forecast curve's total should be a clean multiple.
    assert result["predicted_total_next_24h"] > 0


def test_get_forecast_tool_raises_on_unknown_station(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_fetches(monkeypatch)
    with pytest.raises(mcp_tools.McpToolError, match="Unknown station_id"):
        mcp_tools.get_forecast_tool("does-not-exist")


def test_get_actual_vs_predicted_tool_returns_expected_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_fetches(monkeypatch)
    result = mcp_tools.get_actual_vs_predicted_tool(STATION.station_id, as_of=AS_OF)
    assert result["station_id"] == STATION.station_id
    assert result["station_name"] == STATION.name
    assert isinstance(result["now_datetime"], str)
    # Constant count=10.0, _FakeModel triples -> predicted 30.0/row exactly
    # matches actual 10.0*3 in this fixture only coincidentally at the
    # per-row level; assert the report resolved (not None) and errors are
    # internally consistent, without over-asserting an exact number that
    # would just re-derive daily_report's own arithmetic.
    assert result["unresolved_reason"] is None
    assert result["predicted_total"] is not None
    assert result["actual_total"] is not None
    assert result["abs_error"] == pytest.approx(
        abs(result["predicted_total"] - result["actual_total"])
    )


def test_get_actual_vs_predicted_tool_raises_mcp_tool_error_on_malformed_as_of(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_fetches(monkeypatch)
    with pytest.raises(mcp_tools.McpToolError, match="Invalid as_of date"):
        mcp_tools.get_actual_vs_predicted_tool(STATION.station_id, as_of="not-a-date")


def test_get_actual_vs_predicted_tool_raises_on_unknown_station(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_common_fetches(monkeypatch)
    with pytest.raises(mcp_tools.McpToolError, match="Unknown station_id"):
        mcp_tools.get_actual_vs_predicted_tool("does-not-exist")


def test_get_forecast_tool_wraps_fetch_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_common_fetches(monkeypatch)

    def _raise(*a, **k):
        raise BikeCountDataError("station fetch failed")

    monkeypatch.setattr(mcp_tools, "fetch_station_month", _raise)
    with pytest.raises(mcp_tools.McpToolError, match="Could not build a forecast"):
        mcp_tools.get_forecast_tool(STATION.station_id)

"""Tests for `scripts.send_daily_email`'s own logic (not `daily_report`'s -
see `tests/test_daily_report.py` for that).

No live network calls and no real SMTP connection - `fetch_station_month`
and the SMTP class are both monkeypatched/injected with fakes.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from muenster_bike_forecast import inference
from muenster_bike_forecast.data.bike_counts import Station
from scripts import send_daily_email

STATION_ID = "12345"


def _month_frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame({"station_id": [STATION_ID] * rows, "row": list(range(rows))})


# ---------------------------------------------------------------------------
# fetch_station_raw_history
# ---------------------------------------------------------------------------


def test_fetch_station_raw_history_concatenates_available_months(monkeypatch) -> None:
    station = Station(
        station_id=STATION_ID, name="Test Station", start_year=2020, channels=()
    )

    def fake_fetch(station_id: str, year: int, month: int) -> pd.DataFrame | None:
        assert station_id == STATION_ID
        return _month_frame(2)

    monkeypatch.setattr(send_daily_email, "fetch_station_month", fake_fetch)

    result = send_daily_email.fetch_station_raw_history(station, date(2026, 8, 19))

    n_months = len(inference.months_needed(date(2026, 8, 19)))
    assert len(result) == 2 * n_months


def test_fetch_station_raw_history_skips_months_before_start_year(monkeypatch) -> None:
    station = Station(
        station_id=STATION_ID, name="Test Station", start_year=2026, channels=()
    )
    calls = []

    def fake_fetch(station_id: str, year: int, month: int) -> pd.DataFrame | None:
        calls.append(year)
        return _month_frame(1)

    monkeypatch.setattr(send_daily_email, "fetch_station_month", fake_fetch)

    send_daily_email.fetch_station_raw_history(station, date(2026, 1, 15))

    assert all(year >= 2026 for year in calls)


def test_fetch_station_raw_history_raises_when_nothing_available(monkeypatch) -> None:
    station = Station(
        station_id=STATION_ID, name="Test Station", start_year=2020, channels=()
    )
    monkeypatch.setattr(
        send_daily_email, "fetch_station_month", lambda station_id, year, month: None
    )

    with pytest.raises(inference.InferenceError):
        send_daily_email.fetch_station_raw_history(station, date(2026, 8, 19))


# ---------------------------------------------------------------------------
# send_email
# ---------------------------------------------------------------------------


class _FakeSMTP:
    instances: list["_FakeSMTP"] = []

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.login_args: tuple[str, str] | None = None
        self.sent_message = None
        _FakeSMTP.instances.append(self)

    def __enter__(self) -> "_FakeSMTP":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def login(self, sender: str, app_password: str) -> None:
        self.login_args = (sender, app_password)

    def send_message(self, message) -> None:
        self.sent_message = message


def test_send_email_logs_in_and_sends_with_correct_fields() -> None:
    _FakeSMTP.instances.clear()

    send_daily_email.send_email(
        _FakeSMTP,
        "sender@example.com",
        "app-password",
        "recipient@example.com",
        "Test subject",
        "Test body",
    )

    assert len(_FakeSMTP.instances) == 1
    smtp = _FakeSMTP.instances[0]
    assert smtp.host == "smtp.gmail.com"
    assert smtp.login_args == ("sender@example.com", "app-password")
    assert smtp.sent_message["Subject"] == "Test subject"
    assert smtp.sent_message["From"] == "sender@example.com"
    assert smtp.sent_message["To"] == "recipient@example.com"
    assert smtp.sent_message.get_content().strip() == "Test body"

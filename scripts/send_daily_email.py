"""Sends the daily forecast-accuracy email.

Thin orchestration only: fetches live data with the *uncached* functions
(the Streamlit `@st.cache_data`/`@st.cache_resource` wrappers in
`dashboard_common.py` are pointless for a one-shot process that exits
immediately after running), builds each station's `daily_report.
StationAccuracyReport`, asks Claude to explain the biggest deviations, and
sends the resulting multipart/alternative (plain-text + HTML) email via
Gmail.

Run manually for a local dry run (four required env vars):

    GMAIL_ADDRESS=... GMAIL_APP_PASSWORD=... RECIPIENT_EMAIL=... \
    ANTHROPIC_API_KEY=... python scripts/send_daily_email.py

In production this runs via `.github/workflows/daily_forecast_email.yml`,
which supplies the same four environment variables from GitHub Actions
secrets - never hardcode any of them, including the recipient address.
"""

from __future__ import annotations

import logging
import os
import smtplib
import sys
import time
from datetime import date
from email.message import EmailMessage
from pathlib import Path
from typing import Final

import anthropic
import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from muenster_bike_forecast import daily_report, inference
from muenster_bike_forecast.data.bike_counts import (
    BikeCountDataError,
    Station,
    fetch_station_month,
    list_stations,
)
from muenster_bike_forecast.data.calendar import (
    DEFAULT_PUBLIC_HOLIDAY_SUBDIV,
    SchoolHolidayFetchError,
    SchoolHolidaySchemaError,
    fetch_school_holidays,
    public_holidays,
)
from muenster_bike_forecast.data.join import JoinError, combine_weather_parameters
from muenster_bike_forecast.data.semester_dates import SemesterDateRangeError
from muenster_bike_forecast.data.weather import (
    PARAMETER_SPECS,
    WeatherFetchError,
    WeatherSchemaError,
    fetch_hourly_weather,
)
from muenster_bike_forecast.modeling.lag_features import LagFeatureError
from muenster_bike_forecast.modeling.model_table import ModelTableError

logger = logging.getLogger(__name__)

MODEL_PATH: Final[Path] = PROJECT_ROOT / "models" / "production_lightgbm.joblib"

# Duplicated from dashboard_common.FETCH_ERRORS rather than imported, so this
# script never pulls in `streamlit` (dashboard_common imports it) into a
# process with no Streamlit runtime at all - it's a fixed, rarely-changing
# set of exception types, so the small duplication risk is worth the
# decoupling.
SCRIPT_FETCH_ERRORS: Final[tuple[type[Exception], ...]] = (
    BikeCountDataError,
    WeatherFetchError,
    WeatherSchemaError,
    SchoolHolidayFetchError,
    SchoolHolidaySchemaError,
    SemesterDateRangeError,
    JoinError,
    ModelTableError,
    LagFeatureError,
    inference.InferenceError,
)

# Small courtesy delay between per-station fetches against
# raw.githubusercontent.com - this project has previously observed real
# HTTP 429s under repeated fetching in a short window. fetch_station_month
# itself has no retry/backoff, so the per-station SCRIPT_FETCH_ERRORS catch
# below is the actual mitigation; this is just a cheap extra courtesy.
INTER_STATION_DELAY: Final[float] = 0.2


def fetch_station_raw_history(station: Station, as_of: date) -> pd.DataFrame:
    """Fetches one station's raw bike-count rows for the last ~35 days.

    Mirrors `dashboard_common.build_forecast`'s month-fetching logic,
    duplicated here (uncached) since this process exits after one run.

    Args:
        station: Station to fetch.
        as_of: Last date (inclusive) the fetched window should cover.

    Returns:
        Concatenated raw rows across every needed month.

    Raises:
        inference.InferenceError: if no month in the window returned any
            data (e.g. the station started after this window, or every
            month fetch reported a legitimate absence).
    """
    frames = []
    for year, month in inference.months_needed(as_of):
        if year < station.start_year:
            continue
        frame = fetch_station_month(station.station_id, year, month)
        if frame is not None:
            frames.append(frame)
    if not frames:
        raise inference.InferenceError(
            f"No recent bike-count data available for {station.name} "
            f"({station.station_id}) in the last "
            f"{inference.MIN_HISTORY_LOOKBACK.days} days."
        )
    return pd.concat(frames, ignore_index=True)


def send_email(
    smtp_class: type[smtplib.SMTP],
    sender: str,
    app_password: str,
    recipient: str,
    subject: str,
    text_body: str,
    html_body: str,
) -> None:
    """Sends a multipart/alternative (plain-text + HTML) email via Gmail SMTP.

    Args:
        smtp_class: The SMTP class to instantiate (`smtplib.SMTP_SSL` in
            production) - injected so this is mockable in a test.
        sender: Gmail address to send from.
        app_password: Gmail app-specific password for `sender`.
        recipient: Recipient email address.
        subject: Email subject line.
        text_body: Plain-text email body (the `text/plain` part).
        html_body: HTML email body (the `text/html` alternative part).
    """
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    message.set_content(text_body)
    message.add_alternative(html_body, subtype="html")

    with smtp_class("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, app_password)
        smtp.send_message(message)


def main() -> None:
    """Builds and sends the daily forecast-accuracy report email."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    as_of = date.today()
    stations = list_stations()
    model = joblib.load(MODEL_PATH)
    weather_wide_df = combine_weather_parameters(
        {
            parameter: fetch_hourly_weather(parameter, period="recent")
            for parameter in PARAMETER_SPECS
        }
    )
    public_holidays_df = public_holidays(
        as_of.year - 1, as_of.year, subdiv=DEFAULT_PUBLIC_HOLIDAY_SUBDIV
    )
    school_holidays_df = fetch_school_holidays(as_of.year - 1, as_of.year)

    reports: list[daily_report.StationAccuracyReport] = []
    dropped: list[str] = []
    for station in stations:
        try:
            try:
                raw_bike_df = fetch_station_raw_history(station, as_of)
                report = daily_report.build_station_report(
                    station,
                    model,
                    public_holidays_df,
                    school_holidays_df,
                    weather_wide_df,
                    raw_bike_df,
                )
            except SCRIPT_FETCH_ERRORS as exc:
                logger.warning(
                    "Dropping station %s (%s): %s",
                    station.station_id,
                    station.name,
                    exc,
                )
                dropped.append(station.name)
                continue
            reports.append(report)
        finally:
            # Applies on both the success and the dropped-station path - a
            # failed fetch is exactly the situation most likely to precede
            # an incipient rate limit, so it must not skip this courtesy.
            time.sleep(INTER_STATION_DELAY)

    if not reports:
        raise RuntimeError(
            f"All {len(stations)} station(s) were dropped; nothing to report."
        )

    flagged = daily_report.select_top_deviations(reports)

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    explanations: dict[str, str] = {}
    for report in flagged:
        try:
            explanations[report.station_id] = daily_report.generate_explanation(
                client, report
            )
        except anthropic.APIError as exc:
            explanations[report.station_id] = f"(explanation unavailable: {exc})"
            logger.warning("Explanation failed for %s: %s", report.station_id, exc)

    subject = daily_report.format_email_subject(as_of)
    text_body = daily_report.format_email_body(
        as_of, reports, flagged, explanations, dropped
    )
    html_body = daily_report.format_email_body_html(
        as_of, reports, flagged, explanations, dropped
    )
    send_email(
        smtplib.SMTP_SSL,
        os.environ["GMAIL_ADDRESS"],
        os.environ["GMAIL_APP_PASSWORD"],
        os.environ["RECIPIENT_EMAIL"],
        subject,
        text_body,
        html_body,
    )
    logger.info("Sent daily forecast report for %s (%d dropped).", as_of, len(dropped))


if __name__ == "__main__":
    main()

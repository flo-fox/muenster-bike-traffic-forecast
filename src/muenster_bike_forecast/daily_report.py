"""Yesterday's-forecast-vs-actual accuracy report, plus AI explanations.

Pure transforms over already-fetched/assembled data (mirrors `inference.py`'s
role) - no network I/O of any kind, including no `anthropic.Anthropic()`
client construction. `scripts/send_daily_email.py` is the thin orchestration
layer that fetches live data, constructs the Anthropic client, and sends the
resulting email.

"Yesterday's forecast" is not stored anywhere - there is no forecast-log
mechanism in this project. It is instead *recomputed*: `inference.
assemble_feature_history`'s lag/rolling features are computed relative to
each row's own timestamp, not "as of when this function runs", so the row at
roughly "24h before now" (within `YESTERDAY_ROW_TOLERANCE`) can be run
through `inference.predict_24h_ahead` today and yields the same result
running it yesterday would have - provided the upstream source hasn't
retroactively revised the underlying monthly CSV between yesterday and
today, which is not checked for and is an accepted risk.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass
from datetime import date
from typing import Final

import anthropic
import pandas as pd

from muenster_bike_forecast import inference
from muenster_bike_forecast.data.bike_counts import Station

TOP_N_DEVIATIONS: Final[int] = 3
YESTERDAY_LOOKBACK: Final[pd.Timedelta] = pd.Timedelta(hours=24)
# How close a bike-count reading must be to "24h before now" to count as
# yesterday's basis for the accuracy check - a few missed 15-minute
# intervals (a short sensor gap) is still close enough; anything wider is
# treated as "no usable reading near that time".
YESTERDAY_ROW_TOLERANCE: Final[pd.Timedelta] = pd.Timedelta(minutes=45)
# A timing gap under this is routine 15-minute-grid quantization, not worth
# calling out; wider gaps mean select_row_near fell back to a reading that
# isn't really 24h-apart from "now" (see build_station_report) and should
# be disclosed rather than presented as an exact same-instant comparison.
NOTABLE_TIMING_GAP: Final[pd.Timedelta] = pd.Timedelta(minutes=15)
# Same threshold and Europe/Berlin-local convention as
# dashboard_common.STALENESS_WARNING_THRESHOLD / pages/station_forecast.py -
# duplicated rather than imported so this module never pulls in `streamlit`
# (dashboard_common imports it) into a process with no Streamlit runtime.
STALENESS_WARNING_THRESHOLD: Final[pd.Timedelta] = pd.Timedelta(hours=36)
DEFAULT_EXPLANATION_MODEL: Final[str] = "claude-haiku-4-5"
DEFAULT_EXPLANATION_MAX_TOKENS: Final[int] = 160

_EXPLANATION_SYSTEM_PROMPT: Final[str] = (
    "You explain short-term bike-traffic forecast misses for one bike-"
    "counting station in Münster, Germany, using only the facts given to "
    "you in the user message. Write 2-3 sentences (roughly 40-60 words), "
    "plain language, no headers or bullet points. Ground your explanation "
    "strictly in the provided numbers, weather, and calendar flags - never "
    "invent a specific unavailable cause (a named event, road closure, "
    "festival, or similar not present in the data). If nothing in the "
    "given data plausibly explains the deviation, say so plainly instead "
    "of guessing."
)


@dataclass(frozen=True)
class RowContext:
    """Weather/calendar/time context for one feature-history row.

    Used both for display in the email and to ground the AI-explanation
    prompt in `build_explanation_prompt` - every field here is something
    the model is explicitly allowed to reason from.

    Attributes:
        datetime: The row's own timestamp.
        hour: Hour of day (0-23), matching the `hour` feature column.
        day_of_week: Day of week (0=Monday, ..., 6=Sunday), matching the
            `day_of_week` feature column.
        is_public_holiday: German public holiday flag for this date.
        is_school_holiday: NRW school holiday flag for this date.
        is_lecture_period: University lecture-period flag for this date.
        weather_air_temperature_c: Air temperature, or `None` if the
            weather join left this row's value null.
        weather_precipitation_mm: Precipitation, or `None` if null.
        weather_wind_speed_ms: Wind speed, or `None` if null.
        weather_relative_humidity_pct: Relative humidity, or `None` if
            null.
    """

    datetime: pd.Timestamp
    hour: int
    day_of_week: int
    is_public_holiday: bool
    is_school_holiday: bool
    is_lecture_period: bool
    weather_air_temperature_c: float | None
    weather_precipitation_mm: float | None
    weather_wind_speed_ms: float | None
    weather_relative_humidity_pct: float | None

    @classmethod
    def from_feature_row(cls, row: pd.Series) -> "RowContext":
        """Builds a `RowContext` from one row of an `assemble_feature_history` table.

        Args:
            row: A row containing at least `inference.FEATURE_COLS` plus
                `datetime`, e.g. as returned by `inference.latest_feature_row`
                or `select_row_near`.

        Returns:
            The corresponding `RowContext`.
        """
        return cls(
            datetime=row["datetime"],
            hour=int(row["hour"]),
            day_of_week=int(row["day_of_week"]),
            is_public_holiday=bool(row["is_public_holiday"]),
            is_school_holiday=bool(row["is_school_holiday"]),
            is_lecture_period=bool(row["is_lecture_period"]),
            weather_air_temperature_c=_none_if_nan(
                row.get("weather_air_temperature_c")
            ),
            weather_precipitation_mm=_none_if_nan(row.get("weather_precipitation_mm")),
            weather_wind_speed_ms=_none_if_nan(row.get("weather_wind_speed_ms")),
            weather_relative_humidity_pct=_none_if_nan(
                row.get("weather_relative_humidity_pct")
            ),
        )


def _none_if_nan(value: object) -> float | None:
    """Returns `None` for a missing/NaN value, else `float(value)`."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


@dataclass(frozen=True)
class StationAccuracyReport:
    """One station's yesterday-forecast-vs-actual check, plus today's forecast.

    `predicted_for_now`/`abs_error`/`pct_diff`/`prediction_basis_context`
    are all `None` together when no bike-count reading was found close
    enough to "24h before now" (e.g. a sensor gap right at that point,
    `unresolved_reason` explains why) - this degrades gracefully rather
    than dropping the station entirely, so `forecast_24h_ahead` (today's
    fresh forecast) still shows for a station whose accuracy can't be
    checked. A station is only fully dropped from the report (not
    represented by any `StationAccuracyReport` at all) when its raw
    fetch, weather join, or "now" feature row itself is unusable - the
    existing `inference.InferenceError`/`FETCH_ERRORS` case, handled by
    the caller in `scripts/send_daily_email.py`.

    Attributes:
        station_id: Station directory id.
        station_name: Human-readable station name.
        now_datetime: Timestamp of the "now" reading.
        data_age: How old `now_datetime` is relative to when this report
            was built (Europe/Berlin local time, matching
            `pages/station_forecast.py`'s own staleness check) - the
            source has previously gone stale for extended periods (see
            `dashboard_common.STALENESS_WARNING_THRESHOLD`), and without
            this, a stale "now" reading would be reported as if it were
            live. See `is_stale`.
        actual_now: The actual observed `total_count` at `now_datetime` -
            this is also the target yesterday's forecast was trying to
            predict.
        forecast_24h_ahead: Today's fresh prediction for 24h from now.
        now_context: Weather/calendar context at `now_datetime`.
        predicted_for_now: What the model predicts for `now_datetime`,
            computed from the row ~24h before it - i.e. "what yesterday's
            forecast for right now would have been."
        abs_error: `abs(predicted_for_now - actual_now)`.
        pct_diff: `(predicted_for_now - actual_now) / actual_now * 100`;
            positive means the model over-predicted. `None` when
            `actual_now == 0` (percentage is undefined), even though
            `predicted_for_now`/`abs_error` are still populated in that
            case.
        prediction_basis_context: Weather/calendar context ~24h ago, i.e.
            the conditions the "yesterday" prediction was made under.
        prediction_timing_gap: How far `predicted_for_now`'s implied
            target time (`prediction_basis_context.datetime + 24h`) is
            from `now_datetime`. `select_row_near` only guarantees the
            basis row is within `YESTERDAY_ROW_TOLERANCE` of "24h ago", so
            this can be nonzero (up to that tolerance) whenever there was
            a gap right around that point - `predicted_for_now` and
            `actual_now` are then not quite for the same instant, which
            matters given this data's sharp commute-peak transitions.
            `None` iff `predicted_for_now` is `None`.
        unresolved_reason: Human-readable reason `predicted_for_now` is
            `None`, or `None` if it was resolved.
    """

    station_id: str
    station_name: str
    now_datetime: pd.Timestamp
    data_age: pd.Timedelta
    actual_now: float
    forecast_24h_ahead: float
    now_context: RowContext
    predicted_for_now: float | None
    abs_error: float | None
    pct_diff: float | None
    prediction_basis_context: RowContext | None
    prediction_timing_gap: pd.Timedelta | None
    unresolved_reason: str | None

    @property
    def is_stale(self) -> bool:
        """Whether `now_datetime` is older than `STALENESS_WARNING_THRESHOLD`."""
        return self.data_age > STALENESS_WARNING_THRESHOLD


def select_row_near(
    feature_history_df: pd.DataFrame,
    station_id: int | str,
    target_datetime: pd.Timestamp,
    tolerance: pd.Timedelta = YESTERDAY_ROW_TOLERANCE,
) -> pd.Series:
    """Picks the evaluable row closest to `target_datetime`, within tolerance.

    Args:
        feature_history_df: As returned by `inference.assemble_feature_history`.
        station_id: Station to select (matched after casting to `int64`,
            consistent with `inference.assemble_feature_history`).
        target_datetime: The timestamp to find the nearest row to.
        tolerance: Maximum allowed distance from `target_datetime`.

    Returns:
        The selected row, as a `pandas.Series`.

    Raises:
        inference.InferenceError: if no row with a non-null `total_count`
            for `station_id` falls within `tolerance` of `target_datetime`.
    """
    station_rows = feature_history_df.loc[
        feature_history_df["station_id"] == int(station_id)
    ]
    evaluable = station_rows.dropna(subset=["total_count"])
    if evaluable.empty:
        raise inference.InferenceError(
            f"No row with a non-null total_count for station {station_id!r}."
        )
    time_diff = (evaluable["datetime"] - target_datetime).abs()
    closest_index = time_diff.idxmin()
    if time_diff.loc[closest_index] > tolerance:
        raise inference.InferenceError(
            f"No row within {tolerance} of {target_datetime} for station "
            f"{station_id!r}; closest is {time_diff.loc[closest_index]} away."
        )
    return evaluable.loc[closest_index]


def build_station_report(
    station: Station,
    model: object,
    ratio_table: pd.DataFrame,
    public_holidays_df: pd.DataFrame,
    school_holidays_df: pd.DataFrame,
    weather_wide_df: pd.DataFrame,
    raw_bike_df: pd.DataFrame,
) -> StationAccuracyReport:
    """Builds one station's full accuracy report from freshly fetched data.

    Calls `inference.assemble_feature_history` once, then derives both
    "today's forecast" (from the current row) and "yesterday's forecast
    for right now" (from the row ~24h earlier) from that single history,
    per this module's recompute-not-persist design (see module docstring).

    Args:
        station: The station to report on.
        model: A fitted scikit-learn `Pipeline` (or any object exposing
            `.predict`), as loaded from
            `models/production_random_forest.joblib`.
        ratio_table: Static per-station `weekend_weekday_ratio` lookup.
        public_holidays_df: As returned by `data.calendar.public_holidays`.
        school_holidays_df: As returned by `data.calendar.fetch_school_holidays`.
        weather_wide_df: Combined hourly weather, as returned by
            `data.join.combine_weather_parameters`.
        raw_bike_df: `station`'s raw bike-count rows covering at least the
            last `inference.MIN_HISTORY_LOOKBACK`.

    Returns:
        The station's `StationAccuracyReport`.

    Raises:
        inference.InferenceError: if `raw_bike_df`/the feature assembly is
            unusable, or there is no evaluable "now" row for `station` -
            both propagate up so the caller can drop the station entirely,
            the same way `dashboard_common.build_forecast` does. A missing
            ~24h-ago row, by contrast, is caught internally (see
            `StationAccuracyReport`'s docstring) and does not raise.
    """
    history = inference.assemble_feature_history(
        raw_bike_df,
        weather_wide_df,
        public_holidays_df,
        school_holidays_df,
        ratio_table,
    )
    now_row = inference.latest_feature_row(history, station.station_id)
    now_datetime = now_row["datetime"]
    data_age = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None) - now_datetime
    actual_now = float(now_row["total_count"])
    forecast_24h_ahead = inference.predict_24h_ahead(model, now_row)
    now_context = RowContext.from_feature_row(now_row)

    predicted_for_now: float | None = None
    abs_error: float | None = None
    pct_diff: float | None = None
    prediction_basis_context: RowContext | None = None
    prediction_timing_gap: pd.Timedelta | None = None
    unresolved_reason: str | None = None
    try:
        basis_row = select_row_near(
            history, station.station_id, now_datetime - YESTERDAY_LOOKBACK
        )
    except inference.InferenceError as exc:
        unresolved_reason = (
            "No bike-count reading close enough to 24h ago to check "
            f"yesterday's forecast: {exc}"
        )
    else:
        predicted_for_now = inference.predict_24h_ahead(model, basis_row)
        abs_error = abs(predicted_for_now - actual_now)
        pct_diff = (
            (predicted_for_now - actual_now) / actual_now * 100
            if actual_now != 0
            else None
        )
        prediction_basis_context = RowContext.from_feature_row(basis_row)
        prediction_timing_gap = abs(
            (basis_row["datetime"] + YESTERDAY_LOOKBACK) - now_datetime
        )

    return StationAccuracyReport(
        station_id=station.station_id,
        station_name=station.name,
        now_datetime=now_datetime,
        data_age=data_age,
        actual_now=actual_now,
        forecast_24h_ahead=forecast_24h_ahead,
        now_context=now_context,
        predicted_for_now=predicted_for_now,
        abs_error=abs_error,
        pct_diff=pct_diff,
        prediction_basis_context=prediction_basis_context,
        prediction_timing_gap=prediction_timing_gap,
        unresolved_reason=unresolved_reason,
    )


def select_top_deviations(
    reports: list[StationAccuracyReport], n: int = TOP_N_DEVIATIONS
) -> list[StationAccuracyReport]:
    """Picks the `n` reports with the largest absolute error.

    Args:
        reports: Station reports to rank (typically every processed
            station, per the fleet-wide scope decision).
        n: How many to return.

    Returns:
        Up to `n` reports with `abs_error is not None`, sorted by
        `abs_error` descending, `station_id` ascending as a tiebreak.
        Reports with `abs_error is None` (no resolvable yesterday
        comparison) are excluded - there's nothing to rank them by.
    """
    ranked = [report for report in reports if report.abs_error is not None]
    ranked.sort(key=lambda report: (-report.abs_error, report.station_id))
    return ranked[:n]


def _format_context(context: RowContext) -> str:
    """Renders a `RowContext` as labeled plain-text lines for the AI prompt."""

    def _fmt(value: float | None, unit: str) -> str:
        return f"{value:.1f}{unit}" if value is not None else "unknown"

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    return (
        f"  Time: {context.datetime:%Y-%m-%d %H:%M} ({day_names[context.day_of_week]})\n"
        f"  Public holiday: {context.is_public_holiday}\n"
        f"  School holiday: {context.is_school_holiday}\n"
        f"  Lecture period: {context.is_lecture_period}\n"
        f"  Temperature: {_fmt(context.weather_air_temperature_c, 'C')}\n"
        f"  Precipitation: {_fmt(context.weather_precipitation_mm, 'mm')}\n"
        f"  Wind speed: {_fmt(context.weather_wind_speed_ms, 'm/s')}\n"
        f"  Relative humidity: {_fmt(context.weather_relative_humidity_pct, '%')}"
    )


def build_explanation_prompt(report: StationAccuracyReport) -> tuple[str, str]:
    """Builds the (system, user) prompt asking for a deviation explanation.

    Pure string formatting from `report`'s already-computed fields - no
    network call. Only meaningful for a report with a resolved
    `predicted_for_now` (i.e. `report.abs_error is not None`); callers
    should only invoke this for reports returned by `select_top_deviations`.

    Args:
        report: The station's accuracy report to explain.

    Returns:
        `(system, user)` message strings for `generate_explanation`.
    """
    direction = "over-predicted" if (report.pct_diff or 0) >= 0 else "under-predicted"
    pct_text = f" ({report.pct_diff:+.1f}%)" if report.pct_diff is not None else ""
    caveats = []
    if report.is_stale:
        caveats.append(
            f"Caveat: the 'right now' reading is actually {report.data_age} old - "
            "the upstream source has not published newer data yet, so it may not "
            "reflect current real-world conditions."
        )
    if (
        report.prediction_timing_gap
        and report.prediction_timing_gap > NOTABLE_TIMING_GAP
    ):
        caveats.append(
            f"Caveat: no bike-count reading was available at exactly 24h before "
            f"now, so this comparison is offset by about "
            f"{report.prediction_timing_gap} - the two values may not be for "
            "exactly the same moment."
        )
    caveats_text = ("\n" + "\n".join(caveats) + "\n") if caveats else ""
    user = (
        f"Station: {report.station_name} ({report.station_id})\n"
        f"Yesterday's forecast for right now: {report.predicted_for_now:.0f}\n"
        f"Actual value right now: {report.actual_now:.0f}\n"
        f"Absolute error: {report.abs_error:.0f}{pct_text} - the model {direction}.\n"
        f"{caveats_text}"
        "\n"
        "Conditions ~24h ago, when the forecast was made:\n"
        f"{_format_context(report.prediction_basis_context)}\n"
        "\n"
        "Conditions right now:\n"
        f"{_format_context(report.now_context)}"
    )
    return _EXPLANATION_SYSTEM_PROMPT, user


def generate_explanation(
    client: anthropic.Anthropic,
    report: StationAccuracyReport,
    model: str = DEFAULT_EXPLANATION_MODEL,
    max_tokens: int = DEFAULT_EXPLANATION_MAX_TOKENS,
) -> str:
    """Asks Claude to explain one station's forecast deviation.

    Args:
        client: An `anthropic.Anthropic` client (or any object exposing a
            compatible `.messages.create`) - constructed by the caller so
            this function stays testable via a fake client double.
        report: The station's accuracy report to explain.
        model: Anthropic model id to use.
        max_tokens: Max tokens for the response.

    Returns:
        The generated explanation text, stripped of leading/trailing
        whitespace.
    """
    system, user = build_explanation_prompt(report)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text.strip()


def format_email_subject(as_of: date) -> str:
    """Builds the email subject line for a given report date."""
    return f"Bike traffic forecast report - {as_of.isoformat()}"


def format_email_body(
    as_of: date,
    reports: list[StationAccuracyReport],
    flagged: list[StationAccuracyReport],
    explanations: dict[str, str],
    dropped_stations: list[str],
) -> str:
    """Builds the plain-text email body.

    Args:
        as_of: The report date.
        reports: Every successfully-processed station's report.
        flagged: The subset of `reports` (typically from
            `select_top_deviations`) that get an explanation section.
        explanations: `station_id` -> AI-generated explanation text, for
            each station in `flagged`.
        dropped_stations: Names of stations that couldn't be processed at
            all (see `StationAccuracyReport`'s docstring for the
            distinction from an unresolved-but-listed station).

    Returns:
        The full plain-text email body.
    """
    lines: list[str] = [f"Bike traffic forecast report for {as_of.isoformat()}"]

    if flagged and flagged[0].pct_diff is not None:
        top_line = f"{flagged[0].station_name} at {flagged[0].pct_diff:+.1f}%"
    elif flagged:
        top_line = f"{flagged[0].station_name}"
    else:
        top_line = "n/a"
    n_stale = sum(1 for report in reports if report.is_stale)
    stale_note = f", {n_stale} stale" if n_stale else ""
    lines.append(
        f"{len(reports)} station(s) checked, {len(dropped_stations)} dropped"
        f"{stale_note}, top miss: {top_line}"
    )
    lines.append("")

    lines.append(
        f"{'Station':<28}{'Predicted (~24h ago)':>22}{'Actual now':>12}"
        f"{'% diff':>10}{'Forecast +24h':>16}"
    )
    lines.append("-" * 88)
    for report in sorted(reports, key=lambda report: report.station_name):
        predicted = (
            f"{report.predicted_for_now:.0f}"
            if report.predicted_for_now is not None
            else "n/a"
        )
        pct = f"{report.pct_diff:+.1f}%" if report.pct_diff is not None else "n/a"
        lines.append(
            f"{report.station_name:<28}{predicted:>22}{report.actual_now:>12.0f}"
            f"{pct:>10}{report.forecast_24h_ahead:>16.0f}"
        )
        if report.unresolved_reason:
            lines.append(f"    ({report.unresolved_reason})")
        if report.is_stale:
            lines.append(
                f"    (STALE: 'now' reading is {report.data_age} old - source "
                "hasn't published newer data yet)"
            )
        if (
            report.prediction_timing_gap
            and report.prediction_timing_gap > NOTABLE_TIMING_GAP
        ):
            lines.append(
                f"    (predicted/actual are offset by ~{report.prediction_timing_gap}, "
                "not exactly the same moment - no reading was available at exactly 24h ago)"
            )
    lines.append("")

    lines.append("Notable deviations:")
    if not flagged:
        lines.append("  None - no station had a resolvable yesterday comparison.")
    for report in flagged:
        pct = (
            f"{report.pct_diff:+.1f}%"
            if report.pct_diff is not None
            else "undefined (actual was 0)"
        )
        lines.append("")
        lines.append(f"- {report.station_name} ({report.station_id})")
        lines.append(
            f"  Predicted {report.predicted_for_now:.0f}, actual "
            f"{report.actual_now:.0f} ({pct})"
        )
        if report.is_stale:
            lines.append(
                f"  Note: this station's data is {report.data_age} old (stale)."
            )
        lines.append(
            f"  {explanations.get(report.station_id, '(no explanation generated)')}"
        )
    lines.append("")

    if dropped_stations:
        lines.append(f"Dropped (no usable recent data): {', '.join(dropped_stations)}")
    lines.append(
        'Note: "yesterday\'s forecast" is recomputed today from the same '
        "historical window, not a stored value - this assumes the upstream "
        "bike-count source hasn't retroactively revised past data."
    )
    return "\n".join(lines)


def _station_row_notes_html(report: StationAccuracyReport) -> str:
    """Renders one station table row's inline notes (stale/unresolved/offset) as HTML."""
    notes: list[str] = []
    if report.unresolved_reason:
        notes.append(html.escape(report.unresolved_reason))
    if report.is_stale:
        notes.append(
            f"STALE: 'now' reading is {report.data_age} old - source hasn't "
            "published newer data yet"
        )
    if (
        report.prediction_timing_gap
        and report.prediction_timing_gap > NOTABLE_TIMING_GAP
    ):
        notes.append(
            f"predicted/actual offset by ~{report.prediction_timing_gap}, not "
            "exactly the same moment"
        )
    if not notes:
        return ""
    return (
        '<br><span style="font-size:0.85em;color:#666;">' + "; ".join(notes) + "</span>"
    )


def format_email_body_html(
    as_of: date,
    reports: list[StationAccuracyReport],
    flagged: list[StationAccuracyReport],
    explanations: dict[str, str],
    dropped_stations: list[str],
) -> str:
    """Builds the HTML email body.

    Same content, values, and thresholds as `format_email_body` - this is
    an alternate rendering (a styled table plus sectioned deviations
    instead of fixed-width text), meant to be sent as the `text/html`
    alternative alongside `format_email_body`'s `text/plain` part, not a
    replacement for it.

    Args:
        as_of: The report date.
        reports: Every successfully-processed station's report.
        flagged: The subset of `reports` (typically from
            `select_top_deviations`) that get an explanation section.
        explanations: `station_id` -> AI-generated explanation text, for
            each station in `flagged`.
        dropped_stations: Names of stations that couldn't be processed at
            all.

    Returns:
        The full HTML email body, as a standalone `<html>` document.
    """
    if flagged and flagged[0].pct_diff is not None:
        top_line = (
            f"{html.escape(flagged[0].station_name)} at {flagged[0].pct_diff:+.1f}%"
        )
    elif flagged:
        top_line = html.escape(flagged[0].station_name)
    else:
        top_line = "n/a"
    n_stale = sum(1 for report in reports if report.is_stale)
    stale_note = f", {n_stale} stale" if n_stale else ""
    summary = (
        f"{len(reports)} station(s) checked, {len(dropped_stations)} dropped"
        f"{stale_note}, top miss: {top_line}"
    )

    row_cells = '<td style="padding:4px 8px;text-align:right;border-bottom:1px solid #ddd;">{}</td>'
    table_rows = []
    for report in sorted(reports, key=lambda report: report.station_name):
        predicted = (
            f"{report.predicted_for_now:.0f}"
            if report.predicted_for_now is not None
            else "n/a"
        )
        pct = f"{report.pct_diff:+.1f}%" if report.pct_diff is not None else "n/a"
        table_rows.append(
            "<tr>"
            f'<td style="padding:4px 8px;border-bottom:1px solid #ddd;">'
            f"{html.escape(report.station_name)}{_station_row_notes_html(report)}</td>"
            + row_cells.format(predicted)
            + row_cells.format(f"{report.actual_now:.0f}")
            + row_cells.format(pct)
            + row_cells.format(f"{report.forecast_24h_ahead:.0f}")
            + "</tr>"
        )

    deviation_sections = []
    if not flagged:
        deviation_sections.append(
            "<p>None - no station had a resolvable yesterday comparison.</p>"
        )
    for report in flagged:
        pct = (
            f"{report.pct_diff:+.1f}%"
            if report.pct_diff is not None
            else "undefined (actual was 0)"
        )
        stale_html = (
            f"<p><em>Note: this station's data is {report.data_age} old (stale).</em></p>"
            if report.is_stale
            else ""
        )
        explanation = html.escape(
            explanations.get(report.station_id, "(no explanation generated)")
        )
        deviation_sections.append(
            f"<h3>{html.escape(report.station_name)} ({html.escape(report.station_id)})</h3>"
            f"<p>Predicted {report.predicted_for_now:.0f}, actual "
            f"{report.actual_now:.0f} ({pct})</p>"
            f"{stale_html}"
            f"<p>{explanation}</p>"
        )

    dropped_html = (
        f"<p>Dropped (no usable recent data): {html.escape(', '.join(dropped_stations))}</p>"
        if dropped_stations
        else ""
    )

    return (
        '<html><body style="font-family:Arial,Helvetica,sans-serif;color:#222;">'
        f"<h2>Bike traffic forecast report for {as_of.isoformat()}</h2>"
        f"<p>{summary}</p>"
        '<table style="border-collapse:collapse;width:100%;">'
        '<tr style="background:#f0f0f0;text-align:right;">'
        '<th style="padding:4px 8px;text-align:left;">Station</th>'
        '<th style="padding:4px 8px;">Predicted (~24h ago)</th>'
        '<th style="padding:4px 8px;">Actual now</th>'
        '<th style="padding:4px 8px;">% diff</th>'
        '<th style="padding:4px 8px;">Forecast +24h</th>'
        "</tr>" + "".join(table_rows) + "</table>"
        "<h2>Notable deviations</h2>"
        + "".join(deviation_sections)
        + dropped_html
        + '<p style="font-size:0.85em;color:#666;">Note: &quot;yesterday&#x27;s '
        "forecast&quot; is recomputed today from the same historical window, "
        "not a stored value - this assumes the upstream bike-count source "
        "hasn't retroactively revised past data.</p>"
        "</body></html>"
    )

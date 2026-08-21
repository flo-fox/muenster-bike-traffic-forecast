"""Yesterday's-daily-total-vs-actual accuracy report, plus AI explanations.

Pure transforms over already-fetched/assembled data (mirrors `inference.py`'s
role) - no network I/O of any kind, including no `anthropic.Anthropic()`
client construction. `scripts/send_daily_email.py` is the thin orchestration
layer that fetches live data, constructs the Anthropic client, and sends the
resulting email.

"Yesterday's forecast" is not stored anywhere - there is no forecast-log
mechanism in this project. It is instead *recomputed*: `inference.
predict_forecast_curve` can be anchored at any timestamp, so anchoring it at
`now - 24h` reproduces exactly what running it 24h ago would have predicted
for the rolling window ending "now" - provided the upstream source hasn't
retroactively revised the underlying monthly CSV between yesterday and
today, which is not checked for and is an accepted risk.

The accuracy check compares a *daily total* (summed over the rolling 24h
window ending "now"), not a single point reading - a single 15-minute actual
count is noisy (sensor blips, one unlucky/lucky interval) and doesn't
represent the model's real day-ahead skill as well as a window sum does.
This is therefore a genuinely different, coarser metric than the model's own
published single-point 24h-ahead MAE/RMSE (see notebooks 08-17) - not
directly comparable to those numbers.
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
# Same threshold and Europe/Berlin-local convention as
# dashboard_common.STALENESS_WARNING_THRESHOLD / pages/station_forecast.py -
# duplicated rather than imported so this module never pulls in `streamlit`
# (dashboard_common imports it) into a process with no Streamlit runtime.
STALENESS_WARNING_THRESHOLD: Final[pd.Timedelta] = pd.Timedelta(hours=36)
DEFAULT_EXPLANATION_MODEL: Final[str] = "claude-haiku-4-5"
DEFAULT_EXPLANATION_MAX_TOKENS: Final[int] = 160

_EXPLANATION_SYSTEM_PROMPT: Final[str] = (
    "You explain daily-total bike-traffic forecast misses for one bike-"
    "counting station in Münster, Germany - the miss is between a predicted "
    "and actual *total* over a rolling 24-hour window, not a single moment. "
    "Use only the facts given to you in the user message. Write 2-3 "
    "sentences (roughly 40-60 words), plain language, no headers or bullet "
    "points. Ground your explanation strictly in the provided numbers, "
    "weather, and calendar flags - never invent a specific unavailable "
    "cause (a named event, road closure, festival, or similar not present "
    "in the data). If nothing in the given data plausibly explains the "
    "deviation, say so plainly instead of guessing."
)

_DAY_NAMES: Final[list[str]] = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def _none_if_nan(value: object) -> float | None:
    """Returns `None` for a missing/NaN value, else `float(value)`."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def _sum_or_none(series: pd.Series) -> float | None:
    """Sums `series`, or `None` if every value is null.

    Unlike `.mean()`, `.sum()` on an all-null `Series` returns `0.0` (not
    `NaN`) under pandas' default `skipna=True` - passing that straight
    through `_none_if_nan` would misreport "no precipitation data at all"
    as "confirmed zero rain". This checks nullness explicitly instead.
    """
    if series.isna().all():
        return None
    return float(series.sum())


@dataclass(frozen=True)
class WindowContext:
    """Aggregated weather/calendar context over a rolling 24h window.

    Used to ground the AI-explanation prompt in `build_explanation_prompt` -
    every field here is something the model is explicitly allowed to reason
    from.

    Attributes:
        window_start: Timestamp of the earliest row actually present in the
            window - not the theoretical boundary (`window_end - 24h`).
            If the reading right at that boundary is missing (a sensor
            gap), this understates the window's true span by however much
            is missing; the aggregates below are unaffected (still summed/
            averaged over whatever rows exist), only this display value.
        window_end: Timestamp of the latest row actually present - also
            the report's "now".
        day_of_week: Day of week of the window's most recent row (0=Monday,
            ..., 6=Sunday) - used as the single representative day; a
            window spanning midnight technically touches two calendar
            dates, so this is an approximation, same tolerance as other
            accepted-not-detected caveats in this module.
        is_public_holiday: German public holiday flag, same representative-row
            approximation.
        is_school_holiday: NRW school holiday flag, same approximation.
        is_lecture_period: University lecture-period flag, same approximation.
        mean_air_temperature_c: Mean over every window row with a value, or
            `None` if none had one.
        total_precipitation_mm: Sum over the window (a "how much rain fell"
            total reads more naturally than an average), or `None`.
        mean_wind_speed_ms: Mean over the window, or `None`.
        mean_relative_humidity_pct: Mean over the window, or `None`.
    """

    window_start: pd.Timestamp
    window_end: pd.Timestamp
    day_of_week: int
    is_public_holiday: bool
    is_school_holiday: bool
    is_lecture_period: bool
    mean_air_temperature_c: float | None
    total_precipitation_mm: float | None
    mean_wind_speed_ms: float | None
    mean_relative_humidity_pct: float | None

    @classmethod
    def from_window_rows(cls, window_rows: pd.DataFrame) -> "WindowContext":
        """Builds a `WindowContext` from a station's rows within one window.

        Args:
            window_rows: Rows from an `assemble_feature_history` table
                filtered to one station and one rolling window, at least
                one row.

        Returns:
            The corresponding `WindowContext`.
        """
        representative = window_rows.sort_values("datetime").iloc[-1]
        return cls(
            window_start=window_rows["datetime"].min(),
            window_end=window_rows["datetime"].max(),
            day_of_week=int(representative["day_of_week"]),
            is_public_holiday=bool(representative["is_public_holiday"]),
            is_school_holiday=bool(representative["is_school_holiday"]),
            is_lecture_period=bool(representative["is_lecture_period"]),
            mean_air_temperature_c=_none_if_nan(
                window_rows["weather_air_temperature_c"].mean()
            ),
            total_precipitation_mm=_sum_or_none(
                window_rows["weather_precipitation_mm"]
            ),
            mean_wind_speed_ms=_none_if_nan(
                window_rows["weather_wind_speed_ms"].mean()
            ),
            mean_relative_humidity_pct=_none_if_nan(
                window_rows["weather_relative_humidity_pct"].mean()
            ),
        )


@dataclass(frozen=True)
class StationAccuracyReport:
    """One station's yesterday-total-vs-actual-total accuracy check, plus
    today's fresh rolling next-24h forecast.

    `predicted_total`/`actual_total`/`abs_error`/`pct_diff`/`window_context`
    are all `None` together when the accuracy check couldn't be resolved
    (not enough trailing history ~48h back to reconstruct yesterday's
    forecast - `unresolved_reason` explains why) - this degrades gracefully
    rather than dropping the station entirely, so `forecast_summary` (today's fresh
    forecast) still shows for a station whose accuracy can't be checked. A
    station is only fully dropped from the report (not represented by any
    `StationAccuracyReport` at all) when its raw fetch, weather join, or
    "now" feature row itself is unusable - the existing
    `inference.InferenceError`/`FETCH_ERRORS` case, handled by the caller in
    `scripts/send_daily_email.py`.

    Attributes:
        station_id: Station directory id.
        station_name: Human-readable station name.
        now_datetime: Timestamp of the "now" reading - also the accuracy
            window's end.
        data_age: How old `now_datetime` is relative to when this report
            was built (Europe/Berlin local time, matching
            `pages/station_forecast.py`'s own staleness check) - the
            source has previously gone stale for extended periods (see
            `dashboard_common.STALENESS_WARNING_THRESHOLD`), and without
            this, a stale "now" reading would be reported as if it were
            live. See `is_stale`.
        forecast_summary: Today's fresh rolling next-24h forecast (total +
            peak), from `inference.summarize_forecast_curve`.
        predicted_total: Yesterday's predicted total for the last 24h
            (the window `now_datetime - 24h` to `now_datetime`),
            reconstructed by anchoring `inference.predict_forecast_curve`
            at `now_datetime - 24h` and summing the resulting curve - i.e.
            "what yesterday's forecast for the last 24h would have been."
        actual_total: The actual observed total `total_count` over that
            same window (skipping any missing 15-min readings within it -
            `window_rows` always includes at least `now_datetime`'s own
            reading, which `inference.latest_feature_row` already
            guarantees is non-null, so this can never be "no data at all").
        abs_error: `abs(predicted_total - actual_total)`.
        pct_diff: `(predicted_total - actual_total) / actual_total * 100`;
            positive means the model over-predicted. `None` when
            `actual_total == 0` (percentage is undefined), even though
            `predicted_total`/`abs_error` are still populated in that case.
        window_context: Aggregated weather/calendar context for the window,
            for the AI explanation prompt.
        unresolved_reason: Human-readable reason the accuracy check is
            `None`, or `None` if it was resolved.
    """

    station_id: str
    station_name: str
    now_datetime: pd.Timestamp
    data_age: pd.Timedelta
    forecast_summary: inference.ForecastSummary
    predicted_total: float | None
    actual_total: float | None
    abs_error: float | None
    pct_diff: float | None
    window_context: WindowContext | None
    unresolved_reason: str | None

    @property
    def is_stale(self) -> bool:
        """Whether `now_datetime` is older than `STALENESS_WARNING_THRESHOLD`."""
        return self.data_age > STALENESS_WARNING_THRESHOLD


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
    "today's fresh rolling next-24h forecast" (anchored at "now") and
    "yesterday's predicted total for the last 24h" (anchored at
    `now - 24h`) from that single history, per this module's
    recompute-not-persist design (see module docstring). Both anchors reuse
    the identical `inference.predict_forecast_curve` - no new modeling.

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
            ~48h-back window for the accuracy check, by contrast, is caught
            internally (see `StationAccuracyReport`'s docstring) and does
            not raise.
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
    window_start = now_datetime - YESTERDAY_LOOKBACK

    forecast_curve = inference.predict_forecast_curve(
        model, history, station.station_id, now_datetime
    )
    forecast_summary = inference.summarize_forecast_curve(forecast_curve)

    predicted_total: float | None = None
    actual_total: float | None = None
    abs_error: float | None = None
    pct_diff: float | None = None
    window_context: WindowContext | None = None
    unresolved_reason: str | None = None
    try:
        yesterday_curve = inference.predict_forecast_curve(
            model, history, station.station_id, window_start
        )
    except inference.InferenceError as exc:
        unresolved_reason = (
            "Not enough historical data ~48h ago to reconstruct yesterday's "
            f"forecast for the last 24h: {exc}"
        )
    else:
        predicted_total = inference.summarize_forecast_curve(
            yesterday_curve
        ).total_predicted_count

        window_rows = inference.select_window_rows(
            history, station.station_id, now_datetime, YESTERDAY_LOOKBACK
        )
        # window_rows always contains at least `now_row` itself (its own
        # datetime satisfies the bounds above, and `latest_feature_row`
        # already guarantees a non-null total_count) - so a "no actual
        # readings at all" case can't happen here; `.sum()`'s default
        # skipna=True already handles any other missing 15-min readings
        # within the window by simply excluding them from the sum, which
        # is the same accepted-undercount-on-gaps behavior documented on
        # `inference.ForecastSummary.total_predicted_count`.
        actual_total = float(window_rows["total_count"].sum())
        abs_error = abs(predicted_total - actual_total)
        pct_diff = (
            (predicted_total - actual_total) / actual_total * 100
            if actual_total != 0
            else None
        )
        window_context = WindowContext.from_window_rows(window_rows)

    return StationAccuracyReport(
        station_id=station.station_id,
        station_name=station.name,
        now_datetime=now_datetime,
        data_age=data_age,
        forecast_summary=forecast_summary,
        predicted_total=predicted_total,
        actual_total=actual_total,
        abs_error=abs_error,
        pct_diff=pct_diff,
        window_context=window_context,
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


def _format_window_context(context: WindowContext) -> str:
    """Renders a `WindowContext` as labeled plain-text lines for the AI prompt."""

    def _fmt(value: float | None, unit: str) -> str:
        return f"{value:.1f}{unit}" if value is not None else "unknown"

    return (
        f"  Window: {context.window_start:%Y-%m-%d %H:%M} to "
        f"{context.window_end:%Y-%m-%d %H:%M} ({_DAY_NAMES[context.day_of_week]})\n"
        f"  Public holiday: {context.is_public_holiday}\n"
        f"  School holiday: {context.is_school_holiday}\n"
        f"  Lecture period: {context.is_lecture_period}\n"
        f"  Mean temperature: {_fmt(context.mean_air_temperature_c, 'C')}\n"
        f"  Total precipitation: {_fmt(context.total_precipitation_mm, 'mm')}\n"
        f"  Mean wind speed: {_fmt(context.mean_wind_speed_ms, 'm/s')}\n"
        f"  Mean relative humidity: {_fmt(context.mean_relative_humidity_pct, '%')}"
    )


def build_explanation_prompt(report: StationAccuracyReport) -> tuple[str, str]:
    """Builds the (system, user) prompt asking for a deviation explanation.

    Pure string formatting from `report`'s already-computed fields - no
    network call. Only meaningful for a report with a resolved
    `predicted_total` (i.e. `report.abs_error is not None`); callers should
    only invoke this for reports returned by `select_top_deviations`.

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
            f"Caveat: the window's most recent reading is actually "
            f"{report.data_age} old - the upstream source has not "
            "published newer data yet, so it may not reflect current "
            "real-world conditions."
        )
    caveats_text = ("\n" + "\n".join(caveats) + "\n") if caveats else ""
    user = (
        f"Station: {report.station_name} ({report.station_id})\n"
        f"Yesterday's predicted total for the last 24h: {report.predicted_total:.0f}\n"
        f"Actual total over the last 24h: {report.actual_total:.0f}\n"
        f"Absolute error: {report.abs_error:.0f}{pct_text} - the model {direction}.\n"
        f"{caveats_text}"
        "\n"
        "Conditions during this window:\n"
        f"{_format_window_context(report.window_context)}"
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
        f"{'Station':<28}{'Predicted total (last 24h)':>28}{'Actual total (last 24h)':>26}"
        f"{'% diff':>10}{'Predicted total (next 24h)':>28}"
    )
    lines.append("-" * 120)
    for report in sorted(reports, key=lambda report: report.station_name):
        predicted = (
            f"{report.predicted_total:.0f}"
            if report.predicted_total is not None
            else "n/a"
        )
        actual = (
            f"{report.actual_total:.0f}" if report.actual_total is not None else "n/a"
        )
        pct = f"{report.pct_diff:+.1f}%" if report.pct_diff is not None else "n/a"
        lines.append(
            f"{report.station_name:<28}{predicted:>28}{actual:>26}"
            f"{pct:>10}{report.forecast_summary.total_predicted_count:>28.0f}"
        )
        if report.unresolved_reason:
            lines.append(f"    ({report.unresolved_reason})")
        if report.is_stale:
            lines.append(
                f"    (STALE: 'now' reading is {report.data_age} old - source "
                "hasn't published newer data yet)"
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
            f"  Predicted total (last 24h) {report.predicted_total:.0f}, actual total "
            f"{report.actual_total:.0f} ({pct})"
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
        "bike-count source hasn't retroactively revised past data. Accuracy "
        "is checked as a daily total over a rolling 24h window, not a "
        "single reading - not directly comparable to the model's own "
        "published single-point MAE/RMSE."
    )
    return "\n".join(lines)


def _station_row_notes_html(report: StationAccuracyReport) -> str:
    """Renders one station table row's inline notes (stale/unresolved) as HTML."""
    notes: list[str] = []
    if report.unresolved_reason:
        notes.append(html.escape(report.unresolved_reason))
    if report.is_stale:
        notes.append(
            f"STALE: 'now' reading is {report.data_age} old - source hasn't "
            "published newer data yet"
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
            f"{report.predicted_total:.0f}"
            if report.predicted_total is not None
            else "n/a"
        )
        actual = (
            f"{report.actual_total:.0f}" if report.actual_total is not None else "n/a"
        )
        pct = f"{report.pct_diff:+.1f}%" if report.pct_diff is not None else "n/a"
        table_rows.append(
            "<tr>"
            f'<td style="padding:4px 8px;border-bottom:1px solid #ddd;">'
            f"{html.escape(report.station_name)}{_station_row_notes_html(report)}</td>"
            + row_cells.format(predicted)
            + row_cells.format(actual)
            + row_cells.format(pct)
            + row_cells.format(f"{report.forecast_summary.total_predicted_count:.0f}")
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
            f"<p>Predicted total (last 24h) {report.predicted_total:.0f}, actual total "
            f"{report.actual_total:.0f} ({pct})</p>"
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
        '<th style="padding:4px 8px;">Predicted total (last 24h)</th>'
        '<th style="padding:4px 8px;">Actual total (last 24h)</th>'
        '<th style="padding:4px 8px;">% diff</th>'
        '<th style="padding:4px 8px;">Predicted total (next 24h)</th>'
        "</tr>" + "".join(table_rows) + "</table>"
        "<h2>Notable deviations</h2>"
        + "".join(deviation_sections)
        + dropped_html
        + '<p style="font-size:0.85em;color:#666;">Note: &quot;yesterday&#x27;s '
        "forecast&quot; is recomputed today from the same historical window, "
        "not a stored value - this assumes the upstream bike-count source "
        "hasn't retroactively revised past data. Accuracy is checked as a "
        "daily total over a rolling 24h window, not a single reading - not "
        "directly comparable to the model's own published single-point "
        "MAE/RMSE.</p>"
        "</body></html>"
    )

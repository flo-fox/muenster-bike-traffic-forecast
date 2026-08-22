# Münster Bike Traffic Forecast — Claude Project Guide

## Project goal
24h-ahead prediction of bike traffic volume per counting station in Münster,
based on historical 15-minute count data combined with weather data. Existing
visualizations (Klimadashboard Münster, Code-for-Münster "traffic-dynamics")
only show raw data; the only prior forecasting attempts are frozen 2018
regression experiments. Goal here is an actual, active forecasting tool.

## Tech stack
- Python, Jupyter notebooks
- `pandas` for data, `requests` for HTTP
- ML: `HistGradientBoostingRegressor` (sklearn, notebook 08) and `LightGBM`
  (notebook 09) as the tree-ensemble candidates, `Prophet` (notebook 10) as
  a per-station seasonal-decomposition alternative, `MLPRegressor` (sklearn,
  notebook 11) as a first neural-net pass, `RandomForestRegressor` (sklearn,
  notebook 14) as a fourth tree-based class — see "Model selection
  rationale" below for why these were tried and not linear regression/a
  single tree/SVM. **Production model: LightGBM** (`notebooks/
  18_lightgbm_production_model.ipynb`, `models/production_lightgbm.joblib`,
  decided 2026-08-22) — see that entry and the tech-stack history it
  links for why: random forest (notebook 17's production config) was the
  clear winner on the dataset available through 2026-08-21, but a
  routine data refresh the very next day flipped the ranking, with
  GBM/LightGBM edging out random forest; LightGBM and GBM were
  essentially tied on accuracy, and LightGBM was chosen as architecturally
  simpler to keep as the single production artifact (dramatically
  smaller/faster to train than a tuned random-forest alternative that
  could have reclaimed a narrow lead). A sequence-aware architecture
  (LSTM/GRU + a learned `station_id` embedding) was researched but
  deferred rather than built — see "Model selection rationale" for why
  — and remains the option if the tree ensembles' ceiling ever needs
  breaking through.
- `matplotlib` / `plotly` for visualization
- Streamlit for the live dashboard (`app.py`, `dashboard_common.py`, `pages/`)
- `anthropic` SDK for the daily forecast-accuracy email's AI explanations
  (`scripts/send_daily_email.py`, direct API call — see "Planned additions")

## Model selection rationale
Why tree-ensembles and Prophet were tried, and why the classic
alternatives below were not (or not yet) — see notebook 07 for the
underlying data patterns referenced here:

- **Linear regression**: not tried. The target has strong non-linear and
  interaction effects — a sharp hour×day-of-week interaction (bimodal
  weekday commute peaks vs. a flat weekend afternoon plateau), a
  non-monotonic wind-speed relationship, and a thresholded (not linear)
  precipitation effect (near-zero effect until it actually rains, then a
  step down). Capturing this linearly would need hand-built
  interaction/spline terms plus one-hot encoding of 23 stations spanning a
  ~20x volume range — trees get this for free via splits.
- **A single decision tree**: not tried standalone — high-variance on
  ~2.3M rows with a mixed numeric/categorical feature set; it would
  overfit without the boosting/bagging an ensemble provides, which is
  exactly what the two models actually used are.
- **SVM / SVR**: not tried. Kernel SVR training cost scales roughly
  O(n²)-O(n³), impractical at ~2.3M rows without subsampling away most of
  the value of having years of 15-minute history. It also has no native
  categorical/missing-value handling (would need one-hot encoding
  `station_id` and imputing every lag/rolling-feature null near each
  station's start of coverage) — a poor fit for both scale and feature
  shape here.
- **HistGradientBoostingRegressor / LightGBM** (both tried): native
  missing-value support (lag/rolling features are null near each
  station's start of coverage) and native categorical support
  (`station_id`, `hour`, `day_of_week`, …) with no manual encoding, scale
  to 2.3M rows without subsampling, and tree splits capture the
  interaction/threshold effects above automatically. **Result (current,
  post-2026-08-22 data refresh): near-tied, LightGBM very slightly ahead**
  (MAE 10.46 vs. 10.48, RMSE 17.99 vs. 18.09) — the choice between these
  two still barely matters for this problem, and — new since the
  2026-08-21 double-counting fix — **this pair now edges out random
  forest** (see that entry below and the 2026-08-22 "Planned additions"
  entry for the full story).
- **Prophet** (tried): a genuinely different approach (explicit additive
  seasonal decomposition, one model per station) rather than another tree
  ensemble, specifically to test whether per-station fitting handles the
  regime-shift station (`300038855`) better than one global tree model
  that shares structure across all 23 stations. Result (notebook 10,
  current numbers): **worse than both alternatives** (overall MAE
  21.96/RMSE 35.36, vs. 16.66/31.11 baseline and 10.48/18.09 GBM) —
  including on several individual stations, which get *much* worse under
  Prophet (worst: `300037932` -180%, `300038855` -146%, `300037405`
  -117% vs. baseline). Likely cause: Prophet never sees the current
  `total_count` reading as an input (it's a pure trend/seasonality
  extrapolator), while both the baseline and GBM directly exploit it as
  their single strongest signal.
- **MLPRegressor** (tried, notebook 11): a first neural-net pass, per
  user request — sklearn's `MLPRegressor` on the identical feature set,
  with an explicit preprocessing pipeline (median imputation +
  standardization for numeric features, one-hot encoding for
  categoricals) since, unlike the tree models, it has no native
  missing-value or categorical support. **Result (current)**: MAE 10.80,
  a real (not noise-level) ~3% behind GBM (10.48) and LightGBM (10.46),
  RMSE 18.77 similarly ~4% behind both — bigger occasional misses on top
  of a somewhat worse typical-case error. Notably reverses its earlier
  (pre-refresh) flagged-station finding: MLP now *beats* the baseline on
  both previously-flagged stations (`300037405` +32.9%, `300038855`
  +35.7%) rather than regressing on one of them — see the 2026-08-22
  "Planned additions" entry on why several per-station results moved
  this much with the data refresh. Still not competitive enough overall
  to change the "tree splits already capture this data's structure"
  conclusion — a first-pass neural net still adds preprocessing/tuning
  overhead without a performance gain here.
- **Random forest** (tried, notebook 14): a fourth tree-based class,
  `sklearn.ensemble.RandomForestRegressor` on the identical feature set
  and chronological holdout as 08/09/11 — the last open item from this
  list. Like the MLP it has no native missing-value/categorical support
  (same median-imputation + one-hot preprocessing), but no scaling, since
  tree splits are scale-invariant. Needed an empirical detour: an initial
  full-bootstrap configuration (`n_estimators=60`, `max_depth=16`, no row
  subsampling) missed a 9-minute cell timeout — full-size, un-binned
  trees on ~2.16M rows are memory-bandwidth-bound in a way the
  histogram-binned GBM/LightGBM aren't, so parallel speedup across cores
  fell short of expectations. `max_samples=0.15` bootstrap subsampling
  fixed that. **Result (current, base config): no longer the best model
  tried** — MAE 10.93/RMSE 19.04 overall, vs. 10.48/18.09 (GBM) and
  10.46/17.99 (LightGBM) — GBM now beats base random forest on 21 of 23
  stations. The tuned production config (notebook 17: `max_samples=0.30`
  + `weekend_weekday_ratio`, MAE 10.73/RMSE 18.62) narrows the gap but
  GBM still wins on 18 of 23 stations. This is a **reversal** of the
  finding that held through 2026-08-21 (random forest clearly best,
  ~4-5% MAE edge over GBM) — see the 2026-08-22 "Planned additions"
  entry for the fresh-data-refresh context and why this isn't yet being
  treated as fully settled either way. What *is* still robust: per-station,
  random forest improves on the baseline for 23 of 23 stations (up from
  22/23) — the most consistent of any model tried, including on the two
  previously-flagged stations (`300038855`, `300037405`), where notebook
  12 traced the earlier "regime shift" to a sensor-outage artifact rather
  than real traffic change.
- **Sequence-aware architecture (LSTM/GRU + `station_id` embedding)**:
  researched, not built — deferred rather than left simply unconsidered.
  Three structurally different model classes (GBM, LightGBM, MLP)
  converge to within a few percent of each other, and GBM's permutation
  importance (notebook 08, current numbers) drops off after the top
  handful of features (`total_count`, `day_of_week`, `hour`,
  `station_id`, `rolling_mean_2h`, `lag_1w`, `lag_1d`, roughly a 3.5x
  cliff down to the next feature, `rolling_mean_24h`) — together read as
  the current lag/rolling feature set approaching a noise floor rather
  than being under-exploited by any one model class. (Prophet's much
  worse score doesn't fit this convergence story — see above, it's
  explained separately by never seeing `total_count` as an input.)
  Reshaping ~2.3M flat rows into per-station sequences — handling real
  per-station gaps and differing coverage-start dates — was judged a
  bigger lift than notebooks 08-14 combined, so it was deprioritized
  rather than attempted. Random forest's edge over GBM has since
  reversed (see above) rather than persisted, which weakens (but doesn't
  eliminate) the "some headroom exists via a different bias-variance
  tradeoff" argument for revisiting this deferral - the tree-ensemble
  field is now genuinely closer to a three-way near-tie (GBM, LightGBM,
  RF) than a settled random-forest lead. If ever revisited: PyTorch over
  Keras/TensorFlow, a GRU/LSTM with a learned `station_id` embedding.

## Data sources
- **Bike counts**: GitHub repo `od-ms/radverkehr-zaehlstellen` — 15-minute
  counts, 24 stations, data since 2019 or 2023 depending on station. Open
  data, no credentials needed.
- **Weather data**: DWD (Deutscher Wetterdienst) Open Data portal
  (opendata.dwd.de). Open data, no credentials needed.

## Code conventions
- Type hints on all function signatures.
- Docstrings (Google style) for every function/class.
- Prefer pure functions that accept and return data; keep side effects at the
  notebook level.
- Raise meaningful exceptions rather than returning `None` on error.
- Format with **black** (`line-length = 88`).

## Notebook conventions
- One notebook per analysis/modeling stage, numbered (`01_`, `02_`, …).
- Keep notebooks reproducible: a fresh `Restart & Run All` must succeed.
- Store reusable logic in `src/muenster_bike_forecast/`, not inline in
  notebooks.

## Visualization conventions
- Every chart/plot (matplotlib, plotly, or otherwise) must include a source
  description — e.g. a small caption or footer text naming the underlying
  data source(s) (bike counts: `od-ms/radverkehr-zaehlstellen`; weather: DWD
  Open Data; etc.). Add this to existing charts too when touching a
  notebook that has any without one, not just new charts going forward.

## Review checklist
Before treating a step as done, check it from three angles:
- **Data engineer**: does fetched/raw data get schema-validated? Are missing
  15-min intervals and per-station gaps handled explicitly, not silently
  dropped? Are fetch scripts idempotent and reproducible?
- **Data scientist**: does any train/test split respect time order (no future
  leaking into past) and station boundaries? Are baseline metrics reported
  alongside model metrics so improvements are provable, not assumed?
- **Security analyst**: no hardcoded credentials or tokens, even though
  current data sources are open/credential-free. Treat fetched CSV/HTTP
  content as untrusted input (validate shape/types before use, no
  `eval`/`pickle` on external data).

## Planned additions
- Geo/location map done (2026-07-24): station coordinates are geocoded via
  OpenStreetMap Nominatim (`src/muenster_bike_forecast/data/geocode.py`),
  cached to `data/raw/bike_counts/station_locations.csv` (all 23 stations
  resolved; 6 needed a manually-chosen fallback query, documented in the
  cache/notebook rather than fabricated). Section 5 of
  `notebooks/07_descriptive_analysis.ipynb` plots a lon/lat scatter map
  sized/colored by all-time mean traffic.
- Cross-check verified (2026-07-24): none of notebook 07's three modeling
  suggestions made it into notebook 08. It still uses raw
  `weather_precipitation_mm` (not the bucketed form 07 recommends), plain
  separate `hour`/`day_of_week` categoricals (no explicit hour×day-of-week
  interaction feature, despite 07 showing a strong interaction), and no
  per-station weekend/weekday-ratio commuter-vs-leisure flag. Candidate
  follow-up feature-engineering pass for a future modeling notebook — not
  implemented here to keep 09/10's model-class comparison against 08
  apples-to-apples (same feature set, different model only).
- Distance-from-city-center feature: done (2026-07-26,
  `notebooks/13_distance_feature_test.ipynb`). Notebook 07's geo-map
  section had found a moderate correlation (r ≈ -0.51) between a
  station's distance from the Prinzipalmarkt/Dom center and its mean
  traffic (busier stations cluster on the central Altstadt ring; quieter
  ones sit 2-4km out) — but with notable exceptions (*Wolbecker
  Straße*, *Lütkenbecker Str.*), so it wasn't a clean cutoff going in.
  Tested by adding `distance_from_center_km` (haversine distance to the
  same reference point) to notebook 08's exact feature set and
  retraining an otherwise-identical `HistGradientBoostingRegressor` on
  the identical split. **Result: exactly zero effect** — overall and
  per-station MAE/RMSE (including both flagged stations) identical to
  six decimal places with vs. without the feature, `control_prediction
  == plus_distance_prediction` for every test row, and the feature's
  permutation importance is 0.0. Cause: `distance_from_center_km` is a
  deterministic, many-to-one function of `station_id`, which
  `HistGradientBoostingRegressor`'s categorical splits already exploit
  fully — an explicit distance feature adds nothing the model can't
  already extract from `station_id` alone.
- Streamlit dashboard scaffolded and iterated on (2026-07-27): a multipage
  app (`app.py` as a thin `st.navigation` entry point, shared logic in
  `dashboard_common.py`, pages under `pages/`) with three views — **Station
  forecast** (live 24h-ahead forecast for one station, 7-day observed +
  rolling-24h-average + continuous next-24h forecast curve, via
  `src/muenster_bike_forecast/inference.py`), **Station comparison**
  (cross-sectional bar chart + table across all 23 stations), and **City
  map** (live OpenStreetMap view, colored by forecast; markers get a solid
  dark "halo" trace underneath since `go.Scattermap` markers have no
  border property and a sequential colorscale's light end otherwise
  disappears into the tile colors). Also found and fixed: the live-fetch
  lookback window was too tight (10 days) to reliably span two calendar
  months, silently shortening available history when `as_of` fell late in
  a month — widened to 35 days. Along the way, confirmed the upstream bike-
  count source (`od-ms/radverkehr-zaehlstellen`) normally publishes daily
  (verified via its own commit history) but has an active ~3-week gap as of
  this session — the dashboard surfaces this as a visible staleness
  warning rather than silently presenting stale data as current.
- **Daily-vs-15-min display question resolved (2026-08-21)**: the
  2026-07-27 open question (given the source only updates about daily,
  does a 15-minute-resolution forecast overstate the freshness/precision
  actually available?) is now implemented on both the dashboard and the
  daily email, via a new shared pure-aggregation primitive
  (`inference.summarize_forecast_curve`/`ForecastSummary`, built entirely
  from the already-existing `inference.predict_forecast_curve` - no
  retraining, no new modeling):
  - **Dashboard** (`pages/station_forecast.py`): the primary display is
    now "Predicted traffic, next 24h: `<sum>`" + a weekday-qualified
    "Peak expected: `<day HH:MM>`" callout, anchoring
    `predict_forecast_curve` at "now" and aggregating. Deliberately a
    **rolling** next-24h window, not a calendar-day ("tomorrow") total -
    chosen specifically to avoid the extra curve-construction complexity a
    true calendar-day alignment would need. The old single-point "Forecast
    in 24h" metric is kept as demoted "Point-in-time detail" below a
    divider, not removed. The 15-minute-grid forecast curve chart is
    unchanged (still the right resolution for the model's real target
    grain), just no longer the first thing shown.
  - **Daily email** (`daily_report.py`): the backward-looking accuracy
    check moved from "yesterday's point prediction vs. today's single
    actual reading" to "yesterday's predicted total for the last 24h vs.
    the actual total over that same window" - reusing
    `predict_forecast_curve` anchored at `now - 24h` (which naturally
    yields predictions targeting exactly the window ending "now"), summed
    and compared against the sum of actual readings in that window. A
    single 15-minute actual reading is noisy (sensor blips, one
    unlucky/lucky interval) and represents the model's real day-ahead
    skill worse than a window sum does. The forward-looking "Forecast
    +24h" column changed the same way, reusing `ForecastSummary`, but
    **without** the peak-time callout (a 23-station daily digest table is
    a worse fit for a per-station peak-time than the dashboard's
    single-station live view). This whole redesign also quietly retired
    the old `select_row_near`/`YESTERDAY_ROW_TOLERANCE`/timing-gap
    machinery - summing over a window sidesteps "is there a reading at
    exactly the right instant" entirely. **Note**: the email's new
    daily-total accuracy metric is a genuinely different, coarser metric
    than the model's own published single-point 24h-ahead MAE/RMSE (see
    notebooks 08-17) - not directly comparable to those numbers, and the
    email's own closing disclaimer now says so.
  - **Station comparison and City map** (`pages/station_comparison.py`,
    `pages/city_map.py` via `dashboard_common.render_station_map`): same
    "Actual (last 24h)" / "Predicted (next 24h)" framing as the email,
    replacing the old `current_total_count`/`forecast_value` 15-minute
    point fields in `build_fleet_snapshot`'s per-station rows. Needed a
    new small shared primitive, `inference.select_window_rows` (pure row
    selection for "a station's rows within a rolling window ending at some
    timestamp"), factored out of `daily_report.build_station_report`'s
    previously-inline window logic and reused by both
    `dashboard_common.build_forecast` (for `actual_total_24h`) and
    `daily_report.py` (behavior-preserving refactor, no output change).
    No demoted secondary section on these two pages, unlike the station-
    forecast page - a 23-station bar chart/table/map is a worse fit for
    carrying both framings than a single-station detail page.
- **`chronological_split` gained a default 24h embargo (2026-08-17)**:
  `modeling/model_table.py`'s `chronological_split` previously put every
  row strictly before the cutoff into train, but a row's *label* (added
  by `add_forecast_target`) is observed 24h after its own timestamp — so
  a train row just before the cutoff carried a label from just after it,
  inside the nominal test window. Fixed by adding an
  `embargo: pd.Timedelta = DEFAULT_HORIZON` parameter that now excludes
  `[cutoff - embargo, cutoff)` from train by default (pass
  `embargo=pd.Timedelta(0)` to reproduce the old behavior). **Re-run
  complete (2026-08-19)**: all ten notebooks (06, 08, 09, 10, 11, 13, 14,
  15, 16, 17) have had a fresh Restart & Run All against the new embargo
  default — impact was indeed small, as expected (the dropped window is
  ~96 rows/station out of ~2.16M train rows). Test-set metrics barely
  moved since the embargo only trims train rows near the cutoff, not the
  test window itself: baseline unchanged (39.34/77.25), GBM 28.73/54.69
  (was 28.57/54.53), LightGBM 28.36/54.29 (was 28.56/54.44), MLP
  29.13/56.68 (was 28.57/55.57), random forest (notebook 14 config)
  27.42/54.17 (was 27.34/54.13). **The production model's headline
  number is now MAE 27.14 / RMSE 53.71** (`RandomForestRegressor`,
  `max_samples=0.30`, base feature set + `weekend_weekday_ratio`,
  notebook 17), down from the previously-quoted 27.07/53.70 — still the
  best model tried, still the recommended production config, no
  conclusion in this file changes.

  One real bug surfaced along the way, now fixed: notebook 16 has a
  built-in reproduction check against notebook 14's number
  (`np.isclose(...)` against a hardcoded constant) that started printing
  `False` once 14 was re-run with the new embargo default and 16 wasn't
  updated to match — the hardcoded GBM/LightGBM/MLP/RF-target reference
  constants in notebook 16's cell were stale pre-embargo values. Fixed by
  updating those constants to the fresh post-embargo numbers above and
  re-running; the check now prints `True`.

  **Known residual staleness, not fixed**: notebooks 09, 11, 14, 15, and
  17 also carry hardcoded cross-notebook reference constants (e.g.
  notebook 15's `RF_BASELINE_OVERALL` still cites notebook 14's
  *pre-embargo* 27.343144/54.130155 rather than 14's fresh
  27.415694/54.172414) copied in from before the embargo fix, the same
  pattern notebook 16 had. Unlike 16, none of these assert a pass/fail
  check that now reads as visibly wrong — they're display-only
  comparison rows/prose — and every affected difference is sub-1%
  relative, so no ranking or written conclusion in this file or those
  notebooks changes if corrected. Left as-is to avoid another full
  retraining cascade (09 → 11 → 14 → 15 → 16 → 17, since fixing one
  hardcoded reference requires re-running the notebook that hardcodes
  it) for a display-only inconsistency; pick up here if those notebooks
  are touched again for another reason anyway.
- **Daily forecast-accuracy email built (2026-08-19,
  `feature/daily-forecast-email` branch)**: the "daily email" half of the
  2026-07-26 dashboard decision (GitHub Actions + Gmail app password —
  see the earlier "Web dashboard plan" entry above) is now actually
  implemented, not just decided. `scripts/send_daily_email.py` (thin
  orchestration) + `src/muenster_bike_forecast/daily_report.py` (pure
  comparison/formatting logic, tested in `tests/test_daily_report.py`)
  check every station's forecast from ~24h ago against today's actual
  reading, show today's fresh 24h-ahead forecast too, and — new since
  the original 2026-07-26 plan — ask Claude (`claude-haiku-4-5`, direct
  API call) for a short, context-grounded explanation of the top 5
  biggest deviations. Runs via
  `.github/workflows/daily_forecast_email.yml`, ~09:00 Münster local
  time (two DST-gated cron entries, since cron itself isn't DST-aware).
  Design decisions worth knowing before touching this again:
  - **No forecast-persistence mechanism** — "yesterday's forecast" is
    *recomputed* each run, not stored: fetching today's ~35-day window
    inherently includes yesterday's actual counts too, and
    `assemble_feature_history`'s lag/rolling features are computed
    relative to each row's own timestamp, so predicting from the row
    ~24h before now reproduces what running it yesterday would have
    given. **Accepted risk, not detected**: this assumes the upstream
    bike-count source doesn't retroactively revise a past monthly CSV
    between yesterday and today.
  - **Direct Anthropic API call, not an MCP server** — considered as an
    alternative (see the "Next dev priorities" project memory) and
    rejected for this specific use case: an MCP server would need to run
    persistently to be reachable from a one-shot cron job, which doesn't
    fit a process that runs once daily and exits. MCP remains a live,
    separate idea for other future use cases (e.g. interactive
    forecast-accuracy queries from a chat session).
  - **A Claude Pro/Max subscription does not cover this** — confirmed
    with the user; `ANTHROPIC_API_KEY` needs its own
    console.anthropic.com account/billing, set as a GitHub Actions
    secret alongside the three Gmail/recipient secrets (see README.md).
  - **Rate-limit exposure, partially mitigated since 2026-08-21**:
    fleet-wide scope (23 stations × ~2 months per run ≈ 46 requests to
    `raw.githubusercontent.com`) hits the same unthrottled source the
    dashboard does, which has previously shown real HTTP 429s under
    repeated fetching. At the time this was written, no retry/backoff
    existed and none was added here deliberately (would have been
    speculative scope beyond what was asked); `bike_counts.py` later
    gained a real one (`_get_with_retry`, added 2026-08-21 for the
    double-counting-fix retrain's own unattended live fetch — see that
    entry below), and since `list_stations`/`fetch_station_month` are
    the same functions this script calls, the daily email now inherits
    that retry/backoff automatically, without any change needed here.
    Remaining mitigation, unchanged: a per-station `SCRIPT_FETCH_ERRORS`
    catch (skip and continue, same pattern as
    `dashboard_common.build_fleet_snapshot`) plus a small 0.2s courtesy
    delay between station fetches.
  - **Top-5-by-absolute-error default** (`daily_report.TOP_N_DEVIATIONS`)
    — user-confirmed, but arbitrary; revisit if 5 turns out too many/few
    once real emails start arriving.
- **`total_count` double-counting bug found, fixed, and the full model
  chain retrained (2026-08-21/22, `fix/channel-double-counting` branch)**:
  a user manually cross-checking a dashboard number (Neutor: 146) against
  its raw channel CSV found it was exactly double a plausible value.
  Root cause: `compute_total_count` summed a station's "combined" channel
  (numeric id == `station_id`) *on top of* its own directional
  sub-channels, which already equal it — confirmed systemic across **all
  23 stations, 100% match, zero exceptions**, via a new read-only
  diagnostic (`combined_channel_matches_directional_sum`). Fix, per
  explicit user instruction ("use the first column"): `compute_total_count`
  now selects the combined channel directly via `coalesce_channel_columns`
  rather than summing (`fece051`); `inference.py`'s call site updated to
  match. Same session, at the user's request, added a new "Inbound vs.
  outbound imbalance" section to notebook 07 (`compute_directional_totals`
  in `analysis/descriptive.py`) — Weseler Straße's ~28% imbalance is the
  standout; Kanalpromenade Abschnitt 6 (~62%) and Bismarckallee (~7%) are
  flagged unreliable (concurrent-channel-overlap and mid-history
  relabeling issues respectively, not fixed, honestly documented
  instead). Gasselstiege's specific percentage originally quoted here
  (~50%) was corrected 2026-08-22 - the current, correctly-computed
  figure is ~0.1% (essentially balanced), not ~50%; the earlier number
  was a transcription error from before this session's live-data
  refresh, not a change in the underlying data. The methodological
  caveat (concurrent, not sequential, channel generations at this
  station - see `compute_directional_totals`'s docstring) still applies
  and still means any single number quoted for this station, including
  this corrected one, should be treated with the same caution as before.

  The retrain that followed was explicitly two-staged, per the user's
  request, to isolate cause from coincidence:

  - **Stage 1** (`42b4b13`, `8d18184`): re-ran notebooks 08-17 against
    the already-corrected `model_table.csv`, with **no new data
    fetched** — isolating the fix's effect alone. Every model's MAE/RMSE
    dropped by very close to half (baseline 39.34 → 19.67; confirmed via
    independent audit that the old/new per-station baseline-MAE ratio
    was **exactly 0.5000 for all 23 stations**, not just "roughly half"
    in aggregate) — expected, since the bug was double-counting the
    target for every affected station uniformly. The model ranking held:
    random forest still best (notebook 17 production config: MAE
    13.56/RMSE 26.83, a ~6.4% MAE edge over GBM's 14.49; the base
    random-forest config, notebook 14, MAE 13.71, edges GBM by ~5.4%,
    up slightly from the pre-fix ~4.3%) — GBM/LightGBM near-tied, MLP
    tied with GBM on MAE but worse RMSE, Prophet clearly worst. This
    retrain also surfaced (and
    fixed) a much bigger-than-previously-documented case of the
    "hardcoded cross-notebook reference constant" staleness pattern
    already noted below under the 2026-08-17 embargo entry: after the
    2x-rescale, notebooks 09/10/11/14/15/16's hardcoded "reference"
    MAE/RMSE constants (both overall and per-station) were off by
    40-55 percentage points, not sub-1% — e.g. notebook 16 was printing
    "RF beats GBM on 23 of 23 stations" with a fabricated ~49% margin
    (true margin then ~5%), and notebook 10 made Prophet look
    competitive with GBM (actually ~90% worse). All fixed by
    programmatically extracting fresh ground truth from each source
    notebook's own already-executed output (no retraining, no
    transcription risk) rather than hand-typing corrected numbers.
  - **Stage 2** (`567d496`, `ee715ca`): re-fetched live data (notebooks
    01/02, ~2-3 more months since the last fetch, using the new
    `_get_with_retry` backoff below) and re-ran the entire chain 01-17.
    Surfaced a real, previously-latent bug: notebook 03's station-file
    glob crashed on `station_locations.csv` (its `NON_STATION_FILES`
    denylist predated that file, added 2026-07-24) — fixed with a
    positive match on the numeric station-id filename convention
    instead of an ever-growing denylist. **Significant, independently-
    verified finding: on this larger/fresher dataset, the model ranking
    partially reversed** — GBM (MAE 10.48) and LightGBM (10.46) now both
    edge out random forest (production config 10.73, base config
    10.93); GBM beats even the tuned production RF config on 18 of 23
    stations. Random forest's per-station consistency actually
    *improved* (beats baseline on 23/23 stations now, up from 22/23),
    just no longer with the lowest aggregate error. Two independent
    `bike-data-scientist-auditor` passes confirmed this is real (fair
    apples-to-apples comparison — identical train/test rows, features,
    `random_state` — and broad across stations, not a 1-2 station
    artifact), not a bug. **This directly contradicts the "Model
    selection rationale" section's and this file's tech-stack bullet's
    prior claim that random forest is simply "the best model" — both
    have been updated to reflect that no single model class is
    currently the settled winner.** The stale-hardcoded-reference-
    constant issue recurred for Stage 2's fresh numbers (same root
    cause as Stage 1's, not re-fixed a second time given the scale of
    manual work already spent) — every number in this entry and in the
    "Model selection rationale" section above was pulled directly from
    each notebook's own trained-model output, not from any notebook's
    own "vs. GBM"-style comparison table, which should not be trusted
    until that architectural gap is closed (e.g. by having each
    notebook write its metrics to a shared file instead of hardcoding
    cross-notebook constants — a real fix for a bug class that has now
    caused two separate incidents, not yet built).
  - **Resolved 2026-08-22: switched to LightGBM.** The question below
    ("which model to actually ship") was open for a short while after
    the ranking reversal above - kept here rather than deleted, since it
    explains the tradeoffs that were actually weighed. A follow-up
    re-measurement of notebook 16's Section 7 hyperparameter sweep (also
    2026-08-22, against the current data) had added a relevant data
    point: random forest's reference/production configs no longer lead,
    but a more aggressively-tuned config (`max_samples=0.30,
    n_estimators=100, max_depth=18, min_samples_leaf=5`) reaches MAE
    10.475 - narrowly *ahead* of GBM (10.483) again, though not quite
    matching LightGBM (10.461) - at a real cost (~9x the reference
    config's serialized model size - fit-time comparisons across sweep
    configs proved unreliable between repeat runs, see notebook 16
    Section 7), and not re-checked per-station the way the reference
    config was. So random forest *could* still lead with enough tuning;
    it just no longer did so "for free" at the previously-recommended
    settings. User's decision: **switch to LightGBM** (`notebooks/
    18_lightgbm_production_model.ipynb`, `models/
    production_lightgbm.joblib`) - essentially tied with GBM on accuracy
    (10.461 vs. 10.483 MAE) but architecturally simpler to keep as the
    single production artifact, and dramatically smaller/faster to train
    than the tuned random-forest alternative (0.3MB vs. random forest's
    42.8MB, ~6s fit vs. minutes). This also retired
    `weekend_weekday_ratio` from the live inference path entirely
    (`inference.py`, `dashboard_common.py`, `scripts/send_daily_email.py`,
    `daily_report.py`) - that feature was only ever validated against
    random forest (notebook 15's own scope), never against LightGBM, so
    shipping it unvalidated with a different model class wasn't done;
    `models/weekend_weekday_ratio.csv` and `production_random_forest.joblib`
    were removed (not kept as a "rollback" hedge - `inference.FEATURE_COLS`
    losing `weekend_weekday_ratio` means the old RF artifact couldn't be
    fed by the changed `inference.py` in isolation anyway, so a real
    rollback is "revert the commit," which git history already provides;
    keeping a stale, already-incompatible 42.8MB binary around bought
    nothing). A `FixedCategoryCaster` transformer
    (`src/muenster_bike_forecast/modeling/lightgbm_features.py`) is baked
    into the saved `Pipeline` so LightGBM's categorical-encoding
    consistency requirement (same fixed categories at fit and predict
    time) holds even for `inference.predict_24h_ahead`'s single-row
    predictions - verified both by a dedicated unit test
    (`tests/test_lightgbm_features.py`) and directly inside notebook 18
    itself (single-row vs. batch prediction on a real test row, bit-
    identical).
  - **Also added this session**: `_get_with_retry` exponential-backoff
    wrapper in `bike_counts.py` (5s base, doubling, 3 retries) around
    `list_stations`/`fetch_station_month`, specifically for this
    retrain's unattended live fetch — closes discarded retryable-status
    responses to avoid leaking sockets, per code review. Two new
    on-demand review subagents: `bike-data-auditor` (cross-checks
    displayed numbers against raw source data — the role a human did
    manually to catch this whole bug) and `bike-data-scientist-auditor`
    (independently re-derives modeling methodology — leakage, fair
    baselines, metric correctness, effect size vs. noise; used
    extensively during this retrain to verify the ranking reversal
    above). Both on-demand only, not wired into any hook.

## What Claude should avoid
- Do not add speculative features beyond what is asked.
- Do not rewrite working code just for style; only touch what is needed.
- Do not commit to an ML library or web framework before it's actually needed
  for the step at hand.

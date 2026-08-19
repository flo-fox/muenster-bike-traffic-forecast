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
  notebook 14) as a fourth tree-based class and currently the best model
  tried — see "Model selection rationale" below for why these were tried
  and not linear regression/a single tree/SVM, and for the random-forest
  result. A sequence-aware architecture (LSTM/GRU + a learned `station_id`
  embedding) was researched but deferred rather than built — see "Model
  selection rationale" for why — and remains the option if the tree
  ensembles' ceiling ever needs breaking through.
- `matplotlib` / `plotly` for visualization
- Possibly a web dashboard later (framework still open)

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
  interaction/threshold effects above automatically. Result: near-tied
  (28.57 vs. 28.56 MAE overall) — the choice between these two barely
  matters for this problem.
- **Prophet** (tried): a genuinely different approach (explicit additive
  seasonal decomposition, one model per station) rather than another tree
  ensemble, specifically to test whether per-station fitting handles the
  regime-shift station (`300038855`) better than one global tree model
  that shares structure across all 23 stations. Result (notebook 10):
  **worse than both alternatives** (overall MAE 54.30/RMSE 92.01, vs.
  39.34/77.25 baseline and 28.57/54.53 GBM) — including on the two
  flagged stations, which get *worse* under Prophet, not better. Likely
  cause: Prophet never sees the current `total_count` reading as an
  input (it's a pure trend/seasonality extrapolator), while both the
  baseline and GBM directly exploit it as their single strongest signal.
- **MLPRegressor** (tried, notebook 11): a first neural-net pass, per
  user request — sklearn's `MLPRegressor` on the identical feature set,
  with an explicit preprocessing pipeline (median imputation +
  standardization for numeric features, one-hot encoding for
  categoricals) since, unlike the tree models, it has no native
  missing-value or categorical support. Result: MAE 28.57 essentially
  tied with GBM (28.57) and LightGBM (28.56), RMSE 55.57 slightly worse
  than both (~54.5) — bigger occasional misses despite a comparable
  typical-case error. Does not fix either flagged station cleanly: worse
  than both trees on the mild regression (`300037405`, -16.5% vs.
  baseline vs. their ~-8%), modestly better than both trees but still far
  from the baseline on the severe regime-shift station (`300038855`,
  +4-7% over the trees, still -102% vs. baseline). Confirms the same
  conclusion as the tree-vs-Prophet comparison: this data's structure is
  already well captured by tree splits, so a first-pass neural net adds
  preprocessing/tuning overhead without a performance gain.
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
  cut the actual fit to 91.4s. **Result: the best model tried so far** —
  MAE 27.34/RMSE 54.13 overall, vs. 28.57/54.53 (GBM), 28.56/54.44
  (LightGBM), 28.57/55.57 (MLP) — a real ~4.3% MAE improvement, not a
  near-tie like the other three are with each other. Per-station, it
  improves on the baseline for 22 of 23 stations (the most consistent of
  any model tried), including a large apparent win on `300038855` — but
  notebook 12 (run in parallel) subsequently found that station's test
  window is ~90% sensor-gap-or-all-zero artifact, not real traffic, so
  that specific win is likely a near-zero prediction scoring well against
  a corrupted near-zero target rather than genuine regime-shift
  robustness; treat it with the same caution as the smaller, less
  artifact-prone win on the other flagged station (`300037405`, +14.8%
  vs. baseline).
- **Sequence-aware architecture (LSTM/GRU + `station_id` embedding)**:
  researched, not built — deferred rather than left simply unconsidered.
  Three structurally different model classes tried so far (GBM,
  LightGBM, MLP) converge to within ~0.05 MAE of each other (28.56-
  28.57), and GBM's permutation importance (notebook 08) drops off
  sharply after the top handful of features (`total_count`, `day_of_week`,
  `hour`, `station_id`, `lag_1w`, `rolling_mean_2h`, roughly a 3.7x cliff
  down to the next feature, `lag_1d`) — together read as the current
  lag/rolling feature set approaching a noise floor rather than being
  under-exploited by any one model class. (Prophet's much worse score,
  54.30 MAE, doesn't fit this convergence story — see above, it's
  explained separately by never seeing `total_count` as an input.)
  Reshaping ~2.3M flat rows into per-station sequences — handling real
  per-station gaps and differing coverage-start dates — was judged a
  bigger lift than notebooks 08-14 combined, so it was deprioritized
  rather than attempted. Random forest's subsequent ~4.3% MAE gain
  (notebook 14) shows some headroom still exists within the current
  feature set via a different bias-variance tradeoff, which doesn't
  overturn the deferral but is worth noting against the "noise floor"
  framing. If ever revisited: PyTorch over Keras/TensorFlow, a GRU/LSTM
  with a learned `station_id` embedding.

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
- **Open question for next session**: given the source only updates about
  daily (per above), does a 15-minute-resolution forecast curve overstate
  the freshness/precision actually available? Discussed 2026-07-27:
  leaning towards keeping the *model* at 15-minute resolution (that's the
  target's real grain, and what the continuous-curve feature depends on)
  but adding a **daily-aggregated headline number** (e.g. total predicted
  traffic for the next calendar day, maybe a peak-hour callout) as the
  primary display, since that better matches how often the underlying data
  actually changes — a pure aggregation of predictions already computed,
  no retraining needed. Tradeoff not yet resolved: a daily aggregate is
  more honest about freshness but loses the intraday shape (e.g. "busier
  morning rush than usual") the current curve shows. Not implemented yet —
  pick up here.
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

## What Claude should avoid
- Do not add speculative features beyond what is asked.
- Do not rewrite working code just for style; only touch what is needed.
- Do not commit to an ML library or web framework before it's actually needed
  for the step at hand.

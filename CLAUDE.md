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
  notebook 11) as a first neural-net pass — see "Model selection rationale"
  below for why these and not linear regression/a single tree/SVM. Random
  forest (sklearn-only, no new install) remains an open, untried option;
  a sequence-aware architecture (LSTM/temporal model, or embeddings for
  `station_id`) remains open too if the tree ensembles' ceiling ever needs
  breaking through.
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
- **Random forest**: not ruled out, just not tried yet — a plausible cheap
  sklearn-only alternative (no new dependency) if revisited later.
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
  preprocessing/tuning overhead without a performance gain. A genuinely
  different architecture (sequence model over raw 15-min data, or learned
  `station_id` embeddings) remains untried and is the more plausible
  place a neural net could still add value.

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
- Check whether distance-from-city-center is a useful model feature.
  Notebook 07's geo-map section found a moderate correlation (r ≈ -0.51)
  between a station's distance from the Prinzipalmarkt/Dom center and its
  mean traffic (busier stations cluster on the central Altstadt ring;
  quieter ones sit 2-4km out) — but with notable exceptions (*Wolbecker
  Straße*, *Lütkenbecker Str.*), so it's not a clean cutoff. Not yet tried
  as an actual model feature in 08/09/10; worth testing given it's cheap
  to compute from `station_locations.csv` and may explain some of what
  `station_id` alone currently has to capture as an opaque per-station
  effect. Not yet started.

## What Claude should avoid
- Do not add speculative features beyond what is asked.
- Do not rewrite working code just for style; only touch what is needed.
- Do not commit to an ML library or web framework before it's actually needed
  for the step at hand.

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
- ML library still open (e.g. scikit-learn / LightGBM / Prophet) — decide
  during the modeling notebooks, not upfront
- `matplotlib` / `plotly` for visualization
- Possibly a web dashboard later (framework still open)

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

## What Claude should avoid
- Do not add speculative features beyond what is asked.
- Do not rewrite working code just for style; only touch what is needed.
- Do not commit to an ML library or web framework before it's actually needed
  for the step at hand.

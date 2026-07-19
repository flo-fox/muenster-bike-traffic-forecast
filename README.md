# Münster Bike Traffic Forecast

24-hour-ahead prediction of bike traffic volume per counting station in
Münster, combining historical 15-minute count data with weather data.

## Why

Münster already has good raw-data visualizations (Klimadashboard Münster,
Code-for-Münster's "traffic-dynamics"), but no active forecasting tool — the
only prior attempt is a set of frozen 2018 regression experiments. This
project aims to close that gap.

## Data sources

- **Bike counts**: [`od-ms/radverkehr-zaehlstellen`](https://github.com/od-ms/radverkehr-zaehlstellen)
  — 15-minute counts from 24 stations across the city, data since 2019/2023
  depending on station.
- **Weather data**: [DWD Open Data](https://opendata.dwd.de) (Deutscher
  Wetterdienst).

Both sources are open data — no credentials required.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Layout

- `notebooks/` — numbered analysis/modeling notebooks
- `src/muenster_bike_forecast/` — reusable data loading, feature engineering,
  and modeling code
- `data/raw/` — raw downloaded data (gitignored, regenerate from notebooks)
- `tests/` — unit tests

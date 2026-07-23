# Münster Bike Traffic Forecast

24-hour-ahead prediction of bike traffic volume per counting station in
Münster, combining historical 15-minute count data with weather data.

## Why

Münster already has good raw-data visualizations (Klimadashboard Münster,
Code-for-Münster's "traffic-dynamics"), but no active forecasting tool — the
only prior attempt is a set of frozen 2018 regression experiments. This
project aims to close that gap.

## Data sources & attribution

- **Bike counts**: [`od-ms/radverkehr-zaehlstellen`](https://github.com/od-ms/radverkehr-zaehlstellen)
  — 15-minute counts from 24 stations across the city, data since 2019/2023
  depending on station. Published by the City of Münster under
  [Datenlizenz Deutschland – Namensnennung 2.0](https://www.govdata.de/dl-de/by-2-0)
  (dl-de/by-2-0). *Attribution: Datenquelle Stadt Münster, dl-de/by-2-0.*
- **Weather data**: [DWD Open Data](https://opendata.dwd.de) (Deutscher
  Wetterdienst), hourly station observations. Licensed
  [CC BY 4.0](https://opendata.dwd.de/climate_environment/CDC/Nutzungsbedingungen_German.pdf).
  *Attribution: © Deutscher Wetterdienst (DWD).*
- **NRW school holidays**: fetched from the [OpenHolidays API](https://openholidaysapi.org/),
  licensed [ODbL 1.0](https://github.com/openpotato/openholidaysapi.data).
  *Attribution: contains data from OpenHolidays API, ODbL-1.0.* ODbL's
  share-alike clause applies to any *published derivative database*
  containing this data (not automatically to a dashboard's visual output,
  which ODbL treats separately as a "Produced Work") — revisit before
  publishing a raw combined dataset built on this data. See
  `src/muenster_bike_forecast/data/calendar.py`.
- **NRW university semester/lecture-period dates**: [NRW Ministry of
  Culture and Science (MKW)](https://www.mkw.nrw/service/vorlesungszeiten).
  Page content is under standard copyright; the specific lecture-period
  dates transcribed into this project are used as factual reference data.
  See `src/muenster_bike_forecast/data/semester_dates.py` for full
  provenance (which years are ministry-sourced vs. extrapolated).
- **Public holidays**: computed via the
  [`holidays`](https://pypi.org/project/holidays/) Python library
  ([MIT](https://github.com/vacanza/holidays/blob/dev/LICENSE)), no
  external fetch.

Bike-count and weather data require attribution but are otherwise freely
reusable, including commercially. School-holiday data carries additional
share-alike obligations for published derivatives — see the note above.
All other dependencies (pandas, requests, scikit-learn, matplotlib,
jupyter, pytest, black) use standard permissive OSS licenses (BSD/MIT/
Apache-family).

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

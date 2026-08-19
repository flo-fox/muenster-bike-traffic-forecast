"""Streamlit dashboard entry point: live 24h-ahead bike-traffic forecast.

This file only bootstraps `sys.path` (so `pages/` and `dashboard_common.py`
can import `muenster_bike_forecast`), configures the page, and wires up the
multipage navigation. Each page under `pages/` is its own self-contained
view; shared caching/fetch/chart logic lives in `dashboard_common.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

st.set_page_config(
    page_title="Münster Bike Traffic Forecast", page_icon="🚲", layout="wide"
)

PAGES_DIR = PROJECT_ROOT / "pages"
pages = [
    st.Page(
        str(PAGES_DIR / "station_forecast.py"),
        title="Station forecast",
        icon="📍",
        default=True,
    ),
    st.Page(
        str(PAGES_DIR / "station_comparison.py"), title="Station comparison", icon="📊"
    ),
    st.Page(str(PAGES_DIR / "city_map.py"), title="City map", icon="🗺️"),
]
st.navigation(pages).run()

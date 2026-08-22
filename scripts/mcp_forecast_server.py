"""Local (stdio) MCP server exposing forecast-query tools for this project.

Thin entry point only - the actual logic lives in
`muenster_bike_forecast.mcp_tools` so it's testable without a running
server. Run directly (`python scripts/mcp_forecast_server.py`) or via an
MCP client configured for stdio transport - see README.md for how to
register it with Claude Code.

Local/stdio only, deliberately: reachable only from sessions running on
this machine, no hosting cost, no auth needed. A remote/HTTP-reachable
version (for the Claude mobile app, say) is a separate, larger piece of
work - not attempted here.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mcp.server import MCPServer

from muenster_bike_forecast.mcp_tools import (
    McpToolError,
    get_actual_vs_predicted_tool,
    get_forecast_tool,
    list_stations_tool,
)

mcp = MCPServer(
    name="muenster-bike-forecast",
    description=(
        "24h-ahead bike-traffic forecasts for Münster counting stations, "
        "backed by the project's live LightGBM production model."
    ),
)


@mcp.tool()
def list_stations() -> list[dict[str, object]]:
    """Lists every bike-counting station with live data available.

    Returns one entry per station with its id, human-readable name, and
    the first year the source repo has data for it. Call this first to
    find a station's id before calling get_forecast or
    get_actual_vs_predicted.
    """
    try:
        return list_stations_tool()
    except McpToolError as exc:
        raise RuntimeError(str(exc)) from exc


@mcp.tool()
def get_forecast(station_id: str, as_of: str | None = None) -> dict[str, object]:
    """Gets a live 24h-ahead bike-traffic forecast for one station.

    Fetches fresh bike-count and weather data and runs the project's
    production model. Returns the predicted total traffic over the next
    24 hours, when/how high it's expected to peak, and the actual total
    over the last 24 hours for context.

    Args:
        station_id: The station to forecast - get valid ids from
            list_stations.
        as_of: Optional date (YYYY-MM-DD) to anchor the live-data fetch
            to; defaults to today.
    """
    try:
        return get_forecast_tool(station_id, as_of)
    except McpToolError as exc:
        raise RuntimeError(str(exc)) from exc


@mcp.tool()
def get_actual_vs_predicted(
    station_id: str, as_of: str | None = None
) -> dict[str, object]:
    """Compares yesterday's predicted bike traffic to what actually happened.

    Note: this project keeps no forecast history, so this always compares
    the most recent available ~24h window against what was predicted for
    it - not an arbitrary past date. `as_of` only controls which live-data
    snapshot is fetched (defaults to today), not which day is being
    checked.

    Args:
        station_id: The station to check - get valid ids from
            list_stations.
        as_of: Optional date (YYYY-MM-DD) to anchor the live-data fetch
            to; defaults to today.
    """
    try:
        return get_actual_vs_predicted_tool(station_id, as_of)
    except McpToolError as exc:
        raise RuntimeError(str(exc)) from exc


if __name__ == "__main__":
    mcp.run(transport="stdio")

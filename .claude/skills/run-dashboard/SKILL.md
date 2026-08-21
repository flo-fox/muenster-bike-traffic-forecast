---
name: run-dashboard
description: This skill should be used when the user asks to "show me the app", "run the dashboard", "start the app", "open the Streamlit app", "launch the dashboard", or wants to see the Münster bike traffic forecast dashboard working.
---

# Run the Münster bike traffic forecast dashboard

The dashboard is a Streamlit multipage app: `app.py` (thin
`st.navigation` entry point) plus `pages/station_forecast.py`,
`pages/station_comparison.py`, `pages/city_map.py`, all sharing logic
from `dashboard_common.py`. This skill captures the launch recipe
worked out on 2026-08-17 so it doesn't get rediscovered each session.

## Check whether it's already running

Before launching, check port 8501 first — a prior session may have left
it up:

```
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501
```

`200` means it's already running — skip straight to "Hand off to the
user" below, **unless** `dashboard_common.py` or any `pages/*.py` file
has been edited since that server was launched. Confirmed 2026-08-21: a
running server showed a stale-code `KeyError` after such edits, and the
Streamlit app's own "Clear cache" menu action did *not* fix it — only a
full stop-and-relaunch did. Streamlit's hot-reload/cache-invalidation
doesn't reliably catch every change to a shared module imported by
multiple pages, especially across several edits in one session. When in
doubt after editing dashboard code, restart rather than reuse. Anything
other than `200` (connection refused, `000`) means it needs launching.

## Launch it

Run this as its own backgrounded command — do not wrap it in a
subshell like `(cmd &)` and then `sleep`/`curl` in the same tool call.
That pattern was tried on 2026-08-17 and the backgrounded subshell died
along with the parent command, leaving no server running. Instead pass
`run_in_background: true` directly on the `streamlit run` invocation
itself:

```
streamlit run app.py --server.headless true --server.port 8501
```

Wait a few seconds, then read the background task's output file (or run
a fresh `curl` health check as above) to confirm the `Uvicorn server
started` / `Local URL: http://localhost:8501` lines appear before
telling the user it's up.

## Hand off to the user

No browser-automation tooling (`claude-in-chrome`) has been reliably
available in past sessions on this machine — check whether it's
connected before assuming it isn't (it may have been installed since).
If it's not available, don't guess at what the page looks like: tell
the user the app is running at `http://localhost:8501` and let them
open it themselves, offering to tail the server log if something looks
wrong. If browser tooling *is* available, drive it there and actually
look at the rendered page rather than just confirming the HTTP status
code — a 200 only proves the server started, not that a page renders
correctly.

## Known live-data caveat

The station list and forecasts depend on live fetches from
`raw.githubusercontent.com` (station index) and DWD. Both are cached
via `st.cache_data` in `dashboard_common.py` with TTLs from 15 minutes
to 24 hours, but every *process restart* clears that cache and
re-fetches immediately. Restarting the server repeatedly in a short
window (e.g. while iterating on dashboard code) can trip
`raw.githubusercontent.com`'s anonymous per-IP rate limit and produce a
`429 Too Many Requests` error in the station list — this is a rate
limit, not a GitHub outage (`githubstatus.com` will still read `200`
during it). It clears on its own; no code fix needed.

## Stopping it

Background streamlit processes started this way keep running after the
conversation ends. If asked to stop it, find and end the process rather
than leaving it orphaned:

```
# Windows
tasklist | grep -i streamlit
taskkill //PID <pid> //F
```

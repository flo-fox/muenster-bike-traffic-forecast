---
name: verify-notebook
description: This skill should be used when the user asks to "verify a notebook", "check a notebook still runs", "does Restart and Run All still work", or wants to confirm a notebook is reproducible after editing it or its `src/` dependencies.
---

# Verify a notebook's Restart & Run All

CLAUDE.md's notebook conventions require "a fresh Restart & Run All
must succeed" for every notebook, but nothing currently enforces this
automatically - the `run_tests.py`/`check_conventions.py` PostToolUse
hooks run on every edit and would make editing any notebook slow if
they also executed it in full, so this is deliberately an on-demand
skill instead of an automatic hook.

## When to use it

Run this after editing a notebook directly, or after editing a
`src/muenster_bike_forecast/*` function that one or more notebooks
import - a signature change can silently break a notebook that isn't
open. It is not meant to run on every edit; use it when reproducibility
specifically needs checking.

## Steps

1. Run the verification script from the project root:

   ```
   python .claude/skills/verify-notebook/scripts/verify_notebook.py notebooks/NN_name.ipynb [timeout_seconds]
   ```

   Default timeout is 300s. It executes the notebook into a throwaway
   temp copy via `jupyter nbconvert --execute` - the committed notebook
   and its saved outputs are never touched, so this is safe to run on
   notebooks with meaningful existing outputs.

2. Exit code 0 + `Restart & Run All succeeded.` means it's reproducible.
   Exit code 1 means it failed; the printed message includes the
   failing cell (when the partial output notebook was recoverable) and
   the last ~15 lines of stderr, which normally include the actual
   Python traceback.

3. If it fails, fix the root cause (a stale import path, a renamed
   `src/` function, a hardcoded path assumption) rather than just
   re-running to see if it was transient - unless the notebook fetches
   live data (see caveat below), a failure is almost always real.

## Live-data notebooks: a real caveat, not a flaky test

`01_fetch_bike_counts.ipynb`, `02_fetch_weather.ipynb`, and
`04_fetch_calendar_features.ipynb` hit live external sources
(`od-ms/radverkehr-zaehlstellen` on GitHub, DWD Open Data). Verifying
these:
- Can fail for reasons outside this repo entirely (source downtime,
  rate limiting - `raw.githubusercontent.com` returning `429 Too Many
  Requests` on repeated fetches in a short window has already happened
  in this project's dashboard work, see the `run-dashboard` skill).
  A failure in one of these three notebooks specifically is worth a
  second look before assuming the code is broken.
- Is genuinely slower and consumes the sources' rate limits for real -
  don't verify these three in a tight loop or as a first troubleshooting
  step; verify the notebooks downstream of them (03 onward, which read
  the already-fetched local CSVs) instead when the goal is checking
  modeling-pipeline reproducibility rather than the fetch step itself.

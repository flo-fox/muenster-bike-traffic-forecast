---
name: new-notebook
description: This skill should be used when the user asks to "create a new notebook", "start a new analysis notebook", "scaffold a notebook", or wants to begin a new numbered analysis/modeling stage under `notebooks/`.
---

# Scaffold a new numbered notebook

This project keeps one notebook per analysis/modeling stage, numbered
`01_`, `02_`, ... up through `17_` so far (see CLAUDE.md's "Notebook
conventions"). This skill creates the next one with the right number,
a standard `src/`-import bootstrap cell, and a conventions-checklist
reminder, instead of copy-pasting an old notebook's boilerplate by
hand.

## Steps

1. Ask the user for a short title if they haven't given one (e.g. "18
   holiday feature test" -> title `"Holiday Feature Test"`).
2. Run the scaffold script from the project root:

   ```
   python .claude/skills/new-notebook/scripts/scaffold_notebook.py "Title Of The Notebook"
   ```

   It scans `notebooks/` for the highest existing `NN_` prefix, picks
   the next number, slugifies the title, and writes
   `notebooks/NN_slug.ipynb` with:
   - a markdown title cell (`# NN - Title` + a `TODO: motivation` line
     to replace with the actual reasoning for the notebook, matching
     how e.g. `13_distance_feature_test.ipynb` opens by referencing
     the finding that motivated it)
   - a code cell with the standard path-bootstrap so
     `from muenster_bike_forecast...` imports work whether the
     notebook runs from `notebooks/` or the project root
   - a markdown checklist cell mirroring CLAUDE.md's review checklist
     (type hints/docstrings on `src/` calls, chart data-source
     captions, black formatting, Restart & Run All)
3. Open the new notebook and replace the `TODO: motivation` line with
   the real reasoning before adding any analysis cells - every existing
   notebook opens by stating why it exists, not just what it does.
4. Fill in the actual imports/analysis. The bootstrap cell only
   includes `pandas` - add whichever `muenster_bike_forecast.*` and
   third-party imports the notebook actually needs, following the
   pattern in `13_distance_feature_test.ipynb`'s import cell.

## Notes

- The script refuses to overwrite an existing file at the computed
  path (raises `FileExistsError`) rather than silently clobbering one.
- Re-running the script twice in the same session produces two
  different numbers correctly, since it re-scans `notebooks/` each
  time rather than caching the "next number".

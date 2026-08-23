"""Shared cross-notebook model-metrics registry.

Every modeling notebook (08-18) computes its own overall + per-station
MAE/RMSE via ``model_table.compute_baseline_metrics``. Before this module
existed, notebooks that wanted to compare against an *earlier* notebook's
result hardcoded that result as a Python constant (e.g.
``GBM_REFERENCE_OVERALL_MAE = 14.486977``) - a copy that goes stale the
moment the source notebook is re-run against new data. This caused two
real, documented incidents (see CLAUDE.md's "total_count double-counting
bug" entry): notebook 16 once printed "RF beats GBM on 23 of 23 stations"
with a fabricated ~49% margin because its hardcoded GBM reference was
stale, and notebook 10 made Prophet look competitive with GBM for the
same reason.

This module replaces the copy with a shared, on-disk registry
(``data/model_metrics/registry.csv``, committed to git - small,
derived, and load-bearing in the same way as
``data/processed/station_locations.csv``): a notebook calls
`write_metrics` right after computing its own metrics, and any
downstream notebook calls `read_metrics` instead of hardcoding a copy.
Re-running a notebook overwrites only its own entry; every other
notebook's entry is untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

REGISTRY_COLUMNS: Final[list[str]] = [
    "notebook_id",
    "model_name",
    "group",
    "mae",
    "rmse",
    "n_rows",
]
OVERALL_GROUP: Final[str] = "overall"


class MetricsRegistryError(Exception):
    """Raised when the shared metrics registry cannot be read or written as expected.

    Covers a registry file missing an expected column, a `write_metrics`
    call whose overall/per-station frames don't match
    `model_table.compute_baseline_metrics`'s own contract, and a
    `read_metrics` call for a (notebook_id, model_name) pair that has
    never been written - a clear error here, not a silently wrong or
    missing value, is the entire point of this module.
    """


def load_registry(path: Path) -> pd.DataFrame:
    """Loads the shared metrics registry CSV, or an empty frame if absent.

    Args:
        path: Path to the registry CSV (see `REGISTRY_COLUMNS`).

    Returns:
        DataFrame with `REGISTRY_COLUMNS`. Empty (zero rows, correct
        columns) if `path` does not exist yet - the first notebook to
        ever write is expected to start from nothing, same convention
        as `geocode.load_location_cache`.

    Raises:
        MetricsRegistryError: if the file exists but is missing an
            expected column.
    """
    if not path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    df = pd.read_csv(path)
    missing = set(REGISTRY_COLUMNS) - set(df.columns)
    if missing:
        raise MetricsRegistryError(
            f"{path} is missing expected column(s): {sorted(missing)}."
        )
    return df[REGISTRY_COLUMNS]


def _validate_overall(overall_metrics: pd.DataFrame) -> None:
    missing = {"mae", "rmse", "n_rows"} - set(overall_metrics.columns)
    if missing:
        raise MetricsRegistryError(
            f"overall_metrics is missing expected column(s): {sorted(missing)}."
        )
    if len(overall_metrics) != 1:
        raise MetricsRegistryError(
            f"overall_metrics must have exactly one row, got {len(overall_metrics)}."
        )
    if "group" in overall_metrics.columns:
        (group_value,) = overall_metrics["group"].tolist()
        if group_value != OVERALL_GROUP:
            raise MetricsRegistryError(
                f"overall_metrics's single row must have group == "
                f"{OVERALL_GROUP!r}, got {group_value!r}."
            )


def _validate_per_station(per_station_metrics: pd.DataFrame) -> None:
    missing = {"mae", "rmse", "n_rows"} - set(per_station_metrics.columns)
    if missing:
        raise MetricsRegistryError(
            f"per_station_metrics is missing expected column(s): {sorted(missing)}."
        )
    if per_station_metrics.empty:
        raise MetricsRegistryError("per_station_metrics must not be empty.")
    if "group" in per_station_metrics.columns:
        if (per_station_metrics["group"] == OVERALL_GROUP).any():
            raise MetricsRegistryError(
                f"per_station_metrics must not contain a {OVERALL_GROUP!r} "
                "row - pass that in overall_metrics instead."
            )


def write_metrics(
    path: Path,
    notebook_id: str,
    model_name: str,
    overall_metrics: pd.DataFrame,
    per_station_metrics: pd.DataFrame,
) -> Path:
    """Writes one notebook+model's metrics into the shared registry.

    Merges into the existing registry keyed by (notebook_id, model_name,
    group): re-running this notebook overwrites only its own
    (notebook_id, model_name) rows, never another notebook's or another
    model_name's existing entries.

    Args:
        path: Registry CSV path; created (with parent dirs) if missing.
        notebook_id: Writing notebook's filename stem, e.g.
            ``"08_gradient_boosting_model"``.
        model_name: Short label for this specific model/variant, e.g.
            ``"gradient_boosting"``, ``"random_forest_combined"``. Must
            be used consistently between the writer and every reader.
        overall_metrics: As returned by
            ``model_table.compute_baseline_metrics(..., group_col=None)``
            - exactly one row, ``group == "overall"``.
        per_station_metrics: As returned by
            ``model_table.compute_baseline_metrics(..., group_col="station_id")``.

    Returns:
        `path`.

    Raises:
        MetricsRegistryError: if `overall_metrics` does not have exactly
            one row with ``group == "overall"``, if `per_station_metrics`
            is empty or contains a ``group == "overall"`` row, or if
            either frame is missing a `mae`/`rmse`/`n_rows` column.
    """
    _validate_overall(overall_metrics)
    _validate_per_station(per_station_metrics)

    overall_row = overall_metrics.copy()
    overall_row["group"] = OVERALL_GROUP
    per_station_rows = per_station_metrics.copy()
    per_station_rows["group"] = per_station_rows["group"].astype(str)

    new_rows = pd.concat([overall_row, per_station_rows], ignore_index=True)
    new_rows["notebook_id"] = notebook_id
    new_rows["model_name"] = model_name
    new_rows = new_rows[REGISTRY_COLUMNS]

    existing = load_registry(path)
    is_this_entry = (existing["notebook_id"] == notebook_id) & (
        existing["model_name"] == model_name
    )
    combined = pd.concat([existing[~is_this_entry], new_rows], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    combined[REGISTRY_COLUMNS].to_csv(path, index=False)
    return path


def read_metrics(
    path: Path, notebook_id: str, model_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reads one notebook+model's metrics back out of the shared registry.

    Args:
        path: Registry CSV path.
        notebook_id: As passed to the original `write_metrics` call.
        model_name: As passed to the original `write_metrics` call.

    Returns:
        `(overall_metrics, per_station_metrics)`, each shaped like
        `model_table.compute_baseline_metrics`'s own output
        (``group``/``mae``/``rmse``/``n_rows``), so callers use them as a
        drop-in for what they used to get from a hardcoded dict, e.g.
        ``read_metrics(...)[1].set_index("group")["mae"]`` replaces
        ``pd.Series(GBM_REFERENCE_PER_STATION_MAE)``.
        `per_station_metrics["group"]` is cast back to `int64` (the
        registry stores it as `str` on disk, since a CSV column can't
        hold both the string ``"overall"`` and integer station ids), so
        it aligns directly with a locally-computed
        ``compute_baseline_metrics(..., group_col="station_id")`` result
        without the caller needing its own cast - getting this wrong
        would silently misalign (all-NaN) when combined via
        ``pd.DataFrame({...})``, exactly the class of bug this module
        exists to prevent.

    Raises:
        MetricsRegistryError: if `path` does not exist, if no rows match
            (notebook_id, model_name) (message includes what was
            requested and which (notebook_id, model_name) pairs *are*
            available, to make a typo'd model_name obvious rather than a
            silent empty result), or if the matching rows have no
            "overall" row.
    """
    registry = load_registry(path)
    if registry.empty:
        raise MetricsRegistryError(
            f"{path} does not exist or is empty - nothing has been written yet."
        )

    matches = registry[
        (registry["notebook_id"] == notebook_id)
        & (registry["model_name"] == model_name)
    ]
    if matches.empty:
        available = sorted(set(zip(registry["notebook_id"], registry["model_name"])))
        raise MetricsRegistryError(
            f"No entry for (notebook_id={notebook_id!r}, model_name={model_name!r}). "
            f"Available (notebook_id, model_name) pairs: {available}."
        )

    overall = matches[matches["group"] == OVERALL_GROUP].reset_index(drop=True)
    if overall.empty:
        raise MetricsRegistryError(
            f"(notebook_id={notebook_id!r}, model_name={model_name!r}) has no "
            f"{OVERALL_GROUP!r} row."
        )

    per_station = matches[matches["group"] != OVERALL_GROUP].reset_index(drop=True)
    per_station = per_station.copy()
    per_station["group"] = per_station["group"].astype("int64")

    return overall[REGISTRY_COLUMNS], per_station[REGISTRY_COLUMNS]

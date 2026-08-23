"""Tests for `muenster_bike_forecast.modeling.metrics_registry`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from muenster_bike_forecast.modeling.metrics_registry import (
    MetricsRegistryError,
    REGISTRY_COLUMNS,
    load_registry,
    read_metrics,
    write_metrics,
)
from muenster_bike_forecast.modeling.model_table import compute_baseline_metrics


def _synthetic_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_id": [1, 1, 2, 2, 3, 3],
            "prediction": [10.0, 12.0, 20.0, 22.0, 5.0, 7.0],
            "target": [11.0, 11.0, 21.0, 21.0, 6.0, 6.0],
        }
    )


def _synthetic_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _synthetic_df()
    overall = compute_baseline_metrics(
        df, prediction_col="prediction", target_col="target"
    )
    per_station = compute_baseline_metrics(
        df, prediction_col="prediction", target_col="target", group_col="station_id"
    )
    return overall, per_station


# ---------------------------------------------------------------------------
# load_registry
# ---------------------------------------------------------------------------


def test_load_registry_returns_empty_frame_if_file_missing(tmp_path: Path) -> None:
    registry = load_registry(tmp_path / "does_not_exist.csv")
    assert registry.empty
    assert list(registry.columns) == REGISTRY_COLUMNS


def test_load_registry_raises_on_missing_column(tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    pd.DataFrame({"notebook_id": ["08"], "model_name": ["gbm"]}).to_csv(
        path, index=False
    )
    with pytest.raises(MetricsRegistryError, match="missing expected column"):
        load_registry(path)


# ---------------------------------------------------------------------------
# write_metrics / read_metrics round trip
# ---------------------------------------------------------------------------


def test_write_then_read_metrics_round_trips_overall_and_per_station(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.csv"
    overall, per_station = _synthetic_metrics()

    write_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting", overall, per_station
    )
    read_overall, read_per_station = read_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting"
    )

    assert read_overall["mae"].iloc[0] == pytest.approx(overall["mae"].iloc[0])
    assert read_overall["rmse"].iloc[0] == pytest.approx(overall["rmse"].iloc[0])
    assert read_overall["group"].iloc[0] == "overall"

    expected_per_station = per_station.set_index("group")["mae"].sort_index()
    actual_per_station = read_per_station.set_index("group")["mae"].sort_index()
    pd.testing.assert_series_equal(
        actual_per_station, expected_per_station, check_dtype=False
    )


def test_write_metrics_does_not_clobber_other_notebooks_or_models(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.csv"
    overall, per_station = _synthetic_metrics()

    write_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting", overall, per_station
    )
    write_metrics(path, "09_lightgbm_model", "lightgbm", overall, per_station)

    read_overall, _ = read_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting"
    )
    assert read_overall["mae"].iloc[0] == pytest.approx(overall["mae"].iloc[0])

    registry = load_registry(path)
    pairs = set(zip(registry["notebook_id"], registry["model_name"]))
    assert ("08_gradient_boosting_model", "gradient_boosting") in pairs
    assert ("09_lightgbm_model", "lightgbm") in pairs


def test_write_metrics_overwrites_its_own_entry_on_rerun(tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    overall, per_station = _synthetic_metrics()

    write_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting", overall, per_station
    )

    new_overall = overall.copy()
    new_overall["mae"] = 999.0
    new_per_station = per_station.copy()
    new_per_station["mae"] = 999.0
    write_metrics(
        path,
        "08_gradient_boosting_model",
        "gradient_boosting",
        new_overall,
        new_per_station,
    )

    registry = load_registry(path)
    this_entry = registry[
        (registry["notebook_id"] == "08_gradient_boosting_model")
        & (registry["model_name"] == "gradient_boosting")
    ]
    # Exactly one overall row + one row per station, not duplicated.
    assert len(this_entry) == 1 + len(per_station)
    read_overall, _ = read_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting"
    )
    assert read_overall["mae"].iloc[0] == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# write_metrics validation
# ---------------------------------------------------------------------------


def test_write_metrics_raises_on_overall_metrics_wrong_row_count(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.csv"
    _, per_station = _synthetic_metrics()
    two_rows = pd.DataFrame(
        {
            "group": ["overall", "overall"],
            "mae": [1.0, 2.0],
            "rmse": [1.0, 2.0],
            "n_rows": [1, 1],
        }
    )
    with pytest.raises(MetricsRegistryError, match="exactly one row"):
        write_metrics(path, "08", "gbm", two_rows, per_station)


def test_write_metrics_raises_on_per_station_metrics_containing_overall_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.csv"
    overall, per_station = _synthetic_metrics()
    bad_per_station = pd.concat(
        [per_station, overall.assign(group="overall")], ignore_index=True
    )
    with pytest.raises(MetricsRegistryError, match="must not contain"):
        write_metrics(path, "08", "gbm", overall, bad_per_station)


def test_write_metrics_raises_on_per_station_metrics_empty(tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    overall, _ = _synthetic_metrics()
    empty = pd.DataFrame(columns=["group", "mae", "rmse", "n_rows"])
    with pytest.raises(MetricsRegistryError, match="must not be empty"):
        write_metrics(path, "08", "gbm", overall, empty)


def test_write_metrics_raises_on_missing_metric_columns(tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    overall, per_station = _synthetic_metrics()
    incomplete_overall = overall.drop(columns=["rmse"])
    with pytest.raises(MetricsRegistryError, match="missing expected column"):
        write_metrics(path, "08", "gbm", incomplete_overall, per_station)


# ---------------------------------------------------------------------------
# read_metrics errors
# ---------------------------------------------------------------------------


def test_read_metrics_raises_on_missing_registry_file(tmp_path: Path) -> None:
    with pytest.raises(MetricsRegistryError, match="does not exist"):
        read_metrics(tmp_path / "does_not_exist.csv", "08", "gbm")


def test_read_metrics_raises_on_unknown_notebook_or_model(tmp_path: Path) -> None:
    path = tmp_path / "registry.csv"
    overall, per_station = _synthetic_metrics()
    write_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting", overall, per_station
    )

    with pytest.raises(MetricsRegistryError, match="lightgbm") as exc_info:
        read_metrics(path, "08_gradient_boosting_model", "lightgbm")
    # Error names both what was requested and what's available.
    assert "gradient_boosting" in str(exc_info.value)


# ---------------------------------------------------------------------------
# the specific correctness bug this module exists to prevent
# ---------------------------------------------------------------------------


def test_read_metrics_per_station_group_is_int64_after_round_trip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.csv"
    overall, per_station = _synthetic_metrics()
    write_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting", overall, per_station
    )

    _, read_per_station = read_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting"
    )
    assert read_per_station["group"].dtype == "int64"
    assert sorted(read_per_station["group"].tolist()) == sorted(
        per_station["group"].tolist()
    )


def test_read_metrics_result_aligns_with_a_freshly_computed_compute_baseline_metrics_call(
    tmp_path: Path,
) -> None:
    path = tmp_path / "registry.csv"
    overall, per_station = _synthetic_metrics()
    write_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting", overall, per_station
    )
    _, read_per_station = read_metrics(
        path, "08_gradient_boosting_model", "gradient_boosting"
    )

    # A second, independently-computed per-station frame over the same
    # station ids (as a downstream notebook would have from its own model).
    df = _synthetic_df()
    df["prediction"] = df["prediction"] + 1.0
    fresh_per_station = compute_baseline_metrics(
        df, prediction_col="prediction", target_col="target", group_col="station_id"
    )

    combined = pd.DataFrame(
        {
            "reference_mae": read_per_station.set_index("group")["mae"],
            "fresh_mae": fresh_per_station.set_index("group")["mae"],
        }
    )
    # If the dtype mismatch bug were present, this union-of-index combine
    # would silently produce NaN rows instead of 3 fully-aligned rows.
    assert len(combined) == 3
    assert not combined.isna().any().any()

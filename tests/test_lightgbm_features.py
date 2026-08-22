"""Tests for `muenster_bike_forecast.modeling.lightgbm_features`."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline

from muenster_bike_forecast.modeling.lightgbm_features import FixedCategoryCaster


def test_fit_transform_casts_to_category_dtype_with_fixed_categories() -> None:
    df = pd.DataFrame({"station": ["a", "b", "a", "c"], "value": [1, 2, 3, 4]})
    caster = FixedCategoryCaster(["station"])

    out = caster.fit(df).transform(df)

    assert isinstance(out["station"].dtype, pd.CategoricalDtype)
    assert list(out["station"].cat.categories) == ["a", "b", "c"]
    assert out["value"].dtype == df["value"].dtype  # untouched column


def test_transform_on_a_later_disjoint_frame_reuses_fit_time_categories() -> None:
    train = pd.DataFrame({"station": ["a", "b", "c"]})
    caster = FixedCategoryCaster(["station"]).fit(train)

    later = pd.DataFrame({"station": ["b"]})  # single value, disjoint from train order
    out = caster.transform(later)

    assert list(out["station"].cat.categories) == ["a", "b", "c"]
    assert out["station"].iloc[0] == "b"


def test_transform_maps_unseen_category_to_nan_not_an_error() -> None:
    train = pd.DataFrame({"station": ["a", "b"]})
    caster = FixedCategoryCaster(["station"]).fit(train)

    out = caster.transform(pd.DataFrame({"station": ["z"]}))

    assert pd.isna(out["station"].iloc[0])


def test_single_row_prediction_matches_batch_prediction() -> None:
    """The correctness property this transformer exists for.

    `inference.predict_24h_ahead` constructs its input as
    `pd.DataFrame([row])` - a single-row frame. If categorical encoding
    were derived per-batch (e.g. a naive `.astype("category")`) rather
    than from a fixed, fit-time category set, that single-row prediction
    could silently diverge from what the same row gets predicted as when
    it's part of a larger batch.
    """
    rng = np.random.default_rng(0)
    n = 200
    train_df = pd.DataFrame(
        {
            "station": rng.choice(["a", "b", "c", "d"], size=n),
            "hour": rng.integers(0, 24, size=n),
            "x": rng.normal(size=n),
        }
    )
    y_train = train_df["x"] + train_df["hour"] * 0.1

    model = Pipeline(
        [
            ("cast_categorical", FixedCategoryCaster(["station", "hour"])),
            ("lgbm", LGBMRegressor(random_state=0, verbosity=-1, n_estimators=20)),
        ]
    )
    model.fit(train_df, y_train, lgbm__categorical_feature=["station", "hour"])

    batch_df = pd.DataFrame(
        {"station": ["a", "b", "c"], "hour": [8, 12, 20], "x": [0.5, -0.3, 1.2]}
    )
    batch_predictions = model.predict(batch_df)

    for i in range(len(batch_df)):
        single_row_df = batch_df.iloc[[i]]
        single_row_prediction = model.predict(single_row_df)[0]
        assert single_row_prediction == batch_predictions[i], (
            f"Row {i}: single-row prediction ({single_row_prediction}) "
            f"does not match its batch prediction ({batch_predictions[i]})."
        )

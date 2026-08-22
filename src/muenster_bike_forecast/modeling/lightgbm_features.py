"""Fixed-categories casting for LightGBM's production `Pipeline`.

`LGBMRegressor` needs pandas `category`-dtype columns with a *consistent*
set of categories at both `.fit()` and `.predict()` time. Naively casting
via `.astype("category")` separately at fit and predict time is risky for
this project's live-inference path: `inference.predict_24h_ahead` scores
one row at a time, and casting a freshly-built single-row DataFrame to
`category` dtype would derive its categories from whatever's in that one
row, not from the full training set. This transformer avoids that by
storing each categorical column's fixed category set once, at `.fit()`
time (on the full training set), and reapplying that exact fixed set at
`.transform()` - so a single row and a full batch of the same underlying
values are always encoded identically, regardless of what else is (or
isn't) present in the batch being transformed.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FixedCategoryCaster(BaseEstimator, TransformerMixin):
    """Casts named columns to `category` dtype with a fit-time-fixed category list.

    Args:
        categorical_columns: Columns to cast; every other column in `X`
            passes through untouched.
    """

    def __init__(self, categorical_columns: Sequence[str]):
        """Stores which columns to cast; no fitting happens here.

        Args:
            categorical_columns: Columns to cast to `category` dtype.
        """
        self.categorical_columns = list(categorical_columns)

    def fit(self, X: pd.DataFrame, y: object = None) -> "FixedCategoryCaster":
        """Records each categorical column's fixed category set from `X`.

        Args:
            X: Training features; must contain every column in
                `categorical_columns`.
            y: Ignored (present for scikit-learn's `Pipeline` API).

        Returns:
            self.
        """
        self.categories_ = {
            column: pd.Index(pd.unique(X[column].dropna())).sort_values()
            for column in self.categorical_columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Casts `categorical_columns` to `category` dtype using the fixed set.

        A value present in `X` but absent from the fit-time category set
        (e.g. a station id that didn't exist in training) becomes `NaN`,
        not an error - the same missing-value handling LightGBM already
        applies elsewhere in this feature set (lag/rolling nulls near a
        station's start of coverage).

        Args:
            X: Features to transform; must contain every column in
                `categorical_columns`.

        Returns:
            Copy of `X` with `categorical_columns` cast to `category`
            dtype.
        """
        out = X.copy()
        for column in self.categorical_columns:
            categories = self.categories_[column]
            # Explicitly null out any value outside the fixed category set
            # before casting - assigning it directly during the cast is
            # deprecated (pandas warns it will raise in a future version).
            masked = out[column].where(out[column].isin(categories))
            out[column] = masked.astype(pd.CategoricalDtype(categories=categories))
        return out

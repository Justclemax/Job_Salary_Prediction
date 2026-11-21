"""Model zoo. Every model is a full pipeline predicting salary in dollars.

All regressors are trained on log(salary) through TransformedTargetRegressor:
the data is multiplicative (spread grows with the mean), so errors are
closer to homoscedastic in log space. Predictions are returned in dollars.
"""

from typing import Any

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from salary import config as C
from salary.features import make_preprocessor

MODEL_NAMES = ["dummy", "ridge", "random_forest", "xgboost", "lightgbm", "catboost"]
BOOSTERS = ["xgboost", "lightgbm", "catboost"]
DISPLAY_NAMES = {
    "dummy": "Baseline (médiane)",
    "ridge": "Ridge (log-linéaire)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "ensemble": "Ensemble (moyenne des boosters)",
}

DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "dummy": {"strategy": "median"},
    "ridge": {"alpha": 1.0},
    "random_forest": {"n_estimators": 150, "min_samples_leaf": 20, "max_features": 1.0},
    "xgboost": {"n_estimators": 800, "learning_rate": 0.05, "max_depth": 6,
                "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 10},
    "lightgbm": {"n_estimators": 800, "learning_rate": 0.05, "num_leaves": 63,
                 "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
                 "min_child_samples": 50},
    "catboost": {"iterations": 1000, "learning_rate": 0.08, "depth": 6},
}


def _estimator(name: str, params: dict[str, Any]):
    p = {**DEFAULT_PARAMS[name], **params}
    if name == "dummy":
        return DummyRegressor(**p), "tree"
    if name == "ridge":
        return Ridge(**p), "linear"
    if name == "random_forest":
        return RandomForestRegressor(n_jobs=-1, random_state=C.SEED, **p), "tree"
    if name == "xgboost":
        return XGBRegressor(tree_method="hist", n_jobs=-1, random_state=C.SEED, **p), "tree"
    if name == "lightgbm":
        return LGBMRegressor(n_jobs=-1, random_state=C.SEED, verbose=-1, **p), "tree"
    if name == "catboost":
        return (
            CatBoostRegressor(cat_features=tuple(C.CATEGORICAL), random_seed=C.SEED,
                              verbose=False, thread_count=-1, **p),
            "native",
        )
    raise ValueError(f"Unknown model: {name}")


def build_model(name: str, params: dict[str, Any] | None = None) -> TransformedTargetRegressor:
    est, kind = _estimator(name, params or {})
    pipe = Pipeline([("prep", make_preprocessor(kind)), ("model", est)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp,
                                      check_inverse=False)


def build_quantile_model(alpha: float) -> Pipeline:
    """LightGBM quantile regressor for prediction intervals (trained on raw salary)."""
    est = LGBMRegressor(objective="quantile", alpha=alpha, n_estimators=500,
                        learning_rate=0.05, num_leaves=31, min_child_samples=100,
                        n_jobs=-1, random_state=C.SEED, verbose=-1)
    return Pipeline([("prep", make_preprocessor("tree")), ("model", est)])


def inner_model(fitted: TransformedTargetRegressor):
    """Return (preprocessor, estimator) of a fitted model."""
    pipe = fitted.regressor_
    return pipe.named_steps["prep"], pipe.named_steps["model"]


class AveragingEnsemble(BaseEstimator, RegressorMixin):
    """Geometric mean of already-fitted models (averaging in log space)."""

    def __init__(self, models: dict[str, Any] | None = None):
        self.models = models

    def fit(self, X, y=None):  # members are fitted beforehand
        return self

    def predict(self, X):
        logs = np.column_stack([np.log(m.predict(X)) for m in self.models.values()])
        return np.exp(logs.mean(axis=1))

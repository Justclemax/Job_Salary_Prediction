"""Optuna hyper-parameter search for the gradient-boosting models.

Search runs on a subsample of the TRAIN set with K-fold CV; the test set is never used.
"""

import optuna
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score

from salary import config as C
from salary.models import build_model


def _space(name: str, trial: optuna.Trial) -> dict:
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 60),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 300, 1500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
        }
    raise ValueError(f"No search space for {name}")


def tune(name: str, train: pd.DataFrame, n_trials: int = 20, sample: int = 50_000,
         cv: int = 3, timeout: int | None = 600) -> tuple[dict, pd.DataFrame]:
    sub = train.sample(n=min(sample, len(train)), random_state=C.SEED)
    X, y = sub[C.FEATURES], sub[C.TARGET]
    folds = KFold(cv, shuffle=True, random_state=C.SEED)

    def objective(trial: optuna.Trial) -> float:
        model = build_model(name, _space(name, trial))
        scores = cross_val_score(model, X, y, cv=folds,
                                 scoring="neg_root_mean_squared_error")
        return -scores.mean()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "duration"))
    return study.best_params, trials

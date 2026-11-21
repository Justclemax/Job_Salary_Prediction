"""Train, tune, cross-validate and evaluate every model; write artifacts.

    uv run python -m salary.train                 # full run
    uv run python -m salary.train --fast          # quick smoke run (subsample, few trials)
    uv run python -m salary.train --models xgboost lightgbm --trials 30

Outputs
    models/<name>.joblib, models/ensemble.joblib, models/quantile_{low,high}.joblib
    models/metrics.json                     metrics + params + metadata
    reports/models/predictions_test.parquet test-set predictions of every model
    reports/models/segment_errors.csv       per-segment errors of the selected model
    reports/models/cv_folds.csv             per-fold CV metrics
    reports/models/optuna_<name>.csv        tuning trials
    reports/models/shap_sample.parquet, shap_importance.csv
"""

import argparse
import json
import platform
import subprocess
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import KFold

from salary import config as C
from salary.data import file_md5, load_raw, split
from salary.evaluate import interval_metrics, regression_metrics, segment_errors, summarize_folds
from salary.explain import importance, shap_values
from salary.models import (
    BOOSTERS,
    DEFAULT_PARAMS,
    DISPLAY_NAMES,
    MODEL_NAMES,
    AveragingEnsemble,
    build_model,
    build_quantile_model,
)
from salary.tuning import tune

REPORT_DIR = C.REPORTS_DIR / "models"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=C.ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _versions() -> dict[str, str]:
    import catboost  # noqa: E401
    import lightgbm
    import sklearn
    import xgboost

    return {"python": platform.python_version(), "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__}


def split_calibration(train: pd.DataFrame, frac: float = 0.15):
    cal = train.sample(frac=frac, random_state=C.SEED)
    return train.drop(cal.index), cal


def cross_validate_oof(model, X: pd.DataFrame, y: pd.Series, folds: KFold):
    """Manual K-fold returning per-fold metrics and out-of-fold predictions."""
    oof = np.zeros(len(y))
    fold_metrics = []
    for tr, va in folds.split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
        fold_metrics.append(regression_metrics(y.iloc[va], oof[va]))
    return fold_metrics, oof


def run(models: list[str], n_trials: int, cv: int, fast: bool, tune_models: bool) -> dict:
    t0 = time.time()
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    train, test = split(df)
    if not fast:
        train.to_parquet(C.DATA_PROCESSED / "train.parquet")
        test.to_parquet(C.DATA_PROCESSED / "test.parquet")
    if fast:  # smoke run: small subsample, nothing below is representative
        train = train.sample(20_000, random_state=C.SEED)
        test = test.sample(5_000, random_state=C.SEED)
    X_tr, y_tr = train[C.FEATURES], train[C.TARGET]
    X_te, y_te = test[C.FEATURES], test[C.TARGET]
    folds = KFold(cv, shuffle=True, random_state=C.SEED)

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    oofs: dict[str, np.ndarray] = {}
    fold_rows = []
    preds = test[C.FEATURES + [C.TARGET]].copy()

    for name in models:
        print(f"\n=== {DISPLAY_NAMES[name]} ===", flush=True)
        params: dict = {}
        if tune_models and name in BOOSTERS:
            ts = time.time()
            params, trials = tune(name, train, n_trials=n_trials,
                                  sample=10_000 if fast else 50_000)
            trials.to_csv(REPORT_DIR / f"optuna_{name}.csv", index=False)
            print(f"  tuning {time.time() - ts:.0f}s  best={params}", flush=True)
        model = build_model(name, params)

        fold_metrics, oofs[name] = cross_validate_oof(model, X_tr, y_tr, folds)
        for i, fm in enumerate(fold_metrics):
            fold_rows.append({"model": name, "fold": i, **fm})

        ts = time.time()
        model.fit(X_tr, y_tr)
        fit_time = time.time() - ts
        ts = time.time()
        p_te = model.predict(X_te)
        predict_ms_per_1k = (time.time() - ts) / len(X_te) * 1000 * 1000
        p_tr = model.predict(X_tr)

        results[name] = {
            "display_name": DISPLAY_NAMES[name],
            "params": {**DEFAULT_PARAMS[name], **params},
            "tuned": bool(params),
            "cv": summarize_folds(fold_metrics),
            "train": regression_metrics(y_tr, p_tr),
            "test": regression_metrics(y_te, p_te),
            "fit_time_s": fit_time,
            "predict_ms_per_1k": predict_ms_per_1k,
        }
        results[name]["overfit_gap_rmse"] = results[name]["test"]["rmse"] - results[name]["train"]["rmse"]
        preds[f"pred_{name}"] = p_te
        fitted[name] = model
        joblib.dump(model, C.MODELS_DIR / f"{name}.joblib", compress=3)
        print(f"  CV RMSE {results[name]['cv']['rmse_mean']:,.0f} ± {results[name]['cv']['rmse_std']:,.0f}"
              f" | test RMSE {results[name]['test']['rmse']:,.0f}  R² {results[name]['test']['r2']:.4f}"
              f" | fit {fit_time:.1f}s", flush=True)

    boosters = [b for b in BOOSTERS if b in fitted]
    if len(boosters) >= 2:
        ens = AveragingEnsemble({b: fitted[b] for b in boosters})
        oof_ens = np.exp(np.mean([np.log(oofs[b]) for b in boosters], axis=0))
        ens_folds = [regression_metrics(y_tr.iloc[va], oof_ens[va]) for _, va in folds.split(X_tr)]
        for i, fm in enumerate(ens_folds):
            fold_rows.append({"model": "ensemble", "fold": i, **fm})
        ts = time.time()
        p_te = ens.predict(X_te)
        predict_ms_per_1k = (time.time() - ts) / len(X_te) * 1000 * 1000
        results["ensemble"] = {
            "display_name": DISPLAY_NAMES["ensemble"],
            "params": {"members": boosters},
            "cv": summarize_folds(ens_folds),
            "train": regression_metrics(y_tr, ens.predict(X_tr)),
            "test": regression_metrics(y_te, p_te),
            "fit_time_s": sum(results[b]["fit_time_s"] for b in boosters),
            "predict_ms_per_1k": predict_ms_per_1k,
        }
        results["ensemble"]["overfit_gap_rmse"] = results["ensemble"]["test"]["rmse"] - results["ensemble"]["train"]["rmse"]
        preds["pred_ensemble"] = p_te
        fitted["ensemble"] = ens
        joblib.dump(ens, C.MODELS_DIR / "ensemble.joblib", compress=3)

    # Model selection on CV RMSE only (never on the test set).
    best = min(results, key=lambda n: results[n]["cv"]["rmse_mean"])
    print(f"\nSelected model (lowest CV RMSE): {best}")

    # 80 % prediction interval: conformalized quantile regression (Romano et al., 2019).
    # Quantile models are fit on 85 % of train; the other 15 % calibrates the width so
    # that coverage on unseen data is guaranteed to be >= 80 % (exchangeability).
    print("\n=== Quantile models (10 % / 90 %) + conformal calibration ===", flush=True)
    target_cov = 0.80
    q_fit, q_cal = split_calibration(train)
    q_low = build_quantile_model(0.10).fit(q_fit[C.FEATURES], q_fit[C.TARGET])
    q_high = build_quantile_model(0.90).fit(q_fit[C.FEATURES], q_fit[C.TARGET])
    y_cal = q_cal[C.TARGET].to_numpy()
    scores = np.maximum(q_low.predict(q_cal[C.FEATURES]) - y_cal,
                        y_cal - q_high.predict(q_cal[C.FEATURES]))
    n_cal = len(scores)
    offset = float(np.quantile(scores, min(1.0, np.ceil((n_cal + 1) * target_cov) / n_cal)))
    joblib.dump(q_low, C.MODELS_DIR / "quantile_low.joblib", compress=3)
    joblib.dump(q_high, C.MODELS_DIR / "quantile_high.joblib", compress=3)
    raw_low, raw_high = q_low.predict(X_te), q_high.predict(X_te)
    preds["q_low"], preds["q_high"] = raw_low - offset, raw_high + offset
    interval = {
        "conformal_offset": offset,
        "raw": interval_metrics(y_te, raw_low, raw_high),
        **interval_metrics(y_te, preds["q_low"], preds["q_high"]),
    }
    print(f"  raw coverage {interval['raw']['coverage']:.1f}% -> conformal "
          f"{interval['coverage']:.1f}% (target 80%), mean width {interval['mean_width']:,.0f}$")

    preds.to_parquet(REPORT_DIR / "predictions_test.parquet")
    pd.DataFrame(fold_rows).to_csv(REPORT_DIR / "cv_folds.csv", index=False)
    segment_errors(test, y_te, preds[f"pred_{best}"],
                   by=["job_title", "location", "education_level", "company_size",
                       "remote_work", "industry"]).to_csv(REPORT_DIR / "segment_errors.csv", index=False)

    # SHAP on the best single tree model (the ensemble is explained through its best member).
    shap_model = best if best in BOOSTERS + ["random_forest"] else min(
        boosters or ["random_forest"], key=lambda n: results[n]["cv"]["rmse_mean"])
    sample = X_te.sample(min(3000, len(X_te)), random_state=C.SEED)
    sv, base = shap_values(fitted[shap_model], sample)
    pd.concat([sample.reset_index(drop=True),
               sv.add_prefix("shap_").reset_index(drop=True)], axis=1).to_parquet(
        REPORT_DIR / "shap_sample.parquet")
    importance(sv).to_csv(REPORT_DIR / "shap_importance.csv", index=False)

    out = {
        "best_model": best,
        "shap_model": shap_model,
        "shap_base_value_log": base,
        "interval": {"alpha_low": 0.10, "alpha_high": 0.90, "target_coverage": 80.0, **interval},
        "models": results,
        "metadata": {
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "data_md5": file_md5(),
            "n_train": len(train), "n_test": len(test),
            "cv_folds": cv, "optuna_trials": n_trials if tune_models else 0,
            "seed": C.SEED, "fast": fast, "target_transform": "log",
            "duration_s": round(time.time() - t0, 1),
            "versions": _versions(),
        },
    }
    (C.MODELS_DIR / "metrics.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nDone in {out['metadata']['duration_s']:.0f}s -> {C.MODELS_DIR / 'metrics.json'}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODEL_NAMES, choices=MODEL_NAMES)
    ap.add_argument("--trials", type=int, default=20, help="Optuna trials per booster")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--fast", action="store_true", help="smoke run on a subsample")
    a = ap.parse_args()
    run(a.models, n_trials=3 if a.fast else a.trials, cv=3 if a.fast else a.cv,
        fast=a.fast, tune_models=not a.no_tune)


if __name__ == "__main__":
    main()

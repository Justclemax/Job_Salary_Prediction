"""Smoke tests on trained artifacts (skipped until `python -m salary.train` has run)."""

import json

import joblib
import numpy as np
import pytest

from salary import config as C
from salary.explain import explain_one

METRICS = C.MODELS_DIR / "metrics.json"
pytestmark = pytest.mark.skipif(not METRICS.exists(), reason="models not trained")


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text())


def test_best_model_predicts_profile(metrics, profile):
    model = joblib.load(C.MODELS_DIR / f"{metrics['best_model']}.joblib")
    pred = model.predict(profile)[0]
    assert 30_000 < pred < 400_000


def test_shap_reconstructs_prediction(metrics, profile):
    model = joblib.load(C.MODELS_DIR / f"{metrics['shap_model']}.joblib")
    ex = explain_one(model, profile)
    recon = ex.attrs["base_salary"] * np.exp(ex["shap_log"].sum())
    assert recon == pytest.approx(model.predict(profile)[0], rel=1e-3)


def test_interval_contains_prediction(metrics, profile):
    lo = joblib.load(C.MODELS_DIR / "quantile_low.joblib").predict(profile)[0]
    hi = joblib.load(C.MODELS_DIR / "quantile_high.joblib").predict(profile)[0]
    off = metrics["interval"]["conformal_offset"]
    assert lo - off < hi + off
    assert metrics["interval"]["coverage"] >= 78


WORLD = C.REPORTS_DIR / "world" / "summary.json"


@pytest.mark.skipif(not WORLD.exists(), reason="world analysis not run")
def test_world_reports_are_consistent():
    import pandas as pd

    s = json.loads(WORLD.read_text())
    assert s["n"] > 10_000 and s["n_countries"] >= 20
    assert 0.3 < s["best_cv"]["r2_mean"] < 0.9
    grid = pd.read_parquet(C.REPORTS_DIR / "world" / "prediction_grid.parquet")
    assert grid[["pred_usd", "pred_ppp"]].gt(0).all().all()
    ate = pd.read_csv(C.REPORTS_DIR / "world" / "ate.csv")
    assert ate["dml_pct_ci_low"].le(ate["dml_pct"]).all() and ate["dml_pct"].le(ate["dml_pct_ci_high"]).all()

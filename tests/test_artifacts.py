"""Smoke tests on trained artifacts (skipped until `python -m salary.train` has run).

`models/metrics.json` is committed but the `.joblib` files are not, so the model tests
check for the files they load, not only for the metrics.
"""

import json

import joblib
import numpy as np
import pytest

from salary import config as C
from salary.explain import explain_one

METRICS = C.MODELS_DIR / "metrics.json"


def _trained() -> bool:
    if not METRICS.exists():
        return False
    m = json.loads(METRICS.read_text())
    names = {m["best_model"], m["shap_model"], "quantile_low", "quantile_high"}
    return all((C.MODELS_DIR / f"{n}.joblib").exists() for n in names)


needs_models = pytest.mark.skipif(not _trained(), reason="models not trained (no .joblib artifacts)")


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS.read_text())


@needs_models
def test_best_model_predicts_profile(metrics, profile):
    model = joblib.load(C.MODELS_DIR / f"{metrics['best_model']}.joblib")
    pred = model.predict(profile)[0]
    assert 30_000 < pred < 400_000


@needs_models
def test_shap_reconstructs_prediction(metrics, profile):
    model = joblib.load(C.MODELS_DIR / f"{metrics['shap_model']}.joblib")
    ex = explain_one(model, profile)
    recon = ex.attrs["base_salary"] * np.exp(ex["shap_log"].sum())
    assert recon == pytest.approx(model.predict(profile)[0], rel=1e-3)


@needs_models
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
    # The world simulator converts every grid country into its own currency.
    bc = pd.read_csv(C.REPORTS_DIR / "world" / "by_country.csv").set_index("country")
    assert set(grid["country"]) <= set(bc.index)
    assert bc["fx"].gt(0).all() and bc["currency"].str.fullmatch(r"[A-Z]{3}").all()
    assert bc.loc["États-Unis", "fx"] == 1 and bc.loc["France", "currency"] == "EUR"


RUST_SCHEMA = C.MODELS_DIR / "rust" / "schema.json"


@pytest.mark.skipif(not (METRICS.exists() and RUST_SCHEMA.exists()), reason="no Rust bundle")
def test_rust_bundle_matches_trained_models():
    schema = json.loads(RUST_SCHEMA.read_text())
    metrics = json.loads(METRICS.read_text())
    assert schema["model"]["trained_at"] == metrics["metadata"]["trained_at"], (
        "models/rust est périmé : relancer `uv run python -m salary.export`")

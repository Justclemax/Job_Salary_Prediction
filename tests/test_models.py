import numpy as np
import pytest
from sklearn.base import clone

from salary import config as C
from salary.evaluate import regression_metrics
from salary.models import MODEL_NAMES, AveragingEnsemble, build_model

FAST = {"random_forest": {"n_estimators": 10}, "xgboost": {"n_estimators": 50},
        "lightgbm": {"n_estimators": 50}, "catboost": {"iterations": 50}}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_fits_and_predicts_dollars(small, name):
    model = clone(build_model(name, FAST.get(name)))  # clone must work (CatBoost gotcha)
    model.fit(small[C.FEATURES], small[C.TARGET])
    pred = model.predict(small[C.FEATURES])
    assert pred.shape == (len(small),)
    assert (pred > 10_000).all() and (pred < 1_000_000).all()
    if name != "dummy":
        assert regression_metrics(small[C.TARGET], pred)["r2"] > 0.5


def test_ensemble_is_geometric_mean(small):
    X, y = small[C.FEATURES], small[C.TARGET]
    a = build_model("ridge").fit(X, y)
    b = build_model("lightgbm", FAST["lightgbm"]).fit(X, y)
    ens = AveragingEnsemble({"a": a, "b": b})
    np.testing.assert_allclose(ens.predict(X), np.sqrt(a.predict(X) * b.predict(X)), rtol=1e-9)


def test_metrics_perfect_prediction():
    m = regression_metrics([100.0, 200.0], [100.0, 200.0])
    assert m["rmse"] == 0 and m["r2"] == 1 and m["within_10pct"] == 100

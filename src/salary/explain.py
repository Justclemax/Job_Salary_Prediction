"""SHAP explanations for the tree models.

Models are trained on log(salary), so SHAP values are additive in log space:
    log(pred) = base + sum(phi_i)   <=>   pred = exp(base) * prod(exp(phi_i))
exp(phi_i) is the multiplicative effect of feature i on the predicted salary.
"""

import numpy as np
import pandas as pd

from salary import config as C
from salary.models import inner_model


def shap_values(fitted, X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Return (SHAP values in log space with one column per feature, base value)."""
    prep, est = inner_model(fitted)
    Xt = prep.transform(X[C.FEATURES])
    if type(est).__name__ == "CatBoostRegressor":
        from catboost import Pool

        raw = est.get_feature_importance(Pool(Xt, cat_features=C.CATEGORICAL),
                                         type="ShapValues")
        values, base = raw[:, :-1], float(raw[0, -1])
    else:
        import shap

        explainer = shap.TreeExplainer(est)
        values = explainer.shap_values(Xt)
        base = float(np.ravel(explainer.expected_value)[0])
    return pd.DataFrame(values, columns=list(Xt.columns), index=X.index), base


def importance(shap_df: pd.DataFrame) -> pd.DataFrame:
    imp = shap_df.abs().mean().sort_values(ascending=False)
    return pd.DataFrame({"feature": imp.index, "mean_abs_shap": imp.values,
                         "mean_pct_effect": (np.exp(imp.values) - 1) * 100})


def explain_one(fitted, row: pd.DataFrame) -> pd.DataFrame:
    """Per-feature contribution for a single prediction, as % multiplicative effect."""
    vals, base = shap_values(fitted, row)
    phi = vals.iloc[0]
    out = pd.DataFrame({
        "feature": phi.index,
        "value": [row.iloc[0][f] for f in phi.index],
        "shap_log": phi.values,
        "pct_effect": (np.exp(phi.values) - 1) * 100,
    })
    out.attrs["base_salary"] = float(np.exp(base))
    return out.reindex(out["shap_log"].abs().sort_values(ascending=False).index)

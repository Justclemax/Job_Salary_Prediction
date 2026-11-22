"""Export the LightGBM models to a flat JSON bundle for the Rust inference server (serving/).

    uv run python -m salary.export            # writes models/rust/

Bundle (format "salary-forest/1"):
- `lightgbm.json`, `quantile_low.json`, `quantile_high.json`: each forest flattened into
  struct-of-arrays. Node i sends x to `left[i]` when x[split_feature[i]] <= threshold[i],
  else `right[i]`; a negative child c is the leaf `~c` (bitwise not). The raw score is the
  sum of one leaf per tree, then `exp` for the log-target point model.
- `schema.json`: feature order, ordinal codes of every category (from config), numeric
  ranges, the conformal offset of the 80 % interval and model metadata.
- `parity.json`: test profiles with the Python predictions, used by `cargo test` to check
  that Rust reproduces Python to within 1e-9 (relative).
"""

import json

import numpy as np
import pandas as pd

from salary import config as C
from salary.data import load_split
from salary.models import inner_model

OUT = C.MODELS_DIR / "rust"
FORMAT = "salary-forest/1"


def flatten(booster, name: str, transform: str) -> dict:
    """Flatten a LightGBM booster made only of numerical `<=` splits."""
    dump = booster.dump_model()
    if dump["num_tree_per_iteration"] != 1 or dump.get("average_output"):
        raise ValueError(f"{name} : seuls les boosters de régression simples sont exportables")
    feat, thr, left, right, leaves, roots = [], [], [], [], [], []

    def visit(node: dict) -> int:
        if "leaf_value" in node or "split_index" not in node:
            leaves.append(float(node["leaf_value"]))
            return ~(len(leaves) - 1)
        if node["decision_type"] != "<=" or node["missing_type"] != "None":
            raise ValueError(f"{name} : split non supporté ({node['decision_type']}, {node['missing_type']})")
        i = len(feat)
        feat.append(int(node["split_feature"]))
        thr.append(float(node["threshold"]))
        left.append(0)
        right.append(0)
        left[i] = visit(node["left_child"])
        right[i] = visit(node["right_child"])
        return i

    for tree in dump["tree_info"]:
        roots.append(visit(tree["tree_structure"]))
    return {"format": FORMAT, "name": name, "objective": dump["objective"].split()[0],
            "transform": transform, "n_features": len(dump["feature_names"]),
            "feature_names": dump["feature_names"], "roots": roots, "split_feature": feat,
            "threshold": thr, "left": left, "right": right, "leaf_value": leaves}


def schema(metrics: dict, feature_order: list[str]) -> dict:
    features = []
    for f in feature_order:
        if f in C.NUMERIC:
            lo, hi = C.NUMERIC_RANGES[f]
            features.append({"kind": "numeric", "name": f, "min": lo, "max": hi})
        else:
            features.append({"kind": "categorical", "name": f, "levels": C.CATEGORY_LEVELS[f]})
    iv = metrics["interval"]
    lgb = metrics["models"]["lightgbm"]
    return {
        "format": FORMAT, "features": features,
        "interval": {"alpha_low": iv["alpha_low"], "alpha_high": iv["alpha_high"],
                     "conformal_offset": iv.get("conformal_offset", 0.0), "coverage": iv["coverage"]},
        "model": {"name": "lightgbm", "cv_rmse": lgb["cv"]["rmse_mean"], "test_rmse": lgb["test"]["rmse"],
                  "test_r2": lgb["test"]["r2"], "trained_at": metrics["metadata"]["trained_at"],
                  "data_md5": metrics["metadata"]["data_md5"], "fast": metrics["metadata"]["fast"]},
    }


def python_predictions(point, q_low, q_high, X: pd.DataFrame, offset: float) -> pd.DataFrame:
    """Same arithmetic as the app's simulator (app/views/prediction.py)."""
    pred = point.predict(X)
    low = np.minimum(np.maximum(0.0, q_low.predict(X) - offset), pred)
    high = np.maximum(q_high.predict(X) + offset, pred)
    return pd.DataFrame({"salary": pred, "low": low, "high": high})


def run(n_parity: int = 500) -> dict:
    import joblib

    metrics = json.loads((C.MODELS_DIR / "metrics.json").read_text())
    point = joblib.load(C.MODELS_DIR / "lightgbm.joblib")
    q_low = joblib.load(C.MODELS_DIR / "quantile_low.joblib")
    q_high = joblib.load(C.MODELS_DIR / "quantile_high.joblib")
    if not (point.func is np.log and point.inverse_func is np.exp):
        raise ValueError("lightgbm.joblib doit avoir une cible en log")

    OUT.mkdir(parents=True, exist_ok=True)
    forests = {"lightgbm": (inner_model(point)[1].booster_, "exp"),
               "quantile_low": (q_low.named_steps["model"].booster_, "identity"),
               "quantile_high": (q_high.named_steps["model"].booster_, "identity")}
    order = None
    for name, (booster, transform) in forests.items():
        f = flatten(booster, name, transform)
        order = order or f["feature_names"]
        if f["feature_names"] != order or order != C.NUMERIC + C.CATEGORICAL:
            raise ValueError(f"{name} : ordre des variables inattendu {f['feature_names']}")
        (OUT / f"{name}.json").write_text(json.dumps(f, separators=(",", ":")))
    sch = schema(metrics, order)
    (OUT / "schema.json").write_text(json.dumps(sch, indent=2, ensure_ascii=False))

    _, test = load_split()
    X = test[C.FEATURES].sample(n_parity, random_state=C.SEED).reset_index(drop=True)
    preds = python_predictions(point, q_low, q_high, X, sch["interval"]["conformal_offset"])
    (OUT / "parity.json").write_text(json.dumps(
        {"profiles": X.to_dict(orient="records"), **{k: preds[k].tolist() for k in preds}},
        ensure_ascii=False))
    sizes = {p.name: p.stat().st_size for p in sorted(OUT.glob("*.json"))}
    return {"out": str(OUT.relative_to(C.ROOT)), "files": sizes,
            "n_trees": {n: len(b.dump_model()["tree_info"]) for n, (b, _) in forests.items()}}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

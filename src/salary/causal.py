"""Causal analysis: what would a change in X do to salary, all else being equal?

Approach (DoWhy's model -> identify -> estimate -> refute, plus EconML Double ML):
1. Causal graph. Testable implication of "every feature is a root cause of salary
   and no feature causes another": pairwise independence of the features. The data
   satisfies it (max association ~0.009, see salary.stats), so the backdoor set for
   each treatment is simply "all other features".
2. Estimators compared for each treatment: naive difference, backdoor linear
   regression (DoWhy), inverse propensity weighting (DoWhy, binary treatments),
   and LinearDML with gradient-boosted nuisance models (EconML).
3. Refutation tests: placebo treatment, random common cause, data subset.
4. Heterogeneity (CATE) by job title and location with LinearDML.

Caveat: the dataset is very likely synthetic. These estimates describe its
data-generating process, not the real labor market.

    uv run python -m salary.causal     # writes reports/causal/*
"""

import json
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor

from salary import config as cfg
from salary.data import load_raw

OUT = cfg.REPORTS_DIR / "causal"


@dataclass(frozen=True)
class Treatment:
    key: str
    label: str
    source: str  # original column (excluded from the adjustment set)
    binary: bool
    make: Callable[[pd.DataFrame], pd.Series]
    filter: Callable[[pd.DataFrame], pd.Series] | None = None
    unit: str = ""


TREATMENTS = [
    Treatment("education_advanced", "Master/PhD vs ≤ Licence", "education_level", True,
              lambda d: d["education_level"].isin(["Master", "PhD"])),
    Treatment("large_company", "Grande entreprise (Large/Enterprise) vs autres", "company_size", True,
              lambda d: d["company_size"].isin(["Large", "Enterprise"])),
    Treatment("remote_yes", "Télétravail total vs présentiel", "remote_work", True,
              lambda d: d["remote_work"].eq("Yes"), lambda d: d["remote_work"].ne("Hybrid")),
    Treatment("experience_years", "+1 année d'expérience", "experience_years", False,
              lambda d: d["experience_years"].astype(float), unit="par an"),
    Treatment("certifications", "+1 certification", "certifications", False,
              lambda d: d["certifications"].astype(float), unit="par certification"),
    Treatment("skills_count", "+1 compétence", "skills_count", False,
              lambda d: d["skills_count"].astype(float), unit="par compétence"),
]
TREATMENT_BY_KEY = {t.key: t for t in TREATMENTS}


def dag_edges() -> list[tuple[str, str]]:
    return [(f, cfg.TARGET) for f in cfg.FEATURES]


def dag_dot() -> str:
    lines = ["digraph {", "  rankdir=LR;", '  node [shape=box, style="rounded"];']
    lines += [f'  "{a}" -> "{b}";' for a, b in dag_edges()]
    return "\n".join(lines + ["}"])


def _confounders(df: pd.DataFrame, t: Treatment) -> pd.DataFrame:
    """One-hot encoded adjustment set: every feature except the treatment's source column."""
    cols = [c for c in cfg.FEATURES if c != t.source]
    cats = [c for c in cols if c in cfg.CATEGORICAL]
    W = pd.get_dummies(df[cols], columns=cats, drop_first=True, dtype=float)
    W.columns = [c.replace(" ", "_").replace("/", "_") for c in W.columns]
    return W


def _prepare(df: pd.DataFrame, t: Treatment) -> tuple[pd.DataFrame, list[str]]:
    d = df[t.filter(df)] if t.filter else df
    W = _confounders(d, t)
    data = W.copy()
    data["T"] = t.make(d).values if not t.binary else t.make(d).astype(bool).values
    data["Y"] = d[cfg.TARGET].astype(float).values
    data["logY"] = np.log(data["Y"])
    return data.reset_index(drop=True), list(W.columns)


def _naive(data: pd.DataFrame, binary: bool, outcome: str) -> float:
    if binary:
        return float(data.loc[data["T"], outcome].mean() - data.loc[~data["T"], outcome].mean())
    return float(np.polyfit(data["T"], data[outcome], 1)[0])


def _dowhy(data: pd.DataFrame, W: list[str], binary: bool, refute: bool, n_sims: int):
    from dowhy import CausalModel

    model = CausalModel(data=data[W + ["T", "Y"]], treatment="T", outcome="Y", common_causes=W)
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    lin = model.estimate_effect(estimand, method_name="backdoor.linear_regression",
                                confidence_intervals=True, test_significance=True)
    ci = np.ravel(lin.get_confidence_intervals())
    res = {"ols_effect": float(lin.value), "ols_ci_low": float(ci[0]), "ols_ci_high": float(ci[1]),
           "backdoor_set": "all other features"}
    if binary:
        ipw = model.estimate_effect(estimand, method_name="backdoor.propensity_score_weighting",
                                    method_params={"weighting_scheme": "ips_weight"})
        res["ipw_effect"] = float(ipw.value)
    refutations = []
    if refute:
        for method, kw in [
            ("placebo_treatment_refuter", {"placebo_type": "permute"}),
            ("random_common_cause", {}),
            ("data_subset_refuter", {"subset_fraction": 0.8}),
        ]:
            r = model.refute_estimate(estimand, lin, method_name=method,
                                      num_simulations=n_sims, random_seed=cfg.SEED, **kw)
            rr = r.refutation_result or {}
            refutations.append({
                "refuter": method, "estimated_effect": float(r.estimated_effect),
                "new_effect": float(np.ravel(r.new_effect)[0]),
                "p_value": float(rr.get("p_value", np.nan)) if rr else np.nan,
            })
    return res, refutations


def _dml(data: pd.DataFrame, W: list[str], binary: bool, outcome: str,
         X: pd.DataFrame | None = None):
    from econml.dml import LinearDML

    model_y = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31,
                            verbose=-1, random_state=cfg.SEED)
    model_t = (LGBMClassifier if binary else LGBMRegressor)(
        n_estimators=200, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=cfg.SEED)
    est = LinearDML(model_y=model_y, model_t=model_t, discrete_treatment=binary,
                    cv=3, random_state=cfg.SEED)
    T = data["T"].astype(int) if binary else data["T"]
    est.fit(data[outcome], T, X=X, W=data[W])
    return est


def estimate_treatment(df: pd.DataFrame, t: Treatment, refute: bool = True,
                       n_sims: int = 20) -> tuple[dict, list[dict]]:
    data, W = _prepare(df, t)
    row = {"treatment": t.key, "label": t.label, "binary": t.binary, "unit": t.unit,
           "n": len(data), "naive_effect": _naive(data, t.binary, "Y")}
    dw, refutations = _dowhy(data, W, t.binary, refute, n_sims)
    row.update(dw)
    est = _dml(data, W, t.binary, "Y")
    lo, hi = est.ate_interval(alpha=0.05)
    row.update({"dml_effect": float(est.ate()), "dml_ci_low": float(lo), "dml_ci_high": float(hi)})
    est_log = _dml(data, W, t.binary, "logY")
    lo, hi = est_log.ate_interval(alpha=0.05)
    row.update({
        "naive_pct": (np.exp(_naive(data, t.binary, "logY")) - 1) * 100,
        "dml_pct": (np.exp(float(est_log.ate())) - 1) * 100,
        "dml_pct_ci_low": (np.exp(float(lo)) - 1) * 100,
        "dml_pct_ci_high": (np.exp(float(hi)) - 1) * 100,
    })
    for r in refutations:
        r["treatment"] = t.key
    return row, refutations


def cate(df: pd.DataFrame, t: Treatment, by: str) -> pd.DataFrame:
    """Effect of `t` for each level of `by` (LinearDML with `by` as effect modifier)."""
    data, W = _prepare(df, t)
    d = df[t.filter(df)] if t.filter else df
    levels = cfg.CATEGORY_LEVELS[by]
    X = pd.get_dummies(pd.Categorical(d[by].values, categories=levels), drop_first=True, dtype=float)
    W = [w for w in W if not w.startswith(f"{by}_")]
    X_eval = pd.get_dummies(pd.Categorical(levels, categories=levels), drop_first=True, dtype=float)
    rows = []
    for outcome in ["Y", "logY"]:
        est = _dml(data, W, t.binary, outcome, X=X.values)
        eff = est.effect(X_eval.values)
        lo, hi = est.effect_interval(X_eval.values, alpha=0.05)
        for lvl, e, a, b in zip(levels, eff, lo, hi, strict=False):
            rows.append({"treatment": t.key, "by": by, "level": lvl, "outcome": outcome,
                         "effect": float(e), "ci_low": float(a), "ci_high": float(b)})
    out = pd.DataFrame(rows)
    wide = out[out.outcome == "Y"].drop(columns="outcome").reset_index(drop=True)
    pct = out[out.outcome == "logY"].reset_index(drop=True)
    wide["effect_pct"] = (np.exp(pct["effect"]) - 1) * 100
    wide["pct_ci_low"] = (np.exp(pct["ci_low"]) - 1) * 100
    wide["pct_ci_high"] = (np.exp(pct["ci_high"]) - 1) * 100
    return wide


def run_all(df: pd.DataFrame | None = None, sample: int = 80_000, refute: bool = True,
            n_sims: int = 20) -> dict:
    warnings.filterwarnings("ignore")
    logging.getLogger("dowhy").setLevel(logging.ERROR)
    df = load_raw() if df is None else df
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=cfg.SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dag.dot").write_text(dag_dot())

    rows, refs = [], []
    for t in TREATMENTS:
        print(f"ATE: {t.key}", flush=True)
        row, r = estimate_treatment(df, t, refute=refute, n_sims=n_sims)
        rows.append(row)
        refs += r
    ate = pd.DataFrame(rows)
    ate.to_csv(OUT / "ate.csv", index=False)
    pd.DataFrame(refs).to_csv(OUT / "refutations.csv", index=False)

    cates = []
    for key in ["education_advanced", "experience_years", "large_company"]:
        for by in ["job_title", "location"]:
            print(f"CATE: {key} by {by}", flush=True)
            cates.append(cate(df, TREATMENT_BY_KEY[key], by))
    pd.concat(cates, ignore_index=True).to_csv(OUT / "cate.csv", index=False)

    out = {"n_sample": len(df), "n_treatments": len(TREATMENTS),
           "max_naive_vs_adjusted_gap_pct": float(
               ((ate["naive_effect"] - ate["dml_effect"]).abs() / ate["dml_effect"].abs()).max() * 100)}
    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))

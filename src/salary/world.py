"""The same study on real data: Stack Overflow Developer Survey + World Bank PPP.

Steps (all outputs in reports/world/):
1. Load and harmonise the survey (sources.stackoverflow), convert to PPP (sources.worldbank).
2. Descriptives by country and job; pairwise feature associations. Unlike the
   synthetic data, the features are correlated (country <-> education, age <-> experience...).
3. Log-linear OLS (HC3) with type-II ANOVA: share of explained variance per factor.
4. Causal effects with an explicit adjustment set per treatment (a DAG, not "all
   other features"): naive vs DoWhy OLS vs Double ML, plus refuters, CATE by country.
5. Cross-validated models (country-median baseline, Ridge, LightGBM, CatBoost) and
   SHAP importance; a prediction grid job x education x experience x remote x country
   for the app's world simulator.
6. Comparison with the synthetic dataset (country ratios to the USA, causal effects).

    uv run python -W ignore -m salary.world          # ~5 min, downloads ~260 MB once
"""

import argparse
import json
import logging
import warnings
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import stats as sps
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from salary import config as cfg
from salary.causal import _dml, _dowhy, _naive
from salary.evaluate import regression_metrics, summarize_folds
from salary.sources import stackoverflow as so
from salary.sources import worldbank as wb
from salary.stats import _correlation_ratio, _cramers_v

OUT = cfg.REPORTS_DIR / "world"
PPP_YEAR = 2025
NUM, CAT = so.FEATURES_NUM, so.FEATURES_CAT

# Synthetic dataset location -> French country label used for real data.
SYNTH_COUNTRY = {"USA": "États-Unis", "Germany": "Allemagne", "UK": "Royaume-Uni", "India": "Inde",
                 "Canada": "Canada", "Netherlands": "Pays-Bas", "Australia": "Australie",
                 "Sweden": "Suède", "Singapore": "Singapour"}


# ----------------------------------------------------------------------------- data
def load(force: bool = False) -> pd.DataFrame:
    df = so.load(force=force)
    df["iso3"] = df["country"].map(so.ISO3)
    df["salary_ppp"] = wb.to_ppp(df, year=PPP_YEAR)
    return df


# ----------------------------------------------------------------------------- descriptives
def by_group(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col)
    out = pd.DataFrame({
        "n": g.size(),
        "median_usd": g["salary_usd"].median(),
        "q1_usd": g["salary_usd"].quantile(0.25),
        "q3_usd": g["salary_usd"].quantile(0.75),
        "median_ppp": g["salary_ppp"].median(),
        "median_experience": g["experience_years"].median(),
        "share_master_plus": g["education"].apply(lambda s: s.isin(["Master", "Doctorat / prof."]).mean() * 100),
        "share_remote": g["remote"].apply(lambda s: s.eq("Télétravail").mean() * 100),
    })
    return out.sort_values("median_usd", ascending=False).reset_index()


def associations(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise association between features, 0-1 scale (Cramér V, eta, |Spearman|)."""
    rows = []
    for a, b in combinations(NUM + [c for c in CAT if c != "year"], 2):
        ca, cb = a in CAT, b in CAT
        if ca and cb:
            v, _ = _cramers_v(df[a], df[b])
            kind = "Cramér V"
        elif ca or cb:
            cat, num = (a, b) if ca else (b, a)
            v, kind = _correlation_ratio(df[cat], df[num]), "eta"
        else:
            v, kind = abs(sps.spearmanr(df[a], df[b])[0]), "|Spearman|"
        rows.append({"feature_a": a, "feature_b": b, "measure": kind, "value": float(v)})
    return pd.DataFrame(rows).sort_values("value", ascending=False).reset_index(drop=True)


def log_linear(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """log(salary_usd) ~ all features; type-II ANOVA shares and HC3 fit statistics."""
    import statsmodels.api as sm

    formula = "np.log(salary_usd) ~ experience_years + I(experience_years ** 2) + " + " + ".join(
        f"C({c})" for c in CAT)
    fit = smf.ols(formula, data=df).fit(cov_type="HC3")
    aov = sm.stats.anova_lm(smf.ols(formula, data=df).fit(), typ=2)
    ss = aov["sum_sq"].drop("Residual")
    names = {f"C({c})": c for c in CAT} | {"experience_years": "experience_years",
                                          "I(experience_years ** 2)": "experience_years"}
    share = ss.groupby(ss.index.map(names)).sum()
    out = pd.DataFrame({"feature": share.index,
                        "share_explained_pct": share.values / share.sum() * 100})
    out = out.sort_values("share_explained_pct", ascending=False).reset_index(drop=True)
    return out, {"r2": float(fit.rsquared), "r2_adj": float(fit.rsquared_adj), "n": int(fit.nobs)}


# ----------------------------------------------------------------------------- causal
@dataclass(frozen=True)
class RealTreatment:
    key: str
    label: str
    binary: bool
    adjust: tuple[str, ...]  # backdoor set chosen from the DAG below
    make: object
    keep: object = None
    unit: str = ""
    synthetic_key: str = ""


# Assumed causal graph. Adjustment sets below block every backdoor path in it and
# leave mediators out (job after education, remote after company size), so each
# estimate is a total effect. Unobserved causes (skill, city, gender, negotiation)
# are not in the graph: real-data estimates rely on that strong assumption.
DAG_EDGES = [
    ("country", "education"), ("country", "job_family"), ("country", "company_size"),
    ("country", "remote"), ("country", "experience_years"), ("country", "salary"),
    ("age", "education"), ("age", "experience_years"), ("age", "job_family"), ("age", "salary"),
    ("education", "job_family"), ("education", "experience_years"), ("education", "salary"),
    ("experience_years", "job_family"), ("experience_years", "salary"),
    ("company_size", "remote"), ("company_size", "salary"),
    ("job_family", "remote"), ("job_family", "salary"), ("remote", "salary"), ("year", "salary"),
]
TREATMENTS = [
    RealTreatment("education_advanced", "Master ou doctorat vs licence", True,
                  ("country", "age", "year"),
                  lambda d: d["education"].isin(["Master", "Doctorat / prof."]),
                  lambda d: d["education"].isin(["Licence", "Master", "Doctorat / prof."]),
                  synthetic_key="education_advanced"),
    RealTreatment("large_company", "Entreprise ≥ 1 000 salariés vs plus petite", True,
                  ("country", "age", "education", "experience_years", "job_family", "year"),
                  lambda d: d["company_size"].isin(["1 000–4 999", "5 000–9 999", "10 000 +"]),
                  synthetic_key="large_company"),
    RealTreatment("remote_full", "Télétravail total vs présentiel", True,
                  ("country", "age", "education", "experience_years", "job_family", "company_size", "year"),
                  lambda d: d["remote"].eq("Télétravail"),
                  lambda d: d["remote"].ne("Hybride"), synthetic_key="remote_yes"),
    RealTreatment("experience_years", "+1 année d'expérience", False,
                  ("country", "education", "year"),
                  lambda d: d["experience_years"].astype(float), unit="par an",
                  synthetic_key="experience_years"),
]


def _prepare(df: pd.DataFrame, t: RealTreatment) -> tuple[pd.DataFrame, list[str]]:
    d = df[t.keep(df)] if t.keep else df
    cats = [c for c in t.adjust if c in CAT]
    W = pd.get_dummies(d[list(t.adjust)], columns=cats, drop_first=True, dtype=float)
    W.columns = [f"w{i}" for i in range(W.shape[1])]  # dowhy-safe names
    data = W.copy()
    data["T"] = t.make(d).astype(bool).values if t.binary else t.make(d).values
    data["Y"] = d["salary_usd"].values
    data["logY"] = np.log(data["Y"])
    return data.reset_index(drop=True), list(W.columns)


def estimate(df: pd.DataFrame, t: RealTreatment, n_sims: int = 10) -> tuple[dict, list[dict]]:
    data, W = _prepare(df, t)
    row = {"treatment": t.key, "label": t.label, "unit": t.unit, "n": len(data),
           "adjustment_set": ", ".join(t.adjust),
           "naive_effect": _naive(data, t.binary, "Y"),
           "naive_pct": (np.exp(_naive(data, t.binary, "logY")) - 1) * 100}
    dw, refs = _dowhy(data, W, t.binary, refute=True, n_sims=n_sims)
    dw.pop("backdoor_set")
    row.update(dw)
    est = _dml(data, W, t.binary, "Y")
    lo, hi = est.ate_interval(alpha=0.05)
    row.update({"dml_effect": float(est.ate()), "dml_ci_low": float(lo), "dml_ci_high": float(hi)})
    est = _dml(data, W, t.binary, "logY")
    lo, hi = est.ate_interval(alpha=0.05)
    row.update({"dml_pct": (np.exp(float(est.ate())) - 1) * 100,
                "dml_pct_ci_low": (np.exp(float(lo)) - 1) * 100,
                "dml_pct_ci_high": (np.exp(float(hi)) - 1) * 100})
    row["naive_vs_adjusted_gap_pct"] = abs(row["naive_pct"] - row["dml_pct"]) / abs(row["dml_pct"]) * 100
    for r in refs:
        r["treatment"] = t.key
    return row, refs


def cate_by_country(df: pd.DataFrame, t: RealTreatment, top: int = 12) -> pd.DataFrame:
    """Effect (in %) of `t` in each of the `top` largest countries (LinearDML, country as modifier)."""
    keep = df["country"].value_counts().index[:top]
    d = df[df["country"].isin(keep)]
    d = d[t.keep(d)] if t.keep else d
    t_local = RealTreatment(t.key, t.label, t.binary, tuple(c for c in t.adjust if c != "country"),
                            t.make, None, t.unit)
    data, W = _prepare(d, t_local)
    levels = list(keep)
    X = pd.get_dummies(pd.Categorical(d["country"].values, categories=levels), drop_first=True, dtype=float)
    X_eval = pd.get_dummies(pd.Categorical(levels, categories=levels), drop_first=True, dtype=float)
    est = _dml(data, W, t.binary, "logY", X=X.values)
    eff = est.effect(X_eval.values)
    lo, hi = est.effect_interval(X_eval.values, alpha=0.05)
    n = d["country"].value_counts()
    return pd.DataFrame({"treatment": t.key, "country": levels, "n": [int(n[c]) for c in levels],
                         "effect_pct": (np.exp(eff) - 1) * 100, "pct_ci_low": (np.exp(lo) - 1) * 100,
                         "pct_ci_high": (np.exp(hi) - 1) * 100})


# ----------------------------------------------------------------------------- models
class CountryMedian:
    """Baseline: predict the median salary of the person's country."""

    def fit(self, X, y):
        self.m_ = pd.Series(np.asarray(y), index=X.index).groupby(X["country"]).median()
        self.g_ = float(np.median(y))
        return self

    def predict(self, X):
        return X["country"].map(self.m_).fillna(self.g_).to_numpy()


def _as_category(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=CATEGORY_ORDER[c])
    return X


CATEGORY_ORDER: dict[str, list[str]] = {}


def build(name: str):
    log = {"func": np.log, "inverse_func": np.exp}
    if name == "country_median":
        return CountryMedian()
    if name == "ridge":
        prep = ColumnTransformer([("num", StandardScaler(), NUM),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
        return TransformedTargetRegressor(Pipeline([("prep", prep), ("model", Ridge(alpha=1.0))]), **log)
    if name == "lightgbm":
        return TransformedTargetRegressor(LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, cat_smooth=20,
            verbose=-1, random_state=cfg.SEED), **log)
    if name == "catboost":
        return TransformedTargetRegressor(CatBoostRegressor(
            iterations=1200, learning_rate=0.04, depth=6, l2_leaf_reg=5, loss_function="RMSE",
            cat_features=tuple(CAT), random_seed=cfg.SEED, verbose=0, allow_writing_files=False), **log)
    raise ValueError(name)


MODELS = {"country_median": "Médiane du pays", "ridge": "Ridge (log)",
          "lightgbm": "LightGBM", "catboost": "CatBoost"}


def _X(df: pd.DataFrame, name: str) -> pd.DataFrame:
    X = df[NUM + CAT]
    return _as_category(X) if name == "lightgbm" else X


def cross_validate(df: pd.DataFrame, k: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    kf = KFold(k, shuffle=True, random_state=cfg.SEED)
    rows, oof = [], {}
    y = df["salary_usd"].to_numpy()
    for name in MODELS:
        pred = np.zeros(len(df))
        folds = []
        for tr, va in kf.split(df):
            m = build(name).fit(_X(df.iloc[tr], name), y[tr])
            pred[va] = m.predict(_X(df.iloc[va], name))
            folds.append(regression_metrics(y[va], pred[va]))
        rows.append({"model": name, "label": MODELS[name], **summarize_folds(folds)})
        oof[name] = pred
        print(f"  CV {name}: R² {rows[-1]['r2_mean']:.3f}", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(oof)


def shap_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    from catboost import Pool

    cb = model.regressor_
    sv = cb.get_feature_importance(Pool(X, cat_features=list(CAT)), type="ShapValues")[:, :-1]
    imp = np.abs(sv).mean(axis=0)
    return (pd.DataFrame({"feature": X.columns, "mean_abs_shap": imp,
                          "typical_effect_pct": (np.exp(imp) - 1) * 100})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))


EXPERIENCE_GRID = [0, 2, 5, 10, 15, 20, 30]


def _age_for(exp: int) -> str:
    return "18-24" if exp < 2 else "25-34" if exp < 10 else "35-44" if exp < 20 else "45-54" if exp < 30 else "55-64"


def prediction_grid(model, df: pd.DataFrame) -> pd.DataFrame:
    """Predicted salary for every job x education x experience x remote x country.

    Company size is fixed to the most common level, the survey year to the latest one
    and the age bracket follows experience.
    """
    company = df["company_size"].mode()[0]
    year = max(df["year"])
    edu = ["Licence", "Master", "Doctorat / prof.", "Supérieur sans diplôme"]
    rows = [{"experience_years": e, "country": c, "job_family": j, "education": ed,
             "company_size": company, "remote": r, "age": _age_for(e), "year": year}
            for j, ed, e, r, c in product(so.LEVELS["job_family"], edu, EXPERIENCE_GRID,
                                          so.REMOTE_LEVELS, sorted(df["country"].unique()))]
    grid = pd.DataFrame(rows)[NUM + CAT]
    grid["pred_usd"] = model.predict(grid).round(-1)
    pl = (df["salary_usd"] / df["salary_ppp"]).groupby(df["country"]).first()  # price level
    grid["pred_ppp"] = (grid["pred_usd"] / grid["country"].map(pl)).round(-1)
    grid.attrs["company_size"] = company
    return grid


# ----------------------------------------------------------------------------- comparison
def compare_countries(real: pd.DataFrame) -> pd.DataFrame:
    synth = pd.read_csv(cfg.DATA_RAW, usecols=["location", "salary"])
    s = synth[synth["location"].isin(SYNTH_COUNTRY)].groupby("location")["salary"].median()
    s.index = s.index.map(SYNTH_COUNTRY)
    r = real.groupby("country")[["salary_usd", "salary_ppp"]].median()
    out = pd.DataFrame({"synthetic_usd": s}).join(r, how="inner")
    us = out.loc["États-Unis"]
    out["synthetic_ratio_usa"] = out["synthetic_usd"] / us["synthetic_usd"] * 100
    out["real_ratio_usa"] = out["salary_usd"] / us["salary_usd"] * 100
    out["real_ppp_ratio_usa"] = out["salary_ppp"] / us["salary_ppp"] * 100
    return out.rename(columns={"salary_usd": "real_usd", "salary_ppp": "real_ppp"}) \
              .sort_values("real_usd", ascending=False).rename_axis("country").reset_index()


def compare_effects(ate: pd.DataFrame) -> pd.DataFrame:
    p = cfg.REPORTS_DIR / "causal" / "ate.csv"
    if not p.exists():
        return pd.DataFrame()
    synth = pd.read_csv(p).set_index("treatment")
    rows = []
    for t in TREATMENTS:
        r = ate.set_index("treatment").loc[t.key]
        s = synth.loc[t.synthetic_key] if t.synthetic_key in synth.index else None
        rows.append({"treatment": t.key, "label": t.label,
                     "synthetic_naive_pct": float(s["naive_pct"]) if s is not None else np.nan,
                     "synthetic_dml_pct": float(s["dml_pct"]) if s is not None else np.nan,
                     "real_naive_pct": float(r["naive_pct"]), "real_dml_pct": float(r["dml_pct"]),
                     "real_dml_pct_ci_low": float(r["dml_pct_ci_low"]),
                     "real_dml_pct_ci_high": float(r["dml_pct_ci_high"])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def run_all(force_download: bool = False, n_sims: int = 10) -> dict:
    warnings.filterwarnings("ignore")
    logging.getLogger("dowhy").setLevel(logging.ERROR)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dag_edges.json").write_text(json.dumps(DAG_EDGES))

    print("Données…", flush=True)
    df = load(force_download)
    CATEGORY_ORDER.update({c: sorted(df[c].unique()) for c in CAT})
    by_country = by_group(df, "country")
    by_country["iso3"] = by_country["country"].map(so.ISO3)
    pl = wb.price_level(by_country["iso3"].tolist(), PPP_YEAR).set_index("iso3")
    by_country["price_level"] = by_country["iso3"].map(pl["price_level"])
    by_country.to_csv(OUT / "by_country.csv", index=False)
    by_group(df, "job_family").to_csv(OUT / "by_job.csv", index=False)
    assoc = associations(df)
    assoc.to_csv(OUT / "associations.csv", index=False)
    df.sample(min(len(df), 6000), random_state=cfg.SEED).to_parquet(OUT / "sample.parquet")

    print("Modèle log-linéaire…", flush=True)
    shares, ols = log_linear(df)
    shares.to_csv(OUT / "anova_shares.csv", index=False)

    print("Causalité…", flush=True)
    rows, refs = [], []
    for t in TREATMENTS:
        print(f"  ATE {t.key}", flush=True)
        r, rf = estimate(df, t, n_sims=n_sims)
        rows.append(r)
        refs += rf
    ate = pd.DataFrame(rows)
    ate.to_csv(OUT / "ate.csv", index=False)
    pd.DataFrame(refs).to_csv(OUT / "refutations.csv", index=False)
    by_key = {t.key: t for t in TREATMENTS}
    pd.concat([cate_by_country(df, by_key[k]) for k in ("education_advanced", "remote_full")],
              ignore_index=True).to_csv(OUT / "cate_country.csv", index=False)

    print("Modèles…", flush=True)
    cv, oof = cross_validate(df)
    cv.to_csv(OUT / "model_cv.csv", index=False)
    best = cv.loc[cv["rmse_mean"].idxmin(), "model"]
    oof_df = df[["country", "job_family", "salary_usd"]].assign(pred=oof[best].to_numpy())
    oof_df.sample(min(len(df), 6000), random_state=cfg.SEED).to_parquet(OUT / "oof_sample.parquet")
    final = build("catboost").fit(df[NUM + CAT], df["salary_usd"])
    shap_importance(final, df[NUM + CAT].sample(4000, random_state=cfg.SEED)).to_csv(
        OUT / "shap_importance.csv", index=False)
    grid = prediction_grid(final, df)
    grid.to_parquet(OUT / "prediction_grid.parquet")

    print("Comparaison avec les données synthétiques…", flush=True)
    compare_countries(df).to_csv(OUT / "compare_countries.csv", index=False)
    compare_effects(ate).to_csv(OUT / "compare_effects.csv", index=False)

    summary = {
        "source": so.ATTRIBUTION, "ppp_source": "Banque mondiale, PA.NUS.PPP et PA.NUS.FCRF",
        "ppp_year": PPP_YEAR, "years": sorted(df["year"].unique().tolist()),
        "n": len(df), "n_countries": int(df["country"].nunique()),
        "n_jobs": int(df["job_family"].nunique()),
        "median_usd": float(df["salary_usd"].median()),
        "max_association": float(assoc["value"].max()),
        "top_association": assoc.iloc[0][["feature_a", "feature_b", "measure", "value"]].to_dict(),
        "log_ols": ols, "best_model": best,
        "best_cv": cv.set_index("model").loc[best].drop("label").to_dict(),
        "max_naive_vs_adjusted_gap_pct": float(ate["naive_vs_adjusted_gap_pct"].max()),
        "grid_company_size": grid.attrs["company_size"],
        "grid_model": "catboost",  # within one CV std of the best model, native SHAP
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=float))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true", help="re-télécharger les sources")
    args = ap.parse_args()
    print(json.dumps(run_all(force_download=args.refresh), indent=2, ensure_ascii=False, default=float))
